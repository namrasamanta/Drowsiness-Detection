"""Browser-based drowsiness detection server.

Turns this PC into a web server: visitors open the page, the browser asks
for camera permission, frames stream to this PC for analysis, and results
(alerts, metrics, landmark overlay) render live back in their browser.
Each visitor gets an independent detection session.

Run:    python web_server.py [--port 8000]
Local:  http://localhost:8000

Hosting online: browsers only allow camera access on HTTPS pages (localhost
is the one exception), so expose the port through a tunnel that terminates
TLS for you and share the https URL it prints:

    cloudflared tunnel --url http://localhost:8000
    (or: ngrok http 8000)
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import dlib
import numpy as np
from imutils import face_utils

from Drowsiness_Detection import (eye_aspect_ratio, head_pose_degrees,
                                  mouth_aspect_ratio)

L_START, L_END = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
R_START, R_END = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
M_START, M_END = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# same defaults as the desktop script; calibration is a little shorter
# because browser upload rates are lower than a local webcam's fps
CFG = {
    "ear_thresh": 0.25,
    "closed_secs": 3.0,
    "perclos_window": 30.0,
    "perclos_thresh": 0.30,
    "yawn_thresh": 0.45,
    "yawn_secs": 1.0,
    "yawns_per_minute": 2,
    "pitch_thresh": 20.0,
    "nod_secs": 1.5,
    "yaw_limit": 25.0,
    "distraction_secs": 3.0,
    "distraction_alarm_secs": 6.0,
    "face_lost_secs": 2.0,
    "face_lost_alarm_secs": 4.0,
    "calibrate_secs": 4.0,
    "detect_width": 320,
    "session_idle_secs": 120.0,
    "events_dir": "events",
    "status_log_secs": 30.0,
}


class Session:
    """Per-visitor detection state, fed one frame at a time."""

    def __init__(self, sid="local"):
        now = time.monotonic()
        self.sid = sid
        self.started = now
        self.last_seen = now
        self.calibrated = False
        self.calib_end = None
        self.ear_samples = []
        self.pitch_samples = []
        self.ear_thresh = CFG["ear_thresh"]
        self.baseline_pitch = 0.0
        self.closed_since = None
        self.yawn_since = None
        self.yawn_open = False
        self.nod_since = None
        self.away_since = None
        self.last_face_time = None
        self.eye_history = deque()   # (timestamp, closed?)
        self.yawn_times = deque()    # timestamps of completed yawns
        self.perclos = 0.0
        self.perclos_active = False
        self.alert_active = False
        self.last_alert_end = 0.0
        self.last_status_log = 0.0
        self.ip = None

    def process(self, frame, detector, predictor):
        now = time.monotonic()
        self.last_seen = now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale = CFG["detect_width"] / gray.shape[1]
        small = cv2.resize(gray, (CFG["detect_width"],
                                  int(gray.shape[0] * scale)))
        found = detector(small, 0)

        alerts, warnings = [], []
        metrics = {}
        landmarks = None
        state = "waiting"

        if found:
            r = max(found, key=lambda r: r.width() * r.height())
            rect = dlib.rectangle(
                int(r.left() / scale), int(r.top() / scale),
                int(r.right() / scale), int(r.bottom() / scale))
            self.last_face_time = now
            shape = face_utils.shape_to_np(predictor(gray, rect))

            left_eye = shape[L_START:L_END]
            right_eye = shape[R_START:R_END]
            mouth = shape[M_START:M_END]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            pitch, yaw = head_pose_degrees(shape, frame.shape)
            looking_away = yaw is not None and abs(yaw) > CFG["yaw_limit"]

            landmarks = {
                "left_eye": left_eye.astype(int).tolist(),
                "right_eye": right_eye.astype(int).tolist(),
                "mouth": mouth[12:20].astype(int).tolist(),  # inner lips
            }
            metrics = {"ear": round(ear, 3), "mar": round(mar, 3)}
            if pitch is not None:
                metrics["pitch"] = round(pitch - self.baseline_pitch, 1)
            if yaw is not None:
                metrics["yaw"] = round(yaw, 1)

            if not self.calibrated:
                # learn this person's normal pose and eye openness first;
                # only facing-forward frames count, and the window extends
                # until enough of them are seen
                state = "calibrating"
                self.calib_end = self.calib_end or now + CFG["calibrate_secs"]
                if not looking_away:
                    self.ear_samples.append(ear)
                    if pitch is not None and abs(pitch) < 35:
                        self.pitch_samples.append(pitch)
                if now >= self.calib_end and len(self.ear_samples) >= 20:
                    self.calibrated = True
                    self.ear_samples.sort()
                    self.pitch_samples.sort()
                    if self.pitch_samples:
                        mid = self.pitch_samples[len(self.pitch_samples) // 2]
                        self.baseline_pitch = max(-25.0, min(25.0, mid))
                    # 75th percentile approximates open-eye EAR even if the
                    # person blinked during calibration
                    open_ear = self.ear_samples[(len(self.ear_samples) * 3) // 4]
                    self.ear_thresh = min(CFG["ear_thresh"],
                                          max(0.18, 0.75 * open_ear))
            elif looking_away:
                # landmarks are unreliable at an angle; pause the eye/mouth
                # checks, but a sustained look-away is distraction
                state = "looking_away"
                self.closed_since = self.yawn_since = self.nod_since = None
                self.away_since = self.away_since or now
                away_for = now - self.away_since
                if away_for >= CFG["distraction_alarm_secs"]:
                    alerts.append(("distracted-alarm",
                                   "EYES OFF ROAD - LOOK AHEAD"))
                elif away_for >= CFG["distraction_secs"]:
                    warnings.append(("distracted", "LOOKING AWAY TOO LONG"))
            else:
                state = "tracking"
                self.away_since = None

                # 1. microsleep: sustained eye closure
                eyes_closed = ear < self.ear_thresh
                if eyes_closed:
                    self.closed_since = self.closed_since or now
                    if now - self.closed_since >= CFG["closed_secs"]:
                        alerts.append(("microsleep", "MICROSLEEP - EYES CLOSED"))
                else:
                    self.closed_since = None

                # 2. PERCLOS samples (evaluated below, outside this branch)
                self.eye_history.append((now, eyes_closed))

                # 3. yawning: sustained wide-open mouth, counted per minute
                if mar > CFG["yawn_thresh"]:
                    self.yawn_since = self.yawn_since or now
                    if (now - self.yawn_since >= CFG["yawn_secs"]
                            and not self.yawn_open):
                        self.yawn_open = True
                        self.yawn_times.append(now)
                else:
                    self.yawn_since = None
                    self.yawn_open = False
                while self.yawn_times and now - self.yawn_times[0] > 60:
                    self.yawn_times.popleft()
                if len(self.yawn_times) >= CFG["yawns_per_minute"]:
                    warnings.append(("yawns",
                                     f"FATIGUE - {len(self.yawn_times)} YAWNS "
                                     "IN LAST MINUTE"))

                # 4. head nod: pitched down relative to calibrated baseline
                if (pitch is not None
                        and pitch - self.baseline_pitch > CFG["pitch_thresh"]):
                    self.nod_since = self.nod_since or now
                    if now - self.nod_since >= CFG["nod_secs"]:
                        alerts.append(("headdrop", "HEAD DROP - NODDING OFF"))
                else:
                    self.nod_since = None
        else:
            self.closed_since = self.yawn_since = None
            self.nod_since = self.away_since = None
            if self.last_face_time is not None:
                lost_for = now - self.last_face_time
                if lost_for > CFG["face_lost_alarm_secs"]:
                    alerts.append(("facelost-alarm",
                                   "HEAD DOWN / FACE NOT VISIBLE"))
                elif lost_for > CFG["face_lost_secs"]:
                    warnings.append(("facelost", "FACE NOT VISIBLE"))
                else:
                    state = "tracking"

        # PERCLOS from the rolling history every frame, so it survives brief
        # look-aways; hysteresis keeps it from flickering at the threshold
        while (self.eye_history
               and now - self.eye_history[0][0] > CFG["perclos_window"]):
            self.eye_history.popleft()
        if self.calibrated and self.eye_history:
            self.perclos = (sum(c for _, c in self.eye_history)
                            / len(self.eye_history))
            window_full = (now - self.eye_history[0][0]
                           >= CFG["perclos_window"] * 0.8)
            if self.perclos_active:
                self.perclos_active = self.perclos > CFG["perclos_thresh"] * 0.9
            else:
                self.perclos_active = (window_full
                                       and self.perclos > CFG["perclos_thresh"])
            if self.perclos_active:
                warnings.append(("perclos",
                                 f"FATIGUE - EYES CLOSED {self.perclos:.0%} "
                                 f"OF LAST {CFG['perclos_window']:.0f}s"))
        else:
            self.perclos_active = False

        tracked = landmarks is not None and state == "tracking"
        self._log_events(alerts, warnings, tracked, now)

        metrics["perclos"] = round(self.perclos, 2)
        metrics["yawns_per_min"] = len(self.yawn_times)
        return {
            "state": state,
            "calibrated": self.calibrated,
            "alerts": [text for _, text in alerts],
            "warnings": [text for _, text in warnings],
            "metrics": metrics,
            "landmarks": landmarks,
        }

    def _log(self, text):
        with open(os.path.join(CFG["events_dir"], "log.txt"), "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ")
                    + f"[{self.sid[:8]}] " + text + "\n")

    def _log_events(self, alerts, warnings, tracked, now):
        """Text-only evidence trail: no frames are ever written to disk on
        the server. Each incident gets one log line when it starts (alerts
        re-firing within 5 s count as the same incident), and while tracked
        and alarm-free an AWAKE line is written every N seconds so the
        record also shows when someone stayed alert."""
        if not CFG["events_dir"]:
            return

        if alerts:
            if (not self.alert_active
                    and now - self.last_alert_end > 5.0):
                self._log("ALERT " + " | ".join(text for _, text in alerts))
            self.alert_active = True
        elif self.alert_active:
            self.alert_active = False
            self.last_alert_end = now

        if (tracked and not alerts
                and now - self.last_status_log >= CFG["status_log_secs"]):
            self.last_status_log = now
            status = "AWAKE"
            if warnings:
                status += " - " + " | ".join(text for _, text in warnings)
            self._log(status)


def main():
    p = argparse.ArgumentParser(
        description="Serve drowsiness detection to browsers")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model",
                   default="models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--events-dir", default="events",
                   help="folder for the text event log (log.txt); no images "
                        "are stored — pass an empty string to disable logging")
    args = p.parse_args()
    CFG["events_dir"] = args.events_dir

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model)
    index_path = Path(__file__).parent / "web" / "index.html"

    sessions = {}
    sessions_lock = threading.Lock()
    # dlib isn't safe for concurrent use of one predictor; serialize analysis
    pipeline_lock = threading.Lock()

    def get_session(sid, ip):
        with sessions_lock:
            now = time.monotonic()
            for key in [k for k, s in sessions.items()
                        if now - s.last_seen > CFG["session_idle_secs"]]:
                del sessions[key]
            if sid not in sessions:
                session = Session(sid)
                session.ip = ip
                sessions[sid] = session
                print(time.strftime("[%H:%M:%S]"),
                      f"new session {sid[:8]} from {ip} "
                      f"({len(sessions)} active)")
                if CFG["events_dir"]:
                    os.makedirs(CFG["events_dir"], exist_ok=True)
                    session._log(f"SESSION START from {ip}")
            return sessions[sid]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def client_ip(self):
            # behind the tunnel the socket peer is localhost; Cloudflare
            # passes the visitor's real address in CF-Connecting-IP
            # (ngrok and most proxies use X-Forwarded-For)
            return (self.headers.get("CF-Connecting-IP")
                    or (self.headers.get("X-Forwarded-For") or "")
                    .split(",")[0].strip()
                    or self.client_address[0])

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, index_path.read_bytes(),
                           "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/analyze":
                self._send(404, b"not found", "text/plain")
                return
            sid = self.headers.get("X-Session", "")
            length = int(self.headers.get("Content-Length", 0))
            if not sid or not (0 < length <= 2_000_000):
                self._send(400, b"bad request", "text/plain")
                return
            data = self.rfile.read(length)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                self._send(400, b"bad image", "text/plain")
                return
            session = get_session(sid, self.client_ip())
            with pipeline_lock:
                result = session.process(frame, detector, predictor)
            self._send(200, json.dumps(result).encode(), "application/json")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving on http://localhost:{args.port}")
    print("To host online (browsers require HTTPS for camera access), run:")
    print(f"  cloudflared tunnel --url http://localhost:{args.port}")
    print(f"  (or: ngrok http {args.port})")
    print("then share the https:// URL it prints. Ctrl+C stops the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
