"""Real-time driver drowsiness detection.

Combines four fatigue signals from the 68-point dlib facial landmarks:
  1. Microsleep  - eyes closed continuously for N frames
  2. PERCLOS     - fraction of eye closure over a rolling window
  3. Yawning     - mouth aspect ratio spikes, counted per minute
  4. Head nod    - head pitched down (solvePnP pose), or face lost from view

Run:  python Drowsiness_Detection.py [--camera 0] [--no-sound] ...
Quit: press 'q' in the video window.
"""

import argparse
import math
import threading
import time
from collections import deque

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance

try:
    import winsound
except ImportError:
    winsound = None  # non-Windows: fall back to terminal bell

# 3D reference model (mm) for nose tip, chin, eye corners, mouth corners
HEAD_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype="double")
HEAD_LANDMARK_IDXS = [30, 8, 36, 45, 48, 54]

GREEN, YELLOW, RED, WHITE = (0, 200, 0), (0, 220, 255), (0, 0, 255), (255, 255, 255)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    # inner-lip points 60-67 of the 68-point model (mouth slice starts at 48)
    inner = mouth[12:20]
    A = distance.euclidean(inner[1], inner[7])
    B = distance.euclidean(inner[2], inner[6])
    C = distance.euclidean(inner[3], inner[5])
    D = distance.euclidean(inner[0], inner[4])
    return (A + B + C) / (3.0 * D)


def head_pose_degrees(shape, frame_shape):
    """(pitch, yaw) of the head in degrees.

    Positive pitch = chin toward chest; yaw = turning left/right.
    """
    h, w = frame_shape[:2]
    image_points = np.array([shape[i] for i in HEAD_LANDMARK_IDXS], dtype="double")
    camera_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],
        [0, 0, 1],
    ], dtype="double")
    ok, rvec, _ = cv2.solvePnP(HEAD_MODEL_POINTS, image_points, camera_matrix,
                               np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw = angles[0], angles[1]
    # frontal pose decomposes near +/-180; fold it back around 0
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return -pitch, yaw


class Alarm:
    """Plays a repeating beep on a background thread while active."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._on = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            if self._on.is_set() and self.enabled:
                if winsound:
                    winsound.Beep(1200, 400)
                    winsound.Beep(900, 400)
                else:
                    print("\a", end="", flush=True)
                    time.sleep(0.8)
            else:
                time.sleep(0.05)

    def set(self, active):
        if active:
            self._on.set()
        else:
            self._on.clear()


def parse_args():
    p = argparse.ArgumentParser(description="Multi-signal drowsiness detection")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--ear-thresh", type=float, default=0.25,
                   help="eye aspect ratio below this counts as closed")
    p.add_argument("--closed-secs", type=float, default=3.0,
                   help="seconds of continuous eye closure before microsleep alert")
    p.add_argument("--perclos-window", type=float, default=30.0,
                   help="rolling window in seconds for PERCLOS")
    p.add_argument("--perclos-thresh", type=float, default=0.30,
                   help="PERCLOS fraction that triggers a fatigue warning")
    p.add_argument("--yawn-thresh", type=float, default=0.45,
                   help="mouth aspect ratio above this counts as yawning")
    p.add_argument("--yawn-secs", type=float, default=1.0,
                   help="seconds of continuous open mouth to register one yawn")
    p.add_argument("--yawns-per-minute", type=int, default=2,
                   help="yawns within 60s that trigger a fatigue warning")
    p.add_argument("--pitch-thresh", type=float, default=20.0,
                   help="downward head pitch in degrees that counts as nodding")
    p.add_argument("--nod-secs", type=float, default=1.5,
                   help="seconds of sustained downward head pitch before alert")
    p.add_argument("--face-lost-secs", type=float, default=2.0,
                   help="seconds without a detected face before warning")
    p.add_argument("--face-lost-alarm-secs", type=float, default=4.0,
                   help="seconds without a detected face before the alarm sounds")
    p.add_argument("--absent-secs", type=float, default=20.0,
                   help="seconds without a face before assuming the driver "
                        "left: alarm stops, state resets, recalibrates on return")
    p.add_argument("--detect-every", type=int, default=5,
                   help="run face detection every Nth frame once locked on "
                        "(landmarks still run every frame)")
    p.add_argument("--detect-width", type=int, default=320,
                   help="downscaled frame width used for face detection")
    p.add_argument("--calibrate-secs", type=float, default=5.0,
                   help="startup seconds spent learning the driver's normal "
                        "head angle and eye openness (0 to disable)")
    p.add_argument("--yaw-limit", type=float, default=25.0,
                   help="head turn in degrees beyond which eye/mouth checks "
                        "pause (mirror and shoulder checks)")
    p.add_argument("--headless", action="store_true",
                   help="run without a video window (Ctrl+C to stop)")
    p.add_argument("--no-sound", action="store_true", help="disable the audio alarm")
    p.add_argument("--model", default="models/shape_predictor_68_face_landmarks.dat",
                   help="path to the dlib 68-landmark model")
    return p.parse_args()


def main():
    args = parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model)
    # upper-body detector distinguishes "head down, driver slumped in the
    # seat" (keep alarming) from "driver left the car" (go quiet)
    body_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_upperbody.xml")
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (m_start, m_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    # ask the camera for a small frame so high-res capture never costs us
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    alarm = Alarm(enabled=not args.no_sound)

    closed_since = None
    yawn_since = None
    nod_since = None
    yawn_open = False
    perclos = 0.0
    perclos_active = False
    ear_thresh = args.ear_thresh
    baseline_pitch = 0.0
    calibrated = args.calibrate_secs <= 0
    calib_end = None
    ear_samples = []
    pitch_samples = []
    eye_history = deque()   # (timestamp, closed?)
    yawn_times = deque()    # timestamps of completed yawns
    last_face_time = None
    bad_reads = 0
    last_status = None
    fps = 0.0
    last_frame_time = None
    face_rect = None
    frame_idx = 0
    last_body_time = None
    prev_seat_roi = None

    if args.headless:
        print("Drowsiness detection running headless - press Ctrl+C to stop.")
    else:
        print("Drowsiness detection running - press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            bad_reads += 1
            if bad_reads > 30:
                print("Camera stopped delivering frames, exiting.")
                break
            time.sleep(0.05)
            continue
        bad_reads = 0

        if frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        if last_frame_time is not None and now > last_frame_time:
            fps = 0.9 * fps + 0.1 / (now - last_frame_time)
        last_frame_time = now
        frame_idx += 1

        # detection is ~97% of the frame cost, so run it downscaled and,
        # once locked on, only every Nth frame; landmarks run every frame
        if face_rect is None or frame_idx % args.detect_every == 0:
            scale = args.detect_width / gray.shape[1]
            small = cv2.resize(gray, (args.detect_width,
                                      int(gray.shape[0] * scale)))
            found = detector(small, 0)
            if found:
                r = max(found, key=lambda r: r.width() * r.height())
                face_rect = dlib.rectangle(
                    int(r.left() / scale), int(r.top() / scale),
                    int(r.right() / scale), int(r.bottom() / scale))
            else:
                face_rect = None

        alerts = []    # (key, text) - red, alarm sounds
        warnings = []  # (key, text) - yellow, no alarm

        if face_rect is not None:
            last_face_time = now
            prev_seat_roi = None
            shape = face_utils.shape_to_np(predictor(gray, face_rect))

            left_eye = shape[l_start:l_end]
            right_eye = shape[r_start:r_end]
            mouth = shape[m_start:m_end]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            pitch, yaw = head_pose_degrees(shape, frame.shape)

            for pts in (left_eye, right_eye, mouth):
                cv2.drawContours(frame, [cv2.convexHull(pts)], -1, GREEN, 1)

            looking_away = yaw is not None and abs(yaw) > args.yaw_limit

            if not calibrated:
                # learn this driver's normal pose and eye openness first, so
                # camera angle and eye shape don't skew the thresholds; only
                # facing-forward frames count, and the window extends until
                # enough of them are seen - a head-down or turned-away start
                # must not poison the baseline
                calib_end = calib_end or now + args.calibrate_secs
                if not looking_away:
                    ear_samples.append(ear)
                    if pitch is not None and abs(pitch) < 35:
                        pitch_samples.append(pitch)
                cv2.putText(frame, "CALIBRATING - look ahead normally", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
                if now >= calib_end and len(ear_samples) >= 30:
                    calibrated = True
                    ear_samples.sort()
                    pitch_samples.sort()
                    if pitch_samples:
                        baseline_pitch = pitch_samples[len(pitch_samples) // 2]
                        baseline_pitch = max(-25.0, min(25.0, baseline_pitch))
                    # 75th percentile approximates open-eye EAR even if the
                    # driver blinked or drooped during calibration
                    open_ear = ear_samples[(len(ear_samples) * 3) // 4]
                    ear_thresh = min(args.ear_thresh, max(0.18, 0.75 * open_ear))
                    print(f"Calibrated: baseline pitch {baseline_pitch:+.0f} deg, "
                          f"eye-closed threshold {ear_thresh:.2f}")
            elif looking_away:
                # eye/mouth landmarks are unreliable at an angle; a mirror or
                # shoulder check must not accumulate toward any alert
                closed_since = yawn_since = nod_since = None
            else:
                # 1. microsleep: sustained eye closure
                eyes_closed = ear < ear_thresh
                if eyes_closed:
                    closed_since = closed_since or now
                    if now - closed_since >= args.closed_secs:
                        alerts.append(("microsleep", "MICROSLEEP - EYES CLOSED"))
                else:
                    closed_since = None

                # 2. collect PERCLOS samples (evaluated below, outside this
                # branch, so the warning survives brief look-aways)
                eye_history.append((now, eyes_closed))

                # 3. yawning: sustained wide-open mouth, counted per minute
                if mar > args.yawn_thresh:
                    yawn_since = yawn_since or now
                    if now - yawn_since >= args.yawn_secs and not yawn_open:
                        yawn_open = True
                        yawn_times.append(now)
                else:
                    yawn_since = None
                    yawn_open = False
                while yawn_times and now - yawn_times[0] > 60:
                    yawn_times.popleft()
                if len(yawn_times) >= args.yawns_per_minute:
                    warnings.append(("yawns", f"FATIGUE - {len(yawn_times)} YAWNS "
                                     "IN LAST MINUTE"))

                # 4. head nod: pitched down relative to calibrated baseline
                if pitch is not None and pitch - baseline_pitch > args.pitch_thresh:
                    nod_since = nod_since or now
                    if now - nod_since >= args.nod_secs:
                        alerts.append(("headdrop", "HEAD DROP - NODDING OFF"))
                else:
                    nod_since = None

            hud = [f"FPS {fps:.0f}", f"EAR {ear:.2f}", f"MAR {mar:.2f}",
                   f"PERCLOS {perclos:.0%}", f"YAWNS/MIN {len(yawn_times)}"]
            if pitch is not None:
                hud.append(f"PITCH {pitch - baseline_pitch:+.0f}")
            if yaw is not None:
                hud.append(f"YAW {yaw:+.0f}")
            if looking_away:
                hud.append("LOOKING AWAY")
            for i, text in enumerate(hud):
                cv2.putText(frame, text, (10, frame.shape[0] - 10 - 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        else:
            closed_since = yawn_since = nod_since = None
            # face gone - check whether someone is still in the seat, so a
            # slumped driver (head down, torso present) is never mistaken
            # for an empty seat. Two independent checks: the upper-body
            # cascade (weak for seated close-ups) and motion in the seat
            # region - even a still person breathes and sways, while an
            # empty seat is static. Every 3rd frame is plenty.
            if frame_idx % 3 == 0:
                scale = args.detect_width / gray.shape[1]
                small = cv2.resize(gray, (args.detect_width,
                                          int(gray.shape[0] * scale)))
                found_body = (not body_cascade.empty() and
                              len(body_cascade.detectMultiScale(
                                  small, 1.1, 3, minSize=(60, 60))) > 0)
                sh, sw = small.shape
                roi = small[sh // 3:, sw // 6:5 * sw // 6]
                motion = False
                if prev_seat_roi is not None and prev_seat_roi.shape == roi.shape:
                    diff = cv2.absdiff(roi, prev_seat_roi)
                    motion = (diff > 20).mean() > 0.015
                prev_seat_roi = roi
                if found_body or motion:
                    last_body_time = now
            body_present = (last_body_time is not None
                            and now - last_body_time < 5.0)

            # a fully dropped head usually leaves the detector's view entirely,
            # so prolonged face loss escalates from warning to alarm; a long
            # loss with no body either means the driver left - go quiet, but
            # never while a body is still in the seat
            if last_face_time is not None:
                lost_for = now - last_face_time
                if lost_for > args.absent_secs and not body_present:
                    last_face_time = None
                    last_body_time = None
                    eye_history.clear()
                    yawn_times.clear()
                    perclos = 0.0
                    perclos_active = False
                    ear_thresh = args.ear_thresh
                    baseline_pitch = 0.0
                    calibrated = args.calibrate_secs <= 0
                    calib_end = None
                    ear_samples = []
                    pitch_samples = []
                    print("Driver left the frame - waiting, will recalibrate on return.")
                elif lost_for > args.face_lost_alarm_secs:
                    if body_present:
                        alerts.append(("headdown-body",
                                       "HEAD DOWN - DRIVER NOT RESPONDING"))
                    else:
                        alerts.append(("facelost-alarm",
                                       "HEAD DOWN / DRIVER NOT VISIBLE"))
                elif lost_for > args.face_lost_secs:
                    warnings.append(("facelost", "FACE NOT VISIBLE"))
            else:
                warnings.append(("waiting", "WAITING FOR DRIVER"))

        # PERCLOS warning: evaluated from the rolling history every frame -
        # even during look-aways and momentary detection misses - so it can't
        # flap; hysteresis keeps a value at the threshold from flickering too
        while eye_history and now - eye_history[0][0] > args.perclos_window:
            eye_history.popleft()
        if calibrated and eye_history:
            perclos = sum(c for _, c in eye_history) / len(eye_history)
            window_full = now - eye_history[0][0] >= args.perclos_window * 0.8
            if perclos_active:
                perclos_active = perclos > args.perclos_thresh * 0.9
            else:
                perclos_active = window_full and perclos > args.perclos_thresh
            if perclos_active:
                warnings.append(("perclos", f"FATIGUE - EYES CLOSED {perclos:.0%} "
                                 f"OF LAST {args.perclos_window:.0f}s"))
        else:
            perclos_active = False

        alarm.set(bool(alerts))

        y = 30
        for _, text in alerts:
            cv2.putText(frame, f"! {text} !", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            y += 28
        for _, text in warnings:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            y += 24

        # log on signal-set changes only, so varying numbers don't spam
        status = tuple(key for key, _ in alerts + warnings)
        if status != last_status:
            line = " | ".join(text for _, text in alerts + warnings) or "OK"
            print(time.strftime("[%H:%M:%S]"), line)
            last_status = status

        if not args.headless:
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    alarm.set(False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
