# Drowsiness Detection 😴 🚗

Real-time driver drowsiness detection from a webcam, using OpenCV and dlib's
68-point facial landmarks. Instead of only checking for closed eyes, it
combines four fatigue signals:

| Signal | How it's measured | Response |
|---|---|---|
| **Microsleep** | Eye Aspect Ratio (EAR) below threshold for ~20 consecutive frames | 🔴 Alert + alarm |
| **PERCLOS** | % of eye closure over a rolling 30 s window (standard fatigue metric) | 🟡 Warning |
| **Yawning** | Mouth Aspect Ratio (MAR) spikes, ≥2 yawns within a minute | 🟡 Warning |
| **Head nodding** | Head pitch from solvePnP pose estimation; face lost from view | 🔴 Alert / 🟡 Warning |

Red alerts sound a repeating audio alarm (Windows `winsound`; terminal bell elsewhere).

## Setup

```
pip install -r requirements.txt
```

The dlib 68-landmark model is included in `models/`.

## Run

```
python Drowsiness_Detection.py
```

Press `q` in the video window to quit. Useful flags:

```
--camera 1              use a different webcam
--no-sound              disable the audio alarm
--ear-thresh 0.25       eye-closed threshold (lower = less sensitive)
--closed-secs 3.0       seconds of closed eyes before microsleep alert
--yawn-thresh 0.55      mouth-open threshold for yawns
--pitch-thresh 20       downward head angle (degrees) that counts as nodding
```

Run `python Drowsiness_Detection.py --help` for the full list.

## Low-spec devices

Face detection dominates the frame cost, so it runs on a downscaled frame
and, once a face is locked, only every 5th frame — landmarks (which are
~30x cheaper) still run every frame, so alert timing is unaffected. This
cuts per-frame CPU work by roughly 10x versus naive per-frame detection.
On very weak hardware you can push further:

```
--detect-every 10       re-detect the face less often
--detect-width 240      detect on an even smaller frame (don't go below ~240)
```

The bottom-left HUD shows live FPS. Alert thresholds are measured in
seconds, not frames, so behavior stays consistent at any frame rate.

## How it works

Each eye is described by 6 landmark points; the EAR is the ratio of the
vertical eye openings to the horizontal width — it collapses toward 0 when
the eye closes. The MAR does the same for the inner lips. Head pose is
recovered by fitting six landmarks (nose, chin, eye and mouth corners) to a
3D face model with `cv2.solvePnP`, and the pitch angle indicates a drooping
head. Live values for all metrics are shown in the bottom-left HUD.

## Credits

Based on the original [Drowsiness_Detection](https://github.com/akshaybahadur21/Drowsiness_Detection)
by Akshay Bahadur (MIT license) and the EAR technique from
[PyImageSearch](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/).
