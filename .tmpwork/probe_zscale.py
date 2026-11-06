"""Probe 3 — does MediaPipe's z scale with image scale?

Method (all in memory, NO image files written):
  capture one frame -> run landmark detection on the frame resized by factor s
  -> compare interocular distance, z-difference, face size.

If z scales with the face size, then z_diff/face_scale is a scale-invariant cue.
If z is model-relative (constant), it is not.
"""
import sys
sys.path.insert(0, r"D:\project\python\asisten")

import cv2
import numpy as np

from src.face_recognition.landmark_detector import LandmarkDetector

det = LandmarkDetector(static_image_mode=True, refine_landmarks=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA UNAVAILABLE")
    raise SystemExit
ok, frame = cap.read()
cap.release()
if not ok:
    print("NO FRAME")
    raise SystemExit
print(f"base frame shape = {frame.shape}")

base = None
for s in (1.0, 0.85, 0.70, 0.55, 1.15):
    if s == 1.0:
        img = frame
    else:
        nh, nw = int(frame.shape[0] * s), int(frame.shape[1] * s)
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lm = det.detect_single(img)
    if lm is None:
        print(f"  s={s}: NO FACE")
        continue
    p = lambda idx: np.array(lm[idx][:2])
    eye_a = (np.array(lm[159][:2]) + np.array(lm[145][:2])) / 2
    eye_b = (np.array(lm[386][:2]) + np.array(lm[374][:2])) / 2
    inter = float(np.linalg.norm(eye_a - eye_b))
    face_h = float(np.linalg.norm(p(10) - p(152)))
    cheek_w = float(np.linalg.norm(p(234) - p(454)))
    fs = float(np.sqrt(face_h * cheek_w))
    zdiff = float(lm[1][2] - lm[2][2])
    zrange = float(np.ptp([l[2] for l in lm]))
    if base is None:
        base = dict(inter=inter, face_h=face_h, zdiff=zdiff, fs=fs, zrange=zrange)
    print(f"  s={s:.2f}: inter={inter:.5f} (x{inter/base['inter']:.3f})  "
          f"face_h={face_h:.5f} (x{face_h/base['face_h']:.3f})  "
          f"fs={fs:.5f} (x{fs/base['fs']:.3f})  "
          f"zdiff={zdiff:+.6f} (x{zdiff/base['zdiff']:.3f})  "
          f"zrange={zrange:.5f} (x{zrange/base['zrange']:.3f})  "
          f"zdiff/fs={zdiff/fs:+.5f} (x{(zdiff/fs)/(base['zdiff']/base['fs']):.3f})  "
          f"zdiff/zrange={zdiff/zrange:+.5f}")

det.close()
