"""Probe 4 — simulate the subject being FARTHER AWAY by padding the frame.

The face then occupies 1/k of the frame, so all normalised x/y distances must
shrink by 1/k if the pipeline is truly image-relative.  We check whether the
z-difference shrinks too (=> z is image-relative, safe to divide by face size)
or stays constant (=> z is model-relative, i.e. NOT comparable across distance).

All processing in memory; NO image files are written.
"""
import sys
sys.path.insert(0, r"D:\project\python\asisten")

import cv2
import numpy as np

from src.face_recognition.landmark_detector import LandmarkDetector

det = LandmarkDetector(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA UNAVAILABLE")
    raise SystemExit
ok, frame = cap.read()
cap.release()
if not ok:
    print("NO FRAME")
    raise SystemExit
h, w = frame.shape[:2]
print(f"base frame shape = {frame.shape}")

base = None
for k in (1.0, 1.25, 1.5, 1.75):
    nh, nw = int(h * k), int(w * k)
    canvas = np.zeros((nh, nw, 3), dtype=np.uint8)
    oy, ox = (nh - h) // 2, (nw - w) // 2
    canvas[oy:oy + h, ox:ox + w] = frame
    lm = det.detect_single(canvas)
    if lm is None:
        print(f"  k={k}: NO FACE")
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
        base = dict(inter=inter, fs=fs, zdiff=zdiff, zrange=zrange)
    print(f"  k={k:.1f} (face is {1/k:.3f}x of frame): "
          f"inter={inter:.5f} (expect x{1/k:.3f}, got x{inter/base['inter']:.3f})  "
          f"fs={fs:.5f} (got x{fs/base['fs']:.3f})  "
          f"zdiff={zdiff:+.6f} (got x{zdiff/base['zdiff']:.3f})  "
          f"zrange={zrange:.5f} (got x{zrange/base['zrange']:.3f})  "
          f"zdiff/fs={zdiff/fs:+.5f} (got x{(zdiff/fs)/(base['zdiff']/base['fs']):.3f})  "
          f"zdiff/zrange={zdiff/zrange:+.5f} (got x{(zdiff/zrange)/(base['zdiff']/base['zrange']):.3f})")

det.close()
