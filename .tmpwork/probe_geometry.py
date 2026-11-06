"""Probe 2 — cheek/brow side conventions + z-cue stability across frames.
Prints NUMBERS ONLY.
"""
import sys
sys.path.insert(0, r"D:\project\python\asisten")

import cv2
import numpy as np

from src.face_recognition.landmark_detector import LandmarkDetector

print("index map (x only):")
for name, idx in [('eyeA_inner(133)', 133), ('eyeA_outer(33)', 33),
                  ('eyeB_inner(362)', 362), ('eyeB_outer(263)', 263),
                  ('cheek_234', 234), ('cheek_454', 454),
                  ('brow_70', 70), ('brow_300', 300),
                  ('nose_left(49)', 49), ('nose_right(275)', 275),
                  ('mouth_61', 61), ('mouth_291', 291),
                  ('forehead(10)', 10), ('chin(152)', 152)]:
    print(f"  {name:<20} idx={idx}")

det = LandmarkDetector(static_image_mode=False, refine_landmarks=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA UNAVAILABLE")
    raise SystemExit

rows = []
for i in range(40):
    ok, frame = cap.read()
    if not ok:
        continue
    lm = det.detect_single(frame)
    if lm is None:
        continue
    p = lambda idx: np.array(lm[idx][:2])
    face_h = float(np.linalg.norm(p(10) - p(152)))
    cheek_w = float(np.linalg.norm(p(234) - p(454)))
    inter = float(np.linalg.norm((np.array(lm[159][:2]) + np.array(lm[145][:2])) / 2
                                 - (np.array(lm[386][:2]) + np.array(lm[374][:2])) / 2))
    zdiff = float(lm[1][2] - lm[2][2])
    fs = float(np.sqrt(max(face_h, 1e-9) * max(cheek_w, 1e-9)))
    rows.append((face_h, cheek_w, inter, zdiff, fs,
                 p(234)[0], p(454)[0], p(133)[0], p(362)[0],
                 p(70)[0], p(300)[0]))
    if i == 0:
        print("\nfirst frame x-coords:")
        print(f"  cheek_234.x={rows[0][5]:.4f}  cheek_454.x={rows[0][6]:.4f}")
        print(f"  eyeA_inner133.x={rows[0][7]:.4f}  eyeB_inner362.x={rows[0][8]:.4f}")
        print(f"  brow70.x={rows[0][9]:.4f}  brow300.x={rows[0][10]:.4f}")
cap.release()
det.close()

a = np.array(rows)
print(f"\nframes with landmarks: {len(rows)}")
for name, col in [("face_h", 0), ("cheek_w", 1), ("interocular", 2), ("z_diff", 3), ("face_scale", 4)]:
    v = a[:, col]
    print(f"  {name:<12} mean={v.mean():+.5f} std={v.std(ddof=1):.5f} "
          f"min={v.min():+.5f} max={v.max():+.5f} cv={v.std(ddof=1)/abs(v.mean()):.4f}")

zd = a[:, 3]
fs = a[:, 4]
zn = zd / fs
print(f"  z_diff/scale mean={zn.mean():+.5f} std={zn.std(ddof=1):.5f} "
      f"cv={zn.std(ddof=1)/abs(zn.mean()):.4f}")
print(f"  corr(z_diff, face_scale) = {np.corrcoef(zd, fs)[0,1]:+.4f}")
print(f"  corr(z_diff/scale, face_scale) = {np.corrcoef(zn, fs)[0,1]:+.4f}")
