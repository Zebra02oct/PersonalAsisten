"""Verification harness for src/face_detection/detector.py."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.face_detection.detector import (
    FaceDetector,
    OptimizedFaceDetector,
    open_camera,
    run_demo,
)

contoh = ROOT / "contoh"
images = sorted(contoh.glob("*.png"))

# --- 1. Module imports cleanly ---------------------------------------------
print("1. import OK")

# --- 2. Cascade resolves, and is cwd-independent ----------------------------
os.chdir("C:\\Windows")            # deliberately somewhere else entirely
d = FaceDetector()
print("2. cascade resolved to:", d.cascade_path)
assert d.cascade_path.is_absolute()
assert d.cascade_path.exists()
assert not d.face_cascade.empty()
os.chdir(ROOT)

# explicit bad path -> FileNotFoundError
try:
    FaceDetector(cascade_path="does_not_exist.xml")
    raise AssertionError("expected FileNotFoundError")
except FileNotFoundError as e:
    print("   explicit missing path -> FileNotFoundError OK:", e)

# --- 3. Detection actually runs on the sample images ------------------------
for img in images:
    bgr = cv2.imread(str(img))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces_color = d.detect(bgr)
    faces_gray = d.detect(gray)
    print(f"3. {img.name}: color={faces_color} gray={faces_gray}")
    assert faces_color == faces_gray, "colour and grayscale must agree"

# --- 4. Optimized detector: parameter validation ----------------------------
for bad in (0, -1):
    try:
        OptimizedFaceDetector(detect_every_n_frames=bad)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        print(f"4. detect_every_n_frames={bad} -> ValueError OK: {e}")

for bad in (0, -0.5):
    try:
        OptimizedFaceDetector(detection_scale=bad)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        print(f"   detection_scale={bad} -> ValueError OK: {e}")

# --- 5. Optimized detector runs and reuses results between detections -------
opt = OptimizedFaceDetector(detect_every_n_frames=3, detection_scale=0.5)
bgr = cv2.imread(str(images[-1]))
res = [opt.detect(bgr) for _ in range(9)]
print("5. optimized frames 1..9:", res)
assert res[0] == [] and res[1] == [], "no detection before the first Nth frame"
assert res[2], "detection must happen on the Nth frame"
assert res[3] == res[2] == res[8], "results must be cached between detections"

# --- 6. min_size is scaled consistently between the two modes ----------------
# standard mode: min face = 30px in the ORIGINAL frame
# optimized mode at scale 0.5 must search for 15px in the HALF-SIZE frame,
# i.e. the same 30px face in original coordinates.
seen = {}
for scale in (1.0, 0.5, 0.25):
    o = OptimizedFaceDetector(detect_every_n_frames=1, detection_scale=scale)
    o.detect(bgr)
    seen[scale] = o.last_faces
print("6. faces per detection_scale:", {k: len(v) for k, v in seen.items()})

o = OptimizedFaceDetector(detect_every_n_frames=1, detection_scale=0.5)
o.detect(bgr)
o2 = OptimizedFaceDetector(detect_every_n_frames=1, detection_scale=0.5)
o2.detect(bgr, min_size=(800, 800))
print("   default min_size ->", o.last_faces, " min_size=(800,800) ->", o2.last_faces)
assert o.last_faces, "the sample image must contain a detectable face"
assert o2.last_faces == [], "min_size must reach the cascade (scaled)"

# --- 7. Empty / invalid input is handled ------------------------------------
assert d.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
assert d.detect(None) == []
print("7. empty/None input -> [] OK")

# --- 8. Camera failure path is a clean error, not a crash -------------------
class _FakeCap:
    def __init__(self, *a, **k):
        pass
    def isOpened(self):
        return False
    def release(self):
        pass

real_vc = cv2.VideoCapture
cv2.VideoCapture = _FakeCap
try:
    open_camera(0)
    raise AssertionError("expected RuntimeError")
except RuntimeError as e:
    print("8. camera open failure -> RuntimeError OK:", str(e).splitlines()[0])

    # run_demo must exit(1) instead of raising
    try:
        run_demo(use_optimized=False, camera_index=0)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        print("   run_demo with a broken camera -> SystemExit", e.code)
    except Exception as e:
        print("   run_demo raised", type(e).__name__, e)
        raise
finally:
    cv2.VideoCapture = real_vc

print("\nALL DETECTOR CHECKS PASSED")
