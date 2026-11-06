"""Headless exercise of run_demo() with a fake camera + fake GUI backend."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.face_detection import detector as det

events = []


class FakeCap:
    def __init__(self, *a, **k):
        self._released = False
        self._n = 0

    def isOpened(self):
        return True

    def read(self):
        self._n += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:300, 200:400] = 200
        return True, frame

    def release(self):
        events.append("release")
        self._released = True


KEYS = iter([255] * 5 + [ord('o')] + [255] * 5 + [ord('q')])


def fake_waitKey(delay):
    try:
        return next(KEYS)
    except StopIteration:
        return ord('q')


orig = {
    "VideoCapture": cv2.VideoCapture,
    "namedWindow": cv2.namedWindow,
    "resizeWindow": cv2.resizeWindow,
    "imshow": cv2.imshow,
    "waitKey": cv2.waitKey,
    "destroyAllWindows": cv2.destroyAllWindows,
}

cv2.VideoCapture = FakeCap
cv2.namedWindow = lambda *a, **k: events.append("namedWindow")
cv2.resizeWindow = lambda *a, **k: events.append("resizeWindow")
cv2.imshow = lambda *a, **k: events.append("imshow")
cv2.waitKey = fake_waitKey
cv2.destroyAllWindows = lambda: events.append("destroyAllWindows")

try:
    det.run_demo(use_optimized=False, camera_index=0)
finally:
    for k, v in orig.items():
        setattr(cv2, k, v)

print("\nevents:", events)
assert "release" in events, "camera was not released"
assert "destroyAllWindows" in events, "window was not closed"
print("\nHEADLESS DEMO OK")
