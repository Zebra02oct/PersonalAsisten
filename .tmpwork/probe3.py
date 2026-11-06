import sys, os
sys.path.insert(0, r"D:\project\python\asisten")
import cv2, numpy as np
out = r"D:\project\python\asisten\.tmpwork"
cap = cv2.VideoCapture(0)
for i in range(3):
    ok, f = cap.read()
    if ok:
        p = os.path.join(out, f"cap_{i}.png")
        cv2.imwrite(p, f)
        print("saved", p, f.shape)
cap.release()
