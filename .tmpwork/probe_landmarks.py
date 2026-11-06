"""Empirical MediaPipe landmark probe — prints NUMBERS ONLY, never images.

Goal:
  1. Determine which index is the upper lid vs lower lid of each eye.
  2. Determine the true eye centre (midpoint of the lids / iris centre).
  3. Determine which iris landmark set (468-472 vs 473-477) belongs to which eye.
  4. Report nose z-difference stability across camera distance (for the z-cue feature).
"""
import sys
sys.path.insert(0, r"D:\project\python\asisten")

import math
import cv2
import numpy as np

from src.face_recognition.landmark_detector import LandmarkDetector

# Eye landmark rings used by MediaPipe (image-left eye and image-right eye)
EYE_A = {'outer': 33, 'inner': 133, 'cands': [159, 145, 160, 158, 153, 144]}
EYE_B = {'outer': 263, 'inner': 362, 'cands': [386, 374, 384, 380, 373, 382]}


def analyse_eye(lm, eye, label):
    outer = np.array(lm[eye['outer']][:2])
    inner = np.array(lm[eye['inner']][:2])
    axis = inner - outer
    L = np.linalg.norm(axis)
    axis = axis / L
    # perpendicular pointing "down" the face (toward the chin)
    perp = np.array([axis[1], -axis[0]])
    if perp[1] < 0:
        perp = -perp

    print(f"\n--- {label} (outer={eye['outer']}, inner={eye['inner']}) axis_len={L:.5f}")
    offs = {}
    for idx in eye['cands']:
        p = np.array(lm[idx][:2])
        s = float((p - outer) @ perp)
        t = float((p - outer) @ axis)
        offs[idx] = s
        print(f"   idx {idx:>3}: perp_offset={s:+.5f}  along_axis={t:.5f}")
    top = min(offs, key=offs.get)
    bot = max(offs, key=offs.get)
    print(f"   => TOP lid index = {top} (offset {offs[top]:+.5f}), "
          f"BOTTOM lid index = {bot} (offset {offs[bot]:+.5f})")
    mid = (np.array(lm[top][:2]) + np.array(lm[bot][:2])) / 2.0
    print(f"   => geometric eye centre (mid of lids) = ({mid[0]:.5f}, {mid[1]:.5f})")
    return mid


def main():
    det = LandmarkDetector(static_image_mode=True, refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA UNAVAILABLE")
        return
    ok, frame = cap.read()
    if not ok:
        print("NO FRAME")
        cap.release()
        return
    lm = det.detect_single(frame)
    cap.release()
    if lm is None:
        print("NO FACE DETECTED")
        return
    print(f"frame={frame.shape}  n_landmarks={len(lm)}")

    mid_a = analyse_eye(lm, EYE_A, "EYE_A (33/133)")
    mid_b = analyse_eye(lm, EYE_B, "EYE_B (263/362)")

    # iris assignment (only present with refine_landmarks=True)
    if len(lm) > 477:
        c468 = np.array(lm[468][:2])
        c473 = np.array(lm[473][:2])
        print(f"\niris468={c468}  iris473={c473}")
        print(f"dist(iris468, midA)={np.linalg.norm(c468-mid_a):.5f}  "
              f"dist(iris468, midB)={np.linalg.norm(c468-mid_b):.5f}")
        print(f"dist(iris473, midA)={np.linalg.norm(c473-mid_a):.5f}  "
              f"dist(iris473, midB)={np.linalg.norm(c473-mid_b):.5f}")
        r_a = np.mean([np.linalg.norm(np.array(lm[i][:2]) - c468) for i in (469, 470, 471, 472)])
        r_b = np.mean([np.linalg.norm(np.array(lm[i][:2]) - c473) for i in (474, 475, 476, 477)])
        print(f"iris radius 468-set={r_a:.5f}  473-set={r_b:.5f}")
        print(f"midA-midB distance = {np.linalg.norm(mid_a-mid_b):.5f}")
    else:
        print("\nno refined iris landmarks (len<478)")

    # yaw / roll diagnostics
    le = mid_a
    re = mid_b
    nose = np.array(lm[1][:2])
    eye_mid = (le + re) / 2
    inter = np.linalg.norm(re - le)
    yaw = (nose[0] - eye_mid[0]) / inter if inter > 0 else 0.0
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    print(f"\nyaw_ratio(nose offset / interocular) = {yaw:+.4f}")
    print(f"roll (eye line angle, deg) = {roll:+.2f}")
    print(f"nose tip z = {lm[1][2]:+.6f}   nose bottom z = {lm[2][2]:+.6f}  "
          f"z-diff = {lm[1][2]-lm[2][2]:+.6f}")
    det.close()


if __name__ == "__main__":
    main()
