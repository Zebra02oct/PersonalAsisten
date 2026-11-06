"""End-to-end check: real MediaPipe eye geometry -> BiometricScanner readout.

Deliberately does NOT import src/face_recognition/* (owned by the other agent);
it uses MediaPipe directly with the public FaceMesh landmark indices.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp

from src.face_analysis.biometric_scanner import BiometricScanner

RIGHT_EYE = [362, 263, 384, 386, 380, 382, 373, 374]
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145]

contoh = Path(__file__).resolve().parent.parent / "contoh"

with mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
) as mesh:
    for img in sorted(contoh.glob("*.png")):
        bgr = cv2.imread(str(img))
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        faces = res.multi_face_landmarks or []
        print(f"\n--- {img.name} ({w}x{h}) ---")
        if not faces:
            print("   no face detected by MediaPipe")
            continue
        lm = faces[0].landmark
        print(f"   landmarks: {len(lm)}")

        scanner = BiometricScanner()
        for name, idxs in (("right", RIGHT_EYE), ("left", LEFT_EYE)):
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in idxs]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            radius = max(max(xs) - min(xs), max(ys) - min(ys)) // 2 + 30
            for _ in range(8):
                scanner.draw_scanner(bgr.copy(), (cx, cy), radius)
            print(f"   {name:>5} eye centre=({cx},{cy}) radius={radius} -> {scanner.bio_data}")

        # stability across repeated frames with the same geometry
        scanner2 = BiometricScanner()
        seen = set()
        for _ in range(30):
            scanner2.draw_scanner(bgr.copy(), (cx, cy), radius)
            seen.add(tuple(sorted(scanner2.bio_data.items())))
        print(f"   distinct readouts over 30 identical frames: {len(seen)} -> {seen}")

print("\nE2E OK")
