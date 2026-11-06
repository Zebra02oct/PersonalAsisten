"""Verification harness for src/face_analysis/biometric_scanner.py (readout determinism)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.face_analysis.biometric_scanner import BiometricScanner

frame = np.zeros((480, 640, 3), dtype=np.uint8)

# --- 1. No eye detected -> placeholders, not invented numbers ---------------
s = BiometricScanner()
assert s.bio_data == {'IRIS': '--', 'PATTERN': '--', 'DIA': '--', 'PUPIL': '--'}, s.bio_data
print("1. no-eye placeholders OK ->", s.bio_data)

# --- 2. Determinism: same geometry -> same readout across fresh instances ---
a = BiometricScanner()
b = BiometricScanner()
for _ in range(5):
    a.draw_scanner(frame.copy(), (320, 240), 60)
    b.draw_scanner(frame.copy(), (320, 240), 60)
assert a.bio_data == b.bio_data, (a.bio_data, b.bio_data)
print("2. deterministic across instances OK ->", a.bio_data)

# --- 3. Stability across many frames of the SAME eye (with jitter) ----------
s = BiometricScanner()
readouts = []
for i in range(60):
    # simulate landmark jitter of +-2 px
    cx = 320 + (2 if i % 2 else -2)
    cy = 240 + (1 if i % 3 else -1)
    s.draw_scanner(frame.copy(), (cx, cy), 60)
    readouts.append(dict(s.bio_data))

labels = {(r['IRIS'], r['PATTERN']) for r in readouts}
assert len(labels) == 1, f"labels not stable: {labels}"
dias = {r['DIA'] for r in readouts}
pups = {r['PUPIL'] for r in readouts}
print("3. stable labels:", labels, " DIA values seen:", sorted(dias), " PUPIL values seen:", sorted(pups))
assert len(dias) <= 2, dias           # tiny jitter -> tiny numeric wobble only
assert len(pups) <= 2, pups

# --- 4. Values TRACK the geometry (different eye size -> different values) --
sizes = {}
for r_ in (30, 45, 60, 90, 120):
    s2 = BiometricScanner()
    for _ in range(10):
        s2.draw_scanner(frame.copy(), (320, 240), r_)
    sizes[r_] = s2.bio_data
    print(f"   radius {r_:>3}px -> {s2.bio_data}")

dias = [sizes[r]['DIA'] for r in (30, 45, 60, 90, 120)]
nums = [float(d.replace('mm', '')) for d in dias]
assert nums == sorted(nums), nums   # monotonically increasing with eye size
assert len(set(nums)) == len(nums), nums

# --- 5. Different eye geometry -> possibly different label ----------------
labels_seen = set()
for r_ in (25, 40, 55, 70, 85, 100, 115, 130):
    for cx in (150, 320, 480):
        s3 = BiometricScanner()
        for _ in range(5):
            s3.draw_scanner(frame.copy(), (cx, 240), r_)
        labels_seen.add((s3.bio_data['IRIS'], s3.bio_data['PATTERN']))
print("5. distinct label pairs across geometries:", len(labels_seen), sorted(labels_seen))
assert len(labels_seen) > 1

# --- 6. draw_minimal also syncs -------------------------------------------
s4 = BiometricScanner()
s4.draw_minimal(frame.copy(), (320, 240), 60)
assert s4.bio_data['DIA'] != '--', s4.bio_data
print("6. draw_minimal syncs OK ->", s4.bio_data)

# --- 7. Public API unchanged ------------------------------------------------
for name in ('start_scan', 'stop_scan', 'update', 'draw_scanner', 'draw_minimal',
             'reset_profile', 'bio_data', 'scan_active', 'scan_progress', 'scan_angle'):
    assert hasattr(s, name), name
print("7. public API OK")

# --- 8. reset_profile clears the readout -----------------------------------
s.reset_profile()
assert s.bio_data['DIA'] == '--', s.bio_data
print("8. reset_profile OK ->", s.bio_data)

print("\nALL SCANNER CHECKS PASSED")
