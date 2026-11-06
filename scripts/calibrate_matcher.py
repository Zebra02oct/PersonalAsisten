"""
Calibration harness for the shared biometric matcher.

This script measures the two distributions the login threshold has to
separate, evaluates several candidate metrics and prints the numbers.
It is *the* source of the value of
``src.face_recognition.matcher.MATCH_CONFIDENCE_THRESHOLD``.

What it measures
----------------
1. SAME-PERSON distances
   A burst of webcam frames is captured in memory, the 30 geometric features
   are extracted from every frame that contains a detectable face, five of
   those frames are used as the "enrollment" set (exactly what the real
   enrollment produces) and every remaining frame is used as a query.
   The distance of each query to the reference signature is recorded.

2. SAME-PERSON under POSE CHANGE
   The landmark cloud of a real frame is rotated around the vertical axis
   through the nose (a genuine 3-D rotation of the MediaPipe point cloud) to
   synthesise a head that is turned by 10 / 20 / 30 degrees.  This is what
   the enrollment captures for the "turn left" / "turn right" samples.

3. DIFFERENT-PERSON distances
   A synthetic face is built by deforming the *real* landmark cloud with
   genuinely different human proportions: face aspect ratio, eye separation,
   eye size, nose length, nose depth, mouth width, jaw width and face height
   are each rescaled by an anatomically plausible factor drawn at random.
   The deformed landmarks are pushed through the *same* FeatureExtractor, so
   the resulting vectors live in exactly the same space as a real face.

   NOTE: this is a simulation, not a second real human.  The webcam in front
   of the machine only ever sees one person, so a real "impostor" sample
   cannot be collected here.  The deformation ranges are documented below.

4. METRIC VARIANT GRID
   Several combinations of spread estimator and distance aggregation are
   evaluated on exactly the same data so the best separating one can be
   chosen from measurements instead of guesses.

NO IMAGE FILE IS EVER WRITTEN OR READ - every frame is processed in memory and
only numbers are printed.

Usage:
    python scripts/calibrate_matcher.py [--frames 240] [--enroll 5]
                                         [--impostors 200] [--json out.json]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.face_recognition.landmark_detector import LandmarkDetector
from src.face_recognition.feature_extractor import FeatureExtractor


# ---------------------------------------------------------------------------
# Landmark groups used by the synthetic deformation
# ---------------------------------------------------------------------------
GROUPS = {
    "left_eye": [33, 133, 159, 145, 158, 153, 160, 144, 157, 154, 155, 173],
    "right_eye": [362, 263, 386, 374, 380, 382, 384, 373, 385, 387, 398],
    "left_brow": [70, 107, 66, 105, 46, 53, 52, 65, 55, 70],
    "right_brow": [300, 336, 296, 334, 276, 283, 282, 295, 285, 300],
    "nose": [1, 2, 4, 5, 6, 19, 94, 97, 98, 129, 168, 195, 197, 327, 358,
             49, 275, 64, 294, 115, 344, 220, 440],
    "mouth": [61, 291, 13, 14, 0, 37, 267, 78, 308, 40, 310, 11, 12, 17,
              18, 84, 87, 88, 95, 146, 181, 191, 199, 208],
    "jaw_left": [93, 132, 58, 172, 136, 150, 149, 176, 148,
                 377, 400, 378, 379, 365, 397],
    "jaw_right": [323, 361, 288, 397, 365, 379, 378, 400, 377],
    "cheek_l": [50, 101, 118, 117, 205, 206, 207, 187, 111, 117, 118],
    "cheek_r": [280, 330, 347, 346, 425, 426, 427, 411, 312, 346, 347],
    "forehead": [10, 151, 9, 8, 107, 336, 297, 332, 284, 332, 333],
    "iris_left": [468, 469, 470, 471, 472],
    "iris_right": [473, 474, 475, 476, 477],
}

# Anatomically plausible deformation ranges.  A human face aspect ratio
# (width / height) varies roughly between 0.65 and 0.95, so a factor of
# 0.78 .. 1.30 around the reference face spans a good part of that range.
DEFORM_RANGES = {
    "aspect": (0.78, 1.30),     # overall width of the face
    "height": (0.90, 1.15),     # forehead -> chin
    "eye_sep": (0.82, 1.22),    # distance between the two eyes
    "eye_size": (0.82, 1.22),   # size of each eye
    "nose_len": (0.80, 1.25),   # vertical length of the nose
    "nose_depth": (0.55, 1.70), # protrusion of the nose (z)
    "mouth_w": (0.78, 1.28),    # width of the mouth
    "jaw_w": (0.85, 1.20),      # width of the jaw / cheeks
}


def _pt(landmarks, idx):
    return np.array(landmarks[idx][:3], dtype=float)


def deform_face(landmarks, rng, params=None):
    """
    Build a synthetic face with different proportions.

    The real landmark cloud is deformed by rescaling anatomically meaningful
    groups (eyes, brows, nose, mouth, jaw, cheeks).  Every deformation is
    applied *after* the global aspect / height change and about the group's
    own (already transformed) centre, so the result is a clean, anatomically
    plausible face rather than a random scatter of landmarks.

    The deformed cloud stays in the same MediaPipe landmark space, so the
    normal feature extractor can be applied to it.
    """
    lm = np.array([[p[0], p[1], p[2]] for p in landmarks], dtype=float)
    if params is None:
        params = {k: rng.uniform(*v) for k, v in DEFORM_RANGES.items()}

    # --- 1. global aspect ratio + face height, about the face centre ------
    face_c = (lm[234] + lm[454]) / 2.0
    cx, cy = float(face_c[0]), float(face_c[1])
    lm[:, 0] = cx + (lm[:, 0] - cx) * params["aspect"]
    lm[:, 1] = cy + (lm[:, 1] - cy) * params["height"]

    # group centres in the *transformed* space
    eye_l_c = (lm[133] + lm[33]) / 2.0
    eye_r_c = (lm[362] + lm[263]) / 2.0
    nose_c = lm[1].copy()
    mouth_c = lm[0].copy()

    # --- 2. eye size (each eye rescaled about its own centre) ------------
    for idx in GROUPS["left_eye"] + GROUPS["iris_left"]:
        v = lm[idx] - eye_l_c
        v[0] *= params["eye_size"]
        v[1] *= params["eye_size"]
        lm[idx] = eye_l_c + v
    for idx in GROUPS["right_eye"] + GROUPS["iris_right"]:
        v = lm[idx] - eye_r_c
        v[0] *= params["eye_size"]
        v[1] *= params["eye_size"]
        lm[idx] = eye_r_c + v

    # --- 3. eye separation: move the two eye groups apart ------------------
    half = float(np.linalg.norm(eye_r_c[:2] - eye_l_c[:2])) / 2.0
    delta = (params["eye_sep"] - 1.0) * half
    for idx in GROUPS["left_eye"] + GROUPS["iris_left"]:
        lm[idx][0] -= delta
    for idx in GROUPS["right_eye"] + GROUPS["iris_right"]:
        lm[idx][0] += delta

    # --- 4. nose length and depth (about the nose tip) -------------------
    for idx in GROUPS["nose"]:
        v = lm[idx] - nose_c
        v[1] *= params["nose_len"]
        v[2] *= params["nose_depth"]
        lm[idx] = nose_c + v

    # --- 5. mouth width (about the mouth centre) -------------------------
    for idx in GROUPS["mouth"]:
        v = lm[idx] - mouth_c
        v[0] *= params["mouth_w"]
        lm[idx] = mouth_c + v

    # --- 6. jaw width (about the mouth centre, both sides) ---------------
    for idx in GROUPS["jaw_left"] + GROUPS["jaw_right"]:
        v = lm[idx] - mouth_c
        v[0] *= params["jaw_w"]
        lm[idx] = mouth_c + v

    # --- 7. cheeks (about the face centre) -------------------------------
    for idx in GROUPS["cheek_l"]:
        lm[idx][0] = cx + (lm[idx][0] - cx) * params["jaw_w"]
    for idx in GROUPS["cheek_r"]:
        lm[idx][0] = cx + (lm[idx][0] - cx) * params["jaw_w"]

    # --- 8. eyebrows follow the eyes -------------------------------------
    for idx in GROUPS["left_brow"]:
        lm[idx][0] -= delta * 0.8
    for idx in GROUPS["right_brow"]:
        lm[idx][0] += delta * 0.8

    return lm, params


def rotate_yaw(landmarks, degrees):
    """
    Rotate the 3-D landmark cloud around the vertical axis through the nose.

    This synthesises a head that is turned by ``degrees`` - the landmarks
    stay anatomically consistent because the whole cloud is rotated rigidly.
    """
    lm = np.array([[p[0], p[1], p[2]] for p in landmarks], dtype=float)
    nose = lm[1].copy()
    ang = math.radians(degrees)
    c, s = math.cos(ang), math.sin(ang)

    rel = lm - nose
    x = rel[:, 0] * c + rel[:, 2] * s
    z = -rel[:, 0] * s + rel[:, 2] * c
    lm[:, 0] = nose[0] + x
    lm[:, 2] = nose[2] + z
    return lm


# ---------------------------------------------------------------------------
# Candidate metrics
# ---------------------------------------------------------------------------
def build_std(observed, mode):
    """Spread estimators."""
    observed = np.asarray(observed, dtype=float)
    positive = observed[observed > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    if mode == "raw":
        return np.maximum(observed, 1e-6)
    if mode == "floor":                     # max(observed, 0.5 * median)
        return np.maximum(observed, max(0.5 * median, 1e-6))
    if mode == "floor1":                    # max(observed, 1.0 * median)
        return np.maximum(observed, max(median, 1e-6))
    if mode == "shrink":                    # 50 % of the variance pooled
        pooled = median ** 2
        return np.sqrt(np.maximum(0.5 * observed ** 2 + 0.5 * pooled, 1e-12))
    if mode == "shrink75":
        pooled = median ** 2
        return np.sqrt(np.maximum(0.25 * observed ** 2 + 0.75 * pooled, 1e-12))
    if mode == "shrink25":
        pooled = median ** 2
        return np.sqrt(np.maximum(0.75 * observed ** 2 + 0.25 * pooled, 1e-12))
    if mode == "pooled":                    # fully pooled
        return np.full_like(observed, max(median, 1e-6))
    raise ValueError(mode)


def aggregate(z, agg, param):
    """Distance aggregations over the z-scores."""
    z2 = z * z
    if agg == "rms":
        return float(math.sqrt(np.mean(z2)))
    if agg == "mae":
        return float(np.mean(np.abs(z)))
    if agg == "max":
        return float(np.max(np.abs(z)))
    if agg == "topk":
        k = int(param)
        s = np.sort(z2)[::-1][:k]
        return float(math.sqrt(np.mean(s)))
    if agg == "trim":                     # RMS after dropping the top-k |z|
        k = int(param)
        s = np.sort(z2)[:len(z2) - k] if k else z2
        return float(math.sqrt(np.mean(s)))
    if agg == "lse":
        T = float(param)
        m = np.max(z2)
        return float(math.sqrt(T * (m + math.log(np.mean(np.exp((z2 - m) / T))))))
    if agg == "pnorm":
        p = float(param)
        return float(np.sum(np.abs(z) ** p) ** (1.0 / p))
    raise ValueError(agg)


def distance_of(query, ref_features, ref_std, agg, param, clip):
    z = (np.asarray(query, dtype=float) - ref_features) / ref_std
    z = np.clip(z, -clip, clip)
    return aggregate(z, agg, param)


def stats_line(name, d):
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return f"  {name:<30} (none)"
    return (f"  {name:<30} n={d.size:<4} min={d.min():6.3f} p05={np.percentile(d,5):6.3f} "
            f"p50={np.percentile(d,50):6.3f} p95={np.percentile(d,95):6.3f} "
            f"p99={np.percentile(d,99):6.3f} max={d.max():6.3f}")


def capture(frames_wanted, detector, extractor, pause):
    """Capture frames from the webcam and extract features. No image files."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA UNAVAILABLE")
        return [], []

    features, landmark_sets = [], []
    attempts = 0
    max_attempts = frames_wanted * 3
    while len(features) < frames_wanted and attempts < max_attempts:
        attempts += 1
        ok, frame = cap.read()
        if not ok:
            continue
        lm = detector.detect_single(frame)
        if lm:
            try:
                feats = extractor.extract(lm)
            except ValueError:
                feats = None
            if feats is not None:
                features.append(feats)
                landmark_sets.append(
                    np.array([[p[0], p[1], p[2]] for p in lm], dtype=float))
        if pause:
            time.sleep(pause)
    cap.release()
    return features, landmark_sets


def main():
    ap = argparse.ArgumentParser(description="Calibrate the biometric matcher")
    ap.add_argument("--frames", type=int, default=240,
                    help="number of webcam frames to capture")
    ap.add_argument("--enroll", type=int, default=5,
                    help="number of enrollment samples")
    ap.add_argument("--impostors", type=int, default=200,
                    help="number of synthetic different-person faces")
    ap.add_argument("--pause", type=float, default=0.12,
                    help="pause between frames so the burst covers real time")
    ap.add_argument("--json", default=None,
                    help="optional path to write the raw numbers to (JSON)")
    ap.add_argument("--cache", default=".tmpwork/agent1/calib_cache.npz",
                    help="path of the capture cache (.npz, no images)")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="reuse a cached capture instead of using the camera")
    args = ap.parse_args()

    rng = np.random.default_rng(20241103)

    print("=" * 96)
    print("  BIOMETRIC MATCHER CALIBRATION")
    print("=" * 96)
    print(f"  frames to capture  : {args.frames} (pause {args.pause}s)")
    print(f"  enrollment samples : {args.enroll}")
    print(f"  synthetic impostors: {args.impostors}")
    print()

    detector = LandmarkDetector(static_image_mode=False, refine_landmarks=True)
    extractor = FeatureExtractor()

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists() and args.reuse_cache:
        blob = np.load(cache, allow_pickle=False)
        feats_arr = blob["features"]
        lm_arr = blob["landmarks"]
        features = [f for f in feats_arr]
        landmark_sets = [l for l in lm_arr]
        capture_time = 0.0
        print(f"\n  reusing cached capture from {cache} "
              f"({len(features)} frames)")
    else:
        t0 = time.time()
        features, landmark_sets = capture(args.frames, detector, extractor, args.pause)
        capture_time = time.time() - t0
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache,
                                features=np.array(features),
                                landmarks=np.array(landmark_sets))
            print(f"\n  capture cached to {cache}")
    detector.close()

    if len(features) < args.enroll + 10:
        print(f"ERROR: only {len(features)} usable frames captured - "
              f"cannot calibrate. Is somebody in front of the camera?")
        return 1

    print(f"\n  captured {len(features)} usable frames in {capture_time:.1f}s "
          f"(face detected + {len(features[0])} features extracted)")

    feats = np.array(features)

    # ---- reference built from the enrollment samples ----------------------
    idx = np.linspace(0, len(feats) - 1, args.enroll).astype(int)
    enroll_feats = feats[idx]
    ref_mean = enroll_feats.mean(axis=0)
    observed_std = enroll_feats.std(axis=0, ddof=1)
    print(f"  enrollment frame indices : {idx.tolist()}")
    print(f"  observed std (min/med/max): {observed_std.min():.6f} / "
          f"{np.median(observed_std):.6f} / {observed_std.max():.6f}")

    enroll_set = set(idx.tolist())
    same_query = [f for i, f in enumerate(feats) if i not in enroll_set]

    # ---- synthetic impostors ---------------------------------------------
    base = landmark_sets[len(landmark_sets) // 2]
    impostor_feats = []
    for _ in range(args.impostors):
        synth, _p = deform_face(base, rng)
        try:
            impostor_feats.append(extractor.extract(synth))
        except ValueError:
            pass
    impostor_feats = np.array(impostor_feats)
    print(f"  synthetic impostors built: {len(impostor_feats)}")

    # how different are the impostors really, in raw feature terms?
    ref_norm = float(np.linalg.norm(ref_mean))
    rel_same = np.array([float(np.linalg.norm(f - ref_mean)) / ref_norm
                         for f in same_query])
    rel_imp = np.array([float(np.linalg.norm(f - ref_mean)) / ref_norm
                        for f in impostor_feats])
    print("\n  relative feature-vector difference (||q - ref|| / ||ref||):")
    print(stats_line("SAME person", rel_same))
    print(stats_line("DIFFERENT person", rel_imp))

    # ---- pose variants ---------------------------------------------------
    pose_feats = {}
    for deg in (10, 20, 30):
        vals = []
        for lm in landmark_sets:
            try:
                vals.append(extractor.extract(rotate_yaw(lm, deg)))
            except ValueError:
                pass
        pose_feats[deg] = np.array(vals)

    # ---- per-dimension diagnostic ----------------------------------------
    print("\n" + "=" * 96)
    print("  PER-DIMENSION DIAGNOSTIC (std estimator = floor)")
    print("=" * 96)
    std_floor = build_std(observed_std, "floor")
    z_same = (np.array(same_query) - ref_mean) / std_floor
    z_imp = (impostor_feats - ref_mean) / std_floor
    print(f"  {'dim':>4} {'mean':>11} {'std':>10} {'same |z| p50':>13} "
          f"{'same |z| p99':>13} {'imp |z| p50':>12} {'imp |z| min':>12}")
    for i in range(feats.shape[1]):
        print(f"  {i:>4} {ref_mean[i]:>11.5f} {std_floor[i]:>10.6f} "
              f"{np.percentile(np.abs(z_same[:,i]),50):>13.3f} "
              f"{np.percentile(np.abs(z_same[:,i]),99):>13.3f} "
              f"{np.percentile(np.abs(z_imp[:,i]),50):>12.3f} "
              f"{np.abs(z_imp[:,i]).min():>12.3f}")

    # ---- metric variant grid ---------------------------------------------
    variants = []
    for std_mode in ("raw", "floor", "floor1", "shrink", "shrink75",
                     "shrink25", "pooled"):
        for agg, param, clip in (
            ("rms", 0, 3.0),
            ("rms", 0, 5.0),
            ("rms", 0, 8.0),
            ("mae", 0, 3.0),
            ("mae", 0, 5.0),
            ("trim", 1, 3.0),
            ("trim", 2, 3.0),
            ("trim", 3, 3.0),
            ("trim", 2, 5.0),
            ("trim", 3, 5.0),
            ("max", 0, 3.0),
            ("max", 0, 5.0),
            ("topk", 3, 5.0),
            ("topk", 5, 5.0),
            ("topk", 8, 5.0),
            ("topk", 5, 8.0),
            ("lse", 2.0, 5.0),
            ("lse", 4.0, 5.0),
            ("lse", 4.0, 8.0),
            ("pnorm", 3.0, 5.0),
            ("pnorm", 4.0, 8.0),
        ):
            variants.append((std_mode, agg, param, clip))

    print("\n" + "=" * 96)
    print("  METRIC VARIANT GRID")
    print("  separation = impostor_p01 - same_p99   (higher is better)")
    print("  'bal'      = distance threshold with equal tails (impostor_p01)")
    print("=" * 96)
    print(f"  {'std':<9} {'agg':<6} {'par':>5} {'clip':>5} "
          f"{'same p50':>9} {'same p99':>9} {'same max':>9} "
          f"{'imp p01':>9} {'imp p50':>9} {'sep':>8} {'bal':>7}")

    results = []
    for std_mode, agg, param, clip in variants:
        std = build_std(observed_std, std_mode)
        d_same = np.array([distance_of(f, ref_mean, std, agg, param, clip)
                           for f in same_query])
        d_imp = np.array([distance_of(f, ref_mean, std, agg, param, clip)
                          for f in impostor_feats])
        s99 = float(np.percentile(d_same, 99))
        s_max = float(d_same.max())
        i01 = float(np.percentile(d_imp, 1))
        i50 = float(np.percentile(d_imp, 50))
        sep = i01 - s99
        # balanced threshold: reject 1 % of impostors, accept 99 % of same
        bal = i01
        results.append(dict(std=std_mode, agg=agg, param=param, clip=clip,
                            same_p50=float(np.percentile(d_same, 50)),
                            same_p99=s99, same_max=s_max,
                            imp_p01=i01, imp_p50=i50, sep=sep, bal=bal))
        print(f"  {std_mode:<9} {agg:<6} {param:>5} {clip:>5.1f} "
              f"{np.percentile(d_same,50):>9.3f} {s99:>9.3f} {s_max:>9.3f} "
              f"{i01:>9.3f} {i50:>9.3f} {sep:>8.3f} {bal:>7.3f}")

    results.sort(key=lambda r: (-r["sep"], r["same_p99"]))
    print("\n  BEST 8 VARIANTS (by separation):")
    for r in results[:8]:
        print(f"    std={r['std']:<9} agg={r['agg']:<6} par={r['param']:>4} "
              f"clip={r['clip']:.1f}  sep={r['sep']:+.3f}  "
              f"same p99={r['same_p99']:.3f}  imp p01={r['imp_p01']:.3f}")

    best = results[0]
    std = build_std(observed_std, best["std"])
    agg, param, clip = best["agg"], best["param"], best["clip"]
    d_same = np.array([distance_of(f, ref_mean, std, agg, param, clip)
                       for f in same_query])
    d_imp = np.array([distance_of(f, ref_mean, std, agg, param, clip)
                      for f in impostor_feats])

    print("\n" + "=" * 96)
    print(f"  DETAIL FOR THE SELECTED VARIANT "
          f"(std={best['std']}, agg={best['agg']}, par={param}, clip={clip})")
    print("=" * 96)
    print(stats_line("SAME person (out of sample)", d_same))
    for deg, pf in pose_feats.items():
        d = np.array([distance_of(f, ref_mean, std, agg, param, clip) for f in pf])
        print(stats_line(f"SAME person (yaw {deg} deg)", d))
    print(stats_line("DIFFERENT person (synthetic)", d_imp))

    # threshold recommendation: keep the same-person false rejection at ~1 %
    same_thr = float(np.percentile(d_same, 99))
    imp_at = float(np.mean(d_imp <= same_thr))
    print(f"\n  threshold at same-person p99 = {same_thr:.3f}")
    print(f"    -> impostors accepted: {imp_at:.2%}")
    imp_thr = float(np.percentile(d_imp, 1))
    same_at = float(np.mean(d_same <= imp_thr))
    print(f"  threshold at impostor p01   = {imp_thr:.3f}")
    print(f"    -> same-person accepted: {same_at:.2%}")

    if imp_thr > same_thr:
        rec = math.sqrt(same_thr * imp_thr)
        print(f"\n  RECOMMENDED distance threshold = {rec:.3f}")
        print(f"  RECOMMENDED confidence (exp(-d^2/2)) = {math.exp(-0.5 * rec**2):.4f}")
    else:
        rec = (same_thr + imp_thr) / 2
        print(f"\n  WARNING: distributions still overlap.")
        print(f"  fallback distance threshold = {rec:.3f}")

    payload = {
        "n_frames": len(features),
        "enroll_indices": idx.tolist(),
        "observed_std": observed_std.tolist(),
        "variants": results,
        "best": best,
        "same": d_same.tolist(),
        "impostor": d_imp.tolist(),
        "pose": {str(k): [distance_of(f, ref_mean, std, agg, param, clip) for f in v]
                 for k, v in pose_feats.items()},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  raw numbers written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
