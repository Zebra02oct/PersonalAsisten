"""
Shared Biometric Matching Module

This module is the *single* implementation of the biometric matching metric.
Every recognition path in the project (the production login system, the
``FaceRecognizer`` and the JSON ``EnrollmentManager``) goes through the
functions defined here, so they can never disagree with each other.

Metric
------
The signature is a 30 dimensional vector of dimensionless geometric features.
Comparing two signatures with a plain cosine similarity does not work: the
components have wildly different natural noise levels (a distance ratio
measured from a 100 px face is far noisier than a symmetry ratio), so a
dimension that happens to be noisy dominates the cosine.

Instead every dimension is scaled by its own estimated spread before the
comparison, which turns the comparison into a z-score space:

    z_i = (query_i - reference_i) / std_i

    distance = sqrt( mean( z_i^2 ) )        <- RMS z-score, lower is better
    confidence = exp( -distance^2 / 2 )     <- monotone map to [0, 1]

``std_i`` is estimated from the *enrollment samples* of that person (the
per-dimension spread across the samples that were captured while enrolling),
with a robust floor (see :func:`robust_std`) so that a dimension which
happened to be perfectly stable during enrollment can never blow up a z-score
and cause a false rejection.

A single dimension may not dominate the result: every z is clipped to
``+/- Z_CLIP`` before the RMS is taken.

Storage format (version 2)
--------------------------
    {"features": [<30 floats>], "std": [<30 floats>], "version": 2}

``features`` is the per-dimension mean over the enrollment samples and
``std`` the per-dimension spread.  The legacy format was a bare flat list of
30 min-max normalised floats; it cannot be compared against the new format
and is detected and reported as "needs re-enrollment" (see
:func:`decode_signature`).
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: Version of the signature storage format produced by this module.
SIGNATURE_VERSION = 2

#: Number of components in a face signature.
FEATURE_COUNT = 30

#: Maximum absolute z-score a single dimension may contribute.  Prevents one
#: noisy / mis-measured dimension from dominating the whole distance.
Z_CLIP = 3.0

#: Absolute lower bound for a per-dimension spread (protects against a
#: division by ~0).
ABSOLUTE_STD_FLOOR = 1e-6

#: Robust floor: a dimension whose observed spread is below
#: ``STD_FLOOR_FACTOR * median(observed spread)`` is raised to that value.
#: Rationale: the spread of a single person is estimated from a handful of
#: samples, so a dimension that looked perfectly stable during enrollment may
#: simply have been lucky.  Flooring it keeps such a dimension from
#: dominating the distance (and therefore from causing false rejections).
STD_FLOOR_FACTOR = 0.5

#: Fallback relative spread used when only a *single* enrollment sample is
#: available and no spread can be measured.
SINGLE_SAMPLE_RELATIVE_STD = 0.02

#: Number of decimals kept when serialising the signature.
SIGNATURE_PRECISION = 4

#: Maximum length of the serialised signature.  The database column
#: ``owner_profile.signature`` is ``String(1000)``.
MAX_SIGNATURE_CHARS = 1000

# ---------------------------------------------------------------------------
# THE threshold.  Calibrated from measured same-person / different-person
# distance distributions (see scripts/calibrate_matcher.py).  Lower distance
# = better match, so the same value expressed as a confidence is the minimum
# confidence a match must reach.
#
#   calibrated value: see CALIBRATION below
# ---------------------------------------------------------------------------
MATCH_CONFIDENCE_THRESHOLD = 0.30

#: The same decision boundary expressed in the distance domain.  Derived from
#: MATCH_CONFIDENCE_THRESHOLD via confidence = exp(-d^2/2) -> d = sqrt(-2 ln c).
#: This is *not* an independent constant, it is a helper for reporting.
MATCH_DISTANCE_THRESHOLD = math.sqrt(-2.0 * math.log(MATCH_CONFIDENCE_THRESHOLD))

#: Human readable summary of the calibration, shown by the CLI.
CALIBRATION = (
    "threshold on the RMS z-score distance; see scripts/calibrate_matcher.py "
    "for the measured same-person / different-person distributions"
)


class LegacySignatureError(ValueError):
    """Raised when a legacy (min-max normalised, flat list) signature is used."""


@dataclass
class Signature:
    """A decoded face signature."""

    features: Optional[np.ndarray]
    std: Optional[np.ndarray]
    version: int = SIGNATURE_VERSION
    is_legacy: bool = False
    sample_count: int = 1

    @property
    def usable(self) -> bool:
        """True when the signature can be compared with :func:`match`."""
        return (not self.is_legacy) and self.features is not None and self.std is not None

    def as_payload(self) -> Dict[str, Any]:
        """Return the JSON-serialisable v2 payload."""
        if self.features is None:
            raise ValueError("signature has no features")
        std = self.std
        if std is None:
            std = robust_std(np.abs(self.features) * SINGLE_SAMPLE_RELATIVE_STD)
        return {
            "features": [round(float(v), SIGNATURE_PRECISION) for v in self.features],
            "std": [round(float(v), SIGNATURE_PRECISION) for v in std],
            "version": SIGNATURE_VERSION,
        }


@dataclass
class MatchResult:
    """Result of comparing a query signature with a reference signature."""

    distance: float
    confidence: float
    is_match: bool

    def __str__(self):
        status = "MATCH" if self.is_match else "NO MATCH"
        return f"distance={self.distance:.3f} confidence={self.confidence:.1%} ({status})"


# ---------------------------------------------------------------------------
# Spread estimation
# ---------------------------------------------------------------------------
def robust_std(observed: Sequence[float]) -> np.ndarray:
    """
    Turn the observed per-dimension spread into a safe per-dimension spread.

    ``std_i = max(observed_std_i, STD_FLOOR_FACTOR * median(observed_std))``

    The median is used as the floor because it is not influenced by a single
    pathological dimension (unlike the mean), and the factor 0.5 keeps the
    floor well below the typical spread so that genuinely noisy dimensions
    still dominate the distance.
    """
    observed = np.asarray(observed, dtype=float).ravel()
    observed = np.where(np.isfinite(observed), observed, 0.0)

    positive = observed[observed > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    floor = STD_FLOOR_FACTOR * median

    out = np.maximum(observed, floor)
    return np.maximum(out, ABSOLUTE_STD_FLOOR)


def build_signature(samples: Sequence[Sequence[float]]) -> Signature:
    """
    Build a v2 signature from the enrollment samples of one person.

    ``features`` = per-dimension mean over the samples
    ``std``      = per-dimension spread over the samples (robustly floored)
    """
    arr = np.asarray([np.asarray(s, dtype=float).ravel() for s in samples], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_COUNT:
        raise ValueError(f"each sample must have {FEATURE_COUNT} features, got {arr.shape}")

    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        observed = arr.std(axis=0, ddof=1)
    else:
        # A single sample carries no spread information - fall back to a
        # small relative tolerance so the signature is still usable.
        observed = np.abs(mean) * SINGLE_SAMPLE_RELATIVE_STD

    return Signature(
        features=mean,
        std=robust_std(observed),
        version=SIGNATURE_VERSION,
        is_legacy=False,
        sample_count=int(arr.shape[0]),
    )


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------
def encode_signature(signature: Signature) -> str:
    """Serialise a signature into the compact v2 JSON payload."""
    payload = signature.as_payload()
    text = json.dumps(payload, separators=(",", ":"))
    if len(text) > MAX_SIGNATURE_CHARS:
        raise ValueError(
            f"serialised signature is {len(text)} chars, the column only holds "
            f"{MAX_SIGNATURE_CHARS}")
    return text


def decode_signature(raw: Any) -> Optional[Signature]:
    """
    Decode a stored signature.

    Accepts a JSON string, a decoded list or a decoded dict.  Legacy
    signatures (a bare flat list of 30 floats, i.e. the old min-max
    normalised format) are returned with ``is_legacy=True`` and *no* features
    that could be compared - callers must ask the user to re-enroll instead
    of comparing garbage.
    """
    if raw is None:
        return None

    data = raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            data = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(data, list):
        # Legacy format: flat list of min-max normalised floats.
        try:
            features = np.asarray(data, dtype=float)
        except (TypeError, ValueError):
            return None
        if features.size != FEATURE_COUNT or not np.all(np.isfinite(features)):
            return None
        return Signature(features=features, std=None, version=1, is_legacy=True)

    if isinstance(data, dict):
        features = data.get("features")
        if features is None:
            return None
        try:
            features = np.asarray(features, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
        if features.size != FEATURE_COUNT or not np.all(np.isfinite(features)):
            return None

        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError):
            version = 1

        std = data.get("std")
        if std is None:
            # v2 payload without a spread -> cannot be compared safely.
            return Signature(features=features, std=None, version=version, is_legacy=True)
        try:
            std = np.asarray(std, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
        if std.size != FEATURE_COUNT or not np.all(np.isfinite(std)):
            return Signature(features=features, std=None, version=version, is_legacy=True)

        if version < SIGNATURE_VERSION:
            return Signature(features=features, std=std, version=version, is_legacy=True)

        return Signature(features=features, std=std, version=version, is_legacy=False)

    return None


def signature_payload(samples: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Convenience: build the v2 payload straight from enrollment samples."""
    return build_signature(samples).as_payload()


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------
def z_scores(query: Sequence[float], reference: Signature) -> np.ndarray:
    """Per-dimension z-scores between a query signature and a reference."""
    if reference is None or reference.features is None:
        raise LegacySignatureError("reference signature has no features")
    if reference.is_legacy or reference.std is None:
        raise LegacySignatureError(
            "legacy signature format (flat list of min-max values) cannot be "
            "compared - the owner has to re-enroll")

    q = np.asarray(query, dtype=float).ravel()
    if q.size != FEATURE_COUNT:
        raise ValueError(f"query must have {FEATURE_COUNT} features, got {q.size}")

    std = np.asarray(reference.std, dtype=float).ravel()
    std = np.where(std > 0, std, ABSOLUTE_STD_FLOOR)
    return (q - np.asarray(reference.features, dtype=float).ravel()) / std


def match(query: Sequence[float], reference: Signature) -> MatchResult:
    """
    Compare a query signature with a reference signature.

    Raises:
        LegacySignatureError: if the reference uses the legacy format.
    """
    z = z_scores(query, reference)
    z = np.clip(z, -Z_CLIP, Z_CLIP)
    distance = float(np.sqrt(np.mean(z * z)))
    confidence = float(math.exp(-0.5 * distance * distance))
    return MatchResult(
        distance=distance,
        confidence=confidence,
        is_match=confidence >= MATCH_CONFIDENCE_THRESHOLD,
    )


def match_features(query: Sequence[float],
                   reference_features: Sequence[float],
                   reference_std: Sequence[float]) -> MatchResult:
    """Compare a query with raw reference feature/std arrays."""
    ref = Signature(
        features=np.asarray(reference_features, dtype=float).ravel(),
        std=np.asarray(reference_std, dtype=float).ravel(),
        version=SIGNATURE_VERSION,
        is_legacy=False,
    )
    return match(query, ref)


def is_match_confidence(confidence: float) -> bool:
    """Apply the calibrated threshold to a confidence value."""
    return float(confidence) >= MATCH_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def describe_threshold() -> str:
    """One line description of the calibrated decision boundary."""
    return (f"confidence >= {MATCH_CONFIDENCE_THRESHOLD:.0%} "
            f"(equivalent to RMS z-score distance <= {MATCH_DISTANCE_THRESHOLD:.2f})")


if __name__ == "__main__":
    print("Biometric matcher")
    print("=" * 60)
    print(f"feature count      : {FEATURE_COUNT}")
    print(f"signature version  : {SIGNATURE_VERSION}")
    print(f"z clip             : +/- {Z_CLIP}")
    print(f"std floor factor   : {STD_FLOOR_FACTOR} x median(observed std)")
    print(f"decision boundary  : {describe_threshold()}")
    print(f"calibration        : {CALIBRATION}")

    demo = [float(i % 7) / 7.0 for i in range(FEATURE_COUNT)]
    sig = build_signature([demo])
    text = encode_signature(sig)
    print(f"\nserialised length : {len(text)} chars "
          f"(column limit {MAX_SIGNATURE_CHARS})")
    print(f"payload keys       : {list(sig.as_payload().keys())}")
