"""
Feature Extractor Module

Extracts geometric features from facial landmarks to create a face signature.
This signature is used for face recognition without storing actual face images.

Feature Types (30 components, in this exact order):
    [ 0..11] distance    - raw landmark lengths, divided by the per-frame face size
    [12..15] ratio       - dimensionless ratios
    [16..19] angle       - 3 dimensionless angles + 1 nose-depth difference (length)
    [20..27] symmetry    - 4 pairs x (x-symmetry ratio, |dy| length)
    [28..29] proportion  - dimensionless proportions

Scale invariance
-----------------
Every component that carries a *length* is divided by a per-frame face-size
reference derived from the landmarks themselves, so the whole vector is
dimensionless and two samples of the same person can be compared even when
they were taken at different distances from the camera.

The previous implementation applied a per-sample min-max normalisation to the
whole vector.  That forced min=0 / max=1 with all components non-negative on
every single sample, which collapsed every signature into a near-parallel cone
(cosine similarity 0.88 between a real signature and random geometry, versus
0.997 between a real signature and a *completely different* face).  The min-max
step has been removed entirely - it is what made the login system accept
essentially any face.
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Number of components in a face signature.
FEATURE_COUNT = 30

#: Which components are raw lengths and therefore have to be divided by the
#: per-frame face-size reference to become dimensionless.
#:   0..11  distance group                     -> all raw lengths
#:   12..15 ratio group                       -> already dimensionless
#:   16..18 angles (radians / atan2)          -> already dimensionless
#:   19     nose tip z - nose bottom z        -> raw length (MediaPipe z scales
#:                                                  with the face size, verified
#:                                                  empirically)
#:   20..27 symmetry: ratio, |dy| per pair     -> ratio dimensionless, |dy| raw
#:   28..29 proportion group                  -> already dimensionless
SCALE_DEPENDENT_MASK: Tuple[bool, ...] = (
    (True,) * 12
    + (False,) * 4
    + (False, False, False, True)
    + (False, True) * 4
    + (False,) * 2
)
assert len(SCALE_DEPENDENT_MASK) == FEATURE_COUNT


class FeatureExtractor:
    """
    Extracts geometric features from 468 facial landmarks.

    The extracted features form a "face signature" that:
    - Is scale-invariant (raw lengths are divided by the face size)
    - Is dimensionless (every component is a ratio or an angle)
    - Cannot reconstruct the original face
    - Is privacy-friendly
    """

    #: Kept as a class attribute for convenience / backwards compatibility.
    FEATURE_COUNT = FEATURE_COUNT

    # Landmark indices for feature extraction.
    #
    # NOTE - the eye indices below were verified empirically on this machine
    # (MediaPipe 0.10.9, refine_landmarks=True) by projecting the candidate
    # landmarks onto the perpendicular of the eye axis:
    #   eye 33/133 : 159 is the UPPER lid (offset -0.0113), 145 the LOWER lid
    #                (offset +0.0056), 158/153 the inner pair, 160/144 the
    #                outer pair.
    #   eye 263/362: 386 is the UPPER lid (offset -0.0128), 374 the LOWER lid.
    #   iris 468 belongs to the 33/133 eye, iris 473 to the 263/362 eye.
    # The old mapping listed 159 for BOTH 'left_eye_top' and
    # 'left_eye_center', i.e. the "eye centre" was really the upper eyelid.
    # The eye centre is now derived as the midpoint of the two lids.
    KEY_LANDMARKS = {
        # Eyes
        'left_eye_inner': 133,
        'left_eye_outer': 33,
        'left_eye_top': 159,
        'left_eye_bottom': 145,
        'right_eye_inner': 362,
        'right_eye_outer': 263,
        'right_eye_top': 386,
        'right_eye_bottom': 374,

        # Eyebrows
        'left_eyebrow_inner': 70,
        'left_eyebrow_outer': 107,
        'right_eyebrow_inner': 300,
        'right_eyebrow_outer': 336,

        # Nose
        'nose_tip': 1,
        'nose_bottom': 2,
        'nose_left': 49,
        'nose_right': 275,

        # Mouth
        'mouth_left': 61,
        'mouth_right': 291,
        'mouth_top': 13,
        'mouth_bottom': 14,
        'mouth_center': 0,

        # Face contour
        'chin': 152,
        'left_cheek': 234,
        'right_cheek': 454,
        'left_temple': 123,
        'right_temple': 352,
        'forehead': 10,
    }

    def __init__(self, normalize: bool = True):
        """
        Initialize feature extractor.

        Args:
            normalize: If True (default) every length-like component is divided
                by the per-frame face-size reference, which makes the signature
                dimensionless.  If False the raw lengths are returned (they are
                then only comparable between samples taken at the same
                distance).  NOTE: this flag used to control a min-max
                normalisation, which has been removed - it destroyed all
                discriminative information.
        """
        self.normalize = normalize
        self.feature_stats = None  # reserved for future per-dimension stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Extract geometric features from landmarks.

        Args:
            landmarks: List of 468 (x, y, z) normalized landmarks

        Returns:
            numpy array of 30 extracted features (dimensionless)
        """
        features = []

        # 1. Distance features (12) - raw lengths, normalised further below
        features.extend(self._extract_distance_features(landmarks))

        # 2. Ratio features (4) - already dimensionless
        features.extend(self._extract_ratio_features(landmarks))

        # 3. Angle features (4) - 3 dimensionless + 1 raw length
        features.extend(self._extract_angle_features(landmarks))

        # 4. Symmetry features (8) - 4 ratios + 4 raw |dy|
        features.extend(self._extract_symmetry_features(landmarks))

        # 5. Proportion features (2) - already dimensionless
        features.extend(self._extract_proportion_features(landmarks))

        features = np.asarray(features, dtype=float)

        if features.shape != (FEATURE_COUNT,):
            raise ValueError(
                f"Expected {FEATURE_COUNT} features, produced {features.shape[0]}")

        # Normalize the length-like components by the face size.  This is what
        # makes the signature comparable between two captures of the same
        # person taken at different distances from the camera.
        if self.normalize:
            features = self._make_scale_invariant(features, self._face_scale(landmarks))

        return features

    def get_feature_count(self) -> int:
        """Return number of features extracted (30)."""
        return FEATURE_COUNT

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_landmark(self, landmarks: List[Tuple[float, float, float]],
                      name: str) -> np.ndarray:
        """Get a landmark as numpy array."""
        idx = self.KEY_LANDMARKS[name]
        x, y, z = landmarks[idx]
        return np.array([x, y, z])

    def _eye_center(self, landmarks: List[Tuple[float, float, float]],
                    side: str) -> np.ndarray:
        """
        Geometric centre of an eye = midpoint of the upper and lower eyelid.

        The old code used the *upper lid* index (159 / 386) as the eye centre,
        which biased every eye-relative distance downwards.
        """
        top = self._get_landmark(landmarks, f'{side}_eye_top')
        bottom = self._get_landmark(landmarks, f'{side}_eye_bottom')
        return (top + bottom) / 2.0

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return float(np.linalg.norm(p1 - p2))

    def _angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        Returns angle in radians.
        """
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return math.acos(cos_angle)

    def _face_scale(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Per-frame reference length used to turn raw lengths into ratios.

        It is the geometric mean of the face height (forehead -> chin) and the
        face width (cheek -> cheek).  Both are plain Euclidean distances
        between landmarks, therefore they are invariant to in-plane rotation
        and they grow linearly with the apparent size of the face, which makes
        the reference itself scale-invariant.

        Using a *geometric mean* of height and width (rather than the
        inter-ocular distance alone) has a useful side effect: when the head
        is yawed, the apparent width shrinks while the height does not, so
        the reference shrinks a little too and partially compensates the
        foreshortening of the horizontal distances.
        """
        height = self._distance(self._get_landmark(landmarks, 'forehead'),
                                 self._get_landmark(landmarks, 'chin'))
        width = self._distance(self._get_landmark(landmarks, 'left_cheek'),
                               self._get_landmark(landmarks, 'right_cheek'))

        scale = math.sqrt(max(height, 1e-9) * max(width, 1e-9))
        if scale <= 1e-9:
            # Degenerate geometry - fall back to the inter-ocular distance.
            scale = self._distance(self._eye_center(landmarks, 'left'),
                                    self._eye_center(landmarks, 'right'))
        return scale if scale > 1e-9 else 1.0

    def _make_scale_invariant(self, features: np.ndarray,
                              scale: float) -> np.ndarray:
        """Divide every length-like component by the face-size reference."""
        out = features.copy()
        mask = np.array(SCALE_DEPENDENT_MASK, dtype=bool)
        out[mask] = out[mask] / scale
        return out

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------
    def _extract_distance_features(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Extract key distances between facial features (12 raw lengths)."""
        distances = []

        # Eye distances
        eye_left = self._eye_center(landmarks, 'left')
        eye_right = self._eye_center(landmarks, 'right')
        eye_dist = self._distance(eye_left, eye_right)
        distances.append(eye_dist)

        # Eye to nose distances
        nose_tip = self._get_landmark(landmarks, 'nose_tip')
        distances.append(self._distance(eye_left, nose_tip))
        distances.append(self._distance(eye_right, nose_tip))

        # Nose to mouth distances
        mouth_center = self._get_landmark(landmarks, 'mouth_center')
        distances.append(self._distance(nose_tip, mouth_center))

        # Eye to mouth distances
        distances.append(self._distance(eye_left, mouth_center))
        distances.append(self._distance(eye_right, mouth_center))

        # Face height (forehead to chin)
        forehead = self._get_landmark(landmarks, 'forehead')
        chin = self._get_landmark(landmarks, 'chin')
        face_height = self._distance(forehead, chin)
        distances.append(face_height)

        # Eye to chin
        distances.append(self._distance(eye_left, chin))
        distances.append(self._distance(eye_right, chin))

        # Cheek to cheek
        left_cheek = self._get_landmark(landmarks, 'left_cheek')
        right_cheek = self._get_landmark(landmarks, 'right_cheek')
        cheek_dist = self._distance(left_cheek, right_cheek)
        distances.append(cheek_dist)

        # Nose width
        nose_left = self._get_landmark(landmarks, 'nose_left')
        nose_right = self._get_landmark(landmarks, 'nose_right')
        distances.append(self._distance(nose_left, nose_right))

        # Mouth width
        mouth_left = self._get_landmark(landmarks, 'mouth_left')
        mouth_right = self._get_landmark(landmarks, 'mouth_right')
        distances.append(self._distance(mouth_left, mouth_right))

        return distances

    def _extract_ratio_features(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Extract scale-invariant distance ratios (4)."""
        ratios = []

        # Eye spacing relative to face width
        eye_left = self._eye_center(landmarks, 'left')
        eye_right = self._eye_center(landmarks, 'right')
        eye_dist = self._distance(eye_left, eye_right)

        left_cheek = self._get_landmark(landmarks, 'left_cheek')
        right_cheek = self._get_landmark(landmarks, 'right_cheek')
        face_width = self._distance(left_cheek, right_cheek)

        if face_width > 0:
            ratios.append(eye_dist / face_width)

        # Nose to chin relative to face height
        nose_tip = self._get_landmark(landmarks, 'nose_tip')
        chin = self._get_landmark(landmarks, 'chin')
        forehead = self._get_landmark(landmarks, 'forehead')

        nose_to_chin = self._distance(nose_tip, chin)
        face_height = self._distance(forehead, chin)

        if face_height > 0:
            ratios.append(nose_to_chin / face_height)

        # Eye height relative to width
        eye_top = self._get_landmark(landmarks, 'left_eye_top')
        eye_bottom = self._get_landmark(landmarks, 'left_eye_bottom')
        eye_height = self._distance(eye_top, eye_bottom)

        eye_outer = self._get_landmark(landmarks, 'left_eye_outer')
        eye_inner = self._get_landmark(landmarks, 'left_eye_inner')
        eye_width = self._distance(eye_outer, eye_inner)

        if eye_width > 0:
            ratios.append(eye_height / eye_width)

        # Mouth width relative to face width
        mouth_left = self._get_landmark(landmarks, 'mouth_left')
        mouth_right = self._get_landmark(landmarks, 'mouth_right')
        mouth_width = self._distance(mouth_left, mouth_right)

        if face_width > 0:
            ratios.append(mouth_width / face_width)

        return ratios

    def _extract_angle_features(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Extract angular features of the face (4)."""
        angles = []

        # Nose angle (tip relative to eyes)
        nose_tip = self._get_landmark(landmarks, 'nose_tip')
        eye_left = self._eye_center(landmarks, 'left')
        eye_right = self._eye_center(landmarks, 'right')

        angle = self._angle(eye_left, nose_tip, eye_right)
        angles.append(angle)

        # Eye line angle relative to horizontal
        dx = eye_right[0] - eye_left[0]
        dy = eye_right[1] - eye_left[1]
        eye_angle = math.atan2(dy, dx)
        angles.append(eye_angle)

        # Mouth angle
        mouth_left = self._get_landmark(landmarks, 'mouth_left')
        mouth_right = self._get_landmark(landmarks, 'mouth_right')

        dx = mouth_right[0] - mouth_left[0]
        dy = mouth_right[1] - mouth_left[1]
        mouth_angle = math.atan2(dy, dx)
        angles.append(mouth_angle)

        # Nose depth difference (Z).  MediaPipe's z scales with the apparent
        # face size (verified empirically), so this is a length and is
        # divided by the face-size reference like the other lengths.
        nose_bottom = self._get_landmark(landmarks, 'nose_bottom')
        nose_depth_ratio = nose_tip[2] - nose_bottom[2]
        angles.append(nose_depth_ratio)

        return angles

    def _extract_symmetry_features(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Extract facial symmetry measures (8 = 4 pairs x 2)."""
        symmetries = []

        # Face center (approximate)
        left_cheek = self._get_landmark(landmarks, 'left_cheek')
        right_cheek = self._get_landmark(landmarks, 'right_cheek')
        face_center_x = (left_cheek[0] + right_cheek[0]) / 2

        # Left-right distance symmetry for key points
        key_pairs = [
            ('left_eye_inner', 'right_eye_inner'),
            ('left_eyebrow_inner', 'right_eyebrow_inner'),
            ('nose_left', 'nose_right'),
            ('mouth_left', 'mouth_right'),
        ]

        for left_name, right_name in key_pairs:
            left_pt = self._get_landmark(landmarks, left_name)
            right_pt = self._get_landmark(landmarks, right_name)

            # Distance from center for each point
            left_dist = abs(left_pt[0] - face_center_x)
            right_dist = abs(right_pt[0] - face_center_x)

            # Symmetry ratio (1.0 = perfect symmetry)
            total_dist = left_dist + right_dist
            if total_dist > 0:
                symmetry = 1 - abs(left_dist - right_dist) / total_dist
            else:
                symmetry = 1.0

            symmetries.append(symmetry)

            # Y-coordinate symmetry (should be at same height).  This is a raw
            # length and is divided by the face-size reference later on.
            y_diff = abs(left_pt[1] - right_pt[1])
            symmetries.append(y_diff)

        return symmetries

    def _extract_proportion_features(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Extract facial proportion features (2)."""
        proportions = []

        # Golden ratio approximations (face has golden ratio proportions)
        forehead = self._get_landmark(landmarks, 'forehead')
        nose_tip = self._get_landmark(landmarks, 'nose_tip')
        chin = self._get_landmark(landmarks, 'chin')

        upper_height = self._distance(forehead, nose_tip)
        lower_height = self._distance(nose_tip, chin)

        if lower_height > 0:
            proportions.append(upper_height / lower_height)

        # Eye region proportion
        eye_left = self._eye_center(landmarks, 'left')
        nose_bridge = self._get_landmark(landmarks, 'nose_tip')
        eye_to_nose = self._distance(eye_left, nose_bridge)

        eye_right = self._eye_center(landmarks, 'right')
        forehead_y = forehead[1]
        eye_avg_y = (eye_left[1] + eye_right[1]) / 2

        upper_face = abs(forehead_y - eye_avg_y)
        if upper_face > 0:
            proportions.append(eye_to_nose / upper_face)

        return proportions


# Test function
if __name__ == "__main__":
    print("FeatureExtractor Module")
    print("=" * 50)
    extractor = FeatureExtractor()

    print(f"Feature count: {extractor.get_feature_count()}")
    print("\nKey landmarks used:")
    for name, idx in extractor.KEY_LANDMARKS.items():
        print(f"  {name}: {idx}")

    print("\n" + "=" * 50)
    print("Usage:")
    print("  1. Create extractor: extractor = FeatureExtractor()")
    print("  2. Extract features: features = extractor.extract(landmarks)")
    print("  3. features is a numpy array of shape (30,)")
