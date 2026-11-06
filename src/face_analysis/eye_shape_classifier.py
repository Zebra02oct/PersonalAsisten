"""
Eye Shape Classifier

Classifies eye shape based on facial landmarks.
Eye shapes: Almond, Round, Hooded, Monolid, Upturned, Downturned

Based on geometric analysis of eye landmarks:
- Eye aspect ratio (width/height)
- Corner angles
- Upper eyelid curvature
- Visibility of crease (for hooded detection)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class EyeAnalysis:
    """Complete eye analysis result."""
    shape: str
    confidence: float
    measurements: Dict[str, float]
    landmarks: Dict[str, Tuple[int, int]]

    def __str__(self):
        return f"{self.shape} ({self.confidence:.0%} confidence)"


class EyeShapeClassifier:
    """
    Classifies eye shape from facial landmarks.

    Eye Shapes:
    - Almond: Slightly curved top lid, visible iris
    - Round: Circular, large iris visibility
    - Hooded: Upper lid covers part of eyelid
    - Monolid: Flat lid, no visible crease (common in Asian eyes)
    - Upturned: Corners turn upward at outer edges
    - Downturned: Corners turn downward at outer edges
    """

    # Landmark indices for eyes (MediaPipe Face Mesh)
    LEFT_EYE = {
        'corner_inner': 133,
        'corner_outer': 33,
        'top_center': 159,
        'bottom_center': 145,
        'top_inner': 133,
        'top_outer': 33,
        'bottom_inner': 133,
        'bottom_outer': 33,
        'upper_lid_inner': 160,
        'upper_lid_outer': 144,
        'lower_lid_inner': 158,
        'lower_lid_outer': 153,
        'pupil_center': 468,  # Refined landmark if available
    }

    RIGHT_EYE = {
        'corner_inner': 362,
        'corner_outer': 263,
        'top_center': 386,
        'bottom_center': 374,
        'top_inner': 362,
        'top_outer': 263,
        'bottom_inner': 362,
        'bottom_outer': 263,
        'upper_lid_inner': 384,
        'upper_lid_outer': 373,
        'lower_lid_inner': 380,
        'lower_lid_outer': 382,
    }

    def __init__(self):
        """Initialize classifier."""
        pass

    def classify(self, landmarks: List[Tuple[float, float, float]],
                which_eye: str = 'left') -> EyeAnalysis:
        """
        Classify eye shape.

        Args:
            landmarks: List of 468 face landmarks
            which_eye: 'left' or 'right'

        Returns:
            EyeAnalysis with shape classification and details
        """
        eye_indices = self.LEFT_EYE if which_eye == 'left' else self.RIGHT_EYE

        # Get key points
        inner = self._get_landmark(landmarks, eye_indices['corner_inner'])
        outer = self._get_landmark(landmarks, eye_indices['corner_outer'])
        top = self._get_landmark(landmarks, eye_indices['top_center'])
        bottom = self._get_landmark(landmarks, eye_indices['bottom_center'])

        # Calculate basic measurements
        eye_width = self._distance(inner, outer)
        eye_height = self._distance(top, bottom)
        eye_aspect_ratio = eye_height / eye_width if eye_width > 0 else 0

        # Corner angles
        inner_angle = self._calculate_corner_angle(
            self._get_landmark(landmarks, eye_indices['upper_lid_inner']),
            inner,
            self._get_landmark(landmarks, eye_indices['lower_lid_inner'])
        )
        outer_angle = self._calculate_corner_angle(
            self._get_landmark(landmarks, eye_indices['upper_lid_outer']),
            outer,
            self._get_landmark(landmarks, eye_indices['lower_lid_outer'])
        )

        # Eyelid curvature (upper lid arch)
        upper_lid_inner = self._get_landmark(landmarks, eye_indices['upper_lid_inner'])
        upper_lid_outer = self._get_landmark(landmarks, eye_indices['upper_lid_outer'])

        # Sample points between corners to measure curvature
        mid_top = self._get_landmark(landmarks, eye_indices['top_center'])
        curvature = self._measure_curvature(inner, outer, mid_top)

        # Iris visibility (how much iris is visible)
        iris_visibility = self._calculate_iris_visibility(inner, outer, top, bottom)

        # Crease height (for hooded detection)
        crease_height = self._calculate_crease_height(inner, outer, top, landmarks)

        # Store measurements
        measurements = {
            'eye_width': eye_width,
            'eye_height': eye_height,
            'aspect_ratio': eye_aspect_ratio,
            'inner_angle': inner_angle,
            'outer_angle': outer_angle,
            'curvature': curvature,
            'iris_visibility': iris_visibility,
            'crease_height': crease_height,
        }

        # Classify based on measurements
        shape, confidence = self._classify_shape(
            eye_aspect_ratio, inner_angle, outer_angle,
            curvature, iris_visibility, crease_height
        )

        return EyeAnalysis(
            shape=shape,
            confidence=confidence,
            measurements=measurements,
            landmarks={
                'inner': (int(inner[0]), int(inner[1])),
                'outer': (int(outer[0]), int(outer[1])),
                'top': (int(top[0]), int(top[1])),
                'bottom': (int(bottom[0]), int(bottom[1])),
            }
        )

    def classify_both(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[EyeAnalysis, EyeAnalysis]:
        """Classify both eyes."""
        return (
            self.classify(landmarks, 'left'),
            self.classify(landmarks, 'right')
        )

    def _get_landmark(self, landmarks: List[Tuple[float, float, float]], idx: int) -> np.ndarray:
        """Get landmark as numpy array."""
        if idx >= len(landmarks):
            return np.array([0, 0, 0])
        x, y, z = landmarks[idx]
        return np.array([x, y, z])

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance."""
        return np.linalg.norm(p1 - p2)

    def _calculate_corner_angle(self, upper: np.ndarray, corner: np.ndarray, lower: np.ndarray) -> float:
        """Calculate angle at corner (in degrees)."""
        v1 = upper - corner
        v2 = lower - corner

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return math.degrees(math.acos(cos_angle))

    def _measure_curvature(self, p1: np.ndarray, p2: np.ndarray, mid: np.ndarray) -> float:
        """
        Measure curvature of eye lid.
        Returns ratio of arch height to baseline width.
        """
        baseline = self._distance(p1, p2)
        arch_height = self._distance(mid, (p1 + p2) / 2)
        return arch_height / baseline if baseline > 0 else 0

    def _calculate_iris_visibility(self, inner: np.ndarray, outer: np.ndarray,
                                   top: np.ndarray, bottom: np.ndarray) -> float:
        """
        Calculate how much iris is visible.
        1.0 = fully visible, 0.0 = completely covered.
        """
        eye_height = self._distance(top, bottom)
        iris_diameter = 0.4  # Approximate iris diameter relative to eye height

        # If iris would extend beyond eyelids, it's partially covered
        ideal_iris_height = eye_height * iris_diameter
        visible_ratio = min(1.0, ideal_iris_height / (eye_height * 0.3))

        return visible_ratio

    def _calculate_crease_height(self, inner: np.ndarray, outer: np.ndarray,
                                top: np.ndarray,
                                landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculate eyelid crease height relative to eye.
        Higher value = more hooded appearance.
        """
        # Look at the area between inner and outer corners
        # If there's significant tissue above the eye, crease is high
        eye_mid = (inner + outer) / 2

        # Check forehead area (landmark around brow)
        brow_idx = 65 if inner[0] < 0.5 else 296  # Approximate brow landmark
        if brow_idx < len(landmarks):
            brow = self._get_landmark(landmarks, brow_idx)
            forehead_height = self._distance(eye_mid, brow)
            eye_height = self._distance(top, eye_mid)
            return forehead_height / (forehead_height + eye_height)
        return 0.0

    def _classify_shape(self, aspect_ratio: float, inner_angle: float,
                       outer_angle: float, curvature: float,
                       iris_visibility: float, crease_height: float) -> Tuple[str, float]:
        """
        Classify eye shape based on measurements.

        Returns: (shape_name, confidence)
        """
        scores = {}

        # Almond: Medium aspect ratio, slight curvature, good iris visibility
        scores['Almond'] = (
            0.8 if 0.25 < aspect_ratio < 0.4 else 0.4
        ) * (
            0.9 if 0.1 < curvature < 0.25 else 0.6
        ) * (
            0.9 if iris_visibility > 0.7 else 0.5
        ) * (
            0.8 if crease_height < 0.4 else 0.5
        )

        # Round: Higher aspect ratio, circular shape
        scores['Round'] = (
            0.9 if aspect_ratio > 0.35 else 0.5
        ) * (
            0.8 if curvature > 0.2 else 0.5
        ) * (
            0.9 if iris_visibility > 0.8 else 0.6
        )

        # Hooded: Low iris visibility, high crease
        scores['Hooded'] = (
            0.9 if iris_visibility < 0.7 else 0.4
        ) * (
            0.9 if crease_height > 0.5 else 0.5
        ) * (
            0.7 if curvature > 0.15 else 0.5
        )

        # Monolid: Low curvature, very low crease
        scores['Monolid'] = (
            0.9 if curvature < 0.15 else 0.4
        ) * (
            0.9 if crease_height < 0.3 else 0.5
        ) * (
            0.7 if aspect_ratio < 0.35 else 0.5
        )

        # Upturned: Outer corner higher than inner
        scores['Upturned'] = (
            0.9 if outer_angle > inner_angle else 0.4
        ) * (
            0.8 if curvature > 0.15 else 0.5
        )

        # Downturned: Inner corner higher than outer
        scores['Downturned'] = (
            0.9 if inner_angle > outer_angle else 0.4
        ) * (
            0.8 if curvature > 0.15 else 0.5
        )

        # Find best match
        best_shape = max(scores, key=scores.get)
        confidence = scores[best_shape]

        # Normalize confidence
        confidence = min(1.0, confidence * 1.5)

        return best_shape, confidence

    def get_shape_description(self, shape: str) -> str:
        """Get human-readable description of eye shape."""
        descriptions = {
            'Almond': 'Classic eye shape with slightly pointed corners. '
                     'Upper lid creates a gentle arch, iris partially visible.',
            'Round': 'Large, circular eyes with maximum iris visibility. '
                    'Both lids are curved, creating a wide appearance.',
            'Hooded': 'Extra skin folds over the upper lid, partially '
                     'covering the eyelid. Creates a smoldering look.',
            'Monolid': 'Flat lid without a crease. Common in Asian eye shapes. '
                      'Clean line from brow to lash.',
            'Upturned': 'Outer corners of the eyes lift upward. '
                       'Creates a cat-like, almond shape.',
            'Downturned': 'Outer corners of the eyes tilt downward. '
                         'Soft, wide appearance.',
        }
        return descriptions.get(shape, 'Unknown eye shape')


# Demo/test
if __name__ == "__main__":
    print("EyeShapeClassifier Module")
    print("=" * 50)
    classifier = EyeShapeClassifier()

    print("\nEye Shapes Detected:")
    shapes = ['Almond', 'Round', 'Hooded', 'Monolid', 'Upturned', 'Downturned']
    for shape in shapes:
        print(f"  - {shape}")

    print("\n" + "=" * 50)
    print("Usage:")
    print("  classifier = EyeShapeClassifier()")
    print("  result = classifier.classify(landmarks, 'left')")
    print(f"  print(result.shape)  # e.g., 'Almond'")
    print(f"  print(result.confidence)  # e.g., 0.85")
    print(f"  print(classifier.get_shape_description(result.shape))")
