"""
Face Recognizer Module

Matches face signatures against enrolled profiles.
Uses multiple distance metrics for robust matching.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .enrollment import EnrollmentManager


@dataclass
class RecognitionResult:
    """Result of a recognition attempt."""
    user_id: str
    name: str
    confidence: float  # 0.0 - 1.0 (1.0 = perfect match)
    distance: float
    is_match: bool

    def __str__(self):
        status = "✓ MATCH" if self.is_match else "✗ NO MATCH"
        return f"{self.name}: {self.confidence:.2%} ({status})"


class FaceRecognizer:
    """
    Face recognition using geometric feature matching.

    Matching Strategy:
    - Uses cosine similarity and Euclidean distance
    - Combines both for robustness
    - Configurable confidence threshold
    """

    # Default threshold for recognition (0.0 - 1.0)
    # Higher = stricter matching
    DEFAULT_THRESHOLD = 0.65

    def __init__(self,
                 enrollment_manager: EnrollmentManager = None,
                 confidence_threshold: float = DEFAULT_THRESHOLD):
        """
        Initialize face recognizer.

        Args:
            enrollment_manager: Manager with enrolled profiles
            confidence_threshold: Minimum confidence for a match
        """
        self.enrollment = enrollment_manager or EnrollmentManager()
        self.threshold = confidence_threshold

    def recognize(self, signature: np.ndarray) -> Optional[RecognitionResult]:
        """
        Recognize a face against enrolled profiles.

        Args:
            signature: Face signature from FeatureExtractor

        Returns:
            RecognitionResult or None if no face detected
        """
        if signature is None or len(signature) == 0:
            return None

        all_profiles = self.enrollment.get_all_signatures()

        if not all_profiles:
            return None

        best_match = None
        best_confidence = 0.0
        best_distance = float('inf')

        for user_id, (name, enrolled_signature) in all_profiles.items():
            confidence, distance = self._calculate_match(signature, enrolled_signature)

            if confidence > best_confidence:
                best_confidence = confidence
                best_distance = distance
                best_match = user_id

        if best_match:
            is_match = best_confidence >= self.threshold
            return RecognitionResult(
                user_id=best_match,
                name=all_profiles[best_match][0],
                confidence=best_confidence,
                distance=best_distance,
                is_match=is_match
            )

        return None

    def recognize_top_n(self, signature: np.ndarray,
                       n: int = 3) -> List[RecognitionResult]:
        """
        Get top N possible matches.

        Args:
            signature: Face signature
            n: Number of top results to return

        Returns:
            List of RecognitionResult sorted by confidence
        """
        if signature is None or len(signature) == 0:
            return []

        all_profiles = self.enrollment.get_all_signatures()

        if not all_profiles:
            return []

        results = []
        for user_id, (name, enrolled_signature) in all_profiles.items():
            confidence, distance = self._calculate_match(signature, enrolled_signature)
            results.append(RecognitionResult(
                user_id=user_id,
                name=name,
                confidence=confidence,
                distance=distance,
                is_match=confidence >= self.threshold
            ))

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:n]

    def _calculate_match(self, sig1: np.ndarray,
                        sig2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate match confidence between two signatures.

        Uses combination of:
        1. Cosine similarity (angle between vectors)
        2. Euclidean distance (magnitude of difference)

        Returns:
            (confidence, distance) tuple
            confidence is 0.0-1.0 where 1.0 is perfect match
        """
        # Ensure numpy arrays
        sig1 = np.array(sig1)
        sig2 = np.array(sig2)

        # 1. Cosine similarity
        dot_product = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)

        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0
        else:
            cosine_sim = dot_product / (norm1 * norm2)

        # Cosine similarity is in [-1, 1], convert to [0, 1]
        cosine_confidence = (cosine_sim + 1) / 2

        # 2. Euclidean distance (normalized)
        distance = np.linalg.norm(sig1 - sig2)

        # Convert distance to confidence (inverse relationship)
        # Using exponential decay: exp(-distance * scale)
        # Tune scale so that distance of 0.5 gives ~50% confidence
        distance_confidence = np.exp(-distance * 2)

        # 3. Combined confidence
        # Weight cosine more as it's more robust for face matching
        combined_confidence = 0.7 * cosine_confidence + 0.3 * distance_confidence

        return combined_confidence, distance

    def set_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.threshold = max(0.0, min(1.0, threshold))

    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.threshold


class StreamingRecognizer:
    """
    Recognizer with temporal smoothing for video streams.
    Reduces jitter by averaging results over multiple frames.
    """

    def __init__(self, recognizer: FaceRecognizer = None,
                 smoothing_frames: int = 5,
                 required_matches: int = 3):
        """
        Initialize streaming recognizer.

        Args:
            recognizer: Base recognizer to use
            smoothing_frames: Number of frames to average
            required_matches: Number of matching frames to confirm identity
        """
        self.recognizer = recognizer or FaceRecognizer()
        self.smoothing_frames = smoothing_frames
        self.required_matches = required_matches

        self.frame_history = []
        self.name_history = []
        self.current_identity = None

    def process_frame(self, signature: np.ndarray) -> Optional[RecognitionResult]:
        """
        Process a frame and return smoothed recognition result.

        Args:
            signature: Face signature from current frame

        Returns:
            RecognitionResult or None
        """
        if signature is None:
            self.frame_history = []
            self.name_history = []
            return None

        # Get result for current frame
        result = self.recognizer.recognize(signature)

        # Update history
        self.frame_history.append(signature)
        if len(self.frame_history) > self.smoothing_frames:
            self.frame_history.pop(0)

        if result:
            self.name_history.append(result.name)
            if len(self.name_history) > self.smoothing_frames:
                self.name_history.pop(0)
        else:
            self.name_history = []

        # If we have enough matches, confirm identity
        if len(self.name_history) >= self.required_matches:
            # Count occurrences of each name
            name_counts = {}
            for name in self.name_history[-self.required_matches:]:
                name_counts[name] = name_counts.get(name, 0) + 1

            # Check if any name meets threshold
            threshold_count = self.required_matches * 0.6  # 60% of required
            confirmed_name = None
            for name, count in name_counts.items():
                if count >= threshold_count:
                    confirmed_name = name
                    break

            if confirmed_name:
                # Get the most confident result for this name
                for r in self.recognizer.recognize_top_n(signature, n=5):
                    if r.name == confirmed_name:
                        self.current_identity = r
                        return r

        self.current_identity = result
        return result

    def get_current_identity(self) -> Optional[RecognitionResult]:
        """Get the most recent confirmed identity."""
        return self.current_identity

    def reset(self):
        """Reset recognition state."""
        self.frame_history = []
        self.name_history = []
        self.current_identity = None


# Test function
if __name__ == "__main__":
    print("FaceRecognizer Module")
    print("=" * 50)

    recognizer = FaceRecognizer()
    print(f"Default threshold: {recognizer.get_threshold()}")

    print("\nMatching Methods:")
    print("  1. Cosine Similarity - angle between vectors")
    print("  2. Euclidean Distance - magnitude of difference")
    print("  3. Combined - weighted average (70% cosine, 30% distance)")

    print("\n" + "=" * 50)
    print("Usage:")
    print("  recognizer = FaceRecognizer()")
    print("  result = recognizer.recognize(signature)")
    print("  if result and result.is_match:")
    print("      print(f'Hello, {result.name}!')")

    # Demo with fake data
    print("\n" + "=" * 50)
    print("Demo (with fake signatures):")

    # Create fake signatures
    sig1 = np.random.rand(32)
    sig2 = sig1 + np.random.rand(32) * 0.1  # Similar
    sig3 = np.random.rand(32)  # Different

    # Test matching
    conf12, dist12 = recognizer._calculate_match(sig1, sig2)
    conf13, dist13 = recognizer._calculate_match(sig1, sig3)

    print(f"  Similar signatures: {conf12:.2%} confidence")
    print(f"  Different signatures: {conf13:.2%} confidence")
