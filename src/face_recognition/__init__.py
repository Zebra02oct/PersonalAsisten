# Face Recognition Module - MediaPipe based
# Phase 1.5: Educational face recognition with landmark-based approach

from .landmark_detector import LandmarkDetector
from .feature_extractor import FeatureExtractor
from .enrollment import EnrollmentManager
from .recognizer import FaceRecognizer

__all__ = [
    'LandmarkDetector',
    'FeatureExtractor',
    'EnrollmentManager',
    'FaceRecognizer'
]
