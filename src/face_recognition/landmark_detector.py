"""
Landmark Detector Module

Uses MediaPipe Face Mesh to detect 468 facial landmarks.
These landmarks are used to create a geometric face signature for recognition.

Key Landmark Groups:
- 0-467: All 468 face mesh landmarks
- Key indices used for recognition:
  - 1: Nose tip
  - 33, 263: Left/Right eye corners
  - 61, 291: Left/Right mouth corners
  - 234: Right face edge
  - 454: Left face edge
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class LandmarkDetector:
    """
    Detects 468 facial landmarks using MediaPipe Face Mesh.

    MediaPipe Face Mesh provides:
    - 468 3D landmarks
    - Face boundary
    - Iris detection (optional)
    - Works with various head poses
    """

    def __init__(self, static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh.

        Args:
            static_image_mode: If True, processes each image independently (slower)
                            If False, uses previous detection for faster tracking
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: If True, refines landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self._initialize_face_mesh(
            static_image_mode,
            max_num_faces,
            refine_landmarks,
            min_detection_confidence,
            min_tracking_confidence
        )

        # Key landmark indices for visualization and features
        self.key_indices = {
            'nose_tip': 1,
            'left_eye_inner': 133,
            'left_eye_outer': 33,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'mouth_left': 61,
            'mouth_right': 291,
            'left_cheek': 234,
            'right_cheek': 454,
            'forehead': 10,
            'chin': 152,
        }

    def _initialize_face_mesh(self, static_mode, max_faces, refine,
                             det_conf, track_conf):
        """Initialize MediaPipe Face Mesh with parameters."""
        return self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def detect(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """
        Detect facial landmarks in a frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            List of faces, where each face is a list of (x, y, z) landmark coordinates.
            Coordinates are normalized to [0, 1].
            Returns empty list if no face detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Optimization

        # Process frame
        results = self.face_mesh.process(rgb_frame)

        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to list of tuples
                face = []
                for landmark in face_landmarks.landmark:
                    face.append((landmark.x, landmark.y, landmark.z))
                faces.append(face)

        return faces

    def detect_single(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Detect landmarks for single face (first detected face).

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of 468 (x, y, z) tuples, or None if no face detected.
        """
        faces = self.detect(frame)
        return faces[0] if faces else None

    def get_key_landmarks(self, landmarks: List[Tuple[float, float, float]],
                         frame_shape: Tuple[int, int]) -> dict:
        """
        Extract key landmarks in pixel coordinates.

        Args:
            landmarks: List of 468 (x, y, z) normalized landmarks
            frame_shape: (height, width) of the frame

        Returns:
            Dictionary of key landmark names to (x, y) pixel coordinates.
        """
        h, w = frame_shape[:2]
        key_landmarks = {}

        for name, idx in self.key_indices.items():
            x, y, z = landmarks[idx]
            key_landmarks[name] = (int(x * w), int(y * h))

        return key_landmarks

    def draw_landmarks(self, frame: np.ndarray,
                       landmarks: List[Tuple[float, float, float]],
                       draw_all: bool = False) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.

        Args:
            frame: BGR image
            landmarks: List of 468 (x, y, z) normalized landmarks
            draw_all: If True, draw all 468 landmarks (busy). If False, draw key only.

        Returns:
            Frame with landmarks drawn.
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        if draw_all:
            # Draw all 468 landmarks
            for i, (x, y, z) in enumerate(landmarks):
                px, py = int(x * w), int(y * h)
                # Color based on region
                if i < 468:
                    color = (0, 255, 0)  # Green
                cv2.circle(annotated_frame, (px, py), 1, color, -1)
        else:
            # Draw only key landmarks
            key_landmarks = self.get_key_landmarks(landmarks, frame.shape)

            # Draw key points
            for name, (x, y) in key_landmarks.items():
                color = (0, 255, 0) if 'eye' not in name and 'mouth' not in name else (0, 165, 255)
                cv2.circle(annotated_frame, (x, y), 3, color, -1)
                cv2.putText(annotated_frame, name, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated_frame

    def draw_face_mesh(self, frame: np.ndarray,
                       landmarks: List[Tuple[float, float, float]],
                       draw_style: str = 'default') -> np.ndarray:
        """
        Draw full face mesh connections (like MediaPipe default).

        Args:
            frame: BGR image
            landmarks: List of 468 (x, y, z) normalized landmarks
            draw_style: 'default', 'static', or 'connections'

        Returns:
            Frame with face mesh drawn.
        """
        annotated_frame = frame.copy()

        # Create FaceMesh landmarks object for drawing
        if not hasattr(self, '_mp_landmarks'):
            self._mp_landmarks = []

        # Convert to MediaPipe format
        from mediapipe.framework.formats import landmark_pb2
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        for landmark in landmarks:
            landmark_proto = face_landmarks_proto.landmark.add()
            landmark_proto.x = landmark[0]
            landmark_proto.y = landmark[1]
            landmark_proto.z = landmark[2]

        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks_proto,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Draw contours
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks_proto,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        return annotated_frame

    def get_face_box(self, landmarks: List[Tuple[float, float, float]],
                     frame_shape: Tuple[int, int],
                     padding: float = 0.2) -> Tuple[int, int, int, int]:
        """
        Get bounding box of face from landmarks.

        Args:
            landmarks: List of 468 (x, y, z) normalized landmarks
            frame_shape: (height, width) of frame
            padding: Extra padding around face (percentage)

        Returns:
            (x, y, width, height) in pixel coordinates.
        """
        h, w = frame_shape[:2]

        # Get all x, y coordinates
        x_coords = [lm[0] * w for lm in landmarks]
        y_coords = [lm[1] * h for lm in landmarks]

        # Calculate bounding box
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Add padding
        box_w = max_x - min_x
        box_h = max_y - min_y
        pad_x = box_w * padding
        pad_y = box_h * padding

        x = max(0, int(min_x - pad_x))
        y = max(0, int(min_y - pad_y))
        width = min(w - x, int(max_x - min_x + 2 * pad_x))
        height = min(h - y, int(max_y - min_y + 2 * pad_y))

        return (x, y, width, height)

    def close(self):
        """Clean up MediaPipe resources."""
        self.face_mesh.close()


# Test function
if __name__ == "__main__":
    print("LandmarkDetector Module")
    print("=" * 50)
    print("Key landmark indices:")
    detector = LandmarkDetector()
    for name, idx in detector.key_indices.items():
        print(f"  {name}: {idx}")
    print("=" * 50)
    print("Usage:")
    print("  1. Initialize: detector = LandmarkDetector()")
    print("  2. Detect: landmarks = detector.detect_single(frame)")
    print("  3. Draw: annotated = detector.draw_landmarks(frame, landmarks)")
    print("  4. Close: detector.close()")
