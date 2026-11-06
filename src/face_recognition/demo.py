"""
Face Recognition Demo - Professional Tech Edition

Shows:
- Face mesh overlay with 468 landmarks
- Highlighted unique facial features
- Clean, professional technology aesthetic
- Feature labels around the face

Based on professional facial recognition system design.
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.face_recognition.landmark_detector import LandmarkDetector
from src.face_recognition.feature_extractor import FeatureExtractor
from src.face_recognition.enrollment import EnrollmentManager
from src.face_recognition.recognizer import FaceRecognizer


class FaceRecognitionDemo:
    """Face recognition with professional tech UI."""

    # Colors
    COLORS = {
        'primary': (255, 180, 50),       # Cyan
        'success': (100, 255, 150),       # Green
        'warning': (80, 80, 255),         # Red/orange
        'eye': (0, 255, 255),             # Yellow
        'nose': (255, 200, 100),          # Orange
        'mouth': (255, 150, 150),          # Pink
        'contour': (150, 200, 255),       # Light blue
        'landmark': (100, 100, 120),      # Gray
        'dark': (15, 15, 25),
        'panel': (25, 25, 40),
        'text': (200, 200, 210),
    }

    # Landmark groups for highlighting
    LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145, 246, 161, 163, 7]
    RIGHT_EYE = [362, 263, 384, 386, 385, 380, 373, 374, 466, 391, 393, 249]
    NOSE = [1, 2, 3, 4, 5, 6, 168, 195, 197]
    MOUTH = [61, 62, 63, 64, 65, 291, 292, 293, 294, 295, 267, 268, 269]
    FACE_CONTOUR = [10, 151, 337, 297, 338, 299, 300, 297, 332, 297, 234, 93, 323, 423]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    def __init__(self):
        # Initialize components
        self.landmark_detector = LandmarkDetector(
            static_image_mode=False,
            refine_landmarks=True
        )
        self.feature_extractor = FeatureExtractor()
        self.enrollment = EnrollmentManager()
        self.recognizer = FaceRecognizer(self.enrollment)

        # Animation
        self.scan_progress = 0.0
        self.glow_intensity = 0.0

        # Camera
        self.cap = None
        self.window_name = "Face Recognition System"

        # Stats
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.recognition_count = 0

    def start(self):
        """Start the demo."""
        print("=" * 60)
        print("Face Recognition System - Tech Edition")
        print("=" * 60)
        print("Controls: e=Enroll, l=List, r=Remove, q=Quit")
        print("-" * 60)

        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1024, 768)

        try:
            self._run_loop()
        finally:
            self._cleanup()

    def _run_loop(self):
        """Main processing loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self._update_fps()
            processed = self._process_frame(frame)
            cv2.imshow(self.window_name, processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('e'):
                self._enroll_face(frame)
            elif key == ord('l'):
                self._list_faces()
            elif key == ord('r'):
                self._remove_face()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame."""
        h, w = frame.shape[:2]

        # Detect landmarks
        landmarks = self.landmark_detector.detect_single(frame)

        if landmarks:
            # Extract features and recognize
            signature = self.feature_extractor.extract(landmarks)
            result = self.recognizer.recognize(signature)

            # Draw face mesh overlay
            annotated = self._draw_face_mesh(frame, landmarks, h, w, result)

            # Draw feature highlights
            annotated = self._draw_feature_highlights(annotated, landmarks, h, w)

        else:
            annotated = frame.copy()

        # Draw HUD
        annotated = self._draw_hud(annotated, h, w)

        return annotated

    def _draw_face_mesh(self, frame: np.ndarray,
                       landmarks: List,
                       h: int, w: int,
                       result) -> np.ndarray:
        """Draw detailed face mesh overlay."""

        # Get face box
        box = self.landmark_detector.get_face_box(landmarks, (h, w), padding=0.15)
        bx, by, bw, bh = box

        # Animated glow behind face
        glow_alpha = (np.sin(self.glow_intensity) + 1) / 2 * 0.15 + 0.05
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 5, by - 5),
                     (bx + bw + 5, by + bh + 5),
                     self.COLORS['primary'], -1)
        frame[:] = cv2.addWeighted(frame, 1 - glow_alpha,
                                  overlay, glow_alpha, 0)

        # Draw face mesh - key points only (not all 468 to keep it clean)
        mesh_groups = [
            (self.FACE_CONTOUR, self.COLORS['contour'], 2),
            (self.LEFT_EYE + self.RIGHT_EYE, self.COLORS['eye'], 2),
            (self.NOSE, self.COLORS['nose'], 2),
            (self.MOUTH, self.COLORS['mouth'], 2),
        ]

        for indices, color, size in mesh_groups:
            for idx in indices:
                if idx < len(landmarks):
                    x, y, z = landmarks[idx]
                    px, py = int(x * w), int(y * h)
                    cv2.circle(frame, (px, py), size, color, -1, cv2.LINE_AA)

        # Draw iris specifically
        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
            if idx < len(landmarks):
                x, y, z = landmarks[idx]
                px, py = int(x * w), int(y * h)
                cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Draw connections for eyes
        eye_connections = [
            (self.LEFT_EYE, self.COLORS['eye']),
            (self.RIGHT_EYE, self.COLORS['eye']),
        ]

        for eye_indices, color in eye_connections:
            points = []
            for idx in eye_indices:
                if idx < len(landmarks):
                    x, y, z = landmarks[idx]
                    points.append((int(x * w), int(y * h)))

            if len(points) > 2:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 1, cv2.LINE_AA)

        # Draw nose bridge
        nose_points = [(int(landmarks[i][0] * w), int(landmarks[i][1] * h))
                      for i in [168, 195, 197] if i < len(landmarks)]
        if len(nose_points) > 1:
            for i in range(len(nose_points) - 1):
                cv2.line(frame, nose_points[i], nose_points[i + 1],
                        self.COLORS['nose'], 2, cv2.LINE_AA)

        # Draw mouth outline
        mouth_points = [(int(landmarks[i][0] * w), int(landmarks[i][1] * h))
                        for i in [61, 291, 0, 13, 14, 17] if i < len(landmarks)]
        if len(mouth_points) > 2:
            pts = np.array(mouth_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, self.COLORS['mouth'], 2, cv2.LINE_AA)

        # Draw face contour
        contour_indices = [10, 151, 337, 297, 338, 299, 300, 297, 332, 234, 93, 323, 423, 352, 123, 50]
        contour_points = [(int(landmarks[i][0] * w), int(landmarks[i][1] * h))
                         for i in contour_indices if i < len(landmarks)]
        if len(contour_points) > 2:
            pts = np.array(contour_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, self.COLORS['contour'], 2, cv2.LINE_AA)

        return frame

    def _draw_feature_highlights(self, frame: np.ndarray,
                                landmarks: List,
                                h: int, w: int) -> np.ndarray:
        """Draw feature labels and highlights."""

        # Calculate face center
        box = self.landmark_detector.get_face_box(landmarks, (h, w), padding=0.15)
        bx, by, bw, bh = box
        face_center_x = bx + bw // 2
        face_center_y = by + bh // 2

        # Feature positions (relative to face box)
        feature_labels = []

        # Left eye
        if len(landmarks) > 159:
            lx, ly = int(landmarks[159][0] * w), int(landmarks[159][1] * h)
            feature_labels.append(('LEFT EYE', (lx, ly - 40), self.COLORS['eye']))

        # Right eye
        if len(landmarks) > 386:
            rx, ry = int(landmarks[386][0] * w), int(landmarks[386][1] * h)
            feature_labels.append(('RIGHT EYE', (rx, ry - 40), self.COLORS['eye']))

        # Nose
        if len(landmarks) > 1:
            nx, ny = int(landmarks[1][0] * w), int(landmarks[1][1] * h)
            feature_labels.append(('NOSE', (nx + 50, ny), self.COLORS['nose']))

        # Mouth
        if len(landmarks) > 13:
            mx, my = int(landmarks[13][0] * w), int(landmarks[13][1] * h)
            feature_labels.append(('MOUTH', (mx, my + 50), self.COLORS['mouth']))

        # Draw labels
        for text, pos, color in feature_labels:
            x, y = pos

            # Draw connecting line
            line_end_x = x + 30 if 'EYE' in text else x - 30
            cv2.line(frame, (line_end_x, y), pos, color, 1, cv2.LINE_AA)

            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.35, 1)

            bg_x1 = pos[0] - 5
            bg_y1 = pos[1] - th - 3
            bg_x2 = pos[0] + tw + 5
            bg_y2 = pos[1] + 3

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                        self.COLORS['panel'], -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                        color, 1, cv2.LINE_AA)

            cv2.putText(frame, text, (pos[0], pos[1] - 2),
                      font, 0.35, color, 1, cv2.LINE_AA)

        return frame

    def _draw_hud(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """Draw top HUD bar."""

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 45), self.COLORS['dark'], -1, cv2.LINE_AA)

        # Title
        title = "FACIAL RECOGNITION SYSTEM"
        cv2.putText(frame, title, (w // 2 - 140, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['primary'], 1, cv2.LINE_AA)

        # Status dot
        dot_x = w - 80
        dot_y = 22
        dot_radius = 6

        # Pulsing effect
        pulse = (np.sin(time.time() * 3) + 1) / 2
        dot_color = (
            int(100 * pulse + 100),
            int(255 * pulse + 50),
            int(150 * pulse + 100)
        )
        cv2.circle(frame, (dot_x, dot_y), dot_radius, dot_color, -1, cv2.LINE_AA)

        cv2.putText(frame, "LIVE", (dot_x + 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['success'], 1, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1, cv2.LINE_AA)

        # Bottom feature legend
        legend_y = h - 30
        features = [
            ('EYE', self.COLORS['eye']),
            ('NOSE', self.COLORS['nose']),
            ('MOUTH', self.COLORS['mouth']),
            ('CONTOUR', self.COLORS['contour']),
        ]

        start_x = w // 2 - 180
        for i, (name, color) in enumerate(features):
            x = start_x + i * 90

            # Color dot
            cv2.circle(frame, (x, legend_y), 5, color, -1, cv2.LINE_AA)

            # Label
            cv2.putText(frame, name, (x + 12, legend_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1, cv2.LINE_AA)

        # Bottom info
        info_text = "Unique Facial Features Analysis"
        cv2.putText(frame, info_text, (w // 2 - 120, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 120), 1, cv2.LINE_AA)

        return frame

    def _update_fps(self):
        """Update FPS and animations."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time

        if elapsed >= 1.0:
            self.fps = int(self.frame_count / elapsed)
            self.frame_count = 0
            self.fps_start_time = time.time()

        # Update glow animation
        self.glow_intensity += 0.05

    def _enroll_face(self, frame: np.ndarray):
        """Enroll face."""
        landmarks = self.landmark_detector.detect_single(frame)
        if landmarks is None:
            print("[WARNING] No face detected")
            return

        signature = self.feature_extractor.extract(landmarks)

        print("\n" + "=" * 40)
        print("ENROLLMENT")
        print("=" * 40)
        name = input("Name: ").strip()
        if name:
            user_id, is_new = self.enrollment.enroll(name, signature)
            print(f"[OK] Enrolled as '{name}'")
        print("=" * 40)

    def _list_faces(self):
        """List enrolled faces."""
        profiles = self.enrollment.list_profiles()
        print("\n" + "=" * 40)
        print(f"Enrolled: {len(profiles)}")
        for p in profiles:
            print(f"  - {p['name']}")
        print("=" * 40)

    def _remove_face(self):
        """Remove face."""
        profiles = self.enrollment.list_profiles()
        if not profiles:
            return

        print("\n" + "=" * 40)
        for i, p in enumerate(profiles):
            print(f"  {i + 1}. {p['name']}")
        print("  0. Cancel")
        print("=" * 40)

        try:
            choice = int(input("Choice: "))
            if 1 <= choice <= len(profiles):
                self.enrollment.remove(profiles[choice - 1]['user_id'])
                print("[OK] Removed")
        except ValueError:
            pass

    def _cleanup(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.close()
        print("[OK] Closed")


def main():
    demo = FaceRecognitionDemo()
    demo.start()


if __name__ == "__main__":
    main()
