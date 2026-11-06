"""
Guided Face Enrollment Module

Handles the guided enrollment process with pose instructions.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

from src.face_recognition.landmark_detector import LandmarkDetector
from src.face_recognition.feature_extractor import FeatureExtractor
from src.face_analysis.eye_shape_classifier import EyeShapeClassifier


class GuidedEnrollment:
    """Guided face enrollment with pose instructions."""

    COLORS = {
        'primary': (255, 180, 50),
        'success': (100, 255, 150),
        'warning': (80, 80, 255),
        'eye': (0, 255, 255),
        'dark': (15, 15, 25),
        'text': (200, 200, 210),
    }

    def __init__(self, landmark_detector: LandmarkDetector,
                 feature_extractor: FeatureExtractor,
                 eye_classifier: EyeShapeClassifier,
                 profile_manager):
        self.landmark_detector = landmark_detector
        self.feature_extractor = feature_extractor
        self.eye_classifier = eye_classifier
        self.profile_manager = profile_manager

        # Enrollment settings
        self.enrollment_samples: List[np.ndarray] = []
        self.enrollment_count = 0
        self.enrollment_target = 5

        # Camera
        self.cap = None
        self.window_name = "Face Enrollment"

        # Pose tracking
        self.current_pose_index = 0
        self.pose_start_time = 0
        self.pose_valid_duration = 0
        self.pose_required_duration = 1.5

    def run(self) -> bool:
        """Run the guided enrollment process."""
        self._print_header()

        name, dob = self._get_user_info()
        if not name or not dob:
            return False

        self._print_poses()
        self._run_capture_loop(name, dob)

        return self.enrollment_count >= self.enrollment_target

    def _print_header(self):
        print("\n" + "=" * 60)
        print("  FACE ENROLLMENT SYSTEM - GUIDED")
        print("  This will guide you through 5 samples with different poses.")
        print("=" * 60)

    def _get_user_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get user information."""
        print("\n  Enter your information:")

        name = input("  Name: ").strip()
        if not name:
            print("  ✗ Name is required!")
            return None, None

        dob = input("  Date of Birth (YYYY-MM-DD): ").strip()
        if not dob:
            print("  ✗ Date of birth is required!")
            return None, None

        try:
            datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            print("  ✗ Invalid date format! Use YYYY-MM-DD")
            return None, None

        return name, dob

    def _print_poses(self):
        """Print pose instructions."""
        poses = [
            "Look straight at the camera",
            "Turn head slightly to the LEFT",
            "Turn head slightly to the RIGHT",
            "Slight smile (optional)",
            "Look straight again",
        ]
        print("\n  POSE INSTRUCTIONS:")
        for i, pose in enumerate(poses, 1):
            print(f"    {i}. {pose}")
        print("\n  System will auto-capture when pose is correct.\n")

    def _run_capture_loop(self, name: str, dob: str):
        """Run the main capture loop."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("  ✗ Cannot open camera!")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        self.enrollment_samples = []
        self.enrollment_count = 0
        self.current_pose_index = 0
        self.pose_start_time = 0
        self.pose_valid_duration = 0

        poses = [
            ("Look straight at the camera", "straight"),
            ("Turn head LEFT", "left"),
            ("Turn head RIGHT", "right"),
            ("Slight smile", "smile"),
            ("Look straight", "straight2"),
        ]

        try:
            while self.enrollment_count < self.enrollment_target:
                ret, frame = self.cap.read()
                if not ret:
                    break

                annotated = self._process_frame(frame, poses)
                cv2.imshow(self.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n  Enrollment cancelled.")
                    break

            if self.enrollment_count >= self.enrollment_target:
                self._save_enrollment(name, dob)

        finally:
            self._cleanup()

    def _process_frame(self, frame: np.ndarray, poses: List) -> np.ndarray:
        """Process enrollment frame."""
        h, w = frame.shape[:2]
        annotated = frame.copy()

        landmarks = self.landmark_detector.detect_single(frame)

        current_pose, pose_type = poses[self.current_pose_index]

        if landmarks:
            # Draw face mesh
            annotated = self._draw_face_mesh(annotated, landmarks, h, w)

            # Check face validity
            face_box = self.landmark_detector.get_face_box(landmarks, (h, w))
            face_valid = face_box[2] > 80 and face_box[3] > 80

            if face_valid:
                # Check if face is centered
                face_center_x = (face_box[0] + face_box[2] // 2) / w
                pose_correct = 0.4 < face_center_x < 0.6

                if pose_correct:
                    if self.pose_start_time == 0:
                        self.pose_start_time = time.time()

                    self.pose_valid_duration = time.time() - self.pose_start_time

                    # Auto-capture when pose held long enough
                    if self.pose_valid_duration >= self.pose_required_duration:
                        signature = self.feature_extractor.extract(landmarks)
                        self.enrollment_samples.append(signature)
                        self.enrollment_count += 1
                        print(f"  ✓ Captured sample {self.enrollment_count}/{self.enrollment_target}")

                        # Flash effect
                        annotated = self._draw_capture_flash(annotated, h, w)

                        self.pose_start_time = 0
                        self.pose_valid_duration = 0

                        if self.enrollment_count < self.enrollment_target:
                            self.current_pose_index = (self.current_pose_index + 1) % len(poses)
                else:
                    self.pose_start_time = 0
                    pose_correct = False
            else:
                self.pose_start_time = 0
                pose_correct = False

            # Draw UI overlay
            annotated = self._draw_overlay(annotated, h, w, poses, pose_correct)
        else:
            cv2.putText(annotated, "Position your face in the frame...",
                       (w // 2 - 180, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        return annotated

    def _draw_face_mesh(self, frame: np.ndarray, landmarks: List, h: int, w: int) -> np.ndarray:
        """Draw face mesh."""
        # Simple face contour
        contour_indices = [10, 151, 337, 297, 338, 299, 300, 332, 234, 93, 323, 423, 352]
        for idx in contour_indices:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 2, self.COLORS['primary'], -1)

        # Eyes
        LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145]
        RIGHT_EYE = [362, 263, 384, 386, 380, 382, 373, 374]
        for idx in LEFT_EYE + RIGHT_EYE:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 2, self.COLORS['eye'], -1)

        return frame

    def _draw_overlay(self, frame: np.ndarray, h: int, w: int, poses: List, pose_correct: bool) -> np.ndarray:
        """Draw enrollment overlay."""
        current_pose, _ = poses[self.current_pose_index]

        # Pose instruction box
        cv2.rectangle(frame, (w // 2 - 200, 60),
                     (w // 2 + 200, 140),
                     (20, 20, 35), -1)
        cv2.rectangle(frame, (w // 2 - 200, 60),
                     (w // 2 + 200, 140),
                     self.COLORS['success'] if pose_correct else self.COLORS['warning'],
                     2)

        cv2.putText(frame, f"Sample {self.enrollment_count + 1}/{self.enrollment_target}",
                   (w // 2 - 100, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['primary'], 1)

        cv2.putText(frame, current_pose,
                   (w // 2 - 100, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.COLORS['success'] if pose_correct else self.COLORS['warning'], 1)

        # Progress bar
        bar_y = h - 100
        bar_x = w // 2 - 150
        bar_w = 300
        bar_h = 20
        progress = self.enrollment_count / self.enrollment_target

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (40, 40, 50), -1)

        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                         self.COLORS['success'], -1)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     self.COLORS['primary'], 1)

        # Hold progress
        if pose_correct and self.pose_valid_duration > 0:
            hold_progress = min(1.0, self.pose_valid_duration / self.pose_required_duration)
            hold_text = f"Hold still... {int(hold_progress * 100)}%"
            color = self.COLORS['success'] if hold_progress >= 1.0 else (0, 255, 255)
            cv2.putText(frame, hold_text, (w // 2 - 60, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 45), self.COLORS['dark'], -1)
        cv2.putText(frame, "ENROLLMENT MODE - GUIDED SAMPLES",
                   (w // 2 - 130, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['primary'], 1)

        cv2.putText(frame, "Q to quit",
                   (w - 100, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def _draw_capture_flash(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """Draw capture flash effect."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def _save_enrollment(self, name: str, dob: str):
        """Save enrollment to database."""
        print(f"\n  Captured {self.enrollment_count} samples.")

        avg_signature = np.mean(self.enrollment_samples, axis=0).tolist()

        # Get eye shapes from last frame
        landmarks = self.landmark_detector.detect_single(self.cap.read()[1])
        if landmarks:
            eye_left = self.eye_classifier.classify(landmarks, 'left')
            eye_right = self.eye_classifier.classify(landmarks, 'right')
            eye_left_shape = eye_left.shape
            eye_right_shape = eye_right.shape
        else:
            eye_left_shape = "unknown"
            eye_right_shape = "unknown"

        success, message = self.profile_manager.enroll(
            name=name,
            date_of_birth=dob,
            signature=avg_signature,
            eye_shape_left=eye_left_shape,
            eye_shape_right=eye_right_shape,
            sample_count=self.enrollment_count
        )

        if success:
            print("\n  ✓✓✓ ENROLLMENT SUCCESSFUL ✓✓✓")
            print(f"  {message}")
        else:
            print(f"\n  ✗ Enrollment failed: {message}")

    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.close()
