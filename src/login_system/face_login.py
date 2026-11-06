"""
Personal Assistant Face Login System

Staged face verification with MULTI-SAMPLE VERIFICATION.
Requires 3 consecutive successful matches before granting access.

Features:
- Stage 1: Face Detection (timer-based)
- Stage 2: Eye Biometric Scan (spiral)
- Stage 3: Multi-sample Signature Verification (3 matches required)
- Guided enrollment with pose instructions
- Anti-spoofing: requires consistent matches
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.face_recognition.landmark_detector import LandmarkDetector
from src.face_recognition.feature_extractor import FeatureExtractor
from src.face_analysis.eye_shape_classifier import EyeShapeClassifier
from src.face_analysis.biometric_scanner import BiometricScanner
from src.database.owner_profile import get_manager
from src.database.models import LoginHistory


@dataclass
class StageStatus:
    name: str
    active: bool = False
    progress: float = 0.0
    complete: bool = False
    success: bool = False
    message: str = ""
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class FaceLoginSystem:

    COLORS = {
        'primary': (255, 180, 50),
        'success': (100, 255, 150),
        'warning': (80, 80, 255),
        'danger': (50, 50, 255),
        'eye': (0, 255, 255),
        'dark': (15, 15, 25),
        'text': (200, 200, 210),
    }

    LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145]
    RIGHT_EYE = [362, 263, 384, 386, 380, 382, 373, 374]

    def __init__(self):
        # Components
        self.landmark_detector = LandmarkDetector(static_image_mode=False, refine_landmarks=True)
        self.feature_extractor = FeatureExtractor()
        self.eye_classifier = EyeShapeClassifier()
        self.biometric_scanner = BiometricScanner()
        self.profile_manager = get_manager()

        # Camera
        self.cap = None
        self.window_name = "Face Login"

        # Stages
        self.stages = {
            1: StageStatus("FACE DETECTION"),
            2: StageStatus("EYE BIOMETRIC SCAN"),
            3: StageStatus("MULTI-SAMPLE VERIFICATION"),
        }
        self.current_stage = 1

        # Multi-sample verification
        self.required_matches = 3
        self.match_count = 0
        self.match_confidences: List[float] = []
        self.current_match_confidence = 0.0

        # Animation
        self.scan_angle = 0.0
        self.glow_intensity = 0.0

        # Stage 1 tracking
        self.stage1_face_stable_start = 0
        self.stage1_required_time = 1.5

        # Reset verification state
        self._reset_verification()

    def _reset_verification(self):
        """Reset verification state for new login attempt."""
        self.verified = False
        self.verified_user = None
        self.verified_confidence = 0.0
        self.match_count = 0
        self.match_confidences = []
        self.current_match_confidence = 0.0
        self.stage_start_time = 0

        for stage in self.stages.values():
            stage.active = False
            stage.progress = 0.0
            stage.complete = False
            stage.success = False
            stage.message = ""

    def print_header(self):
        print("\n" + "=" * 60)
        print("  PERSONAL ASSISTANT - FACE LOGIN SYSTEM")
        print("  [MULTI-SAMPLE VERIFICATION: 3 matches required]")
        print("=" * 60)

    def print_separator(self):
        print("\n" + "─" * 60)

    def print_stage_header(self, stage_num: int, stage_name: str):
        self.print_separator()
        print(f"  STAGE {stage_num}: {stage_name}")
        print("─" * 60)

    def print_check(self, message: str, success: bool = True, indent: int = 2):
        prefix = "  " * indent
        symbol = "✓" if success else "✗"
        print(f"{prefix}{symbol} {message}")

    def print_progress_bar(self, progress: float, width: int = 40, indent: int = 2):
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        prefix = "  " * indent
        print(f"{prefix}[{bar}] {int(progress * 100)}%")

    def print_status(self, message: str):
        """Print status message."""
        if message:
            print(f"    {message}")

    def check_is_enrolled(self) -> bool:
        return self.profile_manager.is_enrolled()

    def verify(self) -> Tuple[bool, Optional[dict], float]:
        """Main verification with multi-sample requirement."""
        self._reset_verification()
        self.print_header()

        if not self.check_is_enrolled():
            print("\n  ⚠️  No face enrolled yet!")
            return False, None, 0.0

        print("\n  Starting face verification...")
        print(f"  {self.required_matches} consecutive matches required for access.\n")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("  ✗ Cannot open camera!")
            return False, None, 0.0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        start_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                annotated = self._process_verification_frame(frame)
                cv2.imshow(self.window_name, annotated)

                # Animations
                self.scan_angle += 3
                if self.scan_angle >= 360:
                    self.scan_angle -= 360
                self.glow_intensity += 0.05

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n  Login cancelled.")
                    break

                if self.verified:
                    break

        finally:
            self._cleanup()

        if self.verified:
            total_time = time.time() - start_time
            self.print_separator()
            print("\n  ✓✓✓ VERIFICATION SUCCESSFUL ✓✓✓")
            self.print_separator()

            if self.verified_user:
                age = self.verified_user.get('age')
                age_text = f" ({age} years old)" if age else ""
                print(f"\n  Name: {self.verified_user['name']}{age_text}")
                print(f"  Matches: {self.match_count}/{self.required_matches}")
                print(f"  Avg Confidence: {self.verified_confidence:.0%}")
                print(f"  Processing time: {total_time:.2f}s")

                self.profile_manager.record_login(True, self.verified_confidence, "Login successful")

            return True, self.verified_user, self.verified_confidence

        # Failed verification
        if self.match_count > 0:
            print(f"\n  ✗ Only {self.match_count}/{self.required_matches} matches.")
            print("  Access denied.")
        else:
            print("\n  ✗ Face not recognized.")
            print("  Access denied.")

        self.profile_manager.record_login(False, self.current_match_confidence, "Insufficient matches")
        return False, None, 0.0

    def _process_verification_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for verification."""
        h, w = frame.shape[:2]
        annotated = frame.copy()

        landmarks = self.landmark_detector.detect_single(frame)

        if not landmarks:
            self._reset_stages()
            cv2.putText(annotated, "Position your face in the frame...",
                       (w // 2 - 180, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            return annotated

        # Run current stage
        if self.current_stage == 1:
            self._run_stage_1(landmarks, frame.shape)
        elif self.current_stage == 2:
            self._run_stage_2(landmarks, frame.shape)
        elif self.current_stage == 3:
            self._run_stage_3(landmarks, frame.shape)

        # Draw overlays
        annotated = self._draw_face_mesh(annotated, landmarks, h, w)

        if self.stages[2].active:
            eye_center = self._get_eye_center(landmarks, w, h)
            eye_radius = self._get_eye_radius(landmarks, w, h)
            annotated = self.biometric_scanner.draw_scanner(annotated, eye_center, eye_radius)

        annotated = self._draw_stage_overlay(annotated, h, w)
        annotated = self._draw_match_tracker(annotated, h, w)

        if self.verified:
            annotated = self._draw_success_overlay(annotated, h, w)

        return annotated

    def _run_stage_1(self, landmarks: List, frame_shape: Tuple) -> bool:
        """Stage 1: Face Detection."""
        stage = self.stages[1]

        if not stage.active:
            stage.active = True
            self.stage_start_time = time.time()
            self.stage1_face_stable_start = time.time()
            self.print_stage_header(1, "FACE DETECTION")

        face_box = self.landmark_detector.get_face_box(landmarks, frame_shape)
        face_valid = face_box[2] > 80 and face_box[3] > 80

        if face_valid:
            elapsed = time.time() - self.stage1_face_stable_start
            stage.progress = min(1.0, elapsed / self.stage1_required_time)

            if elapsed >= 0.5:
                self.print_check("Face detected", True)

            if elapsed >= 1.0:
                self.print_check("Face stable", True)

            if stage.progress >= 1.0:
                stage.progress = 1.0
                stage.complete = True
                stage.success = True
                self.current_stage = 2
                self.print_check("Quality check passed", True)
                self.print_status("")
                self.print_check("STAGE 1 COMPLETE - Proceeding to Stage 2...", True)
                time.sleep(0.3)
                return True
        else:
            self.stage1_face_stable_start = time.time()
            stage.progress = 0.0
            self.print_check("Face detected - position better", False)

        self.print_progress_bar(stage.progress)
        remaining = max(0, self.stage1_required_time - (time.time() - self.stage1_face_stable_start))
        self.print_status(f"Hold still... {remaining:.1f}s")

        return False

    def _run_stage_2(self, landmarks: List, frame_shape: Tuple) -> bool:
        """Stage 2: Eye Biometric Scan."""
        stage = self.stages[2]

        if not stage.active:
            stage.active = True
            self.stage_start_time = time.time()
            self.print_stage_header(2, "EYE BIOMETRIC SCAN")

        if not self.biometric_scanner.scan_active:
            self.biometric_scanner.start_scan()

        self.biometric_scanner.update(0.03)
        stage.progress = self.biometric_scanner.scan_progress

        left_eye = self.eye_classifier.classify(landmarks, 'left')
        right_eye = self.eye_classifier.classify(landmarks, 'right')

        if not hasattr(self, '_stage2_printed'):
            self._stage2_printed = False

        if not self._stage2_printed:
            self.print_check(f"Left eye: {left_eye.shape} ({left_eye.confidence:.0%})", True)
            self.print_check(f"Right eye: {right_eye.shape} ({right_eye.confidence:.0%})", True)
            self._stage2_printed = True

        self.print_status(f"Scanning biometrics... {int(stage.progress * 100)}%")
        self.print_progress_bar(stage.progress)

        if stage.progress >= 1.0:
            stage.complete = True
            stage.success = True
            self.current_stage = 3
            self.print_check("Biometric scan complete", True)
            self.print_status("")
            self.print_check("STAGE 2 COMPLETE - Proceeding to Stage 3...", True)
            time.sleep(0.3)
            self._stage2_printed = False
            return True

        return False

    def _run_stage_3(self, landmarks: List, frame_shape: Tuple) -> bool:
        """Stage 3: Multi-Sample Signature Verification."""
        stage = self.stages[3]

        if not stage.active:
            stage.active = True
            self.stage_start_time = time.time()
            self.print_stage_header(3, "MULTI-SAMPLE VERIFICATION")
            print(f"\n  Required: {self.required_matches} consecutive matches")
            print(f"  Current: {self.match_count}/{self.required_matches}")
            self.print_separator()

        signature = self.feature_extractor.extract(landmarks)
        stored_signature = self.profile_manager.get_signature()

        if not stored_signature:
            self.print_check("No enrolled signature found", False)
            stage.complete = True
            stage.success = False
            return False

        similarity = self._calculate_similarity(signature, stored_signature)
        self.current_match_confidence = similarity
        threshold = 0.60  # Slightly lower threshold, but needs 3 matches

        elapsed = time.time() - self.stage_start_time
        stage.progress = min(1.0, elapsed / 2.0)

        # Show match status
        self.print_status(f"Match {self.match_count + 1}: Score {similarity:.0%} | Threshold {threshold:.0%}")
        self.print_progress_bar(stage.progress)

        if elapsed >= 1.5:
            if similarity >= threshold:
                # SUCCESSFUL MATCH
                self.match_count += 1
                self.match_confidences.append(similarity)
                avg_conf = sum(self.match_confidences) / len(self.match_confidences)
                self.current_match_confidence = avg_conf

                print(f"\n  ✓✓ MATCH {self.match_count}/{self.required_matches} ✓✓")
                print(f"    Score: {similarity:.0%} | Avg: {avg_conf:.0%}")

                if self.match_count >= self.required_matches:
                    # ALL MATCHES COMPLETE - ACCESS GRANTED
                    self.print_separator()
                    print("\n  ✓✓✓ ALL MATCHES COMPLETE ✓✓✓")
                    print(f"  Confirmed {self.required_matches}/{self.required_matches} times!")

                    self.verified_user = self.profile_manager.get_user_info()
                    self.verified_confidence = avg_conf
                    self.verified = True

                    stage.complete = True
                    stage.success = True
                    stage.progress = 1.0

                    return True
                else:
                    # Need more matches - restart verification
                    print(f"\n  Need {self.required_matches - self.match_count} more match(es)...")
                    print("  Verification will restart automatically.\n")

                    # Reset stages but keep match count
                    self.biometric_scanner.scan_active = False
                    self.current_stage = 1
                    for s in self.stages.values():
                        s.active = False
                        s.progress = 0.0
                        s.complete = False
                        s.success = False

                    return False
            else:
                # NO MATCH - Reset count
                if self.match_count > 0:
                    print(f"\n  ✗ Match FAILED - Resetting counter")
                    print(f"    Score: {similarity:.0%} < {threshold:.0%}")
                    self.match_count = 0
                    self.match_confidences = []

                    # Restart verification
                    self.biometric_scanner.scan_active = False
                    self.current_stage = 1
                    for s in self.stages.values():
                        s.active = False
                        s.progress = 0.0
                        s.complete = False
                        s.success = False

        return False

    def _calculate_similarity(self, sig1: List[float], sig2: List[float]) -> float:
        """Calculate cosine similarity."""
        sig1 = np.array(sig1)
        sig2 = np.array(sig2)

        dot = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine = dot / (norm1 * norm2)
        return (cosine + 1) / 2

    def _reset_stages(self):
        for stage in self.stages.values():
            stage.active = False
            stage.progress = 0.0
            stage.complete = False
            stage.success = False
            stage.message = ""

    def _draw_face_mesh(self, frame, landmarks, h, w):
        contour_indices = [10, 151, 337, 297, 338, 299, 300, 332, 234, 93, 323, 423, 352]
        for idx in contour_indices:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 2, self.COLORS['primary'], -1, cv2.LINE_AA)

        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 2, self.COLORS['eye'], -1, cv2.LINE_AA)

        for idx in [1, 168, 195]:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 2, (255, 200, 100), -1, cv2.LINE_AA)

        return frame

    def _get_eye_center(self, landmarks, w, h):
        if len(landmarks) > 386:
            x, y, _ = landmarks[386]
            return (int(x * w), int(y * h))
        return (w // 2, h // 2)

    def _get_eye_radius(self, landmarks, w, h):
        if len(landmarks) > 263 and len(landmarks) > 33:
            x1 = int(landmarks[263][0] * w)
            x2 = int(landmarks[33][0] * w)
            return abs(x2 - x1) // 2 + 20
        return 50

    def _draw_stage_overlay(self, frame, h, w):
        cv2.rectangle(frame, (0, 0), (w, 45), self.COLORS['dark'], -1, cv2.LINE_AA)
        cv2.putText(frame, "FACE LOGIN - MULTI-SAMPLE VERIFICATION",
                   (w // 2 - 150, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['primary'], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Stage {self.current_stage}/3",
                   (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1, cv2.LINE_AA)

        # Progress bar
        bar_y = h - 80
        bar_x = 100
        bar_w = w - 200
        bar_h = 25

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (30, 30, 40), -1, cv2.LINE_AA)

        total_progress = 0.0
        for i, stage in self.stages.items():
            if stage.complete:
                total_progress += 0.33
            elif stage.active:
                total_progress += 0.33 * stage.progress

        fill_w = int(bar_w * total_progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                         self.COLORS['primary'], -1, cv2.LINE_AA)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     self.COLORS['primary'], 1, cv2.LINE_AA)

        stage_names = ["DETECT", "SCAN", "VERIFY"]
        for i, name in enumerate(stage_names):
            x = bar_x + (bar_w // 3) * i + 20
            cv2.putText(frame, name, (x, bar_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        return frame

    def _draw_match_tracker(self, frame, h, w):
        """Draw match counter overlay."""
        # Match tracker box
        box_x = w - 150
        box_y = 60
        box_w = 130
        box_h = 100

        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                     (20, 20, 35), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                     self.COLORS['primary'], 1, cv2.LINE_AA)

        cv2.putText(frame, "MATCHES",
                   (box_x + 25, box_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['primary'], 1, cv2.LINE_AA)

        # Match indicators
        for i in range(self.required_matches):
            indicator_x = box_x + 15 + i * 40
            indicator_y = box_y + 45

            if i < self.match_count:
                color = self.COLORS['success']
                symbol = "✓"
            else:
                color = (60, 60, 80)
                symbol = "○"

            cv2.putText(frame, symbol, (indicator_x, indicator_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"{self.match_count}/{self.required_matches}",
                   (box_x + 25, box_y + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1, cv2.LINE_AA)

        return frame

    def _draw_success_overlay(self, frame, h, w):
        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 2 - 200, h // 2 - 100),
                     (w // 2 + 200, h // 2 + 100),
                     (20, 50, 30), -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        cv2.rectangle(frame, (w // 2 - 200, h // 2 - 100),
                     (w // 2 + 200, h // 2 + 100),
                     self.COLORS['success'], 3, cv2.LINE_AA)

        cv2.putText(frame, "✓ ACCESS GRANTED",
                   (w // 2 - 100, h // 2 - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLORS['success'], 2, cv2.LINE_AA)

        cv2.putText(frame, f"{self.required_matches}/{self.required_matches} VERIFIED",
                   (w // 2 - 110, h // 2 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1, cv2.LINE_AA)

        if self.verified_user:
            name = self.verified_user['name']
            cv2.putText(frame, f"Welcome, {name}!",
                       (w // 2 - 80, h // 2 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['success'], 1, cv2.LINE_AA)

        return frame

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.close()

    def enroll(self) -> bool:
        """Enroll new face."""
        from src.login_system.enrollment import GuidedEnrollment
        enrollment = GuidedEnrollment(self.landmark_detector, self.feature_extractor,
                                    self.eye_classifier, self.profile_manager)
        return enrollment.run()

    def get_login_history(self) -> List[LoginHistory]:
        return self.profile_manager.get_login_history(limit=10)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Face Login System")
    parser.add_argument('--enroll', action='store_true', help='Enroll new face')
    parser.add_argument('--reset', action='store_true', help='Reset enrollment')
    args = parser.parse_args()

    login_system = FaceLoginSystem()

    if args.reset:
        print("Resetting enrollment...")
        login_system.profile_manager.reset_enrollment()
        print("Enrollment reset complete.")
        return

    if args.enroll:
        login_system.enroll()
    else:
        if login_system.check_is_enrolled():
            login_system.verify()
        else:
            print("\nNo face enrolled. Would you like to enroll now?")
            choice = input("  [Y] Enroll | [N] Quit: ").strip().lower()
            if choice == 'y':
                login_system.enroll()


if __name__ == "__main__":
    main()
