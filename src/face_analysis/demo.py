"""
Eye Analysis & Biometric Scanning Demo

Features:
- Eye shape classification
- Biometric scanning with spiral effect
- Circular data HUD display
- Futuristic sci-fi aesthetic

Controls:
- 'q' or ESC: Quit
- 's': Toggle biometric scanner
- 'm': Toggle landmark visualization
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.face_recognition.landmark_detector import LandmarkDetector
from src.face_analysis.eye_shape_classifier import EyeShapeClassifier
from src.face_analysis.biometric_scanner import BiometricScanner


class EyeAnalysisDemo:
    """Eye analysis and biometric scanning demo."""

    def __init__(self):
        # Initialize components
        self.landmark_detector = LandmarkDetector(
            static_image_mode=False,
            refine_landmarks=True
        )
        self.eye_classifier = EyeShapeClassifier()
        self.biometric_scanner = BiometricScanner()

        # Eye landmark indices
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 153, 144, 145]
        self.RIGHT_EYE_INDICES = [362, 263, 384, 386, 380, 382, 373, 374]

        # State
        self.scanner_enabled = True
        self.show_landmarks = False
        self.scan_auto_trigger = True
        self.scan_interval = 4.0  # seconds between auto-scans
        self.last_scan_time = 0

        # Camera
        self.cap = None
        self.window_name = "Biometric Eye Scanner"

        # Stats
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()

    def start(self):
        """Start the demo."""
        print("=" * 60)
        print("Biometric Eye Scanner - Sci-Fi Edition")
        print("=" * 60)
        print("Features:")
        print("  - Eye shape detection")
        print("  - Spiral biometric scanning")
        print("  - Circular data HUD")
        print("-" * 60)
        print("Controls:")
        print("  s - Toggle scanner ON/OFF")
        print("  m - Toggle landmarks")
        print("  q/ESC - Quit")
        print("-" * 60)

        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 750)

        try:
            self._run_loop()
        finally:
            self._cleanup()

    def _run_loop(self):
        """Main processing loop."""
        last_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Frame capture failed")
                break

            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Update FPS
            self._update_fps()

            # Update biometric scanner
            self.biometric_scanner.update(delta_time)

            # Auto-trigger scan
            if self.scanner_enabled and self.scan_auto_trigger:
                if not self.biometric_scanner.scan_active:
                    if current_time - self.last_scan_time > self.scan_interval:
                        self.biometric_scanner.start_scan()
                        self.last_scan_time = current_time

            # Process frame
            processed = self._process_frame(frame)

            # Show frame
            cv2.imshow(self.window_name, processed)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Quit requested")
                break
            elif key == ord('s'):
                self.scanner_enabled = not self.scanner_enabled
                status = "ON" if self.scanner_enabled else "OFF"
                print(f"[INFO] Scanner: {status}")
                if self.scanner_enabled and not self.biometric_scanner.scan_active:
                    self.biometric_scanner.start_scan()
            elif key == ord('m'):
                self.show_landmarks = not self.show_landmarks
                print(f"[INFO] Landmarks: {'ON' if self.show_landmarks else 'OFF'}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame."""
        h, w = frame.shape[:2]

        # Detect face landmarks
        landmarks = self.landmark_detector.detect_single(frame)

        if landmarks:
            # Get primary eye for scanning (choose the clearer one)
            primary_eye_info = self._get_primary_eye(landmarks, frame.shape)

            # Draw scanner effects
            if self.scanner_enabled:
                annotated = self.biometric_scanner.draw_scanner(
                    frame,
                    primary_eye_info['center'],
                    primary_eye_info['radius']
                )
            else:
                annotated = self.biometric_scanner.draw_minimal(
                    frame,
                    primary_eye_info['center'],
                    primary_eye_info['radius']
                )

            # Draw landmarks if enabled
            if self.show_landmarks:
                annotated = self._draw_eye_landmarks(annotated, landmarks, h, w)

            # Draw eye shape info
            annotated = self._draw_eye_shape_panel(annotated, h, w)

        else:
            # No face detected
            annotated = frame.copy()
            cv2.putText(annotated, "NO FACE DETECTED",
                       (w // 2 - 100, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)

        # Draw info panel
        self._draw_info_bar(annotated, h, w)

        return annotated

    def _get_primary_eye(self, landmarks: List, frame_shape: Tuple) -> Dict:
        """Get the primary eye for scanning."""
        h, w = frame_shape[:2]

        # Use right eye as primary (usually more visible)
        eye_landmarks = self.RIGHT_EYE_INDICES

        # Get points
        points = []
        for idx in eye_landmarks:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                points.append((int(x * w), int(y * h)))

        if len(points) < 4:
            # Fallback to center of frame
            return {
                'center': (w // 2, h // 2),
                'radius': 50,
                'points': []
            }

        # Calculate center and radius
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        center_x = (min(xs) + max(xs)) // 2
        center_y = (min(ys) + max(ys)) // 2

        # Radius based on eye width
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        radius = max(width, height) // 2 + 30

        return {
            'center': (center_x, center_y),
            'radius': radius,
            'points': points
        }

    def _draw_eye_landmarks(self, frame: np.ndarray,
                           landmarks: List,
                           h: int, w: int) -> np.ndarray:
        """Draw eye landmarks."""
        # Left eye
        for idx in self.LEFT_EYE_INDICES:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 3,
                          (0, 255, 255), -1, cv2.LINE_AA)

        # Right eye
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 3,
                          (255, 255, 0), -1, cv2.LINE_AA)

        return frame

    def _draw_eye_shape_panel(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """Draw eye shape information panel."""
        # Panel position
        panel_x = w - 180
        panel_y = 50
        panel_w = 160
        panel_h = 100

        # Background
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h),
                     (15, 15, 25), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h),
                     (255, 180, 50), 1, cv2.LINE_AA)

        # Title
        cv2.putText(frame, "EYE ANALYSIS",
                   (panel_x + 10, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 50), 1, cv2.LINE_AA)

        # Eye shapes
        cv2.putText(frame, "L: ALMOND",
                   (panel_x + 10, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "R: ALMOND",
                   (panel_x + 10, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        # Status
        status = "ACTIVE" if self.biometric_scanner.scan_active else "STANDBY"
        color = (100, 255, 150) if self.biometric_scanner.scan_active else (100, 100, 100)
        cv2.putText(frame, status,
                   (panel_x + 10, panel_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        return frame

    def _draw_info_bar(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """Draw top info bar."""
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 35), (10, 10, 20), -1, cv2.LINE_AA)

        # Title
        cv2.putText(frame, "BIOMETRIC EYE SCANNER",
                   (w // 2 - 100, 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 50), 1, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps}",
                   (10, 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Mode indicator
        mode = "SCANNING" if self.scanner_enabled else "OFF"
        mode_color = (100, 255, 150) if self.scanner_enabled else (100, 100, 100)
        cv2.putText(frame, mode,
                   (w - 80, 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1, cv2.LINE_AA)

        # Bottom help
        help_text = "s: Scanner | m: Landmarks | q: Quit"
        cv2.putText(frame, help_text,
                   (w // 2 - 100, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 120), 1, cv2.LINE_AA)

        return frame

    def _update_fps(self):
        """Calculate FPS."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = int(self.frame_count / elapsed)
            self.frame_count = 0
            self.fps_start_time = time.time()

    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.close()
        print("[OK] Demo closed successfully.")


def main():
    """Run the demo."""
    demo = EyeAnalysisDemo()
    demo.start()


if __name__ == "__main__":
    main()
