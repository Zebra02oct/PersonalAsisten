"""
Face Detection Module - Phase 1 (complete)

Real-time face detection using OpenCV Haar Cascade.
Displays webcam feed with bounding boxes around detected faces.

Optimized for better FPS performance.

Notes
-----
The cascade XML is resolved relative to THIS FILE (``__file__``), never
relative to the current working directory, so the detector keeps working no
matter where the script is launched from.  If the bundled copy is missing we
fall back to the copy shipped inside the installed ``cv2`` package.
"""

import cv2
import sys
import time
from pathlib import Path


class FaceDetector:
    """Face detector using Haar Cascade classifier."""

    #: Name of the Haar cascade used for frontal-face detection.
    CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

    def __init__(self, cascade_path: str = None):
        """
        Initialize the face detector.

        Args:
            cascade_path: Path to Haar cascade XML file.  When None the file
                          bundled next to this module is used, falling back to
                          the copy shipped inside the ``cv2`` package.

        Raises:
            FileNotFoundError: If no cascade XML can be found.
            ValueError: If the cascade file exists but cannot be parsed.
        """
        self.cascade_path = self._resolve_cascade_path(cascade_path)

        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))

        if self.face_cascade.empty():
            raise ValueError(
                f"Failed to load face cascade classifier: {self.cascade_path}"
            )

    @classmethod
    def _resolve_cascade_path(cls, cascade_path=None) -> Path:
        """
        Locate the Haar cascade XML.

        Resolution order:
          1. the explicit ``cascade_path`` argument;
          2. the copy bundled next to this module;
          3. the copy shipped inside the installed ``cv2`` package.

        All lookups are absolute, so the result does not depend on the
        current working directory.
        """
        if cascade_path is not None:
            path = Path(cascade_path)
            if not path.exists():
                raise FileNotFoundError(f"Cascade file not found: {path}")
            return path

        bundled = Path(__file__).resolve().parent / cls.CASCADE_FILENAME
        if bundled.exists():
            return bundled

        # Fallback: the cascade shipped with the OpenCV wheel.
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        fallback = Path(data_dir) / cls.CASCADE_FILENAME if data_dir else None
        if fallback is not None and fallback.exists():
            return fallback

        raise FileNotFoundError(
            "Haar cascade file not found. Looked for:\n"
            f"  - {bundled}\n"
            f"  - {fallback}\n"
            "Reinstall opencv-contrib-python or pass cascade_path=..."
        )

    def detect(self, gray_image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image.

        Args:
            gray_image: Grayscale image from cv2.cvtColor.  A 3- or 4-channel
                        BGR/BGRA frame is accepted as well and converted
                        internally, so the raw camera frame can be passed
                        directly.
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for validation
            min_size: Minimum face size

        Returns:
            List of lists (x, y, width, height) for each detected face
        """
        if gray_image is None or getattr(gray_image, "size", 0) == 0:
            return []

        # The cascade works on a single channel - convert once here instead
        # of letting OpenCV raise on an unexpected number of channels.
        if gray_image.ndim == 3:
            if gray_image.shape[2] == 4:
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGRA2GRAY)
            else:
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        return faces.tolist() if len(faces) > 0 else []


class OptimizedFaceDetector:
    """
    Optimized face detector with frame skipping and resolution scaling.
    """

    def __init__(self, cascade_path: str = None, detect_every_n_frames: int = 3,
                 detection_scale: float = 0.5):
        """
        Initialize optimized face detector.

        Args:
            cascade_path: Path to Haar cascade XML file
            detect_every_n_frames: Run detection every N frames (default: 3)
            detection_scale: Scale factor for detection resolution
                           Lower = faster but less accurate

        Raises:
            ValueError: If detect_every_n_frames < 1 or detection_scale <= 0.
                        Both would otherwise fail later with a
                        ZeroDivisionError / cv2.resize error.
        """
        if detect_every_n_frames < 1:
            raise ValueError(
                f"detect_every_n_frames must be >= 1, got {detect_every_n_frames}"
            )
        if detection_scale <= 0:
            raise ValueError(
                f"detection_scale must be > 0, got {detection_scale}"
            )

        self.base_detector = FaceDetector(cascade_path)
        self.detect_every_n_frames = detect_every_n_frames
        self.detection_scale = detection_scale
        self.frame_count = 0
        self.last_faces = []  # Store last detected faces to reuse

    def detect(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces with optimization.
        Only runs heavy detection every N frames.

        Args:
            frame: BGR camera frame (a grayscale frame is accepted too).
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for validation
            min_size: Minimum face size, expressed in the resolution of the
                      ORIGINAL frame.  It is scaled down together with the
                      frame so that the optimized mode finds the same faces
                      as the standard mode.

        Returns:
            List of tuples (x, y, width, height) in the original resolution.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return self.last_faces

        self.frame_count += 1

        # Only detect every N frames
        if self.frame_count % self.detect_every_n_frames == 0:
            # Create smaller version for faster detection
            small_frame = cv2.resize(
                frame,
                None,
                fx=self.detection_scale,
                fy=self.detection_scale,
                interpolation=cv2.INTER_AREA
            )

            # Convert to grayscale at lower resolution
            if small_frame.ndim == 3:
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = small_frame

            # The minimum face size must be scaled with the frame, otherwise
            # the optimized mode silently searches for faces that are
            # 1/detection_scale times larger than in the standard mode.
            scaled_min_size = (
                max(1, int(round(min_size[0] * self.detection_scale))),
                max(1, int(round(min_size[1] * self.detection_scale))),
            )

            # Detect faces
            faces = self.base_detector.detect(
                gray,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=scaled_min_size,
            )

            # Scale coordinates back to original resolution
            scale = 1.0 / self.detection_scale
            self.last_faces = [
                (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
                for x, y, w, h in faces
            ]

        return self.last_faces


def open_camera(camera_index: int = 0):
    """
    Open the webcam.

    Returns:
        An opened cv2.VideoCapture.

    Raises:
        RuntimeError: If the camera cannot be opened (no webcam, device
                      busy, or an invalid index).
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        # Release the failed handle instead of leaking it.
        cap.release()
        raise RuntimeError(
            "Cannot open webcam. Please check:\n"
            "  - Webcam is connected\n"
            "  - No other application is using the webcam\n"
            f"  - Camera index {camera_index} exists"
        )

    return cap


def run_demo(use_optimized: bool = False, camera_index: int = 0):
    """
    Run the face detection demo.

    Args:
        use_optimized: Use optimized version with frame skipping
        camera_index: Index of the camera to open (0 = default webcam)
    """
    print("=" * 50)
    print("Face Detection Demo - Phase 1")
    print("=" * 50)
    print(f"Mode: {'Optimized' if use_optimized else 'Standard'}")
    print("-" * 50)

    # Initialize detector
    try:
        if use_optimized:
            detector = OptimizedFaceDetector(
                detect_every_n_frames=3,  # Detect every 3rd frame
                detection_scale=0.5       # Process at 50% resolution
            )
            print("[OK] Optimized face detector loaded")
        else:
            detector = FaceDetector()
            print("[OK] Face detector loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load face detector: {e}")
        sys.exit(1)

    # Open webcam
    print("[INFO] Opening webcam...")
    try:
        cap = open_camera(camera_index)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Set camera resolution (optional - uncomment if needed)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[OK] Webcam opened successfully")
    print("[INFO] Controls:")
    print("        'q' or ESC - Quit")
    print("        'o' - Toggle optimized mode")
    print("-" * 50)

    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    display_optimized = use_optimized

    # Create window
    window_name = "Face Detection Demo"
    window_created = False

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        window_created = True

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Failed to capture frame")
                break

            # Detect faces
            faces = detector.detect(frame)

            # Draw bounding boxes
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

            # Draw info overlay
            info_text = f"Faces: {len(faces)} | FPS: {current_fps} | Mode: {'OPT' if display_optimized else 'STD'}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = int(fps_frame_count / elapsed)
                fps_frame_count = 0
                fps_start_time = time.time()

            # Show frame
            cv2.imshow(window_name, frame)

            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("[INFO] Quit requested")
                break
            elif key == ord('o'):
                # Toggle optimized mode (restart with new detector)
                display_optimized = not display_optimized
                if display_optimized:
                    detector = OptimizedFaceDetector(detect_every_n_frames=3, detection_scale=0.5)
                    print("[INFO] Switched to OPTIMIZED mode")
                else:
                    detector = FaceDetector()
                    print("[INFO] Switched to STANDARD mode")
    finally:
        # Always release the camera and close the window, even if the loop
        # above raised (e.g. no GUI backend available).
        cap.release()
        if window_created:
            cv2.destroyAllWindows()
        print("[OK] Camera released, window closed")

    print("-" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    # Check command line argument
    use_optimized = '--optimized' in sys.argv or '-o' in sys.argv
    run_demo(use_optimized=use_optimized)
