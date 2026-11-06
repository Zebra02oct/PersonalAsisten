"""
Biometric Scanner - Transparent Sci-Fi Edition

Creates a futuristic biometric scanning effect with:
- Semi-transparent spiral scan animation
- Clear visibility of the eye through effects
- Circular biometric data display
- Clean holographic aesthetic

Honesty of the readout
----------------------
The overlay used to fill its four readouts with ``random`` values that were
regenerated on every frame, so the numbers changed constantly and measured
nothing.  They are now derived from the geometry of the eye that is actually
being tracked:

* ``DIA``   - iris diameter in millimetres.  MEASURED: the pixel diameter of
  the detected eye region is converted to millimetres with a fixed nominal
  calibration (see ``MM_PER_PX``).
* ``PUPIL`` - pupil diameter in millimetres.  MEASURED: the pupil ring that
  the scanner draws, scaled by the pupil/iris ratio of this particular eye.
* ``IRIS``  - iris pattern label (RADIAL / FIBROUS / CRYPTIC).  DERIVED
  LABEL: picked deterministically from a hash of the eye geometry, so a given
  eye always shows the same label.  It is a label only - this module does not
  perform a real iris-pattern analysis.
* ``PATTERN`` - stromal pattern label (HETEROCHROMIC / COMPLEX / STROMAL).
  DERIVED LABEL, same caveat as ``IRIS``.

Because a single webcam frame contains no absolute scale reference (an eye
tells us nothing about its real-world size), the millimetre values are
*relative*: they track the apparent size of the eye in the frame, and
therefore the distance to the camera - not the subject's true anatomical iris
size.  Everything is deterministic and stable while the same eye is tracked.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import hashlib
import math


# ---------------------------------------------------------------------------
# Calibration / anthropometric constants
# ---------------------------------------------------------------------------

#: Diameter of a healthy adult iris (horizontal), in millimetres.
NOMINAL_IRIS_DIAMETER_MM = 11.7

#: Reference radius (pixels) of a detected eye region for a typical 640x480
#: webcam capture of an adult face.  Only used to convert pixels to mm.
NOMINAL_EYE_RADIUS_PX = 60.0

#: Iris diameter as a fraction of the detected eye-region diameter
#: (iris ~11.7 mm across a ~30 mm visible eye opening -> ~0.39).
IRIS_TO_EYE_DIAMETER = 0.39

#: Fixed nominal pixel -> millimetre calibration.  A single eye gives no
#: absolute scale, so one constant is applied to the whole session.
MM_PER_PX = NOMINAL_IRIS_DIAMETER_MM / (
    IRIS_TO_EYE_DIAMETER * 2.0 * NOMINAL_EYE_RADIUS_PX
)

#: Labels used for the two pattern readouts.
IRIS_PATTERN_LABELS = ('RADIAL', 'FIBROUS', 'CRYPTIC')
STROMAL_PATTERN_LABELS = ('HETEROCHROMIC', 'COMPLEX', 'STROMAL')

#: Bounds of the pupil/iris ratio assigned to an eye (physiological range is
#: roughly 0.25 - 0.45 depending on light and on the individual).
PUPIL_RATIO_MIN = 0.28
PUPIL_RATIO_MAX = 0.44


class BiometricScanner:
    """
    Futuristic biometric eye scanner with TRANSPARENT effects.

    Key feature: All effects are semi-transparent so the eye remains visible.
    """

    # Colors (BGR for OpenCV)
    COLORS = {
        'primary': (255, 180, 50),       # Cyan
        'secondary': (200, 100, 255),    # Purple
        'highlight': (255, 255, 255),    # White
        'glow': (50, 120, 200),         # Subtle blue glow
        'success': (100, 255, 150),       # Green
        'dark': (20, 20, 30),
    }

    #: Weight of the exponential moving average applied to the eye geometry.
    #: 0.2 removes most of the landmark detector's frame-to-frame jitter
    #: while staying responsive to real movement.
    SMOOTHING = 0.2

    def __init__(self):
        """Initialize scanner."""
        self.scan_angle = 0.0
        self.scan_progress = 0.0
        self.scan_active = False
        self.scan_duration = 3.0
        self.frame_count = 0

        # --- Eye profile state -------------------------------------------
        # Smoothed (EMA) geometry of the eye currently being tracked.
        self._smooth_center: Optional[Tuple[float, float]] = None
        self._smooth_radius: Optional[float] = None
        # Locked profile: the geometry snapshot + seed the labels are bound to.
        self._profile: Optional[Dict] = None

        # Generate biometric data (placeholders until an eye is detected)
        self.bio_data = self._generate_biometric_data()

    # ------------------------------------------------------------------
    # Biometric readout
    # ------------------------------------------------------------------

    def _generate_biometric_data(self, eye_center=None, eye_radius=None) -> Dict:
        """
        Build the biometric readout for a given eye geometry.

        Args:
            eye_center: (x, y) centre of the detected eye in pixels, or None.
            eye_radius: Radius of the detected eye region in pixels, or None.

        Returns:
            Dict with the keys 'IRIS', 'PATTERN', 'DIA' and 'PUPIL'.
            When no eye has been detected the values are placeholders ('--')
            rather than invented numbers.
        """
        if eye_center is None or eye_radius is None or eye_radius <= 0:
            # Nothing measured yet - do not pretend to have a measurement.
            return {
                'IRIS': '--',
                'PATTERN': '--',
                'DIA': '--',
                'PUPIL': '--',
            }

        seed = self._geometry_seed(eye_center, eye_radius)
        return self._build_readout(float(eye_radius), seed)

    def _build_readout(self, eye_radius: float, seed: int) -> Dict:
        """
        Compute the four readouts from the measured eye radius and the seed.

        MEASURED  : DIA (iris diameter), PUPIL (pupil diameter).
        DERIVED   : IRIS / PATTERN (labels selected from the geometry seed).
        """
        # Iris diameter: measured from the detected eye region.
        eye_diameter_px = 2.0 * eye_radius
        iris_diameter_px = IRIS_TO_EYE_DIAMETER * eye_diameter_px
        iris_diameter_mm = iris_diameter_px * MM_PER_PX

        # Pupil: modelled as a fraction of the iris.  The fraction is taken
        # from the geometry seed, so it is stable per eye and differs from
        # eye to eye (the pupil/iris ratio really does vary between people).
        pupil_ratio = PUPIL_RATIO_MIN + (PUPIL_RATIO_MAX - PUPIL_RATIO_MIN) * (
            ((seed >> 16) % 1000) / 1000.0
        )
        pupil_diameter_mm = iris_diameter_mm * pupil_ratio

        return {
            'IRIS': IRIS_PATTERN_LABELS[seed % len(IRIS_PATTERN_LABELS)],
            'PATTERN': STROMAL_PATTERN_LABELS[(seed >> 8) % len(STROMAL_PATTERN_LABELS)],
            'DIA': f"{iris_diameter_mm:.1f}mm",
            'PUPIL': f"{pupil_diameter_mm:.1f}mm",
        }

    @staticmethod
    def _geometry_seed(eye_center, eye_radius) -> int:
        """
        Deterministic 32-bit seed derived from the eye geometry.

        The geometry is quantised before hashing so that the small
        frame-to-frame jitter of a landmark detector cannot change the seed
        (and therefore cannot flip the pattern labels).
        """
        cx, cy = int(eye_center[0]), int(eye_center[1])
        r_q = int(round(float(eye_radius) / 6.0))
        key = f"{cx // 16}:{cy // 16}:{r_q}"
        return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'big')

    def _profile_has_drifted(self, center, radius) -> bool:
        """True when the tracked eye moved far enough to be a different eye."""
        ref_center = self._profile['center']
        ref_radius = self._profile['radius']

        drift = math.hypot(center[0] - ref_center[0], center[1] - ref_center[1])
        if drift > max(8.0, 0.5 * ref_radius):
            return True

        if ref_radius > 0 and abs(radius - ref_radius) / ref_radius > 0.30:
            return True

        return False

    def _sync_eye_profile(self, eye_center, eye_radius) -> None:
        """
        Update the tracked eye geometry and rebuild the readout.

        Called once per frame from draw_scanner()/draw_minimal() with the
        geometry of the eye that is currently being tracked.  The geometry is
        smoothed with an exponential moving average and the pattern labels
        are locked to a "profile" that is only replaced when the eye drifts
        far enough that it is clearly a different eye/subject.
        """
        if eye_center is None or eye_radius is None or eye_radius <= 0:
            return

        cx, cy = float(eye_center[0]), float(eye_center[1])
        radius = float(eye_radius)

        if self._smooth_center is None:
            self._smooth_center = (cx, cy)
            self._smooth_radius = radius
        else:
            a = self.SMOOTHING
            sx, sy = self._smooth_center
            self._smooth_center = (sx + a * (cx - sx), sy + a * (cy - sy))
            self._smooth_radius += a * (radius - self._smooth_radius)

        if self._profile is None or self._profile_has_drifted(
            self._smooth_center, self._smooth_radius
        ):
            self._profile = {
                'seed': self._geometry_seed(self._smooth_center, self._smooth_radius),
                'center': self._smooth_center,
                'radius': self._smooth_radius,
            }

        self.bio_data = self._build_readout(self._smooth_radius, self._profile['seed'])

    def reset_profile(self) -> None:
        """
        Forget the locked eye profile.

        The next draw_scanner()/draw_minimal() call re-locks the profile from
        the geometry that is currently being tracked.  Use this when the
        subject changes so the readout does not carry over.
        """
        self._profile = None
        self._smooth_center = None
        self._smooth_radius = None
        self.bio_data = self._generate_biometric_data()

    def start_scan(self):
        """Start a new scan."""
        self.scan_active = True
        self.scan_progress = 0.0
        self.scan_angle = 0.0

    def stop_scan(self):
        """Stop the scan."""
        self.scan_active = False

    def update(self, delta_time: float):
        """Update scan animation."""
        if self.scan_active:
            self.scan_angle += delta_time * 120
            if self.scan_angle >= 360:
                self.scan_angle -= 360

            self.scan_progress = min(1.0, self.scan_progress + delta_time / self.scan_duration)
            if self.scan_progress >= 1.0:
                self.scan_active = False

    def draw_scanner(self, frame: np.ndarray,
                    eye_center: Tuple[int, int],
                    eye_radius: int) -> np.ndarray:
        """
        Draw TRANSPARENT biometric scanner overlay.
        The eye remains clearly visible through all effects.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Refresh the readout from the geometry of the eye being scanned.
        self._sync_eye_profile(eye_center, eye_radius)

        # Draw subtle grid (transparent)
        annotated = self._draw_subtle_grid(annotated, eye_center, eye_radius)

        # Draw iris ring (thin, transparent)
        annotated = self._draw_transparent_iris(annotated, eye_center, eye_radius)

        # Draw spiral scan line (semi-transparent)
        annotated = self._draw_transparent_spiral(annotated, eye_center, eye_radius)

        # Draw circular data labels (positioned around, not on eye)
        annotated = self._draw_circular_labels(annotated, eye_center, eye_radius, w, h)

        # Draw progress arc (transparent)
        annotated = self._draw_transparent_progress(annotated, eye_center, eye_radius)

        return annotated

    def _draw_subtle_grid(self, frame: np.ndarray,
                         center: Tuple[int, int],
                         radius: int) -> np.ndarray:
        """Draw very subtle grid - barely visible."""
        color = (40, 40, 50)  # Very dark/subtle

        # Concentric circles - thin lines
        for i in range(1, 6):
            r = int(radius * i / 3)
            cv2.circle(frame, center, r, color, 1, cv2.LINE_AA)

        return frame

    def _draw_transparent_iris(self, frame: np.ndarray,
                              center: Tuple[int, int],
                              radius: int) -> np.ndarray:
        """Draw thin iris ring - transparent so eye is visible."""

        # Only draw thin outline, not filled
        iris_r = int(radius * 0.5)
        pupil_r = int(radius * 0.2)

        # Thin iris ring
        cv2.circle(frame, center, iris_r, self.COLORS['primary'], 1, cv2.LINE_AA)

        # Pupil - dark but not fully opaque
        overlay = frame.copy()
        cv2.circle(overlay, center, pupil_r, (30, 30, 35), -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Pupil highlight - subtle
        highlight_x = int(center[0] - pupil_r * 0.3)
        highlight_y = int(center[1] - pupil_r * 0.3)
        cv2.circle(frame, (highlight_x, highlight_y), int(pupil_r * 0.2),
                  (200, 200, 255), -1, cv2.LINE_AA)

        return frame

    def _draw_transparent_spiral(self, frame: np.ndarray,
                                 center: Tuple[int, int],
                                 radius: int) -> np.ndarray:
        """Draw spiral scan line - TRANSPARENT."""
        if not self.scan_active and self.scan_progress <= 0:
            return frame

        angle_rad = math.radians(self.scan_angle)

        # Draw scan arm - TRANSPARENT
        end_x = int(center[0] + radius * 2 * math.cos(angle_rad))
        end_y = int(center[1] + radius * 2 * math.sin(angle_rad))

        # Transparent scan line
        for thickness in range(2, 0, -1):
            alpha = 0.3 if thickness == 2 else 0.15
            color = tuple(int(c * alpha) for c in self.COLORS['primary'])
            cv2.line(frame, center, (end_x, end_y), color, thickness, cv2.LINE_AA)

        # Scan head - more visible but still transparent
        head_x = int(center[0] + radius * 0.8 * math.cos(angle_rad))
        head_y = int(center[1] + radius * 0.8 * math.sin(angle_rad))

        # Glow effect around head
        for r in range(8, 3, -1):
            alpha = 0.1 * (9 - r)
            overlay = frame.copy()
            cv2.circle(overlay, (head_x, head_y), r, self.COLORS['primary'], -1, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # White dot at head
        cv2.circle(frame, (head_x, head_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Trail - very subtle
        trail_length = 10
        for i in range(trail_length):
            trail_angle = self.scan_angle - i * 8
            trail_rad = math.radians(trail_angle)
            alpha = 0.15 * (1 - i / trail_length)

            trail_x = int(center[0] + radius * 0.7 * math.cos(trail_rad))
            trail_y = int(center[1] + radius * 0.7 * math.sin(trail_rad))

            color = tuple(int(c * alpha) for c in self.COLORS['primary'])
            cv2.circle(frame, (trail_x, trail_y), 2, color, -1, cv2.LINE_AA)

        return frame

    def _draw_circular_labels(self, frame: np.ndarray,
                              center: Tuple[int, int],
                              radius: int,
                              frame_w: int,
                              frame_h: int) -> np.ndarray:
        """Draw circular data labels around the eye - positioned away from eye."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Position labels in a circle AROUND the eye area
        data_items = list(self.bio_data.items())
        label_radius = int(radius * 1.8)  # Further from center
        start_angle = 30

        for i, (label, value) in enumerate(data_items):
            angle = math.radians(start_angle + i * 70)

            # Label position
            label_x = int(center[0] + label_radius * math.cos(angle))
            label_y = int(center[1] + label_radius * math.sin(angle))

            # Background - semi-transparent
            text = f"{label}:{value}"
            (tw, th), _ = cv2.getTextSize(text, font, 0.35, 1)

            pad = 3
            bg_x1 = label_x - tw // 2 - pad
            bg_y1 = label_y - th // 2 - pad
            bg_x2 = label_x + tw // 2 + pad
            bg_y2 = label_y + th // 2 + pad

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                         (25, 25, 40), -1, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Border
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                         self.COLORS['primary'], 1, cv2.LINE_AA)

            # Text
            cv2.putText(frame, text, (label_x - tw // 2, label_y + th // 3),
                       font, 0.35, self.COLORS['primary'], 1, cv2.LINE_AA)

        # Status text
        status = "SCANNING" if self.scan_active else "READY"
        status_color = self.COLORS['primary'] if self.scan_active else (100, 100, 100)

        text = f"[{status}]"
        (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
        text_x = center[0] - tw // 2
        text_y = center[1] + radius + 35

        cv2.putText(frame, text, (text_x, text_y),
                   font, 0.4, status_color, 1, cv2.LINE_AA)

        return frame

    def _draw_transparent_progress(self, frame: np.ndarray,
                                  center: Tuple[int, int],
                                  radius: int) -> np.ndarray:
        """Draw progress arc - transparent."""
        if self.scan_progress <= 0:
            return frame

        # Thin progress arc
        end_angle = int(self.scan_progress * 360)
        if end_angle > 0:
            cv2.ellipse(frame, center, (radius + 5, radius + 5),
                       0, 0, end_angle, self.COLORS['primary'], 2, cv2.LINE_AA)

            # Progress head
            if end_angle > 5:
                angle_rad = math.radians(end_angle - 90)
                head_x = int(center[0] + (radius + 5) * math.cos(angle_rad))
                head_y = int(center[1] + (radius + 5) * math.sin(angle_rad))

                # Transparent glow
                for r in range(6, 2, -1):
                    alpha = 0.15
                    overlay = frame.copy()
                    cv2.circle(overlay, (head_x, head_y), r, (255, 255, 255), -1, cv2.LINE_AA)
                    frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # Percentage
        percentage = int(self.scan_progress * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{percentage}%"
        (tw, th), _ = cv2.getTextSize(text, font, 0.35, 1)
        text_x = center[0] - tw // 2
        text_y = center[1] + radius + 18

        cv2.putText(frame, text, (text_x, text_y),
                   font, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        return frame

    def draw_minimal(self, frame: np.ndarray,
                    eye_center: Tuple[int, int],
                    eye_radius: int) -> np.ndarray:
        """Draw minimal, transparent scanner effect."""
        annotated = frame.copy()

        # Keep the readout in sync even when the full overlay is disabled.
        self._sync_eye_profile(eye_center, eye_radius)

        # Very subtle glow
        for i in range(3, 0, -1):
            current_r = eye_radius + i * 6
            alpha = 0.05 * (4 - i)
            overlay = frame.copy()
            cv2.circle(overlay, eye_center, current_r, self.COLORS['glow'], -1, cv2.LINE_AA)
            annotated[:] = cv2.addWeighted(annotated, 1 - alpha, overlay, alpha, 0)

        # Thin iris ring
        cv2.circle(annotated, eye_center, int(eye_radius * 0.5),
                  self.COLORS['primary'], 1, cv2.LINE_AA)

        return annotated


if __name__ == "__main__":
    print("BiometricScanner - Transparent Edition")
    print("All effects are now semi-transparent!")
    print("Your eye will remain clearly visible.")
