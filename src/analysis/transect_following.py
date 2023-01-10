"""
Autonomous transect line following with PID heading controller.

The pipeline detects a coloured guide rope on the pool floor using
HSV segmentation, locates it with Hough line detection, computes
the lateral offset and angular error, and runs a PID controller to
produce a normalised steering command in [-1, 1].

The PID integral term is clamped (anti-windup) to prevent accumulation
during long periods without line detection.
"""

import time

import cv2
import numpy as np


# Default transect line HSV (yellow rope)
LINE_HSV_LOWER = np.array([20, 80, 80])
LINE_HSV_UPPER = np.array([40, 255, 255])

LINE_MIN_AREA = 800
HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 60
HOUGH_MAX_GAP = 20


class PIDController:
    """Discrete PID with derivative filtering and integral anti-windup."""

    def __init__(self, kp=0.6, ki=0.05, kd=0.15, windup_limit=1.0, dt=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup_limit = windup_limit
        self._dt = dt  # None = use real wall-clock time

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._derivative_filter = 0.0
        self._alpha = 0.7  # low-pass for derivative

    def compute(self, error):
        """Compute control output for a given error signal.

        Args:
            error: current error value (normalised offset)

        Returns:
            control output in approximately [-1, 1]
        """
        now = time.monotonic()
        if self._prev_time is None:
            dt = self._dt or 0.033
        else:
            dt = self._dt or max(now - self._prev_time, 0.001)
        self._prev_time = now

        self._integral += error * dt
        self._integral = np.clip(
            self._integral, -self.windup_limit, self.windup_limit
        )

        raw_derivative = (error - self._prev_error) / dt
        self._derivative_filter = (
            self._alpha * self._derivative_filter
            + (1 - self._alpha) * raw_derivative
        )
        self._prev_error = error

        output = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * self._derivative_filter
        )
        return float(np.clip(output, -1.0, 1.0))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._derivative_filter = 0.0


class TransectFollower:
    """Full transect following pipeline with HSV detection and PID control."""

    def __init__(
        self,
        hsv_lower=None,
        hsv_upper=None,
        min_area=LINE_MIN_AREA,
        hough_threshold=HOUGH_THRESHOLD,
        hough_min_length=HOUGH_MIN_LENGTH,
        hough_max_gap=HOUGH_MAX_GAP,
        pid_kp=0.6,
        pid_ki=0.05,
        pid_kd=0.15,
        deadband=0.05,
    ):
        self.hsv_lower = hsv_lower if hsv_lower is not None else LINE_HSV_LOWER
        self.hsv_upper = hsv_upper if hsv_upper is not None else LINE_HSV_UPPER
        self.min_area = min_area
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap
        self.deadband = deadband

        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        self._consecutive_lost = 0
        self._max_lost = 30

    def _build_mask(self, frame):
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return mask

    def _largest_contour(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if not valid:
            return None
        return max(valid, key=cv2.contourArea)

    def _compute_line_angle(self, mask):
        """Median angle of Hough lines detected in mask."""
        lines = cv2.HoughLinesP(
            mask, 1, np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(a)
        return float(np.median(angles))

    def process(self, frame):
        """Run one pipeline iteration.

        Args:
            frame: BGR image

        Returns:
            dict with keys:
                detected (bool), offset_norm, angle_deg,
                steering, line_center, mask
        """
        fh, fw = frame.shape[:2]
        mask = self._build_mask(frame)
        cnt = self._largest_contour(mask)

        if cnt is None:
            self._consecutive_lost += 1
            if self._consecutive_lost >= self._max_lost:
                self.pid.reset()
            return {
                "detected": False,
                "offset_norm": 0.0,
                "angle_deg": 0.0,
                "steering": 0.0,
                "line_center": None,
                "mask": mask,
            }

        self._consecutive_lost = 0
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else fw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else fh // 2

        offset_px = cx - fw // 2
        offset_norm = offset_px / (fw // 2)

        angle_deg = self._compute_line_angle(mask)

        # Dead-band: suppress small corrections
        error = offset_norm if abs(offset_norm) > self.deadband else 0.0
        steering = self.pid.compute(error)

        return {
            "detected": True,
            "offset_norm": round(offset_norm, 3),
            "angle_deg": round(angle_deg, 1),
            "steering": round(steering, 3),
            "line_center": (cx, cy),
            "mask": mask,
        }

    def draw_overlay(self, frame, result):
        """Annotate frame with detection result and steering command."""
        out = frame.copy()
        fh, fw = out.shape[:2]
        mask = result["mask"]

        # Tint detected line region
        tint = np.zeros_like(out)
        tint[mask > 0] = (0, 240, 180)
        out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)

        # Centre reference
        cv2.line(out, (fw // 2, 0), (fw // 2, fh), (200, 200, 200), 1)

        if result["detected"]:
            lc = result["line_center"]
            cv2.circle(out, lc, 10, (0, 50, 255), -1)
            cv2.line(out, (fw // 2, fh), lc, (0, 220, 0), 2)

            offset_str = f"offset: {result['offset_norm']:+.3f}"
            angle_str = f"angle:  {result['angle_deg']:+.1f} deg"
            steer_str = f"steer:  {result['steering']:+.3f}"
            for i, s in enumerate([offset_str, angle_str, steer_str]):
                cv2.putText(
                    out, s, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2
                )
        else:
            cv2.putText(
                out, "LINE NOT DETECTED", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2
            )
        return out

