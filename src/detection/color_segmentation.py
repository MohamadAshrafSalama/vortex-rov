"""
Multi-color HSV segmentation with morphological post-processing.

Each color is represented by one or two HSV ranges (red wraps around 0/180).
Morphological opening removes noise; closing fills gaps in larger blobs.
Contours are filtered by area and circularity to reject non-target shapes.
"""

import cv2
import numpy as np


# Default HSV ranges for competition colors
DEFAULT_HSV_RANGES = {
    "red": {
        "lower1": np.array([0, 100, 60]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([165, 100, 60]),
        "upper2": np.array([180, 255, 255]),
    },
    "blue": {
        "lower": np.array([100, 80, 50]),
        "upper": np.array([130, 255, 255]),
    },
    "green": {
        "lower": np.array([40, 60, 50]),
        "upper": np.array([85, 255, 255]),
    },
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([35, 255, 255]),
    },
    "orange": {
        "lower": np.array([10, 120, 100]),
        "upper": np.array([20, 255, 255]),
    },
    "white": {
        "lower": np.array([0, 0, 200]),
        "upper": np.array([180, 40, 255]),
    },
    "brown": {
        "lower": np.array([8, 60, 30]),
        "upper": np.array([20, 200, 130]),
    },
}


def _circularity(contour):
    """Compute 4*pi*area / perimeter^2. Circle = 1.0, jagged = near 0."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


class ColorSegmenter:
    """HSV-based color detection for all competition target colors."""

    def __init__(
        self,
        hsv_ranges=None,
        kernel_size=7,
        open_iters=1,
        close_iters=2,
        min_area=400,
        max_area=80000,
        min_circularity=0.25,
    ):
        self.hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        ksz = kernel_size | 1  # ensure odd
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
        self._open_iters = open_iters
        self._close_iters = close_iters

    def build_mask(self, hsv, color):
        """Create a binary mask for a named color in an HSV image."""
        params = self.hsv_ranges.get(color)
        if params is None:
            raise KeyError(f"Unknown color: {color}")

        if "lower1" in params:
            mask = cv2.inRange(hsv, params["lower1"], params["upper1"])
            mask2 = cv2.inRange(hsv, params["lower2"], params["upper2"])
            mask = cv2.bitwise_or(mask, mask2)
        else:
            mask = cv2.inRange(hsv, params["lower"], params["upper"])

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self._open_kernel, iterations=self._open_iters
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._close_kernel, iterations=self._close_iters
        )
        return mask

    def find_contours(self, mask):
        """Return filtered contours sorted by area descending."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            if _circularity(cnt) < self.min_circularity:
                continue
            valid.append(cnt)
        valid.sort(key=cv2.contourArea, reverse=True)
        return valid

    def detect(self, frame, colors=None):
        """Run detection on a BGR frame for the specified colors.

        Args:
            frame: BGR image
            colors: list of color names to check, or None for all

        Returns:
            list of detection dicts: {color, contour, bbox, center, area, circularity}
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target_colors = colors if colors else list(self.hsv_ranges.keys())
        detections = []

        for color in target_colors:
            try:
                mask = self.build_mask(hsv, color)
            except KeyError:
                continue
            for cnt in self.find_contours(mask):
                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append({
                    "color": color,
                    "contour": cnt,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "area": cv2.contourArea(cnt),
                    "circularity": round(_circularity(cnt), 3),
                })

        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes, centroids, and labels on a copy of frame."""
        color_bgr = {
            "red": (0, 0, 220),
            "blue": (220, 80, 0),
            "green": (0, 200, 0),
            "yellow": (0, 220, 220),
            "orange": (0, 140, 255),
            "white": (220, 220, 220),
            "brown": (42, 85, 139),
        }
        out = frame.copy()
        for det in detections:
            bgr = color_bgr.get(det["color"], (200, 200, 200))
            x, y, w, h = det["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
            cv2.circle(out, det["center"], 5, bgr, -1)
            label = f"{det['color']} {det['area']:.0f}px"
            cv2.putText(out, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)
        return out

    def color_mask_overlay(self, frame, color, alpha=0.35):
        """Return frame with detected color region highlighted."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.build_mask(hsv, color)
        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 200)
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
