"""
Colored box detection and classification for MATE ROV 2024.

Detects rectangular boxes submerged on the pool floor. Uses HSV
segmentation per color, polygon approximation to confirm box shape,
and sorts candidates by apparent size (proxy for distance).
"""

import cv2
import numpy as np

from .color_segmentation import ColorSegmenter, DEFAULT_HSV_RANGES


class BoxDetector:
    """Detect and classify colored rectangular boxes in underwater frames."""

    # Colors expected to appear as box targets in the competition
    BOX_COLORS = ["red", "blue", "green", "yellow", "orange"]

    def __init__(
        self,
        min_area=1500,
        max_area=60000,
        approx_epsilon=0.04,
        aspect_min=0.45,
        aspect_max=2.2,
        kernel_size=7,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.approx_epsilon = approx_epsilon
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

        self._segmenter = ColorSegmenter(
            hsv_ranges=DEFAULT_HSV_RANGES,
            kernel_size=kernel_size,
            open_iters=1,
            close_iters=2,
            min_area=min_area,
            max_area=max_area,
            min_circularity=0.1,
        )

    def _is_box_shape(self, contour):
        """Check whether contour approximates a rectangular polygon."""
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, self.approx_epsilon * perimeter, True
        )
        # Accept quadrilaterals and some 5-6 vertex approximations
        if len(approx) < 4 or len(approx) > 7:
            return False, None
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return False, None
        aspect = w / h
        if not (self.aspect_min < aspect < self.aspect_max):
            return False, None
        return True, approx

    def _compute_solidity(self, contour):
        """Ratio of contour area to convex hull area. Boxes have high solidity."""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            return 0.0
        return cv2.contourArea(contour) / hull_area

    def detect(self, frame):
        """Detect all colored boxes in a BGR frame.

        Returns:
            list of dicts with keys:
                color, bbox (x,y,w,h), center, area, distance_rank,
                solidity, approx_corners
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = []

        for color in self.BOX_COLORS:
            try:
                mask = self._segmenter.build_mask(hsv, color)
            except KeyError:
                continue

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue

                is_box, approx = self._is_box_shape(cnt)
                if not is_box:
                    continue

                solidity = self._compute_solidity(cnt)
                if solidity < 0.65:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                results.append({
                    "color": color,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "area": area,
                    "solidity": round(solidity, 3),
                    "approx_corners": approx,
                })

        # Sort by area descending — larger apparent size means closer to camera
        results.sort(key=lambda d: d["area"], reverse=True)
        for rank, det in enumerate(results):
            det["distance_rank"] = rank

        return results

    def draw(self, frame, detections):
        """Annotate frame with detected boxes."""
        color_bgr = {
            "red": (0, 0, 220),
            "blue": (220, 60, 0),
            "green": (0, 200, 0),
            "yellow": (0, 220, 220),
            "orange": (0, 130, 255),
        }
        out = frame.copy()
        for det in detections:
            bgr = color_bgr.get(det["color"], (180, 180, 180))
            x, y, w, h = det["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)

            if det["approx_corners"] is not None:
                cv2.drawContours(out, [det["approx_corners"]], -1, bgr, 1)

            cv2.circle(out, det["center"], 6, bgr, -1)

            label = (
                f"{det['color'].upper()} "
                f"area={det['area']:.0f} "
                f"sol={det['solidity']:.2f}"
            )
            cv2.putText(
                out, label, (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1
            )
            rank_label = f"#{det['distance_rank'] + 1}"
            cv2.putText(
                out, rank_label, (x + w - 28, y + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        return out

    def primary_target(self, detections, preferred_color=None):
        """Select the closest (largest) box, optionally filtered by color."""
        candidates = detections
        if preferred_color:
            candidates = [d for d in detections if d["color"] == preferred_color]
        if not candidates:
            candidates = detections
        if not candidates:
            return None
        return candidates[0]

