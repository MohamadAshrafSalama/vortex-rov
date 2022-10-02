"""
POI marker detection with spatial offset computation.

Markers are small coloured objects (buoys, tags, discs) placed at
points of interest. Detection pipeline: HSV mask -> morphology ->
contour filter by area and aspect ratio -> centroid extraction.
Spatial offset is the vector from the frame centre to the marker,
normalised to [-1, 1] range for downstream coordinate projection.
"""

import cv2
import numpy as np

from .color_segmentation import ColorSegmenter, DEFAULT_HSV_RANGES


class MarkerDetector:
    """Detect POI markers by color and compute frame-relative positions."""

    MARKER_COLORS = ["red", "blue", "green", "yellow", "orange"]

    def __init__(
        self,
        min_area=200,
        max_area=8000,
        aspect_min=0.25,
        aspect_max=4.0,
        kernel_size=5,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

        self._segmenter = ColorSegmenter(
            hsv_ranges=DEFAULT_HSV_RANGES,
            kernel_size=kernel_size,
            open_iters=1,
            close_iters=1,
            min_area=min_area,
            max_area=max_area,
            min_circularity=0.20,
        )

    def detect(self, frame):
        """Detect markers in a BGR frame.

        Returns list of dicts: {color, center, bbox, area, offset_norm,
                                 angle_from_center_deg}
        """
        fh, fw = frame.shape[:2]
        frame_cx = fw // 2
        frame_cy = fh // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = []

        for color in self.MARKER_COLORS:
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

                x, y, w, h = cv2.boundingRect(cnt)
                if h == 0:
                    continue
                aspect = w / h
                if not (self.aspect_min < aspect < self.aspect_max):
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Normalised offset: negative = left/up, positive = right/down
                dx_norm = (cx - frame_cx) / frame_cx
                dy_norm = (cy - frame_cy) / frame_cy

                angle = np.degrees(
                    np.arctan2(cy - frame_cy, cx - frame_cx)
                )

                results.append({
                    "color": color,
                    "center": (cx, cy),
                    "bbox": (x, y, w, h),
                    "area": area,
                    "offset_norm": (round(dx_norm, 3), round(dy_norm, 3)),
                    "angle_from_center_deg": round(angle, 1),
                })

        return results

    def draw(self, frame, detections):
        """Draw detected markers with colour-coded circles and offset lines."""
        color_bgr = {
            "red": (0, 0, 220),
            "blue": (220, 60, 0),
            "green": (0, 200, 0),
            "yellow": (0, 220, 220),
            "orange": (0, 130, 255),
        }
        fh, fw = frame.shape[:2]
        out = frame.copy()
        cv2.line(out, (fw // 2, 0), (fw // 2, fh), (80, 80, 80), 1)
        cv2.line(out, (0, fh // 2), (fw, fh // 2), (80, 80, 80), 1)

        for det in detections:
            bgr = color_bgr.get(det["color"], (200, 200, 200))
            cx, cy = det["center"]
            x, y, w, h = det["bbox"]

            cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
            cv2.circle(out, (cx, cy), 6, bgr, -1)
            cv2.line(out, (fw // 2, fh // 2), (cx, cy), bgr, 1)

            dx, dy = det["offset_norm"]
            label = f"{det['color']} ({dx:+.2f}, {dy:+.2f})"
            cv2.putText(
                out, label, (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1
            )
        return out

    def build_roi_crops(self, frame, detections, padding=10):
        """Extract cropped sub-images around each detected marker."""
        fh, fw = frame.shape[:2]
        crops = []
        for det in detections:
            x, y, w, h = det["bbox"]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(fw, x + w + padding)
            y2 = min(fh, y + h + padding)
            crops.append({
                "color": det["color"],
                "crop": frame[y1:y2, x1:x2].copy(),
                "roi": (x1, y1, x2 - x1, y2 - y1),
            })
        return crops
