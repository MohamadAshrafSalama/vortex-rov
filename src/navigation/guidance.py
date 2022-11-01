"""
Directional guidance engine with visual arrow overlays.

Computes the pixel offset between a target detection and the frame
centre, maps it to a cardinal/compound direction string, draws an
overlay arrow, and estimates monocular distance using the known
apparent width of a reference object at a known distance.
"""

import math

import cv2
import numpy as np


class GuidanceEngine:
    """Compute ROV navigation commands from detection positions."""

    ARROW_COLOR = (0, 240, 120)
    CENTER_COLOR = (80, 80, 255)

    def __init__(
        self,
        frame_width=1280,
        frame_height=720,
        centering_threshold_px=35,
        reference_width_px=110,
        reference_distance_m=1.0,
        arrow_scale=60,
    ):
        self.fw = frame_width
        self.fh = frame_height
        self.threshold = centering_threshold_px
        self.ref_width_px = reference_width_px
        self.ref_dist_m = reference_distance_m
        self.arrow_scale = arrow_scale

        self._cx = frame_width // 2
        self._cy = frame_height // 2

    def compute_offset(self, detection):
        """Return signed pixel offset (dx, dy) from frame centre."""
        cx, cy = detection.get("center") or (
            detection["bbox"][0] + detection["bbox"][2] // 2,
            detection["bbox"][1] + detection["bbox"][3] // 2,
        )
        return cx - self._cx, cy - self._cy

    def direction_string(self, dx, dy):
        """Map pixel offset to a human-readable direction label."""
        horiz = ""
        vert = ""
        if abs(dx) > self.threshold:
            horiz = "RIGHT" if dx > 0 else "LEFT"
        if abs(dy) > self.threshold:
            vert = "DOWN" if dy > 0 else "UP"
        if not horiz and not vert:
            return "CENTERED"
        parts = [p for p in [vert, horiz] if p]
        return " + ".join(parts)

    def estimate_distance(self, bbox_width_px):
        """Estimate depth using monocular apparent-size method.

        Assumes linear inverse relationship: d = (W_ref * D_ref) / W_obs.
        Returns distance in metres or inf if width is zero.
        """
        if bbox_width_px <= 0:
            return float("inf")
        return round((self.ref_width_px * self.ref_dist_m) / bbox_width_px, 2)

    def compute(self, detection):
        """Full guidance output for a single detection.

        Returns:
            dict with direction, offset, distance_m, normalized_offset,
                  angle_deg, bbox, color/class_name
        """
        dx, dy = self.compute_offset(detection)
        direction = self.direction_string(dx, dy)
        bbox = detection.get("bbox", (0, 0, 0, 0))
        box_w = bbox[2]
        distance_m = self.estimate_distance(box_w)

        norm_x = dx / (self._cx + 1e-6)
        norm_y = dy / (self._cy + 1e-6)
        angle_deg = round(math.degrees(math.atan2(dy, dx)), 1)

        return {
            "direction": direction,
            "offset_px": (dx, dy),
            "offset_norm": (round(norm_x, 3), round(norm_y, 3)),
            "angle_deg": angle_deg,
            "distance_m": distance_m,
            "centered": direction == "CENTERED",
            "bbox": bbox,
            "label": detection.get("color") or detection.get("class_name", ""),
        }

    def draw_guidance(self, frame, guidance):
        """Draw directional arrow, centring crosshair, and distance label."""
        out = frame.copy()
        cx, cy = self._cx, self._cy

        # Centre crosshair
        cv2.line(out, (cx - 20, cy), (cx + 20, cy), self.CENTER_COLOR, 2)
        cv2.line(out, (cx, cy - 20), (cx, cy + 20), self.CENTER_COLOR, 2)
        cv2.circle(out, (cx, cy), 30, self.CENTER_COLOR, 1)

        dx, dy = guidance["offset_px"]
        direction = guidance["direction"]

        if direction != "CENTERED":
            # Normalise arrow to arrow_scale pixels
            length = math.hypot(dx, dy) + 1e-6
            ex = int(cx + (dx / length) * self.arrow_scale)
            ey = int(cy + (dy / length) * self.arrow_scale)
            cv2.arrowedLine(
                out, (cx, cy), (ex, ey), self.ARROW_COLOR, 3, tipLength=0.25
            )

        # HUD text
        lines = [
            f"DIR:  {direction}",
            f"DIST: {guidance['distance_m']:.2f} m",
            f"OFF:  ({dx:+d}, {dy:+d}) px",
            f"ANG:  {guidance['angle_deg']:+.1f} deg",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                out, text, (10, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.ARROW_COLOR, 2
            )

        label = guidance.get("label", "")
        if label:
            x, y, w, h = guidance["bbox"]
            cv2.putText(
                out, label.upper(), (x, max(y - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 2
            )

        return out

    def draw_multi(self, frame, guidances):
        """Draw guidance for multiple detections — highlights primary target."""
        out = frame.copy()
        for i, g in enumerate(guidances):
            color = self.ARROW_COLOR if i == 0 else (150, 150, 150)
            x, y, w, h = g["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                out, f"#{i + 1} {g['label']} {g['distance_m']:.1f}m",
                (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )
        if guidances:
            out = self.draw_guidance(out, guidances[0])
        return out

    def update_frame_size(self, width, height):
        self.fw = width
        self.fh = height
        self._cx = width // 2
        self._cy = height // 2
