"""
Spatial POI mapping: pixel coordinates -> real-world coordinates.

Detected markers are projected from image space to a floor-plane
coordinate system using a pinhole camera model. Sightings within a
configurable proximity threshold are merged to prevent duplicate POIs.
Confirmed POIs (sighted >= min_sightings times) are exported as JSON.
"""

import json
import math
import os
import time

import numpy as np


class SpatialPoint:
    """A single point of interest with sighting history."""

    def __init__(self, poi_id, x_m, y_m, color, frame_idx, timestamp):
        self.id = poi_id
        self.x_m = x_m
        self.y_m = y_m
        self.color = color
        self.sightings = 1
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        # Running mean of position
        self._sum_x = x_m
        self._sum_y = y_m

    def update(self, x_m, y_m, frame_idx, timestamp):
        self.sightings += 1
        self._sum_x += x_m
        self._sum_y += y_m
        self.x_m = self._sum_x / self.sightings
        self.y_m = self._sum_y / self.sightings
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp

    def distance_to(self, x_m, y_m):
        return math.hypot(self.x_m - x_m, self.y_m - y_m)

    def to_dict(self):
        return {
            "id": self.id,
            "x_m": round(self.x_m, 4),
            "y_m": round(self.y_m, 4),
            "color": self.color,
            "sightings": self.sightings,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
        }


class POIMapper:
    """Transform detections into a georeferenced POI map."""

    def __init__(
        self,
        focal_length_px=600.0,
        camera_altitude_m=0.8,
        frame_width=1280,
        frame_height=720,
        merge_radius_m=0.12,
        min_sightings=3,
    ):
        """
        Args:
            focal_length_px: camera focal length in pixels (from calibration)
            camera_altitude_m: approximate floor-to-camera distance
            frame_width, frame_height: image dimensions
            merge_radius_m: spatial threshold for merging nearby sightings
            min_sightings: confirmations required before POI is deemed confirmed
        """
        self.focal_length_px = focal_length_px
        self.altitude_m = camera_altitude_m
        self.fw = frame_width
        self.fh = frame_height
        self.merge_radius_m = merge_radius_m
        self.min_sightings = min_sightings

        self._pois = []
        self._next_id = 1
        self._frame_idx = 0

    def pixel_to_floor(self, cx_px, cy_px, rov_x_m=0.0, rov_y_m=0.0):
        """Project image pixel to approximate floor-plane coordinates.

        Uses a simple pinhole model: floor_x = (px - cx) * Z / f.
        Adds the ROV's absolute position if known.
        """
        cx_img = self.fw / 2.0
        cy_img = self.fh / 2.0

        floor_x = (cx_px - cx_img) * self.altitude_m / self.focal_length_px
        floor_y = (cy_px - cy_img) * self.altitude_m / self.focal_length_px

        return floor_x + rov_x_m, floor_y + rov_y_m

    def update(self, detections, rov_x_m=0.0, rov_y_m=0.0):
        """Process a list of marker detections from one frame.

        Args:
            detections: list of dicts with 'center' and 'color' keys
            rov_x_m, rov_y_m: ROV absolute position (optional)

        Returns:
            list of (SpatialPoint, is_new) tuples for this frame's updates
        """
        self._frame_idx += 1
        now = time.time()
        events = []

        for det in detections:
            cx, cy = det["center"]
            color = det.get("color", "unknown")
            fx, fy = self.pixel_to_floor(cx, cy, rov_x_m, rov_y_m)

            existing = self._find_nearby(fx, fy, color)
            if existing is not None:
                existing.update(fx, fy, self._frame_idx, now)
                events.append((existing, False))
            else:
                poi = SpatialPoint(
                    self._next_id, fx, fy, color, self._frame_idx, now
                )
                self._pois.append(poi)
                self._next_id += 1
                events.append((poi, True))

        return events

    def _find_nearby(self, x_m, y_m, color):
        """Find an existing POI of the same colour within merge radius."""
        for poi in self._pois:
            if poi.color != color:
                continue
            if poi.distance_to(x_m, y_m) < self.merge_radius_m:
                return poi
        return None

    @property
    def all_pois(self):
        return list(self._pois)

    @property
    def confirmed_pois(self):
        return [p for p in self._pois if p.sightings >= self.min_sightings]

    def save(self, path):
        """Export all POIs to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "total": len(self._pois),
            "confirmed": len(self.confirmed_pois),
            "pois": [p.to_dict() for p in self._pois],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def draw_map(self, width=480, height=480, scale_m=3.0):
        """Render a 2D overhead POI map as a numpy image.

        Args:
            width, height: canvas dimensions in pixels
            scale_m: total real-world extent shown in map
        """
        import cv2
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        color_bgr = {
            "red": (0, 0, 220),
            "blue": (220, 60, 0),
            "green": (0, 200, 0),
            "yellow": (0, 220, 220),
            "orange": (0, 130, 255),
            "unknown": (180, 180, 180),
        }

        # Draw grid lines every 0.5m
        step_px = int(width * 0.5 / scale_m)
        for i in range(0, width, step_px):
            cv2.line(canvas, (i, 0), (i, height), (30, 30, 30), 1)
            cv2.line(canvas, (0, i), (width, i), (30, 30, 30), 1)

        # Origin cross
        ox, oy = width // 2, height // 2
        cv2.line(canvas, (ox - 12, oy), (ox + 12, oy), (60, 60, 80), 2)
        cv2.line(canvas, (ox, oy - 12), (ox, oy + 12), (60, 60, 80), 2)

        for poi in self._pois:
            px = int(ox + poi.x_m * (width / scale_m))
            py = int(oy + poi.y_m * (height / scale_m))
            px = max(4, min(width - 4, px))
            py = max(4, min(height - 4, py))

            c = color_bgr.get(poi.color, (180, 180, 180))
            confirmed = poi.sightings >= self.min_sightings
            radius = 8 if confirmed else 4
            cv2.circle(canvas, (px, py), radius, c, -1)
            if confirmed:
                cv2.circle(canvas, (px, py), radius + 3, c, 1)

            lbl = f"#{poi.id}({poi.sightings})"
            cv2.putText(
                canvas, lbl, (px + 10, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1
            )

        cv2.putText(
            canvas, f"POIs: {len(self._pois)} | Confirmed: {len(self.confirmed_pois)}",
            (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
        )
        return canvas

    def reset(self):
        self._pois.clear()
        self._next_id = 1
        self._frame_idx = 0

