"""
Kalman filter tracker for smooth trajectory estimation.

Each tracked object maintains an independent cv2.KalmanFilter with a
constant-velocity motion model. The state vector is [x, y, vx, vy] and
the measurement is [x, y] (centroid position).

KalmanMultiTracker manages a pool of KalmanTracker instances and handles
object birth and death using the same centroid-distance matching logic.
"""

import math
from collections import OrderedDict

import cv2
import numpy as np


class KalmanTracker:
    """Single-object Kalman filter wrapper (constant velocity model)."""

    def __init__(self, initial_center, process_noise=0.03, measurement_noise=1.0):
        """
        Args:
            initial_center: (x, y) tuple for initialisation
            process_noise: Q scaling — higher = trust motion model less
            measurement_noise: R scaling — higher = trust measurements less
        """
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition: x' = x + vx*dt,  y' = y + vy*dt
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # We observe position only (x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        cx, cy = initial_center
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

        self.predicted = (cx, cy)
        self.corrected = (cx, cy)
        self.age = 0
        self.last_update = 0

    def predict(self):
        """Run Kalman predict step. Returns predicted (x, y)."""
        state = self.kf.predict()
        self.predicted = (int(state[0, 0]), int(state[1, 0]))
        self.age += 1
        return self.predicted

    def update(self, measurement):
        """Correct filter with an observed (x, y) centroid."""
        meas = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        state = self.kf.correct(meas)
        self.corrected = (int(state[0, 0]), int(state[1, 0]))
        self.last_update = self.age
        return self.corrected

    @property
    def velocity(self):
        """Current velocity estimate (vx, vy) in pixels/frame."""
        vx = float(self.kf.statePost[2, 0])
        vy = float(self.kf.statePost[3, 0])
        return (round(vx, 2), round(vy, 2))

    @property
    def position(self):
        return self.corrected


class KalmanMultiTracker:
    """Manage multiple Kalman filter tracks with centroid-distance assignment."""

    def __init__(
        self,
        max_disappeared=25,
        max_distance=100,
        process_noise=0.03,
        measurement_noise=1.0,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.next_id = 0
        self.trackers = OrderedDict()    # id -> KalmanTracker
        self.disappeared = OrderedDict()
        self.meta = OrderedDict()        # id -> last detection dict

    def _centroid(self, detection):
        c = detection.get("center")
        if c:
            return c
        x, y, w, h = detection["bbox"]
        return (x + w // 2, y + h // 2)

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def register(self, detection):
        oid = self.next_id
        cx, cy = self._centroid(detection)
        self.trackers[oid] = KalmanTracker(
            (cx, cy), self.process_noise, self.measurement_noise
        )
        self.disappeared[oid] = 0
        self.meta[oid] = detection
        self.next_id += 1
        return oid

    def deregister(self, oid):
        del self.trackers[oid]
        del self.disappeared[oid]
        del self.meta[oid]

    def update(self, detections):
        """Predict all trackers, then update matched ones with measurements.

        Returns:
            dict of {object_id: {"position": (x,y), "velocity": (vx,vy),
                                  "meta": detection_dict}}
        """
        # Predict step for all trackers
        predicted = {oid: t.predict() for oid, t in self.trackers.items()}

        if not detections:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self._build_output()

        if not self.trackers:
            for det in detections:
                self.register(det)
            return self._build_output()

        obj_ids = list(self.trackers.keys())
        used_objs, used_dets = set(), set()

        # Build cost matrix using predicted positions
        cost_pairs = []
        for i, oid in enumerate(obj_ids):
            pred_pos = predicted[oid]
            for j, det in enumerate(detections):
                d = self._dist(pred_pos, self._centroid(det))
                if d < self.max_distance:
                    cost_pairs.append((d, i, j, oid))
        cost_pairs.sort()

        for d, i, j, oid in cost_pairs:
            if oid in used_objs or j in used_dets:
                continue
            self.trackers[oid].update(self._centroid(detections[j]))
            self.disappeared[oid] = 0
            self.meta[oid] = detections[j]
            used_objs.add(oid)
            used_dets.add(j)

        for oid in obj_ids:
            if oid not in used_objs:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for j, det in enumerate(detections):
            if j not in used_dets:
                self.register(det)

        return self._build_output()

    def _build_output(self):
        out = {}
        for oid, tracker in self.trackers.items():
            out[oid] = {
                "position": tracker.position,
                "velocity": tracker.velocity,
                "meta": self.meta.get(oid, {}),
                "age": tracker.age,
            }
        return out

    def draw(self, frame, tracked):
        """Draw smoothed positions and velocity vectors."""
        out = frame.copy()
        for oid, info in tracked.items():
            px, py = info["position"]
            vx, vy = info["velocity"]
            cv2.circle(out, (px, py), 6, (0, 255, 100), -1)
            # Velocity arrow
            ex = int(px + vx * 8)
            ey = int(py + vy * 8)
            cv2.arrowedLine(out, (px, py), (ex, ey), (255, 200, 0), 2, tipLength=0.3)
            cv2.putText(
                out, f"K{oid}", (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1
            )
        return out
