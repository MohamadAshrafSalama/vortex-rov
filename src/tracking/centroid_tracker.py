"""
Multi-object centroid tracker with Hungarian algorithm assignment.

Each detected bounding box is matched to an existing tracked object by
minimising the Euclidean centroid distance. Unmatched tracks accumulate
a disappeared counter and are pruned once max_disappeared is exceeded.
New detections that cannot be matched to any existing track are
registered as fresh objects.

Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm) for
optimal assignment instead of a greedy nearest-neighbour approach.
"""

import math
from collections import OrderedDict

import numpy as np


class CentroidTracker:
    """Track multiple objects by centroid proximity across frames."""

    def __init__(self, max_disappeared=20, max_distance=90):
        """
        Args:
            max_disappeared: frames an object may remain unseen before removal
            max_distance: pixel distance beyond which no match is made
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.next_id = 0
        self.objects = OrderedDict()      # id -> detection dict
        self.disappeared = OrderedDict()  # id -> frame count since last seen
        self.history = OrderedDict()      # id -> list of past centroids

    def _centroid(self, detection):
        return detection.get("center") or (
            detection["bbox"][0] + detection["bbox"][2] // 2,
            detection["bbox"][1] + detection["bbox"][3] // 2,
        )

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def register(self, detection):
        oid = self.next_id
        self.objects[oid] = detection
        self.disappeared[oid] = 0
        self.history[oid] = [self._centroid(detection)]
        self.next_id += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.history[oid]

    def update(self, detections):
        """Update tracker state with new frame's detections.

        Args:
            detections: list of detection dicts (must contain 'center' or 'bbox')

        Returns:
            OrderedDict of {object_id: detection_dict}
        """
        if not detections:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if not self.objects:
            for det in detections:
                self.register(det)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self._centroid(self.objects[oid]) for oid in obj_ids], dtype=float)
        det_centroids = np.array([self._centroid(d) for d in detections], dtype=float)

        # Build cost matrix
        cost = np.zeros((len(obj_ids), len(detections)), dtype=float)
        for i, oc in enumerate(obj_centroids):
            for j, dc in enumerate(det_centroids):
                cost[i, j] = self._dist(oc, dc)

        # Hungarian assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except ImportError:
            # Fallback: greedy nearest-neighbour
            row_ind, col_ind = self._greedy_assignment(cost)

        used_rows = set()
        used_cols = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            self.objects[oid] = detections[c]
            self.disappeared[oid] = 0
            self.history[oid].append(self._centroid(detections[c]))
            if len(self.history[oid]) > 60:
                self.history[oid].pop(0)
            used_rows.add(r)
            used_cols.add(c)

        for i, oid in enumerate(obj_ids):
            if i not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for j, det in enumerate(detections):
            if j not in used_cols:
                self.register(det)

        return self.objects

    @staticmethod
    def _greedy_assignment(cost):
        """Simple greedy matching as fallback when scipy is unavailable."""
        rows, cols = [], []
        used_r, used_c = set(), set()
        flat = np.argsort(cost.ravel())
        for idx in flat:
            r, c = divmod(int(idx), cost.shape[1])
            if r not in used_r and c not in used_c:
                rows.append(r)
                cols.append(c)
                used_r.add(r)
                used_c.add(c)
        return rows, cols

    def velocity(self, oid, window=5):
        """Estimate recent velocity (dx, dy) in pixels/frame for an object."""
        hist = self.history.get(oid, [])
        if len(hist) < 2:
            return (0.0, 0.0)
        recent = hist[-min(window, len(hist)):]
        dx = (recent[-1][0] - recent[0][0]) / (len(recent) - 1)
        dy = (recent[-1][1] - recent[0][1]) / (len(recent) - 1)
        return (round(dx, 2), round(dy, 2))

    def draw(self, frame, color=(0, 255, 0)):
        """Draw tracked object IDs and centroids on a frame."""
        import cv2
        out = frame.copy()
        for oid, det in self.objects.items():
            cx, cy = self._centroid(det)
            cv2.circle(out, (cx, cy), 5, color, -1)
            cv2.putText(
                out, f"ID:{oid}", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            # Draw trajectory
            hist = self.history.get(oid, [])
            for k in range(1, len(hist)):
                cv2.line(out, hist[k - 1], hist[k], (80, 80, 255), 1)
        return out
