"""
Tests for centroid tracker and Kalman tracker modules.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracking.centroid_tracker import CentroidTracker
from src.tracking.kalman_tracker import KalmanTracker, KalmanMultiTracker


def make_detection(cx, cy, color="blue", area=2000):
    return {
        "center": (cx, cy),
        "color": color,
        "bbox": (cx - 25, cy - 25, 50, 50),
        "area": area,
    }


class TestCentroidTracker:
    def test_register_on_first_frame(self):
        tracker = CentroidTracker()
        dets = [make_detection(100, 100), make_detection(300, 300)]
        objects = tracker.update(dets)
        assert len(objects) == 2

    def test_ids_are_stable_across_frames(self):
        tracker = CentroidTracker()
        dets = [make_detection(100, 100)]
        tracker.update(dets)
        ids_1 = set(tracker.update([make_detection(105, 102)]).keys())
        ids_2 = set(tracker.update([make_detection(108, 104)]).keys())
        assert ids_1 == ids_2, "IDs should remain stable for slowly moving objects"

    def test_deregister_after_max_disappeared(self):
        tracker = CentroidTracker(max_disappeared=3)
        dets = [make_detection(100, 100)]
        tracker.update(dets)
        # Disappear for 4 frames
        for _ in range(4):
            tracker.update([])
        assert len(tracker.objects) == 0

    def test_new_object_gets_new_id(self):
        tracker = CentroidTracker(max_disappeared=3)
        tracker.update([make_detection(100, 100)])
        id_1 = list(tracker.objects.keys())[0]
        for _ in range(4):
            tracker.update([])
        tracker.update([make_detection(100, 100)])
        id_2 = list(tracker.objects.keys())[0]
        assert id_2 > id_1

    def test_velocity_computation(self):
        tracker = CentroidTracker()
        tracker.update([make_detection(0, 0)])
        oid = list(tracker.objects.keys())[0]
        tracker.update([make_detection(10, 0)])
        tracker.update([make_detection(20, 0)])
        vx, vy = tracker.velocity(oid, window=3)
        assert vx > 0, "Expected positive x velocity"

    def test_multiple_objects_no_id_collision(self):
        tracker = CentroidTracker()
        dets = [make_detection(i * 100, 100) for i in range(5)]
        objects = tracker.update(dets)
        ids = list(objects.keys())
        assert len(ids) == len(set(ids)), "All IDs must be unique"

    def test_empty_frame_does_not_crash(self):
        tracker = CentroidTracker()
        result = tracker.update([])
        assert isinstance(result, dict)


class TestKalmanTracker:
    def test_predict_returns_position(self):
        kt = KalmanTracker(initial_center=(200, 150))
        pos = kt.predict()
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_update_refines_position(self):
        kt = KalmanTracker(initial_center=(0, 0))
        for i in range(10):
            kt.predict()
            kt.update((i * 5, 0))
        # After updates the corrected position should be near the last measurement
        cx, _ = kt.position
        assert cx > 0

    def test_velocity_property(self):
        kt = KalmanTracker(initial_center=(0, 0))
        for i in range(5):
            kt.predict()
            kt.update((i * 10, 0))
        vx, vy = kt.velocity
        assert isinstance(vx, float)
        assert isinstance(vy, float)


class TestKalmanMultiTracker:
    def test_register_and_track(self):
        mt = KalmanMultiTracker()
        dets = [make_detection(100, 100), make_detection(400, 300)]
        result = mt.update(dets)
        assert len(result) == 2

    def test_tracks_persist(self):
        mt = KalmanMultiTracker(max_disappeared=5)
        mt.update([make_detection(100, 100)])
        ids_1 = set(mt.update([make_detection(105, 100)]).keys())
        ids_2 = set(mt.update([make_detection(110, 100)]).keys())
        assert ids_1 == ids_2

    def test_output_has_required_keys(self):
        mt = KalmanMultiTracker()
        mt.update([make_detection(50, 50)])
        result = mt.update([make_detection(55, 52)])
        for oid, info in result.items():
            assert "position" in info
            assert "velocity" in info
            assert "age" in info
