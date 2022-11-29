"""
Tests for color segmentation and box detection modules.
"""

import sys
import os
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.color_segmentation import ColorSegmenter, _circularity
from src.detection.box_detector import BoxDetector
from src.detection.marker_detector import MarkerDetector


def make_color_frame(bgr_color, width=640, height=480):
    """Create a uniform-colour frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr_color
    return frame


def make_box_frame(bgr_color, x, y, w, h, frame_w=640, frame_h=480):
    """Create a frame with a single solid colour rectangle."""
    frame = np.full((frame_h, frame_w, 3), (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, -1)
    return frame


class TestCircularity:
    def test_circle_high_circularity(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1
        c = _circularity(contours[0])
        assert c > 0.80, f"Expected high circularity for circle, got {c:.3f}"

    def test_thin_line_low_circularity(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (10, 95), (190, 105), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1
        c = _circularity(contours[0])
        assert c < 0.20, f"Expected low circularity for thin rect, got {c:.3f}"


class TestColorSegmenter:
    def setup_method(self):
        self.segmenter = ColorSegmenter(min_area=300, min_circularity=0.0)

    def test_build_mask_known_color(self):
        """A frame filled with a known HSV colour should produce a non-empty mask."""
        # Pure blue in BGR
        frame = make_color_frame((200, 0, 0))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.segmenter.build_mask(hsv, "blue")
        nonzero = cv2.countNonZero(mask)
        assert nonzero > 1000, f"Expected mask to cover most of the blue frame, got {nonzero}"

    def test_build_mask_unknown_color_raises(self):
        frame = make_color_frame((0, 255, 0))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with pytest.raises(KeyError):
            self.segmenter.build_mask(hsv, "purple")

    def test_detect_returns_list(self):
        frame = make_color_frame((30, 30, 30))  # dark grey, no target colours
        dets = self.segmenter.detect(frame, colors=["red", "blue"])
        assert isinstance(dets, list)

    def test_detect_blue_box(self):
        """A large blue rectangle should be detected."""
        frame = make_box_frame((200, 0, 0), x=100, y=100, w=200, h=150)
        dets = self.segmenter.detect(frame, colors=["blue"])
        assert len(dets) >= 1
        assert dets[0]["color"] == "blue"

    def test_draw_returns_same_shape(self):
        frame = make_color_frame((30, 30, 30))
        dets = self.segmenter.detect(frame)
        drawn = self.segmenter.draw_detections(frame, dets)
        assert drawn.shape == frame.shape


class TestBoxDetector:
    def setup_method(self):
        self.detector = BoxDetector(min_area=500)

    def test_detect_returns_list(self):
        frame = make_color_frame((20, 20, 20))
        result = self.detector.detect(frame)
        assert isinstance(result, list)

    def test_distance_rank_assigned(self):
        frame = make_box_frame((0, 0, 200), x=50, y=50, w=300, h=200)
        dets = self.detector.detect(frame)
        for d in dets:
            assert "distance_rank" in d

    def test_primary_target_returns_none_when_empty(self):
        assert self.detector.primary_target([]) is None

    def test_primary_target_preferred_color(self):
        fake_dets = [
            {"color": "blue", "area": 5000, "bbox": (10, 10, 100, 100),
             "center": (60, 60), "solidity": 0.9,
             "approx_corners": None, "distance_rank": 1},
            {"color": "red", "area": 8000, "bbox": (200, 200, 120, 120),
             "center": (260, 260), "solidity": 0.9,
             "approx_corners": None, "distance_rank": 0},
        ]
        target = self.detector.primary_target(fake_dets, preferred_color="blue")
        assert target["color"] == "blue"


class TestMarkerDetector:
    def setup_method(self):
        self.detector = MarkerDetector(min_area=100)

    def test_detect_returns_list(self):
        frame = make_color_frame((20, 20, 20))
        dets = self.detector.detect(frame)
        assert isinstance(dets, list)

    def test_offset_norm_range(self):
        """Detected offset norms should be in [-1, 1]."""
        frame = make_box_frame((0, 0, 200), x=50, y=50, w=80, h=80)
        dets = self.detector.detect(frame)
        for d in dets:
            dx, dy = d["offset_norm"]
            assert -1.0 <= dx <= 1.0
            assert -1.0 <= dy <= 1.0

    def test_build_roi_crops_count(self):
        fake_dets = [
            {"color": "red", "center": (100, 100), "bbox": (80, 80, 40, 40),
             "area": 1600, "offset_norm": (0.1, 0.1), "angle_from_center_deg": 10.0}
        ]
        frame = make_color_frame((30, 30, 30))
        crops = self.detector.build_roi_crops(frame, fake_dets)
        assert len(crops) == 1
        assert "crop" in crops[0]
