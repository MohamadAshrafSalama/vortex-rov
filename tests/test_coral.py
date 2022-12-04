"""
Tests for coral health analysis module.
"""

import sys
import os
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.coral_health import (
    CoralHealthAnalyzer,
    _histogram_correlation,
    _ssim_score,
    _classify_cell_hsv,
    STATE_HEALTHY, STATE_BLEACHED, STATE_DEAD, STATE_BACKGROUND,
    CHANGE_LABELS,
)


def make_uniform_bgr(bgr_color, size=200):
    return np.full((size, size, 3), bgr_color, dtype=np.uint8)


def make_healthy_patch():
    """Create a patch in the healthy coral HSV range (green-ish)."""
    # HSV (55, 160, 120) -> green
    hsv = np.full((100, 100, 3), (55, 160, 120), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_bleached_patch():
    """Create a patch in the bleached coral HSV range (near-white)."""
    hsv = np.full((100, 100, 3), (0, 20, 230), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestCellClassification:
    def test_healthy_patch_classified_healthy(self):
        patch = make_healthy_patch()
        result = _classify_cell_hsv(patch, min_pixels=50)
        assert result == STATE_HEALTHY, f"Expected HEALTHY, got {result}"

    def test_bleached_patch_classified_bleached(self):
        patch = make_bleached_patch()
        result = _classify_cell_hsv(patch, min_pixels=50)
        assert result == STATE_BLEACHED, f"Expected BLEACHED, got {result}"

    def test_dark_background_classified_background(self):
        patch = make_uniform_bgr((10, 10, 10))
        result = _classify_cell_hsv(patch, min_pixels=100)
        assert result == STATE_BACKGROUND


class TestHistogramCorrelation:
    def test_identical_images_correlation_near_1(self):
        img = make_uniform_bgr((100, 150, 50))
        corr = _histogram_correlation(img, img)
        assert corr > 0.99

    def test_different_images_lower_correlation(self):
        img_a = make_uniform_bgr((200, 50, 20))
        img_b = make_uniform_bgr((20, 200, 50))
        corr = _histogram_correlation(img_a, img_b)
        assert corr < 0.90


class TestSSIM:
    def test_identical_images_ssim_near_1(self):
        img = make_uniform_bgr((120, 80, 200))
        score = _ssim_score(img, img)
        assert score > 0.98

    def test_different_images_lower_ssim(self):
        img_a = make_uniform_bgr((200, 10, 10))
        img_b = make_uniform_bgr((10, 200, 10))
        score = _ssim_score(img_a, img_b)
        assert score < 0.95


class TestCoralHealthAnalyzer:
    def setup_method(self):
        self.analyzer = CoralHealthAnalyzer(grid_size=2, target_size=(200, 200))

    def test_compare_returns_required_keys(self):
        before = make_uniform_bgr((80, 160, 50))
        after = make_uniform_bgr((80, 160, 50))
        result = self.analyzer.compare(before, after)
        for key in ("before_grid", "after_grid", "change_grid",
                    "cell_scores", "health_report", "summary", "mean_severity"):
            assert key in result, f"Missing key: {key}"

    def test_grid_dimensions(self):
        before = make_uniform_bgr((80, 160, 50))
        after = make_uniform_bgr((80, 160, 50))
        result = self.analyzer.compare(before, after)
        assert result["before_grid"].shape == (2, 2)
        assert result["after_grid"].shape == (2, 2)
        assert len(result["health_report"]) == 4

    def test_severity_non_negative(self):
        before = make_healthy_patch()
        after = make_bleached_patch()
        result = self.analyzer.compare(before, after)
        for entry in result["health_report"]:
            assert entry["severity"] >= 0.0

    def test_draw_result_same_spatial_shape(self):
        before = make_uniform_bgr((60, 140, 40))
        after = make_uniform_bgr((200, 200, 200))
        result = self.analyzer.compare(before, after)
        annotated = self.analyzer.draw_result(after, result)
        assert annotated.shape[0] == 200
        assert annotated.shape[1] == 200

    def test_change_labels_in_summary(self):
        before = make_uniform_bgr((60, 140, 40))
        after = make_uniform_bgr((60, 140, 40))
        result = self.analyzer.compare(before, after)
        for label in result["summary"]:
            assert label in CHANGE_LABELS.values()
