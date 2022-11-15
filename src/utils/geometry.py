"""
Geometric utility functions for ROV navigation and image analysis.

All functions are pure — no side effects and no OpenCV display calls.
"""

import math

import cv2
import numpy as np


def euclidean_distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_between(p1, p2, degrees=True):
    """Angle from p1 to p2 measured from positive x-axis."""
    a = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return math.degrees(a) if degrees else a


def normalize_angle(angle_deg):
    """Wrap angle to [-180, 180) degrees."""
    while angle_deg >= 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def rotate_point(px, py, cx, cy, angle_deg):
    """Rotate point (px, py) around centre (cx, cy) by angle_deg."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = px - cx
    dy = py - cy
    rx = cx + cos_a * dx - sin_a * dy
    ry = cy + sin_a * dx + cos_a * dy
    return rx, ry


def compute_homography_points(src_points, dst_points, method=cv2.RANSAC, threshold=5.0):
    """Compute 3x3 homography from corresponding point lists.

    Args:
        src_points: list of (x, y) in source image
        dst_points: list of (x, y) in destination image

    Returns:
        H (3x3 float64) or None if computation failed
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        return None
    src = np.float32(src_points).reshape(-1, 1, 2)
    dst = np.float32(dst_points).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method, threshold)
    return H


def perspective_transform_point(point, H):
    """Apply a 3x3 homography to a single 2D point.

    Returns:
        (x', y') transformed coordinates
    """
    px, py = point
    p = np.array([[[px, py]]], dtype=np.float32)
    tp = cv2.perspectiveTransform(p, H)
    return float(tp[0, 0, 0]), float(tp[0, 0, 1])


def rect_iou(box_a, box_b):
    """Compute Intersection over Union for two (x, y, w, h) rectangles."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def point_in_rect(point, rect):
    """Check whether (x, y) lies inside (rx, ry, rw, rh)."""
    px, py = point
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def bbox_center(bbox):
    """Return (cx, cy) from (x, y, w, h)."""
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def contour_center(contour):
    """Return centroid of a contour, or None if area is zero."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def line_angle_deg(x1, y1, x2, y2):
    """Angle of a line segment in degrees, measured from positive x-axis."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Perpendicular distance from point (px, py) to line through (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length
