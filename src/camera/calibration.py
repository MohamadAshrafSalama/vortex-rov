"""
Underwater camera calibration using OpenCV checkerboard method.

Accounts for barrel distortion common in wide-angle lenses used on ROVs.
Supports saving and loading calibration data as YAML for reproducible runs.
The fisheye module path handles ultra-wide lenses with stronger radial distortion.
"""

import os
import cv2
import numpy as np
import yaml


class CameraCalibrator:
    """Checkerboard-based calibration with persistent YAML storage."""

    def __init__(self, board_rows=9, board_cols=6, square_size_m=0.025):
        """
        Args:
            board_rows: number of inner corners along rows
            board_cols: number of inner corners along columns
            square_size_m: physical size of one checkerboard square in meters
        """
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.square_size = square_size_m

        self._obj_pattern = self._build_object_points()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        self.reprojection_error = None
        self._map1 = None
        self._map2 = None

        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6
        )

    def _build_object_points(self):
        pts = np.zeros((self.board_rows * self.board_cols, 3), np.float32)
        pts[:, :2] = np.mgrid[0:self.board_cols, 0:self.board_rows].T.reshape(-1, 2)
        pts *= self.square_size
        return pts

    def detect_corners(self, frame):
        """Find checkerboard corners in a single frame.

        Returns (corners, gray) on success, (None, gray) on failure.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(
            gray, (self.board_cols, self.board_rows), flags
        )
        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self._criteria
            )
            return corners, gray
        return None, gray

    def calibrate(self, frames):
        """Run calibration on a list of BGR frames.

        Args:
            frames: list of numpy arrays (BGR)

        Returns:
            reprojection_error: mean reprojection error in pixels
        """
        obj_points = []
        img_points = []
        image_size = None

        for frame in frames:
            corners, gray = self.detect_corners(frame)
            if corners is None:
                continue
            obj_points.append(self._obj_pattern)
            img_points.append(corners)
            image_size = gray.shape[::-1]

        if len(obj_points) < 6:
            raise RuntimeError(
                f"Need at least 6 valid frames, got {len(obj_points)}"
            )

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        self.camera_matrix = K
        self.dist_coeffs = D
        self.image_size = image_size
        self.reprojection_error = self._compute_reprojection_error(
            obj_points, img_points, rvecs, tvecs
        )
        self._build_undistort_maps()
        return self.reprojection_error

    def _compute_reprojection_error(self, obj_pts, img_pts, rvecs, tvecs):
        total_error = 0.0
        count = 0
        for i, (obj, img) in enumerate(zip(obj_pts, img_pts)):
            projected, _ = cv2.projectPoints(
                obj, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs
            )
            err = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
            total_error += err
            count += 1
        return total_error / count if count else float("inf")

    def _build_undistort_maps(self, alpha=0.5):
        if self.camera_matrix is None or self.image_size is None:
            return
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, self.image_size, alpha
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            new_K, self.image_size, cv2.CV_16SC2
        )

    def undistort(self, frame):
        """Apply undistortion using precomputed remap tables."""
        if self._map1 is None or self._map2 is None:
            raise RuntimeError("Calibration not loaded. Run calibrate() or load().")
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    def draw_corners(self, frame, corners):
        """Draw detected checkerboard corners on frame for visual inspection."""
        out = frame.copy()
        cv2.drawChessboardCorners(
            out, (self.board_cols, self.board_rows), corners, True
        )
        return out

    def save(self, path):
        """Persist calibration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "board_rows": self.board_rows,
            "board_cols": self.board_cols,
            "square_size_m": float(self.square_size),
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reprojection_error": float(self.reprojection_error),
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        print(f"[Calibration] Saved to {path}. RMS error: {self.reprojection_error:.4f}px")

    def load(self, path):
        """Load calibration from YAML file and rebuild undistort maps."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.board_rows = data["board_rows"]
        self.board_cols = data["board_cols"]
        self.square_size = data["square_size_m"]
        self.image_size = tuple(data["image_size"])
        self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        self.reprojection_error = data.get("reprojection_error", None)
        self._obj_pattern = self._build_object_points()
        self._build_undistort_maps()
        return self

    @property
    def focal_length(self):
        if self.camera_matrix is None:
            return None
        return float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])

    @property
    def principal_point(self):
        if self.camera_matrix is None:
            return None
        return float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])

