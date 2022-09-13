"""
Multi-camera manager with thread-safe frame grabbing.

Supports V4L2 (Linux) and platform-agnostic backends via OpenCV.
Each camera runs a dedicated background thread to keep the buffer
drained and deliver the latest frame without pipeline stalls.
"""

import threading
import time
import cv2
import numpy as np


class CameraCapture:
    """Single camera interface with background grab thread."""

    def __init__(self, index=0, width=1280, height=720, fps=30, backend="auto"):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._backend = self._resolve_backend(backend)

        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._grab_count = 0
        self._drop_count = 0

    @staticmethod
    def _resolve_backend(name):
        mapping = {
            "v4l2": cv2.CAP_V4L2,
            "auto": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "gstreamer": cv2.CAP_GSTREAMER,
        }
        return mapping.get(name, cv2.CAP_ANY)

    def open(self):
        self._cap = cv2.VideoCapture(self.index, self._backend)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open camera at index {self.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Warm up: read a few frames so the sensor gains stabilize
        for _ in range(5):
            self._cap.read()

        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return self

    def _grab_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._grab_count += 1
            else:
                self._drop_count += 1
                time.sleep(0.005)

    def read(self):
        """Return the latest grabbed frame, or None if not yet available."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def read_blocking(self, timeout=2.0):
        """Block until a frame is available or timeout elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    @property
    def actual_width(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else self.width

    @property
    def actual_height(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else self.height

    @property
    def actual_fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else self.fps

    @property
    def stats(self):
        return {"grabbed": self._grab_count, "dropped": self._drop_count}

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()


class CameraManager:
    """Manage multiple cameras and provide a unified frame interface."""

    def __init__(self, configs):
        """
        Args:
            configs: list of dicts, each with keys:
                     index, width, height, fps, backend (all optional)
        """
        self._cameras = {}
        self._active = None
        for cfg in configs:
            idx = cfg.get("index", 0)
            cam = CameraCapture(
                index=idx,
                width=cfg.get("width", 1280),
                height=cfg.get("height", 720),
                fps=cfg.get("fps", 30),
                backend=cfg.get("backend", "auto"),
            )
            self._cameras[idx] = cam

    def open_all(self):
        opened = []
        for idx, cam in self._cameras.items():
            try:
                cam.open()
                opened.append(idx)
            except IOError as exc:
                print(f"[CameraManager] Warning: {exc}")
        if opened:
            self._active = opened[0]
        return opened

    def select(self, index):
        if index not in self._cameras:
            raise KeyError(f"Camera index {index} not registered")
        self._active = index

    def read(self, index=None):
        idx = index if index is not None else self._active
        if idx is None or idx not in self._cameras:
            return None
        return self._cameras[idx].read()

    def read_all(self):
        """Return dict of {index: frame} for all open cameras."""
        return {idx: cam.read() for idx, cam in self._cameras.items()}

    def release_all(self):
        for cam in self._cameras.values():
            cam.release()

    def __enter__(self):
        self.open_all()
        return self

    def __exit__(self, *args):
        self.release_all()

    @property
    def camera_indices(self):
        return list(self._cameras.keys())

    def stats(self):
        return {idx: cam.stats for idx, cam in self._cameras.items()}


def list_available_cameras(max_check=8):
    """Probe camera indices and return those that opened successfully."""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available
