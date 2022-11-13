"""
CSV telemetry logger with timestamps, sensor values, and task events.

Thread-safe — multiple pipeline threads can call log() concurrently.
The logger buffers entries and flushes periodically to avoid blocking
the camera loop on disk writes.
"""

import csv
import os
import threading
import time


FIELDNAMES = [
    "timestamp",
    "elapsed_s",
    "frame_idx",
    "depth_m",
    "heading_deg",
    "temperature_c",
    "pitch_deg",
    "roll_deg",
    "battery_pct",
    "boxes_detected",
    "markers_detected",
    "yolo_detected",
    "transect_offset",
    "transect_steering",
    "task_event",
    "notes",
]


class TelemetryLogger:
    """Write telemetry rows to a CSV file with periodic flushing."""

    def __init__(self, path="results/telemetry.csv", flush_interval=5.0):
        """
        Args:
            path: output CSV file path
            flush_interval: seconds between file flushes
        """
        self.path = path
        self.flush_interval = flush_interval

        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._start_time = None
        self._last_flush = 0.0
        self._frame_idx = 0
        self._open()

    def _open(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES, extrasaction="ignore")
        self._writer.writeheader()
        self._start_time = time.time()
        self._last_flush = self._start_time

    def log(
        self,
        depth_m=0.0,
        heading_deg=0.0,
        temperature_c=20.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        battery_pct=100.0,
        boxes_detected=0,
        markers_detected=0,
        yolo_detected=0,
        transect_offset=0.0,
        transect_steering=0.0,
        task_event="",
        notes="",
    ):
        """Write one telemetry row. Non-blocking under normal conditions."""
        now = time.time()
        row = {
            "timestamp": round(now, 3),
            "elapsed_s": round(now - self._start_time, 3),
            "frame_idx": self._frame_idx,
            "depth_m": round(depth_m, 3),
            "heading_deg": round(heading_deg, 2),
            "temperature_c": round(temperature_c, 2),
            "pitch_deg": round(pitch_deg, 2),
            "roll_deg": round(roll_deg, 2),
            "battery_pct": round(battery_pct, 1),
            "boxes_detected": boxes_detected,
            "markers_detected": markers_detected,
            "yolo_detected": yolo_detected,
            "transect_offset": round(transect_offset, 4),
            "transect_steering": round(transect_steering, 4),
            "task_event": task_event,
            "notes": notes,
        }
        with self._lock:
            self._writer.writerow(row)
            self._frame_idx += 1
            if now - self._last_flush >= self.flush_interval:
                self._file.flush()
                self._last_flush = now

    def log_event(self, event_name, notes=""):
        """Convenience wrapper for logging a named task event."""
        self.log(task_event=event_name, notes=notes)

    def flush(self):
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self):
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None
                self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def row_count(self):
        return self._frame_idx
