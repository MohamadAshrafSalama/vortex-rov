"""
Live operator dashboard using tkinter.

Displays a camera feed with detection overlays alongside a telemetry
panel showing depth, heading, temperature, detection counts, and task
status. Updates run on the main tkinter thread via after() polling to
stay thread-safe with the background camera capture threads.

Usage:
    dash = Dashboard(width=1280, height=720)
    dash.run()        # blocking; call update() from a callback thread
"""

import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class Dashboard:
    """Tkinter-based live telemetry and video dashboard."""

    PANEL_WIDTH = 280
    REFRESH_MS = 33    # ~30 Hz GUI refresh

    TASK_COLORS = {
        "idle":    "#555555",
        "running": "#22aa55",
        "done":    "#3388ff",
        "error":   "#cc3333",
    }

    def __init__(self, width=1280, height=720, title="Vortex ROV 2024 — Dashboard"):
        self.video_w = width
        self.video_h = height
        self._title = title

        # Shared state — written by background, read by GUI
        self._lock = threading.Lock()
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._telemetry = {
            "depth_m": 0.0,
            "heading_deg": 0.0,
            "temperature_c": 20.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "battery_pct": 100.0,
        }
        self._detections = {
            "boxes": 0,
            "markers": 0,
            "yolo": 0,
        }
        self._tasks = {
            "Transect":  "idle",
            "POI Map":   "idle",
            "Coral":     "idle",
            "Photomosaic": "idle",
        }
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
        self._running = False

        self._root = None
        self._video_label = None
        self._telem_vars = {}
        self._task_labels = {}

    # --- Public update API (thread-safe) ---

    def update_frame(self, frame):
        """Push a new BGR frame to the display."""
        with self._lock:
            self._frame = frame.copy()
            self._frame_count += 1
            now = time.monotonic()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def update_telemetry(self, **kwargs):
        """Update any telemetry field by name."""
        with self._lock:
            self._telemetry.update(kwargs)

    def update_detections(self, **kwargs):
        """Update detection counts."""
        with self._lock:
            self._detections.update(kwargs)

    def set_task_status(self, task_name, status):
        """Set task status: 'idle' | 'running' | 'done' | 'error'."""
        with self._lock:
            self._tasks[task_name] = status

    # --- GUI construction ---

    def _build_gui(self):
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.configure(bg="#1a1a1a")
        self._root.resizable(False, False)

        total_w = self.video_w + self.PANEL_WIDTH
        self._root.geometry(f"{total_w}x{self.video_h}")

        # Video canvas
        self._video_label = tk.Label(self._root, bg="#000000")
        self._video_label.place(x=0, y=0, width=self.video_w, height=self.video_h)

        # Side panel
        panel = tk.Frame(
            self._root, bg="#141414", width=self.PANEL_WIDTH, height=self.video_h
        )
        panel.place(x=self.video_w, y=0)

        title_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
        label_font = tkfont.Font(family="Courier", size=10)
        val_font = tkfont.Font(family="Courier", size=10, weight="bold")

        y = 12
        tk.Label(
            panel, text="VORTEX ROV 2024", bg="#141414", fg="#00cc66",
            font=title_font
        ).place(x=10, y=y)
        y += 24

        # FPS
        self._fps_var = tk.StringVar(value="FPS:  --")
        tk.Label(panel, textvariable=self._fps_var, bg="#141414",
                 fg="#888888", font=label_font).place(x=10, y=y)
        y += 22

        tk.Frame(panel, bg="#333333", height=1, width=self.PANEL_WIDTH - 20).place(x=10, y=y)
        y += 10

        # Sensor readouts
        tk.Label(panel, text="SENSORS", bg="#141414", fg="#aaaaaa",
                 font=title_font).place(x=10, y=y)
        y += 22

        sensor_fields = [
            ("depth_m", "Depth", "m"),
            ("heading_deg", "Heading", "deg"),
            ("temperature_c", "Water Temp", "C"),
            ("pitch_deg", "Pitch", "deg"),
            ("roll_deg", "Roll", "deg"),
            ("battery_pct", "Battery", "%"),
        ]

        for key, name, unit in sensor_fields:
            tk.Label(panel, text=f"{name}:", bg="#141414", fg="#888888",
                     font=label_font).place(x=10, y=y)
            var = tk.StringVar(value="--")
            self._telem_vars[key] = (var, unit)
            tk.Label(panel, textvariable=var, bg="#141414", fg="#00ff88",
                     font=val_font).place(x=145, y=y)
            y += 22

        y += 8
        tk.Frame(panel, bg="#333333", height=1, width=self.PANEL_WIDTH - 20).place(x=10, y=y)
        y += 10

        # Detection counts
        tk.Label(panel, text="DETECTIONS", bg="#141414", fg="#aaaaaa",
                 font=title_font).place(x=10, y=y)
        y += 22

        det_fields = [("boxes", "Boxes"), ("markers", "Markers"), ("yolo", "YOLO")]
        self._det_vars = {}
        for key, name in det_fields:
            tk.Label(panel, text=f"{name}:", bg="#141414", fg="#888888",
                     font=label_font).place(x=10, y=y)
            var = tk.StringVar(value="0")
            self._det_vars[key] = var
            tk.Label(panel, textvariable=var, bg="#141414", fg="#ffcc00",
                     font=val_font).place(x=145, y=y)
            y += 22

        y += 8
        tk.Frame(panel, bg="#333333", height=1, width=self.PANEL_WIDTH - 20).place(x=10, y=y)
        y += 10

        # Task status
        tk.Label(panel, text="TASKS", bg="#141414", fg="#aaaaaa",
                 font=title_font).place(x=10, y=y)
        y += 22

        for task_name in self._tasks:
            tk.Label(panel, text=f"{task_name}:", bg="#141414", fg="#888888",
                     font=label_font).place(x=10, y=y)
            lbl = tk.Label(panel, text="IDLE", bg="#141414", fg="#555555",
                           font=val_font)
            lbl.place(x=145, y=y)
            self._task_labels[task_name] = lbl
            y += 22

    def _refresh(self):
        """Called every REFRESH_MS ms on the GUI thread."""
        with self._lock:
            frame = self._frame.copy()
            telem = dict(self._telemetry)
            detections = dict(self._detections)
            tasks = dict(self._tasks)
            fps = self._fps

        # Update video
        if _PIL_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)
            self._video_label.config(image=photo)
            self._video_label.image = photo
        else:
            # Fallback: encode to PNG bytes via tkinter BitmapImage isn't great;
            # skip video if PIL unavailable
            pass

        # Update sensor labels
        self._fps_var.set(f"FPS:  {fps:.1f}")
        for key, (var, unit) in self._telem_vars.items():
            val = telem.get(key, 0.0)
            var.set(f"{val:.1f} {unit}")

        for key, var in self._det_vars.items():
            var.set(str(detections.get(key, 0)))

        for task_name, lbl in self._task_labels.items():
            status = tasks.get(task_name, "idle")
            color = self.TASK_COLORS.get(status, "#555555")
            lbl.config(text=status.upper(), fg=color)

        if self._running:
            self._root.after(self.REFRESH_MS, self._refresh)

    def run(self):
        """Start the dashboard (blocking — must run on main thread)."""
        self._build_gui()
        self._running = True
        self._root.after(self.REFRESH_MS, self._refresh)
        self._root.protocol("WM_DELETE_WINDOW", self.stop)
        self._root.mainloop()

    def stop(self):
        self._running = False
        if self._root:
            self._root.destroy()
