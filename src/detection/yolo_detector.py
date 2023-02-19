"""
YOLOv8 inference wrapper for fine-tuned underwater object detection.

Loads a custom-trained YOLOv8 model (Ultralytics) and runs inference
on frames with configurable confidence, IoU thresholds, and class
filtering. Falls back gracefully if the model file is not found.
"""

import os
import cv2
import numpy as np


class YoloDetector:
    """Wrap Ultralytics YOLOv8 for ROV competition inference."""

    LABEL_COLORS = [
        (0, 200, 80),
        (0, 100, 255),
        (255, 60, 0),
        (0, 220, 220),
        (200, 0, 200),
        (255, 200, 0),
        (0, 140, 255),
        (180, 255, 0),
    ]

    def __init__(
        self,
        model_path="models/rov_yolov8.pt",
        confidence=0.45,
        iou_threshold=0.45,
        input_size=640,
        classes=None,
        device="cpu",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.classes = classes
        self.device = device
        self._model = None
        self._class_names = {}

    def load(self):
        """Load the YOLO model. Raises FileNotFoundError if weights missing."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {self.model_path}. "
                "Train the model or download pre-trained weights."
            )
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            self._class_names = self._model.names
            print(
                f"[YoloDetector] Loaded model with "
                f"{len(self._class_names)} classes from {self.model_path}"
            )
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. Run: pip install ultralytics"
            )
        return self

    @property
    def is_loaded(self):
        return self._model is not None

    def detect(self, frame):
        """Run inference on a BGR frame.

        Returns:
            list of dicts: {class_id, class_name, confidence, bbox (x,y,w,h), center}
        """
        if self._model is None:
            return []

        results = self._model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2
                detections.append({
                    "class_id": cls_id,
                    "class_name": self._class_names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 3),
                    "bbox": (x1, y1, w, h),
                    "xyxy": (x1, y1, x2, y2),
                    "center": (cx, cy),
                })

        return detections

    def draw(self, frame, detections):
        """Draw YOLO detections with class-coloured bounding boxes."""
        out = frame.copy()
        for det in detections:
            cls_id = det["class_id"]
            color = self.LABEL_COLORS[cls_id % len(self.LABEL_COLORS)]
            x1, y1, x2, y2 = det["xyxy"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                out, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return out

    def filter_by_class(self, detections, class_names):
        """Return only detections whose class_name is in class_names list."""
        names = set(class_names)
        return [d for d in detections if d["class_name"] in names]

    def top_confidence(self, detections, n=1):
        """Return the n highest-confidence detections."""
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        return sorted_dets[:n]

