"""YOLO detection model (yolov8 / yolov8-p2 / yolo26): run it on a frame and draw boxes."""

import os

import cv2
from ultralytics import YOLO

COLORS = [
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 56),
    (255, 56, 255), (56, 255, 255), (255, 149, 56), (149, 56, 255),
]


def list_models(folder):
    """Return [(filename, full_path), ...] for every .pt weight in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        (name, os.path.join(folder, name))
        for name in os.listdir(folder)
        if name.lower().endswith(".pt")
    )


class YoloDetector:
    def __init__(self):
        self.model = None
        self.names = {}
        self.conf = 0.25

    def load(self, weights_path):
        """Load any ultralytics detection .pt (yolov8, yolov8-p2, yolo26, ...)."""
        self.model = YOLO(weights_path)
        self.names = self.model.names
        return self

    def predict(self, frame, show_labels=True):
        """Return a copy of frame with detection boxes drawn on it."""
        if self.model is None:
            return frame

        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = frame.copy()
        if result.boxes is None:
            return out

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                cv2.putText(
                    out, f"{self.names.get(cls, cls)} {float(box.conf):.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                )
        return out
