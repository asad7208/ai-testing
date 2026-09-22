"""YOLO segmentation model: run it on a frame and draw masks / boxes."""

import os

import cv2
import numpy as np
from ultralytics import YOLO

COLORS = np.array(
    [
        [255, 56, 56], [56, 255, 56], [56, 56, 255], [255, 255, 56],
        [255, 56, 255], [56, 255, 255], [255, 149, 56], [149, 56, 255],
    ],
    dtype=np.uint8,
)


def list_models(folder):
    """Return [(filename, full_path), ...] for every .pt weight in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        (name, os.path.join(folder, name))
        for name in os.listdir(folder)
        if name.lower().endswith(".pt")
    )


class SegDetector:
    def __init__(self):
        self.model = None
        self.names = {}
        self.conf = 0.25

    def load(self, weights_path):
        """Load a .pt YOLO segmentation model."""
        self.model = YOLO(weights_path)
        self.names = self.model.names
        return self

    def boxes(self, frame, class_name="person"):
        """Return xyxy boxes for one class (all classes if the model has no such name)."""
        if self.model is None:
            return []
        # retina_masks keeps masks at the original frame size; without it they
        # come back at the letterboxed model input and resizing shifts them.
        result = self.model.predict(
            frame, conf=self.conf, retina_masks=True, verbose=False
        )[0]
        if result.boxes is None:
            return []
        wanted = [i for i, n in self.names.items() if n == class_name]
        return [
            tuple(box.xyxy[0].int().tolist())
            for box in result.boxes
            if not wanted or int(box.cls) in wanted
        ]

    def predict(self, frame, show_boxes=True, show_masks=True, opacity=0.5):
        """Return a copy of frame with masks and/or boxes drawn on it."""
        if self.model is None:
            return frame

        # retina_masks keeps masks at the original frame size; without it they
        # come back at the letterboxed model input and resizing shifts them.
        result = self.model.predict(
            frame, conf=self.conf, retina_masks=True, verbose=False
        )[0]
        out = frame.copy()

        if show_masks and result.masks is not None:
            overlay = out.copy()
            for i, mask in enumerate(result.masks.data.cpu().numpy()):
                cls = int(result.boxes.cls[i]) if result.boxes is not None else i
                if mask.shape != out.shape[:2]:
                    mask = cv2.resize(mask, (out.shape[1], out.shape[0]))
                overlay[mask > 0.5] = COLORS[cls % len(COLORS)]
            cv2.addWeighted(overlay, opacity, out, 1 - opacity, 0, out)

        if show_boxes and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls = int(box.cls)
                color = [int(c) for c in COLORS[cls % len(COLORS)]]
                label = f"{self.names.get(cls, cls)} {float(box.conf):.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                )

        return out
