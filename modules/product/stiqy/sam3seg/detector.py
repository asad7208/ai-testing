"""SAM 3 / SAM 3.1 text-prompted segmentation (ultralytics SAM3SemanticPredictor).

Weights are gated on Hugging Face and are not auto-downloaded.
"""

import os

import cv2
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

IMGSZ = 1008  # the SAM 3 backbone is fixed at 1008x1008


def list_models(folder):
    """Return [(filename, full_path), ...] for every sam*.pt weight in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        (name, os.path.join(folder, name))
        for name in os.listdir(folder)
        if name.lower().startswith("sam") and name.lower().endswith(".pt")
    )


class Sam3Segmenter:
    def __init__(self):
        self.predictor = None
        self.texts = []
        self.conf = 0.25
        self.iou = 0.7

    def load(self, weights_path, fp16=True):
        """Build a SAM3SemanticPredictor around the given weights."""
        self.predictor = SAM3SemanticPredictor(
            overrides={
                "model": weights_path,
                "task": "segment",
                "mode": "predict",
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": IMGSZ,
                "device": "0" if torch.cuda.is_available() else "cpu",
                "quantize": 16 if fp16 and torch.cuda.is_available() else 32,
                "save": False,
                "verbose": False,
            }
        )
        return self

    def set_prompt(self, text):
        """Set the concepts to segment from a comma-separated string."""
        self.texts = [t.strip() for t in text.split(",") if t.strip()]

    def predict(self, frame, opacity=0.5, show_boxes=True):
        """Return a copy of frame with the prompted masks blended in."""
        if self.predictor is None or not self.texts:
            return frame

        self.predictor.set_image(frame)
        try:
            result = self.predictor(text=self.texts)[0]
        finally:
            self.predictor.reset_image()

        painted = result.plot(boxes=show_boxes, labels=show_boxes)
        return cv2.addWeighted(frame, 1 - opacity, painted, opacity, 0)
