"""Load a YOLO-format annotated dataset and draw its labels on the images."""

import os

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

FORMATS = ("YOLO segmentation", "YOLO detection")

COLORS = np.array(
    [
        [255, 56, 56], [56, 255, 56], [56, 56, 255], [255, 255, 56],
        [255, 56, 255], [56, 255, 255], [255, 149, 56], [149, 56, 255],
    ],
    dtype=np.uint8,
)


def list_pairs(root):
    """Return [(image_name, image_path, label_path_or_None), ...] for a dataset root.

    Expects the usual YOLO layout: <root>/images/... and <root>/labels/...
    A flat folder holding both images and .txt files also works.
    """
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")
    if not os.path.isdir(images_dir):
        images_dir = labels_dir = root
    if not os.path.isdir(images_dir):
        return []

    pairs = []
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        label = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")
        pairs.append((name, os.path.join(images_dir, name), label if os.path.exists(label) else None))
    return pairs


def load_class_names(root):
    """Read classes.txt / names from the dataset root if there is one."""
    for candidate in (
        "classes.txt", "obj.names",
        os.path.join("labels", "classes.txt"),
        os.path.join("images", "classes.txt"),
    ):
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]
    return []


def read_labels(label_path, width, height, fmt):
    """Parse one YOLO label file into [(cls, points_or_box), ...] in pixel coords."""
    if not label_path:
        return []

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
            if fmt == "YOLO segmentation" and len(values) >= 6:
                points = np.array(values, dtype=np.float32).reshape(-1, 2)
                points[:, 0] *= width
                points[:, 1] *= height
                annotations.append((cls, points.astype(np.int32)))
            else:
                cx, cy, bw, bh = values[:4]
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                annotations.append((cls, np.array([[x1, y1], [x2, y2]], dtype=np.int32)))
    return annotations


def draw(image, annotations, names=None, opacity=0.4, show_labels=True):
    """Draw polygons (segmentation) or rectangles (detection) on a copy of image."""
    out = image.copy()
    overlay = out.copy()
    names = names or []

    for cls, points in annotations:
        color = [int(c) for c in COLORS[cls % len(COLORS)]]
        label = names[cls] if cls < len(names) else str(cls)
        if len(points) == 2:  # detection box
            (x1, y1), (x2, y2) = points
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            anchor = (x1, max(y1 - 6, 12))
        else:  # segmentation polygon
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(out, [points], True, color, 2)
            anchor = (int(points[:, 0].min()), max(int(points[:, 1].min()) - 6, 12))
        if show_labels:
            cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, opacity, out, 1 - opacity, 0)
