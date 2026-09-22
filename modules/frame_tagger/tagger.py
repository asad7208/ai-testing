"""Save a tagged frame: the original and the overlay produced by whatever is running."""

import csv
import os
from datetime import datetime


def save_tag(out_dir, video_path, frame_index, raw, overlay, overlays="", note=""):
    """Write the frame under <out_dir>/<video>/images and the overlay under
    <out_dir>/<video>/overlays, and append a row to tags.csv.

    Returns (original_path, overlay_path).
    """
    import cv2

    stem = os.path.splitext(os.path.basename(video_path))[0]
    images_dir = os.path.join(out_dir, stem, "images")
    overlays_dir = os.path.join(out_dir, stem, "overlays")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    name = f"{stem}_f{frame_index:06d}.png"
    original_path = os.path.join(images_dir, name)
    overlay_path = os.path.join(overlays_dir, name)
    cv2.imwrite(original_path, raw)
    cv2.imwrite(overlay_path, overlay)

    log_path = os.path.join(out_dir, "tags.csv")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["time", "video", "frame", "overlays", "note", "original", "overlay"]
            )
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            video_path,
            frame_index,
            overlays,
            note,
            original_path,
            overlay_path,
        ])

    return original_path, overlay_path
