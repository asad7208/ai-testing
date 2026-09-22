"""Load the config and list video files from the configured folder."""

import json
import os

DEFAULT_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"]


def load_config(path="config.json"):
    """Read config.json. Returns dict with 'video_folder' and 'extensions'."""
    with open(path) as f:
        config = json.load(f)
    config.setdefault("video_folder", "")
    config.setdefault("extensions", DEFAULT_EXTENSIONS)
    config.setdefault("stiqy_seg_model_folder", "")
    config.setdefault("qt_det_model_folder", "")
    config.setdefault("stiqy_sam_model_folder", "")
    config.setdefault("stiqy_rvm_ckpt_folder", "")
    config.setdefault("output_folder", "output")
    return config


def list_videos(folder, extensions=None):
    """Return [(filename, full_path), ...] for every video in folder, sorted by name."""
    extensions = tuple(e.lower() for e in (extensions or DEFAULT_EXTENSIONS))
    if not os.path.isdir(folder):
        return []
    videos = [
        (name, os.path.join(folder, name))
        for name in os.listdir(folder)
        if name.lower().endswith(extensions)
    ]
    return sorted(videos)
