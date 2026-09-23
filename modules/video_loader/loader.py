"""Load the config and list video files from the configured folder."""

import json
import os

DEFAULT_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"]


def ensure_local_config(path="config.json"):
    """Create config.local.json next to config.json on first run, if it is missing.

    It starts as a copy of config.json so every key is there to edit; the app reads it
    on top of config.json, and it is git-ignored.
    """
    local_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.local.json")
    if not os.path.exists(local_path):
        with open(path) as f:
            defaults = json.load(f)
        with open(local_path, "w") as f:
            json.dump(defaults, f, indent=2)
    return local_path


def load_config(path="config.json"):
    """Read config.json, then apply config.local.json on top of it if present.

    config.json is tracked in git and holds the defaults; config.local.json is
    git-ignored and holds whatever this machine needs to differ. Any "*_folder"
    value may be relative; it is resolved against the folder holding
    config.json, so the app works from any working directory.
    """
    with open(path) as f:
        config = json.load(f)

    local_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.local.json")
    if os.path.exists(local_path):
        with open(local_path) as f:
            config.update(json.load(f))
    config.setdefault("video_folder", "")
    config.setdefault("extensions", DEFAULT_EXTENSIONS)
    config.setdefault("stiqy_seg_model_folder", "")
    config.setdefault("qt_det_model_folder", "")
    config.setdefault("stiqy_sam_model_folder", "")
    config.setdefault("stiqy_rvm_ckpt_folder", "")
    config.setdefault("output_folder", "output")
    config.setdefault("annotation_folder", "")

    base = os.path.dirname(os.path.abspath(path))
    for key, value in config.items():
        if key.endswith("_folder") and value and not os.path.isabs(value):
            config[key] = os.path.normpath(os.path.join(base, value))
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
