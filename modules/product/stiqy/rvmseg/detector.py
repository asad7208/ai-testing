"""RVM segmentation: run a fine-tuned RVM checkpoint on frames.

The segmentation branch is recurrent, so the hidden state r1..r4 is carried from one
frame to the next while playing. Seeking breaks that chain, so the state is reset
whenever the incoming frame is not the successor of the previous one.
"""

import os

import cv2
import numpy as np
import torch
from torch.nn import functional as NF

from .net import MattingNetwork

RESOLUTION = (512, 256)  # (h, w) the checkpoints were trained at
MARGIN = 0.15  # box expansion, matching the crops the checkpoints were trained on


def expand_box(x1, y1, x2, y2, margin=MARGIN, aspect=RESOLUTION[1] / RESOLUTION[0]):
    """Grow the box, then pad its short side to the training aspect (w/h).

    Padding instead of letting the resize squash the crop matters: the model only ever
    saw upright person crops of roughly this shape. The box is deliberately not clamped
    to the frame -- crop_padded replicates the edges instead.
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * (1 + margin), (y2 - y1) * (1 + margin)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return (round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2))


def crop_padded(frame, x1, y1, x2, y2):
    """Crop a possibly out-of-frame box, replicating edges so the aspect is preserved.

    Returns (crop, (vx1, vy1, vx2, vy2), (ox, oy)) where the v* box is the part inside
    the frame and o* its offset within the crop.
    """
    h, w = frame.shape[:2]
    vx1, vy1, vx2, vy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if vx2 <= vx1 or vy2 <= vy1:
        return None, None, None
    inner = frame[vy1:vy2, vx1:vx2]
    top, left, bottom, right = vy1 - y1, vx1 - x1, y2 - vy2, x2 - vx2
    if top or bottom or left or right:
        inner = cv2.copyMakeBorder(inner, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return inner, (vx1, vy1, vx2, vy2), (left, top)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def list_models(folder):
    """Return [(filename, full_path), ...] for every .pth checkpoint in folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        (name, os.path.join(folder, name))
        for name in os.listdir(folder)
        if name.lower().endswith((".pth", ".pt"))
    )


class RvmSegmenter:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 0.5
        self.native = False  # run at source resolution instead of the training size
        self.rec = [None] * 4
        self.last_index = None
        self.box_states = []  # [(box, rec), ...] from the previous frame, for crop mode

    def load(self, checkpoint_path):
        """Load a fine-tuned RVM segmentation checkpoint."""
        model = MattingNetwork("mobilenetv3").eval().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = model
        self.reset_state()
        return self

    def reset_state(self):
        self.rec = [None] * 4
        self.last_index = None
        self.box_states = []

    @torch.no_grad()
    def _run(self, crop_bgr, rec):
        """Run the seg branch on one BGR crop at the training resolution."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (RESOLUTION[1], RESOLUTION[0]), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255)
        with torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
            out = self.model(x[None, None], *rec, segmentation_pass=True)
        return out[0].float().sigmoid()[0, 0], list(out[1:])

    def alpha_from_boxes(self, frame, boxes, frame_index=None):
        """Mask for the whole frame, built by running each person box as its own crop.

        Each box keeps its own recurrent state, carried to the box it overlaps most in
        the next frame (there is no tracker here, so IoU stands in for a track id).
        """
        if frame_index is not None and frame_index != (self.last_index or -2) + 1:
            self.box_states = []
        self.last_index = frame_index

        h0, w0 = frame.shape[:2]
        alpha = np.zeros((h0, w0), np.float32)
        new_states = []

        for box in boxes:
            x1, y1, x2, y2 = expand_box(*box)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop, valid, offset = crop_padded(frame, x1, y1, x2, y2)
            if crop is None:
                continue

            previous = max(self.box_states, key=lambda s: _iou(box, s[0]), default=None)
            rec = previous[1] if previous and _iou(box, previous[0]) > 0.5 else [None] * 4

            prob, rec = self._run(crop, rec)
            new_states.append((box, rec))

            prob = NF.interpolate(
                prob[None], (y2 - y1, x2 - x1), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
            vx1, vy1, vx2, vy2 = valid
            ox, oy = offset
            prob = prob[oy:oy + (vy2 - vy1), ox:ox + (vx2 - vx1)]
            np.maximum(alpha[vy1:vy2, vx1:vx2], prob, out=alpha[vy1:vy2, vx1:vx2])

        self.box_states = new_states
        return (alpha > self.threshold).astype(np.float32) if self.threshold > 0 else alpha

    @torch.no_grad()
    def alpha(self, frame, frame_index=None):
        """Return the mask for a BGR frame as float32 HxW in [0, 1]."""
        if frame_index is not None and frame_index != (self.last_index or -2) + 1:
            self.rec = [None] * 4  # a seek broke the sequence
        self.last_index = frame_index

        h0, w0 = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self.native:
            rgb = cv2.resize(rgb, (RESOLUTION[1], RESOLUTION[0]), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255)
        if self.native:
            # pad to a multiple of 16 so the /16 backbone strides line up
            h, w = x.shape[-2:]
            pad_h, pad_w = (-h) % 16, (-w) % 16
            if pad_h or pad_w:
                x = NF.pad(x[None], (0, pad_w, 0, pad_h), mode="replicate")[0]
        x = x[None, None]

        with torch.autocast("cuda", torch.float16, enabled=self.device == "cuda"):
            out = self.model(x, *self.rec, segmentation_pass=True)
        logit, self.rec = out[0], out[1:]

        prob = logit.float().sigmoid()[0, 0]
        if self.native:
            prob = prob[:, :h0, :w0]
        prob = NF.interpolate(prob[None], (h0, w0), mode="bilinear", align_corners=False)
        prob = prob[0, 0].cpu().numpy()
        return (prob > self.threshold).astype(np.float32) if self.threshold > 0 else prob

    def predict(self, frame, frame_index=None, opacity=0.5, view="overlay", boxes=None):
        """Return the frame with the mask drawn on it ('overlay') or the mask itself ('mask').

        With `boxes` (xyxy person boxes) the mask is built per crop, which is how the
        checkpoints were trained; without them the whole frame is run at once.
        """
        if self.model is None:
            return frame

        if boxes is None:
            alpha = self.alpha(frame, frame_index)
        else:
            alpha = self.alpha_from_boxes(frame, boxes, frame_index)
        if view == "mask":
            return cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        tint = np.zeros_like(frame)
        tint[:] = (56, 255, 56)
        a = (alpha * opacity)[..., None]
        return (frame * (1 - a) + tint * a).astype(np.uint8)
