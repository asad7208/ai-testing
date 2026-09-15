"""Real-time deinterlacing for interlaced source frames."""

import cv2


def deinterlace(frame, field=0):
    """Drop one field and interpolate it back to full height (bob deinterlace).

    field=0 keeps the top field (even rows), field=1 the bottom field.
    """
    h, w = frame.shape[:2]
    return cv2.resize(frame[field::2], (w, h), interpolation=cv2.INTER_LINEAR)
