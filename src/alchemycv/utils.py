"""Utility helpers — image I/O with Unicode path support."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)


def load_image(filepath: str) -> np.ndarray | None:
    """Load an image from *filepath* (Unicode-safe via ``np.fromfile``)."""
    try:
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log.error("cv2.imdecode returned None for %s", filepath)
        return img
    except Exception:
        log.exception("Failed to load image: %s", filepath)
        return None


def save_image(image: np.ndarray, filepath: str) -> None:
    """Save *image* to *filepath* (Unicode-safe via ``cv2.imencode``)."""
    ext = os.path.splitext(filepath)[1]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"cv2.imencode failed for extension {ext}")
    encoded.tofile(filepath)
    log.info("Image saved to %s", filepath)
