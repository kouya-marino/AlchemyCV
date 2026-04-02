"""Stage 1: Pre-processing filters (blur / smoothing)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _ensure_odd(value: int) -> int:
    """Return *value* as a positive odd integer (required by OpenCV kernels)."""
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def process(image: np.ndarray, preproc_name: str, params: dict[str, Any]) -> np.ndarray:
    """Apply the selected pre-processing filter.

    Parameters
    ----------
    image : BGR uint8 image.
    preproc_name : Display name of the selected filter (e.g. ``"Gaussian Blur"``).
    params : Flat dict of captured parameter values keyed with ``preproc_`` prefix.

    Returns
    -------
    Processed image (same dtype/shape contract as input).
    """
    if preproc_name == "None":
        return image

    if preproc_name == "Gaussian Blur":
        ksize = _ensure_odd(params.get("preproc_Kernel_Size", 5))
        log.debug("Gaussian blur ksize=%d", ksize)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    if preproc_name == "Median Blur":
        ksize = _ensure_odd(params.get("preproc_Kernel_Size", 5))
        log.debug("Median blur ksize=%d", ksize)
        return cv2.medianBlur(image, ksize)

    if preproc_name == "Bilateral Filter":
        d = int(params.get("preproc_Diameter", 9))
        sc = int(params.get("preproc_Sigma_Color", 75))
        ss = int(params.get("preproc_Sigma_Space", 75))
        log.debug("Bilateral d=%d sc=%d ss=%d", d, sc, ss)
        return cv2.bilateralFilter(image, d, sc, ss)

    log.warning("Unknown preprocessing filter: %s", preproc_name)
    return image
