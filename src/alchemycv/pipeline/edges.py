"""Stage 6a: Edge detection (Canny, Sobel, Prewitt, Roberts)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Pre-defined convolution kernels
_KERNELS: dict[str, np.ndarray] = {
    "Prewitt_X": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    "Prewitt_Y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "Roberts_X": np.array([[1, 0], [0, -1]]),
    "Roberts_Y": np.array([[0, 1], [-1, 0]]),
}


def _ensure_odd(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def process(image: np.ndarray, edge_name: str, params: dict[str, Any]) -> np.ndarray:
    """Apply edge detection on a **grayscale** input image.

    Returns a single-channel edge map (uint8).
    """
    if edge_name == "Canny":
        t1 = int(params.get("edge_Threshold_1", 50))
        t2 = int(params.get("edge_Threshold_2", 150))
        return cv2.Canny(image, t1, t2)

    if edge_name == "Sobel":
        ksize = _ensure_odd(int(params.get("edge_Kernel_Size", 3)))
        direction = params.get("edge_Direction", "Magnitude")
        return _directional_filter(image, direction, _sobel_factory(ksize))

    if edge_name in ("Prewitt", "Roberts"):
        direction = params.get("edge_Direction", "Magnitude")
        kx = _KERNELS[f"{edge_name}_X"]
        ky = _KERNELS[f"{edge_name}_Y"]
        return _directional_filter(
            image,
            direction,
            lambda img, axis: cv2.filter2D(img, -1, kx if axis == "X" else ky),
        )

    log.warning("Unknown edge detector: %s", edge_name)
    return image


def _sobel_factory(ksize: int):
    """Return a callable ``(image, axis) -> gradient`` for Sobel."""

    def _apply(img: np.ndarray, axis: str) -> np.ndarray:
        dx, dy = (1, 0) if axis == "X" else (0, 1)
        return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)

    return _apply


def _directional_filter(image: np.ndarray, direction: str, grad_fn) -> np.ndarray:
    if direction == "X":
        return cv2.convertScaleAbs(grad_fn(image, "X"))
    if direction == "Y":
        return cv2.convertScaleAbs(grad_fn(image, "Y"))
    # Magnitude
    gx = grad_fn(image, "X").astype(np.float64)
    gy = grad_fn(image, "Y").astype(np.float64)
    return cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
