"""Stage 2: Image enhancement algorithms."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _ensure_odd(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def process(image: np.ndarray, enhance_name: str, params: dict[str, Any]) -> np.ndarray:
    """Apply the selected enhancement algorithm.

    Parameters
    ----------
    image : BGR or grayscale uint8 image.
    enhance_name : Display name (e.g. ``"CLAHE"``).
    params : Flat dict keyed with ``enhancement_`` prefix.
    """
    if enhance_name == "None":
        return image

    if enhance_name == "Histogram Equalization":
        return _hist_equalize(image)

    if enhance_name == "CLAHE":
        clip = float(params.get("enhancement_Clip_Limit", 2))
        tile = int(params.get("enhancement_Tile_Grid_Size", 8))
        return _clahe(image, clip, tile)

    if enhance_name == "Contrast Stretching":
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    if enhance_name == "Gamma Correction":
        gamma = params.get("enhancement_Gamma_x100", 100) / 100.0
        return _gamma_correction(image, gamma)

    if enhance_name == "Log Transform":
        return _log_transform(image)

    if enhance_name == "Single-Scale Retinex":
        sigma = params.get("enhancement_Sigma", 30)
        return _retinex(image, sigma)

    if enhance_name == "Unsharp Masking":
        ksize = _ensure_odd(params.get("enhancement_Kernel_Size", 5))
        alpha = params.get("enhancement_Alpha_x10", 15) / 10.0
        return _unsharp(image, ksize, alpha)

    log.warning("Unknown enhancement: %s", enhance_name)
    return image


# ------------------------------------------------------------------
# Internal implementations
# ------------------------------------------------------------------

def _hist_equalize(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _clahe(image: np.ndarray, clip: float, tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    if len(image.shape) == 2:
        return clahe.apply(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        gamma = 0.01
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def _log_transform(image: np.ndarray) -> np.ndarray:
    max_val = np.max(image)
    if max_val == 0:
        return image
    c = 255 / np.log(1 + float(max_val))
    log_image = c * np.log(image.astype(np.float32) + 1)
    return np.array(log_image, dtype=np.uint8)


def _retinex(image: np.ndarray, sigma: int) -> np.ndarray:
    img_float = image.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    r_log = np.log(img_float) - np.log(blurred)
    result = cv2.normalize(r_log, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def _unsharp(image: np.ndarray, ksize: int, alpha: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
