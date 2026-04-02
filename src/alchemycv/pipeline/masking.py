"""Stage 5: Mask generation (color range, grayscale range, adaptive, Otsu)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..constants import COLOR_CONV_MAP, FILTER_DATA

log = logging.getLogger(__name__)


def _ensure_odd(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def process(
    image: np.ndarray,
    filter_selection: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, int | None]:
    """Generate a binary mask from *image*.

    Returns
    -------
    (mask, otsu_threshold)
        *otsu_threshold* is ``None`` unless Otsu's method was used.
    """
    filter_cfg = FILTER_DATA.get(filter_selection, {})
    filter_type = filter_cfg.get("type")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    if filter_type == "grayscale_range":
        min_v = int(params.get("filter_Min_Value", 0))
        max_v = int(params.get("filter_Max_Value", 127))
        if min_v > max_v:
            min_v = max_v
        return cv2.inRange(gray, min_v, max_v), None

    if filter_type == "adaptive_thresh":
        method_str = params.get("filter_Adaptive_Method", "Gaussian C")
        method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if method_str == "Mean C"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        type_str = params.get("filter_Threshold_Type", "Binary")
        thresh_type = cv2.THRESH_BINARY if type_str == "Binary" else cv2.THRESH_BINARY_INV
        bsize = _ensure_odd(int(params.get("filter_Block_Size", 11)))
        c_val = int(params.get("filter_C_(Constant)", 2))
        return cv2.adaptiveThreshold(gray, 255, method, thresh_type, bsize, c_val), None

    if filter_type == "otsu":
        type_str = params.get("otsu_Threshold_Type", "Binary")
        thresh_type = cv2.THRESH_BINARY if type_str == "Binary" else cv2.THRESH_BINARY_INV
        ret, mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return mask, int(ret)

    if filter_type == "color":
        if image.ndim == 2:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), None
        channels = filter_cfg["channels"]
        lower = np.array([params.get(f"color_{c}_min", 0) for c in channels])
        upper = np.array([params.get(f"color_{c}_max", 255) for c in channels])
        code = COLOR_CONV_MAP.get(filter_selection, -1)
        converted = cv2.cvtColor(image, code) if code != -1 else image
        return cv2.inRange(converted, lower, upper), None

    h, w = image.shape[:2]
    return np.zeros((h, w), dtype=np.uint8), None
