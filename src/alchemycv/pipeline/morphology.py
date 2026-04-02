"""Stage 6b: Morphological operations on binary masks."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..constants import MORPH_OP_MAP, MORPH_SHAPE_MAP

log = logging.getLogger(__name__)


def _ensure_odd(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def process(mask: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply a morphological operation to *mask*.

    Expected keys in *params*: ``Morph Operation``, ``Morph Kernel Shape``,
    ``Morph Kernel Size``, ``Morph Iterations``.
    """
    op_str = params.get("Morph Operation", "Dilate")
    shape_str = params.get("Morph Kernel Shape", "Rectangle")
    k_size = _ensure_odd(int(params.get("Morph Kernel Size", 5)))
    iterations = max(1, int(params.get("Morph Iterations", 1)))

    shape_cv = MORPH_SHAPE_MAP.get(shape_str, cv2.MORPH_RECT)
    op_cv = MORPH_OP_MAP.get(op_str, cv2.MORPH_DILATE)

    kernel = cv2.getStructuringElement(shape_cv, (k_size, k_size))
    return cv2.morphologyEx(mask, op_cv, kernel, iterations=iterations)
