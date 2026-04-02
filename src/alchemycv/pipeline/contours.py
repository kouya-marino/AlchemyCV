"""Stage 7: Contour detection, filtering, and drawing."""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def process(
    mask: np.ndarray,
    image_to_draw_on: np.ndarray,
    min_area: int = 50,
    max_area: int = 1_000_000,
    draw: bool = True,
) -> tuple[np.ndarray, int]:
    """Detect contours in *mask*, optionally draw them on *image_to_draw_on*.

    Returns
    -------
    (result_image, count)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    if draw:
        if image_to_draw_on.ndim == 2:
            image_to_draw_on = cv2.cvtColor(image_to_draw_on, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_to_draw_on, filtered, -1, (0, 255, 0), 2)

    return image_to_draw_on, len(filtered)
