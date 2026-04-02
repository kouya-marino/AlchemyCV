"""Stage 4: Color-space conversion and single-channel extraction."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..constants import CHANNEL_DATA

log = logging.getLogger(__name__)


def process(
    image: np.ndarray,
    color_space: str,
    channel_name: str,
) -> np.ndarray:
    """Extract a single channel from *image* after converting to *color_space*.

    Returns a single-channel (grayscale) image.
    """
    if not channel_name:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    space_info = CHANNEL_DATA.get(color_space)
    if space_info is None:
        log.warning("Unknown color space: %s", color_space)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(image.shape) == 3 and space_info["code"] is not None:
        converted = cv2.cvtColor(image, space_info["code"])
    else:
        converted = image

    if converted.ndim == 2:
        return converted

    channels = cv2.split(converted)
    try:
        idx = space_info["channels"].index(channel_name)
    except ValueError:
        log.warning("Channel %s not in %s", channel_name, color_space)
        idx = 0
    return channels[idx]
