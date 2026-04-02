"""Pipeline orchestrator — runs all stages in order.

This module is the *only* place that knows about the stage ordering.
Every individual stage module is a pure function: image in, image out.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import preprocessing, enhancement, frequency, channels, masking, edges, morphology, contours

log = logging.getLogger(__name__)


def run(
    original: np.ndarray,
    sel: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute the full processing pipeline.

    Parameters
    ----------
    original : The original BGR uint8 image.
    sel : Selector dict with keys like ``preproc``, ``enhancement``, etc.
    params : Flat parameter dict (``{param_key: value, ...}``).

    Returns
    -------
    Dict with keys ``final``, ``mask``, ``enhanced``, ``preprocessed``,
    ``extracted_channel``, ``object_count``, ``otsu_threshold``.
    """
    preprocessed_image = preprocessing.process(original, sel["preproc"], params)
    enhanced_image = enhancement.process(preprocessed_image, sel["enhancement"], params)
    freq_filtered_image = frequency.process(enhanced_image, sel["frequency"], params)

    image_for_masking = freq_filtered_image
    extracted_channel = None

    if sel["channel_enabled"]:
        image_for_masking = channels.process(
            freq_filtered_image, sel["color_space"], sel["channel"]
        )
        extracted_channel = image_for_masking.copy()

    otsu_threshold = None

    if sel["edge_enabled"]:
        edge_source = image_for_masking if sel["channel_enabled"] else freq_filtered_image
        gray = cv2.cvtColor(edge_source, cv2.COLOR_BGR2GRAY) if len(edge_source.shape) == 3 else edge_source
        mask = edges.process(gray, sel["edge_filter"], params)
    else:
        mask, otsu_threshold = masking.process(image_for_masking, sel["filter"], params)

    if mask is None:
        return None

    if sel["morph_enabled"]:
        mask = morphology.process(mask, params)

    final_image = cv2.bitwise_and(freq_filtered_image, freq_filtered_image, mask=mask)

    object_count = None
    if sel["contours_enabled"]:
        final_image, object_count = contours.process(
            mask,
            final_image.copy(),
            min_area=sel["min_area"],
            max_area=sel["max_area"],
            draw=sel["draw_contours"],
        )

    return {
        "final": final_image,
        "mask": mask,
        "enhanced": enhanced_image,
        "preprocessed": preprocessed_image,
        "extracted_channel": extracted_channel,
        "object_count": object_count,
        "otsu_threshold": otsu_threshold,
    }
