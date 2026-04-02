"""Stage 3: Frequency-domain filtering (DFT-based LPF / HPF)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def process(image: np.ndarray, filter_name: str, params: dict[str, Any]) -> np.ndarray:
    """Apply frequency-domain filter and return the spatial result."""
    if filter_name == "None":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    rows, cols = gray.shape
    m_rows = cv2.getOptimalDFTSize(rows)
    m_cols = cv2.getOptimalDFTSize(cols)
    padded = cv2.copyMakeBorder(gray, 0, m_rows - rows, 0, m_cols - cols,
                                cv2.BORDER_CONSTANT, value=0)

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complex_i = cv2.merge(planes)
    cv2.dft(complex_i, complex_i)
    dft_shift = np.fft.fftshift(complex_i)

    D0 = int(params.get("frequency_Cutoff_Freq_(D0)", 30))
    n = int(params.get("frequency_Order_(n)", 2))

    mask = create_filter_mask((m_rows, m_cols), filter_name, D0, n)
    mask_complex = cv2.merge([mask, mask])

    fshift = dft_shift * mask_complex
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    result = np.uint8(img_back)[:rows, :cols]

    if len(image.shape) == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def create_filter_mask(
    shape: tuple[int, int],
    filter_type: str,
    D0: int,
    n: int = 2,
) -> np.ndarray:
    """Generate a 2-D frequency filter mask (float32, same size as *shape*)."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)
    D = np.sqrt((U - crow) ** 2 + (V - ccol) ** 2).astype(np.float32)

    if D0 == 0:
        if "Low-Pass" in filter_type:
            return np.ones((rows, cols), np.float32)
        return np.zeros((rows, cols), np.float32)

    if filter_type == "Ideal Low-Pass":
        mask = (D <= D0).astype(np.float32)
    elif filter_type == "Gaussian Low-Pass":
        mask = np.exp(-(D ** 2) / (2 * D0 ** 2))
    elif filter_type == "Butterworth Low-Pass":
        mask = 1.0 / (1.0 + (D / D0) ** (2 * n))
    elif filter_type == "Ideal High-Pass":
        mask = (D > D0).astype(np.float32)
    elif filter_type == "Gaussian High-Pass":
        mask = 1.0 - np.exp(-(D ** 2) / (2 * D0 ** 2))
    elif filter_type == "Butterworth High-Pass":
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = 1.0 / (1.0 + (D0 / D) ** (2 * n))
        mask[D == 0] = 0.0
    else:
        log.warning("Unknown frequency filter: %s", filter_type)
        mask = np.ones((rows, cols), np.float32)

    return mask.astype(np.float32)
