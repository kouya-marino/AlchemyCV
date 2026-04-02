"""Unit tests for pipeline.frequency."""

import numpy as np
import pytest

from alchemycv.pipeline import frequency


class TestFrequencyFilter:
    def test_none_returns_same(self, bgr_image_100x100):
        result = frequency.process(bgr_image_100x100, "None", {})
        assert result is bgr_image_100x100

    @pytest.mark.parametrize(
        "filter_name",
        [
            "Ideal Low-Pass",
            "Gaussian Low-Pass",
            "Butterworth Low-Pass",
            "Ideal High-Pass",
            "Gaussian High-Pass",
            "Butterworth High-Pass",
        ],
    )
    def test_all_filter_types(self, bgr_image_100x100, filter_name):
        params = {"frequency_Cutoff_Freq_(D0)": 30, "frequency_Order_(n)": 2}
        result = frequency.process(bgr_image_100x100, filter_name, params)
        assert result.shape == bgr_image_100x100.shape
        assert result.dtype == np.uint8

    def test_grayscale_input(self, gray_image_100x100):
        params = {"frequency_Cutoff_Freq_(D0)": 30}
        result = frequency.process(gray_image_100x100, "Ideal Low-Pass", params)
        assert result.ndim == 2

    def test_d0_zero_lowpass(self, bgr_image_100x100):
        """D0=0 for low-pass should return an image (all-pass)."""
        params = {"frequency_Cutoff_Freq_(D0)": 0}
        result = frequency.process(bgr_image_100x100, "Ideal Low-Pass", params)
        assert result.shape == bgr_image_100x100.shape

    def test_d0_zero_highpass(self, bgr_image_100x100):
        """D0=0 for high-pass should return all-zero mask."""
        params = {"frequency_Cutoff_Freq_(D0)": 0}
        result = frequency.process(bgr_image_100x100, "Ideal High-Pass", params)
        assert result.shape == bgr_image_100x100.shape


class TestFilterMask:
    def test_ideal_lowpass_mask_shape(self):
        mask = frequency.create_filter_mask((64, 64), "Ideal Low-Pass", 20)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_ideal_lowpass_center_is_one(self):
        mask = frequency.create_filter_mask((64, 64), "Ideal Low-Pass", 20)
        assert mask[32, 32] == 1.0

    def test_ideal_highpass_center_is_zero(self):
        mask = frequency.create_filter_mask((64, 64), "Ideal High-Pass", 20)
        assert mask[32, 32] == 0.0

    def test_butterworth_highpass_no_nan(self):
        """Butterworth HP at D=0 should not produce NaN."""
        mask = frequency.create_filter_mask((64, 64), "Butterworth High-Pass", 20, n=2)
        assert not np.any(np.isnan(mask))

    def test_unknown_filter_returns_ones(self):
        mask = frequency.create_filter_mask((32, 32), "Unknown", 10)
        np.testing.assert_array_equal(mask, np.ones((32, 32), np.float32))
