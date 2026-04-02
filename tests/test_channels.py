"""Unit tests for pipeline.channels."""

import numpy as np
import pytest

from alchemycv.pipeline import channels


class TestChannelExtraction:
    @pytest.mark.parametrize(
        "space,channel",
        [
            ("Grayscale", "Intensity"),
            ("RGB/BGR", "B"),
            ("RGB/BGR", "G"),
            ("RGB/BGR", "R"),
            ("HSV", "H"),
            ("HSV", "S"),
            ("HSV", "V"),
            ("HLS", "H"),
            ("Lab", "L"),
            ("YCrCb", "Y"),
        ],
    )
    def test_extraction_returns_single_channel(self, bgr_image_100x100, space, channel):
        result = channels.process(bgr_image_100x100, space, channel)
        assert result.ndim == 2
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_empty_channel_name_falls_back_to_gray(self, bgr_image_100x100):
        result = channels.process(bgr_image_100x100, "RGB/BGR", "")
        assert result.ndim == 2

    def test_unknown_color_space(self, bgr_image_100x100):
        result = channels.process(bgr_image_100x100, "FakeSpace", "X")
        assert result.ndim == 2

    def test_grayscale_input(self, gray_image_100x100):
        """Grayscale input should be returned as-is for grayscale space."""
        result = channels.process(gray_image_100x100, "Grayscale", "Intensity")
        assert result.ndim == 2

    def test_invalid_channel_name(self, bgr_image_100x100):
        """Invalid channel name should fall back to index 0."""
        result = channels.process(bgr_image_100x100, "RGB/BGR", "NonExistent")
        assert result.ndim == 2
