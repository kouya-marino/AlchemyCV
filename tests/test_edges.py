"""Unit tests for pipeline.edges."""

import numpy as np
import pytest

from alchemycv.pipeline import edges


class TestEdgeDetection:
    def test_canny(self, gray_image_100x100):
        params = {"edge_Threshold_1": 50, "edge_Threshold_2": 150}
        result = edges.process(gray_image_100x100, "Canny", params)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("direction", ["X", "Y", "Magnitude"])
    def test_sobel_directions(self, gray_image_100x100, direction):
        params = {"edge_Kernel_Size": 3, "edge_Direction": direction}
        result = edges.process(gray_image_100x100, "Sobel", params)
        assert result.shape == (100, 100)

    def test_sobel_even_kernel(self, gray_image_100x100):
        """Even kernel size should be auto-corrected."""
        params = {"edge_Kernel_Size": 4, "edge_Direction": "Magnitude"}
        result = edges.process(gray_image_100x100, "Sobel", params)
        assert result.shape == (100, 100)

    @pytest.mark.parametrize("direction", ["X", "Y", "Magnitude"])
    def test_prewitt_directions(self, gray_image_100x100, direction):
        params = {"edge_Direction": direction}
        result = edges.process(gray_image_100x100, "Prewitt", params)
        assert result.shape == (100, 100)

    @pytest.mark.parametrize("direction", ["X", "Y", "Magnitude"])
    def test_roberts_directions(self, gray_image_100x100, direction):
        params = {"edge_Direction": direction}
        result = edges.process(gray_image_100x100, "Roberts", params)
        assert result.shape == (100, 100)

    def test_unknown_detector(self, gray_image_100x100):
        result = edges.process(gray_image_100x100, "FakeEdge", {})
        assert result is gray_image_100x100

    def test_canny_on_uniform_image(self):
        """Uniform image should produce no edges."""
        uniform = np.full((50, 50), 128, dtype=np.uint8)
        params = {"edge_Threshold_1": 50, "edge_Threshold_2": 150}
        result = edges.process(uniform, "Canny", params)
        assert np.all(result == 0)
