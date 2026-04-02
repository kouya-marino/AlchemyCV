"""Unit tests for pipeline.enhancement."""

import numpy as np

from alchemycv.pipeline import enhancement


class TestEnhancement:
    def test_none_returns_same(self, bgr_image_100x100):
        result = enhancement.process(bgr_image_100x100, "None", {})
        assert result is bgr_image_100x100

    def test_histogram_equalization_bgr(self, bgr_image_100x100):
        result = enhancement.process(bgr_image_100x100, "Histogram Equalization", {})
        assert result.shape == bgr_image_100x100.shape
        assert result.dtype == np.uint8

    def test_histogram_equalization_gray(self, gray_image_100x100):
        result = enhancement.process(gray_image_100x100, "Histogram Equalization", {})
        assert result.shape == gray_image_100x100.shape

    def test_clahe(self, bgr_image_100x100):
        params = {"enhancement_Clip_Limit": 2, "enhancement_Tile_Grid_Size": 8}
        result = enhancement.process(bgr_image_100x100, "CLAHE", params)
        assert result.shape == bgr_image_100x100.shape

    def test_clahe_grayscale(self, gray_image_100x100):
        params = {"enhancement_Clip_Limit": 2, "enhancement_Tile_Grid_Size": 8}
        result = enhancement.process(gray_image_100x100, "CLAHE", params)
        assert result.shape == gray_image_100x100.shape

    def test_contrast_stretching(self, bgr_image_100x100):
        result = enhancement.process(bgr_image_100x100, "Contrast Stretching", {})
        assert result.shape == bgr_image_100x100.shape

    def test_gamma_correction(self, bgr_image_100x100):
        params = {"enhancement_Gamma_x100": 150}
        result = enhancement.process(bgr_image_100x100, "Gamma Correction", params)
        assert result.shape == bgr_image_100x100.shape

    def test_gamma_identity(self, bgr_image_100x100):
        """Gamma=1.0 (100) should return nearly identical image."""
        params = {"enhancement_Gamma_x100": 100}
        result = enhancement.process(bgr_image_100x100, "Gamma Correction", params)
        np.testing.assert_array_equal(result, bgr_image_100x100)

    def test_log_transform(self, bgr_image_100x100):
        result = enhancement.process(bgr_image_100x100, "Log Transform", {})
        assert result.shape == bgr_image_100x100.shape
        assert result.dtype == np.uint8

    def test_log_transform_black_image(self, black_image):
        """Log transform on all-zero image should not crash."""
        result = enhancement.process(black_image, "Log Transform", {})
        assert result is black_image

    def test_retinex(self, bgr_image_100x100):
        params = {"enhancement_Sigma": 30}
        result = enhancement.process(bgr_image_100x100, "Single-Scale Retinex", params)
        assert result.shape == bgr_image_100x100.shape

    def test_unsharp_masking(self, bgr_image_100x100):
        params = {"enhancement_Kernel_Size": 5, "enhancement_Alpha_x10": 15}
        result = enhancement.process(bgr_image_100x100, "Unsharp Masking", params)
        assert result.shape == bgr_image_100x100.shape

    def test_unsharp_even_kernel(self, bgr_image_100x100):
        """Even kernel should be auto-corrected."""
        params = {"enhancement_Kernel_Size": 4, "enhancement_Alpha_x10": 10}
        result = enhancement.process(bgr_image_100x100, "Unsharp Masking", params)
        assert result.shape == bgr_image_100x100.shape

    def test_unknown_returns_original(self, bgr_image_100x100):
        result = enhancement.process(bgr_image_100x100, "FakeFilter", {})
        assert result is bgr_image_100x100
