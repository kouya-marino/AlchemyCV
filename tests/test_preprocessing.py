"""Unit tests for pipeline.preprocessing."""

import numpy as np

from alchemycv.pipeline import preprocessing


class TestPreprocessing:
    def test_none_returns_same(self, bgr_image_100x100):
        result = preprocessing.process(bgr_image_100x100, "None", {})
        assert result is bgr_image_100x100

    def test_gaussian_blur(self, bgr_image_100x100):
        params = {"preproc_Kernel_Size": 5}
        result = preprocessing.process(bgr_image_100x100, "Gaussian Blur", params)
        assert result.shape == bgr_image_100x100.shape
        assert result.dtype == np.uint8

    def test_gaussian_blur_even_kernel_corrected(self, bgr_image_100x100):
        """Even kernel sizes should be auto-corrected to odd."""
        params = {"preproc_Kernel_Size": 4}
        result = preprocessing.process(bgr_image_100x100, "Gaussian Blur", params)
        assert result.shape == bgr_image_100x100.shape

    def test_median_blur(self, bgr_image_100x100):
        params = {"preproc_Kernel_Size": 3}
        result = preprocessing.process(bgr_image_100x100, "Median Blur", params)
        assert result.shape == bgr_image_100x100.shape
        assert result.dtype == np.uint8

    def test_bilateral_filter(self, bgr_image_100x100):
        params = {"preproc_Diameter": 5, "preproc_Sigma_Color": 50, "preproc_Sigma_Space": 50}
        result = preprocessing.process(bgr_image_100x100, "Bilateral Filter", params)
        assert result.shape == bgr_image_100x100.shape

    def test_unknown_filter_returns_original(self, bgr_image_100x100):
        result = preprocessing.process(bgr_image_100x100, "NonExistent", {})
        assert result is bgr_image_100x100

    def test_blur_actually_smooths(self, bgr_image_100x100):
        """Blurred image should have less variance than original."""
        params = {"preproc_Kernel_Size": 15}
        result = preprocessing.process(bgr_image_100x100, "Gaussian Blur", params)
        assert np.var(result.astype(float)) < np.var(bgr_image_100x100.astype(float))

    def test_grayscale_input(self, gray_image_100x100):
        """Preprocessing should handle grayscale input without crashing."""
        params = {"preproc_Kernel_Size": 5}
        result = preprocessing.process(gray_image_100x100, "Gaussian Blur", params)
        assert result.shape == gray_image_100x100.shape

    def test_kernel_size_1(self, bgr_image_100x100):
        """Kernel size 1 should effectively be a no-op."""
        params = {"preproc_Kernel_Size": 1}
        result = preprocessing.process(bgr_image_100x100, "Gaussian Blur", params)
        np.testing.assert_array_equal(result, bgr_image_100x100)
