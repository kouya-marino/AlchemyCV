"""Unit tests for pipeline.masking."""

import numpy as np

from alchemycv.pipeline import masking


class TestMaskGeneration:
    def test_grayscale_range(self, bgr_image_100x100):
        params = {"filter_Min_Value": 50, "filter_Max_Value": 200}
        mask, otsu = masking.process(bgr_image_100x100, "Grayscale Range", params)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert otsu is None
        assert set(np.unique(mask)).issubset({0, 255})

    def test_grayscale_range_min_gt_max(self, bgr_image_100x100):
        """When min > max, min should be clamped to max."""
        params = {"filter_Min_Value": 200, "filter_Max_Value": 50}
        mask, _ = masking.process(bgr_image_100x100, "Grayscale Range", params)
        assert mask.shape == (100, 100)

    def test_adaptive_threshold(self, bgr_image_100x100):
        params = {
            "filter_Adaptive_Method": "Gaussian C",
            "filter_Threshold_Type": "Binary",
            "filter_Block_Size": 11,
            "filter_C_(Constant)": 2,
        }
        mask, otsu = masking.process(bgr_image_100x100, "Adaptive Threshold", params)
        assert mask.shape == (100, 100)
        assert otsu is None

    def test_adaptive_threshold_even_block(self, bgr_image_100x100):
        """Even block size should be auto-corrected to odd."""
        params = {
            "filter_Adaptive_Method": "Mean C",
            "filter_Threshold_Type": "Binary Inverted",
            "filter_Block_Size": 10,
            "filter_C_(Constant)": 0,
        }
        mask, _ = masking.process(bgr_image_100x100, "Adaptive Threshold", params)
        assert mask.shape == (100, 100)

    def test_otsu(self, bgr_image_100x100):
        params = {"otsu_Threshold_Type": "Binary"}
        mask, threshold = masking.process(bgr_image_100x100, "Otsu's Binarization", params)
        assert mask.shape == (100, 100)
        assert threshold is not None
        assert 0 <= threshold <= 255

    def test_otsu_inverted(self, bgr_image_100x100):
        params = {"otsu_Threshold_Type": "Binary Inverted"}
        mask, _ = masking.process(bgr_image_100x100, "Otsu's Binarization", params)
        assert mask.shape == (100, 100)

    def test_color_filter_hsv(self, bgr_image_100x100):
        params = {
            "color_H_min": 0,
            "color_H_max": 179,
            "color_S_min": 0,
            "color_S_max": 255,
            "color_V_min": 0,
            "color_V_max": 255,
        }
        mask, _ = masking.process(bgr_image_100x100, "HSV", params)
        assert mask.shape == (100, 100)
        # Full range should select everything
        assert np.all(mask == 255)

    def test_color_filter_bgr(self, bgr_image_100x100):
        params = {
            "color_B_min": 0,
            "color_B_max": 50,
            "color_G_min": 0,
            "color_G_max": 50,
            "color_R_min": 0,
            "color_R_max": 50,
        }
        mask, _ = masking.process(bgr_image_100x100, "RGB/BGR (Color Filter)", params)
        assert mask.shape == (100, 100)

    def test_color_filter_on_grayscale(self, gray_image_100x100):
        """Color filter on grayscale input should return zeros."""
        params = {
            "color_B_min": 0,
            "color_B_max": 255,
            "color_G_min": 0,
            "color_G_max": 255,
            "color_R_min": 0,
            "color_R_max": 255,
        }
        mask, _ = masking.process(gray_image_100x100, "RGB/BGR (Color Filter)", params)
        assert np.all(mask == 0)

    def test_unknown_filter(self, bgr_image_100x100):
        mask, _ = masking.process(bgr_image_100x100, "NonExistent", {})
        assert mask.shape == (100, 100)
        assert np.all(mask == 0)
