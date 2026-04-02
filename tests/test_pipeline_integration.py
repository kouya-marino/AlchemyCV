"""Integration tests for the full processing pipeline."""

import numpy as np
import pytest

from alchemycv.pipeline import engine


def _base_sel(**overrides):
    """Return a default selector dict with optional overrides."""
    defaults = {
        "preproc": "None",
        "enhancement": "None",
        "frequency": "None",
        "channel_enabled": False,
        "color_space": "Grayscale",
        "channel": "Intensity",
        "filter": "Grayscale Range",
        "edge_enabled": False,
        "edge_filter": "Canny",
        "morph_enabled": False,
        "contours_enabled": False,
        "draw_contours": True,
        "min_area": 50,
        "max_area": 1_000_000,
        "display_mode": "Final Result",
    }
    defaults.update(overrides)
    return defaults


class TestFullPipeline:
    def test_basic_run(self, bgr_image_100x100):
        sel = _base_sel()
        params = {"filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None
        assert "final" in result
        assert "mask" in result
        assert result["final"].shape == bgr_image_100x100.shape

    def test_all_stages_enabled(self, bgr_image_100x100):
        sel = _base_sel(
            preproc="Gaussian Blur",
            enhancement="CLAHE",
            frequency="Gaussian Low-Pass",
            edge_enabled=True,
            edge_filter="Sobel",
            morph_enabled=True,
            contours_enabled=True,
        )
        params = {
            "preproc_Kernel_Size": 5,
            "enhancement_Clip_Limit": 2,
            "enhancement_Tile_Grid_Size": 8,
            "frequency_Cutoff_Freq_(D0)": 30,
            "edge_Kernel_Size": 3,
            "edge_Direction": "Magnitude",
            "Morph Operation": "Close",
            "Morph Kernel Shape": "Ellipse",
            "Morph Kernel Size": 5,
            "Morph Iterations": 2,
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None
        assert result["object_count"] is not None

    def test_channel_extraction_with_masking(self, bgr_image_100x100):
        sel = _base_sel(
            channel_enabled=True,
            color_space="HSV",
            channel="S",
            filter="Grayscale Range",
        )
        params = {"filter_Min_Value": 50, "filter_Max_Value": 200}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None
        assert result["extracted_channel"] is not None
        assert result["extracted_channel"].ndim == 2

    def test_color_filter_mask(self, bgr_image_100x100):
        sel = _base_sel(filter="HSV")
        params = {
            "color_H_min": 0,
            "color_H_max": 179,
            "color_S_min": 0,
            "color_S_max": 255,
            "color_V_min": 100,
            "color_V_max": 255,
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_otsu_returns_threshold(self, bgr_image_100x100):
        sel = _base_sel(filter="Otsu's Binarization")
        params = {"otsu_Threshold_Type": "Binary"}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None
        assert result["otsu_threshold"] is not None

    def test_adaptive_threshold(self, bgr_image_100x100):
        sel = _base_sel(filter="Adaptive Threshold")
        params = {
            "filter_Adaptive_Method": "Gaussian C",
            "filter_Threshold_Type": "Binary",
            "filter_Block_Size": 11,
            "filter_C_(Constant)": 2,
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    @pytest.mark.parametrize("edge_filter", ["Canny", "Sobel", "Prewitt", "Roberts"])
    def test_all_edge_detectors_in_pipeline(self, bgr_image_100x100, edge_filter):
        sel = _base_sel(edge_enabled=True, edge_filter=edge_filter)
        params = {
            "edge_Threshold_1": 50,
            "edge_Threshold_2": 150,
            "edge_Kernel_Size": 3,
            "edge_Direction": "Magnitude",
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    @pytest.mark.parametrize(
        "freq_filter",
        [
            "Ideal Low-Pass",
            "Gaussian Low-Pass",
            "Butterworth Low-Pass",
            "Ideal High-Pass",
            "Gaussian High-Pass",
            "Butterworth High-Pass",
        ],
    )
    def test_all_frequency_filters_in_pipeline(self, bgr_image_100x100, freq_filter):
        sel = _base_sel(frequency=freq_filter, filter="Grayscale Range")
        params = {
            "frequency_Cutoff_Freq_(D0)": 30,
            "frequency_Order_(n)": 2,
            "filter_Min_Value": 0,
            "filter_Max_Value": 200,
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None


class TestEdgeCases:
    def test_single_pixel_image(self):
        img = np.array([[[128, 64, 32]]], dtype=np.uint8)
        sel = _base_sel()
        params = {"filter_Min_Value": 0, "filter_Max_Value": 255}
        result = engine.run(img, sel, params)
        assert result is not None
        assert result["final"].shape[:2] == (1, 1)

    def test_grayscale_input_to_pipeline(self):
        """Pipeline expects BGR but should handle edge cases gracefully."""
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        # Convert to 3-channel to match expected input
        bgr = np.stack([gray, gray, gray], axis=-1)
        sel = _base_sel()
        params = {"filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(bgr, sel, params)
        assert result is not None

    def test_large_image(self):
        """Verify pipeline handles larger images without error."""
        img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        sel = _base_sel(preproc="Median Blur")
        params = {"preproc_Kernel_Size": 3, "filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(img, sel, params)
        assert result is not None
        assert result["final"].shape == (500, 500, 3)

    def test_all_black_image(self, black_image):
        sel = _base_sel()
        params = {"filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(black_image, sel, params)
        assert result is not None

    def test_all_white_image(self, white_image):
        sel = _base_sel()
        params = {"filter_Min_Value": 0, "filter_Max_Value": 255}
        result = engine.run(white_image, sel, params)
        assert result is not None


class TestParameterBoundaries:
    def test_min_kernel_size(self, bgr_image_100x100):
        sel = _base_sel(preproc="Gaussian Blur")
        params = {"preproc_Kernel_Size": 1, "filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_max_kernel_size(self, bgr_image_100x100):
        sel = _base_sel(preproc="Gaussian Blur")
        params = {"preproc_Kernel_Size": 51, "filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_extreme_gamma(self, bgr_image_100x100):
        sel = _base_sel(enhancement="Gamma Correction")
        params = {"enhancement_Gamma_x100": 10, "filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_high_gamma(self, bgr_image_100x100):
        sel = _base_sel(enhancement="Gamma Correction")
        params = {"enhancement_Gamma_x100": 300, "filter_Min_Value": 0, "filter_Max_Value": 127}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_extreme_cutoff_frequency(self, bgr_image_100x100):
        sel = _base_sel(frequency="Butterworth Low-Pass")
        params = {
            "frequency_Cutoff_Freq_(D0)": 250,
            "frequency_Order_(n)": 10,
            "filter_Min_Value": 0,
            "filter_Max_Value": 127,
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_canny_zero_thresholds(self, bgr_image_100x100):
        sel = _base_sel(edge_enabled=True, edge_filter="Canny")
        params = {"edge_Threshold_1": 0, "edge_Threshold_2": 0}
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None

    def test_max_morph_iterations(self, bgr_image_100x100):
        sel = _base_sel(edge_enabled=True, edge_filter="Canny", morph_enabled=True)
        params = {
            "edge_Threshold_1": 50,
            "edge_Threshold_2": 150,
            "Morph Operation": "Dilate",
            "Morph Kernel Shape": "Rectangle",
            "Morph Kernel Size": 3,
            "Morph Iterations": 20,
        }
        result = engine.run(bgr_image_100x100, sel, params)
        assert result is not None
