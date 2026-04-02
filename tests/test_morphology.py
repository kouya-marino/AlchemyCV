"""Unit tests for pipeline.morphology."""

import numpy as np
import pytest

from alchemycv.pipeline import morphology


@pytest.fixture
def binary_mask():
    """Binary mask with a white square in the center."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    return mask


class TestMorphology:
    @pytest.mark.parametrize(
        "operation",
        [
            "Dilate",
            "Erode",
            "Open",
            "Close",
            "Gradient",
            "Top Hat",
            "Black Hat",
        ],
    )
    def test_all_operations(self, binary_mask, operation):
        params = {
            "Morph Operation": operation,
            "Morph Kernel Shape": "Rectangle",
            "Morph Kernel Size": 5,
            "Morph Iterations": 1,
        }
        result = morphology.process(binary_mask, params)
        assert result.shape == binary_mask.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("shape", ["Rectangle", "Ellipse", "Cross"])
    def test_all_kernel_shapes(self, binary_mask, shape):
        params = {
            "Morph Operation": "Dilate",
            "Morph Kernel Shape": shape,
            "Morph Kernel Size": 5,
            "Morph Iterations": 1,
        }
        result = morphology.process(binary_mask, params)
        assert result.shape == binary_mask.shape

    def test_dilate_expands(self, binary_mask):
        params = {
            "Morph Operation": "Dilate",
            "Morph Kernel Shape": "Rectangle",
            "Morph Kernel Size": 5,
            "Morph Iterations": 1,
        }
        result = morphology.process(binary_mask, params)
        assert np.sum(result > 0) > np.sum(binary_mask > 0)

    def test_erode_shrinks(self, binary_mask):
        params = {
            "Morph Operation": "Erode",
            "Morph Kernel Shape": "Rectangle",
            "Morph Kernel Size": 5,
            "Morph Iterations": 1,
        }
        result = morphology.process(binary_mask, params)
        assert np.sum(result > 0) < np.sum(binary_mask > 0)

    def test_even_kernel_corrected(self, binary_mask):
        params = {
            "Morph Operation": "Dilate",
            "Morph Kernel Shape": "Rectangle",
            "Morph Kernel Size": 4,
            "Morph Iterations": 1,
        }
        result = morphology.process(binary_mask, params)
        assert result.shape == binary_mask.shape

    def test_multiple_iterations(self, binary_mask):
        params1 = {
            "Morph Operation": "Erode",
            "Morph Kernel Shape": "Rectangle",
            "Morph Kernel Size": 3,
            "Morph Iterations": 1,
        }
        params3 = {**params1, "Morph Iterations": 3}
        r1 = morphology.process(binary_mask, params1)
        r3 = morphology.process(binary_mask, params3)
        assert np.sum(r3 > 0) < np.sum(r1 > 0)
