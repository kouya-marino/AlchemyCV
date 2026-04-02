"""Unit tests for pipeline.contours."""

import numpy as np
import pytest

from alchemycv.pipeline import contours


@pytest.fixture
def mask_with_objects():
    """Binary mask with 3 distinct white rectangles."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:40, 10:40] = 255  # ~900 px area
    mask[60:90, 60:90] = 255  # ~900 px area
    mask[120:190, 120:190] = 255  # ~4900 px area
    return mask


class TestContours:
    def test_finds_objects(self, mask_with_objects):
        draw_img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, count = contours.process(mask_with_objects, draw_img, min_area=0, max_area=1_000_000)
        assert count == 3

    def test_area_filtering_min(self, mask_with_objects):
        draw_img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, count = contours.process(mask_with_objects, draw_img, min_area=1000, max_area=1_000_000)
        assert count == 1  # Only the large rectangle

    def test_area_filtering_max(self, mask_with_objects):
        draw_img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, count = contours.process(mask_with_objects, draw_img, min_area=0, max_area=1000)
        assert count == 2  # Only the two small rectangles

    def test_no_draw(self, mask_with_objects):
        draw_img = np.zeros((200, 200, 3), dtype=np.uint8)
        original = draw_img.copy()
        result, count = contours.process(mask_with_objects, draw_img, draw=False)
        np.testing.assert_array_equal(result, original)

    def test_draw_on_grayscale(self, mask_with_objects):
        """Drawing on grayscale should auto-convert to BGR."""
        gray_img = np.zeros((200, 200), dtype=np.uint8)
        result, count = contours.process(mask_with_objects, gray_img, draw=True)
        assert result.ndim == 3  # Should be converted to BGR

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        draw_img = np.zeros((50, 50, 3), dtype=np.uint8)
        result, count = contours.process(mask, draw_img)
        assert count == 0

    def test_draw_changes_image(self, mask_with_objects):
        draw_img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, _ = contours.process(mask_with_objects, draw_img, draw=True)
        assert np.any(result != 0)
