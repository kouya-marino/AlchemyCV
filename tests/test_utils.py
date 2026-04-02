"""Unit tests for utils (image I/O)."""

import numpy as np
import pytest

from alchemycv.utils import load_image, save_image


class TestImageIO:
    def test_save_and_load_png(self, tmp_path, bgr_image_100x100):
        filepath = str(tmp_path / "test.png")
        save_image(bgr_image_100x100, filepath)
        loaded = load_image(filepath)
        assert loaded is not None
        assert loaded.shape == bgr_image_100x100.shape
        np.testing.assert_array_equal(loaded, bgr_image_100x100)

    def test_save_and_load_jpg(self, tmp_path, bgr_image_100x100):
        filepath = str(tmp_path / "test.jpg")
        save_image(bgr_image_100x100, filepath)
        loaded = load_image(filepath)
        assert loaded is not None
        assert loaded.shape == bgr_image_100x100.shape
        # JPG is lossy, so just check shape, not exact pixels

    def test_save_and_load_bmp(self, tmp_path, bgr_image_100x100):
        filepath = str(tmp_path / "test.bmp")
        save_image(bgr_image_100x100, filepath)
        loaded = load_image(filepath)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, bgr_image_100x100)

    def test_load_nonexistent_returns_none(self):
        result = load_image("/nonexistent/path/image.png")
        assert result is None

    def test_save_invalid_extension(self, tmp_path, bgr_image_100x100):
        """Unsupported extension should raise an error."""
        filepath = str(tmp_path / "test.xyz")
        with pytest.raises(Exception):
            save_image(bgr_image_100x100, filepath)
