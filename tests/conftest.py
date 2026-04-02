"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def bgr_image_100x100():
    """100x100 BGR uint8 image with varied content."""
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def gray_image_100x100():
    """100x100 grayscale uint8 image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def bgr_image_1x1():
    """Tiny 1x1 BGR image for edge-case testing."""
    return np.array([[[128, 64, 32]]], dtype=np.uint8)


@pytest.fixture
def bgr_image_small():
    """10x10 BGR image for fast tests."""
    np.random.seed(99)
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)


@pytest.fixture
def black_image():
    """100x100 all-black BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """100x100 all-white BGR image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)
