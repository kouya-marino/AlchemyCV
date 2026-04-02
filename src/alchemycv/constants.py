"""Constants and data definitions for AlchemyCV."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any

# ---------------------------------------------------------------------------
# Filter / stage data structures
# ---------------------------------------------------------------------------
# These dicts define available filters, their parameter specs, and defaults.
# Pipeline functions use the 'type' key to dispatch; UI code uses the rest
# to build dynamic controls.  Keeping them here avoids circular imports and
# gives both layers a single source of truth.
# ---------------------------------------------------------------------------

PREPROCESSING_DATA: dict[str, dict[str, Any]] = {
    "None": {"type": "none"},
    "Gaussian Blur": {
        "type": "gaussian",
        "params": {"Kernel Size": {"range": (1, 51), "default": 5}},
    },
    "Median Blur": {
        "type": "median",
        "params": {"Kernel Size": {"range": (1, 51), "default": 5}},
    },
    "Bilateral Filter": {
        "type": "bilateral",
        "params": {
            "Diameter": {"range": (1, 25), "default": 9},
            "Sigma Color": {"range": (1, 150), "default": 75},
            "Sigma Space": {"range": (1, 150), "default": 75},
        },
    },
}

ENHANCEMENT_DATA: dict[str, dict[str, Any]] = {
    "None": {"type": "none"},
    "Histogram Equalization": {"type": "hist_equal"},
    "CLAHE": {
        "type": "clahe",
        "params": {
            "Clip Limit": {"range": (1, 40), "default": 2},
            "Tile Grid Size": {"range": (2, 32), "default": 8},
        },
    },
    "Contrast Stretching": {"type": "contrast_stretch"},
    "Gamma Correction": {
        "type": "gamma",
        "params": {"Gamma x100": {"range": (10, 300), "default": 100}},
    },
    "Log Transform": {"type": "log"},
    "Single-Scale Retinex": {
        "type": "ssr",
        "params": {"Sigma": {"range": (1, 250), "default": 30}},
    },
    "Unsharp Masking": {
        "type": "unsharp",
        "params": {
            "Kernel Size": {"range": (1, 51), "default": 5},
            "Alpha x10": {"range": (1, 50), "default": 15},
        },
    },
}

FREQUENCY_DATA: dict[str, dict[str, Any]] = {
    "None": {"type": "none"},
    "Ideal Low-Pass": {
        "type": "ilpf",
        "params": {"Cutoff Freq (D0)": {"range": (1, 250), "default": 30}},
    },
    "Gaussian Low-Pass": {
        "type": "glpf",
        "params": {"Cutoff Freq (D0)": {"range": (1, 250), "default": 30}},
    },
    "Butterworth Low-Pass": {
        "type": "blpf",
        "params": {
            "Cutoff Freq (D0)": {"range": (1, 250), "default": 30},
            "Order (n)": {"range": (1, 10), "default": 2},
        },
    },
    "Ideal High-Pass": {
        "type": "ihpf",
        "params": {"Cutoff Freq (D0)": {"range": (1, 250), "default": 30}},
    },
    "Gaussian High-Pass": {
        "type": "ghpf",
        "params": {"Cutoff Freq (D0)": {"range": (1, 250), "default": 30}},
    },
    "Butterworth High-Pass": {
        "type": "bhpf",
        "params": {
            "Cutoff Freq (D0)": {"range": (1, 250), "default": 30},
            "Order (n)": {"range": (1, 10), "default": 2},
        },
    },
}

CHANNEL_DATA: dict[str, dict[str, Any]] = {
    "Grayscale": {"code": cv2.COLOR_BGR2GRAY, "channels": ["Intensity"]},
    "RGB/BGR": {"code": None, "channels": ["B", "G", "R"]},
    "HSV": {"code": cv2.COLOR_BGR2HSV, "channels": ["H", "S", "V"]},
    "HLS": {"code": cv2.COLOR_BGR2HLS, "channels": ["H", "L", "S"]},
    "Lab": {"code": cv2.COLOR_BGR2LAB, "channels": ["L", "a", "b"]},
    "YCrCb": {"code": cv2.COLOR_BGR2YCrCb, "channels": ["Y", "Cr", "Cb"]},
}

FILTER_DATA: dict[str, dict[str, Any]] = {
    "RGB/BGR (Color Filter)": {
        "type": "color",
        "channels": ["B", "G", "R"],
        "ranges": [(0, 255), (0, 255), (0, 255)],
    },
    "HSV": {
        "type": "color",
        "channels": ["H", "S", "V"],
        "ranges": [(0, 179), (0, 255), (0, 255)],
    },
    "HLS": {
        "type": "color",
        "channels": ["H", "L", "S"],
        "ranges": [(0, 179), (0, 255), (0, 255)],
    },
    "Lab": {
        "type": "color",
        "channels": ["L", "a", "b"],
        "ranges": [(0, 255), (0, 255), (0, 255)],
    },
    "YCrCb": {
        "type": "color",
        "channels": ["Y", "Cr", "Cb"],
        "ranges": [(0, 255), (0, 255), (0, 255)],
    },
    "Grayscale Range": {
        "type": "grayscale_range",
        "params": {
            "Min Value": {"range": (0, 255), "default": 0},
            "Max Value": {"range": (0, 255), "default": 127},
        },
    },
    "Adaptive Threshold": {
        "type": "adaptive_thresh",
        "params": {
            "Adaptive Method": {
                "options": ["Mean C", "Gaussian C"],
                "default": "Gaussian C",
            },
            "Threshold Type": {
                "options": ["Binary", "Binary Inverted"],
                "default": "Binary",
            },
            "Block Size": {"range": (3, 55), "default": 11},
            "C (Constant)": {"range": (-30, 30), "default": 2},
        },
    },
    "Otsu's Binarization": {
        "type": "otsu",
        "params": {
            "Threshold Type": {
                "options": ["Binary", "Binary Inverted"],
                "default": "Binary",
            }
        },
    },
}

EDGE_DETECTION_DATA: dict[str, dict[str, Any]] = {
    "Canny": {
        "type": "canny",
        "params": {
            "Threshold 1": {"range": (0, 255), "default": 50},
            "Threshold 2": {"range": (0, 255), "default": 150},
        },
    },
    "Sobel": {
        "type": "sobel",
        "params": {
            "Kernel Size": {"range": (1, 31), "default": 3},
            "Direction": {
                "options": ["X", "Y", "Magnitude"],
                "default": "Magnitude",
            },
        },
    },
    "Prewitt": {
        "type": "prewitt",
        "params": {
            "Direction": {
                "options": ["X", "Y", "Magnitude"],
                "default": "Magnitude",
            }
        },
    },
    "Roberts": {
        "type": "roberts",
        "params": {
            "Direction": {
                "options": ["X", "Y", "Magnitude"],
                "default": "Magnitude",
            }
        },
    },
}

# Morphological operation / kernel shape look-ups
MORPH_OPERATIONS = ["Dilate", "Erode", "Open", "Close", "Gradient", "Top Hat", "Black Hat"]
MORPH_KERNEL_SHAPES = ["Rectangle", "Ellipse", "Cross"]

MORPH_OP_MAP: dict[str, int] = {
    "Dilate": cv2.MORPH_DILATE,
    "Erode": cv2.MORPH_ERODE,
    "Open": cv2.MORPH_OPEN,
    "Close": cv2.MORPH_CLOSE,
    "Gradient": cv2.MORPH_GRADIENT,
    "Top Hat": cv2.MORPH_TOPHAT,
    "Black Hat": cv2.MORPH_BLACKHAT,
}

MORPH_SHAPE_MAP: dict[str, int] = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS,
}

# Color-space conversion codes for mask generation
COLOR_CONV_MAP: dict[str, int] = {
    "RGB/BGR (Color Filter)": -1,
    "HSV": cv2.COLOR_BGR2HSV,
    "HLS": cv2.COLOR_BGR2HLS,
    "Lab": cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
}

# Display modes
DISPLAY_MODES = [
    "Final Result",
    "Binary Mask",
    "Enhanced Image",
    "Pre-processed Image",
    "Extracted Channel",
]

# Supported image file types for open dialog
IMAGE_FILETYPES = [
    ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
    ("All files", "*.*"),
]

# Supported save formats
SAVE_FILETYPES = [
    ("PNG Image", "*.png"),
    ("JPEG Image", "*.jpg"),
    ("BMP Image", "*.bmp"),
    ("TIFF Image", "*.tiff"),
    ("WebP Image", "*.webp"),
    ("All Files", "*.*"),
]
