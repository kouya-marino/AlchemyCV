# AlchemyCV - Advanced Computer Vision Tool

AlchemyCV is a powerful desktop application built with Python. It provides a comprehensive suite of tools for real-time image processing, enhancement, filtering, and analysis.

![AlchemyCV Screenshot]  <!-- Replace with a real screenshot URL -->

## Features

- **Multi-Stage Processing Pipeline:** Apply filters in a logical order: Pre-processing, Enhancement, Frequency Filtering, Masking, and Refinement.
- **Rich Filter Library:** Includes Gaussian/Median/Bilateral blurs, Histogram Equalization, CLAHE, Fourier transforms (LPF/HPF), color space filtering (HSV, Lab), thresholding, and more.
- **Advanced Masking:** Generate binary masks from color or grayscale ranges, or use edge detection algorithms like Canny, Sobel, and Prewitt.
- **Contour Analysis:** Automatically detect, count, and draw contours on objects in the image based on area.
- **Interactive UI:**
    - Real-time parameter adjustment with sliders and dropdowns.
    - Zoom and Pan the image display with mouse controls.
    - Status bar showing image dimensions and mouse coordinates.
    - Informative tooltips for key controls.
- **Session Management:** Save and load your complex filter settings to a JSON file.

## Installation

AlchemyCV is designed to be run from a Python virtual environment.

**1. Clone the Repository:**
```bash
git clone https://github.com/kouya-marino/AlchemyCV.git
cd AlchemyCV
