# AlchemyCV - Advanced Computer Vision Tool

AlchemyCV is a powerful desktop application built with Python. It provides a comprehensive suite of tools for real-time image processing, enhancement, filtering, and analysis.

![AlchemyCV Screenshot](https://raw.githubusercontent.com/kouya-marino/AlchemyCV/main/application.png)

## Features

-   **Multi-Stage Processing Pipeline:** Apply filters in a logical order: Pre-processing, Enhancement, Frequency Filtering, Masking, and Refinement.
-   **Rich Filter Library:** Includes Gaussian/Median/Bilateral blurs, Histogram Equalization, CLAHE, Fourier transforms (LPF/HPF), color space filtering (HSV, Lab), thresholding, and more.
-   **Advanced Masking:** Generate binary masks from color or grayscale ranges, or use edge detection algorithms like Canny, Sobel, and Prewitt.
-   **Contour Analysis:** Automatically detect, count, and draw contours on objects in the image based on area.
-   **Interactive UI:**
    -   Real-time parameter adjustment with sliders and dropdowns.
    -   Zoom and Pan the image display with mouse controls.
    -   Status bar showing image dimensions and mouse coordinates.
    -   Informative tooltips for key controls.
-   **Session Management:** Save and load your complex filter settings to a JSON file.

## Installation

The easiest way to install AlchemyCV is with `pip` from the Python Package Index (PyPI).

```bash
pip install alchemycv
```

## Usage

After installation, the application can be launched by simply running the following command in your terminal:

```bash
alchemycv
```

---

## For Developers (Installation from Source)

If you wish to modify the code or contribute to the project, you can install it from source.

**1. Clone the Repository:**

```bash
git clone https://github.com/kouya-marino/AlchemyCV.git
cd AlchemyCV
```

**2. Create and Activate a Virtual Environment:**

It is highly recommended to create a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

**3. Install Dependencies:**

The required libraries are listed in the package configuration and can be installed with `pip`.

```bash
pip install -r requirements.txt
```
*(Note: You will need to create a `requirements.txt` file from your `pyproject.toml` or install them manually if you choose this route).*

**4. Run the Application:**

```bash
python -m src.alchemycv.app
```