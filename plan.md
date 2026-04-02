# AlchemyCV — Improvement & Future Plan

> **Version:** 1.0.0 | **Date:** 2026-04-03
> **Current state:** Monolithic single-file app (`app.py`, 1191 lines, 50+ methods in one class)
> **Stack:** Python 3.8+ / Tkinter / OpenCV / NumPy / Pillow / Matplotlib

---

## Phase 1: Code Quality & Foundation

**Goal:** Make the codebase testable, maintainable, and contributor-friendly.

### 1.1 Modularize `app.py`

Split the monolithic `AdvancedFilterApp` class into focused modules:

```
src/alchemycv/
├── app.py                  # Entry point, window setup, main loop
├── ui/
│   ├── main_window.py      # Top-level layout, menu bar, toolbar
│   ├── control_panel.py    # Left panel with all filter controls
│   ├── canvas.py           # Image canvas, zoom, pan, display
│   └── widgets.py          # Reusable slider+entry, dynamic panels
├── pipeline/
│   ├── engine.py           # Pipeline orchestrator (runs stages in order)
│   ├── preprocessing.py    # Gaussian, Median, Bilateral blur
│   ├── enhancement.py      # CLAHE, Gamma, Retinex, Unsharp, etc.
│   ├── frequency.py        # DFT-based LPF/HPF filters
│   ├── channels.py         # Color space conversion, channel extraction
│   ├── masking.py          # Color range, grayscale range, adaptive, Otsu
│   ├── edges.py            # Canny, Sobel, Prewitt, Roberts
│   ├── morphology.py       # Dilate, Erode, Open, Close, etc.
│   └── contours.py         # Contour detection, area filtering, drawing
├── state/
│   ├── manager.py          # Undo/redo stack, state capture/restore
│   └── settings.py         # Save/load JSON settings, session persistence
└── utils.py                # Image I/O (Unicode paths), format helpers
```

**Why:** The current single-class design makes it impossible to unit-test filters independently, reuse processing logic outside the GUI, or onboard new contributors without understanding 1200 lines at once.

### 1.2 Decouple Processing from UI

- Processing functions should accept NumPy arrays + parameter dicts, return NumPy arrays
- No tkinter imports in `pipeline/` modules
- UI reads parameters from widgets, passes plain dicts to pipeline
- Enables headless/CLI usage and testing without a display

### 1.3 Add Type Hints

- Add type annotations to all public methods and function signatures
- Use `numpy.typing.NDArray` for image parameters
- Add `py.typed` marker for downstream consumers

### 1.4 Replace `print()` with `logging`

- Use Python `logging` module with configurable levels
- `DEBUG` for filter parameter values, `INFO` for pipeline stages, `WARNING` for fallbacks, `ERROR` for failures

### 1.5 Fix Hardcoded Magic Strings

- Define filter names, color spaces, and parameter keys as constants or enums
- Example: `class FilterType(Enum): GAUSSIAN_BLUR = "Gaussian Blur"` etc.
- Eliminates silent bugs from typos in string comparisons

### 1.6 Input Validation

- Enforce odd kernel sizes before passing to OpenCV (currently causes crashes)
- Validate min < max for contour area ranges
- Clamp parameter values to valid ranges in entry widgets
- Show user-friendly error messages instead of silent failures

---

## Phase 2: Testing & CI/CD

**Goal:** Automated quality gates to catch regressions.

### 2.1 Unit Tests (pytest)

- Test each processing function in isolation (input image + params -> expected output)
- Snapshot tests: compare filter outputs against reference images (pixel tolerance)
- State management tests: capture, restore, undo, redo cycles
- Settings serialization round-trip tests

### 2.2 Integration Tests

- Full pipeline tests: load image -> apply filters -> verify output dimensions/channels
- Edge cases: empty image, single-pixel image, very large image, corrupted file
- Parameter boundary tests: min/max values for every slider

### 2.3 CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  lint:    ruff check + ruff format --check
  type:    mypy --strict src/
  test:    pytest --cov=alchemycv -x
  build:   pip install . && alchemycv --version
```

### 2.4 Automated PyPI Publishing

- GitHub Actions workflow triggered on version tags (`v*.*.*`)
- Build sdist + wheel, publish to PyPI
- Auto-generate release notes from commit history

---

## Phase 3: UX Improvements

**Goal:** Make the app more intuitive and responsive.

### 3.1 Tooltips

- Add hover tooltips to every filter parameter explaining what it does
- Example: "Sigma Color — Higher values mix colors from farther pixels (range: 1-150)"

### 3.2 Per-Stage Reset Buttons

- "Reset" button on each LabelFrame to reset only that stage's parameters
- Currently only "Reset All" exists, which is too destructive

### 3.3 Progress Indicator

- Show a progress bar or spinner during heavy operations (FFT on large images)
- Replace the bare `_processing` flag with proper feedback

### 3.4 Before/After Comparison

- Split-view mode: drag a divider to compare original vs. processed
- Toggle with keyboard shortcut (e.g., `Space` to flip between original and result)

### 3.5 Dark Mode

- Add a dark theme option using ttk styles
- Persist theme preference in settings

### 3.6 Better Zoom & Pan

- Zoom to mouse cursor position (not center)
- Minimap/navigator for large zoomed-in images
- Pixel-level inspector on hover (show RGB values)

---

## Phase 4: New Features

**Goal:** Expand capabilities beyond current filter set.

### 4.1 Batch Processing

- Apply current pipeline configuration to an entire folder of images
- Output naming patterns (e.g., `{original}_processed.png`)
- Progress bar with cancel support
- CLI interface: `alchemycv --batch --input ./photos --output ./results --config pipeline.json`

### 4.2 Video Support

- Open video files (mp4, avi, mkv) and process frame-by-frame
- Live webcam feed with real-time filter application
- Export processed video with codec selection
- Frame scrubber to preview specific frames before full export

### 4.3 More Filters & Algorithms

**Segmentation:**
- Watershed segmentation
- GrabCut (interactive foreground extraction)
- K-means color clustering

**Denoising:**
- Non-local means denoising (`cv2.fastNlMeansDenoisingColored`)
- Wiener filter

**Feature Detection:**
- Harris corner detection with visualization
- ORB / SIFT keypoint detection and matching
- Hough line and circle detection

**AI-Powered (Optional, via OpenCV DNN or ONNX):**
- Background removal (U2-Net or similar lightweight model)
- Super-resolution (ESPCN, FSRCNN)
- Style transfer

### 4.4 Region of Interest (ROI)

- Draw rectangles, polygons, or freehand regions on the image
- Apply filters only within selected ROI
- Multiple ROI support with different filter configurations

### 4.5 Layer System

- Stack multiple filter configurations as layers
- Blend modes between layers (multiply, screen, overlay, etc.)
- Opacity control per layer
- Reorder layers via drag-and-drop

### 4.6 Histogram Tools

- Live histogram overlay on the image canvas
- Per-channel histogram with adjustable curves
- Levels adjustment (black point, white point, midtones)

---

## Phase 5: Architecture & Distribution

**Goal:** Make the app accessible to non-developers and extensible by power users.

### 5.1 Plugin System

- Define a simple plugin API:
  ```python
  class AlchemyPlugin:
      name: str
      stage: str  # which pipeline stage it belongs to
      parameters: dict
      def process(self, image: np.ndarray, params: dict) -> np.ndarray: ...
  ```
- Auto-discover plugins from `~/.alchemycv/plugins/` directory
- Hot-reload without restarting the app

### 5.2 Node-Based Pipeline Editor

- Visual node graph for building custom pipelines (like Blender/Unreal nodes)
- Drag connections between filter nodes
- Fork/merge pipeline branches (e.g., process R/G/B channels differently, then merge)
- Save/share pipeline graphs as JSON

### 5.3 Standalone Executables

- **Windows:** `.exe` via PyInstaller or Nuitka
- **macOS:** `.dmg` / `.app` bundle
- **Linux:** `.AppImage` or Flatpak
- Auto-update mechanism (check GitHub releases on startup)

### 5.4 Library Mode

- After modularization (Phase 1), publish processing functions as a standalone library
- `from alchemycv.pipeline import enhance, denoise, detect_edges`
- Enables Jupyter notebook workflows and scripting without the GUI

### 5.5 Localization (i18n)

- Extract all UI strings to resource files
- Support multiple languages
- Community-contributed translations via JSON/YAML files

---

## Phase 6: Documentation & Community

**Goal:** Lower the barrier for users and contributors.

### 6.1 User Documentation

- Tutorial-style guides with screenshots:
  - "Isolating objects by color"
  - "Counting cells in a microscopy image"
  - "Enhancing low-light photos"
- Parameter reference with visual examples of each filter
- Video walkthrough of the full pipeline

### 6.2 Developer Documentation

- Architecture overview with module dependency diagram
- Contributing guide with testing and code style requirements
- API reference for library mode (auto-generated with Sphinx/mkdocs)

### 6.3 Example Presets

- Ship built-in presets for common tasks:
  - "Sharpen Photo", "Vintage Film", "Edge Sketch", "Cell Counter", "License Plate Isolator"
- Community preset sharing via GitHub repository or built-in preset browser

---

## Priority Roadmap

| Priority | Phase | Item | Impact | Effort |
|----------|-------|------|--------|--------|
| 1 | 1.1 | Modularize app.py | Unlocks everything | Large |
| 2 | 1.2 | Decouple processing from UI | Testability + CLI | Medium |
| 3 | 2.1 | Unit tests | Confidence to refactor | Medium |
| 4 | 2.3 | CI pipeline | Catch regressions | Small |
| 5 | 1.5 | Constants/enums for magic strings | Fewer silent bugs | Small |
| 6 | 1.6 | Input validation | Crash prevention | Small |
| 7 | 3.1 | Tooltips | User experience | Small |
| 8 | 3.4 | Before/after comparison | User experience | Medium |
| 9 | 4.1 | Batch processing | Major feature | Medium |
| 10 | 4.2 | Video support | Major feature | Large |
| 11 | 4.3 | More filters | Feature expansion | Medium |
| 12 | 5.1 | Plugin system | Extensibility | Large |
| 13 | 5.3 | Standalone executables | Accessibility | Medium |
| 14 | 5.2 | Node-based editor | Power users | Very Large |

---

## Known Technical Debt

| Issue | Location | Risk |
|-------|----------|------|
| No tests | Entire project | High — regressions go unnoticed |
| Monolithic class | `app.py:20-1175` | High — blocks all improvements |
| Threading without synchronization | `app.py:644-652` | Medium — race conditions on rapid changes |
| Hardcoded filter strings | Throughout | Medium — typos cause silent failures |
| No parameter validation | Sliders/entries | Medium — invalid values crash OpenCV |
| Frequency filter loses color | `app.py:777-814` | Low — converts to grayscale always |
| Undo captures full state | `app.py:556-566` | Low — memory usage with large histories |
| Settings fragile to renames | `app.py:1045-1086` | Low — saved JSON breaks if vars renamed |
