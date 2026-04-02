# AlchemyCV — Improvement & Future Plan

> **Version:** 1.0.0 | **Last updated:** 2026-04-03
> **Stack:** Python 3.8+ / Tkinter / OpenCV / NumPy / Pillow / Matplotlib

---

## Completed

### Phase 1: Code Quality & Foundation ✅

Modularized the 1191-line monolithic `app.py` into 17 focused modules across 4 packages:

```
src/alchemycv/              (~2,230 lines total)
├── app.py                  (446 lines — thin orchestrator)
├── constants.py            (259 lines — centralized filter data, lookup maps)
├── utils.py                (33 lines — Unicode-safe image I/O)
├── pipeline/               (pure functions, zero tkinter imports)
│   ├── engine.py           preprocessing.py   enhancement.py
│   ├── frequency.py        channels.py        masking.py
│   ├── edges.py            morphology.py      contours.py
├── state/
│   ├── manager.py          (UndoManager — capture/restore/undo/redo)
│   └── settings.py         (JSON save/load)
└── ui/
    ├── control_panel.py    canvas.py          widgets.py
```

What was delivered:
- Pipeline functions decoupled from UI — accept `np.ndarray` + `dict`, return `np.ndarray`
- Type hints on all public functions
- `logging` module replaces `print(stderr)`
- Constants consolidated in `constants.py` (filter data, morph maps, color conv maps)
- `_ensure_odd()` validation in every module needing odd kernel sizes

### Phase 2: Testing & CI/CD ✅

- **140 pytest tests** across 11 test modules — all passing
- Coverage: pipeline modules 97-100%, state 83-89%, utils 91%
- **CI pipeline** (`.github/workflows/ci.yml`): ruff lint/format + pytest on Python 3.9/3.11/3.12 + build verification
- **PyPI publishing** (`.github/workflows/publish.yml`): triggered on `v*.*.*` tags, trusted OIDC publishing
- Dev dependencies: `pytest`, `pytest-cov`, `ruff` in `[project.optional-dependencies]`
- All code passes `ruff check` + `ruff format`

---

## Phase 3: UX Improvements

**Goal:** Make the app more intuitive, responsive, and visually polished.

### 3.1 Tooltips

- Add hover tooltips to every filter parameter explaining what it does
- Example: *"Sigma Color — Higher values mix colors from farther pixels (1-150)"*
- Implement a reusable `ToolTip` widget class in `ui/widgets.py`

### 3.2 Per-Stage Reset Buttons

- Add a "Reset" button to each LabelFrame (stages 1-7)
- Resets only that stage's parameters to defaults
- Currently only "Reset All" exists, which is too destructive for exploration

### 3.3 Progress Indicator

- Show a progress bar or spinner in the status area during heavy operations
- Especially useful for FFT on large images and batch processing (Phase 4)
- Replace the bare `_processing` boolean flag with visual feedback

### 3.4 Before/After Comparison

- Split-view mode: vertical divider to compare original vs. processed side-by-side
- Quick toggle: `Space` key to flip between original and current result
- Useful for fine-tuning enhancement parameters

### 3.5 Dark Mode

- Add a dark theme using custom ttk styles
- Toggle via menu or keyboard shortcut
- Persist preference in settings JSON

### 3.6 Better Zoom & Pan

- Zoom centered on mouse cursor position (not canvas center)
- Pixel inspector on hover — show RGB/HSV values at cursor in status bar
- Optional minimap/navigator overlay for large zoomed-in images

### 3.7 Keyboard Shortcuts Guide

- Add Help > Keyboard Shortcuts dialog listing all bindings
- Consider adding customizable shortcut support

---

## Phase 4: New Features

**Goal:** Expand capabilities beyond the current filter set.

### 4.1 Batch Processing

- Apply current pipeline to an entire folder of images
- Output naming: `{original}_processed.{ext}`
- Progress bar with cancel support
- CLI mode: `alchemycv --batch --input ./photos --output ./results --config pipeline.json`
- Leverages the decoupled pipeline — no UI needed for headless batch

### 4.2 More Filters & Algorithms

**Denoising:**
- Non-local means denoising (`cv2.fastNlMeansDenoisingColored`)
- Bilateral denoising with guided filter

**Segmentation:**
- Watershed segmentation with marker-based seeds
- GrabCut (interactive foreground extraction)
- K-means color clustering with configurable K

**Feature Detection:**
- Harris corner detection with visualization overlay
- ORB keypoint detection with descriptor matching
- Hough line and circle detection

**Sharpening:**
- Laplacian sharpening
- High-boost filtering

### 4.3 Region of Interest (ROI)

- Draw rectangles or polygons directly on the canvas
- Apply the pipeline only within selected regions
- Multiple ROI support with independent filter configs
- ROI saved/loaded as part of settings JSON

### 4.4 Histogram & Levels Tools

- Live histogram overlay on the canvas (toggle on/off)
- Per-channel (R/G/B) histogram with adjustable curves
- Levels adjustment: black point, white point, midtone gamma
- These build on the existing `show_histogram()` feature

### 4.5 Video Support

- Open video files (mp4, avi, mkv) via `cv2.VideoCapture`
- Process frame-by-frame with the current pipeline
- Frame scrubber slider for preview
- Export processed video with codec selection
- Live webcam feed with real-time filtering

---

## Phase 5: Architecture & Distribution

**Goal:** Make the app accessible to non-developers and extensible by power users.

### 5.1 Plugin System

- Simple plugin API leveraging the modular pipeline:
  ```python
  class AlchemyPlugin:
      name: str
      stage: str  # pipeline stage this belongs to
      parameters: dict
      def process(self, image: np.ndarray, params: dict) -> np.ndarray: ...
  ```
- Auto-discover from `~/.alchemycv/plugins/`
- Hot-reload without app restart
- Built-in plugin template generator

### 5.2 Library Mode (Headless API)

- Pipeline is already decoupled — formalize as a public API:
  ```python
  from alchemycv.pipeline import preprocessing, enhancement, edges
  result = preprocessing.process(image, "Gaussian Blur", {"preproc_Kernel_Size": 5})
  ```
- Add `__all__` exports and stable API surface
- Enables Jupyter notebook workflows and scripting

### 5.3 Standalone Executables

- **Windows:** `.exe` via PyInstaller or Nuitka
- **macOS:** `.app` bundle via py2app
- **Linux:** `.AppImage` or Flatpak
- GitHub Actions workflow to build all three on release tags
- Auto-update check against GitHub releases

### 5.4 Node-Based Pipeline Editor

- Visual node graph (drag-and-drop) for building custom pipelines
- Connect filter nodes with wires, fork/merge branches
- Process RGB channels independently and recombine
- Save/share pipeline graphs as JSON
- This is a large undertaking — consider using an existing node framework

### 5.5 Localization (i18n)

- Extract all UI strings to resource files (JSON/YAML)
- Language selector in settings
- Community-contributed translations

---

## Phase 6: Documentation & Community

**Goal:** Lower the barrier for users and contributors.

### 6.1 User Documentation

- Tutorial-style guides with screenshots:
  - "Isolating objects by color"
  - "Counting cells in a microscopy image"
  - "Enhancing low-light photos"
  - "Batch processing a photo folder"
- Parameter reference: visual before/after for each filter
- Video walkthrough of the full pipeline

### 6.2 Developer Documentation

- Architecture diagram showing module dependencies
- Contributing guide updated with testing & lint requirements
- API reference for library mode (auto-generated with mkdocs or Sphinx)
- Plugin development guide

### 6.3 Built-in Presets

- Ship preset configs for common workflows:
  - "Sharpen Photo", "Vintage Film", "Edge Sketch", "Cell Counter", "Document Scanner"
- Preset browser in the app with one-click apply
- Community preset sharing via GitHub repo or in-app download

---

## Priority Roadmap

| Priority | Phase | Item | Impact | Effort |
|----------|-------|------|--------|--------|
| 1 | 3.1 | Tooltips | UX — helps new users understand parameters | Small |
| 2 | 3.2 | Per-stage reset buttons | UX — faster experimentation | Small |
| 3 | 3.4 | Before/after comparison | UX — essential for parameter tuning | Medium |
| 4 | 3.6 | Pixel inspector + cursor zoom | UX — precision work | Medium |
| 5 | 4.1 | Batch processing + CLI | Major feature — unlocks automation | Medium |
| 6 | 4.2 | More filters (denoising, segmentation) | Feature expansion | Medium |
| 7 | 4.4 | Histogram & levels tools | Pro feature | Medium |
| 8 | 3.3 | Progress indicator | UX — feedback for slow ops | Small |
| 9 | 3.5 | Dark mode | UX — visual comfort | Small |
| 10 | 5.2 | Library mode (headless API) | Enables scripting/notebooks | Small |
| 11 | 5.3 | Standalone executables | Accessibility — no Python needed | Medium |
| 12 | 5.1 | Plugin system | Extensibility | Large |
| 13 | 4.3 | ROI support | Advanced feature | Large |
| 14 | 4.5 | Video support | Major feature | Large |
| 15 | 5.4 | Node-based editor | Power users | Very Large |
| 16 | 6.1 | User docs & tutorials | Community growth | Medium |
| 17 | 6.3 | Built-in presets | Discoverability | Small |

---

## Remaining Technical Debt

| Issue | Location | Risk | Notes |
|-------|----------|------|-------|
| Threading without cancellation | `app.py` apply_filter | Medium | Rapid changes can queue threads; no way to cancel long FFT ops |
| Frequency filter loses color | `pipeline/frequency.py` | Low | Always converts to grayscale — could support per-channel FFT |
| Undo captures full state | `state/manager.py` | Low | Memory grows with 20 snapshots; could diff instead |
| Settings fragile to renames | `state/settings.py` | Low | Renaming a tkinter var breaks saved JSON files |
| No UI tests | `ui/` package | Low | Hard to test tkinter in CI; consider screenshot regression tests |

---

## Current Stats

| Metric | Value |
|--------|-------|
| Source files | 21 Python files |
| Source lines | ~2,230 |
| Test files | 11 + conftest |
| Test lines | ~1,100 |
| Test count | 140 |
| Pipeline coverage | 97-100% |
| CI checks | lint + format + test (3 Python versions) + build |
| Filters/operations | 50+ across 7 pipeline stages |
