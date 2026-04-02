"""AlchemyCV — main application entry point.

This module wires together the UI, pipeline, and state modules.
It owns the tkinter root window and coordinates events between layers.
"""

from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

from .constants import (
    CHANNEL_DATA,
    EDGE_DETECTION_DATA,
    FILTER_DATA,
    IMAGE_FILETYPES,
    SAVE_FILETYPES,
)
from .pipeline import engine as pipeline_engine
from .pipeline.frequency import create_filter_mask
from .state import settings as settings_io
from .state.manager import UndoManager
from .ui.canvas import ImageCanvas
from .ui.control_panel import ControlPanel
from .utils import load_image, save_image

# Optional matplotlib
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

log = logging.getLogger(__name__)


class AdvancedFilterApp:
    """Top-level application class — thin orchestrator."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AlchemyCV")
        self.root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

        # --- Image state ---
        self.original_cv_image: np.ndarray | None = None

        # --- Tkinter variables (owned here, read by UI and pipeline) ---
        self.selected_preproc = tk.StringVar(value="None")
        self.selected_enhancement = tk.StringVar(value="None")
        self.selected_frequency_filter = tk.StringVar(value="None")
        self.channel_enabled = tk.BooleanVar(value=False)
        self.selected_color_space = tk.StringVar(value=list(CHANNEL_DATA.keys())[0])
        self.selected_channel = tk.StringVar()
        self.selected_filter = tk.StringVar(value=list(FILTER_DATA.keys())[0])
        self.edge_enabled = tk.BooleanVar(value=False)
        self.selected_edge_filter = tk.StringVar(value="Canny")
        self.morph_enabled = tk.BooleanVar(value=False)
        self.param_vars: dict[str, tk.Variable] = {}

        self.contours_enabled = tk.BooleanVar(value=False)
        self.draw_contours = tk.BooleanVar(value=True)
        self.contour_min_area = tk.IntVar(value=50)
        self.contour_max_area = tk.IntVar(value=1_000_000)
        self.object_count_text = tk.StringVar(value="Objects Found: --")
        self.display_mode = tk.StringVar(value="Final Result")
        self.zoom_level_text = tk.StringVar(value="Zoom: 100%")

        # --- Undo / redo ---
        self.undo_mgr = UndoManager()

        # --- Threading ---
        self._processing = False

        # --- Build UI ---
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        self.controls = ControlPanel(main_pane, app=self, apply_filter=self.apply_filter)

        right_container = ttk.Frame(main_pane, padding="10")
        main_pane.add(right_container)
        self.image_canvas = ImageCanvas(right_container, self.zoom_level_text)

        # Wire up filter trace and keyboard shortcuts
        self.selected_filter.trace_add("write", self.controls.on_filter_change)
        self._bind_shortcuts()

        # Initialize UI state
        self.controls.initialize_ui_states()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _: self.open_image())
        self.root.bind("<Control-s>", lambda _: self.save_image())
        self.root.bind("<Control-z>", lambda _: self.undo())
        self.root.bind("<Control-y>", lambda _: self.redo())
        self.root.bind("<Control-0>", lambda _: self.fit_to_screen())
        self.root.bind("<plus>", lambda _: self.image_canvas.zoom_in())
        self.root.bind("<minus>", lambda _: self.image_canvas.zoom_out())
        self.root.bind("<equal>", lambda _: self.image_canvas.zoom_in())

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------

    def open_image(self) -> None:
        filepath = filedialog.askopenfilename(filetypes=IMAGE_FILETYPES)
        if not filepath:
            return
        img = load_image(filepath)
        if img is None:
            messagebox.showerror("Error", f"Failed to open image: {filepath}")
            return
        self.original_cv_image = img
        self.controls.save_button.config(state="normal")
        self.controls.reset_button.config(state="normal")
        self.reset_all_to_defaults()
        self.fit_to_screen()

    def save_image(self) -> None:
        if self.image_canvas.processed_cv_image is None:
            messagebox.showwarning("No Image", "There is no image to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=SAVE_FILETYPES)
        if not filepath:
            return
        try:
            save_image(self.image_canvas.processed_cv_image, filepath)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the image.\nError: {e}")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def save_settings(self) -> None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        try:
            settings_io.save(self, self.param_vars, filepath)
            messagebox.showinfo("Success", "Settings saved successfully.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save settings.\nError: {e}")

    def load_settings(self) -> None:
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        try:
            settings_io.load(self, self.param_vars, filepath)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load settings file.\nError: {e}")
            return
        self.controls.initialize_ui_states()
        self.apply_filter()

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        if self.undo_mgr.undo(self, self.param_vars):
            self.controls.initialize_ui_states()
            self.controls.update_undo_buttons()

    def redo(self) -> None:
        if self.undo_mgr.redo(self, self.param_vars):
            self.controls.initialize_ui_states()
            self.controls.update_undo_buttons()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all_to_defaults(self) -> None:
        if self.original_cv_image is None:
            return
        self.selected_preproc.set("None")
        self.selected_enhancement.set("None")
        self.selected_frequency_filter.set("None")
        self.channel_enabled.set(False)
        self.selected_color_space.set(list(CHANNEL_DATA.keys())[0])
        self.selected_filter.set(list(FILTER_DATA.keys())[0])
        self.edge_enabled.set(False)
        self.selected_edge_filter.set(list(EDGE_DETECTION_DATA.keys())[0])
        self.morph_enabled.set(False)
        self.param_vars["Morph Operation"].set("Dilate")
        self.param_vars["Morph Kernel Shape"].set("Rectangle")
        self.param_vars["Morph Kernel Size"].set(5)
        self.param_vars["Morph Iterations"].set(1)
        self.contours_enabled.set(False)
        self.draw_contours.set(True)
        self.contour_min_area.set(50)
        self.contour_max_area.set(1_000_000)
        self.display_mode.set("Final Result")
        self.controls.initialize_ui_states()
        self.fit_to_screen()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def fit_to_screen(self) -> None:
        if self.original_cv_image is None:
            return
        self.image_canvas.fit_to_screen(self.original_cv_image)
        self.apply_filter()

    # ------------------------------------------------------------------
    # Main pipeline trigger
    # ------------------------------------------------------------------

    def apply_filter(self, _event: Any = None) -> None:
        if self.original_cv_image is None or self._processing:
            return

        # Debounced undo capture
        self.undo_mgr.maybe_push(self, self.param_vars)
        self.controls.update_undo_buttons()

        # Snapshot all tkinter vars on main thread (thread-safe)
        params: dict[str, Any] = {}
        for k, v in self.param_vars.items():
            try:
                params[k] = v.get()
            except tk.TclError:
                pass

        sel = {
            "preproc": self.selected_preproc.get(),
            "enhancement": self.selected_enhancement.get(),
            "frequency": self.selected_frequency_filter.get(),
            "channel_enabled": self.channel_enabled.get(),
            "color_space": self.selected_color_space.get(),
            "channel": self.selected_channel.get(),
            "filter": self.selected_filter.get(),
            "edge_enabled": self.edge_enabled.get(),
            "edge_filter": self.selected_edge_filter.get(),
            "morph_enabled": self.morph_enabled.get(),
            "contours_enabled": self.contours_enabled.get(),
            "draw_contours": self.draw_contours.get(),
            "min_area": self.contour_min_area.get(),
            "max_area": self.contour_max_area.get(),
            "display_mode": self.display_mode.get(),
        }

        self._processing = True

        def _run() -> None:
            try:
                result = pipeline_engine.run(self.original_cv_image, sel, params)
                self.root.after(0, lambda: self._finish_processing(result, sel))
            except Exception as e:
                log.error("Pipeline error: %s", e, exc_info=True)
                self.root.after(0, lambda: setattr(self, "_processing", False))

        threading.Thread(target=_run, daemon=True).start()

    def _finish_processing(self, result: dict[str, Any] | None, sel: dict[str, Any]) -> None:
        self._processing = False
        if result is None:
            return

        # Update display-mode radio button states
        if hasattr(self.controls, "channel_display_rb"):
            self.controls.channel_display_rb.config(state="normal" if sel["channel_enabled"] else "disabled")
        if hasattr(self.controls, "enhanced_display_rb"):
            self.controls.enhanced_display_rb.config(
                state="normal" if sel["enhancement"] != "None" or sel["frequency"] != "None" else "disabled"
            )

        # Update Otsu threshold display
        if result.get("otsu_threshold") is not None and "otsu_Calculated_Threshold" in self.param_vars:
            self.param_vars["otsu_Calculated_Threshold"].set(str(result["otsu_threshold"]))

        # Update contour count
        if result.get("object_count") is not None:
            self.object_count_text.set(f"Objects Found: {result['object_count']}")

        # Display the selected view
        mode = sel["display_mode"]
        view_map = {
            "Final Result": result["final"],
            "Binary Mask": result["mask"],
            "Enhanced Image": result["enhanced"],
            "Pre-processed Image": result["preprocessed"],
        }
        if mode == "Extracted Channel" and result["extracted_channel"] is not None:
            display_img = result["extracted_channel"]
        else:
            display_img = view_map.get(mode, result["final"])

        self.image_canvas.update_image(display_img)

    # ------------------------------------------------------------------
    # Visualization (histogram / spectrum)
    # ------------------------------------------------------------------

    def show_histogram(self) -> None:
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Missing Library", "Matplotlib is required for this feature.")
            return
        if self.original_cv_image is None:
            messagebox.showwarning("No Image", "Open an image first.")
            return
        try:
            from .pipeline import channels, enhancement, frequency, preprocessing

            params: dict[str, Any] = {}
            for k, v in self.param_vars.items():
                try:
                    params[k] = v.get()
                except tk.TclError:
                    pass

            preprocessed = preprocessing.process(self.original_cv_image, self.selected_preproc.get(), params)
            enhanced = enhancement.process(preprocessed, self.selected_enhancement.get(), params)
            freq_filtered = frequency.process(enhanced, self.selected_frequency_filter.get(), params)

            image_for_hist = freq_filtered
            if self.channel_enabled.get():
                image_for_hist = channels.process(
                    freq_filtered, self.selected_color_space.get(), self.selected_channel.get()
                )
                title = f"Histogram of '{self.selected_channel.get()}' Channel"
            elif self.selected_frequency_filter.get() != "None":
                title = "Histogram of Frequency Filtered Image"
            elif self.selected_enhancement.get() != "None":
                title = "Histogram of Enhanced Image"
            elif self.selected_preproc.get() != "None":
                title = "Histogram of Pre-processed Image"
            else:
                title = "Histogram of Original Image"

            gray = cv2.cvtColor(image_for_hist, cv2.COLOR_BGR2GRAY) if image_for_hist.ndim == 3 else image_for_hist
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            win = tk.Toplevel(self.root)
            win.title(title)
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(hist)
            ax.set_xlim([0, 256])
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Pixel Count")
            ax.grid()
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Histogram Error", f"Could not generate histogram.\nError: {e}")

    def show_frequency_spectrum(self) -> None:
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Missing Library", "Matplotlib is required for this feature.")
            return
        if self.original_cv_image is None:
            messagebox.showwarning("No Image", "Open an image first.")
            return
        try:
            from .pipeline import enhancement, preprocessing

            params: dict[str, Any] = {}
            for k, v in self.param_vars.items():
                try:
                    params[k] = v.get()
                except tk.TclError:
                    pass

            preprocessed = preprocessing.process(self.original_cv_image, self.selected_preproc.get(), params)
            enhanced = enhancement.process(preprocessed, self.selected_enhancement.get(), params)

            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            rows, cols = gray.shape
            m_rows = cv2.getOptimalDFTSize(rows)
            m_cols = cv2.getOptimalDFTSize(cols)
            padded = cv2.copyMakeBorder(gray, 0, m_rows - rows, 0, m_cols - cols, cv2.BORDER_CONSTANT, value=0)

            dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude = 20 * np.log1p(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

            filter_type = self.selected_frequency_filter.get()
            D0 = int(params.get("frequency_Cutoff_Freq_(D0)", 30))
            n = int(params.get("frequency_Order_(n)", 2))
            fmask = create_filter_mask((m_rows, m_cols), filter_type, D0, n)

            win = tk.Toplevel(self.root)
            win.title("Frequency Spectrum and Filter")
            fig = Figure(figsize=(8, 4), dpi=100)
            ax1 = fig.add_subplot(121)
            ax1.imshow(magnitude, cmap="gray")
            ax1.set_title("Magnitude Spectrum")
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = fig.add_subplot(122)
            ax2.imshow(fmask, cmap="gray")
            ax2.set_title(f"{filter_type} Mask")
            ax2.set_xticks([])
            ax2.set_yticks([])
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Spectrum Error", f"Could not generate spectrum.\nError: {e}")


def main() -> None:
    """Application entry point."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    AdvancedFilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
