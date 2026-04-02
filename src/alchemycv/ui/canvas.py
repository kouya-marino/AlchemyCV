"""Image display canvas with zoom and pan support."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageCanvas:
    """Manages the right-side image canvas, zoom, and pan."""

    def __init__(self, parent: ttk.Frame, zoom_level_text: tk.StringVar) -> None:
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self._zoom_level_text = zoom_level_text
        self._tk_image: ImageTk.PhotoImage | None = None
        self.processed_cv_image: np.ndarray | None = None

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(parent, background="gray")
        v_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(parent, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        self._image_item = self.canvas.create_image(0, 0, anchor="nw")

        # Zoom controls bar
        zoom_frame = ttk.Frame(parent, padding=5)
        zoom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(zoom_frame, text="Zoom In (+)", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom Out (-)", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Fit to Screen", command=self.fit_to_screen).pack(side=tk.LEFT, padx=5)
        ttk.Label(zoom_frame, textvariable=zoom_level_text).pack(side=tk.RIGHT, padx=5)

        # Mouse bindings
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def update_image(self, cv_image: np.ndarray) -> None:
        self.processed_cv_image = cv_image
        self._refresh()

    def _refresh(self) -> None:
        if self.processed_cv_image is None:
            return

        h, w = self.processed_cv_image.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        if new_w < 1 or new_h < 1:
            return

        interp = cv2.INTER_AREA if self.zoom_factor < 1.0 else cv2.INTER_LINEAR
        zoomed = cv2.resize(self.processed_cv_image, (new_w, new_h), interpolation=interp)

        if len(zoomed.shape) == 2:
            rgb = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        self._tk_image = ImageTk.PhotoImage(image=pil_image)
        self.canvas.itemconfig(self._image_item, image=self._tk_image)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        x_pos = max(0, (canvas_w - new_w) / 2)
        y_pos = max(0, (canvas_h - new_h) / 2)

        self.canvas.coords(self._image_item, x_pos, y_pos)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self._zoom_level_text.set(f"Zoom: {self.zoom_factor:.0%}")

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def zoom_in(self) -> None:
        self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.2)
        self._refresh()

    def zoom_out(self) -> None:
        self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.2)
        self._refresh()

    def fit_to_screen(self, original: np.ndarray | None = None) -> None:
        if original is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            self.canvas.after(50, lambda: self.fit_to_screen(original))
            return
        img_h, img_w = original.shape[:2]
        self.zoom_factor = min(canvas_w / img_w, canvas_h / img_h)
        # Don't refresh here — caller will trigger apply_filter which refreshes

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_ctrl_mousewheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
