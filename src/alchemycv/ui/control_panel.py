"""Left-side control panel — all filter controls and display options."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from ..constants import (
    CHANNEL_DATA,
    DISPLAY_MODES,
    EDGE_DETECTION_DATA,
    ENHANCEMENT_DATA,
    FILTER_DATA,
    FREQUENCY_DATA,
    MORPH_KERNEL_SHAPES,
    MORPH_OPERATIONS,
    PREPROCESSING_DATA,
)
from .widgets import build_dynamic_panel, create_param_row, create_slider_and_entry

# Check matplotlib availability once
try:
    from matplotlib.figure import Figure  # noqa: F401
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ControlPanel:
    """Builds and manages the left-side scrollable control panel."""

    def __init__(
        self,
        parent: tk.PanedWindow,
        app: Any,
        apply_filter: Callable[[], None],
    ) -> None:
        self.app = app
        self._apply = apply_filter

        # --- Widget storage ---
        self.preproc_param_widgets: list[tk.Widget] = []
        self.enhancement_param_widgets: list[tk.Widget] = []
        self.frequency_param_widgets: list[tk.Widget] = []
        self.channel_param_widgets: list[tk.Widget] = []
        self.param_widgets: list[tk.Widget] = []
        self.edge_param_widgets: list[tk.Widget] = []
        self.morph_param_widgets: list[tk.Widget] = []

        # --- Build scrollable container ---
        left_container = ttk.Frame(parent)
        self.canvas_left = tk.Canvas(left_container)
        v_scroll = ttk.Scrollbar(left_container, orient="vertical", command=self.canvas_left.yview)
        h_scroll = ttk.Scrollbar(left_container, orient="horizontal", command=self.canvas_left.xview)
        self.canvas_left.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        left_container.grid_rowconfigure(0, weight=1)
        left_container.grid_columnconfigure(0, weight=1)
        self.canvas_left.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        controls_frame = ttk.Frame(self.canvas_left, padding="10")
        self.canvas_left.create_window((0, 0), window=controls_frame, anchor="nw")
        controls_frame.bind(
            "<Configure>",
            lambda _e: self.canvas_left.configure(scrollregion=self.canvas_left.bbox("all")),
        )
        self.canvas_left.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas_left.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

        parent.add(left_container, width=480, stretch="never")

        # --- Top buttons ---
        self._build_top_buttons(controls_frame)
        self._build_preprocessing(controls_frame)
        self._build_enhancement(controls_frame)
        self._build_frequency(controls_frame)
        self._build_channel_extractor(controls_frame)
        self._build_mask_generation(controls_frame)
        self._build_refinement(controls_frame)
        self._build_contour_analysis(controls_frame)
        self._build_display_options(controls_frame)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_top_buttons(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        top.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(top, text="Open Image", command=self.app.open_image).grid(
            row=0, column=0, sticky="ew", padx=(0, 2)
        )
        self.save_button = ttk.Button(top, text="Save Image", command=self.app.save_image, state="disabled")
        self.save_button.grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(top, text="Load Settings", command=self.app.load_settings).grid(
            row=0, column=2, sticky="ew", padx=2
        )
        ttk.Button(top, text="Save Settings", command=self.app.save_settings).grid(
            row=0, column=3, sticky="ew", padx=(2, 0)
        )

        self.undo_button = ttk.Button(top, text="Undo (Ctrl+Z)", command=self.app.undo, state="disabled")
        self.undo_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 2), pady=(2, 0))
        self.redo_button = ttk.Button(top, text="Redo (Ctrl+Y)", command=self.app.redo, state="disabled")
        self.redo_button.grid(row=1, column=2, columnspan=2, sticky="ew", padx=(2, 0), pady=(2, 0))

        self.reset_button = ttk.Button(
            parent, text="Reset All to Defaults", command=self.app.reset_all_to_defaults, state="disabled"
        )
        self.reset_button.pack(fill=tk.X, pady=(0, 10))

    def _build_preprocessing(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. Pre-processing", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(
            frame, self.app.selected_preproc, self.app.selected_preproc.get(),
            *PREPROCESSING_DATA.keys(), command=self.on_preproc_change,
        ).pack(fill=tk.X)
        self.preproc_parameter_frame = ttk.Frame(frame, padding=(0, 10, 0, 0))
        self.preproc_parameter_frame.pack(fill=tk.X)

    def _build_enhancement(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. Image Enhancement", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(
            frame, self.app.selected_enhancement, self.app.selected_enhancement.get(),
            *ENHANCEMENT_DATA.keys(), command=self.on_enhancement_change,
        ).pack(fill=tk.X)
        self.enhancement_parameter_frame = ttk.Frame(frame, padding=(0, 10, 0, 0))
        self.enhancement_parameter_frame.pack(fill=tk.X)

    def _build_frequency(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3. Frequency Domain Filters", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.OptionMenu(
            frame, self.app.selected_frequency_filter, self.app.selected_frequency_filter.get(),
            *FREQUENCY_DATA.keys(), command=self.on_frequency_filter_change,
        ).pack(fill=tk.X)
        self.frequency_parameter_frame = ttk.Frame(frame, padding=(0, 10, 0, 0))
        self.frequency_parameter_frame.pack(fill=tk.X)
        self.show_spectrum_button = ttk.Button(
            frame, text="Show Frequency Spectrum", command=self.app.show_frequency_spectrum, state="disabled"
        )
        self.show_spectrum_button.pack(fill=tk.X, pady=(5, 0))

    def _build_channel_extractor(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="4. Channel Extractor", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frame, text="Enable", variable=self.app.channel_enabled, command=self.on_channel_viewer_toggle).pack(anchor="w")
        self.channel_controls_container = ttk.Frame(frame)
        self.channel_controls_container.pack(fill=tk.X, pady=5)
        self._create_channel_controls()

    def _create_channel_controls(self) -> None:
        frame1 = create_param_row("Color Space", parent=self.channel_controls_container)
        space_menu = ttk.OptionMenu(
            frame1, self.app.selected_color_space, self.app.selected_color_space.get(),
            *CHANNEL_DATA.keys(), command=self.on_color_space_change,
        )
        space_menu.grid(row=0, column=1, sticky="ew")

        frame2 = create_param_row("Channel", parent=self.channel_controls_container)
        self.channel_menu = ttk.OptionMenu(frame2, self.app.selected_channel, "")
        self.channel_menu.grid(row=0, column=1, sticky="ew")

    def _build_mask_generation(self, parent: ttk.Frame) -> None:
        self.filter_frame = ttk.LabelFrame(parent, text="5. Mask Generation", padding="10")
        self.filter_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.filter_menubutton = ttk.Menubutton(self.filter_frame, textvariable=self.app.selected_filter, direction="flush")
        filter_menu = tk.Menu(self.filter_menubutton, tearoff=False)
        for name in FILTER_DATA.keys():
            filter_menu.add_radiobutton(label=name, variable=self.app.selected_filter)
        self.filter_menubutton["menu"] = filter_menu
        self.filter_menubutton.pack(fill=tk.X)
        self.parameter_frame = ttk.Frame(self.filter_frame, padding=(0, 10, 0, 0))
        self.parameter_frame.pack(fill=tk.X)

    def _build_refinement(self, parent: ttk.Frame) -> None:
        self.refinement_frame = ttk.LabelFrame(parent, text="6. Mask Refinement", padding="10")
        self.refinement_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        # Edge detection
        edge_frame = ttk.LabelFrame(self.refinement_frame, text="Edge Detection (Overrides Mask Generation)", padding=5)
        edge_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(edge_frame, text="Enable", variable=self.app.edge_enabled, command=self.on_edge_toggle).pack(anchor="w")
        self.edge_controls_container = ttk.Frame(edge_frame)
        self.edge_controls_container.pack(fill=tk.X)
        ttk.OptionMenu(
            self.edge_controls_container, self.app.selected_edge_filter, self.app.selected_edge_filter.get(),
            *EDGE_DETECTION_DATA.keys(), command=self.on_edge_filter_change,
        ).pack(fill=tk.X, pady=(5, 0))
        self.edge_parameter_frame = ttk.Frame(self.edge_controls_container, padding=(0, 10, 0, 0))
        self.edge_parameter_frame.pack(fill=tk.X)

        # Morphological operations
        morph_frame = ttk.LabelFrame(self.refinement_frame, text="Morphological Operations", padding=5)
        morph_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(morph_frame, text="Enable", variable=self.app.morph_enabled, command=self.on_morph_toggle).pack(anchor="w")
        self.morph_controls_container = ttk.Frame(morph_frame)
        self.morph_controls_container.pack(fill=tk.X)
        self._create_morph_controls()

    def _create_morph_controls(self) -> None:
        p = self.app.param_vars

        frame = create_param_row("Morph Operation", parent=self.morph_controls_container)
        self.morph_param_widgets.append(frame)
        var = tk.StringVar(value=MORPH_OPERATIONS[0])
        p["Morph Operation"] = var
        menu = ttk.OptionMenu(frame, var, MORPH_OPERATIONS[0], *MORPH_OPERATIONS, command=lambda _: self._apply())
        menu.grid(row=0, column=1, sticky="ew")
        self.morph_param_widgets.append(menu)

        frame = create_param_row("Kernel Shape", parent=self.morph_controls_container)
        self.morph_param_widgets.append(frame)
        var = tk.StringVar(value=MORPH_KERNEL_SHAPES[0])
        p["Morph Kernel Shape"] = var
        menu = ttk.OptionMenu(frame, var, MORPH_KERNEL_SHAPES[0], *MORPH_KERNEL_SHAPES, command=lambda _: self._apply())
        menu.grid(row=0, column=1, sticky="ew")
        self.morph_param_widgets.append(menu)

        frame = create_param_row("Kernel Size", parent=self.morph_controls_container)
        self.morph_param_widgets.append(frame)
        var = tk.IntVar(value=5)
        p["Morph Kernel Size"] = var
        s, e = create_slider_and_entry(frame, var, 1, 51, on_change=self._apply)
        self.morph_param_widgets.extend([s, e])

        frame = create_param_row("Iterations", parent=self.morph_controls_container)
        self.morph_param_widgets.append(frame)
        var = tk.IntVar(value=1)
        p["Morph Iterations"] = var
        s, e = create_slider_and_entry(frame, var, 1, 20, on_change=self._apply)
        self.morph_param_widgets.extend([s, e])

    def _build_contour_analysis(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="7. Contour Analysis", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(frame, text="Enable", variable=self.app.contours_enabled, command=self.on_contour_toggle).pack(anchor="w")
        self.contour_controls_container = ttk.Frame(frame)
        self.contour_controls_container.pack(fill=tk.X)

        ttk.Checkbutton(
            self.contour_controls_container, text="Draw Contours on Image",
            variable=self.app.draw_contours, command=self._apply,
        ).pack(anchor="w", pady=(5, 0))

        min_frame = create_param_row("Min Area", parent=self.contour_controls_container)
        create_slider_and_entry(min_frame, self.app.contour_min_area, 0, 50000, on_change=self._apply)

        max_frame = create_param_row("Max Area", parent=self.contour_controls_container)
        create_slider_and_entry(max_frame, self.app.contour_max_area, 0, 1000000, on_change=self._apply)

        ttk.Label(
            self.contour_controls_container, textvariable=self.app.object_count_text,
            font=("Helvetica", 10, "bold"),
        ).pack(pady=5)

    def _build_display_options(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Display Options", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        for option in DISPLAY_MODES:
            rb = ttk.Radiobutton(frame, text=option, variable=self.app.display_mode, value=option, command=self._apply)
            rb.pack(anchor="w", side=tk.LEFT, expand=True)
            if option == "Extracted Channel":
                self.channel_display_rb = rb
            if option == "Enhanced Image":
                self.enhanced_display_rb = rb

        hist_button = ttk.Button(frame, text="Show Histogram", command=self.app.show_histogram)
        hist_button.pack(anchor="e", expand=True)
        if not MATPLOTLIB_AVAILABLE:
            hist_button.config(state="disabled")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_preproc_change(self, _event: Any = None) -> None:
        build_dynamic_panel(
            PREPROCESSING_DATA, self.app.selected_preproc.get(),
            self.preproc_parameter_frame, self.preproc_param_widgets,
            self.app.param_vars, "preproc", on_change=self._apply,
        )
        self._apply()

    def on_enhancement_change(self, _event: Any = None) -> None:
        build_dynamic_panel(
            ENHANCEMENT_DATA, self.app.selected_enhancement.get(),
            self.enhancement_parameter_frame, self.enhancement_param_widgets,
            self.app.param_vars, "enhancement", on_change=self._apply,
        )
        self._apply()

    def on_frequency_filter_change(self, _event: Any = None) -> None:
        build_dynamic_panel(
            FREQUENCY_DATA, self.app.selected_frequency_filter.get(),
            self.frequency_parameter_frame, self.frequency_param_widgets,
            self.app.param_vars, "frequency", on_change=self._apply,
        )
        state = "normal" if self.app.selected_frequency_filter.get() != "None" else "disabled"
        self.show_spectrum_button.config(state=state)
        self._apply()

    def on_channel_viewer_toggle(self, _event: Any = None) -> None:
        is_enabled = self.app.channel_enabled.get()
        self._set_widget_state(self.channel_controls_container, "normal" if is_enabled else "disabled")
        self._update_filter_menu_states()

        current_filter = self.app.selected_filter.get()
        current_type = FILTER_DATA.get(current_filter, {}).get("type")
        if is_enabled and current_type == "color":
            for name, data in FILTER_DATA.items():
                if data["type"] != "color":
                    self.app.selected_filter.set(name)
                    break
        else:
            self._apply()

    def on_color_space_change(self, _event: Any = None) -> None:
        space = self.app.selected_color_space.get()
        channels = CHANNEL_DATA[space]["channels"]
        if hasattr(self, "channel_menu"):
            menu = self.channel_menu["menu"]
            menu.delete(0, "end")
            for ch in channels:
                menu.add_command(
                    label=ch,
                    command=lambda v=ch: (self.app.selected_channel.set(v), self._apply()),
                )
            self.app.selected_channel.set(channels[0])
        self._apply()

    def on_filter_change(self, *_args: Any) -> None:
        self._build_filter_panel()
        self._apply()

    def on_edge_filter_change(self, _event: Any = None) -> None:
        build_dynamic_panel(
            EDGE_DETECTION_DATA, self.app.selected_edge_filter.get(),
            self.edge_parameter_frame, self.edge_param_widgets,
            self.app.param_vars, "edge", on_change=self._apply,
        )
        self._apply()

    def on_edge_toggle(self) -> None:
        is_enabled = self.app.edge_enabled.get()
        self._set_widget_state(self.edge_controls_container, "normal" if is_enabled else "disabled")
        self._set_widget_state(self.filter_frame, "disabled" if is_enabled else "normal")
        self._apply()

    def on_morph_toggle(self) -> None:
        state = "normal" if self.app.morph_enabled.get() else "disabled"
        self._set_widget_state(self.morph_controls_container, state)
        self._apply()

    def on_contour_toggle(self) -> None:
        state = "normal" if self.app.contours_enabled.get() else "disabled"
        self._set_widget_state(self.contour_controls_container, state)
        if not self.app.contours_enabled.get():
            self.app.object_count_text.set("Objects Found: --")
        self._apply()

    def initialize_ui_states(self) -> None:
        """Set all UI panels to match current variable values."""
        self.on_preproc_change()
        self.on_enhancement_change()
        self.on_frequency_filter_change()
        self.on_color_space_change()
        self.on_filter_change()
        self.on_edge_filter_change()
        self.on_channel_viewer_toggle()
        self.on_edge_toggle()
        self.on_morph_toggle()
        self.on_contour_toggle()

    def update_undo_buttons(self) -> None:
        self.undo_button.config(state="normal" if self.app.undo_mgr.can_undo else "disabled")
        self.redo_button.config(state="normal" if self.app.undo_mgr.can_redo else "disabled")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_widget_state(self, widget: tk.Widget, state: str) -> None:
        try:
            widget.config(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state(child, state)

    def _update_filter_menu_states(self) -> None:
        is_enabled = self.app.channel_enabled.get()
        try:
            menu_name = self.filter_menubutton.cget("menu")
            if not menu_name:
                return
            filter_menu = self.filter_menubutton.nametowidget(menu_name)
        except tk.TclError:
            return
        for i, name in enumerate(FILTER_DATA.keys()):
            is_color = FILTER_DATA[name]["type"] == "color"
            filter_menu.entryconfigure(i, state="disabled" if (is_enabled and is_color) else "normal")

    def _build_filter_panel(self) -> None:
        config = FILTER_DATA.get(self.app.selected_filter.get(), {})
        pv = self.app.param_vars

        for w in self.param_widgets:
            w.destroy()
        self.param_widgets.clear()
        for k in list(pv.keys()):
            if k.startswith("filter_") or k.startswith("color_") or k.startswith("otsu_"):
                del pv[k]

        if config.get("type") == "color":
            for channel, (min_val, max_val) in zip(config["channels"], config["ranges"]):
                frame = create_param_row(f"{channel} Min", parent=self.parameter_frame, label2="Max")
                self.param_widgets.append(frame)
                min_var = tk.IntVar(value=min_val)
                pv[f"color_{channel}_min"] = min_var
                s1, e1 = create_slider_and_entry(frame, min_var, min_val, max_val, on_change=self._apply)
                self.param_widgets.extend([s1, e1])
                max_var = tk.IntVar(value=max_val)
                pv[f"color_{channel}_max"] = max_var
                s2, e2 = create_slider_and_entry(frame, max_var, min_val, max_val, on_change=self._apply, column_offset=3)
                self.param_widgets.extend([s2, e2])

        elif config.get("type") == "otsu":
            params = config["params"]
            p_name = "Threshold Type"
            p_data = params[p_name]
            frame1 = create_param_row(p_name, parent=self.parameter_frame)
            self.param_widgets.append(frame1)
            var = tk.StringVar(value=p_data["default"])
            pv[f"otsu_{p_name.replace(' ', '_')}"] = var
            menu = ttk.OptionMenu(frame1, var, p_data["default"], *p_data["options"], command=lambda _: self._apply())
            menu.grid(row=0, column=1, sticky="ew")
            self.param_widgets.append(menu)

            frame2 = create_param_row("Calculated Threshold:", parent=self.parameter_frame)
            self.param_widgets.append(frame2)
            calc_var = tk.StringVar(value="--")
            pv["otsu_Calculated_Threshold"] = calc_var
            lbl = ttk.Label(frame2, textvariable=calc_var, font=("Helvetica", 10, "bold"))
            lbl.grid(row=0, column=1)

        else:
            build_dynamic_panel(
                FILTER_DATA, self.app.selected_filter.get(),
                self.parameter_frame, self.param_widgets,
                pv, "filter", on_change=self._apply,
            )

    def _on_mousewheel(self, event: tk.Event) -> None:
        event.widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        event.widget.xview_scroll(int(-1 * (event.delta / 120)), "units")
