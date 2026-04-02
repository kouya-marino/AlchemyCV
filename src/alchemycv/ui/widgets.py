"""Reusable widget builders for dynamic parameter panels."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


def create_param_row(
    label1: str,
    parent: ttk.Frame,
    label2: str | None = None,
) -> ttk.Frame:
    """Create a labelled row frame suitable for sliders or option menus."""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=2)
    frame.columnconfigure((1, 4), weight=1)
    ttk.Label(frame, text=label1).grid(row=0, column=0, sticky="w", padx=5)
    if label2:
        ttk.Label(frame, text=label2).grid(row=0, column=3, padx=(10, 0), sticky="w")
    return frame


def create_slider_and_entry(
    parent: ttk.Frame,
    variable: tk.IntVar,
    min_val: int,
    max_val: int,
    on_change: Callable[[], None] | None = None,
    column_offset: int = 0,
) -> tuple[ttk.Scale, ttk.Entry]:
    """Create a linked slider + entry pair inside *parent*."""

    def _on_slider(value: str) -> None:
        variable.set(round(float(value)))
        if on_change:
            on_change()

    slider = ttk.Scale(
        parent,
        from_=min_val,
        to=max_val,
        orient=tk.HORIZONTAL,
        variable=variable,
        command=_on_slider,
    )
    slider.grid(row=0, column=column_offset + 1, sticky="ew", padx=5)

    entry = ttk.Entry(parent, textvariable=variable, width=7)
    entry.grid(row=0, column=column_offset + 2)
    if on_change:
        entry.bind("<Return>", lambda _e: on_change())

    return slider, entry


def build_dynamic_panel(
    data: dict[str, dict[str, Any]],
    selection_name: str,
    parent_frame: ttk.Frame,
    widget_list: list[tk.Widget],
    param_vars: dict[str, tk.Variable],
    param_prefix: str,
    on_change: Callable[[], None] | None = None,
) -> None:
    """Destroy old widgets and rebuild parameter controls for *selection_name*."""
    for w in widget_list:
        w.destroy()
    widget_list.clear()

    # Clean old param vars with this prefix
    for k in list(param_vars.keys()):
        if k.startswith(param_prefix):
            del param_vars[k]

    config = data.get(selection_name, {})
    params = config.get("params", {})

    for p_name, p_data in params.items():
        frame = create_param_row(p_name, parent=parent_frame)
        widget_list.append(frame)
        var_key = f"{param_prefix}_{p_name.replace(' ', '_')}"

        if "options" in p_data:
            var = tk.StringVar(value=p_data["default"])
            menu = ttk.OptionMenu(
                frame,
                var,
                p_data["default"],
                *p_data["options"],
                command=lambda _v: on_change() if on_change else None,
            )
            menu.grid(row=0, column=1, sticky="ew")
            widget_list.append(menu)
        else:
            var = tk.IntVar(value=p_data["default"])
            slider, entry = create_slider_and_entry(
                frame,
                var,
                p_data["range"][0],
                p_data["range"][1],
                on_change=on_change,
            )
            widget_list.extend([slider, entry])

        param_vars[var_key] = var
