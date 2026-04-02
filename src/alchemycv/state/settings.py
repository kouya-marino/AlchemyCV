"""Save / load filter settings as JSON."""

from __future__ import annotations

import json
import logging
import tkinter as tk
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def save(
    obj: Any,
    param_vars: dict[str, tk.Variable],
    filepath: str | Path,
) -> None:
    """Serialize all tkinter Variables on *obj* and *param_vars* to JSON."""
    settings: dict[str, Any] = {}
    for var_name in dir(obj):
        var = getattr(obj, var_name)
        if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar)):
            try:
                settings[var_name] = var.get()
            except tk.TclError:
                pass

    pv_data: dict[str, Any] = {}
    for key, var in param_vars.items():
        try:
            pv_data[key] = var.get()
        except tk.TclError:
            pass
    settings["param_vars"] = pv_data

    with open(filepath, "w") as f:
        json.dump(settings, f, indent=4)
    log.info("Settings saved to %s", filepath)


def load(
    obj: Any,
    param_vars: dict[str, tk.Variable],
    filepath: str | Path,
) -> None:
    """Restore tkinter Variables from a JSON settings file."""
    with open(filepath, "r") as f:
        settings: dict[str, Any] = json.load(f)

    for var_name, value in settings.items():
        if var_name == "param_vars":
            continue
        if hasattr(obj, var_name):
            var = getattr(obj, var_name)
            if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar)):
                try:
                    var.set(value)
                except tk.TclError:
                    pass

    if "param_vars" in settings:
        for key, value in settings["param_vars"].items():
            if key in param_vars:
                try:
                    param_vars[key].set(value)
                except tk.TclError:
                    pass
    log.info("Settings loaded from %s", filepath)
