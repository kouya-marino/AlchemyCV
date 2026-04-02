"""Undo / redo state management for tkinter variable snapshots."""

from __future__ import annotations

import logging
import time
import tkinter as tk
from typing import Any

log = logging.getLogger(__name__)

StateSnapshot = dict[tuple[str, str], Any]


class UndoManager:
    """Manages an undo/redo stack of captured tkinter variable states."""

    def __init__(self, max_undo: int = 20, debounce_sec: float = 0.5) -> None:
        self.undo_stack: list[StateSnapshot] = []
        self.redo_stack: list[StateSnapshot] = []
        self.max_undo = max_undo
        self._debounce_sec = debounce_sec
        self._last_capture_time: float = 0

    # ------------------------------------------------------------------
    # Capture / restore
    # ------------------------------------------------------------------

    @staticmethod
    def capture(obj: Any, param_vars: dict[str, tk.Variable]) -> StateSnapshot:
        """Snapshot all tkinter Variables on *obj* and in *param_vars*."""
        state: StateSnapshot = {}
        for var_name in dir(obj):
            var = getattr(obj, var_name)
            if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar)):
                try:
                    state[("self", var_name)] = var.get()
                except tk.TclError:
                    pass
        for key, var in param_vars.items():
            try:
                state[("param", key)] = var.get()
            except tk.TclError:
                pass
        return state

    @staticmethod
    def restore(
        state: StateSnapshot,
        obj: Any,
        param_vars: dict[str, tk.Variable],
    ) -> None:
        """Apply a previously captured snapshot back onto *obj* / *param_vars*."""
        for (kind, key), value in state.items():
            try:
                if kind == "self":
                    var = getattr(obj, key, None)
                    if isinstance(var, (tk.StringVar, tk.IntVar, tk.BooleanVar)):
                        var.set(value)
                elif kind == "param" and key in param_vars:
                    param_vars[key].set(value)
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Push / pop
    # ------------------------------------------------------------------

    def maybe_push(self, obj: Any, param_vars: dict[str, tk.Variable]) -> bool:
        """Push a new undo state if the debounce window has elapsed.

        Returns ``True`` if a state was actually pushed.
        """
        now = time.time()
        if now - self._last_capture_time < self._debounce_sec:
            return False

        snapshot = self.capture(obj, param_vars)
        if self.undo_stack and self.undo_stack[-1] == snapshot:
            return False

        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._last_capture_time = now
        return True

    def undo(self, obj: Any, param_vars: dict[str, tk.Variable]) -> bool:
        """Pop the last undo state and restore it. Returns ``True`` on success."""
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.capture(obj, param_vars))
        state = self.undo_stack.pop()
        self.restore(state, obj, param_vars)
        return True

    def redo(self, obj: Any, param_vars: dict[str, tk.Variable]) -> bool:
        """Re-apply the last undone state. Returns ``True`` on success."""
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.capture(obj, param_vars))
        state = self.redo_stack.pop()
        self.restore(state, obj, param_vars)
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)
