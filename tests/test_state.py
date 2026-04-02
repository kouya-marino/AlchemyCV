"""Unit tests for state.manager (UndoManager) and state.settings."""

import json
import tkinter as tk

import pytest

from alchemycv.state import settings as settings_io
from alchemycv.state.manager import UndoManager


@pytest.fixture
def tk_root():
    """Create and destroy a tkinter root for variable tests."""
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


class FakeApp:
    """Minimal object with tkinter variables for state capture."""

    def __init__(self, root):
        self.name = tk.StringVar(master=root, value="test")
        self.count = tk.IntVar(master=root, value=0)
        self.flag = tk.BooleanVar(master=root, value=False)


class TestUndoManager:
    def test_initial_state(self):
        mgr = UndoManager()
        assert not mgr.can_undo
        assert not mgr.can_redo

    def test_capture_and_restore(self, tk_root):
        app = FakeApp(tk_root)
        param_vars = {"slider_a": tk.IntVar(master=tk_root, value=10)}

        snapshot = UndoManager.capture(app, param_vars)
        assert ("self", "name") in snapshot
        assert ("self", "count") in snapshot
        assert ("param", "slider_a") in snapshot
        assert snapshot[("self", "name")] == "test"
        assert snapshot[("param", "slider_a")] == 10

        # Modify state
        app.name.set("changed")
        param_vars["slider_a"].set(99)

        # Restore
        UndoManager.restore(snapshot, app, param_vars)
        assert app.name.get() == "test"
        assert param_vars["slider_a"].get() == 10

    def test_push_and_undo(self, tk_root):
        app = FakeApp(tk_root)
        param_vars = {}
        mgr = UndoManager(debounce_sec=0)  # No debounce for testing

        # Push initial state
        mgr.maybe_push(app, param_vars)
        assert mgr.can_undo

        # Modify and undo
        app.name.set("modified")
        mgr.undo(app, param_vars)
        assert app.name.get() == "test"
        assert mgr.can_redo

    def test_redo(self, tk_root):
        app = FakeApp(tk_root)
        param_vars = {}
        mgr = UndoManager(debounce_sec=0)

        # Save state with name="test"
        mgr.maybe_push(app, param_vars)

        # Change to "v2" and save
        app.name.set("v2")
        mgr.maybe_push(app, param_vars)

        # Change to "v3" (current unsaved state)
        app.name.set("v3")

        # Undo: should restore to "v2" snapshot
        mgr.undo(app, param_vars)
        assert app.name.get() == "v2"

        # Undo again: should restore to "test" snapshot
        mgr.undo(app, param_vars)
        assert app.name.get() == "test"

        # Redo: should go back to "v2"
        mgr.redo(app, param_vars)
        assert app.name.get() == "v2"

    def test_max_undo_limit(self, tk_root):
        app = FakeApp(tk_root)
        param_vars = {}
        mgr = UndoManager(max_undo=3, debounce_sec=0)

        for i in range(5):
            app.count.set(i)
            mgr.maybe_push(app, param_vars)

        assert len(mgr.undo_stack) <= 3

    def test_push_clears_redo(self, tk_root):
        app = FakeApp(tk_root)
        param_vars = {}
        mgr = UndoManager(debounce_sec=0)

        mgr.maybe_push(app, param_vars)
        app.count.set(1)
        mgr.maybe_push(app, param_vars)
        mgr.undo(app, param_vars)
        assert mgr.can_redo

        # New push should clear redo
        app.count.set(99)
        mgr.maybe_push(app, param_vars)
        assert not mgr.can_redo

    def test_undo_empty_returns_false(self, tk_root):
        mgr = UndoManager()
        app = FakeApp(tk_root)
        assert mgr.undo(app, {}) is False

    def test_redo_empty_returns_false(self, tk_root):
        mgr = UndoManager()
        app = FakeApp(tk_root)
        assert mgr.redo(app, {}) is False


class TestSettings:
    def test_save_and_load_roundtrip(self, tk_root, tmp_path):
        app = FakeApp(tk_root)
        param_vars = {
            "slider_x": tk.IntVar(master=tk_root, value=42),
            "mode": tk.StringVar(master=tk_root, value="auto"),
        }

        filepath = tmp_path / "settings.json"
        settings_io.save(app, param_vars, filepath)

        # Verify file is valid JSON
        with open(filepath) as f:
            data = json.load(f)
        assert "param_vars" in data
        assert data["param_vars"]["slider_x"] == 42

        # Modify state, then load
        app.name.set("changed")
        param_vars["slider_x"].set(0)
        param_vars["mode"].set("manual")

        settings_io.load(app, param_vars, filepath)
        assert app.name.get() == "test"
        assert param_vars["slider_x"].get() == 42
        assert param_vars["mode"].get() == "auto"

    def test_load_ignores_unknown_keys(self, tk_root, tmp_path):
        app = FakeApp(tk_root)
        param_vars = {}

        filepath = tmp_path / "settings.json"
        with open(filepath, "w") as f:
            json.dump({"nonexistent_var": "foo", "name": "loaded"}, f)

        settings_io.load(app, param_vars, filepath)
        assert app.name.get() == "loaded"

    def test_load_handles_missing_param_vars(self, tk_root, tmp_path):
        app = FakeApp(tk_root)
        param_vars = {"existing": tk.IntVar(master=tk_root, value=0)}

        filepath = tmp_path / "settings.json"
        with open(filepath, "w") as f:
            json.dump({"param_vars": {"existing": 77, "missing": 99}}, f)

        settings_io.load(app, param_vars, filepath)
        assert param_vars["existing"].get() == 77
