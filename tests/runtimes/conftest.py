# tests/runtimes/conftest.py
"""Shared fixtures for the runtimes tests.

The autouse fixture here is a safety fix, not a convenience. ``RuntimeManager``
defaults its state directory to ``~/.llmcore/runtimes``, and that directory is
the user's record of *what is currently costing them money* -- the spec calls
it a safety mechanism rather than a cache, and ``llmcore-runtimes status``
reads it.

Without this redirect, every test that calls ``up()`` wrote a real file there,
so running the test suite left phantom entries claiming a GPU VM was running.
That is precisely the false signal the subsystem exists to prevent: someone
checking whether they are being billed would have been told yes, by their own
test run.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_state(tmp_path, monkeypatch):
    """Point every default state store at a temporary directory."""
    state_dir = str(tmp_path / "runtime-state")
    monkeypatch.setattr("llmcore.runtimes.state.DEFAULT_STATE_DIR", state_dir)
    monkeypatch.setattr("llmcore.runtimes.manager.DEFAULT_STATE_DIR", state_dir, raising=False)
    return state_dir
