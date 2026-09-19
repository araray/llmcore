# tests/conftest.py
"""Shared fixtures for the llmcore test suite.

The 0.52.0 control plane makes the prompt registry mandatory: every
cognitive phase renders its prompts from the grimoire-backed adapter.
``bundled_prompt_registry`` (or the ``make_bundled_prompt_registry``
helper for non-fixture call sites) supplies the bundled-pack adapter so
tests can construct phases/cycles/agents without wiring config.
"""

import pytest


def make_bundled_prompt_registry():
    """Build a fresh adapter over the packaged ``llmcore-builtin`` pack."""
    from grimoire import Grimoire

    from llmcore.agents.prompts.grimoire_adapter import GrimoirePromptRegistryAdapter
    from llmcore.grimoire_runtime import bundled_pack_path

    return GrimoirePromptRegistryAdapter(Grimoire(bundled_pack_path()))


@pytest.fixture(scope="session")
def bundled_prompt_registry():
    """Session-scoped grimoire adapter over the bundled llmcore spell pack.

    The pack is immutable at test time and adapter state is limited to
    in-memory usage metrics, so one instance can serve the whole session.
    """
    return make_bundled_prompt_registry()


# ---------------------------------------------------------------------------
# requires_ollama: skip live-Ollama tests when no server answers
# ---------------------------------------------------------------------------
_OLLAMA_REACHABLE: bool | None = None


def _ollama_reachable() -> bool:
    global _OLLAMA_REACHABLE
    if _OLLAMA_REACHABLE is None:
        import os
        import urllib.request

        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not base.startswith("http"):
            base = f"http://{base}"
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=1.5) as resp:
                _OLLAMA_REACHABLE = resp.status == 200
        except Exception:
            _OLLAMA_REACHABLE = False
    return _OLLAMA_REACHABLE


def pytest_collection_modifyitems(config, items):
    """Skip ``requires_ollama`` tests unless an Ollama server answers."""
    if not any("requires_ollama" in item.keywords for item in items):
        return
    if _ollama_reachable():
        return
    skip = pytest.mark.skip(reason="requires a live Ollama server (start ollama or set OLLAMA_HOST)")
    for item in items:
        if "requires_ollama" in item.keywords:
            item.add_marker(skip)
