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
