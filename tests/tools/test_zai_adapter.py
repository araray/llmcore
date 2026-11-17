# tests/tools/test_zai_adapter.py
"""Tests for the cardctl Z.ai adapter's GLM metadata table.

Regression for PR #60 review: ``glm-5.3-flash`` was added as a narrow entry
without ``context_length`` and ``_enrich_model`` indexed it directly, so a
catalogue fetch containing a Flash model raised ``KeyError``. Narrow entries
now inherit from the family entry they extend at a ``-`` boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.cardctl.adapters.base import NormalizedModel  # noqa: E402
from tools.cardctl.adapters.zai_adapter import _GLM_MODELS, ZaiAdapter, _glm_metadata  # noqa: E402


def _enrich(model_id: str) -> NormalizedModel:
    adapter = ZaiAdapter(api_key="test")
    return adapter._enrich_model(
        NormalizedModel(model_id=model_id, provider="zai"), {"id": model_id}
    )


@pytest.mark.parametrize("model_id", ["glm-5.3-flash", "glm-5.3-flashx", "glm-5.3-flashx-legal"])
def test_flash_models_inherit_family_context(model_id: str) -> None:
    card = _enrich(model_id)
    assert card.context_length == _GLM_MODELS["glm-5.3"]["context_length"]
    assert card.supports_vision is True
    assert card.supports_reasoning is True


def test_siblings_sharing_characters_do_not_inherit() -> None:
    # glm-5v-turbo starts with "glm-5" but is not a "glm-5-" variant.
    assert _glm_metadata("glm-5v-turbo") == _GLM_MODELS["glm-5v-turbo"]
    assert "reasoning" not in _glm_metadata("glm-4.6v")


def test_every_table_entry_resolves_a_context_length() -> None:
    for prefix in _GLM_MODELS:
        assert _glm_metadata(prefix).get("context_length"), prefix


def test_unknown_model_keeps_api_context() -> None:
    adapter = ZaiAdapter(api_key="test")
    nm = NormalizedModel(model_id="glm-9", provider="zai", context_length=4096)
    assert adapter._enrich_model(nm, {"id": "glm-9"}).context_length == 4096
