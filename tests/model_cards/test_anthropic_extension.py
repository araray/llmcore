# tests/model_cards/test_anthropic_extension.py
"""The Anthropic card extension keeps the Models API capability fields.

Regression for PR #60 review: the cards gained ``thinking_types``,
``effort_levels``, ``server_tools`` and more, but ``AnthropicExtension``
ignored undeclared keys, so loading a card silently dropped them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmcore.model_cards import model_supports_native_search
from llmcore.model_cards.schema import ModelCard

_CARDS = Path(__file__).resolve().parents[2] / "src/llmcore/model_cards/default_cards/anthropic"
_CARD_FILES = sorted(_CARDS.glob("*.json"))


@pytest.mark.parametrize("path", _CARD_FILES, ids=lambda p: p.stem)
def test_extension_round_trips_every_field(path: Path) -> None:
    raw = json.loads(path.read_text())
    ext = raw.get("provider_anthropic") or {}
    card = ModelCard.model_validate(raw)
    dumped = card.model_dump(mode="json", exclude_none=True)["provider_anthropic"] if ext else {}
    for key, value in ext.items():
        assert dumped.get(key) == value, f"{path.stem}: provider_anthropic.{key} lost"


def test_server_tools_mark_native_search() -> None:
    with_tools = [
        p
        for p in _CARD_FILES
        if "server_tools" in (json.loads(p.read_text()).get("provider_anthropic") or {})
    ]
    assert with_tools
    card = ModelCard.model_validate(json.loads(with_tools[0].read_text()))
    assert card.provider_anthropic.effort_levels
    assert model_supports_native_search(card) is True
