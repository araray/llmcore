# tests/tools/test_typesafe_adapter.py
"""Tests for the cardctl TypeSafe adapter (tools/cardctl/adapters/typesafe_adapter.py).

Covers:
* registration in the adapter registry (``typesafe`` + ``jev`` alias),
* alias collapsing (``jev-latest``/``jev-preview`` -> ``jev-1.13.0``) and
  unknown-name pass-through in ``_normalize_entries``,
* end-to-end card building with the ``typesafe.toml`` enrichment, validated
  against ``ModelCard`` and compared with the hand-written builtin card, and
* ``fetch_models`` against a mocked ``GET /v1/models``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llmcore.model_cards.schema import ModelCard  # noqa: E402
from tools.cardctl.adapters import get_adapter, list_providers  # noqa: E402
from tools.cardctl.adapters.typesafe_adapter import TypeSafeAdapter  # noqa: E402
from tools.cardctl.core.builder import CardBuilder  # noqa: E402
from tools.cardctl.core.enrichment import EnrichmentStore  # noqa: E402

LISTED = [
    {
        "name": "jev-latest",
        "description": "The latest iteration of TypeSafe's System One Model: Jev",
        "release_date": "2026-09-10T18:38:01.391457+00:00",
    },
    {
        "name": "jev-preview",
        "description": "A preview version of `jev-latest`: should be better in most ways",
        "release_date": "2026-09-10T18:39:06.057655+00:00",
    },
]


def test_registered_in_registry():
    assert "typesafe" in list_providers()
    adapter = get_adapter("typesafe")
    assert isinstance(adapter, TypeSafeAdapter)
    assert isinstance(get_adapter("jev"), TypeSafeAdapter)
    assert adapter.provider_name == "typesafe"
    assert adapter.requires_api_key is True
    assert adapter.api_key_env_var == "TYPESAFE_API_KEY"
    assert adapter.base_url == "https://api.typesafe.ai"
    assert adapter.models_endpoint == "/v1/models"


@pytest.fixture
def adapter():
    return TypeSafeAdapter(api_key="ts-test")


def test_normalize_collapses_aliases(adapter):
    models = adapter._normalize_entries(LISTED)
    assert len(models) == 1
    m = models[0]
    assert m.model_id == "jev-1.13.0"
    assert m.provider == "typesafe"
    assert m.model_type == "decision"
    assert m.aliases == ["jev-latest", "jev-preview"]
    assert m.display_name == "Jev 1.13"
    assert m.context_length == 65_536
    assert m.supports_streaming is False
    assert m.supports_tools is False
    assert m.supports_structured_output is True
    assert m.raw_api_data["_pricing"] == {"input": 0.042, "output": 0.0}
    assert m.raw_api_data["_extension"]["question_types"] == ["noul", "choice", "score"]
    assert m.raw_api_data["_release_dates"]["jev-latest"].startswith("2026-09-10")
    assert "latest iteration" in (m.description or "")


def test_normalize_keeps_unknown_names(adapter):
    models = adapter._normalize_entries(
        [*LISTED, {"name": "jev-2-latest", "description": "next gen"}, {"nope": 1}]
    )
    ids = [m.model_id for m in models]
    assert ids == ["jev-1.13.0", "jev-2-latest"]
    nxt = models[1]
    assert nxt.model_type == "decision"
    assert nxt.aliases == []
    assert nxt.description == "next gen"
    assert "_pricing" not in nxt.raw_api_data


def test_build_card_matches_builtin(adapter):
    models = adapter._normalize_entries(LISTED)
    store = EnrichmentStore.load("typesafe")
    card_dict = CardBuilder("typesafe", store).build(models[0])
    card = ModelCard.model_validate(card_dict)

    builtin_path = _REPO_ROOT / "src/llmcore/model_cards/default_cards/typesafe/jev-1.13.0.json"
    builtin = ModelCard.model_validate_json(builtin_path.read_text())

    assert card.model_id == builtin.model_id == "jev-1.13.0"
    assert card.model_type == builtin.model_type == "decision"
    assert sorted(card.aliases) == sorted(builtin.aliases)
    assert card.context.max_input_tokens == builtin.context.max_input_tokens
    assert card.pricing is not None and builtin.pricing is not None
    assert card.pricing.per_million_tokens.input == builtin.pricing.per_million_tokens.input
    assert card.pricing.per_million_tokens.output == builtin.pricing.per_million_tokens.output
    assert card.capabilities.structured_output is True
    assert card.capabilities.streaming is False
    for key in (
        "endpoint",
        "question_types",
        "state_plus_longest_question_max_tokens",
        "output_tokens_billed",
    ):
        assert card.provider_extension[key] == builtin.provider_extension[key]
    assert card.source == "generated"
    json.dumps(card_dict)  # serialisable


def test_fetch_models_requires_key():
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        import asyncio

        adapter = TypeSafeAdapter()
        adapter._api_key = None
        import os

        env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_API_KEY"}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "environ", env)
            asyncio.run(adapter.fetch_models())


@respx.mock
def test_fetch_models_mocked(adapter):
    import asyncio

    route = respx.get("https://api.typesafe.ai/v1/models").mock(
        return_value=httpx.Response(200, json={"models": LISTED})
    )
    models = asyncio.run(adapter.fetch_models())
    assert route.called
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer ts-test"
    assert [m.model_id for m in models] == ["jev-1.13.0"]
    assert models[0].aliases == ["jev-latest", "jev-preview"]
