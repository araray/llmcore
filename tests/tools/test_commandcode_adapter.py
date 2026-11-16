# tests/tools/test_commandcode_adapter.py
"""Tests for the cardctl Command Code adapter and its enrichment overlay.

Covers:
* registration (``commandcode`` + brand-spelling aliases),
* normalization of the thin ``/models`` rows: routes, decision models, the
  ``created`` stamp that is not a release date,
* capability inheritance from vendor cards, and the refusal to copy a
  placeholder (all-false) capability block,
* the ``commandcode.toml`` overlay: peak / list pricing policy, cache writes
  reaching the card, and exact entries guarding against prefix leaks,
* ``fetch_models`` against a mocked catalog, including Jev's curated entries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llmcore.model_cards.schema import ModelCard  # noqa: E402
from tools.cardctl.adapters import get_adapter, list_providers  # noqa: E402
from tools.cardctl.adapters.commandcode_adapter import (  # noqa: E402
    MODELS_URL,
    CommandCodeAdapter,
    sibling_candidates,
    states_capabilities,
)
from tools.cardctl.core.builder import CardBuilder  # noqa: E402
from tools.cardctl.core.enrichment import EnrichmentStore  # noqa: E402

ROWS = [
    {"id": "claude-opus-5-5", "name": "Claude Opus 5.5", "context_length": 1000000,
     "created": 1794793041, "supported_endpoints": ["/messages"]},
    {"id": "moonshotai/Kimi-K3", "name": "Kimi K3", "context_length": 1000000,
     "supported_endpoints": ["/chat/completions", "/responses"]},
    {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro (latest)", "context_length": 1000000,
     "supported_endpoints": ["/chat/completions", "/responses"]},
    {"id": "stealth/glyph-cluster:free", "name": "Glyph Cluster", "context_length": 256000,
     "supported_endpoints": ["/chat/completions", "/responses"]},
]


def _caps(**flags: bool) -> SimpleNamespace:
    base = {
        "tool_use": False, "function_calling": False, "vision": False, "reasoning": False,
        "json_mode": False, "structured_output": False, "audio_input": False,
        "video_input": False,
    }
    base.update(flags)
    return SimpleNamespace(**base)


def _card(provider: str, model_id: str, caps: SimpleNamespace, max_out: int | None = 64000):
    return SimpleNamespace(
        provider=provider, model_id=model_id, capabilities=caps,
        context=SimpleNamespace(max_output_tokens=max_out),
        architecture=SimpleNamespace(family="fam"),
    )


class FakeRegistry:
    def __init__(self, cards: dict[tuple[str, str], object]):
        self.cards = cards

    def get(self, provider: str, model_id: str):
        return self.cards.get((provider, model_id.lower()))


def _build(model) -> ModelCard:
    builder = CardBuilder("commandcode", EnrichmentStore.load("commandcode"))
    return ModelCard.model_validate(builder.build(model))


class TestRegistration:
    def test_registered_with_aliases(self):
        for name in ("commandcode", "command_code", "command-code"):
            assert name in list_providers()
            assert isinstance(get_adapter(name), CommandCodeAdapter)

    def test_needs_no_key(self):
        assert CommandCodeAdapter.requires_api_key is False


class TestSiblings:
    @pytest.mark.parametrize(
        ("model_id", "first"),
        [
            ("claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
            ("gpt-6-sol", ("openai", "gpt-6-sol")),
            ("moonshotai/Kimi-K3", ("kimi", "Kimi-K3")),
            ("zai-org/GLM-5.3", ("zai", "GLM-5.3")),
            ("google/gemini-3.8-flash", ("google", "gemini-3.8-flash")),
            ("MiniMaxAI/MiniMax-M3", ("friendli", "MiniMaxAI/MiniMax-M3")),
        ],
    )
    def test_candidates(self, model_id, first):
        assert sibling_candidates(model_id)[0] == first

    def test_unknown_org_has_no_sibling(self):
        assert sibling_candidates("sakana/fugu-ultra") == []

    def test_placeholder_blocks_do_not_count(self):
        assert states_capabilities(_card("a", "b", _caps())) is False
        assert states_capabilities(_card("a", "b", _caps(tool_use=True))) is True


class TestNormalize:
    def test_genuine_sibling_supplies_capabilities_and_max_output(self):
        reg = FakeRegistry({("kimi", "kimi-k3"): _card("kimi", "kimi-k3", _caps(
            tool_use=True, vision=True, reasoning=True), max_out=32768)})
        m = CommandCodeAdapter().normalize(ROWS[1], reg)
        assert m.supports_vision and m.supports_reasoning and m.supports_tools
        assert m.max_output_tokens == 32768
        ext = m.raw_api_data["_extension"]
        assert ext["inherits_from"] == "kimi/kimi-k3"
        assert ext["capabilities_from"] == "kimi/kimi-k3"
        assert ext["routes"] == ["/chat/completions", "/responses"]

    def test_placeholder_sibling_gives_max_output_but_not_capabilities(self):
        reg = FakeRegistry({("anthropic", "claude-opus-5-5"): _card(
            "anthropic", "claude-opus-5-5", _caps(), max_out=128000)})
        m = CommandCodeAdapter().normalize(ROWS[0], reg)
        # The all-false placeholder would have said "no vision, no reasoning".
        assert m.supports_vision and m.supports_reasoning and m.supports_tools
        assert m.max_output_tokens == 128000
        assert m.raw_api_data["_extension"]["capabilities_from"] == "family-defaults"

    def test_created_is_not_a_release_date(self):
        card = _build(CommandCodeAdapter().normalize(ROWS[0], FakeRegistry({})))
        assert card.lifecycle.release_date is None

    def test_decision_model(self):
        m = CommandCodeAdapter().normalize(
            {"id": "typesafe/jev", "supported_endpoints": ["/systemone"], "context_length": 32768},
            FakeRegistry({}),
        )
        assert m.model_type == "other"
        assert not m.supports_tools and not m.supports_streaming


class TestEnrichment:
    def test_claude_prices_include_cache_writes(self):
        card = _build(CommandCodeAdapter().normalize(ROWS[0], FakeRegistry({})))
        p = card.pricing.per_million_tokens
        assert (p.input, p.output, p.cached_input, p.cache_write) == (4.0, 20.0, 0.2, 5.0)

    def test_deepseek_card_price_is_peak_with_off_peak_recorded(self):
        card = _build(CommandCodeAdapter().normalize(ROWS[2], FakeRegistry({})))
        assert card.pricing.per_million_tokens.input == 1.32
        assert card.provider_extension["off_peak"]["input"] == 0.66

    def test_prefix_neighbours_do_not_inherit_a_schedule(self):
        store = EnrichmentStore.load("commandcode")
        fast = store.get("deepseek/deepseek-v4-flash-fast").provider_extension
        assert not isinstance(fast.get("off_peak"), dict)
        contributor = store.get("meta/muse-spark-1.3-contributor").provider_extension
        assert contributor["plans"].startswith("every plan")

    def test_free_model(self):
        card = _build(CommandCodeAdapter().normalize(ROWS[3], FakeRegistry({})))
        assert card.pricing.per_million_tokens.input == 0.0
        assert card.provider_extension["daily_request_cap"] == 1000
        assert "free" in card.tags

    def test_every_shipped_card_is_priced(self):
        cards_dir = _REPO_ROOT / "src" / "llmcore" / "model_cards" / "default_cards" / "commandcode"
        unpriced = []
        for path in cards_dir.glob("*.json"):
            card = ModelCard.model_validate_json(path.read_text())
            if card.pricing is None:
                unpriced.append(card.model_id)
        assert unpriced == []
        assert len(list(cards_dir.glob("*.json"))) >= 100


class TestFetch:
    async def test_fetch_adds_jev_and_refuses_bad_shapes(self):
        adapter = CommandCodeAdapter()
        with respx.mock() as mock:
            mock.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": ROWS}))
            models = await adapter.fetch_models()
        ids = {m.model_id for m in models}
        assert {"typesafe/jev", "typesafe/jev-1.14.0", "claude-opus-5-5"} <= ids

        with respx.mock() as mock:
            mock.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
            with pytest.raises(RuntimeError, match="refusing"):
                await adapter.fetch_models()

        with respx.mock() as mock:
            mock.get(MODELS_URL).mock(return_value=httpx.Response(500))
            with pytest.raises(RuntimeError, match="discovery failed"):
                await adapter.fetch_models()
