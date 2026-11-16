# tests/agents/test_model_router_defaults.py
"""ModelRouter defaults come from the bundled cards: current, priced models.

The hand-written table named only retired models (gpt-3.5-turbo, o1-preview,
Claude 3, Gemini 1.5, deepseek-chat/-reasoner) with 2024 prices.
"""

from __future__ import annotations

from llmcore.agents.routing import DEFAULT_MODELS
from llmcore.agents.routing.model_router import _DEFAULT_SPECS, ModelTier
from llmcore.model_cards.registry import get_model_card_registry


def test_every_default_is_active_and_priced_from_its_card():
    registry = get_model_card_registry()
    registry.load()
    assert len(DEFAULT_MODELS) == len(_DEFAULT_SPECS)
    for model_id, info in DEFAULT_MODELS.items():
        card = registry.get(info.provider, model_id)
        assert card is not None and card.lifecycle.status == "active", model_id
        if info.provider != "ollama":
            assert info.cost_per_1k_input == card.pricing.per_million_tokens.input / 1000
        assert info.context_window == card.context.max_input_tokens


def test_every_tier_is_served():
    tiers = {info.tier for info in DEFAULT_MODELS.values()}
    assert tiers == {ModelTier.FAST, ModelTier.BALANCED, ModelTier.CAPABLE}
