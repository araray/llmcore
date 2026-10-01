# tests/routing/test_cards.py
"""Model-card lookups, and an audit that keeps them wired to real namespaces.

The audit exists because of a real bug: routing mapped the provider type
``gemini`` onto a card namespace called ``gemini``, while the cards are filed
under ``google``. Nothing raised -- the lookup returned ``None``, ``None``
means "unknown", and every caller handles unknown quietly. The effect was that
``lowest_cost`` could not price any Gemini target and the pre-call
context-window check never fired for one of the most used providers in the
library.

That failure mode is silent by construction, so it needs a test that fails
loudly instead.
"""

from __future__ import annotations

import pathlib

import pytest

from llmcore.routing.cards import (
    SELF_HOSTED_PROVIDERS,
    canonical_card_provider,
    context_window,
    estimate_cost_usd,
    lookup_card,
)
from llmcore.routing.models import Target


def target(spec: str) -> Target:
    return Target.parse(spec)


class TestCardNamespaceAudit:
    """Every provider type must resolve to a namespace cards are filed under."""

    @pytest.fixture(scope="class")
    def card_providers(self) -> set[str]:
        """The namespaces cards are filed under, read from the packaged tree.

        Deliberately *not* from ``get_model_card_registry()``. That registry is
        a process-wide singleton which other suites reset, reload with user
        cards, and monkeypatch -- so an audit built on it passes alone and
        fails in a full run, for reasons that have nothing to do with what it
        is auditing. The directory listing is the ground truth and is immune to
        all of that.
        """
        import llmcore.model_cards

        cards_dir = (
            pathlib.Path(llmcore.model_cards.__file__).parent / "default_cards"
        )
        return {
            entry.name.lower()
            for entry in cards_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith("_")
        }

    def test_every_provider_type_resolves_to_a_real_namespace(self, card_providers):
        """A type whose canonical namespace has no cards is either genuinely
        card-less (self-hosted, or a vendor we ship none for) or a mapping bug.
        The allow-list below is the full set of the former, so anything new
        showing up here is the latter."""
        from llmcore.providers.manager import PROVIDER_MAP

        # Providers that legitimately have no cards: self-hosted endpoints
        # where the model is whatever the user loaded, and vendors llmcore
        # reaches but ships no card set for.
        expected_cardless = SELF_HOSTED_PROVIDERS | {"groq", "together", "bigmodel"}

        # Filter to classes that actually ship in llmcore: other suites
        # monkeypatch PROVIDER_MAP to register fakes, and auditing a stub named
        # "fake" against the card tree is meaningless.
        real = {
            name: cls
            for name, cls in PROVIDER_MAP.items()
            if getattr(cls, "__module__", "").startswith("llmcore.providers")
        }
        assert real, "PROVIDER_MAP contained no real providers; the filter is wrong"

        unmapped = sorted(
            name
            for name in real
            if canonical_card_provider(name) not in card_providers
            and name not in expected_cardless
        )
        assert not unmapped, (
            f"These provider types resolve to a card namespace with no cards: {unmapped}. "
            f"Either add an alias in llmcore.routing.cards._CARD_PROVIDER_ALIASES or add "
            f"them to this test's expected_cardless set. Namespaces that exist: "
            f"{sorted(card_providers)}"
        )

    def test_the_gemini_regression_specifically(self, card_providers):
        """The bug that prompted this file."""
        assert canonical_card_provider("gemini") == "google"
        assert "google" in card_providers

    def test_aliases_point_at_namespaces_that_exist(self, card_providers):
        from llmcore.routing.cards import _CARD_PROVIDER_ALIASES

        broken = sorted(
            f"{source} -> {destination}"
            for source, destination in _CARD_PROVIDER_ALIASES.items()
            if destination not in card_providers
        )
        assert not broken, f"aliases pointing nowhere: {broken}"


class TestLookups:
    def test_a_known_model_has_a_window_and_a_price(self):
        assert context_window(target("openai:gpt-4o-mini")) == 128_000
        assert estimate_cost_usd(target("openai:gpt-4o-mini"), input_tokens=1_000) > 0

    def test_gemini_targets_are_priced(self):
        """What the alias bug broke."""
        assert context_window(target("gemini:gemini-2.5-flash")) is not None
        assert estimate_cost_usd(target("gemini:gemini-2.5-flash"), input_tokens=1_000) is not None

    def test_an_unknown_model_is_unknown_not_zero(self):
        assert context_window(target("openai:no-such-model-9")) is None
        assert estimate_cost_usd(target("openai:no-such-model-9"), input_tokens=1_000) is None

    def test_a_target_with_no_model_cannot_be_looked_up(self):
        assert lookup_card(target("openai")) is None

    def test_self_hosted_inference_costs_zero_not_unknown(self):
        """Their cards carry no pricing block, and reading that as unknown
        would rank a local model behind a paid API instead of ahead of it."""
        for spec in ("ollama:llama3.3:70b", "vllm:Qwen/Qwen3-30B"):
            assert estimate_cost_usd(target(spec), input_tokens=10_000) == 0.0

    def test_a_configured_instance_name_resolves_via_its_type(self):
        """A card is filed under the provider type, and an instance may be
        called anything."""
        assert context_window(target("my-openai:gpt-4o-mini")) is None
        assert context_window(target("my-openai:gpt-4o-mini"), provider_type="openai") == 128_000

    def test_output_tokens_affect_the_estimate(self):
        cheap = estimate_cost_usd(target("openai:gpt-4o-mini"), input_tokens=1_000)
        dear = estimate_cost_usd(
            target("openai:gpt-4o-mini"), input_tokens=1_000, output_tokens=10_000
        )
        assert dear > cheap

    def test_a_card_store_failure_does_not_break_routing(self, monkeypatch):
        import llmcore.model_cards as model_cards

        def boom():
            raise RuntimeError("card store on fire")

        monkeypatch.setattr(model_cards, "get_model_card_registry", boom)
        assert lookup_card(target("openai:gpt-4o-mini")) is None
