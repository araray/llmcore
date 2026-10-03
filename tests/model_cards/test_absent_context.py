# tests/model_cards/test_absent_context.py
"""A context window nobody stated must be absent, not 128,000.

This file exists because of a measured, repeated mistake. The card generator
defaulted ``max_input_tokens`` to 4,096, which was far too low and produced
broken context budgets at runtime. That was "fixed" by changing the default to
128,000 -- which is far too low for some models and **meaningless** for others.
Both were the same error one layer down: answering with a plausible number
instead of admitting the question had no answer.

Measured before the fix: **986 of 2,279 effective cards (43%) claimed exactly
128,000 tokens.** Among them were 51 speech-to-text models whose input is
audio, 114 text-to-speech voices whose published limit is in characters, and
124 image generators with no token window of any kind.

The field is not cosmetic. It drives ``list-models`` output, context limits in
several providers, runtime sizing, and whether the agent capability pre-check
refuses a run outright. A card that says nothing is read as unknown; a card
that says 128,000 is believed.
"""

from __future__ import annotations

import json
import pathlib

import pytest

#: Model types whose input is not a sequence of text tokens, so a
#: ``max_input_tokens`` is not a value that happens to be unknown -- it is a
#: measurement of the wrong thing.
NO_TOKEN_INPUT = frozenset(
    {"stt", "tts", "audio", "image-generation", "video-generation", "media"}
)


@pytest.fixture(scope="module")
def cards() -> list[tuple[str, dict]]:
    """Every packaged card, read from the tree rather than the registry.

    Deliberately not via ``get_model_card_registry()``: that singleton is
    reset, reloaded with user cards and monkeypatched by other suites, so an
    audit built on it passes alone and fails in a full run for reasons that
    have nothing to do with what it audits.
    """
    import llmcore.model_cards

    root = pathlib.Path(llmcore.model_cards.__file__).parent / "default_cards"
    out: list[tuple[str, dict]] = []
    for provider in sorted(root.iterdir()):
        if not provider.is_dir() or provider.name.startswith("_"):
            continue
        for path in sorted(provider.glob("*.json")):
            out.append((f"{provider.name}/{path.name}", json.loads(path.read_text())))
    assert out, "no packaged cards were found; the audit is looking in the wrong place"
    return out


class TestNoFabricatedWindows:
    def test_no_media_card_claims_the_generator_s_default_window(self, cards):
        """396 did. The rule is specifically about 128,000 rather than "media
        cards have no window at all", because some genuinely do and the vendor
        publishes it: ElevenLabs states a per-model character limit (5,000 to
        40,000) and Whisper's decoder really does stop at 448. Those are real
        numbers in the wrong unit, which is a different problem from a number
        with no referent.

        What no speech, image or video model has is a 128,000-*token* input
        window. That value only ever came from the builder's fallback."""
        offenders = [
            name
            for name, card in cards
            if (card.get("model_type") or "") in NO_TOKEN_INPUT
            and (card.get("context") or {}).get("max_input_tokens") == 128_000
        ]
        assert not offenders, (
            f"{len(offenders)} card(s) of a non-token-input type claim a "
            f"128,000-token input window: {offenders[:10]}. Set `context` to "
            f"null -- llmcore reads an absent context as unknown, and "
            f"`get_context_length()` returns None for it."
        )

    def test_media_windows_that_remain_are_plausible_for_their_type(self, cards):
        """A weaker but still useful guard on the ones that were kept: a text
        prompt limit for a speech or image endpoint is in the thousands, not
        the hundreds of thousands. Deepgram's 1,000,000 is the one outlier and
        is its own published figure, so it is allowed by name rather than by
        the rule."""
        suspicious = [
            (name, (card.get("context") or {}).get("max_input_tokens"))
            for name, card in cards
            if (card.get("model_type") or "") in NO_TOKEN_INPUT
            and (card.get("context") or {}).get("max_input_tokens", 0) > 100_000
            and not name.startswith("deepgram/")
        ]
        assert not suspicious, (
            f"media cards with an implausibly large token window, which is how "
            f"a text model's figure looks when it lands on a media endpoint: "
            f"{suspicious[:10]}"
        )

    def test_the_absent_block_is_null_not_an_empty_object(self, cards):
        """``ModelContext.max_input_tokens`` is required *within* a context
        block, so ``{}`` cannot validate. Absence has to be the whole block."""
        broken = [
            name
            for name, card in cards
            if "context" in card and card["context"] is not None
            and not (card["context"] or {}).get("max_input_tokens")
        ]
        assert not broken, f"cards with a context block but no window: {broken[:10]}"

    def test_a_card_without_a_window_still_loads_and_answers_none(self):
        """The whole point: unknown must be *representable*, not a load error."""
        from llmcore.model_cards.schema import ModelCard

        card = ModelCard.model_validate(
            {
                "model_id": "x",
                "provider": "p",
                "model_type": "image-generation",
                "context": None,
            }
        )
        assert card.get_context_length() is None


class TestTheGeneratorDoesNotInventOne:
    def test_no_source_means_no_context_block(self):
        """The regression that produced all 986. Nothing stated a window, and
        the builder answered anyway."""
        from tools.cardctl.core.builder import CardBuilder
        from tools.cardctl.adapters.base import NormalizedModel
        from tools.cardctl.core.enrichment import EnrichmentStore, ModelEnrichment

        builder = CardBuilder("p", EnrichmentStore({}, {}, {}, {}, {}, []))
        model = NormalizedModel(model_id="m", provider="p", context_length=None)
        assert builder._build_context(model, ModelEnrichment()) is None

    def test_a_stated_window_is_kept(self):
        from tools.cardctl.core.builder import CardBuilder
        from tools.cardctl.adapters.base import NormalizedModel
        from tools.cardctl.core.enrichment import EnrichmentStore, ModelEnrichment

        builder = CardBuilder("p", EnrichmentStore({}, {}, {}, {}, {}, []))
        model = NormalizedModel(model_id="m", provider="p", context_length=262_144)
        built = builder._build_context(model, ModelEnrichment())
        assert built == {"max_input_tokens": 262_144}

    def test_an_enrichment_override_still_wins(self):
        from tools.cardctl.core.builder import CardBuilder
        from tools.cardctl.adapters.base import NormalizedModel
        from tools.cardctl.core.enrichment import EnrichmentStore, ModelEnrichment

        builder = CardBuilder("p", EnrichmentStore({}, {}, {}, {}, {}, []))
        model = NormalizedModel(model_id="m", provider="p", context_length=8192)
        enrichment = ModelEnrichment(overrides={"max_input_tokens": 1_000_000})
        built = builder._build_context(model, enrichment)
        assert built == {"max_input_tokens": 1_000_000}

    def test_an_unrecognised_poe_model_gets_no_guess(self):
        """The Poe adapter's per-family numbers are researched and stay. Its
        catch-all 128,000 and its 4,096 for media endpoints were not."""
        from tools.cardctl.adapters.poe_adapter import _guess_context_length

        assert _guess_context_length("some-model-nobody-has-heard-of") is None
        assert _guess_context_length("whisper-large") is None
        # Researched values are untouched.
        assert _guess_context_length("gpt-4o") == 128_000
        assert _guess_context_length("claude-sonnet-4") == 200_000
