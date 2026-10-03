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


class TestPoeCardsDoNotCarryTheRemovedCatchAll:
    """Poe cards must not state 128,000 unless the family table says so.

    The Poe adapter carries researched per-family windows and used to have a
    catch-all returning 128,000 for everything else. The catch-all is gone, but
    170 cards generated while it existed kept its answer -- including
    `canvas-creator`, `code-editor` and `elevenlabs-music`, which are Poe bots
    rather than frontier LLMs. Fifty of them disagreed with a family rule the
    same file already held, and eight were really 1,000,000, so llmcore was
    truncating work it could have done.

    This asserts the *narrow* thing and not the tempting one. "Cards must agree
    with the family table" is false: 73 cards legitimately disagree because the
    **table is the stale one** -- it answers 200,000 for anything matching
    "claude" and 128,000 for anything matching "deepseek", while the cards know
    `claude-sonnet-5.5` is 1,000,000 and `deepseek-v4-pro` is 1,048,576. The
    table is a fallback for models nothing else can place, so it cannot be used
    as ground truth against a card that was told better.
    """

    @pytest.fixture(scope="class")
    def poe_cards(self) -> list[tuple[str, dict]]:
        import llmcore.model_cards

        root = (
            pathlib.Path(llmcore.model_cards.__file__).parent / "default_cards" / "poe"
        )
        if not root.is_dir():
            pytest.skip("no packaged poe cards")
        return [(p.name, json.loads(p.read_text())) for p in sorted(root.glob("*.json"))]

    def test_a_card_claiming_128000_is_backed_by_a_family_rule(self, poe_cards):
        from tools.cardctl.adapters.poe_adapter import _guess_context_length

        unbacked = [
            name
            for name, card in poe_cards
            if (card.get("context") or {}).get("max_input_tokens") == 128_000
            and _guess_context_length(card.get("model_id", "")) != 128_000
        ]
        assert not unbacked, (
            f"{len(unbacked)} Poe card(s) claim 128,000 with nothing backing it, "
            f"which is what the removed catch-all produced: {unbacked[:10]}"
        )

    def test_the_catch_all_is_still_gone(self):
        """The data fix above is only durable while the generator stays fixed."""
        from tools.cardctl.adapters.poe_adapter import _guess_context_length

        assert _guess_context_length("a-bot-nobody-has-a-rule-for") is None


class TestAnthropicWindowsAreSourcedOrUnknown:
    """No Claude model has ever had a 128,000-token input window.

    They are 200,000, or 1,000,000 for the 5.x generation. So unlike the chat
    cards in general -- where 128,000 is sometimes right -- that value on an
    Anthropic card is always wrong, which makes it the one provider where the
    audit can be absolute.
    """

    @pytest.fixture(scope="class")
    def anthropic_cards(self) -> list[tuple[str, dict]]:
        import llmcore.model_cards

        root = (
            pathlib.Path(llmcore.model_cards.__file__).parent
            / "default_cards"
            / "anthropic"
        )
        if not root.is_dir():
            pytest.skip("no packaged anthropic cards")
        return [(p.name, json.loads(p.read_text())) for p in sorted(root.glob("*.json"))]

    def test_no_claude_card_claims_128000(self, anthropic_cards):
        offenders = [
            name
            for name, card in anthropic_cards
            if (card.get("context") or {}).get("max_input_tokens") == 128_000
        ]
        assert not offenders, (
            f"Claude windows are 200,000 or 1,000,000, never 128,000: {offenders}"
        )

    def test_every_stated_window_is_a_real_claude_window(self, anthropic_cards):
        """A guard against the next plausible-looking number. If Anthropic
        ships a new window this test should fail and be updated deliberately,
        rather than quietly accepting whatever a generator wrote."""
        allowed = {200_000, 1_000_000}
        wrong = [
            (name, (card.get("context") or {}).get("max_input_tokens"))
            for name, card in anthropic_cards
            if (card.get("context") or {}).get("max_input_tokens")
            and (card.get("context") or {}).get("max_input_tokens") not in allowed
        ]
        assert not wrong, f"unrecognised Claude window(s): {wrong}"


class TestCardsAgreeWithProviderFallbackTables:
    """Where a provider ships a researched table, no card may contradict it.

    Two in-repo sources already knew better than the cards did.
    `DEFAULT_OPENAI_TOKEN_LIMITS` says `gpt-4` is 8,000 and `gpt-3.5-turbo` is
    16,000 while their cards claimed 128,000 -- a 16x overstatement on the
    first, so a caller trusting the card would push 128k tokens into an 8k
    model.

    This is the mirror image of the Poe case, and the difference is worth
    stating. A *family-pattern* table (Poe's, which answers 200,000 for
    anything matching "claude") goes stale as models ship and cannot be
    treated as ground truth. An *exact-id* table like this one cannot go stale
    in the same way: it either names a model or it does not, and when it names
    one it was written about that model.
    """

    @pytest.fixture(scope="class")
    def openai_cards(self) -> list[tuple[str, dict]]:
        import llmcore.model_cards

        root = (
            pathlib.Path(llmcore.model_cards.__file__).parent
            / "default_cards"
            / "openai"
        )
        if not root.is_dir():
            pytest.skip("no packaged openai cards")
        return [(p.name, json.loads(p.read_text())) for p in sorted(root.glob("*.json"))]

    def test_no_openai_card_contradicts_the_provider_table(self, openai_cards):
        from llmcore.providers.openai_provider import DEFAULT_OPENAI_TOKEN_LIMITS

        disagreements = []
        for name, card in openai_cards:
            stated = (card.get("context") or {}).get("max_input_tokens")
            known = DEFAULT_OPENAI_TOKEN_LIMITS.get(card.get("model_id", ""))
            if stated and known and stated != known:
                disagreements.append(f"{name}: card={stated} provider={known}")
        assert not disagreements, (
            f"{len(disagreements)} OpenAI card(s) contradict the provider's own "
            f"researched table, which names each model exactly: {disagreements[:10]}"
        )
