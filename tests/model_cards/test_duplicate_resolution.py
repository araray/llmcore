"""Two card files claiming one model_id must resolve by rule, not by luck.

The packaged tree contains 89 model_ids that appear in two files under
different filename spellings (`vendor--model.json` written by the
generator, `vendor__model.json` hand-written). Before this was pinned,
`rglob` order decided which one won, and two of the pairs disagreed about
whether the model had pricing at all -- so `Llama-3.3-70B` kept its price
while `Llama-3.2-11B` silently lost one. Routing reads pricing to rank
targets, so that is a cost decision made by directory iteration order.
"""

import json

import pytest

from llmcore.model_cards.registry import ModelCardRegistry


def write_card(directory, filename, *, model_id, source, priced):
    directory.mkdir(parents=True, exist_ok=True)
    card = {
        "model_id": model_id,
        "display_name": model_id,
        "provider": "anthropic",
        "model_type": "chat",
        "source": source,
        "context": {"max_input_tokens": 200000},
    }
    if priced:
        card["pricing"] = {
            "currency": "USD",
            "per_million_tokens": {"input": 4.0, "output": 20.0},
        }
    (directory / filename).write_text(json.dumps(card))


@pytest.fixture(autouse=True)
def fresh_registry():
    """ModelCardRegistry is a process-wide singleton.

    `ModelCardRegistry()` returns the already-initialised instance and
    `load()` early-returns once `_loaded` is set, so without this every
    test after the first would silently assert against the *previous*
    test's cards.
    """
    ModelCardRegistry.reset_instance()
    yield
    ModelCardRegistry.reset_instance()


@pytest.fixture
def tree(tmp_path):
    """One model_id in two files: hand-written and priced, generated and not."""
    builtin = tmp_path / "builtin" / "anthropic"
    # Names chosen so the unpriced generated card sorts *first* -- i.e. the
    # order that previously lost the price.
    write_card(builtin, "aaa--dup.json", model_id="dup", source="generated", priced=False)
    write_card(builtin, "zzz__dup.json", model_id="dup", source="builtin", priced=True)
    return tmp_path


class TestSameTierDuplicates:
    def test_hand_written_card_wins_regardless_of_filename_order(self, tree):
        r = ModelCardRegistry()
        r.load(builtin_path=tree / "builtin", user_path=tree / "absent")
        card = r.get("anthropic", "dup")
        assert card is not None
        assert card.pricing is not None, "the priced hand-written card must win"
        assert card.pricing.per_million_tokens.input == 4.0

    def test_the_collision_is_recorded(self, tree):
        r = ModelCardRegistry()
        r.load(builtin_path=tree / "builtin", user_path=tree / "absent")
        assert len(r._collisions) == 1
        assert "anthropic/dup" in r._collisions[0]

    def test_loading_twice_gives_the_same_answer(self, tree):
        # Genuinely two loads: the singleton must be reset between them,
        # or this compares one loaded instance against itself.
        results = []
        for _ in range(2):
            ModelCardRegistry.reset_instance()
            registry = ModelCardRegistry()
            registry.load(builtin_path=tree / "builtin", user_path=tree / "absent")
            card = registry.get("anthropic", "dup")
            results.append(card.pricing.per_million_tokens.input if card.pricing else None)
        assert results[0] == results[1] == 4.0

    def test_generated_wins_when_it_is_the_only_card(self, tmp_path):
        # Precedence must not mean "drop generated cards".
        write_card(tmp_path / "builtin" / "anthropic", "only.json",
                   model_id="solo", source="generated", priced=True)
        r = ModelCardRegistry()
        r.load(builtin_path=tmp_path / "builtin", user_path=tmp_path / "absent")
        assert r.get("anthropic", "solo") is not None


class TestCrossTierOverrideStillWorks:
    """A user card must still replace a builtin one -- that is the feature."""

    def test_user_card_overrides_builtin_even_though_builtin_outranks(self, tmp_path):
        write_card(tmp_path / "builtin" / "anthropic", "m.json",
                   model_id="m", source="builtin", priced=False)
        write_card(tmp_path / "user" / "anthropic", "m.json",
                   model_id="m", source="generated", priced=True)
        r = ModelCardRegistry()
        r.load(builtin_path=tmp_path / "builtin", user_path=tmp_path / "user")
        card = r.get("anthropic", "m")
        assert card.source == "user"
        assert card.pricing is not None, (
            "origin precedence applies within a tier, never across tiers"
        )
