"""Pricing for things that are not tokens.

A third of some catalogues is not billed per token: images per image or per
megapixel, video per clip or per second, speech per character,
transcription per audio minute. `per_million_tokens` cannot express any of
it — and because that field was *required*, such a model could not have a
`ModelPricing` at all. That is why those cards carried `pricing: null` and
were invisible to every cost path, and why providers like elevenlabs,
deepgram and fal sit on the unpriceable list.

Comparing rates is only meaningful between models priced in the same unit;
no conversion between dollars-per-image and dollars-per-token is attempted
or implied.
"""

from __future__ import annotations

import pytest

from llmcore.model_cards.schema import (
    ModelCard,
    ModelContext,
    ModelPricing,
    TokenPricing,
    UnitPricing,
)


class TestRepresentable:
    def test_a_model_with_no_token_rates_can_be_priced_at_all(self):
        # The point of the change: this used to be a validation error.
        pricing = ModelPricing(per_unit=UnitPricing(per_image=0.03))
        assert pricing.per_million_tokens is None
        assert pricing.per_unit.is_priced()

    def test_an_empty_unit_block_is_not_priced(self):
        assert UnitPricing().is_priced() is False

    def test_token_only_pricing_still_validates(self):
        pricing = ModelPricing(per_million_tokens=TokenPricing(input=4.0, output=20.0))
        assert pricing.per_unit is None


class TestCost:
    @pytest.mark.parametrize("field,keyword,qty,rate,expected", [
        ("per_image", "images", 4, 0.03, 0.12),
        ("per_megapixel", "megapixels", 2.5, 0.01, 0.025),
        ("per_video", "videos", 3, 1.20, 3.60),
        ("per_video_second", "video_seconds", 5, 0.24, 1.20),
        ("per_audio_minute", "audio_minutes", 10, 0.006, 0.06),
        ("per_character", "characters", 1000, 0.00003, 0.03),
        ("per_second", "seconds", 90, 0.002, 0.18),
        ("per_request", "requests", 7, 0.01, 0.07),
    ])
    def test_each_unit_is_priced(self, field, keyword, qty, rate, expected):
        pricing = ModelPricing(per_unit=UnitPricing(**{field: rate}))
        assert pricing.get_cost(0, 0, **{keyword: qty}) == pytest.approx(expected)

    def test_token_and_unit_rates_add(self):
        # A model may charge for its prompt *and* per image.
        pricing = ModelPricing(
            per_million_tokens=TokenPricing(input=4.0, output=20.0),
            per_unit=UnitPricing(per_image=0.03),
        )
        assert pricing.get_cost(1_000_000, 0, images=2) == pytest.approx(4.06)

    def test_a_unit_the_model_does_not_price_contributes_nothing(self):
        pricing = ModelPricing(per_unit=UnitPricing(per_image=0.03))
        assert pricing.get_cost(0, 0, videos=5) == pytest.approx(0.0)

    def test_but_that_is_reported_rather_than_silently_free(self):
        # "no rate" and "free" must be distinguishable by the caller.
        unit = UnitPricing(per_image=0.03)
        assert unit.unpriced_units({"videos": 5}) == ["videos"]
        assert unit.unpriced_units({"images": 5}) == []

    def test_a_zero_quantity_is_not_an_unpriced_unit(self):
        assert UnitPricing(per_image=0.03).unpriced_units({"videos": 0}) == []

    def test_the_batch_discount_applies_to_unit_pricing_too(self):
        pricing = ModelPricing(
            per_unit=UnitPricing(per_image=0.10), batch_discount_percent=50
        )
        assert pricing.get_cost(0, 0, images=10, batch=True) == pytest.approx(0.5)


class TestTokenPathsStayHonest:
    def test_token_counts_against_a_unit_only_model_cost_nothing(self):
        # Not zero because it is free -- zero because the card states no
        # token rate, and inventing one would be worse.
        pricing = ModelPricing(per_unit=UnitPricing(per_image=0.03))
        assert pricing.get_cost(1_000_000, 100_000) == pytest.approx(0.0)

    def test_rates_for_returns_none_rather_than_zeros(self):
        pricing = ModelPricing(per_unit=UnitPricing(per_image=0.03))
        assert pricing.rates_for(1_000) is None

    def test_existing_token_pricing_is_unchanged(self):
        pricing = ModelPricing(per_million_tokens=TokenPricing(input=4.0, output=20.0))
        assert pricing.get_cost(1_000_000, 100_000) == pytest.approx(6.0)

    def test_omitting_units_changes_nothing(self):
        pricing = ModelPricing(
            per_million_tokens=TokenPricing(input=4.0, output=20.0),
            per_unit=UnitPricing(per_image=0.03),
        )
        assert pricing.get_cost(1_000, 500) == pytest.approx(
            pricing.get_cost(1_000, 500, images=0)
        )


class TestRegistryReporting:
    def _card(self, pricing):
        return ModelCard(model_id="m", display_name="m", provider="gpuai",
                         model_type="chat",
                         context=ModelContext(max_input_tokens=1000),
                         pricing=pricing)

    def test_a_unit_only_card_omits_token_keys_rather_than_zeroing_them(self, tmp_path):
        import json

        from llmcore.model_cards.registry import ModelCardRegistry

        directory = tmp_path / "builtin" / "gpuai"
        directory.mkdir(parents=True)
        card = self._card(ModelPricing(per_unit=UnitPricing(per_image=0.03)))
        (directory / "m.json").write_text(card.model_dump_json())

        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry()
        registry.load(builtin_path=tmp_path / "builtin", user_path=tmp_path / "none")
        info = registry.get_pricing("gpuai", "m")
        ModelCardRegistry.reset_instance()

        assert "input" not in info          # absent, not 0.0
        assert info["per_unit"] == {"per_image": 0.03}
