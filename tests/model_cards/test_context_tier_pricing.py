"""Context-tiered pricing must actually be applied.

`ModelPricing.context_tiers` was declared, documented and populated on five
packaged cards -- and read by nothing. `get_cost` used the flat rates
regardless of prompt size, so a long prompt on a model that charges more for
long context was priced at roughly half its true cost. These tests pin both
the bracket semantics and the rate resolution.
"""

import pytest

from llmcore.model_cards.schema import ContextTier, ModelPricing, TokenPricing


def gemini_like() -> ModelPricing:
    """Gemini 2.5 Pro's declared shape: $1.25 to 200k, $2.50 beyond."""
    return ModelPricing(
        per_million_tokens=TokenPricing(
            input=1.25, output=10.0, cached_input=0.31
        ),
        context_tiers=[
            ContextTier(threshold_tokens=200_000, input_price=1.25, output_price=10.0),
            ContextTier(threshold_tokens=1_048_576, input_price=2.5, output_price=15.0),
        ],
    )


class TestBracketSelection:
    """`threshold_tokens` is an inclusive upper bound, not a floor."""

    @pytest.mark.parametrize("prompt,expected", [
        (1, 200_000),
        (199_999, 200_000),
        (200_000, 200_000),          # inclusive
        (200_001, 1_048_576),        # one token over moves bracket
        (500_000, 1_048_576),
        (1_048_576, 1_048_576),
    ])
    def test_smallest_covering_tier_wins(self, prompt, expected):
        tier = gemini_like().tier_for(prompt)
        assert tier is not None and tier.threshold_tokens == expected

    def test_prompt_beyond_every_tier_uses_the_largest(self):
        # "Past the last bracket" must never mean "back to the cheapest".
        tier = gemini_like().tier_for(5_000_000)
        assert tier.threshold_tokens == 1_048_576
        assert tier.input_price == 2.5

    def test_no_tiers_means_flat_rates(self):
        flat = ModelPricing(per_million_tokens=TokenPricing(input=4.0, output=20.0))
        assert flat.tier_for(10_000_000) is None
        assert flat.rates_for(10_000_000).input == 4.0


class TestRateResolution:
    def test_long_prompt_uses_the_higher_rates(self):
        rates = gemini_like().rates_for(500_000)
        assert rates.input == 2.5
        assert rates.output == 15.0

    def test_cached_rate_keeps_its_discount_ratio(self):
        # cached_input is a discount off input, so it must move with input.
        # Carrying 0.31 across the boundary would make cached reads 25% of
        # input at one tier and 12% at the next, which no vendor publishes.
        base = gemini_like()
        short = base.rates_for(100_000)
        long = base.rates_for(500_000)
        assert short.cached_input == pytest.approx(0.31)
        assert long.cached_input == pytest.approx(0.62)
        assert long.cached_input / long.input == pytest.approx(
            short.cached_input / short.input
        )

    def test_a_tier_may_state_its_own_cached_rate(self):
        pricing = ModelPricing(
            per_million_tokens=TokenPricing(input=1.0, output=4.0, cached_input=0.1),
            context_tiers=[
                ContextTier(threshold_tokens=1_000, input_price=1.0, output_price=4.0),
                ContextTier(threshold_tokens=10_000, input_price=2.0, output_price=8.0,
                            cached_input=0.5),
            ],
        )
        # Explicit beats the ratio rule: 0.5, not 0.2.
        assert pricing.rates_for(5_000).cached_input == pytest.approx(0.5)

    def test_absent_base_cached_rate_stays_absent(self):
        pricing = ModelPricing(
            per_million_tokens=TokenPricing(input=1.0, output=4.0),
            context_tiers=[
                ContextTier(threshold_tokens=1_000, input_price=1.0, output_price=4.0),
                ContextTier(threshold_tokens=10_000, input_price=2.0, output_price=8.0),
            ],
        )
        assert pricing.rates_for(5_000).cached_input is None


class TestCost:
    def test_long_prompt_costs_roughly_double(self):
        p = gemini_like()
        short = p.get_cost(100_000, 10_000)
        long = p.get_cost(500_000, 10_000)
        # 500k at $2.50 + 10k at $15.00
        assert long == pytest.approx(500_000 / 1e6 * 2.5 + 10_000 / 1e6 * 15.0)
        assert long > short

    def test_flat_pricing_would_have_understated_it(self):
        p = gemini_like()
        flat = 500_000 / 1e6 * 1.25 + 10_000 / 1e6 * 10.0
        assert p.get_cost(500_000, 10_000) / flat == pytest.approx(1.93, abs=0.02)

    def test_cache_write_tokens_count_toward_the_bracket(self):
        # They are part of the prompt even though providers report them
        # separately from input_tokens.
        p = gemini_like()
        assert p.tier_for(150_000).threshold_tokens == 200_000
        under = p.get_cost(150_000, 0)
        over = p.get_cost(150_000, 0, cache_write_tokens=100_000)
        # 250k total prompt crosses into the higher bracket, so the fresh
        # input is repriced too -- not merely the written tokens.
        assert over > under + (100_000 / 1e6 * 1.25)

    def test_tiers_are_used_regardless_of_declaration_order(self):
        p = gemini_like()
        p.context_tiers = list(reversed(p.context_tiers))
        assert p.tier_for(100_000).threshold_tokens == 200_000

    def test_omitting_new_arguments_changes_nothing_for_flat_models(self):
        flat = ModelPricing(per_million_tokens=TokenPricing(input=4.0, output=20.0))
        assert flat.get_cost(1_000, 500) == pytest.approx(
            1_000 / 1e6 * 4.0 + 500 / 1e6 * 20.0
        )
