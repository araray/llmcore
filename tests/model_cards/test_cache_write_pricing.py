"""Pricing must account for cache writes.

On real agent traffic, cache reads and writes were ~97% of all input
tokens, so a cost model that only prices fresh input is not approximating
the answer -- it is answering a different question. These tests pin the
semantics, especially the one that is easy to get backwards: a cache write
costs *more* than fresh input, while a cache read costs much less.
"""

import pytest

from llmcore.model_cards.schema import ModelPricing, TokenPricing


def pricing(**kwargs) -> ModelPricing:
    """Claude Opus 5.5's published rates unless overridden."""
    rates = {"input": 4.0, "output": 20.0, "cached_input": 0.2, "cache_write": 5.0}
    rates.update(kwargs)
    return ModelPricing(per_million_tokens=TokenPricing(**rates))


class TestCacheWriteRate:
    def test_cache_write_is_priced_at_its_own_rate(self):
        # 1M written at $5.00/Mtok, nothing else.
        cost = pricing().get_cost(0, 0, cache_write_tokens=1_000_000)
        assert cost == pytest.approx(5.0)

    def test_cache_write_costs_more_than_fresh_input(self):
        written = pricing().get_cost(0, 0, cache_write_tokens=1_000_000)
        fresh = pricing().get_cost(1_000_000, 0)
        assert written > fresh

    def test_cache_read_costs_far_less_than_fresh_input(self):
        read = pricing().get_cost(1_000_000, 0, cached_tokens=1_000_000)
        fresh = pricing().get_cost(1_000_000, 0)
        assert read == pytest.approx(0.2)
        assert read < fresh / 10

    def test_falls_back_to_input_rate_when_unstated(self):
        # Understates the real premium, but inventing a multiplier would be
        # worse; the fallback is at least a published number.
        cost = pricing(cache_write=None).get_cost(0, 0, cache_write_tokens=1_000_000)
        assert cost == pytest.approx(4.0)

    def test_cache_writes_are_additional_not_a_subset_of_input(self):
        # Providers report cache_creation_input_tokens alongside
        # input_tokens, so the two must add rather than overlap.
        both = pricing().get_cost(1_000_000, 0, cache_write_tokens=1_000_000)
        assert both == pytest.approx(4.0 + 5.0)

    def test_omitting_the_argument_changes_nothing(self):
        # Backward compatibility: every existing caller keeps its answer.
        assert pricing().get_cost(1_000, 500) == pytest.approx(
            pricing().get_cost(1_000, 500, 0, 0)
        )


class TestRealisticAgentTurn:
    """The shape that actually shows up in harness traffic."""

    def test_cache_dominated_turn_is_mostly_not_fresh_input(self):
        # Measured medians for one claude-code turn: a few fresh tokens,
        # ~922k read from cache, ~1k out.
        cost = pricing().get_cost(
            input_tokens=6, output_tokens=962, cached_tokens=0,
            cache_write_tokens=0,
        )
        cached = pricing().get_cost(
            input_tokens=922_403, output_tokens=962, cached_tokens=922_403,
        )
        # Without cache pricing the same turn would be charged at $4/Mtok.
        uncached = pricing().get_cost(input_tokens=922_403, output_tokens=962)
        assert cached < uncached / 10
        assert cost < cached

    def test_ignoring_cache_writes_understates_a_cold_turn(self):
        warm = pricing().get_cost(100_000, 1_000, cached_tokens=100_000)
        cold = pricing().get_cost(
            100_000, 1_000, cached_tokens=100_000, cache_write_tokens=100_000
        )
        assert cold > warm
        assert cold - warm == pytest.approx(0.5)  # 100k at $5/Mtok
