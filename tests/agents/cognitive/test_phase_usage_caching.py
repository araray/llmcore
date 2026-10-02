# tests/agents/cognitive/test_phase_usage_caching.py
"""Prompt caching must reach the agent's cost accounting.

The circuit breaker's ``COST_LIMIT`` acts on the number ``extract_usage``
produces, and on cache-heavy agent traffic cache reads are the
overwhelming majority of input tokens. Before this was handled, the same
underlying turn priced differently depending only on which convention the
provider used to report it:

* Anthropic-style reports the prompt count as *fresh tokens only*, so the
  cached bulk never reached pricing — measured 10.6x under, and the
  breaker tripped at iteration 52 instead of 5.
* OpenAI-style reports the whole prompt, so every cached token was charged
  at the fresh-input rate — 18.2x over, tripping at iteration 1 and
  killing every run.

These tests pin both conventions to the same answer.
"""

from __future__ import annotations

import pytest

from llmcore.agents.cognitive.phases.usage import extract_usage

MODEL = ("anthropic", "claude-opus-5-5")

#: One real cache-heavy turn: almost nothing fresh, ~922k read from cache.
FRESH, CACHED, OUT = 6, 922_403, 962


def anthropic_shape(**extra):
    """`input_tokens` excludes cache reads; counters sit alongside."""
    usage = {
        "input_tokens": FRESH,
        "output_tokens": OUT,
        "cache_read_input_tokens": CACHED,
        "cache_creation_input_tokens": 0,
        # the provider's OpenAI-compatible aliases, which alias the *fresh* count
        "prompt_tokens": FRESH,
        "completion_tokens": OUT,
        "total_tokens": FRESH + OUT,
    }
    usage.update(extra)
    return {"usage": usage}


def openai_shape(**extra):
    """`prompt_tokens` includes the cached part, broken out in details."""
    usage = {
        "prompt_tokens": FRESH + CACHED,
        "completion_tokens": OUT,
        "total_tokens": FRESH + CACHED + OUT,
        "prompt_tokens_details": {"cached_tokens": CACHED},
    }
    usage.update(extra)
    return {"usage": usage}


class TestConventionsAgree:
    def test_both_report_the_whole_prompt(self):
        a = extract_usage(anthropic_shape(), *MODEL)
        o = extract_usage(openai_shape(), *MODEL)
        assert a.prompt_tokens == o.prompt_tokens == FRESH + CACHED

    def test_both_report_the_same_cached_count(self):
        a = extract_usage(anthropic_shape(), *MODEL)
        o = extract_usage(openai_shape(), *MODEL)
        assert a.cached_tokens == o.cached_tokens == CACHED

    def test_both_cost_the_same(self):
        a = extract_usage(anthropic_shape(), *MODEL)
        o = extract_usage(openai_shape(), *MODEL)
        assert a.cost == pytest.approx(o.cost)

    def test_cached_tokens_are_a_subset_never_additional(self):
        # The contract get_cost relies on; violating it double-counts.
        for shape in (anthropic_shape(), openai_shape()):
            u = extract_usage(shape, *MODEL)
            assert u.cached_tokens <= u.prompt_tokens


class TestCostIsCacheAware:
    def test_cached_reads_cost_far_less_than_fresh_input(self):
        u = extract_usage(anthropic_shape(), *MODEL)
        from llmcore.model_cards import get_model_card_registry

        pricing = get_model_card_registry().get(*MODEL).pricing
        as_fresh = pricing.get_cost(FRESH + CACHED, OUT)
        assert u.cost < as_fresh / 10

    def test_cached_bulk_is_not_dropped(self):
        # The Anthropic failure: pricing only the fresh count.
        u = extract_usage(anthropic_shape(), *MODEL)
        from llmcore.model_cards import get_model_card_registry

        pricing = get_model_card_registry().get(*MODEL).pricing
        fresh_only = pricing.get_cost(FRESH, OUT)
        assert u.cost > fresh_only * 5

    def test_cache_writes_are_charged(self):
        without = extract_usage(anthropic_shape(), *MODEL)
        with_writes = extract_usage(
            anthropic_shape(cache_creation_input_tokens=100_000), *MODEL
        )
        assert with_writes.cache_write_tokens == 100_000
        assert with_writes.cost > without.cost


class TestNoCacheInformation:
    """A plain usage block must behave exactly as it did before."""

    def test_plain_openai_block_is_unchanged(self):
        u = extract_usage(
            {"usage": {"prompt_tokens": 1000, "completion_tokens": 500,
                       "total_tokens": 1500}}, *MODEL)
        assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (1000, 500, 1500)
        assert u.cached_tokens == 0 and u.cache_write_tokens == 0

    def test_missing_total_is_still_derived(self):
        u = extract_usage(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, *MODEL)
        assert u.total_tokens == 15

    def test_no_usage_block_returns_none(self):
        assert extract_usage({"no": "usage"}, *MODEL) is None
        assert extract_usage("not a dict", *MODEL) is None

    def test_all_zero_usage_returns_none(self):
        assert extract_usage(
            {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            *MODEL) is None

    def test_unknown_model_keeps_cost_none_but_still_counts_tokens(self):
        u = extract_usage(anthropic_shape(), "nonexistent", "nope")
        assert u.cost is None
        assert u.cached_tokens == CACHED


class TestDefensive:
    def test_zero_fresh_input_still_prices_the_cache(self):
        # Anthropic really does report input_tokens=0 on a fully cached prompt.
        u = extract_usage(
            {"usage": {"input_tokens": 0, "output_tokens": 10,
                       "cache_read_input_tokens": 500_000,
                       "prompt_tokens": 0, "completion_tokens": 10}}, *MODEL)
        assert u.prompt_tokens == 500_000
        assert u.cost > 0

    def test_cached_count_larger_than_prompt_is_clamped(self):
        u = extract_usage(
            {"usage": {"prompt_tokens": 100, "completion_tokens": 10,
                       "prompt_tokens_details": {"cached_tokens": 999_999}}}, *MODEL)
        assert u.cached_tokens == 100

    def test_garbage_counters_do_not_raise(self):
        u = extract_usage(
            {"usage": {"prompt_tokens": "abc", "completion_tokens": None,
                       "total_tokens": 42, "cache_read_input_tokens": {}}}, *MODEL)
        assert u is not None and u.total_tokens == 42
