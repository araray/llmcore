# tests/routing/test_pools.py
"""Pools: eligibility, the seven selection strategies, and order tiers.

The behaviour under test is mostly about *restraint*: a pool must not try a
target it knows will fail, must not treat "unknown" as "zero", and must not
silently drop a member it was told to keep as a spare.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from llmcore.routing.models import (
    Balance,
    FailureKind,
    Outcome,
    RoutingRequest,
    SelectionStrategy,
    Target,
)
from llmcore.routing.pools import Pool, select
from llmcore.routing.state import InMemoryRoutingState

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def state() -> InMemoryRoutingState:
    return InMemoryRoutingState()


def pool_of(*specs: str, **kwargs) -> Pool:
    return Pool.from_config(kwargs.pop("name", "p"), {"targets": list(specs), **kwargs})


def keys(candidates, *, eligible=True) -> list[str]:
    return [c.target.key for c in candidates if c.eligible is eligible]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPoolFromConfig:
    def test_a_bare_list_is_a_pool(self):
        """What people write before they need a strategy."""
        p = Pool.from_config("cheap", ["openai:gpt-4o-mini", "gemini:gemini-2.0-flash"])
        assert len(p.targets) == 2
        assert p.strategy is None, "no strategy means defer to config, not 'priority'"

    def test_a_single_spec_is_a_one_member_pool(self):
        assert Pool.from_config("solo", "openai:gpt-4o").targets[0].model == "gpt-4o"

    def test_routing_params_are_stripped_from_the_target(self):
        """Forwarding `weight` to a vendor would be a 400."""
        p = pool_of("openai:gpt-4o?weight=3&temperature=0.2&order=1")
        target = p.targets[0]
        assert target.params == {"temperature": 0.2}
        assert p.weight_of(target) == 3.0
        assert p.order_of(target) == 1

    def test_explicit_tables_also_work(self):
        p = Pool.from_config(
            "p",
            {
                "targets": ["openai:gpt-4o", "anthropic:claude-opus-4-1"],
                "weights": {"openai:gpt-4o": 5},
                "orders": {"anthropic:claude-opus-4-1": 2},
            },
        )
        assert p.weight_of(Target.parse("openai:gpt-4o")) == 5.0
        assert p.order_of(Target.parse("anthropic:claude-opus-4-1")) == 2

    def test_defaults_apply_to_unlisted_members(self):
        p = pool_of("openai:gpt-4o")
        assert p.weight_of(p.targets[0]) == 1.0
        assert p.order_of(p.targets[0]) == 0

    def test_an_empty_pool_is_rejected_with_an_example(self):
        with pytest.raises(ValueError, match="declares no targets"):
            Pool.from_config("p", {"targets": []})

    @pytest.mark.parametrize(
        ("field", "value"),
        [("strategy", "teleport"), ("affinity", "sticky")],
    )
    def test_a_bad_enum_value_falls_back_rather_than_raising(self, field, value):
        """A typo in one pool must not stop llmcore from starting."""
        p = Pool.from_config("p", {"targets": ["openai:gpt-4o"], field: value})
        assert p.strategy is None if field == "strategy" else p.affinity == "session"

    def test_a_non_numeric_weight_is_ignored(self):
        p = pool_of("openai:gpt-4o?weight=heavy")
        assert p.weight_of(p.targets[0]) == 1.0

    def test_session_affinity_is_the_default(self):
        """Spec 3.5: failing over mid-conversation changes the model, which
        drops the cached prefix and invalidates preserved reasoning."""
        assert pool_of("openai:gpt-4o").affinity_is_sticky is True
        assert pool_of("openai:gpt-4o", affinity="none").affinity_is_sticky is False

    def test_membership_accepts_specs_and_targets(self):
        p = pool_of("openai:gpt-4o?temperature=0.2")
        assert "openai:gpt-4o" in p
        assert Target.parse("openai:gpt-4o?temperature=0.9") in p, "params are not identity"
        assert "openai:gpt-5.4" not in p


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


class TestEligibility:
    @pytest.mark.asyncio
    async def test_a_cooling_target_is_skipped_with_the_time_left(self, state):
        p = pool_of("openai:gpt-4o", "anthropic:claude-opus-4-1")
        await state.record(
            "openai:gpt-4o",
            Outcome("openai:gpt-4o", ok=False, failure=FailureKind.RATE_LIMIT),
            now=NOW,
        )
        chosen, candidates = select(p, state=state, now=NOW)
        assert chosen.key == "anthropic:claude-opus-4-1"
        skipped = next(c for c in candidates if not c.eligible)
        assert "cooling down" in skipped.reason and "rate_limit" in skipped.reason
        assert skipped.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_an_auth_failure_removes_a_target_for_the_process(self, state):
        p = pool_of("openai:gpt-4o", "anthropic:claude-opus-4-1")
        await state.record(
            "openai:gpt-4o", Outcome("openai:gpt-4o", ok=False, failure=FailureKind.AUTH)
        )
        chosen, candidates = select(p, state=state, now=NOW + timedelta(days=2))
        assert chosen.key == "anthropic:claude-opus-4-1"
        assert "unusable" in next(c for c in candidates if not c.eligible).reason

    def test_already_tried_targets_are_excluded(self, state):
        p = pool_of("openai:gpt-4o", "anthropic:claude-opus-4-1")
        chosen, candidates = select(p, state=state, exclude={"openai:gpt-4o"}, now=NOW)
        assert chosen.key == "anthropic:claude-opus-4-1"
        assert "already tried" in next(c for c in candidates if not c.eligible).reason

    def test_a_prompt_too_large_for_a_window_skips_that_member(self, state):
        """The card already knows this will fail; spending a request to be
        told so is pure waste."""
        p = pool_of("openai:gpt-4o", "anthropic:claude-opus-4-1")  # 128k vs 200k
        request = RoutingRequest(prompt="x" * 640_000)  # ~160k tokens
        chosen, candidates = select(p, state=state, request=request, now=NOW)
        assert chosen.key == "anthropic:claude-opus-4-1"
        assert "context window" in next(c for c in candidates if not c.eligible).reason

    def test_an_unknown_window_is_never_a_reason_to_skip(self, state):
        """Plenty of routable models ship no card -- a private vLLM
        checkpoint, a fresh Ollama pull."""
        p = pool_of("openai:no-card-model-9")
        chosen, _ = select(p, state=state, request=RoutingRequest(prompt="x" * 400_000), now=NOW)
        assert chosen is not None

    def test_a_caller_can_override_the_token_requirement(self, state):
        """After a real ContextLengthError the true number is known."""
        p = pool_of("openai:gpt-4o")
        chosen, _ = select(p, state=state, min_context_tokens=500_000, now=NOW)
        assert chosen is None

    def test_an_exhausted_pool_still_explains_itself(self, state):
        p = pool_of("openai:gpt-4o")
        chosen, candidates = select(p, state=state, exclude={"openai:gpt-4o"}, now=NOW)
        assert chosen is None
        assert len(candidates) == 1 and candidates[0].reason


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_priority_follows_declared_order(self, state):
        p = pool_of("anthropic:claude-opus-4-1", "openai:gpt-4o")
        chosen, candidates = select(p, state=state, strategy=SelectionStrategy.PRIORITY, now=NOW)
        assert chosen.key == "anthropic:claude-opus-4-1"
        assert keys(candidates) == ["anthropic:claude-opus-4-1", "openai:gpt-4o"]

    def test_round_robin_rotates_with_the_cursor(self, state):
        p = pool_of("a:1", "b:2", "c:3")
        picked = [
            select(p, state=state, strategy=SelectionStrategy.ROUND_ROBIN, cursor=i, now=NOW)[0].key
            for i in range(4)
        ]
        assert picked == ["a:1", "b:2", "c:3", "a:1"]

    def test_weighted_favours_weight_over_many_draws(self, state):
        p = pool_of("a:1?weight=1", "b:2?weight=9")
        rng = random.Random(0)
        picks = [
            select(p, state=state, strategy=SelectionStrategy.WEIGHTED, rng=rng, now=NOW)[0].key
            for _ in range(400)
        ]
        assert 0.75 < picks.count("b:2") / len(picks) < 0.98

    def test_weighted_orders_the_whole_list(self, state):
        """So the fallback order after a failure is weighted too, not random."""
        p = pool_of("a:1?weight=1", "b:2?weight=1", "c:3?weight=1")
        _, candidates = select(
            p, state=state, strategy=SelectionStrategy.WEIGHTED, rng=random.Random(3), now=NOW
        )
        assert len(keys(candidates)) == 3

    def test_a_zero_weight_member_sorts_last_but_stays_reachable(self, state):
        """Weight 0 means 'do not normally pick this', which is how a spare is
        parked -- dropping it would make the spare useless."""
        p = pool_of("a:1?weight=0", "b:2?weight=1")
        chosen, candidates = select(
            p, state=state, strategy=SelectionStrategy.WEIGHTED, rng=random.Random(1), now=NOW
        )
        assert chosen.key == "b:2"
        assert keys(candidates) == ["b:2", "a:1"]

    @pytest.mark.asyncio
    async def test_lowest_latency_uses_the_smoothed_average(self, state):
        p = pool_of("slow:1", "fast:2")
        await state.record("slow:1", Outcome("slow:1", ok=True, latency_seconds=4.0))
        await state.record("fast:2", Outcome("fast:2", ok=True, latency_seconds=0.5))
        chosen, candidates = select(
            p, state=state, strategy=SelectionStrategy.LOWEST_LATENCY, now=NOW
        )
        assert chosen.key == "fast:2"
        assert keys(candidates) == ["fast:2", "slow:1"]

    @pytest.mark.asyncio
    async def test_lowest_latency_explores_an_unmeasured_target_first(self, state):
        """It cannot prefer low latency without a measurement, and one call is
        all it takes to get one."""
        p = pool_of("measured:1", "fresh:2")
        await state.record("measured:1", Outcome("measured:1", ok=True, latency_seconds=0.1))
        chosen, _ = select(p, state=state, strategy=SelectionStrategy.LOWEST_LATENCY, now=NOW)
        assert chosen.key == "fresh:2"

    def test_lowest_cost_uses_card_pricing(self, state):
        p = pool_of("anthropic:claude-opus-4-1", "openai:gpt-4o-mini")
        chosen, candidates = select(
            p,
            state=state,
            strategy=SelectionStrategy.LOWEST_COST,
            request=RoutingRequest(prompt="x" * 4_000),
            now=NOW,
        )
        assert chosen.key == "openai:gpt-4o-mini"
        scores = {c.target.key: c.score for c in candidates}
        assert scores["openai:gpt-4o-mini"] < scores["anthropic:claude-opus-4-1"]

    def test_lowest_cost_prefers_self_hosted_inference(self, state):
        """A model running on your own machine has no per-token vendor charge,
        which is a fact rather than a special case for local models."""
        p = pool_of("openai:gpt-4o", "ollama:llama3.3:70b")
        chosen, _ = select(
            p,
            state=state,
            strategy=SelectionStrategy.LOWEST_COST,
            request=RoutingRequest(prompt="hello"),
            now=NOW,
        )
        assert chosen.key == "ollama:llama3.3:70b"

    def test_an_unpriced_target_never_outranks_a_priced_one(self, state):
        """Treating unknown as zero is how a cost strategy becomes the most
        expensive thing in the system."""
        p = pool_of("openai:gpt-4o", "openai:no-such-card-model")
        chosen, candidates = select(
            p,
            state=state,
            strategy=SelectionStrategy.LOWEST_COST,
            request=RoutingRequest(prompt="hi"),
            now=NOW,
        )
        assert chosen.key == "openai:gpt-4o"
        assert keys(candidates) == ["openai:gpt-4o", "openai:no-such-card-model"]
        assert next(c for c in candidates if c.target.model == "no-such-card-model").score is None

    @pytest.mark.asyncio
    async def test_least_busy_counts_in_flight_requests(self, state):
        p = pool_of("a:1", "b:2")
        async with state.in_flight("a:1"):
            chosen, _ = select(p, state=state, strategy=SelectionStrategy.LEAST_BUSY, now=NOW)
            assert chosen.key == "b:2"
        chosen, _ = select(p, state=state, strategy=SelectionStrategy.LEAST_BUSY, now=NOW)
        assert chosen.key == "a:1", "the counter must come back down"

    @pytest.mark.asyncio
    async def test_most_credits_ranks_known_balances_first(self, state):
        p = pool_of("poor:1", "rich:2")
        await state.record_balance("poor:1", Balance(amount=1.0))
        await state.record_balance("rich:2", Balance(amount=500.0))
        chosen, candidates = select(
            p, state=state, strategy=SelectionStrategy.MOST_CREDITS, now=NOW
        )
        assert chosen.key == "rich:2"
        assert keys(candidates) == ["rich:2", "poor:1"]

    @pytest.mark.asyncio
    async def test_an_unknown_balance_ranks_after_a_known_one_not_as_zero(self, state):
        """Most vendors expose no balance at all; reading that as empty would
        demote every provider that simply has no endpoint."""
        p = pool_of("unknown:1", "known:2")
        await state.record_balance("known:2", Balance(amount=0.5))
        chosen, candidates = select(
            p, state=state, strategy=SelectionStrategy.MOST_CREDITS, now=NOW
        )
        assert chosen.key == "known:2"
        assert keys(candidates) == ["known:2", "unknown:1"]

    @pytest.mark.asyncio
    async def test_an_observed_credit_failure_feeds_most_credits(self, state):
        """This is what makes the strategy useful where no probe exists: the
        signal arrives from failure rather than from polling."""
        p = pool_of("broke:1", "unknown:2")
        await state.record(
            "broke:1",
            Outcome("broke:1", ok=False, failure=FailureKind.INSUFFICIENT_CREDIT),
            now=NOW,
        )
        # Past the cooldown, so it is eligible again but known to be empty.
        later = NOW + timedelta(minutes=20)
        chosen, candidates = select(
            p, state=state, strategy=SelectionStrategy.MOST_CREDITS, now=later
        )
        assert chosen.key == "unknown:2"
        assert keys(candidates) == ["unknown:2", "broke:1"]

    @pytest.mark.asyncio
    async def test_a_probed_empty_balance_starts_a_cooldown(self, state):
        """The probe already said the next call will fail; spending it to find
        out would be wasteful."""
        await state.record_balance("broke:1", Balance(amount=0.0))
        assert not state.health_sync("broke:1").is_available()

    def test_the_pools_own_strategy_applies_by_default(self, state):
        p = pool_of("anthropic:claude-opus-4-1", "openai:gpt-4o-mini", strategy="lowest_cost")
        chosen, _ = select(p, state=state, request=RoutingRequest(prompt="hi"), now=NOW)
        assert chosen.key == "openai:gpt-4o-mini"

    def test_a_request_can_override_the_pools_strategy(self, state):
        """Config is a warm-up, not a cage -- including at pool level."""
        p = pool_of("anthropic:claude-opus-4-1", "openai:gpt-4o-mini", strategy="lowest_cost")
        chosen, _ = select(
            p,
            state=state,
            strategy=SelectionStrategy.PRIORITY,
            request=RoutingRequest(prompt="hi"),
            now=NOW,
        )
        assert chosen.key == "anthropic:claude-opus-4-1"


# ---------------------------------------------------------------------------
# Order tiers
# ---------------------------------------------------------------------------


class TestOrderTiers:
    def test_a_higher_tier_is_held_back_entirely(self, state):
        """'Use my own GPU, and only pay a vendor if it is down.'"""
        p = pool_of("ollama:llama3.3:70b", "openai:gpt-4o?order=1")
        chosen, candidates = select(p, state=state, now=NOW)
        assert chosen.key == "ollama:llama3.3:70b"
        held = next(c for c in candidates if not c.eligible)
        assert held.target.key == "openai:gpt-4o" and "order tier" in held.reason

    @pytest.mark.asyncio
    async def test_the_next_tier_opens_when_the_first_is_unusable(self, state):
        p = pool_of("ollama:llama3.3:70b", "openai:gpt-4o?order=1")
        await state.record(
            "ollama:llama3.3:70b",
            Outcome("ollama:llama3.3:70b", ok=False, failure=FailureKind.TIMEOUT),
            now=NOW,
        )
        chosen, _ = select(p, state=state, now=NOW)
        assert chosen.key == "openai:gpt-4o"

    def test_a_strategy_applies_within_a_tier(self, state):
        p = pool_of(
            "anthropic:claude-opus-4-1",
            "openai:gpt-4o-mini",
            "openai:gpt-4o?order=1",
            strategy="lowest_cost",
        )
        chosen, candidates = select(p, state=state, request=RoutingRequest(prompt="hi"), now=NOW)
        assert chosen.key == "openai:gpt-4o-mini"
        assert keys(candidates) == ["openai:gpt-4o-mini", "anthropic:claude-opus-4-1"]

    def test_tiers_need_not_start_at_zero(self, state):
        p = Pool.from_config(
            "p",
            {"targets": ["a:1?order=5", "b:2?order=9"]},
        )
        chosen, _ = select(p, state=state, now=NOW)
        assert chosen.key == "a:1"


class TestMostCreditsBands:
    """The band order `most_credits` uses, stated as tests.

    The obvious implementation -- sort all known balances descending -- puts a
    confirmed zero ahead of an unknown, which is backwards.
    """

    @pytest.mark.asyncio
    async def test_known_empty_ranks_behind_unknown(self, state):
        p = pool_of("empty:1", "unknown:2")
        await state.record_balance("empty:1", Balance(amount=0.0))
        later = NOW + timedelta(minutes=20)
        chosen, candidates = select(
            p, state=state, strategy=SelectionStrategy.MOST_CREDITS, now=later
        )
        assert chosen.key == "unknown:2"
        assert keys(candidates) == ["unknown:2", "empty:1"]

    @pytest.mark.asyncio
    async def test_all_three_bands_order_correctly(self, state):
        p = pool_of("empty:1", "unknown:2", "funded:3", "richer:4")
        await state.record_balance("empty:1", Balance(amount=0.0), now=NOW)
        await state.record_balance("funded:3", Balance(amount=5.0), now=NOW)
        await state.record_balance("richer:4", Balance(amount=50.0), now=NOW)
        later = NOW + timedelta(minutes=20)
        _, candidates = select(p, state=state, strategy=SelectionStrategy.MOST_CREDITS, now=later)
        assert keys(candidates) == ["richer:4", "funded:3", "unknown:2", "empty:1"]
