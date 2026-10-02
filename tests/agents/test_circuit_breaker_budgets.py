# tests/agents/test_circuit_breaker_budgets.py
"""The runaway guard must sit above normal work, and must be visible coming.

Two problems, both measured on 4,140 real agent turns priced from model
cards with cache-aware accounting:

* `max_total_cost` defaulted to $1.00 while the *median* turn costs $1.04,
  so the guard cut off slightly over half of ordinary turns mid-run. It only
  looked harmless before cost accounting was made cache-aware, which had
  been understating Anthropic-shaped runs roughly tenfold.
* The breaker reports a budget only once it is already gone. On a long run
  that means discarding most of the work, so the result now carries a
  projection a host can act on earlier.

The defaults are declared in four places — the breaker's pydantic config,
its dataclass fallback, its `__init__` signature, and `default_config.toml`
— and the TOML is the one that actually loads. A test pins them together,
because they had already drifted.
"""

from __future__ import annotations

import pytest

from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker, TripReason
from llmcore.config.agents_config import CircuitBreakerConfig, load_agents_config


class TestDefaultsAgreeEverywhere:
    def test_breaker_and_config_section_agree(self):
        breaker, section = AgentCircuitBreaker(), CircuitBreakerConfig()
        assert breaker.config.max_total_cost == section.max_total_cost
        assert (
            breaker.config.max_execution_time_seconds
            == section.max_execution_time_seconds
        )

    def test_the_shipped_toml_agrees_with_the_code(self):
        # The TOML overrides the code defaults, so a mismatch here means the
        # documented default is not the effective one.
        loaded = load_agents_config().circuit_breaker
        section = CircuitBreakerConfig()
        assert loaded.max_total_cost == section.max_total_cost
        assert loaded.max_execution_time_seconds == section.max_execution_time_seconds


class TestBudgetSitsAboveNormalWork:
    #: Measured percentiles of real turn cost, at frontier rates.
    MEDIAN_TURN = 1.04
    P90_TURN = 10.02

    def test_the_median_turn_does_not_trip_the_default(self):
        assert CircuitBreakerConfig().max_total_cost > self.MEDIAN_TURN

    def test_a_p90_turn_does_not_trip_the_default(self):
        # A guard that fires on 10% of ordinary work is an outage, not a guard.
        assert CircuitBreakerConfig().max_total_cost > self.P90_TURN

    def test_a_genuine_runaway_still_trips(self):
        breaker = AgentCircuitBreaker(max_iterations=1000)
        breaker.start()
        result = None
        for i in range(1, 400):
            result = breaker.check(iteration=i, progress=i / 400, cost=0.20)
            if result.tripped:
                break
        assert result.tripped
        assert result.reason == TripReason.COST_LIMIT


class TestProjection:
    def test_projection_extrapolates_to_the_iteration_budget(self):
        breaker = AgentCircuitBreaker(max_iterations=20, max_total_cost=1000.0)
        breaker.start()
        for i in range(1, 5):
            result = breaker.check(iteration=i, progress=i / 20, cost=0.50)
        # $0.50/iteration over 20 iterations.
        assert result.projected_total_cost == pytest.approx(10.0, rel=0.01)

    def test_projection_sees_a_trip_coming_before_it_happens(self):
        breaker = AgentCircuitBreaker(max_iterations=20, max_total_cost=5.0)
        breaker.start()
        result = breaker.check(iteration=1, progress=0.05, cost=1.0)
        assert result.tripped is False
        # One iteration in, the projection already exceeds the budget.
        assert result.projected_total_cost > breaker.config.max_total_cost

    def test_budget_fraction_tracks_spend(self):
        breaker = AgentCircuitBreaker(max_iterations=20, max_total_cost=10.0)
        breaker.start()
        result = breaker.check(iteration=1, progress=0.05, cost=2.5)
        assert result.cost_budget_fraction == pytest.approx(0.25)

    def test_no_projection_before_anything_is_priced(self):
        # Guessing from nothing would be worse than saying nothing.
        breaker = AgentCircuitBreaker(max_iterations=20)
        breaker.start()
        result = breaker.check(iteration=1, progress=0.0, cost=0.0)
        assert result.projected_total_cost == 0.0

    def test_zero_budget_does_not_divide_by_zero(self):
        breaker = AgentCircuitBreaker(max_iterations=20, max_total_cost=0.0)
        breaker.start()
        result = breaker.check(iteration=1, progress=0.0, cost=0.0)
        assert result.cost_budget_fraction == 0.0


class TestOtherTripsUnaffected:
    def test_iteration_limit_still_trips(self):
        breaker = AgentCircuitBreaker(max_iterations=3, max_total_cost=1000.0)
        breaker.start()
        result = None
        for i in range(1, 10):
            result = breaker.check(iteration=i, progress=i / 10, cost=0.0)
            if result.tripped:
                break
        assert result.tripped and result.reason == TripReason.MAX_ITERATIONS

    def test_an_explicit_low_budget_is_still_respected(self):
        # Raising the default must not make the knob unusable for callers
        # who deliberately want a tight leash.
        breaker = AgentCircuitBreaker(max_iterations=100, max_total_cost=0.10)
        breaker.start()
        result = breaker.check(iteration=1, progress=0.01, cost=0.50)
        assert result.tripped and result.reason == TripReason.COST_LIMIT
