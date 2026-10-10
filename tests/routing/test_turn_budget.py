# tests/routing/test_turn_budget.py
"""The per-turn step and spend budget.

Why this exists at all is a measured claim, not a preference: routing by
predicted prompt complexity is worth ~2.2% of this project's spend, while turns
over 200 steps are 62% of it. So the quantity worth bounding is the one that is
observable directly and exactly, rather than the one a classifier guesses.

Most of what is tested here is the behaviour around *not knowing*. A budget
that treats an unpriced call as free never trips, and this subsystem's cost
model has failed that way at seven separate layers.
"""

from __future__ import annotations

import pytest

from llmcore.routing.budget import (
    BudgetAction,
    BudgetPolicy,
    BudgetState,
    TurnBudget,
)


class TestNoDefaultLimits:
    def test_an_unconfigured_policy_bounds_nothing(self):
        """The agent circuit breaker shipped with a $1.00 default that would
        have cut off 50.7% of this project's normal turns. A number nobody
        chose is worse than no number."""
        policy = BudgetPolicy()
        assert policy.max_steps is None
        assert policy.max_cost_usd is None
        assert policy.is_bounded is False

    def test_an_unbounded_budget_never_exceeds(self):
        budget = TurnBudget()
        for _ in range(10_000):
            budget.record(cost_usd=1.0)
        verdict = budget.check()
        assert verdict.state is BudgetState.OK
        assert verdict.should_stop is False

    @pytest.mark.parametrize(
        "kwargs",
        [{"max_steps": 0}, {"max_steps": -1}, {"max_cost_usd": 0},
         {"warn_at_fraction": 0}, {"warn_at_fraction": 1.5}],
    )
    def test_a_nonsensical_dial_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            BudgetPolicy(**kwargs)


class TestSteps:
    def test_the_step_ceiling_trips(self):
        budget = TurnBudget(BudgetPolicy(max_steps=3, action=BudgetAction.STOP))
        for _ in range(3):
            assert budget.check().should_stop is False
            budget.record(cost_usd=0.01)
        verdict = budget.check()
        assert verdict.state is BudgetState.EXCEEDED
        assert verdict.should_stop is True
        assert "3/3 steps" in verdict.reason

    def test_warning_comes_before_the_ceiling(self):
        budget = TurnBudget(BudgetPolicy(max_steps=10, warn_at_fraction=0.8))
        for _ in range(8):
            budget.record(cost_usd=0.01)
        verdict = budget.check()
        assert verdict.state is BudgetState.WARNING
        assert "8 of 10 steps" in verdict.reason

    def test_report_records_without_stopping(self):
        """For measuring before enforcing, which is how a limit gets chosen
        rather than guessed."""
        budget = TurnBudget(BudgetPolicy(max_steps=1, action=BudgetAction.REPORT))
        budget.record(cost_usd=0.01)
        verdict = budget.check()
        assert verdict.state is BudgetState.EXCEEDED
        assert verdict.should_stop is False

    def test_warn_does_not_stop_either(self):
        budget = TurnBudget(BudgetPolicy(max_steps=1, action=BudgetAction.WARN))
        budget.record(cost_usd=0.01)
        assert budget.check().should_stop is False


class TestUnknownCostIsNotZero:
    def test_an_unpriced_step_still_counts_as_a_step(self):
        budget = TurnBudget(BudgetPolicy(max_steps=2, action=BudgetAction.STOP))
        budget.record(cost_usd=None)
        budget.record(cost_usd=None)
        assert budget.check().should_stop is True

    def test_an_unpriced_step_does_not_add_zero_to_the_spend(self):
        """Adding 0.0 would be indistinguishable from a genuinely free call,
        and would hold the mean cost per step down as well."""
        budget = TurnBudget(BudgetPolicy(max_cost_usd=1.0))
        budget.record(cost_usd=0.50)
        budget.record(cost_usd=None)
        assert budget.cost_usd == pytest.approx(0.50)
        assert budget.cost_per_step == pytest.approx(0.50)
        assert budget.cost_is_partial is True

    def test_a_partially_priced_turn_says_the_ceiling_is_unenforceable(self):
        budget = TurnBudget(BudgetPolicy(max_cost_usd=10.0))
        budget.record(cost_usd=0.10)
        budget.record(cost_usd=None)
        verdict = budget.check()
        assert verdict.cost_is_partial is True
        assert "cannot be enforced" in verdict.reason

    def test_a_floor_already_over_the_ceiling_still_trips(self):
        """The conclusion does not depend on the unknown gap: what is known is
        already too much."""
        budget = TurnBudget(BudgetPolicy(max_cost_usd=1.0, action=BudgetAction.STOP))
        budget.record(cost_usd=1.5)
        budget.record(cost_usd=None)
        verdict = budget.check()
        assert verdict.should_stop is True
        assert "unpriced step" in verdict.reason

    def test_a_fully_unpriced_turn_has_no_cost_per_step(self):
        budget = TurnBudget()
        budget.record(cost_usd=None)
        assert budget.cost_per_step is None


class TestProjection:
    def test_it_warns_about_the_course_before_the_arrival(self):
        budget = TurnBudget(BudgetPolicy(max_steps=10, max_cost_usd=1.0))
        for _ in range(2):
            budget.record(cost_usd=0.15)
        verdict = budget.check()
        # 0.15/step x 10 steps = $1.50, over the $1.00 ceiling, while only
        # $0.30 has actually been spent.
        assert verdict.projected_cost_usd == pytest.approx(1.50)
        assert verdict.state is BudgetState.WARNING
        assert "on course for" in verdict.reason

    def test_there_is_no_projection_without_a_step_limit(self):
        budget = TurnBudget(BudgetPolicy(max_cost_usd=1.0))
        budget.record(cost_usd=0.10)
        assert budget.check().projected_cost_usd is None

    def test_there_is_no_projection_without_a_priced_step(self):
        """Extrapolating from zero observations gives a confident 0.00, which
        reads as 'this turn is free'."""
        budget = TurnBudget(BudgetPolicy(max_steps=10, max_cost_usd=1.0))
        budget.record(cost_usd=None)
        assert budget.check().projected_cost_usd is None


class TestNoMidTurnConstrain:
    def test_constrain_is_not_an_action(self):
        """Deliberately absent. Changing model inside a conversation changes
        behaviour, and with preserved-thinking models it invalidates reasoning
        blocks already in the history -- so the design's `constrain` action was
        resolved as unsafe rather than implemented."""
        assert set(BudgetAction) == {
            BudgetAction.REPORT,
            BudgetAction.WARN,
            BudgetAction.STOP,
        }
        with pytest.raises(ValueError):
            BudgetAction("constrain")


class TestAuthority:
    def test_a_policy_override_keeps_the_meter(self):
        """`caller > policy > config`: a caller may raise its own limit, but
        not wipe what the turn has already spent by doing so."""
        budget = TurnBudget(BudgetPolicy(max_steps=2))
        budget.record(cost_usd=0.25)
        budget.record(cost_usd=0.25)
        assert budget.check().state is BudgetState.EXCEEDED

        raised = budget.with_policy(BudgetPolicy(max_steps=10))
        assert raised.steps == 2
        assert raised.cost_usd == pytest.approx(0.50)
        assert raised.check().state is BudgetState.OK


class TestFromConfig:
    def _get(self, **values):
        return lambda key, default=None: values.get(key, default)

    def test_an_empty_config_is_unbounded(self):
        assert BudgetPolicy.from_config(self._get()).is_bounded is False

    def test_dials_are_read(self):
        policy = BudgetPolicy.from_config(
            self._get(**{
                "routing.budget.max_steps": 200,
                "routing.budget.max_cost_usd": 5.0,
                "routing.budget.warn_at_fraction": 0.5,
                "routing.budget.action": "stop",
            })
        )
        assert policy.max_steps == 200
        assert policy.max_cost_usd == pytest.approx(5.0)
        assert policy.warn_at_fraction == pytest.approx(0.5)
        assert policy.action is BudgetAction.STOP

    def test_an_unknown_action_falls_back_to_warn_rather_than_stopping(self):
        """A typo must not become an outage, and must not silently become
        enforcement either."""
        policy = BudgetPolicy.from_config(
            self._get(**{"routing.budget.action": "halt-everything"})
        )
        assert policy.action is BudgetAction.WARN

    def test_a_non_numeric_limit_is_ignored_rather_than_coerced(self):
        """`float()` of a stray object or a mock succeeds, and would install a
        budget nobody configured. `float(MagicMock())` returning 1.0 has
        already caused one bug in this codebase."""
        from unittest.mock import MagicMock

        policy = BudgetPolicy.from_config(
            self._get(**{
                "routing.budget.max_steps": MagicMock(),
                "routing.budget.max_cost_usd": "lots",
            })
        )
        assert policy.max_steps is None
        assert policy.max_cost_usd is None

    def test_a_boolean_is_not_a_limit(self):
        """`True` is an int, and `int(True) == 1` would be a one-step budget."""
        policy = BudgetPolicy.from_config(
            self._get(**{"routing.budget.max_steps": True})
        )
        assert policy.max_steps is None


class TestVerdictSerialisation:
    def test_it_is_json_safe(self):
        import json

        budget = TurnBudget(BudgetPolicy(max_steps=4, max_cost_usd=1.0))
        budget.record(cost_usd=0.3, input_tokens=100, output_tokens=20)
        payload = budget.check().as_dict()
        assert json.loads(json.dumps(payload))["steps"] == 1
        assert payload["state"] == "warning" or payload["state"] == "ok"
