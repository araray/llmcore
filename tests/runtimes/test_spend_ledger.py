# tests/runtimes/test_spend_ledger.py
"""The daily spend ledger and cap.

The gap this fills was found by writing the agent-lens migration guide:
agent-lens keeps a spend ledger and a ``budget_units_per_day`` cap, and llmcore
had **no per-day budget at all**. Its ceilings were per *runtime*
(``max_spend``) and per *turn* (``[routing.budget]``) — so ten runtimes, each
under its own ceiling, could spend ten times the intent with nothing objecting.

Two properties carry most of the weight here, and both are the same lesson this
subsystem keeps relearning: **one cap cannot span two units**, because there is
no defensible rate between Colab's compute units and dollars; and **unknown
spend is not zero**, because a day of unpriceable runtimes must not read as a
free day.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmcore.runtimes.ledger import SpendEntry, SpendLedger
from llmcore.runtimes.models import CostUnit

NOW = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)


def entry(amount, *, unit=CostUnit.USD, at=NOW, name="r"):
    return SpendEntry(at=at, runtime="gpuai", name=name, amount=amount, unit=unit)


@pytest.fixture
def ledger(tmp_path):
    return SpendLedger(tmp_path / "spend.jsonl")


class TestRecordingAndTotals:
    def test_an_empty_ledger_totals_zero_without_a_file(self, ledger):
        total = ledger.total(CostUnit.USD, now=NOW)
        assert total.amount == 0.0
        assert total.runs == 0
        assert total.is_partial is False

    def test_entries_round_trip(self, ledger):
        ledger.record(entry(1.25))
        reloaded = SpendLedger(ledger.path)
        assert [e.amount for e in reloaded.entries()] == [1.25]

    def test_totals_sum_within_the_day(self, ledger):
        ledger.record(entry(1.50, at=NOW - timedelta(hours=2)))
        ledger.record(entry(2.25, at=NOW - timedelta(minutes=5)))
        total = ledger.total(CostUnit.USD, now=NOW)
        assert total.amount == pytest.approx(3.75)
        assert total.runs == 2

    def test_yesterday_is_not_in_todays_total(self, ledger):
        """The window is the UTC day, so a cap has a stateable reset time. A
        cap that moved with the machine's timezone would not."""
        ledger.record(entry(99.0, at=NOW - timedelta(days=1)))
        ledger.record(entry(1.0, at=NOW))
        assert ledger.total(CostUnit.USD, now=NOW).amount == pytest.approx(1.0)


class TestUnitsDoNotMix:
    def test_a_total_counts_only_its_own_unit(self, ledger):
        ledger.record(entry(5.0, unit=CostUnit.USD))
        ledger.record(entry(40.0, unit=CostUnit.COMPUTE_UNIT))
        assert ledger.total(CostUnit.USD, now=NOW).amount == pytest.approx(5.0)
        assert ledger.total(CostUnit.COMPUTE_UNIT, now=NOW).amount == pytest.approx(40.0)

    def test_per_day_reporting_keeps_units_apart(self, ledger):
        ledger.record(entry(5.0, unit=CostUnit.USD))
        ledger.record(entry(40.0, unit=CostUnit.COMPUTE_UNIT))
        days = ledger.days()
        day = NOW.date()
        assert set(days[day]) == {CostUnit.USD, CostUnit.COMPUTE_UNIT}


class TestUnknownIsNotZero:
    def test_an_unpriced_run_counts_as_a_run_not_as_zero(self, ledger):
        ledger.record(entry(None))
        total = ledger.total(CostUnit.USD, now=NOW)
        assert total.runs == 1
        assert total.unpriced_runs == 1
        assert total.amount == 0.0
        assert total.is_partial is True

    def test_a_partial_total_says_so_when_rendered(self, ledger):
        ledger.record(entry(10.0))
        ledger.record(entry(None))
        rendered = ledger.total(CostUnit.USD, now=NOW).rendered()
        assert "$10.00" in rendered
        assert "1 unpriced run" in rendered

    def test_a_fully_priced_total_does_not_add_a_caveat(self, ledger):
        ledger.record(entry(10.0))
        assert ledger.total(CostUnit.USD, now=NOW).rendered() == "$10.00"


class TestRobustness:
    def test_a_corrupt_line_does_not_lose_the_good_ones(self, ledger):
        ledger.record(entry(1.0))
        with ledger.path.open("a") as handle:
            handle.write("{not json at all\n")
        ledger.record(entry(2.0))
        reloaded = SpendLedger(ledger.path)
        assert reloaded.total(CostUnit.USD, now=NOW).amount == pytest.approx(3.0)

    def test_an_unknown_unit_in_the_file_does_not_raise(self, ledger):
        with ledger.path.open("a") as handle:
            handle.write(
                '{"at": "2026-10-10T10:00:00+00:00", "runtime": "x", "name": "y", '
                '"amount": 1.0, "unit": "galleons"}\n'
            )
        assert SpendLedger(ledger.path).entries()[0].unit is CostUnit.COMPUTE_UNIT

    def test_a_non_numeric_amount_becomes_unknown_not_an_error(self, ledger):
        with ledger.path.open("a") as handle:
            handle.write(
                '{"at": "2026-10-10T10:00:00+00:00", "runtime": "x", "name": "y", '
                '"amount": "lots", "unit": "usd"}\n'
            )
        total = SpendLedger(ledger.path).total(CostUnit.USD, now=NOW)
        assert total.unpriced_runs == 1

    def test_a_failed_write_never_raises(self, tmp_path, monkeypatch):
        """``record`` is called from teardown. If recording the cost of
        releasing a runtime could fail the release, the ledger would be a way
        to keep paying for something."""
        ledger = SpendLedger(tmp_path / "nested" / "spend.jsonl")

        def refuse(*a, **k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr("pathlib.Path.mkdir", refuse)
        ledger.record(entry(1.0))  # must not raise

    def test_pruning_keeps_recent_entries_only(self, ledger):
        ledger.record(entry(1.0, at=NOW - timedelta(days=200)))
        ledger.record(entry(2.0, at=NOW))
        removed = ledger.prune(keep_days=90, now=NOW)
        assert removed == 1
        assert [e.amount for e in SpendLedger(ledger.path).entries()] == [2.0]

    def test_pruning_nothing_rewrites_nothing(self, ledger):
        ledger.record(entry(2.0, at=NOW))
        assert ledger.prune(keep_days=90, now=NOW) == 0


class TestTheCap:
    """Enforced before ``up()``, because a cap applied afterwards is not one."""

    def _manager(self, tmp_path, **config):
        from llmcore.runtimes.manager import RuntimeManager

        values = {"runtimes.enabled": True, "runtimes.state_dir": str(tmp_path)}
        values.update(config)
        return RuntimeManager(config_get=lambda key, default=None: values.get(key, default))

    def _plan(self, unit=CostUnit.USD):
        from llmcore.runtimes.models import ModelSpec, Plan

        return Plan(
            spec=ModelSpec(repo_id="a/b"), sku="1xa100", cost_unit=unit,
            estimated_cost_per_hour=1.0, fits=True,
        )

    def test_no_cap_means_no_refusal(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.ledger.record(entry(1_000_000.0))
        manager._enforce_daily_budget(self._plan())  # must not raise

    def test_the_cap_refuses_once_reached(self, tmp_path):
        from llmcore.runtimes.manager import DailyBudgetExceededError

        manager = self._manager(tmp_path, **{"runtimes.budget.max_usd_per_day": 10.0})
        manager.ledger.record(entry(10.0))
        with pytest.raises(DailyBudgetExceededError, match="today's provisioning budget"):
            manager._enforce_daily_budget(self._plan())

    def test_under_the_cap_is_allowed(self, tmp_path):
        manager = self._manager(tmp_path, **{"runtimes.budget.max_usd_per_day": 10.0})
        manager.ledger.record(entry(9.99))
        manager._enforce_daily_budget(self._plan())

    def test_the_other_unit_is_unaffected(self, tmp_path):
        """A dollar cap must not be spent down by Colab compute units."""
        manager = self._manager(tmp_path, **{"runtimes.budget.max_usd_per_day": 10.0})
        manager.ledger.record(entry(500.0, unit=CostUnit.COMPUTE_UNIT))
        manager._enforce_daily_budget(self._plan())

    def test_a_plan_with_no_unit_is_not_blocked(self, tmp_path):
        """Refusing it would turn "we do not know the price" into "you may not
        have it", which is a different policy than the one configured."""
        manager = self._manager(tmp_path, **{"runtimes.budget.max_usd_per_day": 10.0})
        manager.ledger.record(entry(50.0))
        manager._enforce_daily_budget(self._plan(unit=None))

    def test_the_refusal_mentions_the_unpriced_runs(self, tmp_path):
        from llmcore.runtimes.manager import DailyBudgetExceededError

        manager = self._manager(tmp_path, **{"runtimes.budget.max_usd_per_day": 10.0})
        manager.ledger.record(entry(10.0))
        manager.ledger.record(entry(None))
        with pytest.raises(DailyBudgetExceededError, match="could not be priced"):
            manager._enforce_daily_budget(self._plan())

    def test_a_nonsense_cap_is_ignored_rather_than_coerced(self, tmp_path):
        """``float(MagicMock())`` returns 1.0 and has caused a bug here before."""
        from unittest.mock import MagicMock

        manager = self._manager(
            tmp_path, **{"runtimes.budget.max_usd_per_day": MagicMock()}
        )
        assert manager._daily_caps == {}

    def test_a_boolean_cap_is_ignored(self, tmp_path):
        manager = self._manager(tmp_path, **{"runtimes.budget.max_usd_per_day": True})
        assert manager._daily_caps == {}


class TestRecordingOnTeardown:
    def test_a_released_runtime_is_written_to_the_ledger(self, tmp_path):
        from llmcore.runtimes.manager import RuntimeManager
        from llmcore.runtimes.models import RuntimeHandle

        manager = RuntimeManager(
            config_get=lambda k, d=None: {"runtimes.state_dir": str(tmp_path)}.get(k, d)
        )
        manager.record_spend(
            RuntimeHandle(
                name="qwen", runtime="gpuai", external_id="inst-1",
                base_url="http://x/v1", served_model="a/b",
                spend_used=3.25, spend_unit=CostUnit.USD,
            )
        )
        total = manager.ledger.total(CostUnit.USD)
        assert total.amount == pytest.approx(3.25)
        assert total.runs == 1

    def test_an_unpriced_runtime_is_recorded_as_unknown(self, tmp_path):
        from llmcore.runtimes.manager import RuntimeManager
        from llmcore.runtimes.models import RuntimeHandle

        manager = RuntimeManager(
            config_get=lambda k, d=None: {"runtimes.state_dir": str(tmp_path)}.get(k, d)
        )
        manager.record_spend(
            RuntimeHandle(
                name="mystery", runtime="deepinfra", external_id="d-1",
                base_url="http://x/v1", served_model="a/b",
                spend_used=None, spend_unit=CostUnit.USD,
            )
        )
        total = manager.ledger.total(CostUnit.USD)
        assert total.unpriced_runs == 1
        assert total.is_partial is True


class TestTheCapStopsProvisioningForReal:
    """The unit tests above call ``_enforce_daily_budget`` directly. This one
    goes through ``up()``, because the property that matters is not "the check
    returns an error" but "the backend is never asked to provision" -- a cap
    that refuses *after* compute is assigned is not a cap.
    """

    def _manager(self, tmp_path, backend, cap=10.0):
        from llmcore.runtimes.manager import RuntimeManager

        values = {
            "runtimes.enabled": True,
            "runtimes.state_dir": str(tmp_path),
            "runtimes.budget.max_usd_per_day": cap,
            "runtimes.defaults.confirm_spend": False,
        }
        return RuntimeManager(
            {"fake": backend},
            config_get=lambda key, default=None: values.get(key, default),
        )

    def _priced_plan(self):
        from llmcore.runtimes.models import ModelSpec, Plan

        return Plan(
            spec=ModelSpec(repo_id="a/b"),
            sku="1xa100",
            cost_unit=CostUnit.USD,
            estimated_cost_per_hour=2.0,
            fits=True,
        )

    @pytest.mark.asyncio
    async def test_the_backend_is_never_asked_when_the_budget_is_spent(self, tmp_path):
        from llmcore.runtimes.manager import DailyBudgetExceededError
        from llmcore.runtimes.testing import FakeRuntime

        backend = FakeRuntime("fake")
        manager = self._manager(tmp_path, backend)
        manager.ledger.record(entry(12.0))

        with pytest.raises(DailyBudgetExceededError):
            await manager.up(
                "a/b", name="r", backend="fake", plan=self._priced_plan(),
                confirm_spend=True, attach=False,
            )
        assert backend.called("up") == 0
        assert backend.running == {}

    @pytest.mark.asyncio
    async def test_provisioning_proceeds_under_the_cap(self, tmp_path):
        from llmcore.runtimes.testing import FakeRuntime

        backend = FakeRuntime("fake")
        manager = self._manager(tmp_path, backend)
        manager.ledger.record(entry(1.0))

        await manager.up(
            "a/b", name="r", backend="fake", plan=self._priced_plan(),
            confirm_spend=True, attach=False,
        )
        assert backend.called("up") == 1

    @pytest.mark.asyncio
    async def test_stopping_a_runtime_adds_it_to_the_ledger(self, tmp_path):
        from llmcore.runtimes.testing import FakeRuntime

        backend = FakeRuntime("fake")
        manager = self._manager(tmp_path, backend)
        handle = await manager.up(
            "a/b", name="r", backend="fake", plan=self._priced_plan(),
            confirm_spend=True, attach=False,
        )
        handle.spend_used = 4.5
        handle.spend_unit = CostUnit.USD
        manager.state.save(handle)

        assert await manager.down("r") is True
        total = manager.ledger.total(CostUnit.USD)
        assert total.amount == pytest.approx(4.5)
        assert total.runs == 1
