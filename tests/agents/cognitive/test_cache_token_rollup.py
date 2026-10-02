# tests/agents/cognitive/test_cache_token_rollup.py
"""Cache token counts must survive the whole way to session stats.

The chain is PhaseUsage -> CycleIteration -> EnhancedAgentState -> the host
-> record_agent_usage -> SessionTokenStats. `SessionTokenStats` has always
had a `total_cached_tokens` field and always read it from the interaction
record — but nothing ever wrote that key, so it was permanently zero even
once the cycle had measured it. These tests pin every link.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.cognitive.models import (
    CycleIteration,
    EnhancedAgentState,
    PhaseUsage,
    PlanOutput,
    ThinkOutput,
)


def usage(prompt, completion, cached=0, written=0, cost=0.0):
    return PhaseUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        cache_write_tokens=written,
        cost=cost,
        provider="anthropic",
        model="claude-opus-5-5",
    )


class TestIterationRollup:
    def _iteration(self):
        it = CycleIteration(iteration_number=1)
        it.plan_output = PlanOutput(
            plan_steps=["a"],
            usage=usage(500_000, 100, cached=490_000, written=1_000, cost=0.11),
        )
        it.think_output = ThinkOutput(
            thought="t",
            usage=usage(600_000, 200, cached=590_000, written=0, cost=0.13),
        )
        it.update_token_totals_from_phases()
        return it

    def test_cached_tokens_sum_across_phases(self):
        assert self._iteration().total_cached_tokens == 1_080_000

    def test_cache_writes_sum_across_phases(self):
        assert self._iteration().total_cache_write_tokens == 1_000

    def test_prompt_total_is_unaffected(self):
        # Regression guard: cached is a subset, so it must not be added on.
        assert self._iteration().total_prompt_tokens == 1_100_000

    def test_history_summary_exposes_them(self):
        summary = self._iteration().to_history_summary()
        assert summary["total_cached_tokens"] == 1_080_000
        assert summary["total_cache_write_tokens"] == 1_000

    def test_iteration_without_cache_data_reports_zero(self):
        it = CycleIteration(iteration_number=1)
        it.plan_output = PlanOutput(plan_steps=["a"], usage=usage(1_000, 100, cost=0.01))
        it.update_token_totals_from_phases()
        assert it.total_cached_tokens == 0
        assert it.total_cache_write_tokens == 0


class TestStateRollup:
    def test_state_accumulates_across_iterations(self):
        state = EnhancedAgentState(goal="g", session_id="s")
        for _ in range(3):
            it = CycleIteration(iteration_number=1)
            it.plan_output = PlanOutput(
                plan_steps=["a"], usage=usage(1_000, 10, cached=900, written=50)
            )
            it.update_token_totals_from_phases()
            state.add_iteration(it)
        assert state.total_cached_tokens == 2_700
        assert state.total_cache_write_tokens == 150


@pytest.mark.asyncio
class TestRecordAgentUsage:
    """record_agent_usage is the boundary where the counts were dropped."""

    def _core(self, session):
        core = object.__new__(__import__("llmcore.api", fromlist=["LLMCore"]).LLMCore)
        manager = MagicMock()
        manager.load_or_create_session = AsyncMock(return_value=session)
        manager.save_session = AsyncMock()
        core._session_manager = manager
        return core

    def _session(self):
        session = MagicMock()
        session.metadata = {}
        session.messages = []
        return session

    async def test_counts_reach_the_interaction_record(self):
        session = self._session()
        core = self._core(session)
        await core.record_agent_usage("s", [{
            "provider": "anthropic", "model": "claude-opus-5-5",
            "prompt_tokens": 922_409, "completion_tokens": 962,
            "cached_tokens": 922_403, "cache_write_tokens": 1_000, "cost": 0.2,
        }])
        record = session.metadata["interactions"][0]
        assert record["cached_tokens"] == 922_403
        assert record["cache_write_tokens"] == 1_000

    async def test_cached_count_is_clamped_to_the_prompt(self):
        # A cached count above the prompt is a reporting error; letting it
        # through would imply a cache-hit ratio over 100%.
        session = self._session()
        core = self._core(session)
        await core.record_agent_usage("s", [{
            "prompt_tokens": 100, "completion_tokens": 10,
            "cached_tokens": 999_999,
        }])
        assert session.metadata["interactions"][0]["cached_tokens"] == 100

    async def test_records_without_cache_keys_still_work(self):
        session = self._session()
        core = self._core(session)
        await core.record_agent_usage("s", [{
            "prompt_tokens": 1_000, "completion_tokens": 500, "cost": 0.01,
        }])
        record = session.metadata["interactions"][0]
        assert record["cached_tokens"] == 0
        assert record["cache_write_tokens"] == 0
        assert record["total_tokens"] == 1_500


@pytest.mark.asyncio
class TestSessionStats:
    async def test_stats_aggregate_both_counts(self):
        from llmcore.api import LLMCore

        session = MagicMock()
        session.messages = []
        session.metadata = {"interactions": [
            {"prompt_tokens": 1_000, "completion_tokens": 100,
             "cached_tokens": 900, "cache_write_tokens": 50, "model": "m"},
            {"prompt_tokens": 2_000, "completion_tokens": 200,
             "cached_tokens": 1_800, "cache_write_tokens": 0, "model": "m"},
        ]}
        core = object.__new__(LLMCore)
        manager = MagicMock()
        manager.load_or_create_session = AsyncMock(return_value=session)
        core._session_manager = manager

        stats = await core.get_session_token_stats("s")
        assert stats.total_cached_tokens == 2_700
        assert stats.total_cache_write_tokens == 50
        # cached is a subset, so prompt totals must not include it twice
        assert stats.total_prompt_tokens == 3_000

    async def test_legacy_records_without_the_keys_aggregate_to_zero(self):
        from llmcore.api import LLMCore

        session = MagicMock()
        session.messages = []
        session.metadata = {"interactions": [
            {"prompt_tokens": 10, "completion_tokens": 5, "model": "m"},
        ]}
        core = object.__new__(LLMCore)
        manager = MagicMock()
        manager.load_or_create_session = AsyncMock(return_value=session)
        core._session_manager = manager

        stats = await core.get_session_token_stats("s")
        assert stats.total_cached_tokens == 0
        assert stats.total_cache_write_tokens == 0
