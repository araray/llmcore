# tests/agents/cognitive/test_phase_usage.py
"""Token/cost accounting tests (plan §2.7).

``extract_usage`` turns provider usage blocks into typed ``PhaseUsage``
records with model-card pricing; iteration totals sum them; state totals
roll up on ``complete_iteration``; the circuit breaker receives PER-ITERATION
cost deltas (its ``check`` adds internally — running totals double-count);
``record_agent_usage`` writes session interactions that
``get_session_token_stats`` aggregates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from llmcore.agents.cognitive.models import (
    ConfidenceLevel,
    CycleIteration,
    EnhancedAgentState,
    PlanOutput,
    ReflectOutput,
    TerminationReason,
    ThinkOutput,
    ValidateOutput,
    ValidationResult,
)
from llmcore.agents.cognitive.phases.cycle import CognitiveCycle
from llmcore.agents.cognitive.phases.usage import PhaseUsage, extract_usage
from llmcore.config.agents_config import AgentsConfig


def _response(prompt: int, completion: int, total: int | None = None) -> dict:
    usage: dict = {"prompt_tokens": prompt, "completion_tokens": completion}
    if total is not None:
        usage["total_tokens"] = total
    return {"choices": [{"message": {"content": "x"}}], "usage": usage}


# =============================================================================
# extract_usage shapes
# =============================================================================


class TestExtractUsage:
    def test_reads_usage_block(self):
        usage = extract_usage(_response(100, 50, 150), "test_provider", "test_model")

        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.provider == "test_provider"
        assert usage.model == "test_model"

    def test_total_computed_when_missing(self):
        usage = extract_usage(_response(10, 5), "p", "m")
        assert usage is not None
        assert usage.total_tokens == 15

    @pytest.mark.parametrize(
        "response",
        [
            None,
            "not a dict",
            {"choices": []},
            {"usage": None},
            {"usage": "text"},
            {"usage": {}},
            {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        ],
    )
    def test_absent_usage_returns_none(self, response):
        assert extract_usage(response, "p", "m") is None

    def test_cost_from_model_card_pricing(self):
        usage = extract_usage(_response(1000, 500), "openai", "gpt-4o")

        assert usage is not None
        # gpt-4o card: $2.5/M input + $10/M output.
        assert usage.cost == pytest.approx(0.0025 + 0.005)

    def test_unknown_model_cost_is_none(self):
        usage = extract_usage(_response(1000, 500), "no_such_provider", "no_such_model")
        assert usage is not None
        assert usage.cost is None

    def test_missing_provider_or_model_cost_is_none(self):
        assert extract_usage(_response(10, 5), None, "gpt-4o").cost is None
        assert extract_usage(_response(10, 5), "openai", None).cost is None

    def test_pricing_lookup_memoized(self):
        from llmcore.agents.cognitive.phases.usage import _pricing_for

        _pricing_for.cache_clear()
        extract_usage(_response(10, 5), "openai", "gpt-4o")
        extract_usage(_response(20, 10), "openai", "gpt-4o")

        info = _pricing_for.cache_info()
        assert info.hits >= 1
        assert info.misses == 1


# =============================================================================
# Iteration totals + state roll-up
# =============================================================================


def _usage(prompt: int, completion: int, cost: float | None) -> PhaseUsage:
    return PhaseUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost=cost,
        provider="p",
        model="m",
    )


class TestIterationTotals:
    def _full_iteration(self) -> CycleIteration:
        iteration = CycleIteration(iteration_number=1)
        iteration.plan_output = PlanOutput(
            plan_steps=["a"], tokens_used=15, usage=_usage(10, 5, 0.01)
        )
        iteration.think_output = ThinkOutput(
            thought="t", reasoning_tokens=30, usage=_usage(20, 10, 0.02)
        )
        iteration.validate_output = ValidateOutput(
            result=ValidationResult.APPROVED,
            confidence=ConfidenceLevel.HIGH,
            tokens_used=12,
            usage=_usage(8, 4, None),  # unknown pricing
        )
        iteration.reflect_output = ReflectOutput(
            evaluation="e", tokens_used=9, usage=_usage(6, 3, 0.03)
        )
        return iteration

    def test_update_token_totals_sums_phase_usage(self):
        iteration = self._full_iteration()

        total = iteration.update_token_totals_from_phases()

        assert total == 15 + 30 + 12 + 9
        assert iteration.total_tokens_used == total
        assert iteration.total_prompt_tokens == 10 + 20 + 8 + 6
        assert iteration.total_completion_tokens == 5 + 10 + 4 + 3
        assert iteration.total_cost == pytest.approx(0.06)

    def test_history_summary_includes_cost_fields(self):
        iteration = self._full_iteration()
        iteration.update_token_totals_from_phases()

        summary = iteration.to_history_summary()

        assert summary["total_tokens_used"] == 66
        assert summary["total_prompt_tokens"] == 44
        assert summary["total_completion_tokens"] == 22
        assert summary["total_cost"] == pytest.approx(0.06)

    def test_complete_iteration_rolls_tokens_and_cost_into_state(self):
        state = EnhancedAgentState(goal="g", session_id="s")
        iteration = state.start_iteration(1)
        iteration.think_output = ThinkOutput(
            thought="t", reasoning_tokens=30, usage=_usage(20, 10, 0.02)
        )
        iteration.update_token_totals_from_phases()

        state.complete_iteration(success=True)

        assert state.total_tokens_used == 30
        assert state.total_cost == pytest.approx(0.02)

    def test_state_cost_survives_snapshot_round_trip(self):
        state = EnhancedAgentState(goal="g", session_id="s")
        state.total_cost = 0.125
        state.total_tokens_used = 42

        restored = EnhancedAgentState.from_resume_snapshot(state.to_resume_snapshot())

        assert restored.total_cost == pytest.approx(0.125)
        assert restored.total_tokens_used == 42


# =============================================================================
# Circuit breaker: per-iteration cost deltas (no double-count)
# =============================================================================


def _costed_iteration(number: int, cost: float, tokens: int = 100) -> CycleIteration:
    iteration = CycleIteration(iteration_number=number)
    iteration.think_output = MagicMock()
    iteration.think_output.is_final_answer = False
    iteration.think_output.proposed_action = None
    iteration.observe_output = None
    iteration.reflect_output = MagicMock()
    iteration.reflect_output.step_completed = False
    iteration.update_output = MagicMock()
    iteration.update_output.should_continue = True
    iteration.total_tokens_used = tokens
    iteration.total_cost = cost
    iteration.mark_completed(success=True)
    return iteration


def _make_cycle(bundled_prompt_registry) -> CognitiveCycle:
    tool_manager = Mock()
    tool_manager.get_tool_definitions = Mock(side_effect=lambda names=None: [])
    return CognitiveCycle(
        provider_manager=Mock(),
        memory_manager=Mock(),
        storage_manager=Mock(),
        tool_manager=tool_manager,
        prompt_registry=bundled_prompt_registry,
    )


class TestCircuitBreakerCostDeltas:
    @pytest.mark.asyncio
    async def test_cost_limit_trips_on_true_total_not_double_count(
        self, bundled_prompt_registry
    ):
        """Three iterations at $0.40 total $1.20 — the $1.00 limit must trip
        on iteration 3. Passing running totals (the old bug) would feed the
        breaker 0.4 + 0.8 and trip a run that only spent $0.80."""
        config = AgentsConfig()
        config.circuit_breaker.max_total_cost = 1.0
        config.circuit_breaker.max_iterations = 100
        config.circuit_breaker.max_execution_time_seconds = 3600
        config.circuit_breaker.progress_stall_threshold = 50

        cycle = _make_cycle(bundled_prompt_registry)
        calls: list[int] = []

        async def costed_run_iteration(*args, **kwargs):
            calls.append(1)
            return _costed_iteration(len(calls), cost=0.4)

        cycle.run_iteration = Mock(side_effect=costed_run_iteration)  # type: ignore[method-assign]

        state = EnhancedAgentState(goal="g", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=10,
            agents_config=config,
        )

        assert "circuit breaker" in result.lower()
        assert "cost" in result.lower()
        assert state.termination_reason == TerminationReason.CIRCUIT_BREAKER.value
        # Tripped at the true $1.20 crossing — NOT at iteration 2.
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_streaming_updates_carry_tokens_and_cost(
        self, bundled_prompt_registry
    ):
        config = AgentsConfig()
        config.circuit_breaker.enabled = False

        cycle = _make_cycle(bundled_prompt_registry)
        counter: list[int] = []

        async def run_iteration(*args, **kwargs):
            counter.append(1)
            state = kwargs.get("agent_state") or args[0]
            if len(counter) >= 2:
                state.is_finished = True
                state.final_answer = "done"
            return _costed_iteration(len(counter), cost=0.01, tokens=42)

        cycle.run_iteration = Mock(side_effect=run_iteration)  # type: ignore[method-assign]

        state = EnhancedAgentState(goal="g", session_id="s")
        updates = []
        async for update in cycle.run_streaming(
            agent_state=state,
            session_id="s",
            max_iterations=5,
            agents_config=config,
        ):
            updates.append(update)

        per_iteration = [u for u in updates if u.tokens_used]
        assert per_iteration, "expected at least one per-iteration update"
        assert all(u.tokens_used == 42 for u in per_iteration)
        assert all(u.cost == pytest.approx(0.01) for u in per_iteration)


# =============================================================================
# record_agent_usage → get_session_token_stats
# =============================================================================


class _SessionStore:
    """Minimal async session manager backed by real ChatSession objects."""

    def __init__(self, session):
        self.sessions = {session.id: session}
        self.saved = []

    async def load_or_create_session(self, session_id):
        return self.sessions.get(session_id)

    async def save_session(self, session):
        self.saved.append(session)
        self.sessions[session.id] = session


class TestRecordAgentUsage:
    def _llm_with_session(self):
        from llmcore.models import ChatSession

        session = ChatSession(id="darwin-session", name="Darwin run")
        llm = MagicMock()
        llm._session_manager = _SessionStore(session)
        return llm, session

    @pytest.mark.asyncio
    async def test_records_normalize_and_persist(self):
        from llmcore.api import LLMCore

        llm, session = self._llm_with_session()

        await LLMCore.record_agent_usage(
            llm,
            "darwin-session",
            [
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.001,
                },
                {"model": "gpt-4o", "prompt_tokens": "20", "completion_tokens": 10},
            ],
        )

        interactions = session.metadata["interactions"]
        assert len(interactions) == 2
        first = interactions[0]
        assert first["provider"] == "openai"
        assert first["model"] == "gpt-4o"
        assert first["prompt_tokens"] == 100
        assert first["completion_tokens"] == 50
        assert first["total_tokens"] == 150
        assert first["cost"] == pytest.approx(0.001)
        assert first["source"] == "darwin"
        assert first["timestamp"]
        second = interactions[1]
        assert second["provider"] == "unknown"
        assert second["prompt_tokens"] == 20
        assert second["total_tokens"] == 30
        assert second["cost"] is None
        # The session was persisted.
        assert llm._session_manager.saved

    @pytest.mark.asyncio
    async def test_token_stats_report_darwin_turns(self):
        from llmcore.api import LLMCore

        llm, _ = self._llm_with_session()

        await LLMCore.record_agent_usage(
            llm,
            "darwin-session",
            [
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "timestamp": "2026-07-08T10:00:00Z",
                },
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                    "timestamp": "2026-07-08T10:01:00Z",
                },
            ],
        )

        stats = await LLMCore.get_session_token_stats(llm, "darwin-session")

        assert stats.total_prompt_tokens == 300
        assert stats.total_completion_tokens == 130
        assert stats.total_tokens == 430
        assert stats.interaction_count == 2
        assert stats.by_model["gpt-4o"]["count"] == 2

    @pytest.mark.asyncio
    async def test_invalid_records_skipped(self):
        from llmcore.api import LLMCore

        llm, session = self._llm_with_session()

        await LLMCore.record_agent_usage(
            llm,
            "darwin-session",
            ["not-a-dict", {"prompt_tokens": "junk", "completion_tokens": None}],
        )

        interactions = session.metadata["interactions"]
        # The non-dict record is skipped; the junk-valued one normalizes to 0.
        assert len(interactions) == 1
        assert interactions[0]["prompt_tokens"] == 0
        assert interactions[0]["total_tokens"] == 0


# =============================================================================
# Forced finalize usage accounting
# =============================================================================


class TestForcedFinalizeUsage:
    @pytest.mark.asyncio
    async def test_force_finalize_rolls_usage_into_state(self, bundled_prompt_registry):
        provider = Mock()
        provider.get_name = Mock(return_value="test_provider")
        provider.default_model = "test_model"
        provider.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "Synthesized"}}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30},
            }
        )
        provider.extract_response_content = Mock(return_value="Synthesized")

        cycle = _make_cycle(bundled_prompt_registry)
        cycle.provider_manager.get_provider = Mock(return_value=provider)

        state = EnhancedAgentState(goal="g", session_id="s")
        finalized = await cycle._force_finalize(
            state, reason=TerminationReason.SYNTHESIS_FALLBACK
        )

        assert finalized is True
        assert state.total_tokens_used == 30
