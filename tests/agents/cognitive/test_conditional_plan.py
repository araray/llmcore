# tests/agents/cognitive/test_conditional_plan.py
"""Conditional PLAN tests (plan §2.5).

Plan-before-observe measurably hurts simple/knowledge tasks, so the PLAN
phase is gated by ``PlanningConfig.mode``: ``always``/``first`` keep the
legacy predicate, ``complex_only`` (the new default) additionally requires
moderate/complex goal complexity, and ``on_failure`` plans only after a
failed action or an explicit replan request. Reflection-driven replans are
capped by ``max_replans`` in the UPDATE phase.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llmcore.agents.cognitive.models import (
    EnhancedAgentState,
    PerceiveOutput,
    PlanOutput,
    ReflectInput,
    ReflectOutput,
    ThinkInput,
    ThinkOutput,
    UpdateInput,
)
from llmcore.agents.cognitive.phases.cycle import CognitiveCycle, _should_plan
from llmcore.agents.cognitive.phases.update import update_phase
from llmcore.config.agents_config import AgentsConfig, PlanMode, PlanningConfig
from llmcore.models import ToolCall, ToolResult

CYCLE_MODULE = "llmcore.agents.cognitive.phases.cycle"

#: Heuristic GoalClassifier outcomes (verified): short imperative goals are
#: "simple"; the multi-clause goals below hit "moderate"/"complex" patterns.
SIMPLE_GOAL = "What is 2+2?"
MODERATE_GOAL = "Analyze the codebase, refactor the auth module, and add tests"
COMPLEX_GOAL = (
    "Research the history of Rome and write a comprehensive report with citations"
)


def _config(mode: PlanMode, escalation: bool = True) -> AgentsConfig:
    config = AgentsConfig()
    config.circuit_breaker.enabled = False
    config.planning.mode = mode
    config.planning.plan_on_failure_escalation = escalation
    return config


def _state(goal: str = SIMPLE_GOAL, plan: list[str] | None = None) -> EnhancedAgentState:
    state = EnhancedAgentState(goal=goal, session_id="s")
    if plan:
        state.plan = plan
        state.plan_steps_status = ["pending"] * len(plan)
    return state


# =============================================================================
# _should_plan matrix
# =============================================================================


class TestShouldPlanMatrix:
    @pytest.mark.parametrize("mode", [PlanMode.ALWAYS, PlanMode.FIRST])
    def test_always_and_first_keep_legacy_predicate(self, mode):
        config = _config(mode)

        # First iteration plans regardless of plan state.
        assert _should_plan(_state(plan=["a"]), 1, config) is True
        # Later iterations: only an empty plan or a replan request plans.
        assert _should_plan(_state(plan=["a"]), 2, config) is False
        assert _should_plan(_state(), 3, config) is True

        state = _state(plan=["a"])
        state.set_working_memory("plan_needs_update", True)
        assert _should_plan(state, 2, config) is True

    def test_complex_only_skips_simple_goals(self):
        config = _config(PlanMode.COMPLEX_ONLY)
        assert _should_plan(_state(SIMPLE_GOAL), 1, config) is False

    @pytest.mark.parametrize("goal", [MODERATE_GOAL, COMPLEX_GOAL])
    def test_complex_only_plans_moderate_and_complex(self, goal):
        config = _config(PlanMode.COMPLEX_ONLY)
        assert _should_plan(_state(goal), 1, config) is True

    def test_complex_only_working_memory_complexity_wins(self):
        config = _config(PlanMode.COMPLEX_ONLY)

        # single_agent stamps goal_complexity: it overrides the heuristic.
        state = _state(SIMPLE_GOAL)
        state.set_working_memory("goal_complexity", "complex")
        assert _should_plan(state, 1, config) is True

        state = _state(COMPLEX_GOAL)
        state.set_working_memory("goal_complexity", "simple")
        assert _should_plan(state, 1, config) is False

    def test_complex_only_caches_heuristic_classification(self):
        config = _config(PlanMode.COMPLEX_ONLY)
        state = _state(SIMPLE_GOAL)

        _should_plan(state, 1, config)

        assert state.get_working_memory("goal_complexity") == "simple"

    def test_on_failure_requires_failure_or_replan_request(self):
        config = _config(PlanMode.ON_FAILURE)

        # Never plans proactively — not even on the first iteration.
        assert _should_plan(_state(COMPLEX_GOAL), 1, config) is False

        state = _state(COMPLEX_GOAL)
        state.set_working_memory("last_action_failed", True)
        assert _should_plan(state, 2, config) is True

        state = _state(COMPLEX_GOAL)
        state.set_working_memory("plan_needs_update", True)
        assert _should_plan(state, 2, config) is True

    def test_escalation_plans_after_failure_with_empty_plan(self):
        config = _config(PlanMode.COMPLEX_ONLY, escalation=True)

        state = _state(SIMPLE_GOAL)
        state.set_working_memory("last_action_failed", True)
        assert _should_plan(state, 2, config) is True

        # A plan already exists: no escalation.
        state = _state(SIMPLE_GOAL, plan=["a"])
        state.set_working_memory("last_action_failed", True)
        assert _should_plan(state, 2, config) is False

    def test_escalation_disabled_keeps_gate_closed(self):
        config = _config(PlanMode.COMPLEX_ONLY, escalation=False)
        state = _state(SIMPLE_GOAL)
        state.set_working_memory("last_action_failed", True)
        assert _should_plan(state, 2, config) is False

    def test_default_mode_is_complex_only(self):
        state = _state(SIMPLE_GOAL)
        assert _should_plan(state, 1, AgentsConfig()) is False


# =============================================================================
# run_iteration integration
# =============================================================================


def _tool_manager(result: ToolResult) -> Mock:
    manager = Mock()
    manager.is_tool_loaded = Mock(return_value=True)
    manager.get_tool_names = Mock(return_value=["calculator"])
    manager.get_tool_definitions = Mock(side_effect=lambda names=None: [])
    manager.execute_tool = AsyncMock(return_value=result)
    return manager


def _make_cycle(bundled_prompt_registry, tool_manager: Mock, config: AgentsConfig):
    return CognitiveCycle(
        provider_manager=Mock(),
        memory_manager=Mock(),
        storage_manager=Mock(),
        tool_manager=tool_manager,
        prompt_registry=bundled_prompt_registry,
        agents_config=config,
    )


class TestCyclePlanGating:
    async def _run(self, goal, think_outputs, bundled_prompt_registry, config, tool_result=None):
        tool_manager = _tool_manager(
            tool_result or ToolResult(tool_call_id="c1", content="4", is_error=False)
        )
        cycle = _make_cycle(bundled_prompt_registry, tool_manager, config)
        state = EnhancedAgentState(goal=goal, session_id="s")

        plan_mock = AsyncMock(return_value=PlanOutput(plan_steps=["do it"], reasoning="r"))
        with (
            patch(
                f"{CYCLE_MODULE}.perceive_phase",
                AsyncMock(return_value=PerceiveOutput(retrieved_context=[])),
            ),
            patch(f"{CYCLE_MODULE}.plan_phase", plan_mock),
            patch(f"{CYCLE_MODULE}.think_phase", AsyncMock(side_effect=think_outputs)),
            patch(
                f"{CYCLE_MODULE}.reflect_phase",
                AsyncMock(
                    return_value=ReflectOutput(evaluation="ok", progress_estimate=0.5)
                ),
            ),
        ):
            for _ in think_outputs:
                await cycle.run_iteration(
                    agent_state=state, session_id="s", skip_validation=True
                )
        return plan_mock, state

    @pytest.mark.asyncio
    async def test_simple_goal_skips_plan_phase(self, bundled_prompt_registry):
        plan_mock, _ = await self._run(
            SIMPLE_GOAL,
            [ThinkOutput(thought="direct", is_final_answer=True, final_answer="4")],
            bundled_prompt_registry,
            _config(PlanMode.COMPLEX_ONLY),
        )
        plan_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_complex_goal_runs_plan_phase_and_stamps_baseline(
        self, bundled_prompt_registry
    ):
        plan_mock, state = await self._run(
            COMPLEX_GOAL,
            [ThinkOutput(thought="direct", is_final_answer=True, final_answer="done")],
            bundled_prompt_registry,
            _config(PlanMode.COMPLEX_ONLY),
        )
        plan_mock.assert_called_once()
        assert state.get_working_memory("initial_plan_version") == state.plan_version

    @pytest.mark.asyncio
    async def test_last_action_failed_flag_tracks_act_result(self, bundled_prompt_registry):
        failing = ToolResult(tool_call_id="c1", content="boom", is_error=True)
        _plan_mock, state = await self._run(
            SIMPLE_GOAL,
            [
                ThinkOutput(
                    thought="try",
                    proposed_action=ToolCall(
                        id="c1", name="calculator", arguments={"expression": "1/0"}
                    ),
                )
            ],
            bundled_prompt_registry,
            _config(PlanMode.COMPLEX_ONLY, escalation=False),
            tool_result=failing,
        )
        assert state.get_working_memory("last_action_failed") is True

        _plan_mock, state = await self._run(
            SIMPLE_GOAL,
            [
                ThinkOutput(
                    thought="try",
                    proposed_action=ToolCall(
                        id="c1", name="calculator", arguments={"expression": "2+2"}
                    ),
                )
            ],
            bundled_prompt_registry,
            _config(PlanMode.COMPLEX_ONLY, escalation=False),
        )
        assert state.get_working_memory("last_action_failed") is False

    @pytest.mark.asyncio
    async def test_failure_escalation_plans_on_next_iteration(self, bundled_prompt_registry):
        failing = ToolResult(tool_call_id="c1", content="boom", is_error=True)
        think_outputs = [
            ThinkOutput(
                thought="try",
                proposed_action=ToolCall(
                    id="c1", name="calculator", arguments={"expression": "1/0"}
                ),
            ),
            ThinkOutput(
                thought="retry differently",
                proposed_action=ToolCall(
                    id="c2", name="calculator", arguments={"expression": "1/1"}
                ),
            ),
        ]
        plan_mock, _ = await self._run(
            SIMPLE_GOAL,
            think_outputs,
            bundled_prompt_registry,
            _config(PlanMode.COMPLEX_ONLY, escalation=True),
            tool_result=failing,
        )
        # Iteration 1: gate closed (simple goal). Iteration 2: escalation
        # (failed observation + empty plan) triggers exactly one PLAN.
        plan_mock.assert_called_once()


# =============================================================================
# Replan cap (UPDATE phase)
# =============================================================================


def _replan_reflection(new_plan: list[str]) -> ReflectOutput:
    return ReflectOutput(
        evaluation="needs a new plan",
        progress_estimate=0.5,
        plan_needs_update=True,
        updated_plan=new_plan,
    )


class TestReplanCap:
    @pytest.mark.asyncio
    async def test_cap_allows_max_replans_then_exhausts(self):
        state = _state(MODERATE_GOAL, plan=["original"])
        state.plan_version = 1
        state.set_working_memory("initial_plan_version", 1)
        config = AgentsConfig()  # max_replans=3

        for i in range(3):
            await update_phase(
                agent_state=state,
                update_input=UpdateInput(
                    reflection=_replan_reflection([f"step {i}"]), current_state=state
                ),
                agents_config=config,
            )
            assert state.plan == [f"step {i}"]

        assert state.plan_version == 4
        assert not state.get_working_memory("replan_budget_exhausted", False)

        await update_phase(
            agent_state=state,
            update_input=UpdateInput(
                reflection=_replan_reflection(["blocked"]), current_state=state
            ),
            agents_config=config,
        )

        assert state.plan == ["step 2"]  # unchanged
        assert state.plan_version == 4
        assert state.get_working_memory("replan_budget_exhausted") is True

    @pytest.mark.asyncio
    async def test_zero_budget_blocks_first_replan(self):
        state = _state(MODERATE_GOAL, plan=["original"])
        state.plan_version = 1
        state.set_working_memory("initial_plan_version", 1)
        config = AgentsConfig()
        config.planning.max_replans = 0

        await update_phase(
            agent_state=state,
            update_input=UpdateInput(
                reflection=_replan_reflection(["blocked"]), current_state=state
            ),
            agents_config=config,
        )

        assert state.plan == ["original"]
        assert state.get_working_memory("replan_budget_exhausted") is True

    @pytest.mark.asyncio
    async def test_missing_baseline_uses_current_version(self):
        """A plan seeded outside the PLAN phase has no recorded baseline —
        the current version becomes it, so the budget still applies."""
        state = _state(MODERATE_GOAL, plan=["seeded"])
        state.plan_version = 2
        config = AgentsConfig()

        await update_phase(
            agent_state=state,
            update_input=UpdateInput(
                reflection=_replan_reflection(["updated"]), current_state=state
            ),
            agents_config=config,
        )

        assert state.plan == ["updated"]
        assert state.get_working_memory("initial_plan_version") == 2

    @pytest.mark.asyncio
    async def test_update_phase_without_agents_config_defaults(self):
        state = _state(MODERATE_GOAL, plan=["original"])
        state.plan_version = 1

        await update_phase(
            agent_state=state,
            update_input=UpdateInput(
                reflection=_replan_reflection(["updated"]), current_state=state
            ),
        )

        assert state.plan == ["updated"]


# =============================================================================
# Empty-plan rendering (the gated PLAN leaves simple goals plan-less)
# =============================================================================


class TestEmptyPlanRendering:
    def test_reflect_messages_render_with_empty_plan(self, bundled_prompt_registry):
        from llmcore.agents.cognitive.phases.reflect import _generate_reflection_messages

        messages = _generate_reflection_messages(
            reflect_input=ReflectInput(
                goal=SIMPLE_GOAL,
                plan=[],
                current_step_index=0,
                last_action=None,
                observation="tool output",
                iteration_number=1,
            ),
            prompt_registry=bundled_prompt_registry,
        )
        rendered = "\n".join(m.content for m in messages)
        assert "CURRENT PLAN:" in rendered
        assert "EVALUATION:" in rendered

    def test_think_messages_render_with_fallback_step(self, bundled_prompt_registry):
        from llmcore.agents.cognitive.phases.think import _generate_thinking_messages

        state = EnhancedAgentState(goal=SIMPLE_GOAL)
        messages = _generate_thinking_messages(
            think_input=ThinkInput(goal=SIMPLE_GOAL, current_step="Complete the goal"),
            agent_state=state,
            prompt_registry=bundled_prompt_registry,
        )
        rendered = "\n".join(m.content for m in messages)
        assert "Complete the goal" in rendered


# =============================================================================
# Config surface
# =============================================================================


class TestPlanningConfig:
    def test_defaults(self):
        config = AgentsConfig()
        assert isinstance(config.planning, PlanningConfig)
        assert config.planning.mode is PlanMode.COMPLEX_ONLY
        assert config.planning.max_replans == 3
        assert config.planning.plan_on_failure_escalation is True

    def test_plan_mode_members(self):
        assert {m.value for m in PlanMode} == {
            "always",
            "first",
            "complex_only",
            "on_failure",
        }
