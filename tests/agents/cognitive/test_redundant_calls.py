# tests/agents/cognitive/test_redundant_calls.py
"""Redundant-call detector tests (plan §2.4).

The post-THINK choke point never re-executes a non-finish action whose
name+arguments signature already ran: VALIDATE/ACT are skipped, a corrective
observation is fed into REFLECT, and once one signature has been seen
``redundancy.force_finalize_after`` times the loops route to forced
finalization (subject to the same budget guard as convergence 2.2).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llmcore.agents.cognitive.models import (
    EnhancedAgentState,
    PerceiveOutput,
    PlanOutput,
    ReflectOutput,
    TerminationReason,
    ThinkOutput,
)
from llmcore.agents.cognitive.phases.cycle import (
    REDUNDANCY_FORCE_FINALIZE_KEY,
    CognitiveCycle,
)
from llmcore.config.agents_config import AgentsConfig, RedundancyConfig
from llmcore.models import Tool, ToolCall, ToolResult

CYCLE_MODULE = "llmcore.agents.cognitive.phases.cycle"

FINISH_TOOL = Tool(
    name="finish",
    description="Complete the task with a final answer",
    parameters={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)


def _calc_call(call_id: str, expression: str = "2+2") -> ToolCall:
    return ToolCall(id=call_id, name="calculator", arguments={"expression": expression})


def _finalize_native_response(answer: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "f1",
                            "function": {"name": "finish", "arguments": {"answer": answer}},
                        }
                    ]
                }
            }
        ],
        "usage": {"total_tokens": 30},
    }


def _mock_provider(response, content: str = "") -> Mock:
    provider = Mock()
    provider.get_name = Mock(return_value="test_provider")
    provider.default_model = "test_model"
    provider.chat_completion = AsyncMock(return_value=response)
    provider.extract_response_content = Mock(return_value=content)
    return provider


def _tool_manager() -> Mock:
    manager = Mock()
    manager.is_tool_loaded = Mock(return_value=True)
    manager.get_tool_names = Mock(return_value=["calculator", "finish"])
    manager.get_tool_definitions = Mock(
        side_effect=lambda names=None: [FINISH_TOOL] if names else []
    )
    manager.execute_tool = AsyncMock(
        return_value=ToolResult(tool_call_id="c1", content="4", is_error=False)
    )
    return manager


@pytest.fixture
def agents_config() -> AgentsConfig:
    config = AgentsConfig()
    config.circuit_breaker.enabled = False
    return config


def _make_cycle(
    bundled_prompt_registry,
    agents_config: AgentsConfig | None = None,
    provider: Mock | None = None,
) -> tuple[CognitiveCycle, Mock, Mock]:
    provider_manager = Mock()
    provider_manager.get_provider = Mock(return_value=provider or Mock())
    tool_manager = _tool_manager()
    cycle = CognitiveCycle(
        provider_manager=provider_manager,
        memory_manager=Mock(),
        storage_manager=Mock(),
        tool_manager=tool_manager,
        prompt_registry=bundled_prompt_registry,
        agents_config=agents_config,
    )
    return cycle, tool_manager, provider_manager


def _phase_patches(think_side_effect):
    """Patch PERCEIVE/PLAN/THINK/REFLECT; VALIDATE/ACT/OBSERVE/UPDATE run real.

    skip_validation=True keeps the LLM judge out of the way, so a clean
    calculator call flows through the real ACT/OBSERVE path with zero LLM
    calls — the choke point's dedupe is then the only variable under test.
    """
    return (
        patch(
            f"{CYCLE_MODULE}.perceive_phase",
            AsyncMock(return_value=PerceiveOutput(retrieved_context=[])),
        ),
        patch(
            f"{CYCLE_MODULE}.plan_phase",
            AsyncMock(return_value=PlanOutput(plan_steps=["do it"], reasoning="r")),
        ),
        patch(f"{CYCLE_MODULE}.think_phase", AsyncMock(side_effect=think_side_effect)),
        patch(
            f"{CYCLE_MODULE}.reflect_phase",
            AsyncMock(return_value=ReflectOutput(evaluation="ok", progress_estimate=0.5)),
        ),
    )


async def _run_iterations(
    actions: list[ToolCall],
    bundled_prompt_registry,
    agents_config: AgentsConfig | None = None,
):
    """Run one run_iteration per action with THINK patched to propose it."""
    cycle, tool_manager, _ = _make_cycle(bundled_prompt_registry, agents_config)
    state = EnhancedAgentState(goal="Test goal", session_id="s")

    think_outputs = [
        ThinkOutput(thought=f"use the tool ({i})", proposed_action=action)
        for i, action in enumerate(actions)
    ]

    iterations = []
    p1, p2, p3, p4 = _phase_patches(think_outputs)
    with p1, p2, p3, p4:
        for _ in actions:
            iterations.append(
                await cycle.run_iteration(
                    agent_state=state, session_id="s", skip_validation=True
                )
            )
    return iterations, state, tool_manager


# =============================================================================
# Choke-point behavior in run_iteration
# =============================================================================


class TestRedundantCallDedupe:
    @pytest.mark.asyncio
    async def test_identical_action_skips_validate_and_act(self, bundled_prompt_registry):
        iterations, _state, tool_manager = await _run_iterations(
            [_calc_call("c1"), _calc_call("c2")], bundled_prompt_registry
        )

        # Only the first proposal executed.
        tool_manager.execute_tool.assert_awaited_once()

        first, second = iterations
        assert first.act_output is not None
        assert first.observe_output.observation.startswith("Executed calculator")

        # The repeat skipped VALIDATE/ACT but still ran REFLECT/UPDATE.
        assert second.validate_output is None
        assert second.act_output is None
        assert second.observe_output is not None
        assert second.observe_output.observation.startswith("REPEATED ACTION")
        assert second.reflect_output is not None
        assert second.update_output is not None

    @pytest.mark.asyncio
    async def test_corrective_observation_contents(self, bundled_prompt_registry):
        iterations, _, _ = await _run_iterations(
            [_calc_call("c1"), _calc_call("c2")], bundled_prompt_registry
        )

        observation = iterations[1].observe_output.observation
        assert (
            "you already ran calculator with identical arguments in iteration 1"
            in observation
        )
        # The recorded preview carries the first execution's result.
        assert "Result: 4" in observation
        assert observation.endswith("Use that observation or call finish.")

    @pytest.mark.asyncio
    async def test_differing_arguments_both_execute(self, bundled_prompt_registry):
        iterations, _, tool_manager = await _run_iterations(
            [_calc_call("c1", "2+2"), _calc_call("c2", "3+3")], bundled_prompt_registry
        )

        assert tool_manager.execute_tool.await_count == 2
        for iteration in iterations:
            assert iteration.act_output is not None
            assert iteration.observe_output.observation.startswith("Executed calculator")

    @pytest.mark.asyncio
    async def test_redundancy_disabled_executes_repeats(
        self, bundled_prompt_registry, agents_config
    ):
        agents_config.redundancy.enabled = False
        iterations, state, tool_manager = await _run_iterations(
            [_calc_call("c1"), _calc_call("c2")], bundled_prompt_registry, agents_config
        )

        assert tool_manager.execute_tool.await_count == 2
        assert all(i.act_output is not None for i in iterations)
        assert state.lookup_action_signature(_calc_call("cX")) is None

    @pytest.mark.asyncio
    async def test_force_finalize_flag_set_at_threshold(self, bundled_prompt_registry):
        _, state, tool_manager = await _run_iterations(
            [_calc_call("c1"), _calc_call("c2"), _calc_call("c3")],
            bundled_prompt_registry,
        )

        # 1 execution + 2 blocked repeats = count 3 (default threshold).
        tool_manager.execute_tool.assert_awaited_once()
        assert state.get_working_memory(REDUNDANCY_FORCE_FINALIZE_KEY) is True

    @pytest.mark.asyncio
    async def test_flag_not_set_below_threshold(self, bundled_prompt_registry):
        _, state, _ = await _run_iterations(
            [_calc_call("c1"), _calc_call("c2")], bundled_prompt_registry
        )
        assert not state.get_working_memory(REDUNDANCY_FORCE_FINALIZE_KEY, False)

    @pytest.mark.asyncio
    async def test_approval_paused_action_is_not_recorded(self, bundled_prompt_registry):
        """A dangerous action pauses for approval WITHOUT executing — its
        signature must not be recorded, or the post-approval retry of the
        exact same call would be blocked as a repeat."""
        dangerous = ToolCall(
            id="c1", name="calculator", arguments={"command": "sudo rm -rf /tmp/x"}
        )
        iterations, state, tool_manager = await _run_iterations(
            [dangerous], bundled_prompt_registry
        )

        tool_manager.execute_tool.assert_not_awaited()
        assert state.awaiting_human_approval is True
        assert state.lookup_action_signature(dangerous) is None
        assert iterations[0].observe_output.observation.startswith("Executed")


# =============================================================================
# Signature canonicalization + snapshot round-trip (models-level)
# =============================================================================


class TestActionSignatures:
    def test_dict_key_order_canonicalization(self):
        state = EnhancedAgentState(goal="g")
        first = ToolCall(id="1", name="t", arguments={"a": 1, "b": {"x": 1, "y": 2}})
        second = ToolCall(id="2", name="t", arguments={"b": {"y": 2, "x": 1}, "a": 1})

        assert EnhancedAgentState.compute_action_signature(
            first
        ) == EnhancedAgentState.compute_action_signature(second)

        state.record_action_signature(first, result_preview="r")
        entry = state.lookup_action_signature(second)
        assert entry is not None
        assert entry["count"] == 1
        assert entry["result_preview"] == "r"

    def test_signature_ignores_call_id_but_not_name(self):
        same_args = {"q": "rivers"}
        a = ToolCall(id="1", name="search", arguments=dict(same_args))
        b = ToolCall(id="2", name="search", arguments=dict(same_args))
        c = ToolCall(id="1", name="other_search", arguments=dict(same_args))

        assert EnhancedAgentState.compute_action_signature(
            a
        ) == EnhancedAgentState.compute_action_signature(b)
        assert EnhancedAgentState.compute_action_signature(
            a
        ) != EnhancedAgentState.compute_action_signature(c)

    def test_record_counts_and_preview_retention(self):
        state = EnhancedAgentState(goal="g")
        call = ToolCall(id="1", name="search", arguments={"q": "rivers"})

        assert state.record_action_signature(call, result_preview="10 results") == 1
        # A later record without a preview keeps the earlier one.
        assert state.record_action_signature(call) == 2
        entry = state.lookup_action_signature(call)
        assert entry == {"count": 2, "iteration": 1, "result_preview": "10 results"}

    def test_snapshot_round_trip_preserves_signatures(self):
        state = EnhancedAgentState(goal="g", session_id="s")
        call = ToolCall(id="1", name="search", arguments={"q": "rivers", "n": 5})
        state.record_action_signature(call, result_preview="10 results")
        state.record_action_signature(call)

        restored = EnhancedAgentState.from_resume_snapshot(state.to_resume_snapshot())

        entry = restored.lookup_action_signature(call)
        assert entry is not None
        assert entry["count"] == 2
        assert entry["result_preview"] == "10 results"


# =============================================================================
# Loop routing to forced finalize
# =============================================================================


class TestRedundancyForcedFinalize:
    def _repeating_think(self):
        async def think(*args, **kwargs):
            return ThinkOutput(thought="same again", proposed_action=_calc_call("cX"))

        return think

    @pytest.mark.asyncio
    async def test_run_until_complete_finalizes_on_redundancy(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("Converged answer"))
        cycle, tool_manager, _provider_manager = _make_cycle(
            bundled_prompt_registry, agents_config, provider
        )
        state = EnhancedAgentState(goal="Loop goal", session_id="s")

        p1, p2, p3, p4 = _phase_patches(self._repeating_think())
        with p1, p2, p3, p4:
            result = await cycle.run_until_complete(
                agent_state=state,
                session_id="s",
                max_iterations=10,
                skip_validation=True,
                agents_config=agents_config,
            )

        assert result == "Converged answer"
        assert state.final_answer == "Converged answer"
        assert state.termination_reason == TerminationReason.FORCED_FINALIZE.value
        # 1 execution + 2 blocked repeats, then the loop finalized — well
        # before the 2.2 budget-edge trigger at max_iterations=10.
        assert len(state.iterations) == 3
        tool_manager.execute_tool.assert_awaited_once()
        provider.chat_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_streaming_finalizes_on_redundancy(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("Streamed answer"))
        cycle, _, _ = _make_cycle(bundled_prompt_registry, agents_config, provider)
        state = EnhancedAgentState(goal="Loop goal", session_id="s")

        updates = []
        p1, p2, p3, p4 = _phase_patches(self._repeating_think())
        with p1, p2, p3, p4:
            async for update in cycle.run_streaming(
                agent_state=state,
                session_id="s",
                max_iterations=10,
                skip_validation=True,
                agents_config=agents_config,
            ):
                updates.append(update)

        final = updates[-1]
        assert final.is_complete is True
        assert final.is_final is True
        assert final.status == "complete"
        assert final.termination_reason == TerminationReason.FORCED_FINALIZE.value
        assert state.final_answer == "Streamed answer"
        assert len(state.iterations) == 3

    @pytest.mark.asyncio
    async def test_budget_one_keeps_legacy_semantics(
        self, bundled_prompt_registry, agents_config
    ):
        """max_iterations=1 (wairu's bounded outer-loop driving pattern) must
        ignore the redundancy flag — the outer driver owns convergence."""
        provider = _mock_provider(_finalize_native_response("never used"))
        cycle, _, _ = _make_cycle(bundled_prompt_registry, agents_config, provider)
        state = EnhancedAgentState(goal="Loop goal", session_id="s")
        state.set_working_memory(REDUNDANCY_FORCE_FINALIZE_KEY, True)

        p1, p2, p3, p4 = _phase_patches(self._repeating_think())
        with p1, p2, p3, p4:
            result = await cycle.run_until_complete(
                agent_state=state,
                session_id="s",
                max_iterations=1,
                skip_validation=True,
                agents_config=agents_config,
            )

        assert "Task incomplete" in result
        assert state.is_finished is False
        provider.chat_completion.assert_not_awaited()


# =============================================================================
# Config surface
# =============================================================================


class TestRedundancyConfig:
    def test_defaults(self):
        config = AgentsConfig()
        assert isinstance(config.redundancy, RedundancyConfig)
        assert config.redundancy.enabled is True
        assert config.redundancy.force_finalize_after == 3
