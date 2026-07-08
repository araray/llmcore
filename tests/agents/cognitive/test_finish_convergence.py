# tests/agents/cognitive/test_finish_convergence.py
"""Finish-tool convergence tests (plan §2.1).

A native ``finish``-tool call is a terminal signal, not a tool execution:
THINK intercepts it before ``proposed_action`` assignment, plan-step finish
shortcuts route to the same final-answer construction, and ACT keeps a
defense-in-depth short-circuit for stale/resumed calls. Every stop site
stamps ``EnhancedAgentState.termination_reason``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from llmcore.agents.cognitive.models import (
    ActInput,
    ConfidenceLevel,
    EnhancedAgentState,
    PlanStepSpec,
    TerminationReason,
    ThinkInput,
    ThinkOutput,
)
from llmcore.agents.cognitive.phases.act import act_phase
from llmcore.agents.cognitive.phases.cycle import StreamingIterationResult
from llmcore.agents.cognitive.phases.think import (
    FINISH_WITHOUT_ANSWER_THOUGHT,
    _parse_think_response,
    think_phase,
)
from llmcore.config.agents_config import AgentsConfig, ConvergenceConfig
from llmcore.models import ToolCall


def _native_response(name: str, arguments) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": name, "arguments": arguments}}
                    ]
                }
            }
        ],
        "usage": {"total_tokens": 10},
    }


# =============================================================================
# _parse_think_response finish-tool matrix
# =============================================================================


class TestParseFinishToolMatrix:
    def test_native_finish_call_terminates(self):
        output = _parse_think_response(
            response_text="Thought: done.",
            response_dict=_native_response("finish", {"answer": "The answer is 42."}),
            tool_manager=object(),
        )

        assert output.is_final_answer is True
        assert output.final_answer == "The answer is 42."
        assert output.final_answer_source == "finish_tool"
        assert output.proposed_action is None
        assert output.confidence == ConfidenceLevel.HIGH

    def test_final_answer_alias_terminates(self):
        output = _parse_think_response(
            response_text="",
            response_dict=_native_response("final_answer", {"answer": "42"}),
            tool_manager=object(),
        )

        assert output.is_final_answer is True
        assert output.final_answer == "42"
        assert output.final_answer_source == "finish_tool"

    def test_json_string_arguments(self):
        output = _parse_think_response(
            response_text="",
            response_dict=_native_response("finish", '{"answer": "forty-two"}'),
            tool_manager=object(),
        )

        assert output.is_final_answer is True
        assert output.final_answer == "forty-two"

    def test_bare_string_arguments_fall_back_to_input_key(self):
        # _coerce_tool_arguments wraps non-JSON strings as {"input": ...}
        output = _parse_think_response(
            response_text="",
            response_dict=_native_response("finish", "the plain answer"),
            tool_manager=object(),
        )

        assert output.is_final_answer is True
        assert output.final_answer == "the plain answer"
        assert output.final_answer_source == "finish_tool"

    def test_empty_answer_is_corrective_not_final(self):
        output = _parse_think_response(
            response_text="Thought: finishing.",
            response_dict=_native_response("finish", {}),
            tool_manager=object(),
        )

        assert output.is_final_answer is False
        assert output.final_answer is None
        assert output.proposed_action is None
        assert output.thought == FINISH_WITHOUT_ANSWER_THOUGHT
        assert output.confidence == ConfidenceLevel.LOW

    def test_whitespace_answer_is_corrective(self):
        output = _parse_think_response(
            response_text="",
            response_dict=_native_response("finish", {"answer": "   "}),
            tool_manager=object(),
        )

        assert output.is_final_answer is False
        assert output.thought == FINISH_WITHOUT_ANSWER_THOUGHT

    def test_min_answer_chars_enforced(self):
        convergence = ConvergenceConfig(min_answer_chars=10)
        output = _parse_think_response(
            response_text="",
            response_dict=_native_response("finish", {"answer": "short"}),
            tool_manager=object(),
            convergence=convergence,
        )

        assert output.is_final_answer is False
        assert output.thought == FINISH_WITHOUT_ANSWER_THOUGHT

    def test_require_nonempty_answer_disabled_accepts_empty(self):
        convergence = ConvergenceConfig(require_nonempty_answer=False)
        output = _parse_think_response(
            response_text="",
            response_dict=_native_response("finish", {}),
            tool_manager=object(),
            convergence=convergence,
        )

        assert output.is_final_answer is True
        assert output.final_answer == ""
        assert output.final_answer_source == "finish_tool"

    def test_custom_finish_tool_names(self):
        convergence = ConvergenceConfig(finish_tool_names=["done"])
        done = _parse_think_response(
            response_text="",
            response_dict=_native_response("done", {"answer": "yes"}),
            tool_manager=object(),
            convergence=convergence,
        )
        assert done.is_final_answer is True

        # "finish" is no longer terminal under the custom name list.
        finish = _parse_think_response(
            response_text="",
            response_dict=_native_response("finish", {"answer": "yes"}),
            tool_manager=object(),
            convergence=convergence,
        )
        assert finish.is_final_answer is False
        assert finish.proposed_action is not None
        assert finish.proposed_action.name == "finish"

    def test_non_finish_native_call_untouched(self):
        output = _parse_think_response(
            response_text="Thought: search it.",
            response_dict=_native_response("semantic_search", {"query": "x"}),
            tool_manager=object(),
        )

        assert output.is_final_answer is False
        assert output.final_answer_source is None
        assert output.proposed_action is not None
        assert output.proposed_action.name == "semantic_search"

    def test_text_final_answer_regression(self):
        """The `Final Answer:` ReAct text parse stays the non-tool fallback."""
        output = _parse_think_response(
            response_text="Thought: all done.\nFinal Answer: it is 42",
            response_dict={"usage": {"total_tokens": 5}},
            tool_manager=object(),
        )

        assert output.is_final_answer is True
        assert output.final_answer == "it is 42"
        assert output.final_answer_source == "text"
        assert output.reasoning_tokens == 5


# =============================================================================
# think_phase integration (native finish + plan-step shortcut)
# =============================================================================


def _mock_provider(response: dict, content: str = ""):
    provider = Mock()
    provider.get_name = Mock(return_value="test_provider")
    provider.default_model = "test_model"
    provider.chat_completion = AsyncMock(return_value=response)
    provider.extract_response_content = Mock(return_value=content)
    return provider


class TestThinkPhaseFinish:
    @pytest.mark.asyncio
    async def test_native_finish_sets_state_and_reason(self, bundled_prompt_registry):
        provider = _mock_provider(
            _native_response("finish", {"answer": "Paris"}), "Thought: I know this."
        )
        provider_manager = Mock()
        provider_manager.get_provider = Mock(return_value=provider)
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(return_value=[])

        state = EnhancedAgentState(goal="Capital of France?")
        output = await think_phase(
            agent_state=state,
            think_input=ThinkInput(goal="Capital of France?", current_step="Answer"),
            provider_manager=provider_manager,
            memory_manager=Mock(),
            tool_manager=tool_manager,
            prompt_registry=bundled_prompt_registry,
        )

        assert output.is_final_answer is True
        assert output.final_answer == "Paris"
        assert output.final_answer_source == "finish_tool"
        assert state.is_finished is True
        assert state.final_answer == "Paris"
        assert state.termination_reason == TerminationReason.FINISH_TOOL.value

    @pytest.mark.asyncio
    async def test_text_final_answer_stamps_text_reason(self, bundled_prompt_registry):
        content = "Thought: easy.\nFinal Answer: Paris"
        provider = _mock_provider({"choices": [{"message": {"content": content}}]}, content)
        provider_manager = Mock()
        provider_manager.get_provider = Mock(return_value=provider)
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(return_value=[])

        state = EnhancedAgentState(goal="Capital of France?")
        output = await think_phase(
            agent_state=state,
            think_input=ThinkInput(goal="Capital of France?", current_step="Answer"),
            provider_manager=provider_manager,
            memory_manager=Mock(),
            tool_manager=tool_manager,
            prompt_registry=bundled_prompt_registry,
        )

        assert output.is_final_answer is True
        assert output.final_answer_source == "text"
        assert state.termination_reason == TerminationReason.FINAL_ANSWER_TEXT.value

    @pytest.mark.asyncio
    async def test_empty_finish_call_reprompts_not_finished(self, bundled_prompt_registry):
        provider = _mock_provider(_native_response("finish", {}), "Thought: finishing.")
        provider_manager = Mock()
        provider_manager.get_provider = Mock(return_value=provider)
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(return_value=[])

        state = EnhancedAgentState(goal="G")
        output = await think_phase(
            agent_state=state,
            think_input=ThinkInput(goal="G", current_step="S"),
            provider_manager=provider_manager,
            memory_manager=Mock(),
            tool_manager=tool_manager,
            prompt_registry=bundled_prompt_registry,
        )

        assert output.is_final_answer is False
        assert output.proposed_action is None
        assert output.thought == FINISH_WITHOUT_ANSWER_THOUGHT
        assert state.is_finished is False
        assert state.termination_reason is None

    @pytest.mark.asyncio
    async def test_plan_step_finish_routes_to_final_answer(self, bundled_prompt_registry):
        provider_manager = Mock()  # must never be called
        state = EnhancedAgentState(goal="G")
        think_input = ThinkInput(
            goal="G",
            current_step="Finish up",
            current_step_spec=PlanStepSpec(
                index=0,
                description="Finish up",
                tool_name="finish",
                input={"answer": "planned answer"},
            ),
        )

        output = await think_phase(
            agent_state=state,
            think_input=think_input,
            provider_manager=provider_manager,
            memory_manager=Mock(),
            tool_manager=Mock(),
            prompt_registry=bundled_prompt_registry,
        )

        assert output.is_final_answer is True
        assert output.final_answer == "planned answer"
        assert output.final_answer_source == "finish_tool"
        assert output.proposed_action is None
        assert state.is_finished is True
        assert state.termination_reason == TerminationReason.FINISH_TOOL.value
        provider_manager.get_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_plan_step_finish_without_answer_falls_through(
        self, bundled_prompt_registry
    ):
        """An empty plan-step finish falls through to a normal THINK call."""
        provider = _mock_provider(
            _native_response("finish", {"answer": "recovered"}), "Thought: recovering."
        )
        provider_manager = Mock()
        provider_manager.get_provider = Mock(return_value=provider)
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(return_value=[])

        state = EnhancedAgentState(goal="G")
        think_input = ThinkInput(
            goal="G",
            current_step="Finish up",
            current_step_spec=PlanStepSpec(
                index=0, description="Finish up", tool_name="finish", input={}
            ),
        )

        output = await think_phase(
            agent_state=state,
            think_input=think_input,
            provider_manager=provider_manager,
            memory_manager=Mock(),
            tool_manager=tool_manager,
            prompt_registry=bundled_prompt_registry,
        )

        provider.chat_completion.assert_awaited()
        assert output.is_final_answer is True
        assert output.final_answer == "recovered"

    @pytest.mark.asyncio
    async def test_plan_step_non_finish_shortcut_unchanged(self, bundled_prompt_registry):
        provider_manager = Mock()
        state = EnhancedAgentState(goal="G")
        think_input = ThinkInput(
            goal="G",
            current_step="Search",
            current_step_spec=PlanStepSpec(
                index=0,
                description="Search",
                tool_name="semantic_search",
                input={"query": "x"},
            ),
        )

        output = await think_phase(
            agent_state=state,
            think_input=think_input,
            provider_manager=provider_manager,
            memory_manager=Mock(),
            tool_manager=Mock(),
            prompt_registry=bundled_prompt_registry,
        )

        assert output.is_final_answer is False
        assert output.proposed_action is not None
        assert output.proposed_action.name == "semantic_search"
        assert state.is_finished is False


# =============================================================================
# ACT defense-in-depth
# =============================================================================


class TestActFinishDefense:
    @pytest.mark.asyncio
    async def test_finish_call_short_circuits_act(self):
        tool_manager = Mock()
        tool_manager.execute_tool = AsyncMock()

        state = EnhancedAgentState(goal="G")
        act_input = ActInput(
            tool_call=ToolCall(id="c1", name="finish", arguments={"answer": "done deal"})
        )

        output = await act_phase(
            agent_state=state, act_input=act_input, tool_manager=tool_manager
        )

        assert output.success is True
        assert output.tool_result.content == "done deal"
        assert state.is_finished is True
        assert state.final_answer == "done deal"
        assert state.termination_reason == TerminationReason.FINISH_TOOL.value
        tool_manager.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_finish_alias_short_circuits_act(self):
        tool_manager = Mock()
        tool_manager.execute_tool = AsyncMock()

        state = EnhancedAgentState(goal="G")
        act_input = ActInput(
            tool_call=ToolCall(id="c1", name="final_answer", arguments={"answer": "ok"})
        )

        output = await act_phase(
            agent_state=state, act_input=act_input, tool_manager=tool_manager
        )

        assert output.success is True
        assert state.is_finished is True
        tool_manager.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_finish_in_act_does_not_finish(self):
        tool_manager = Mock()
        tool_manager.execute_tool = AsyncMock()

        state = EnhancedAgentState(goal="G")
        act_input = ActInput(tool_call=ToolCall(id="c1", name="finish", arguments={}))

        output = await act_phase(
            agent_state=state, act_input=act_input, tool_manager=tool_manager
        )

        assert output.success is True
        assert state.is_finished is False
        assert state.termination_reason is None
        tool_manager.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normal_tool_still_executes(self):
        from llmcore.models import ToolResult

        tool_manager = Mock()
        tool_manager.execute_tool = AsyncMock(
            return_value=ToolResult(tool_call_id="c1", content="4", is_error=False)
        )

        state = EnhancedAgentState(goal="G")
        act_input = ActInput(
            tool_call=ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
        )

        output = await act_phase(
            agent_state=state, act_input=act_input, tool_manager=tool_manager
        )

        assert output.success is True
        tool_manager.execute_tool.assert_awaited_once()
        assert state.is_finished is False


# =============================================================================
# termination_reason plumbing
# =============================================================================


class TestTerminationReasonPlumbing:
    def test_enum_members(self):
        assert {reason.value for reason in TerminationReason} == {
            "finish_tool",
            "final_answer_text",
            "forced_finalize",
            "synthesis_fallback",
            "plan_complete",
            "update_stopped",
            "human_approval_required",
            "circuit_breaker",
            "max_iterations",
            "error",
        }

    def test_state_snapshot_round_trip(self):
        state = EnhancedAgentState(goal="G", session_id="s1")
        state.is_finished = True
        state.final_answer = "42"
        state.termination_reason = TerminationReason.FINISH_TOOL.value

        snapshot = state.to_resume_snapshot()
        assert snapshot["termination_reason"] == "finish_tool"

        restored = EnhancedAgentState.from_resume_snapshot(snapshot)
        assert restored.termination_reason == "finish_tool"
        assert restored.is_finished is True

    def test_state_snapshot_round_trip_none(self):
        state = EnhancedAgentState(goal="G")
        snapshot = state.to_resume_snapshot()
        assert snapshot["termination_reason"] is None
        restored = EnhancedAgentState.from_resume_snapshot(snapshot)
        assert restored.termination_reason is None

    def test_streaming_result_field_defaults(self):
        result = StreamingIterationResult(iteration=1, max_iterations=5, progress=0.1)
        assert result.termination_reason is None

    def test_iteration_update_carries_termination_reason(self):
        from llmcore.agents.single_agent import IterationUpdate

        streaming = StreamingIterationResult(
            iteration=2,
            max_iterations=5,
            progress=1.0,
            is_complete=True,
            is_final=True,
            status="complete",
            stop_reason="finish_tool",
            termination_reason="finish_tool",
        )
        update = IterationUpdate.from_streaming_result(streaming)
        assert update.termination_reason == "finish_tool"
        assert update.stop_reason == "finish_tool"

    def test_agent_result_carries_termination_reason(self):
        from llmcore.agents.single_agent import AgentResult

        state = EnhancedAgentState(goal="G", session_id="s1")
        state.termination_reason = TerminationReason.FINISH_TOOL.value
        result = AgentResult(
            goal="G",
            final_answer="42",
            success=True,
            iteration_count=1,
            total_tokens=10,
            total_time_seconds=0.1,
            session_id="s1",
            agent_state=state,
        )
        assert result.termination_reason == "finish_tool"
        assert result.to_dict()["termination_reason"] == "finish_tool"

    @pytest.mark.asyncio
    async def test_update_phase_stamps_plan_complete(self):
        from llmcore.agents.cognitive.models import ReflectOutput, UpdateInput
        from llmcore.agents.cognitive.phases.update import update_phase

        state = EnhancedAgentState(goal="G")
        reflection = ReflectOutput(evaluation="done", progress_estimate=1.0)

        output = await update_phase(
            agent_state=state,
            update_input=UpdateInput(reflection=reflection, current_state=state),
        )

        assert output.should_continue is False
        assert state.is_finished is True
        assert state.termination_reason == TerminationReason.PLAN_COMPLETE.value

    def test_think_output_final_answer_source_default(self):
        output = ThinkOutput(thought="t")
        assert output.final_answer_source is None

    def test_agents_config_convergence_defaults(self):
        config = AgentsConfig()
        assert config.convergence.finish_tool_names == ["finish", "final_answer"]
        assert config.convergence.require_nonempty_answer is True
        assert config.convergence.min_answer_chars == 1
        assert config.convergence.forced_finalize_enabled is True
        assert config.convergence.finalize_when_remaining == 1
        assert config.convergence.synthesis_on_exhaustion is True
        assert config.convergence.finalize_temperature == 0.2
