# tests/agents/cognitive/test_forced_finalize.py
"""remaining_steps + in-cycle forced finalize tests (plan §2.2).

The invariant under test: the cognitive loop cannot exit un-converged on the
max-iterations / update-stopped paths. The last budgeted iteration becomes a
finalize synthesis pass (``forced_finalize``) and loop exhaustion triggers a
synthesis fallback (``synthesis_fallback``). Only human-approval,
circuit-breaker, and hard-error exits remain un-converged.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from llmcore.agents.cognitive.models import (
    CycleIteration,
    EnhancedAgentState,
    TerminationReason,
    ThinkInput,
)
from llmcore.agents.cognitive.phases.cycle import CognitiveCycle
from llmcore.config.agents_config import AgentsConfig
from llmcore.models import Tool

FINISH_TOOL = Tool(
    name="finish",
    description="Complete the task with a final answer",
    parameters={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)


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


def _mock_provider(response, content: str = ""):
    provider = Mock()
    provider.get_name = Mock(return_value="test_provider")
    provider.default_model = "test_model"
    provider.chat_completion = AsyncMock(return_value=response)
    provider.extract_response_content = Mock(return_value=content)
    return provider


def _non_finishing_iteration(number: int) -> CycleIteration:
    iteration = CycleIteration(iteration_number=number)
    iteration.think_output = MagicMock()
    iteration.think_output.is_final_answer = False
    iteration.think_output.proposed_action = None
    iteration.observe_output = None
    iteration.reflect_output = MagicMock()
    iteration.reflect_output.step_completed = False
    iteration.update_output = MagicMock()
    iteration.update_output.should_continue = True
    iteration.total_tokens_used = 0
    iteration.mark_completed(success=True)
    return iteration


@pytest.fixture
def agents_config():
    config = AgentsConfig()
    config.circuit_breaker.enabled = False
    return config


def _make_cycle(provider, bundled_prompt_registry):
    provider_manager = Mock()
    provider_manager.get_provider = Mock(return_value=provider)

    tool_manager = Mock()
    tool_manager.get_tool_definitions = Mock(
        side_effect=lambda names=None: [FINISH_TOOL] if names else []
    )

    return CognitiveCycle(
        provider_manager=provider_manager,
        memory_manager=Mock(),
        storage_manager=Mock(),
        tool_manager=tool_manager,
        prompt_registry=bundled_prompt_registry,
    )


def _patch_never_finishing(cycle: CognitiveCycle) -> Mock:
    calls: list[dict] = []

    async def fake_run_iteration(*args, **kwargs):
        calls.append(kwargs)
        return _non_finishing_iteration(len(calls))

    mock = Mock(side_effect=fake_run_iteration)
    mock.calls = calls
    cycle.run_iteration = mock  # type: ignore[method-assign]
    return mock


# =============================================================================
# run_until_complete
# =============================================================================


class TestRunUntilCompleteForcedFinalize:
    @pytest.mark.asyncio
    async def test_never_finishing_run_gets_synthesized_answer(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("Synthesized answer"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        run_iteration = _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="Long goal", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=3,
            agents_config=agents_config,
        )

        assert result == "Synthesized answer"
        assert state.is_finished is True
        assert state.final_answer == "Synthesized answer"
        assert state.termination_reason == TerminationReason.FORCED_FINALIZE.value
        # The last budgeted iteration was replaced by the finalize pass.
        assert run_iteration.call_count == 2
        provider.chat_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remaining_iterations_threaded_to_run_iteration(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("A"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        run_iteration = _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=3,
            agents_config=agents_config,
        )

        assert [kwargs.get("remaining_iterations") for kwargs in run_iteration.calls] == [3, 2]

    @pytest.mark.asyncio
    async def test_finalize_prompt_carries_reason_and_goal(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("A"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="Count the rivers", session_id="s")
        await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=2,
            agents_config=agents_config,
        )

        messages = provider.chat_completion.call_args.kwargs["context"]
        rendered = "\n".join(m.content for m in messages)
        assert "forced_finalize" in rendered
        assert "Count the rivers" in rendered

    @pytest.mark.asyncio
    async def test_tool_choice_required_captured(self, bundled_prompt_registry, agents_config):
        provider = _mock_provider(_finalize_native_response("A"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=2,
            agents_config=agents_config,
        )

        call_kwargs = provider.chat_completion.call_args.kwargs
        assert call_kwargs["tools"] == [FINISH_TOOL]
        assert call_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_tool_choice_omitted_without_signature_support(
        self, bundled_prompt_registry, agents_config
    ):
        captured: dict = {}

        class NoToolChoiceProvider:
            default_model = "m"

            def get_name(self):
                return "no-tool-choice"

            async def chat_completion(
                self, context, model=None, stream=False, tools=None, temperature=None
            ):
                captured["tools"] = tools
                captured["context"] = context
                return {"choices": [{"message": {"content": "plain text answer"}}]}

            def extract_response_content(self, response):
                return "plain text answer"

        cycle = _make_cycle(NoToolChoiceProvider(), bundled_prompt_registry)
        _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=2,
            agents_config=agents_config,
        )

        # Text fallback parse: the full response text is the answer.
        assert result == "plain text answer"
        assert state.termination_reason == TerminationReason.FORCED_FINALIZE.value
        assert captured["tools"] == [FINISH_TOOL]

    @pytest.mark.asyncio
    async def test_provider_error_falls_through_to_legacy_exit(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(None)
        provider.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=3,
            agents_config=agents_config,
        )

        # NEVER raises in the exhaustion position: legacy incomplete message,
        # un-finished state, termination_reason=error.
        assert "Task incomplete" in result
        assert state.is_finished is False
        assert state.termination_reason == TerminationReason.ERROR.value

    @pytest.mark.asyncio
    async def test_synthesis_fallback_on_update_stop(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("Recovered answer"))
        cycle = _make_cycle(provider, bundled_prompt_registry)

        async def stopping_iteration(*args, **kwargs):
            iteration = _non_finishing_iteration(1)
            iteration.update_output.should_continue = False
            return iteration

        cycle.run_iteration = Mock(side_effect=stopping_iteration)  # type: ignore[method-assign]

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=5,
            agents_config=agents_config,
        )

        assert result == "Recovered answer"
        assert state.termination_reason == TerminationReason.SYNTHESIS_FALLBACK.value

    @pytest.mark.asyncio
    async def test_human_approval_stop_skips_synthesis(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("never used"))
        cycle = _make_cycle(provider, bundled_prompt_registry)

        async def approval_iteration(*args, **kwargs):
            state = kwargs.get("agent_state") or args[0]
            state.awaiting_human_approval = True
            state.pending_approval_prompt = "Approve?"
            iteration = _non_finishing_iteration(1)
            iteration.update_output.should_continue = False
            return iteration

        cycle.run_iteration = Mock(side_effect=approval_iteration)  # type: ignore[method-assign]

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=5,
            agents_config=agents_config,
        )

        assert "Human approval required" in result
        assert state.termination_reason == TerminationReason.HUMAN_APPROVAL_REQUIRED.value
        provider.chat_completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forced_finalize_disabled_keeps_legacy_behavior(
        self, bundled_prompt_registry, agents_config
    ):
        agents_config.convergence.forced_finalize_enabled = False
        agents_config.convergence.synthesis_on_exhaustion = False

        provider = _mock_provider(_finalize_native_response("never used"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        run_iteration = _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=3,
            agents_config=agents_config,
        )

        assert "Task incomplete" in result
        assert run_iteration.call_count == 3
        assert state.termination_reason == TerminationReason.MAX_ITERATIONS.value
        provider.chat_completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_iteration_budget_keeps_legacy_semantics(
        self, bundled_prompt_registry, agents_config
    ):
        """A bounded run(max_iterations=1) — wairu's outer-loop driving
        pattern — must run its one normal iteration and exit un-converged
        without any finalize/synthesis provider call."""
        provider = _mock_provider(_finalize_native_response("never used"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        run_iteration = _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=1,
            agents_config=agents_config,
        )

        assert "Task incomplete" in result
        assert run_iteration.call_count == 1
        assert state.is_finished is False
        assert state.termination_reason == TerminationReason.MAX_ITERATIONS.value
        provider.chat_completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exhaustion_synthesis_when_in_loop_trigger_disabled(
        self, bundled_prompt_registry, agents_config
    ):
        agents_config.convergence.finalize_when_remaining = 0

        provider = _mock_provider(_finalize_native_response("Exhaustion answer"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        run_iteration = _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        result = await cycle.run_until_complete(
            agent_state=state,
            session_id="s",
            max_iterations=3,
            agents_config=agents_config,
        )

        assert result == "Exhaustion answer"
        assert run_iteration.call_count == 3  # full budget, then synthesis
        assert state.termination_reason == TerminationReason.SYNTHESIS_FALLBACK.value


# =============================================================================
# run_streaming
# =============================================================================


class TestRunStreamingForcedFinalize:
    async def _collect(self, cycle, state, agents_config, max_iterations=3):
        updates = []
        async for update in cycle.run_streaming(
            agent_state=state,
            session_id="s",
            max_iterations=max_iterations,
            agents_config=agents_config,
        ):
            updates.append(update)
        return updates

    @pytest.mark.asyncio
    async def test_forced_finalize_terminal_update(self, bundled_prompt_registry, agents_config):
        provider = _mock_provider(_finalize_native_response("Streamed answer"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        updates = await self._collect(cycle, state, agents_config, max_iterations=3)

        final = updates[-1]
        assert final.is_complete is True
        assert final.is_final is True
        assert final.status == "complete"
        assert final.stop_reason == TerminationReason.FORCED_FINALIZE.value
        assert final.termination_reason == TerminationReason.FORCED_FINALIZE.value
        assert final.message == "Streamed answer"
        assert state.final_answer == "Streamed answer"

    @pytest.mark.asyncio
    async def test_synthesis_fallback_terminal_update_on_exhaustion(
        self, bundled_prompt_registry, agents_config
    ):
        agents_config.convergence.finalize_when_remaining = 0

        provider = _mock_provider(_finalize_native_response("Fallback answer"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        run_iteration = _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        updates = await self._collect(cycle, state, agents_config, max_iterations=3)

        assert run_iteration.call_count == 3
        final = updates[-1]
        assert final.is_complete is True
        assert final.status == "complete"
        assert final.stop_reason == TerminationReason.SYNTHESIS_FALLBACK.value
        assert final.termination_reason == TerminationReason.SYNTHESIS_FALLBACK.value

    @pytest.mark.asyncio
    async def test_update_stop_synthesis_terminal_update(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("Recovered"))
        cycle = _make_cycle(provider, bundled_prompt_registry)

        async def stopping_iteration(*args, **kwargs):
            iteration = _non_finishing_iteration(1)
            iteration.update_output.should_continue = False
            return iteration

        cycle.run_iteration = Mock(side_effect=stopping_iteration)  # type: ignore[method-assign]

        state = EnhancedAgentState(goal="G", session_id="s")
        updates = await self._collect(cycle, state, agents_config, max_iterations=5)

        final = updates[-1]
        assert final.is_complete is True
        assert final.status == "complete"
        assert final.termination_reason == TerminationReason.SYNTHESIS_FALLBACK.value

    @pytest.mark.asyncio
    async def test_provider_error_keeps_max_iterations_yield(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(None)
        provider.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))
        cycle = _make_cycle(provider, bundled_prompt_registry)
        _patch_never_finishing(cycle)

        state = EnhancedAgentState(goal="G", session_id="s")
        updates = await self._collect(cycle, state, agents_config, max_iterations=3)

        final = updates[-1]
        assert final.is_complete is False
        assert final.is_final is True
        assert final.status == "max_iterations"
        assert final.stop_reason == "max_iterations"
        assert final.termination_reason == TerminationReason.ERROR.value

    @pytest.mark.asyncio
    async def test_human_approval_stop_keeps_stopped_update(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("never used"))
        cycle = _make_cycle(provider, bundled_prompt_registry)

        async def approval_iteration(*args, **kwargs):
            state = kwargs.get("agent_state") or args[0]
            state.awaiting_human_approval = True
            iteration = _non_finishing_iteration(1)
            iteration.update_output.should_continue = False
            return iteration

        cycle.run_iteration = Mock(side_effect=approval_iteration)  # type: ignore[method-assign]

        state = EnhancedAgentState(goal="G", session_id="s")
        updates = await self._collect(cycle, state, agents_config, max_iterations=5)

        final = updates[-1]
        assert final.status == "stopped"
        assert final.stop_reason == "human_approval_required"
        assert final.termination_reason == TerminationReason.HUMAN_APPROVAL_REQUIRED.value
        provider.chat_completion.assert_not_awaited()


# =============================================================================
# _force_finalize unit behavior
# =============================================================================


class TestForceFinalizeUnit:
    @pytest.mark.asyncio
    async def test_empty_synthesis_answer_fails_closed(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider({"choices": [{"message": {"content": ""}}]}, content="")
        cycle = _make_cycle(provider, bundled_prompt_registry)

        state = EnhancedAgentState(goal="G", session_id="s")
        finalized = await cycle._force_finalize(
            state, reason=TerminationReason.SYNTHESIS_FALLBACK
        )

        assert finalized is False
        assert state.is_finished is False
        assert state.termination_reason == TerminationReason.ERROR.value

    @pytest.mark.asyncio
    async def test_direct_call_sets_reason_and_answer(
        self, bundled_prompt_registry, agents_config
    ):
        provider = _mock_provider(_finalize_native_response("Direct"))
        cycle = _make_cycle(provider, bundled_prompt_registry)

        state = EnhancedAgentState(goal="G", session_id="s")
        finalized = await cycle._force_finalize(
            state, reason=TerminationReason.FORCED_FINALIZE
        )

        assert finalized is True
        assert state.is_finished is True
        assert state.final_answer == "Direct"
        assert state.termination_reason == TerminationReason.FORCED_FINALIZE.value


# =============================================================================
# remaining_steps plumbing
# =============================================================================


class TestRemainingStepsPlumbing:
    def test_think_input_default_is_none(self):
        assert ThinkInput(goal="G", current_step="S").remaining_steps is None

    def test_run_iteration_signature_backcompat(self):
        parameters = inspect.signature(CognitiveCycle.run_iteration).parameters
        assert "remaining_iterations" in parameters
        assert parameters["remaining_iterations"].default is None

    def test_thinking_messages_render_remaining_steps(self, bundled_prompt_registry):
        from llmcore.agents.cognitive.phases.think import _generate_thinking_messages

        state = EnhancedAgentState(goal="G")
        messages = _generate_thinking_messages(
            think_input=ThinkInput(goal="G", current_step="S", remaining_steps=2),
            agent_state=state,
            prompt_registry=bundled_prompt_registry,
        )
        rendered = "\n".join(m.content for m in messages)
        assert "You have 2 step(s) remaining" in rendered

    def test_thinking_messages_render_unlimited_when_none(self, bundled_prompt_registry):
        from llmcore.agents.cognitive.phases.think import _generate_thinking_messages

        state = EnhancedAgentState(goal="G")
        messages = _generate_thinking_messages(
            think_input=ThinkInput(goal="G", current_step="S"),
            agent_state=state,
            prompt_registry=bundled_prompt_registry,
        )
        rendered = "\n".join(m.content for m in messages)
        assert "You have unlimited step(s) remaining" in rendered
