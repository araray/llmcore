# tests/agents/cognitive/test_skip_validation_guards.py
"""Deterministic guards under skip_validation (plan §2.3).

``skip_validation=True`` skips only the LLM validation judge: the
deterministic pre-checks (tool-registry membership + dangerous-pattern scan)
still run and their outputs are USED, mirroring validate_phase's state
side-effects. ``ValidationConfig.deterministic_guards=False`` is the escape
hatch back to blanket auto-approval.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llmcore.agents.cognitive.models import (
    ConfidenceLevel,
    EnhancedAgentState,
    PerceiveOutput,
    PlanOutput,
    ReflectOutput,
    ThinkOutput,
    ValidationResult,
)
from llmcore.agents.cognitive.phases.cycle import CognitiveCycle
from llmcore.agents.cognitive.phases.validate import deterministic_precheck
from llmcore.config.agents_config import AgentsConfig, ValidationConfig
from llmcore.models import ToolCall, ToolResult

CYCLE_MODULE = "llmcore.agents.cognitive.phases.cycle"


def _tool_manager(loaded: bool = True) -> Mock:
    manager = Mock()
    manager.is_tool_loaded = Mock(return_value=loaded)
    manager.get_tool_names = Mock(return_value=["calculator"])
    manager.get_tool_definitions = Mock(return_value=[])
    manager.execute_tool = AsyncMock(
        return_value=ToolResult(tool_call_id="c1", content="4", is_error=False)
    )
    return manager


async def _run_skip_validation_iteration(
    action: ToolCall,
    tool_manager: Mock,
    bundled_prompt_registry,
    agents_config: AgentsConfig | None = None,
):
    """Run one run_iteration with skip_validation=True and a fixed THINK action.

    PERCEIVE/PLAN/THINK/REFLECT are patched (no LLM); VALIDATE/ACT/OBSERVE/
    UPDATE run for real so the deterministic-guard path is exercised
    end-to-end, including its state side-effects.
    """
    provider_manager = Mock()
    cycle = CognitiveCycle(
        provider_manager=provider_manager,
        memory_manager=Mock(),
        storage_manager=Mock(),
        tool_manager=tool_manager,
        prompt_registry=bundled_prompt_registry,
        agents_config=agents_config,
    )
    state = EnhancedAgentState(goal="Test goal", session_id="s")

    with (
        patch(
            f"{CYCLE_MODULE}.perceive_phase",
            AsyncMock(return_value=PerceiveOutput(retrieved_context=[])),
        ),
        patch(
            f"{CYCLE_MODULE}.plan_phase",
            AsyncMock(return_value=PlanOutput(plan_steps=["do it"], reasoning="r")),
        ),
        patch(
            f"{CYCLE_MODULE}.think_phase",
            AsyncMock(
                return_value=ThinkOutput(thought="use the tool", proposed_action=action)
            ),
        ),
        patch(
            f"{CYCLE_MODULE}.reflect_phase",
            AsyncMock(
                return_value=ReflectOutput(evaluation="ok", progress_estimate=0.5)
            ),
        ),
    ):
        iteration = await cycle.run_iteration(
            agent_state=state, session_id="s", skip_validation=True
        )

    return iteration, state, provider_manager


# =============================================================================
# deterministic_precheck unit behavior
# =============================================================================


class TestDeterministicPrecheck:
    def test_clean_action_returns_none(self):
        action = ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
        assert deterministic_precheck(action, _tool_manager(loaded=True)) is None

    def test_unknown_tool_rejected(self):
        action = ToolCall(id="c1", name="ghost_tool", arguments={})
        output = deterministic_precheck(action, _tool_manager(loaded=False))

        assert output is not None
        assert output.result == ValidationResult.REJECTED
        assert output.requires_human_approval is False
        assert any("not loaded" in concern for concern in output.concerns)

    def test_dangerous_pattern_requires_approval(self):
        action = ToolCall(
            id="c1", name="calculator", arguments={"command": "sudo rm -rf /"}
        )
        output = deterministic_precheck(
            action, _tool_manager(loaded=True), goal="clean up", reasoning="tidy"
        )

        assert output is not None
        assert output.result == ValidationResult.REQUIRES_HUMAN_APPROVAL
        assert output.requires_human_approval is True
        assert output.approval_prompt is not None
        assert "clean up" in output.approval_prompt
        assert output.confidence == ConfidenceLevel.HIGH

    def test_no_tool_manager_skips_registry_but_keeps_danger_scan(self):
        clean = ToolCall(id="c1", name="anything", arguments={"x": 1})
        assert deterministic_precheck(clean, None) is None

        dangerous = ToolCall(id="c2", name="anything", arguments={"cmd": "chmod 777 /etc"})
        output = deterministic_precheck(dangerous, None)
        assert output is not None
        assert output.result == ValidationResult.REQUIRES_HUMAN_APPROVAL

    @pytest.mark.asyncio
    async def test_validate_phase_dedupes_through_precheck(self, bundled_prompt_registry):
        """validate_phase's deterministic early-returns keep their contract."""
        from llmcore.agents.cognitive.models import ValidateInput
        from llmcore.agents.cognitive.phases.validate import validate_phase

        state = EnhancedAgentState(goal="G")
        output = await validate_phase(
            agent_state=state,
            validate_input=ValidateInput(
                goal="Clean up files",
                proposed_action=ToolCall(
                    id="c1", name="execute_shell", arguments={"command": "rm -rf /"}
                ),
                reasoning="cleanup",
            ),
            provider_manager=Mock(),  # must never be reached
            prompt_registry=bundled_prompt_registry,
        )

        assert output.result == ValidationResult.REQUIRES_HUMAN_APPROVAL
        assert output.approval_prompt is not None
        assert "Clean up files" in output.approval_prompt


# =============================================================================
# skip_validation cycle behavior
# =============================================================================


class TestSkipValidationGuards:
    @pytest.mark.asyncio
    async def test_dangerous_args_pause_for_approval(self, bundled_prompt_registry):
        action = ToolCall(
            id="c1", name="calculator", arguments={"command": "sudo rm -rf /tmp/x"}
        )
        tool_manager = _tool_manager(loaded=True)

        iteration, state, provider_manager = await _run_skip_validation_iteration(
            action, tool_manager, bundled_prompt_registry
        )

        assert iteration.validate_output is not None
        assert iteration.validate_output.result == ValidationResult.REQUIRES_HUMAN_APPROVAL
        # State side-effects mirror validate_phase.
        assert state.awaiting_human_approval is True
        assert state.pending_approval_prompt
        assert state.pending_validation is not None
        assert len(state.validation_history) == 1
        # ACT paused instead of executing.
        tool_manager.execute_tool.assert_not_awaited()
        assert iteration.act_output.success is False
        assert "human approval" in iteration.act_output.tool_result.content.lower()
        # UPDATE stops the loop for HITL.
        assert iteration.update_output.should_continue is False
        # Zero LLM calls: the judge stayed skipped.
        provider_manager.get_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_tool_rejected(self, bundled_prompt_registry):
        action = ToolCall(id="c1", name="ghost_tool", arguments={"x": 1})
        tool_manager = _tool_manager(loaded=False)

        iteration, state, provider_manager = await _run_skip_validation_iteration(
            action, tool_manager, bundled_prompt_registry
        )

        assert iteration.validate_output.result == ValidationResult.REJECTED
        assert state.awaiting_human_approval is False
        assert len(state.validation_history) == 1
        tool_manager.execute_tool.assert_not_awaited()
        assert iteration.act_output.success is False
        assert iteration.act_output.tool_result.is_error is True
        provider_manager.get_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_action_auto_approved_with_zero_llm_calls(
        self, bundled_prompt_registry
    ):
        action = ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
        tool_manager = _tool_manager(loaded=True)

        iteration, state, provider_manager = await _run_skip_validation_iteration(
            action, tool_manager, bundled_prompt_registry
        )

        assert iteration.validate_output.result == ValidationResult.APPROVED
        assert iteration.validate_output.suggestions == ["LLM validation skipped"]
        assert state.awaiting_human_approval is False
        assert state.validation_history == []  # fabricated approvals are not history
        tool_manager.execute_tool.assert_awaited_once()
        assert iteration.act_output.success is True
        provider_manager.get_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_deterministic_guards_escape_hatch(self, bundled_prompt_registry):
        """deterministic_guards=False restores blanket auto-approval."""
        config = AgentsConfig()
        config.validation.deterministic_guards = False

        action = ToolCall(
            id="c1", name="ghost_tool", arguments={"command": "sudo rm -rf /tmp/x"}
        )
        tool_manager = _tool_manager(loaded=False)

        iteration, state, provider_manager = await _run_skip_validation_iteration(
            action, tool_manager, bundled_prompt_registry, agents_config=config
        )

        assert iteration.validate_output.result == ValidationResult.APPROVED
        assert iteration.validate_output.suggestions == ["LLM validation skipped"]
        assert state.awaiting_human_approval is False
        tool_manager.execute_tool.assert_awaited_once()
        provider_manager.get_provider.assert_not_called()

    def test_validation_config_defaults(self):
        config = AgentsConfig()
        assert isinstance(config.validation, ValidationConfig)
        assert config.validation.deterministic_guards is True
