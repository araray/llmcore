# tests/agents/cognitive/test_cognitive_phases_advanced.py
"""
Unit tests for advanced cognitive phases and orchestrator.

Tests cover:
- VALIDATE phase implementation
- ACT phase implementation
- OBSERVE phase implementation
- REFLECT phase implementation
- UPDATE phase implementation
- CognitiveCycle orchestrator

References:
    - Dossier: Steps 2.6-2.7 (Advanced Phases & Orchestrator)
"""

from unittest.mock import AsyncMock, Mock

import pytest

from llmcore.agents.cognitive import (
    ActInput,
    CognitiveCycle,
    ConfidenceLevel,
    # Enhanced State
    EnhancedAgentState,
    ObserveInput,
    ReflectInput,
    ReflectOutput,
    UpdateInput,
    ValidateInput,
    ValidationResult,
    act_phase,
    observe_phase,
    reflect_phase,
    update_phase,
    # Phase Functions
    validate_phase,
)
from llmcore.models import ToolCall, ToolResult

# =============================================================================
# VALIDATE PHASE TESTS
# =============================================================================


class TestValidatePhase:
    """Tests for VALIDATE phase implementation."""

    @pytest.mark.asyncio
    async def test_validate_phase_approved(self):
        """Test VALIDATE phase approving a safe action."""
        # Mock provider manager
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test_provider")
        provider.default_model = "test_model"

        # Mock response content - approval
        mock_content = """
APPROVED: yes
CONFIDENCE: high
CONCERNS: None
SUGGESTIONS: None
"""
        # Mock chat_completion to return dict (like real provider)
        mock_response = {
            "choices": [{"message": {"content": mock_content}}],
            "usage": {"total_tokens": 80, "prompt_tokens": 50, "completion_tokens": 30},
        }
        provider.chat_completion = AsyncMock(return_value=mock_response)
        provider.extract_response_content = Mock(return_value=mock_content)
        provider_manager.get_provider = Mock(return_value=provider)

        # Create state and input
        state = EnhancedAgentState(goal="Test")
        validate_input = ValidateInput(
            goal="List directory contents",
            proposed_action=ToolCall(
                id="call_1", name="execute_shell", arguments={"command": "ls -la"}
            ),
            reasoning="Need to see what files are present",
            risk_tolerance="medium",
        )

        # Execute phase
        output = await validate_phase(
            agent_state=state, validate_input=validate_input, provider_manager=provider_manager
        )

        # Verify
        assert output.result == ValidationResult.APPROVED
        assert output.confidence == ConfidenceLevel.HIGH
        assert not output.requires_human_approval

    @pytest.mark.asyncio
    async def test_validate_phase_dangerous_pattern(self):
        """Test VALIDATE phase catching dangerous patterns."""
        provider_manager = Mock()

        state = EnhancedAgentState(goal="Test")
        validate_input = ValidateInput(
            goal="Clean up files",
            proposed_action=ToolCall(
                id="call_1",
                name="execute_shell",
                arguments={"command": "rm -rf /"},  # Dangerous!
            ),
            reasoning="Remove all files",
            risk_tolerance="medium",
        )

        # Execute phase (should catch dangerous pattern without calling LLM)
        output = await validate_phase(
            agent_state=state, validate_input=validate_input, provider_manager=provider_manager
        )

        # Verify
        assert output.result == ValidationResult.REQUIRES_HUMAN_APPROVAL
        assert output.requires_human_approval
        assert len(output.concerns) > 0
        assert "dangerous" in output.concerns[0].lower()

    @pytest.mark.asyncio
    async def test_validate_phase_low_confidence(self):
        """Test VALIDATE phase with low confidence requiring HITL."""
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test")
        provider.default_model = "test_model"

        mock_response = Mock()
        mock_response.content = """
APPROVED: yes
CONFIDENCE: low
CONCERNS:
- Uncertain about side effects
- Limited context available
SUGGESTIONS:
- Request human review
"""
        mock_response.usage = Mock(total_tokens=90)

        provider.chat = AsyncMock(return_value=mock_response)
        provider_manager.get_provider = Mock(return_value=provider)

        state = EnhancedAgentState(goal="Test")
        validate_input = ValidateInput(
            goal="Modify system config",
            proposed_action=ToolCall(
                id="call_1",
                name="save_file",
                arguments={"path": "/etc/config", "content": "new config"},
            ),
            reasoning="Update configuration",
            risk_tolerance="low",
        )

        output = await validate_phase(
            agent_state=state, validate_input=validate_input, provider_manager=provider_manager
        )

        # Low confidence should trigger HITL
        assert output.result == ValidationResult.REQUIRES_HUMAN_APPROVAL
        assert output.requires_human_approval


# =============================================================================
# ACT PHASE TESTS
# =============================================================================


class TestActPhase:
    """Tests for ACT phase implementation."""

    @pytest.mark.asyncio
    async def test_act_phase_success(self):
        """Test ACT phase successful execution."""
        # Mock tool manager
        tool_manager = Mock()
        tool_result = ToolResult(tool_call_id="call_1", content="4", is_error=False)
        tool_manager.execute_tool = AsyncMock(return_value=tool_result)

        # Create state and input
        state = EnhancedAgentState(goal="Calculate")
        act_input = ActInput(
            tool_call=ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"}),
            validation_result=ValidationResult.APPROVED,
        )

        # Execute phase
        output = await act_phase(agent_state=state, act_input=act_input, tool_manager=tool_manager)

        # Verify
        assert output.success
        assert output.tool_result.content == "4"
        assert not output.tool_result.is_error
        assert output.execution_time_ms > 0
        assert state.total_tool_calls == 1

    @pytest.mark.asyncio
    async def test_act_phase_rejected(self):
        """Test ACT phase with rejected validation."""
        tool_manager = Mock()

        state = EnhancedAgentState(goal="Test")
        act_input = ActInput(
            tool_call=ToolCall(id="call_1", name="dangerous_op", arguments={}),
            validation_result=ValidationResult.REJECTED,
        )

        output = await act_phase(agent_state=state, act_input=act_input, tool_manager=tool_manager)

        # Should not execute
        assert not output.success
        assert output.tool_result.is_error
        assert "rejected" in output.tool_result.content.lower()

    @pytest.mark.asyncio
    async def test_act_phase_with_retry(self):
        """Test ACT phase retry logic."""
        tool_manager = Mock()

        # Fail first time, succeed second time
        error_result = ToolResult(tool_call_id="call_1", content="Network error", is_error=True)
        success_result = ToolResult(tool_call_id="call_1", content="Success", is_error=False)

        tool_manager.execute_tool = AsyncMock(side_effect=[error_result, success_result])

        state = EnhancedAgentState(goal="Test")
        act_input = ActInput(
            tool_call=ToolCall(id="call_1", name="api_call", arguments={}),
            validation_result=ValidationResult.APPROVED,
        )

        output = await act_phase(
            agent_state=state, act_input=act_input, tool_manager=tool_manager, max_retries=1
        )

        # Should succeed on retry
        assert output.success
        assert output.tool_result.content == "Success"


# =============================================================================
# OBSERVE PHASE TESTS
# =============================================================================


class TestObservePhase:
    """Tests for OBSERVE phase implementation."""

    @pytest.mark.asyncio
    async def test_observe_phase_success(self):
        """Test OBSERVE phase with successful result."""
        state = EnhancedAgentState(goal="Calculate")
        observe_input = ObserveInput(
            action_taken=ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"}),
            action_result=ToolResult(tool_call_id="call_1", content="4", is_error=False),
            expected_outcome="The result should be 4",
        )

        output = await observe_phase(agent_state=state, observe_input=observe_input)

        # Verify
        assert "calculator" in output.observation
        assert output.matches_expectation is True
        assert len(output.insights) > 0
        assert not output.follow_up_needed

    @pytest.mark.asyncio
    async def test_observe_phase_error(self):
        """Test OBSERVE phase with error result."""
        state = EnhancedAgentState(goal="Test")
        observe_input = ObserveInput(
            action_taken=ToolCall(
                id="call_1", name="execute_shell", arguments={"command": "invalid"}
            ),
            action_result=ToolResult(
                tool_call_id="call_1", content="Command not found", is_error=True
            ),
        )

        output = await observe_phase(agent_state=state, observe_input=observe_input)

        # Errors should need follow-up
        assert output.follow_up_needed
        assert "ERROR" in output.observation
        assert len(output.insights) > 0


# =============================================================================
# REFLECT PHASE TESTS
# =============================================================================


class TestReflectPhase:
    """Tests for REFLECT phase implementation."""

    @pytest.mark.asyncio
    async def test_reflect_phase_progress(self):
        """Test REFLECT phase evaluating progress."""
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test")
        provider.default_model = "test_model"

        # Mock response content
        mock_content = """
EVALUATION: Successfully completed calculation step.
PROGRESS: 50%
INSIGHTS:
- Calculation tool works correctly
- Result matches expectations
PLAN_UPDATE: no
STEP_COMPLETED: yes
NEXT_FOCUS: Move to next step in plan
"""
        # Mock chat_completion to return dict (like real provider)
        mock_response = {
            "choices": [{"message": {"content": mock_content}}],
            "usage": {"total_tokens": 120, "prompt_tokens": 70, "completion_tokens": 50},
        }
        provider.chat_completion = AsyncMock(return_value=mock_response)
        provider.extract_response_content = Mock(return_value=mock_content)
        provider_manager.get_provider = Mock(return_value=provider)

        state = EnhancedAgentState(goal="Calculate factorial")
        state.plan = ["Calculate 10!", "Format result", "Return answer"]

        reflect_input = ReflectInput(
            goal="Calculate factorial of 10",
            plan=state.plan,
            current_step_index=0,
            last_action=ToolCall(id="call_1", name="calculator", arguments={}),
            observation="Result: 3628800",
            iteration_number=1,
        )

        output = await reflect_phase(
            agent_state=state, reflect_input=reflect_input, provider_manager=provider_manager
        )

        # Verify
        assert output.progress_estimate == 0.5
        assert not output.plan_needs_update
        assert output.step_completed
        assert len(output.insights) > 0
        assert output.next_focus is not None

    @pytest.mark.asyncio
    async def test_reflect_phase_plan_update(self):
        """Test REFLECT phase recommending plan update."""
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test")
        provider.default_model = "test_model"

        # Mock response content
        mock_content = """
EVALUATION: Approach not working, need to change strategy.
PROGRESS: 25%
INSIGHTS:
- Current approach inefficient
- Better method available
PLAN_UPDATE: yes
STEP_COMPLETED: no
NEXT_FOCUS: Try alternative approach

UPDATED PLAN:
1. Use optimized algorithm
2. Cache intermediate results
3. Return final answer
"""
        # Mock chat_completion to return dict (like real provider)
        mock_response = {
            "choices": [{"message": {"content": mock_content}}],
            "usage": {"total_tokens": 130, "prompt_tokens": 80, "completion_tokens": 50},
        }
        provider.chat_completion = AsyncMock(return_value=mock_response)
        provider.extract_response_content = Mock(return_value=mock_content)
        provider_manager.get_provider = Mock(return_value=provider)

        state = EnhancedAgentState(goal="Optimize")
        state.plan = ["Step 1", "Step 2"]

        reflect_input = ReflectInput(
            goal="Optimize calculation",
            plan=state.plan,
            current_step_index=0,
            last_action=ToolCall(id="call_1", name="execute", arguments={}),
            observation="Too slow",
            iteration_number=2,
        )

        output = await reflect_phase(
            agent_state=state, reflect_input=reflect_input, provider_manager=provider_manager
        )

        # Verify
        assert output.plan_needs_update
        assert output.updated_plan is not None
        assert len(output.updated_plan) == 3


# =============================================================================
# UPDATE PHASE TESTS
# =============================================================================


class TestUpdatePhase:
    """Tests for UPDATE phase implementation."""

    @pytest.mark.asyncio
    async def test_update_phase_basic(self):
        """Test UPDATE phase with basic state updates."""
        state = EnhancedAgentState(goal="Test")
        state.plan = ["Step 1", "Step 2", "Step 3"]
        state.plan_steps_status = ["pending", "pending", "pending"]
        state.current_plan_step_index = 0

        # Create reflection output
        reflect_output = ReflectOutput(
            evaluation="Step completed",
            progress_estimate=0.33,
            insights=["Learned X", "Learned Y"],
            plan_needs_update=False,
            step_completed=True,
            next_focus="Continue to step 2",
        )

        update_input = UpdateInput(reflection=reflect_output, current_state=state)

        output = await update_phase(agent_state=state, update_input=update_input)

        # Verify
        assert state.progress_estimate == 0.33
        assert state.plan_steps_status[0] == "completed"
        assert state.current_plan_step_index == 1
        assert state.get_working_memory("next_focus") == "Continue to step 2"
        assert len(state.get_working_memory("insights", [])) == 2
        assert output.should_continue

    @pytest.mark.asyncio
    async def test_update_phase_plan_update(self):
        """Test UPDATE phase applying plan update."""
        state = EnhancedAgentState(goal="Test")
        state.plan = ["Old step 1", "Old step 2"]
        old_version = state.plan_version

        reflect_output = ReflectOutput(
            evaluation="Plan needs update",
            progress_estimate=0.2,
            insights=[],
            plan_needs_update=True,
            updated_plan=["New step 1", "New step 2", "New step 3"],
            step_completed=False,
        )

        update_input = UpdateInput(reflection=reflect_output, current_state=state)

        output = await update_phase(agent_state=state, update_input=update_input)

        # Verify plan was updated
        assert len(state.plan) == 3
        assert state.plan[0] == "New step 1"
        assert state.plan_version > old_version
        assert "plan_updated" in output.state_updates

    @pytest.mark.asyncio
    async def test_update_phase_completion(self):
        """Test UPDATE phase detecting task completion."""
        state = EnhancedAgentState(goal="Test")
        state.plan = ["Step 1"]
        state.plan_steps_status = ["pending"]

        # 100% progress
        reflect_output = ReflectOutput(
            evaluation="Task complete",
            progress_estimate=1.0,
            insights=["Task finished"],
            plan_needs_update=False,
            step_completed=True,
        )

        update_input = UpdateInput(reflection=reflect_output, current_state=state)

        output = await update_phase(agent_state=state, update_input=update_input)

        # Should not continue
        assert not output.should_continue
        assert state.is_finished


# =============================================================================
# COGNITIVE CYCLE ORCHESTRATOR TESTS
# =============================================================================


class TestCognitiveCycle:
    """Tests for CognitiveCycle orchestrator."""

    @pytest.mark.asyncio
    async def test_cognitive_cycle_initialization(self):
        """Test CognitiveCycle initialization."""
        provider_manager = Mock()
        memory_manager = Mock()
        storage_manager = Mock()
        tool_manager = Mock()

        cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            storage_manager=storage_manager,
            tool_manager=tool_manager,
        )

        assert cycle.provider_manager == provider_manager
        assert cycle.memory_manager == memory_manager
        assert cycle.storage_manager == storage_manager
        assert cycle.tool_manager == tool_manager

    @pytest.mark.asyncio
    async def test_cognitive_cycle_single_iteration(self):
        """Test running a single cognitive iteration."""
        # Create comprehensive mocks
        provider_manager = self._create_mock_provider_manager()
        memory_manager = self._create_mock_memory_manager()
        storage_manager = self._create_mock_storage_manager()
        tool_manager = self._create_mock_tool_manager()

        cycle = CognitiveCycle(
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            storage_manager=storage_manager,
            tool_manager=tool_manager,
        )

        state = EnhancedAgentState(goal="Calculate 2+2", session_id="test_session")

        # Run iteration
        iteration = await cycle.run_iteration(agent_state=state, session_id="test_session")

        # Verify iteration completed
        assert iteration is not None
        assert iteration.iteration_number == 1
        assert len(state.iterations) == 1

    def _create_mock_provider_manager(self):
        """Create mock provider manager."""
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test")
        provider.default_model = "test_model"

        # Mock LLM responses
        async def mock_chat(messages, model, **kwargs):
            # Return different responses based on context
            user_content = messages[-1].content if messages else ""

            response = Mock()
            if "PLAN" in user_content:
                response.content = "PLAN:\n1. Step 1\nREASONING: Simple plan"
            elif "Thought:" in user_content or "ReAct" in user_content:
                response.content = "Thought: Calculate\nFinal Answer: 4"
            elif "VALIDATE" in user_content:
                response.content = "APPROVED: yes\nCONFIDENCE: high"
            elif "REFLECT" in user_content:
                response.content = (
                    "EVALUATION: Done\nPROGRESS: 100%\nPLAN_UPDATE: no\nSTEP_COMPLETED: yes"
                )
            else:
                response.content = "Default response"

            response.usage = Mock(total_tokens=100)
            return response

        provider.chat = AsyncMock(side_effect=mock_chat)
        provider_manager.get_provider = Mock(return_value=provider)

        return provider_manager

    def _create_mock_memory_manager(self):
        """Create mock memory manager."""
        memory_manager = Mock()
        memory_manager.retrieve_relevant_context = AsyncMock(
            return_value=[Mock(content="Context item")]
        )
        return memory_manager

    def _create_mock_storage_manager(self):
        """Create mock storage manager."""
        storage_manager = Mock()
        storage_manager.store_episode = AsyncMock(return_value=None)
        return storage_manager

    def _create_mock_tool_manager(self):
        """Create mock tool manager."""
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(
            return_value=[{"function": {"name": "calculator", "description": "Calculate"}}]
        )
        tool_manager.execute_tool = AsyncMock(
            return_value=ToolResult(tool_call_id="call_1", content="4", is_error=False)
        )
        return tool_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
