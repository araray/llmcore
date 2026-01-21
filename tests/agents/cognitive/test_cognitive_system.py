# tests/agents/cognitive/test_cognitive_system.py
"""
Unit tests for the Enhanced Cognitive Cycle System.

Tests cover:
- Cognitive cycle models (EnhancedAgentState, CycleIteration, Phase I/O)
- PERCEIVE phase implementation
- Enhanced PLAN phase implementation
- Enhanced THINK phase implementation

References:
    - Dossier: Steps 2.4-2.5 (Cognitive Cycle Models & Enhanced Phases)
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from llmcore.agents.cognitive import (
    # Enums
    CognitivePhase,
    ConfidenceLevel,
    # Iteration Tracking
    CycleIteration,
    # Enhanced State
    EnhancedAgentState,
    IterationStatus,
    # Phase I/O Models
    PerceiveInput,
    PerceiveOutput,
    PlanInput,
    PlanOutput,
    ThinkInput,
    ThinkOutput,
    ValidateInput,
    ValidateOutput,
    ValidationResult,
    # Phase Functions
    perceive_phase,
    plan_phase,
    think_phase,
)
from llmcore.models import ToolCall, ToolResult

# =============================================================================
# MODEL TESTS
# =============================================================================


class TestCognitiveEnums:
    """Tests for cognitive enums."""

    def test_cognitive_phase_enum(self):
        """Test CognitivePhase enum values."""
        assert CognitivePhase.PERCEIVE.value == "perceive"
        assert CognitivePhase.PLAN.value == "plan"
        assert CognitivePhase.THINK.value == "think"
        assert CognitivePhase.VALIDATE.value == "validate"
        assert CognitivePhase.ACT.value == "act"
        assert CognitivePhase.OBSERVE.value == "observe"
        assert CognitivePhase.REFLECT.value == "reflect"
        assert CognitivePhase.UPDATE.value == "update"

    def test_iteration_status_enum(self):
        """Test IterationStatus enum values."""
        assert IterationStatus.IN_PROGRESS.value == "in_progress"
        assert IterationStatus.COMPLETED.value == "completed"
        assert IterationStatus.FAILED.value == "failed"

    def test_validation_result_enum(self):
        """Test ValidationResult enum values."""
        assert ValidationResult.APPROVED.value == "approved"
        assert ValidationResult.REJECTED.value == "rejected"

    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"


class TestPhaseInputOutputModels:
    """Tests for phase input/output models."""

    def test_perceive_input(self):
        """Test PerceiveInput model."""
        input_model = PerceiveInput(
            goal="Test goal", context_query="specific query", force_refresh=True
        )

        assert input_model.goal == "Test goal"
        assert input_model.context_query == "specific query"
        assert input_model.force_refresh is True

    def test_perceive_output(self):
        """Test PerceiveOutput model."""
        output = PerceiveOutput(
            retrieved_context=["context1", "context2"],
            working_memory_snapshot={"key": "value"},
            environmental_state={"sandbox": True},
        )

        assert len(output.retrieved_context) == 2
        assert output.working_memory_snapshot["key"] == "value"
        assert output.environmental_state["sandbox"] is True
        assert isinstance(output.perceived_at, datetime)

    def test_plan_input(self):
        """Test PlanInput model."""
        input_model = PlanInput(
            goal="Calculate factorial", context="Math problem", constraints="Must be efficient"
        )

        assert input_model.goal == "Calculate factorial"
        assert input_model.context == "Math problem"
        assert input_model.constraints == "Must be efficient"

    def test_plan_output(self):
        """Test PlanOutput model."""
        output = PlanOutput(
            plan_steps=["Step 1", "Step 2", "Step 3"],
            reasoning="Strategic approach",
            estimated_iterations=6,
            risks_identified=["Risk 1", "Risk 2"],
        )

        assert len(output.plan_steps) == 3
        assert output.reasoning == "Strategic approach"
        assert output.estimated_iterations == 6
        assert len(output.risks_identified) == 2

    def test_think_output(self):
        """Test ThinkOutput model."""
        tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "10!"})

        output = ThinkOutput(
            thought="I should use the calculator",
            proposed_action=tool_call,
            is_final_answer=False,
            confidence=ConfidenceLevel.HIGH,
        )

        assert "calculator" in output.thought
        assert output.proposed_action.name == "calculator"
        assert output.confidence == ConfidenceLevel.HIGH

    def test_validate_output(self):
        """Test ValidateOutput model."""
        output = ValidateOutput(
            result=ValidationResult.APPROVED,
            confidence=ConfidenceLevel.HIGH,
            concerns=["Minor issue"],
            suggestions=["Could improve by..."],
        )

        assert output.result == ValidationResult.APPROVED
        assert output.confidence == ConfidenceLevel.HIGH
        assert len(output.concerns) == 1
        assert len(output.suggestions) == 1


class TestCycleIteration:
    """Tests for CycleIteration model."""

    def test_create_iteration(self):
        """Test creating an iteration."""
        iteration = CycleIteration(iteration_number=1)

        assert iteration.iteration_number == 1
        assert iteration.status == IterationStatus.IN_PROGRESS
        assert isinstance(iteration.started_at, datetime)
        assert iteration.completed_at is None

    def test_mark_completed_success(self):
        """Test marking iteration as completed successfully."""
        iteration = CycleIteration(iteration_number=1)
        iteration.mark_completed(success=True)

        assert iteration.status == IterationStatus.COMPLETED
        assert iteration.completed_at is not None
        assert iteration.duration_ms > 0

    def test_mark_completed_failure(self):
        """Test marking iteration as failed."""
        iteration = CycleIteration(iteration_number=1)
        iteration.mark_completed(success=False)

        assert iteration.status == IterationStatus.FAILED
        assert iteration.completed_at is not None

    def test_mark_interrupted(self):
        """Test marking iteration as interrupted."""
        iteration = CycleIteration(iteration_number=1)
        iteration.mark_interrupted("Human intervention")

        assert iteration.status == IterationStatus.INTERRUPTED
        assert iteration.error == "Interrupted: Human intervention"

    def test_phases_completed(self):
        """Test tracking completed phases."""
        iteration = CycleIteration(iteration_number=1)

        # Initially no phases completed
        assert len(iteration.phases_completed) == 0

        # Add phase outputs
        iteration.perceive_output = PerceiveOutput()
        iteration.plan_output = PlanOutput(plan_steps=["step1"])

        completed = iteration.phases_completed
        assert CognitivePhase.PERCEIVE in completed
        assert CognitivePhase.PLAN in completed
        assert len(completed) == 2


class TestEnhancedAgentState:
    """Tests for EnhancedAgentState model."""

    def test_create_enhanced_state(self):
        """Test creating enhanced agent state."""
        state = EnhancedAgentState(goal="Test goal", session_id="session-123")

        assert state.goal == "Test goal"
        assert state.session_id == "session-123"
        assert len(state.iterations) == 0
        assert state.progress_estimate == 0.0
        assert state.overall_confidence == ConfidenceLevel.MEDIUM

    def test_start_iteration(self):
        """Test starting a new iteration."""
        state = EnhancedAgentState(goal="Test")

        iteration = state.start_iteration(iteration_number=1)

        assert iteration.iteration_number == 1
        assert state.current_iteration == iteration

    def test_complete_iteration(self):
        """Test completing an iteration."""
        state = EnhancedAgentState(goal="Test")

        iteration = state.start_iteration(1)
        iteration.total_tokens_used = 100

        state.complete_iteration(success=True)

        assert len(state.iterations) == 1
        assert state.current_iteration is None
        assert state.total_tokens_used == 100

    def test_add_multiple_iterations(self):
        """Test adding multiple iterations."""
        state = EnhancedAgentState(goal="Test")

        for i in range(3):
            iteration = state.start_iteration(i + 1)
            iteration.mark_completed(success=True)
            state.complete_iteration()

        assert len(state.iterations) == 3
        assert state.iteration_count == 3
        assert state.successful_iterations == 3
        assert state.failed_iterations == 0

    def test_update_plan(self):
        """Test updating the plan."""
        state = EnhancedAgentState(goal="Test")

        state.update_plan(["Step 1", "Step 2", "Step 3"], "Better approach")

        assert len(state.plan) == 3
        assert state.plan_version == 1
        assert state.plan_updated_at is not None
        assert len(state.plan_steps_status) == 3

    def test_working_memory(self):
        """Test working memory operations."""
        state = EnhancedAgentState(goal="Test")

        # Set values
        state.set_working_memory("key1", "value1")
        state.set_working_memory("key2", 42)

        # Get values
        assert state.get_working_memory("key1") == "value1"
        assert state.get_working_memory("key2") == 42
        assert state.get_working_memory("nonexistent", "default") == "default"

        # Clear
        state.clear_working_memory()
        assert len(state.working_memory) == 0

    def test_metrics_properties(self):
        """Test metrics calculation properties."""
        state = EnhancedAgentState(goal="Test")

        # Add some iterations
        for i in range(5):
            iteration = CycleIteration(iteration_number=i + 1)
            iteration.mark_completed(success=i < 3)  # 3 success, 2 failures
            state.add_iteration(iteration)

        assert state.iteration_count == 5
        assert state.successful_iterations == 3
        assert state.failed_iterations == 2


# =============================================================================
# PHASE IMPLEMENTATION TESTS
# =============================================================================


class TestPerceivePhase:
    """Tests for PERCEIVE phase implementation."""

    @pytest.mark.asyncio
    async def test_perceive_phase_basic(self):
        """Test basic PERCEIVE phase execution."""
        # Create mock memory manager
        memory_manager = Mock()
        memory_manager.retrieve_relevant_context = AsyncMock(
            return_value=[Mock(content="Context 1"), Mock(content="Context 2")]
        )

        # Create state
        state = EnhancedAgentState(goal="Test goal")
        state.set_working_memory("key", "value")

        # Create input
        perceive_input = PerceiveInput(goal="Test goal")

        # Execute phase
        output = await perceive_phase(
            agent_state=state, perceive_input=perceive_input, memory_manager=memory_manager
        )

        # Verify
        assert len(output.retrieved_context) == 2
        assert "Context 1" in output.retrieved_context
        assert output.working_memory_snapshot["key"] == "value"
        assert isinstance(output.perceived_at, datetime)

    @pytest.mark.asyncio
    async def test_perceive_with_sandbox(self):
        """Test PERCEIVE phase with active sandbox."""
        memory_manager = Mock()
        memory_manager.retrieve_relevant_context = AsyncMock(return_value=[])

        # Mock sandbox
        sandbox = Mock()
        sandbox.get_info = Mock(
            return_value={"provider": "docker", "status": "ready", "access_level": "restricted"}
        )

        state = EnhancedAgentState(goal="Test")
        perceive_input = PerceiveInput(goal="Test")

        output = await perceive_phase(
            agent_state=state,
            perceive_input=perceive_input,
            memory_manager=memory_manager,
            sandbox=sandbox,
        )

        # Verify sandbox state captured
        assert output.environmental_state["has_sandbox"] is True
        assert output.environmental_state["sandbox_provider"] == "docker"


class TestPlanPhase:
    """Tests for enhanced PLAN phase implementation."""

    @pytest.mark.asyncio
    async def test_plan_phase_basic(self):
        """Test basic PLAN phase execution."""
        # Mock provider manager
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test_provider")
        provider.default_model = "test_model"

        # Mock response content
        mock_content = """
PLAN:
1. Understand the problem
2. Break down into steps
3. Execute solution

REASONING:
This is a methodical approach.

RISKS:
- Time constraints
- Resource availability
"""
        # Mock chat_completion to return dict (like real provider)
        mock_response = {
            "choices": [{"message": {"content": mock_content}}],
            "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        }
        provider.chat_completion = AsyncMock(return_value=mock_response)
        provider.extract_response_content = Mock(return_value=mock_content)
        provider_manager.get_provider = Mock(return_value=provider)

        # Create state and input
        state = EnhancedAgentState(goal="Test")
        plan_input = PlanInput(goal="Calculate factorial", context="Math problem")

        # Execute phase
        output = await plan_phase(
            agent_state=state, plan_input=plan_input, provider_manager=provider_manager
        )

        # Verify
        assert len(output.plan_steps) == 3
        assert "Understand the problem" in output.plan_steps[0]
        assert "methodical" in output.reasoning.lower()
        assert len(output.risks_identified) == 2
        assert state.plan_version == 1


class TestThinkPhase:
    """Tests for enhanced THINK phase implementation."""

    @pytest.mark.asyncio
    async def test_think_phase_with_action(self):
        """Test THINK phase proposing an action."""
        # Mock provider manager
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test_provider")
        provider.default_model = "test_model"

        # Mock response content
        mock_content = """
Thought: I need to calculate the factorial using the calculator tool.
Action: calculator
Action Input: {"expression": "10!"}
"""
        # Mock chat_completion to return dict (like real provider)
        mock_response = {
            "choices": [{"message": {"content": mock_content}}],
            "usage": {"total_tokens": 150, "prompt_tokens": 80, "completion_tokens": 70},
        }
        provider.chat_completion = AsyncMock(return_value=mock_response)
        provider.extract_response_content = Mock(return_value=mock_content)
        provider_manager.get_provider = Mock(return_value=provider)

        # Mock memory and tool managers
        memory_manager = Mock()
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(
            return_value=[
                {"function": {"name": "calculator", "description": "Perform calculations"}}
            ]
        )

        # Create state and input
        state = EnhancedAgentState(goal="Calculate factorial")
        think_input = ThinkInput(
            goal="Calculate factorial of 10",
            current_step="Perform calculation",
            available_tools=tool_manager.get_tool_definitions(),
        )

        # Execute phase
        output = await think_phase(
            agent_state=state,
            think_input=think_input,
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            tool_manager=tool_manager,
        )

        # Verify
        assert "calculate" in output.thought.lower()
        assert output.proposed_action is not None
        assert output.proposed_action.name == "calculator"
        assert not output.is_final_answer
        assert state.pending_tool_call == output.proposed_action

    @pytest.mark.asyncio
    async def test_think_phase_with_final_answer(self):
        """Test THINK phase providing final answer."""
        provider_manager = Mock()
        provider = Mock()
        provider.get_name = Mock(return_value="test")
        provider.default_model = "test_model"

        # Mock response content
        mock_content = """
Thought: I have calculated the result and can provide the final answer.
Final Answer: The factorial of 10 is 3,628,800
"""
        # Mock chat_completion to return dict (like real provider)
        mock_response = {
            "choices": [{"message": {"content": mock_content}}],
            "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        }
        provider.chat_completion = AsyncMock(return_value=mock_response)
        provider.extract_response_content = Mock(return_value=mock_content)
        provider_manager.get_provider = Mock(return_value=provider)

        memory_manager = Mock()
        tool_manager = Mock()
        tool_manager.get_tool_definitions = Mock(return_value=[])

        state = EnhancedAgentState(goal="Calculate factorial")
        think_input = ThinkInput(
            goal="Calculate factorial of 10", current_step="Provide result", available_tools=[]
        )

        output = await think_phase(
            agent_state=state,
            think_input=think_input,
            provider_manager=provider_manager,
            memory_manager=memory_manager,
            tool_manager=tool_manager,
        )

        # Verify
        assert output.is_final_answer
        assert "3,628,800" in output.final_answer
        assert output.confidence == ConfidenceLevel.HIGH
        assert state.is_finished


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCognitiveIntegration:
    """Integration tests for cognitive components."""

    @pytest.mark.asyncio
    async def test_complete_iteration_lifecycle(self):
        """Test a complete iteration lifecycle through multiple phases."""
        # Setup
        state = EnhancedAgentState(goal="Calculate factorial of 10", session_id="test_session")

        # Start iteration
        iteration = state.start_iteration(1)

        # Simulate phase outputs
        iteration.perceive_output = PerceiveOutput(
            retrieved_context=["Math context"],
            working_memory_snapshot={},
            environmental_state={"sandbox": False},
        )

        iteration.plan_output = PlanOutput(
            plan_steps=["Step 1", "Step 2"], reasoning="Strategic approach", estimated_iterations=4
        )

        iteration.think_output = ThinkOutput(
            thought="Using calculator",
            proposed_action=ToolCall(
                id="call_1", name="calculator", arguments={"expression": "10!"}
            ),
            is_final_answer=False,
            confidence=ConfidenceLevel.HIGH,
        )

        # Complete iteration
        iteration.total_tokens_used = 500
        iteration.mark_completed(success=True)
        state.complete_iteration()

        # Verify
        assert state.iteration_count == 1
        assert state.successful_iterations == 1
        assert state.total_tokens_used == 500
        assert len(iteration.phases_completed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
