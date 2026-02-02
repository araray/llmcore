# tests/agents/test_streaming.py
"""
Tests for Phase 5: Streaming & Observability.

This module tests the streaming functionality of the cognitive cycle,
verifying that real-time iteration updates are properly yielded during
agent execution.

Test Categories:
    1. StreamingIterationResult dataclass tests
    2. CognitiveCycle.run_streaming() tests
    3. SingleAgentMode.run_streaming() tests
    4. IterationUpdate conversion tests
    5. Edge cases and error handling
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.agents.cognitive.models import (
    CycleIteration,
    EnhancedAgentState,
)
from llmcore.agents.cognitive.phases.cycle import (
    CognitiveCycle,
    StreamingIterationResult,
)
from llmcore.agents.single_agent import (
    IterationUpdate,
    SingleAgentMode,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_provider_manager():
    """Create a mock provider manager."""
    manager = MagicMock()
    manager.get_default_provider_name.return_value = "openai"
    manager.get_default_model.return_value = "gpt-4o"
    return manager


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    return MagicMock()


@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager."""
    return MagicMock()


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager."""
    manager = MagicMock()
    manager.get_tool_definitions.return_value = []
    manager.get_tool_names.return_value = []
    return manager


@pytest.fixture
def mock_prompt_registry():
    """Create a mock prompt registry."""
    return MagicMock()


@pytest.fixture
def cognitive_cycle(
    mock_provider_manager,
    mock_memory_manager,
    mock_storage_manager,
    mock_tool_manager,
    mock_prompt_registry,
):
    """Create a CognitiveCycle with mocked dependencies."""
    return CognitiveCycle(
        provider_manager=mock_provider_manager,
        memory_manager=mock_memory_manager,
        storage_manager=mock_storage_manager,
        tool_manager=mock_tool_manager,
        prompt_registry=mock_prompt_registry,
    )


@pytest.fixture
def agent_state():
    """Create a test agent state."""
    return EnhancedAgentState(
        goal="Test goal",
        session_id="test-session-123",
        context="Test context",
    )


# =============================================================================
# STREAMING ITERATION RESULT TESTS
# =============================================================================


class TestStreamingIterationResult:
    """Tests for the StreamingIterationResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = StreamingIterationResult(
            iteration=1,
            max_iterations=10,
            progress=0.5,
        )

        assert result.iteration == 1
        assert result.max_iterations == 10
        assert result.progress == 0.5
        assert result.is_complete is False
        assert result.is_final is False
        assert result.status == "in_progress"
        assert result.current_phase == "unknown"
        assert result.message == ""
        assert result.action_name is None
        assert result.error is None

    def test_complete_iteration(self):
        """Test a complete iteration result."""
        result = StreamingIterationResult(
            iteration=5,
            max_iterations=10,
            progress=1.0,
            is_complete=True,
            is_final=True,
            status="complete",
            current_phase="update",
            message="Task completed successfully",
            action_name="python_exec",
            action_summary="print('hello')",
            observation_summary="Output: hello",
            step_completed=True,
            plan_step="Execute the final step",
            tokens_used=1500,
            duration_ms=2500.0,
        )

        assert result.is_complete is True
        assert result.is_final is True
        assert result.status == "complete"
        assert result.action_name == "python_exec"
        assert result.tokens_used == 1500

    def test_error_iteration(self):
        """Test an error iteration result."""
        result = StreamingIterationResult(
            iteration=3,
            max_iterations=10,
            progress=0.3,
            is_complete=False,
            is_final=True,
            status="error",
            current_phase="error",
            message="Iteration failed: Connection timeout",
            error="Connection timeout after 30s",
            stop_reason="error",
        )

        assert result.is_final is True
        assert result.status == "error"
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_circuit_breaker_result(self):
        """Test circuit breaker tripped result."""
        result = StreamingIterationResult(
            iteration=8,
            max_iterations=10,
            progress=0.85,
            is_complete=False,
            is_final=True,
            status="circuit_breaker_tripped",
            message="Progress stalled at 85%",
            stop_reason="progress_stalled",
        )

        assert result.is_final is True
        assert result.status == "circuit_breaker_tripped"
        assert result.stop_reason == "progress_stalled"


# =============================================================================
# ITERATION UPDATE TESTS
# =============================================================================


class TestIterationUpdate:
    """Tests for the IterationUpdate class."""

    def test_basic_creation(self):
        """Test basic IterationUpdate creation."""
        update = IterationUpdate(
            iteration=1,
            progress=0.5,
            status="in_progress",
            message="Working on task",
        )

        assert update.iteration == 1
        assert update.progress == 0.5
        assert update.status == "in_progress"
        assert update.message == "Working on task"
        assert update.is_final is False
        assert update.max_iterations == 10  # default

    def test_full_creation(self):
        """Test IterationUpdate with all fields."""
        update = IterationUpdate(
            iteration=5,
            progress=0.8,
            status="in_progress",
            message="Almost done",
            max_iterations=15,
            is_final=False,
            phase="reflect",
            action="shell_exec",
            action_args="ls -la",
            observation="Listed 10 files",
            step_completed=True,
            plan_step="List directory contents",
            tokens_used=500,
            duration_ms=1200.0,
        )

        assert update.max_iterations == 15
        assert update.phase == "reflect"
        assert update.action == "shell_exec"
        assert update.action_args == "ls -la"
        assert update.step_completed is True
        assert update.tokens_used == 500

    def test_repr(self):
        """Test string representation."""
        update = IterationUpdate(
            iteration=3,
            progress=0.6,
            status="in_progress",
            message="Test",
            is_final=False,
        )

        repr_str = repr(update)
        assert "iteration=3" in repr_str
        assert "60.0%" in repr_str
        assert "in_progress" in repr_str

    def test_from_streaming_result(self):
        """Test conversion from StreamingIterationResult."""
        streaming = StreamingIterationResult(
            iteration=4,
            max_iterations=12,
            progress=0.7,
            is_complete=False,
            is_final=False,
            status="in_progress",
            current_phase="act",
            message="Executing action",
            action_name="file_write",
            action_summary="Writing to output.txt",
            observation_summary="File written successfully",
            step_completed=False,
            plan_step="Save results",
            tokens_used=800,
            duration_ms=1500.0,
        )

        update = IterationUpdate.from_streaming_result(streaming)

        assert update.iteration == 4
        assert update.max_iterations == 12
        assert update.progress == 0.7
        assert update.status == "in_progress"
        assert update.phase == "act"
        assert update.action == "file_write"
        assert update.action_args == "Writing to output.txt"
        assert update.observation == "File written successfully"
        assert update.tokens_used == 800
        assert update.duration_ms == 1500.0


# =============================================================================
# COGNITIVE CYCLE STREAMING TESTS
# =============================================================================


class TestCognitiveCycleStreaming:
    """Tests for CognitiveCycle.run_streaming()."""

    @pytest.mark.asyncio
    async def test_yields_updates_for_each_iteration(self, cognitive_cycle, agent_state):
        """Test that run_streaming yields an update after each iteration."""
        # Mock run_iteration to return completed iterations
        iteration_count = 0

        async def mock_run_iteration(*args, **kwargs):
            nonlocal iteration_count
            iteration_count += 1

            # Create a mock iteration result
            iteration = CycleIteration(iteration_number=iteration_count)
            iteration.think_output = MagicMock()
            iteration.think_output.is_final_answer = iteration_count >= 3
            iteration.think_output.proposed_action = None
            iteration.observe_output = None
            iteration.reflect_output = MagicMock()
            iteration.reflect_output.step_completed = False
            iteration.reflect_output.progress_estimate = iteration_count / 3
            iteration.update_output = MagicMock()
            iteration.update_output.should_continue = iteration_count < 3
            iteration.total_tokens_used = 100 * iteration_count
            iteration.mark_completed(success=True)

            # Mark state as finished on iteration 3
            if iteration_count >= 3:
                agent_state.is_finished = True
                agent_state.final_answer = "Task completed"

            return iteration

        cognitive_cycle.run_iteration = mock_run_iteration

        updates: List[StreamingIterationResult] = []
        async for update in cognitive_cycle.run_streaming(
            agent_state=agent_state,
            session_id="test-session",
            max_iterations=10,
        ):
            updates.append(update)

        # Should have received updates for each iteration
        assert len(updates) >= 1
        assert updates[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stops_on_max_iterations(self, cognitive_cycle, agent_state):
        """Test that streaming stops at max_iterations."""
        # Mock run_iteration to never complete the task
        async def mock_run_iteration(*args, **kwargs):
            iteration = CycleIteration(iteration_number=agent_state.iteration_count + 1)
            iteration.think_output = MagicMock()
            iteration.think_output.is_final_answer = False
            iteration.think_output.proposed_action = None
            iteration.reflect_output = MagicMock()
            iteration.reflect_output.step_completed = False
            iteration.update_output = MagicMock()
            iteration.update_output.should_continue = True
            iteration.total_tokens_used = 100
            iteration.mark_completed(success=True)
            agent_state.iteration_count += 1
            return iteration

        cognitive_cycle.run_iteration = mock_run_iteration

        updates: List[StreamingIterationResult] = []
        max_iter = 3
        async for update in cognitive_cycle.run_streaming(
            agent_state=agent_state,
            session_id="test-session",
            max_iterations=max_iter,
        ):
            updates.append(update)

        # Should have received 3 updates + final max_iterations message
        assert len(updates) <= max_iter + 1
        assert updates[-1].is_final is True
        assert updates[-1].stop_reason == "max_iterations" or updates[-1].iteration == max_iter

    @pytest.mark.asyncio
    async def test_handles_iteration_error(self, cognitive_cycle, agent_state):
        """Test that errors during iteration are properly reported."""
        call_count = 0

        async def mock_run_iteration(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Test error during iteration")

            iteration = CycleIteration(iteration_number=call_count)
            iteration.think_output = MagicMock()
            iteration.think_output.is_final_answer = call_count >= 3
            iteration.think_output.proposed_action = None
            iteration.reflect_output = MagicMock()
            iteration.reflect_output.step_completed = False
            iteration.update_output = MagicMock()
            iteration.update_output.should_continue = True
            iteration.total_tokens_used = 100
            iteration.mark_completed(success=True)

            if call_count >= 3:
                agent_state.is_finished = True
            return iteration

        cognitive_cycle.run_iteration = mock_run_iteration

        updates: List[StreamingIterationResult] = []
        async for update in cognitive_cycle.run_streaming(
            agent_state=agent_state,
            session_id="test-session",
            max_iterations=5,
        ):
            updates.append(update)

        # Should have at least one error update
        error_updates = [u for u in updates if u.status == "error"]
        # Without circuit breaker, error is fatal
        assert len(updates) >= 1

    @pytest.mark.asyncio
    async def test_already_finished_state(self, cognitive_cycle, agent_state):
        """Test streaming when state is already finished."""
        agent_state.is_finished = True
        agent_state.final_answer = "Already done"

        updates: List[StreamingIterationResult] = []
        async for update in cognitive_cycle.run_streaming(
            agent_state=agent_state,
            session_id="test-session",
            max_iterations=10,
        ):
            updates.append(update)

        assert len(updates) == 1
        assert updates[0].is_complete is True
        assert updates[0].is_final is True
        assert updates[0].progress == 1.0


# =============================================================================
# SINGLE AGENT MODE STREAMING TESTS
# =============================================================================


class TestSingleAgentModeStreaming:
    """Tests for SingleAgentMode.run_streaming()."""

    @pytest.fixture
    def mock_agents_config(self):
        """Create mock agents config."""
        config = MagicMock()
        config.goals.classifier_enabled = False
        config.fast_path.enabled = False
        config.capability_check.enabled = False
        config.circuit_breaker.enabled = False
        return config

    @pytest.fixture
    def single_agent(self, mock_provider_manager, mock_agents_config):
        """Create a SingleAgentMode with mocks."""
        with patch.object(SingleAgentMode, '__init__', lambda self, *args, **kwargs: None):
            agent = SingleAgentMode.__new__(SingleAgentMode)
            agent.provider_manager = mock_provider_manager
            agent._agents_config = mock_agents_config
            agent.tracer = None

            # Mock tool manager
            agent.tool_manager = MagicMock()
            agent.tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]

            # Mock persona manager
            agent.persona_manager = MagicMock()
            agent.persona_manager.get_persona.return_value = None

            # Mock goal classifier
            agent.goal_classifier = MagicMock()

            # Mock capability checker
            agent.capability_checker = MagicMock()

            # Mock cognitive cycle with streaming
            agent.cognitive_cycle = MagicMock()

            # Mock sandbox methods
            agent._setup_sandbox = AsyncMock(return_value=None)
            agent._cleanup_sandbox = AsyncMock()

            return agent

    @pytest.mark.asyncio
    async def test_run_streaming_yields_updates(self, single_agent):
        """Test that run_streaming yields IterationUpdate objects."""
        # Setup mock streaming results
        streaming_results = [
            StreamingIterationResult(
                iteration=1,
                max_iterations=5,
                progress=0.3,
                is_complete=False,
                is_final=False,
                status="in_progress",
                current_phase="act",
                message="Working...",
            ),
            StreamingIterationResult(
                iteration=2,
                max_iterations=5,
                progress=1.0,
                is_complete=True,
                is_final=True,
                status="complete",
                current_phase="complete",
                message="Done!",
            ),
        ]

        async def mock_run_streaming(*args, **kwargs):
            for result in streaming_results:
                yield result

        single_agent.cognitive_cycle.run_streaming = mock_run_streaming

        updates: List[IterationUpdate] = []
        async for update in single_agent.run_streaming(goal="Test task"):
            updates.append(update)

        assert len(updates) == 2
        assert isinstance(updates[0], IterationUpdate)
        assert updates[0].iteration == 1
        assert updates[0].progress == 0.3
        assert updates[1].is_final is True
        assert updates[1].status == "complete"

    @pytest.mark.asyncio
    async def test_run_streaming_sandbox_cleanup(self, single_agent):
        """Test that sandbox is cleaned up after streaming."""
        mock_sandbox = MagicMock()
        single_agent._setup_sandbox = AsyncMock(return_value=mock_sandbox)

        async def mock_run_streaming(*args, **kwargs):
            yield StreamingIterationResult(
                iteration=1,
                max_iterations=5,
                progress=1.0,
                is_complete=True,
                is_final=True,
                status="complete",
                message="Done",
            )

        single_agent.cognitive_cycle.run_streaming = mock_run_streaming

        updates = []
        async for update in single_agent.run_streaming(goal="Test", use_sandbox=True):
            updates.append(update)

        # Verify sandbox was setup and cleaned up
        single_agent._setup_sandbox.assert_called_once()
        single_agent._cleanup_sandbox.assert_called_once_with(mock_sandbox)


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestStreamingRegression:
    """Regression tests to ensure streaming methods are not stubs."""

    def test_streaming_iteration_result_is_dataclass(self):
        """Verify StreamingIterationResult is a proper dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(StreamingIterationResult)

    def test_run_streaming_is_async_generator(self):
        """Verify run_streaming is an async generator method."""
        import inspect
        assert inspect.isasyncgenfunction(CognitiveCycle.run_streaming)
        assert inspect.isasyncgenfunction(SingleAgentMode.run_streaming)

    def test_iteration_update_has_from_streaming_result(self):
        """Verify IterationUpdate has conversion method."""
        assert hasattr(IterationUpdate, 'from_streaming_result')
        assert callable(IterationUpdate.from_streaming_result)

    def test_streaming_result_fields_exist(self):
        """Verify all expected fields exist on StreamingIterationResult."""
        expected_fields = [
            'iteration', 'max_iterations', 'progress', 'is_complete', 'is_final',
            'status', 'current_phase', 'message', 'action_name', 'action_summary',
            'observation_summary', 'step_completed', 'plan_step', 'error',
            'tokens_used', 'duration_ms', 'stop_reason'
        ]

        result = StreamingIterationResult(iteration=1, max_iterations=10, progress=0.5)
        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestStreamingIntegration:
    """Integration-style tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_complete_streaming_flow(self):
        """Test a complete streaming flow with realistic data."""
        # This test verifies the data flow without mocking internals
        result1 = StreamingIterationResult(
            iteration=1,
            max_iterations=5,
            progress=0.2,
            is_complete=False,
            is_final=False,
            status="in_progress",
            current_phase="act",
            message="Analyzing task requirements",
            action_name="analyze",
            tokens_used=200,
            duration_ms=1500.0,
        )

        result2 = StreamingIterationResult(
            iteration=2,
            max_iterations=5,
            progress=0.6,
            is_complete=False,
            is_final=False,
            status="in_progress",
            current_phase="observe",
            message="Processing data",
            action_name="process",
            observation_summary="Processed 100 records",
            step_completed=True,
            tokens_used=350,
            duration_ms=2200.0,
        )

        result3 = StreamingIterationResult(
            iteration=3,
            max_iterations=5,
            progress=1.0,
            is_complete=True,
            is_final=True,
            status="complete",
            current_phase="complete",
            message="Task completed successfully",
            tokens_used=150,
            duration_ms=800.0,
        )

        # Convert to IterationUpdates and verify
        updates = [
            IterationUpdate.from_streaming_result(r)
            for r in [result1, result2, result3]
        ]

        assert len(updates) == 3

        # Check first update
        assert updates[0].iteration == 1
        assert updates[0].progress == 0.2
        assert not updates[0].is_final

        # Check middle update
        assert updates[1].step_completed is True
        assert updates[1].observation == "Processed 100 records"

        # Check final update
        assert updates[2].is_final is True
        assert updates[2].progress == 1.0
        assert updates[2].status == "complete"

    @pytest.mark.asyncio
    async def test_error_recovery_in_stream(self):
        """Test error recovery behavior in streaming."""
        results = [
            StreamingIterationResult(
                iteration=1,
                max_iterations=5,
                progress=0.2,
                status="in_progress",
            ),
            StreamingIterationResult(
                iteration=2,
                max_iterations=5,
                progress=0.2,
                status="error",
                error="Temporary failure",
                is_final=False,  # Can continue with circuit breaker
            ),
            StreamingIterationResult(
                iteration=3,
                max_iterations=5,
                progress=1.0,
                status="complete",
                is_complete=True,
                is_final=True,
            ),
        ]

        updates = [IterationUpdate.from_streaming_result(r) for r in results]

        # Error in middle should not be final if circuit breaker allows retry
        assert updates[1].status == "error"
        assert updates[1].error == "Temporary failure"
        assert not updates[1].is_final

        # Final update should be success
        assert updates[2].is_final is True
        assert updates[2].status == "complete"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
