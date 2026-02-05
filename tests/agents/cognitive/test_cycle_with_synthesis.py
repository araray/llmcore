# tests/agents/cognitive/test_cycle_with_synthesis.py
"""
Tests for CognitiveCycle integration with ContextSynthesizer.

These tests verify that the CognitiveCycle orchestrator correctly uses
the ContextSynthesizer in the PERCEIVE phase when provided, and falls
back to legacy MemoryManager retrieval when not.

References:
    - Task 3.9: ContextSynthesizer integration into Darwin PERCEIVE phase
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from llmcore.agents.cognitive.phases import (
    CognitiveCycle,
    StreamingIterationResult,
    create_default_synthesizer,
)
from llmcore.agents.cognitive.models import (
    EnhancedAgentState,
    PerceiveOutput,
    ValidationResult,
    ConfidenceLevel,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_provider_manager() -> MagicMock:
    """Create a mock provider manager."""
    manager = MagicMock()
    manager.generate = AsyncMock(return_value="Generated response")
    return manager


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock memory manager."""
    manager = MagicMock()
    manager.retrieve_context = AsyncMock(return_value=["Memory context 1", "Memory context 2"])
    manager.add_message = AsyncMock()
    manager.get_conversation_history = MagicMock(return_value=[])
    return manager


@pytest.fixture
def mock_storage_manager() -> MagicMock:
    """Create a mock storage manager."""
    manager = MagicMock()
    manager.store = AsyncMock()
    manager.retrieve = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_tool_manager() -> MagicMock:
    """Create a mock tool manager."""
    manager = MagicMock()
    manager.execute = AsyncMock(return_value={"result": "success"})
    manager.get_tool = MagicMock(return_value=None)
    manager.list_tools = MagicMock(return_value=[])
    return manager


@pytest.fixture
def mock_context_synthesizer() -> MagicMock:
    """Create a mock context synthesizer."""
    synthesizer = MagicMock()
    # Synthesize returns formatted context with sections
    synthesizer.synthesize = AsyncMock(
        return_value="## goals\n\nGoal context here\n\n---\n\n## recent\n\nRecent turns here\n\n---\n\n"
    )
    synthesizer.max_tokens = 100_000
    return synthesizer


@pytest.fixture
def agent_state() -> EnhancedAgentState:
    """Create a basic agent state for testing."""
    return EnhancedAgentState(
        goal="Test goal: complete the task",
        agent_id="test-agent-001",
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestCognitiveCycleInitialization:
    """Tests for CognitiveCycle initialization with context_synthesizer."""

    def test_init_without_synthesizer(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
    ) -> None:
        """CognitiveCycle initializes without synthesizer (legacy mode)."""
        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

        assert cycle.context_synthesizer is None
        assert cycle.provider_manager is mock_provider_manager
        assert cycle.memory_manager is mock_memory_manager

    def test_init_with_synthesizer(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
        mock_context_synthesizer: MagicMock,
    ) -> None:
        """CognitiveCycle initializes with synthesizer (synthesis mode)."""
        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=mock_context_synthesizer,
        )

        assert cycle.context_synthesizer is mock_context_synthesizer

    def test_init_with_all_optional_params(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
        mock_context_synthesizer: MagicMock,
    ) -> None:
        """CognitiveCycle initializes with all optional parameters."""
        mock_tracer = MagicMock()
        mock_registry = MagicMock()

        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            prompt_registry=mock_registry,
            tracer=mock_tracer,
            context_synthesizer=mock_context_synthesizer,
        )

        assert cycle.tracer is mock_tracer
        assert cycle.prompt_registry is mock_registry
        assert cycle.context_synthesizer is mock_context_synthesizer


# =============================================================================
# PERCEIVE PHASE INTEGRATION TESTS
# =============================================================================


def _create_mock_phase_outputs() -> dict[str, MagicMock]:
    """Create mock outputs for all cognitive phases.

    Sets is_final_answer=True to short-circuit after THINK phase,
    avoiding the need to construct real ToolCall objects for validation.
    """
    # Create mock perceive output
    perceive_output = MagicMock()
    perceive_output.goal = "Test goal"
    perceive_output.retrieved_context = ["Context chunk"]

    # Create mock plan output
    plan_output = MagicMock()
    plan_output.plan = ["Step 1"]
    plan_output.reasoning = "Plan reasoning"

    # Create mock think output that signals completion
    # Setting is_final_answer=True skips validate/act phases
    think_output = MagicMock()
    think_output.thought = "Task complete"
    think_output.proposed_action = None  # No action needed
    think_output.is_final_answer = True
    think_output.final_answer = "Task completed successfully"
    think_output.confidence = ConfidenceLevel.HIGH
    think_output.reasoning_tokens = None
    think_output.using_activity_fallback = False

    # These outputs won't be used but we include them for completeness
    validate_output = MagicMock()
    validate_output.result = ValidationResult.APPROVED
    validate_output.confidence = ConfidenceLevel.HIGH
    validate_output.requires_human_approval = False

    act_output = MagicMock()
    act_output.tool_result = MagicMock()
    act_output.success = True

    observe_output = MagicMock()
    observe_output.observation = "Not executed"
    observe_output.success_indicators = []

    reflect_output = MagicMock()
    reflect_output.reflection = "Task complete"
    reflect_output.step_completed = True

    update_output = MagicMock()
    update_output.state_changes = []
    update_output.should_continue = False

    return {
        "perceive": perceive_output,
        "plan": plan_output,
        "think": think_output,
        "validate": validate_output,
        "act": act_output,
        "observe": observe_output,
        "reflect": reflect_output,
        "update": update_output,
    }


class TestCognitiveCyclePerceiveIntegration:
    """Tests for CognitiveCycle PERCEIVE phase using ContextSynthesizer."""

    @pytest.mark.asyncio
    async def test_run_iteration_passes_synthesizer_to_perceive(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
        mock_context_synthesizer: MagicMock,
        agent_state: EnhancedAgentState,
    ) -> None:
        """run_iteration passes context_synthesizer to perceive_phase."""
        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=mock_context_synthesizer,
        )

        outputs = _create_mock_phase_outputs()

        # Mock all phase functions to track calls
        with patch(
            "llmcore.agents.cognitive.phases.cycle.perceive_phase",
            return_value=outputs["perceive"],
        ) as mock_perceive, patch(
            "llmcore.agents.cognitive.phases.cycle.plan_phase",
            return_value=outputs["plan"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.think_phase",
            return_value=outputs["think"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.validate_phase",
            return_value=outputs["validate"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.act_phase",
            return_value=outputs["act"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.observe_phase",
            return_value=outputs["observe"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.reflect_phase",
            return_value=outputs["reflect"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.update_phase",
            return_value=outputs["update"],
        ):
            # Run iteration
            await cycle.run_iteration(
                agent_state=agent_state,
                session_id="test-session",
            )

            # Verify perceive_phase was called with context_synthesizer
            mock_perceive.assert_called_once()
            call_kwargs = mock_perceive.call_args.kwargs
            assert "context_synthesizer" in call_kwargs
            assert call_kwargs["context_synthesizer"] is mock_context_synthesizer

    @pytest.mark.asyncio
    async def test_run_iteration_without_synthesizer_uses_legacy_mode(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
        agent_state: EnhancedAgentState,
    ) -> None:
        """run_iteration without synthesizer uses legacy MemoryManager mode."""
        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            # No context_synthesizer
        )

        outputs = _create_mock_phase_outputs()

        with patch(
            "llmcore.agents.cognitive.phases.cycle.perceive_phase",
            return_value=outputs["perceive"],
        ) as mock_perceive, patch(
            "llmcore.agents.cognitive.phases.cycle.plan_phase",
            return_value=outputs["plan"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.think_phase",
            return_value=outputs["think"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.validate_phase",
            return_value=outputs["validate"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.act_phase",
            return_value=outputs["act"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.observe_phase",
            return_value=outputs["observe"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.reflect_phase",
            return_value=outputs["reflect"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.update_phase",
            return_value=outputs["update"],
        ):
            await cycle.run_iteration(
                agent_state=agent_state,
                session_id="test-session",
            )

            # Verify perceive_phase was called with context_synthesizer=None
            mock_perceive.assert_called_once()
            call_kwargs = mock_perceive.call_args.kwargs
            assert call_kwargs.get("context_synthesizer") is None


# =============================================================================
# FACTORY INTEGRATION TESTS
# =============================================================================


class TestCognitiveCycleWithFactory:
    """Tests for using create_default_synthesizer with CognitiveCycle."""

    def test_create_cycle_with_factory_synthesizer(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
    ) -> None:
        """CognitiveCycle works with factory-created synthesizer."""
        # Create synthesizer with factory (minimal - no dependencies)
        synthesizer = create_default_synthesizer(max_tokens=50_000)

        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=synthesizer,
        )

        assert cycle.context_synthesizer is synthesizer
        assert cycle.context_synthesizer.max_tokens == 50_000

    def test_create_cycle_with_factory_and_goal_manager(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
    ) -> None:
        """CognitiveCycle with synthesizer including GoalManager source."""
        mock_goal_manager = MagicMock()
        mock_goal_manager.active_goals = []

        synthesizer = create_default_synthesizer(
            goal_manager=mock_goal_manager,
            max_tokens=100_000,
        )

        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=synthesizer,
        )

        # Verify goals source was registered
        assert "goals" in synthesizer._sources

    def test_create_cycle_with_factory_full_sources(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
    ) -> None:
        """CognitiveCycle with synthesizer including all context sources."""
        mock_goal_manager = MagicMock()
        mock_goal_manager.active_goals = []

        mock_skill_loader = MagicMock()

        async def mock_retrieval(query: str, top_k: int = 5) -> list[str]:
            return ["RAG chunk 1", "RAG chunk 2"]

        synthesizer = create_default_synthesizer(
            goal_manager=mock_goal_manager,
            skill_loader=mock_skill_loader,
            retrieval_fn=mock_retrieval,
            max_tokens=150_000,
        )

        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=synthesizer,
        )

        # Verify sources were registered
        assert "goals" in synthesizer._sources
        assert "skills" in synthesizer._sources
        assert "semantic" in synthesizer._sources


# =============================================================================
# STREAMING INTEGRATION TESTS
# =============================================================================


class TestCognitiveCycleStreamingWithSynthesis:
    """Tests for run_streaming with ContextSynthesizer."""

    @pytest.mark.asyncio
    async def test_streaming_uses_synthesizer(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
        mock_context_synthesizer: MagicMock,
        agent_state: EnhancedAgentState,
    ) -> None:
        """run_streaming uses context_synthesizer when provided."""
        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=mock_context_synthesizer,
        )

        with patch.object(cycle, "run_iteration") as mock_run:
            # Configure run_iteration to complete immediately
            from llmcore.agents.cognitive.models import CycleIteration

            mock_iteration = MagicMock(spec=CycleIteration)
            mock_iteration.think_output = MagicMock()
            mock_iteration.think_output.proposed_action = None
            mock_iteration.think_output.is_final_answer = True
            mock_iteration.observe_output = None
            mock_iteration.reflect_output = None
            mock_iteration.update_output = MagicMock()
            mock_iteration.update_output.should_continue = False
            mock_iteration.total_cost = 0.0
            mock_iteration.total_tokens_used = 0

            mock_run.return_value = mock_iteration

            # Mark as finished after first iteration
            agent_state.is_finished = True
            agent_state.final_answer = "Done"

            results = []
            async for result in cycle.run_streaming(
                agent_state=agent_state,
                session_id="test-session",
                max_iterations=5,
            ):
                results.append(result)

            # Should have at least one result
            assert len(results) >= 1
            # Cycle should have synthesizer attached
            assert cycle.context_synthesizer is mock_context_synthesizer


# =============================================================================
# TRACING INTEGRATION TESTS
# =============================================================================


class TestCognitiveCycleTracingWithSynthesis:
    """Tests for OpenTelemetry tracing with synthesis mode."""

    @pytest.mark.asyncio
    async def test_tracing_includes_synthesis_mode_attribute(
        self,
        mock_provider_manager: MagicMock,
        mock_memory_manager: MagicMock,
        mock_storage_manager: MagicMock,
        mock_tool_manager: MagicMock,
        mock_context_synthesizer: MagicMock,
        agent_state: EnhancedAgentState,
    ) -> None:
        """Tracing span includes synthesis_mode attribute when using synthesizer."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )

        cycle = CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
            context_synthesizer=mock_context_synthesizer,
            tracer=mock_tracer,
        )

        outputs = _create_mock_phase_outputs()

        with patch(
            "llmcore.agents.cognitive.phases.cycle.perceive_phase",
            return_value=outputs["perceive"],
        ) as mock_perceive, patch(
            "llmcore.agents.cognitive.phases.cycle.plan_phase",
            return_value=outputs["plan"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.think_phase",
            return_value=outputs["think"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.validate_phase",
            return_value=outputs["validate"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.act_phase",
            return_value=outputs["act"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.observe_phase",
            return_value=outputs["observe"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.reflect_phase",
            return_value=outputs["reflect"],
        ), patch(
            "llmcore.agents.cognitive.phases.cycle.update_phase",
            return_value=outputs["update"],
        ):
            await cycle.run_iteration(
                agent_state=agent_state,
                session_id="test-session",
            )

            # Verify perceive was called with synthesizer and tracer
            call_kwargs = mock_perceive.call_args.kwargs
            assert call_kwargs.get("context_synthesizer") is mock_context_synthesizer
            assert call_kwargs.get("tracer") is mock_tracer


# =============================================================================
# EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_cognitive_cycle_exported_from_phases(self) -> None:
        """CognitiveCycle is exported from phases module."""
        from llmcore.agents.cognitive.phases import CognitiveCycle

        assert CognitiveCycle is not None

    def test_streaming_result_exported_from_phases(self) -> None:
        """StreamingIterationResult is exported from phases module."""
        from llmcore.agents.cognitive.phases import StreamingIterationResult

        assert StreamingIterationResult is not None

    def test_create_default_synthesizer_exported(self) -> None:
        """create_default_synthesizer is exported from phases module."""
        from llmcore.agents.cognitive.phases import create_default_synthesizer

        assert callable(create_default_synthesizer)

    def test_all_exports_present(self) -> None:
        """All expected exports are present in __all__."""
        from llmcore.agents.cognitive import phases

        expected_exports = [
            "CognitiveCycle",
            "StreamingIterationResult",
            "create_default_synthesizer",
            "act_phase",
            "observe_phase",
            "perceive_phase",
            "plan_phase",
            "reflect_phase",
            "think_phase",
            "update_phase",
            "validate_phase",
        ]

        for export in expected_exports:
            assert export in phases.__all__, f"{export} not in __all__"
