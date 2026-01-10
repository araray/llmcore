# tests/agents/test_memory_and_manager.py
"""
Unit tests for Memory Integration and Enhanced Agent Manager.

Tests cover:
- CognitiveMemoryIntegrator (recording, retrieval, consolidation)
- AgentManager (original functionality preserved)
- EnhancedAgentManager (inheritance, modes, routing, integration)
- Backward compatibility
- Complete system integration

References:
    - Dossier: Steps 2.10-2.11 (Memory Integration & Enhanced AgentManager)
    - Integration Audit: INTEGRATION_AUDIT.md
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llmcore.agents import AgentResult
from llmcore.agents.cognitive import (
    CycleIteration,
    EnhancedAgentState,
    IterationStatus,
    ReflectOutput,
)
from llmcore.agents.manager import AgentManager, AgentMode, EnhancedAgentManager
from llmcore.agents.memory import CognitiveMemoryIntegrator

# =============================================================================
# MEMORY INTEGRATOR TESTS
# =============================================================================


class TestCognitiveMemoryIntegrator:
    """Tests for CognitiveMemoryIntegrator."""

    @pytest.fixture
    def mock_managers(self):
        """Create mock memory and storage managers."""
        memory_manager = Mock()
        storage_manager = Mock()

        memory_manager.retrieve_relevant_context = AsyncMock(return_value=[])
        memory_manager.store_memory = AsyncMock(return_value=None)
        storage_manager.store_episode = AsyncMock(return_value=None)
        storage_manager.get_episodes = AsyncMock(return_value=[])

        return memory_manager, storage_manager

    def test_integrator_initialization(self, mock_managers):
        """Test CognitiveMemoryIntegrator initialization."""
        memory_manager, storage_manager = mock_managers

        integrator = CognitiveMemoryIntegrator(
            memory_manager=memory_manager, storage_manager=storage_manager
        )

        assert integrator.memory_manager == memory_manager
        assert integrator.storage_manager == storage_manager

    @pytest.mark.asyncio
    async def test_record_iteration(self, mock_managers):
        """Test recording an iteration."""
        memory_manager, storage_manager = mock_managers

        integrator = CognitiveMemoryIntegrator(
            memory_manager=memory_manager, storage_manager=storage_manager
        )

        # Create test iteration
        state = EnhancedAgentState(goal="Test goal", session_id="session-123")
        iteration = state.start_iteration(1)
        iteration.reflect_output = ReflectOutput(
            evaluation="Success",
            progress_estimate=0.5,
            insights=["Learned something", "Discovered pattern"],
            plan_needs_update=False,
            step_completed=True,
        )
        iteration.mark_completed(success=True)

        # Record iteration
        await integrator.record_iteration(
            iteration=iteration, agent_state=state, session_id="session-123", extract_learnings=True
        )

        # Verify episode was stored
        assert storage_manager.store_episode.called

        # Verify learnings were extracted and stored
        assert memory_manager.store_memory.call_count == 2  # 2 insights

    @pytest.mark.asyncio
    async def test_retrieve_context(self, mock_managers):
        """Test retrieving context."""
        memory_manager, storage_manager = mock_managers

        # Setup mock returns
        semantic_item = Mock()
        semantic_item.content = "Semantic context item"
        memory_manager.retrieve_relevant_context = AsyncMock(return_value=[semantic_item])

        episode = Mock()
        episode.data = {"content": "Episodic context item"}
        storage_manager.get_episodes = AsyncMock(return_value=[episode])

        integrator = CognitiveMemoryIntegrator(
            memory_manager=memory_manager, storage_manager=storage_manager
        )

        # Retrieve context
        context = await integrator.retrieve_context(
            goal="Test goal", current_step="Test step", session_id="session-123", max_items=5
        )

        # Verify
        assert len(context) > 0
        assert "Semantic context item" in context
        assert "Episodic context item" in context

    @pytest.mark.asyncio
    async def test_consolidate_session_memory(self, mock_managers):
        """Test consolidating session memory."""
        memory_manager, storage_manager = mock_managers

        integrator = CognitiveMemoryIntegrator(
            memory_manager=memory_manager, storage_manager=storage_manager
        )

        # Create state with multiple iterations containing insights
        state = EnhancedAgentState(goal="Test goal", session_id="session-123")

        for i in range(3):
            iteration = state.start_iteration(i + 1)
            iteration.reflect_output = ReflectOutput(
                evaluation=f"Iteration {i + 1}",
                progress_estimate=0.33 * (i + 1),
                insights=[f"Insight {i + 1}A", f"Insight {i + 1}B"],
                plan_needs_update=False,
                step_completed=True,
            )
            iteration.mark_completed(success=True)
            state.complete_iteration(success=True)

        state.is_finished = True

        # Consolidate
        await integrator.consolidate_session_memory(session_id="session-123", agent_state=state)

        # Verify consolidated memory was stored
        assert memory_manager.store_memory.called
        call_args = memory_manager.store_memory.call_args

        # Check content
        content = call_args[1]["content"]
        assert "Session Learning" in content
        assert "Test goal" in content

        # Check metadata
        metadata = call_args[1]["metadata"]
        assert metadata["type"] == "session_learning"
        assert metadata["session_id"] == "session-123"


# =============================================================================
# ORIGINAL AGENT MANAGER TESTS
# =============================================================================


class TestAgentManager:
    """Tests for original AgentManager (preserved functionality)."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for AgentManager."""
        provider_manager = Mock()
        memory_manager = Mock()
        storage_manager = Mock()

        return {
            "provider_manager": provider_manager,
            "memory_manager": memory_manager,
            "storage_manager": storage_manager,
        }

    def test_manager_initialization(self, mock_components):
        """Test AgentManager initialization."""
        manager = AgentManager(**mock_components)

        assert manager._provider_manager is not None
        assert manager._memory_manager is not None
        assert manager._storage_manager is not None
        assert manager._tool_manager is not None
        assert not manager.sandbox_enabled

    def test_manager_has_original_methods(self, mock_components):
        """Test that original methods are present."""
        manager = AgentManager(**mock_components)

        # Original methods should exist
        assert hasattr(manager, "run_agent_loop")
        assert hasattr(manager, "initialize_sandbox")
        assert hasattr(manager, "shutdown_sandbox")
        assert hasattr(manager, "get_available_tools")
        assert hasattr(manager, "get_tool_definitions")
        assert hasattr(manager, "cleanup")

    def test_sandbox_properties(self, mock_components):
        """Test sandbox property accessors."""
        manager = AgentManager(**mock_components)

        assert manager.sandbox_enabled is False
        assert manager.sandbox_integration is None


# =============================================================================
# ENHANCED AGENT MANAGER TESTS
# =============================================================================


class TestEnhancedAgentManager:
    """Tests for EnhancedAgentManager."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for EnhancedAgentManager.

        NOTE: EnhancedAgentManager does NOT accept tool_manager as a parameter.
        It creates its own ToolManager internally through the parent class.
        """
        provider_manager = Mock()
        memory_manager = Mock()
        storage_manager = Mock()

        # Setup async mocks
        memory_manager.retrieve_relevant_context = AsyncMock(return_value=[])
        memory_manager.store_memory = AsyncMock(return_value=None)
        storage_manager.store_episode = AsyncMock(return_value=None)
        storage_manager.get_episodes = AsyncMock(return_value=[])

        return {
            "provider_manager": provider_manager,
            "memory_manager": memory_manager,
            "storage_manager": storage_manager,
        }

    def test_manager_initialization(self, mock_components):
        """Test EnhancedAgentManager initialization."""
        manager = EnhancedAgentManager(**mock_components)

        assert manager._provider_manager is not None
        assert manager._memory_manager is not None
        assert manager._storage_manager is not None
        # ToolManager is created internally by parent class
        assert manager._tool_manager is not None
        assert manager.persona_manager is not None
        assert manager.memory_integrator is not None
        assert manager.single_agent is not None
        assert manager.default_mode == AgentMode.SINGLE

    def test_manager_with_custom_default_mode(self, mock_components):
        """Test initialization with custom default mode."""
        manager = EnhancedAgentManager(**mock_components, default_mode=AgentMode.LEGACY)

        assert manager.default_mode == AgentMode.LEGACY

    @pytest.mark.asyncio
    async def test_run_single_mode(self, mock_components):
        """Test running in single mode."""
        with patch("llmcore.agents.single_agent.SingleAgentMode") as MockSingleAgent:
            # Setup mock
            mock_single_instance = AsyncMock()
            mock_result = AgentResult(
                goal="Test",
                final_answer="Done",
                success=True,
                iteration_count=3,
                total_tokens=500,
                total_time_seconds=5.0,
                session_id="session-123",
            )
            mock_single_instance.run = AsyncMock(return_value=mock_result)
            MockSingleAgent.return_value = mock_single_instance

            manager = EnhancedAgentManager(**mock_components)
            manager.single_agent = mock_single_instance

            # Run
            result = await manager.run(goal="Test goal", mode=AgentMode.SINGLE, persona="analyst")

            # Verify
            assert result.success
            assert result.final_answer == "Done"
            assert mock_single_instance.run.called

    @pytest.mark.asyncio
    async def test_run_legacy_mode(self, mock_components):
        """Test running in legacy mode."""
        # The legacy mode runs through _run_cognitive_loop which expects
        # a proper async provider. We need to mock the internal run method.
        manager = EnhancedAgentManager(**mock_components)

        # Mock the internal _run_cognitive_loop to avoid provider issues
        mock_result = AgentResult(
            goal="Test goal",
            final_answer="Legacy result",
            success=True,
            iteration_count=1,
            total_tokens=100,
            total_time_seconds=1.0,
            session_id="session-123",
        )

        # Patch _run_cognitive_loop directly since legacy mode uses it
        with patch.object(manager, "_run_cognitive_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = "Legacy result"

            # Run in legacy mode
            result = await manager.run(goal="Test goal", mode=AgentMode.LEGACY)

            # Verify
            assert result.final_answer == "Legacy result"
            assert mock_loop.called

    @pytest.mark.asyncio
    async def test_run_uses_default_mode(self, mock_components):
        """Test that run() uses default mode when not specified."""
        # Setup mock single agent instance
        mock_single_instance = AsyncMock()
        mock_result = AgentResult(
            goal="Test",
            final_answer="Done",
            success=True,
            iteration_count=1,
            total_tokens=100,
            total_time_seconds=1.0,
            session_id="session-123",
        )
        mock_single_instance.run = AsyncMock(return_value=mock_result)

        manager = EnhancedAgentManager(**mock_components, default_mode=AgentMode.SINGLE)
        # Directly set the single_agent (same pattern as test_run_single_mode)
        manager.single_agent = mock_single_instance

        # Run without specifying mode
        result = await manager.run(goal="Test")

        # Should use default (SINGLE)
        assert mock_single_instance.run.called

    def test_create_persona(self, mock_components):
        """Test creating a persona through manager."""
        manager = EnhancedAgentManager(**mock_components)

        persona = manager.create_persona(
            name="Test Persona", description="A test persona", traits=[], risk_tolerance=None
        )

        assert persona.name == "Test Persona"
        assert persona.description == "A test persona"

    def test_list_personas(self, mock_components):
        """Test listing personas."""
        manager = EnhancedAgentManager(**mock_components)

        personas = manager.list_personas()

        # Should have at least built-in personas
        assert len(personas) >= 5

    def test_get_persona(self, mock_components):
        """Test getting a specific persona."""
        manager = EnhancedAgentManager(**mock_components)

        analyst = manager.get_persona("analyst")

        assert analyst is not None
        assert analyst.id == "analyst"
        assert analyst.name == "Data Analyst"

    @pytest.mark.asyncio
    async def test_consolidate_memory(self, mock_components):
        """Test memory consolidation."""
        manager = EnhancedAgentManager(**mock_components)

        state = EnhancedAgentState(goal="Test", session_id="session-123")
        iteration = state.start_iteration(1)
        iteration.reflect_output = ReflectOutput(
            evaluation="Success",
            progress_estimate=1.0,
            insights=["Insight 1"],
            plan_needs_update=False,
            step_completed=True,
        )
        state.complete_iteration(success=True)

        # Should not raise
        await manager.consolidate_memory(session_id="session-123", agent_state=state)

    def test_set_default_mode(self, mock_components):
        """Test setting default mode."""
        manager = EnhancedAgentManager(**mock_components)

        assert manager.default_mode == AgentMode.SINGLE

        manager.set_default_mode(AgentMode.LEGACY)

        assert manager.default_mode == AgentMode.LEGACY

    def test_get_capabilities(self, mock_components):
        """Test getting capabilities."""
        manager = EnhancedAgentManager(**mock_components)

        capabilities = manager.get_capabilities()

        assert "modes" in capabilities
        assert "personas" in capabilities
        assert "cognitive_phases" in capabilities
        assert capabilities["cognitive_phases"] == 8
        assert capabilities["memory_integration"] is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSystemIntegration:
    """Integration tests for complete system."""

    @pytest.fixture
    def mock_components(self):
        """Create comprehensive mock components."""
        provider_manager = Mock()
        memory_manager = Mock()
        storage_manager = Mock()
        # tool_manager = Mock()

        # Setup comprehensive async mocks
        memory_manager.retrieve_relevant_context = AsyncMock(return_value=[])
        memory_manager.store_memory = AsyncMock(return_value=None)
        storage_manager.store_episode = AsyncMock(return_value=None)
        storage_manager.get_episodes = AsyncMock(return_value=[])

        return {
            "provider_manager": provider_manager,
            "memory_manager": memory_manager,
            "storage_manager": storage_manager,
            # "tool_manager": tool_manager,  # Removed - not accepted by EnhancedAgentManager
        }

    def test_complete_initialization(self, mock_components):
        """Test complete system initialization."""
        manager = EnhancedAgentManager(**mock_components)

        # All components should be initialized
        assert manager.persona_manager is not None
        assert manager.memory_integrator is not None
        assert manager.single_agent is not None

        # Built-in personas should be loaded
        personas = manager.list_personas()
        assert len(personas) >= 5

        # Capabilities should be complete
        capabilities = manager.get_capabilities()
        assert AgentMode.SINGLE.value in capabilities["modes"]
        assert AgentMode.LEGACY.value in capabilities["modes"]

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mock_components):
        """Test end-to-end workflow with mocks."""
        with patch("llmcore.agents.single_agent.SingleAgentMode") as MockSingleAgent:
            # Setup complete mock chain
            mock_single_instance = AsyncMock()

            # Create realistic result
            state = EnhancedAgentState(goal="Test", session_id="session-123")
            iteration = state.start_iteration(1)
            iteration.reflect_output = ReflectOutput(
                evaluation="Success",
                progress_estimate=1.0,
                insights=["Key insight"],
                plan_needs_update=False,
                step_completed=True,
            )
            state.complete_iteration(success=True)
            state.is_finished = True

            mock_result = AgentResult(
                goal="Test workflow",
                final_answer="Workflow complete",
                success=True,
                iteration_count=1,
                total_tokens=250,
                total_time_seconds=3.5,
                session_id="session-123",
                persona_used="Assistant",
                agent_state=state,
            )

            mock_single_instance.run = AsyncMock(return_value=mock_result)
            MockSingleAgent.return_value = mock_single_instance

            # Initialize manager
            manager = EnhancedAgentManager(**mock_components)
            manager.single_agent = mock_single_instance

            # Run agent
            result = await manager.run(
                goal="Test workflow", mode=AgentMode.SINGLE, persona="assistant"
            )

            # Verify result
            assert result.success
            assert result.iteration_count == 1
            assert result.persona_used == "Assistant"

            # Consolidate memory
            await manager.consolidate_memory(
                session_id="session-123", agent_state=result.agent_state
            )

            # Verify memory storage was called
            assert mock_components["memory_manager"].store_memory.called


# =============================================================================
# INHERITANCE & BACKWARD COMPATIBILITY TESTS
# =============================================================================


class TestInheritanceAndCompatibility:
    """Tests for proper inheritance and backward compatibility."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        provider_manager = Mock()
        memory_manager = Mock()
        storage_manager = Mock()

        memory_manager.retrieve_relevant_context = AsyncMock(return_value=[])
        memory_manager.store_memory = AsyncMock(return_value=None)
        storage_manager.store_episode = AsyncMock(return_value=None)
        storage_manager.get_episodes = AsyncMock(return_value=[])

        return {
            "provider_manager": provider_manager,
            "memory_manager": memory_manager,
            "storage_manager": storage_manager,
        }

    def test_enhanced_inherits_from_original(self, mock_components):
        """Test proper inheritance."""
        manager = EnhancedAgentManager(**mock_components)

        assert isinstance(manager, AgentManager)
        assert isinstance(manager, EnhancedAgentManager)

    def test_original_methods_available_in_enhanced(self, mock_components):
        """Test all original methods are accessible."""
        manager = EnhancedAgentManager(**mock_components)

        # Original public methods
        assert callable(manager.run_agent_loop)
        assert callable(manager.initialize_sandbox)
        assert callable(manager.shutdown_sandbox)
        assert callable(manager.cleanup)
        assert callable(manager.get_available_tools)
        assert callable(manager.get_tool_definitions)

    def test_new_methods_only_in_enhanced(self, mock_components):
        """Test new methods are only in EnhancedAgentManager."""
        original = AgentManager(**mock_components)
        enhanced = EnhancedAgentManager(**mock_components)

        # Original should NOT have new methods
        assert not hasattr(original, "run")
        assert not hasattr(original, "create_persona")
        assert not hasattr(original, "set_default_mode")

        # Enhanced SHOULD have new methods
        assert callable(enhanced.run)
        assert callable(enhanced.create_persona)
        assert callable(enhanced.set_default_mode)

    def test_both_managers_have_same_constructor_signature(self):
        """Test that both managers accept same basic arguments."""
        pm = Mock()
        mm = Mock()
        sm = Mock()

        # Both should accept same 3 required args
        original = AgentManager(pm, mm, sm)
        enhanced = EnhancedAgentManager(pm, mm, sm)

        assert original is not None
        assert enhanced is not None

    def test_enhanced_has_additional_attributes(self, mock_components):
        """Test enhanced manager has Layer 2 attributes."""
        manager = EnhancedAgentManager(**mock_components)

        assert hasattr(manager, "persona_manager")
        assert hasattr(manager, "memory_integrator")
        assert hasattr(manager, "single_agent")
        assert hasattr(manager, "default_mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
