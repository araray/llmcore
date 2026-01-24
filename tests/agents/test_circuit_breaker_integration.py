# tests/agents/test_circuit_breaker_integration.py
"""
G3 Phase 5: Circuit Breaker Integration Tests.

Tests that the circuit breaker is properly integrated into
CognitiveCycle.run_until_complete() and stops execution on:
- Repeated identical errors
- Progress stalls
- Time/cost limits
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add source to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in CognitiveCycle."""

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        return MagicMock()

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        return MagicMock()

    @pytest.fixture
    def mock_storage_manager(self):
        """Create a mock storage manager."""
        return MagicMock()

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager."""
        manager = MagicMock()
        manager.get_tool_definitions.return_value = []
        return manager

    @pytest.fixture
    def cognitive_cycle(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
        mock_tool_manager,
    ):
        """Create a cognitive cycle instance."""
        from llmcore.agents.cognitive.phases.cycle import CognitiveCycle

        return CognitiveCycle(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            tool_manager=mock_tool_manager,
        )

    @pytest.fixture
    def agents_config(self):
        """Create an agents config with circuit breaker enabled."""
        from llmcore.config.agents_config import AgentsConfig
        config = AgentsConfig()
        config.circuit_breaker.enabled = True
        config.circuit_breaker.max_same_errors = 3
        config.circuit_breaker.max_iterations = 15
        config.circuit_breaker.progress_stall_threshold = 5
        return config

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_repeated_errors(
        self, cognitive_cycle, agents_config
    ):
        """Test that circuit breaker trips after N identical errors."""
        from llmcore.agents.cognitive.models import EnhancedAgentState

        agent_state = EnhancedAgentState(
            goal="Test goal",
            session_id="test-session",
        )

        # Mock run_iteration to always raise the same error
        with patch.object(
            cognitive_cycle,
            'run_iteration',
            side_effect=Exception("Model does not support tools"),
        ):
            result = await cognitive_cycle.run_until_complete(
                agent_state=agent_state,
                session_id="test-session",
                max_iterations=10,
                agents_config=agents_config,
            )

        # Should have tripped on repeated error
        assert "circuit breaker" in result.lower()
        assert "repeated_error" in result.lower()
        # Should not have completed all 10 iterations
        assert agent_state.iteration_count <= 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_disabled(self, cognitive_cycle):
        """Test that circuit breaker can be disabled."""
        from llmcore.config.agents_config import AgentsConfig
        from llmcore.agents.cognitive.models import EnhancedAgentState

        # Disable circuit breaker
        config = AgentsConfig()
        config.circuit_breaker.enabled = False

        agent_state = EnhancedAgentState(
            goal="Test goal",
            session_id="test-session",
        )

        # Mock run_iteration to always raise the same error
        with patch.object(
            cognitive_cycle,
            'run_iteration',
            side_effect=Exception("Model does not support tools"),
        ):
            result = await cognitive_cycle.run_until_complete(
                agent_state=agent_state,
                session_id="test-session",
                max_iterations=5,
                agents_config=config,
            )

        # Without circuit breaker, should fail on first error
        assert "Task failed" in result

    @pytest.mark.asyncio
    async def test_circuit_breaker_max_iterations(
        self, cognitive_cycle, agents_config
    ):
        """Test that circuit breaker respects max_iterations limit."""
        from llmcore.agents.cognitive.models import EnhancedAgentState, CycleIteration

        agent_state = EnhancedAgentState(
            goal="Test goal",
            session_id="test-session",
        )

        # Set low max iterations
        agents_config.circuit_breaker.max_iterations = 3

        # Create a mock iteration that doesn't complete
        mock_iteration = MagicMock(spec=CycleIteration)
        mock_iteration.think_output = None
        mock_iteration.update_output = MagicMock()
        mock_iteration.update_output.should_continue = True
        mock_iteration.total_cost = 0.0

        with patch.object(
            cognitive_cycle,
            'run_iteration',
            new_callable=AsyncMock,
            return_value=mock_iteration,
        ):
            result = await cognitive_cycle.run_until_complete(
                agent_state=agent_state,
                session_id="test-session",
                max_iterations=10,  # Higher than circuit breaker limit
                agents_config=agents_config,
            )

        # Should have stopped at circuit breaker limit (3), not max_iterations (10)
        assert "circuit breaker" in result.lower() or "incomplete" in result.lower()


class TestCircuitBreakerUnit:
    """Unit tests for AgentCircuitBreaker."""

    def test_trip_on_max_iterations(self):
        """Test that breaker trips on max iterations."""
        from llmcore.agents.resilience.circuit_breaker import (
            AgentCircuitBreaker,
            TripReason,
        )

        breaker = AgentCircuitBreaker(max_iterations=5)
        breaker.start()

        for i in range(10):
            result = breaker.check(iteration=i, progress=0.1 * i)
            if result.tripped:
                assert result.reason == TripReason.MAX_ITERATIONS
                assert i == 5
                break
        else:
            pytest.fail("Breaker should have tripped")

    def test_trip_on_repeated_error(self):
        """Test that breaker trips on repeated errors."""
        from llmcore.agents.resilience.circuit_breaker import (
            AgentCircuitBreaker,
            TripReason,
        )

        breaker = AgentCircuitBreaker(max_same_errors=3, max_iterations=100)
        breaker.start()

        for i in range(10):
            result = breaker.check(
                iteration=i,
                progress=0.0,
                error="Same error message" if i < 5 else None,
            )
            if result.tripped:
                assert result.reason == TripReason.REPEATED_ERROR
                break
        else:
            pytest.fail("Breaker should have tripped on repeated error")

    def test_trip_on_progress_stall(self):
        """Test that breaker trips on progress stall."""
        from llmcore.agents.resilience.circuit_breaker import (
            AgentCircuitBreaker,
            TripReason,
        )

        breaker = AgentCircuitBreaker(
            progress_stall_threshold=3,
            max_iterations=100,
        )
        breaker.start()

        for i in range(10):
            # Progress stuck at 0.5
            result = breaker.check(iteration=i, progress=0.5)
            if result.tripped:
                assert result.reason == TripReason.PROGRESS_STALL
                break
        else:
            pytest.fail("Breaker should have tripped on progress stall")

    def test_trip_on_cost_limit(self):
        """Test that breaker trips on cost limit."""
        from llmcore.agents.resilience.circuit_breaker import (
            AgentCircuitBreaker,
            TripReason,
        )

        breaker = AgentCircuitBreaker(max_total_cost=0.05, max_iterations=100)
        breaker.start()

        for i in range(10):
            result = breaker.check(iteration=i, progress=0.1 * i, cost=0.02)
            if result.tripped:
                assert result.reason == TripReason.COST_LIMIT
                break
        else:
            pytest.fail("Breaker should have tripped on cost limit")

    def test_no_trip_on_normal_progress(self):
        """Test that breaker doesn't trip with normal progress."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(max_iterations=10)
        breaker.start()

        for i in range(5):
            result = breaker.check(iteration=i, progress=0.2 * i)
            assert not result.tripped, f"Unexpected trip at iteration {i}"

    def test_factory_presets(self):
        """Test factory preset configurations."""
        from llmcore.agents.resilience.circuit_breaker import create_circuit_breaker

        strict = create_circuit_breaker("strict")
        assert strict.config.max_iterations == 5
        assert strict.config.max_same_errors == 2

        permissive = create_circuit_breaker("permissive")
        assert permissive.config.max_iterations == 50
        assert permissive.config.max_same_errors == 5

        custom = create_circuit_breaker("default", max_iterations=25)
        assert custom.config.max_iterations == 25

    def test_breaker_status(self):
        """Test breaker status reporting."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(max_iterations=10, max_total_cost=1.0)
        breaker.start()

        # Check some iterations
        breaker.check(iteration=0, progress=0.1, cost=0.1)
        breaker.check(iteration=1, progress=0.2, cost=0.1)

        status = breaker.get_status()
        assert status["state"] == "open"
        assert status["tripped"] is False
        assert status["total_cost"] == 0.2
        assert status["error_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
