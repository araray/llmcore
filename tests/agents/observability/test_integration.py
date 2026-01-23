# tests/agents/observability/test_integration.py
"""
Integration Tests for Observability in Agent Execution.

These tests verify that the observability module is properly integrated
into the agent execution flow, ensuring events are logged and metrics
are collected during agent operations.

Phase 8 - Integration & Polish
References:
    - Continuation Guide: Section 7 (Testing Requirements)
    - Master Plan: Section 29.9 (Phase 8 Integration)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# IMPORTS
# =============================================================================

# Check if AgentManager is available (conftest creates dummy packages that don't include it)
try:
    from llmcore.agents import AgentManager
    AGENT_MANAGER_AVAILABLE = True
except ImportError:
    AGENT_MANAGER_AVAILABLE = False

# Check if EnhancedAgentManager is available
try:
    from llmcore.agents import EnhancedAgentManager
    ENHANCED_AGENT_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_AGENT_MANAGER_AVAILABLE = False

# Skip marker for tests that require AgentManager
requires_agent_manager = pytest.mark.skipif(
    not AGENT_MANAGER_AVAILABLE,
    reason="AgentManager not available in isolated test environment"
)

requires_enhanced_agent_manager = pytest.mark.skipif(
    not ENHANCED_AGENT_MANAGER_AVAILABLE,
    reason="EnhancedAgentManager not available in isolated test environment"
)


class TestObservabilityFactoryUnit:
    """Unit tests for observability factory functions."""

    def test_observability_components_dataclass(self):
        """Test ObservabilityComponents dataclass defaults."""
        from llmcore.agents.observability_factory import ObservabilityComponents

        # Default should be disabled
        components = ObservabilityComponents()
        assert components.enabled is False
        assert components.logger is None
        assert components.metrics is None
        assert components.config == {}
        assert bool(components) is False

    def test_observability_components_enabled(self):
        """Test ObservabilityComponents when enabled."""
        from llmcore.agents.observability_factory import ObservabilityComponents

        mock_logger = MagicMock()
        mock_metrics = MagicMock()

        components = ObservabilityComponents(
            enabled=True,
            logger=mock_logger,
            metrics=mock_metrics,
            config={"enabled": True},
        )

        assert components.enabled is True
        assert components.logger is mock_logger
        assert components.metrics is mock_metrics
        assert bool(components) is True

    @pytest.mark.asyncio
    async def test_observability_components_close(self):
        """Test ObservabilityComponents.close() method."""
        from llmcore.agents.observability_factory import ObservabilityComponents

        mock_logger = AsyncMock()
        mock_logger.close = AsyncMock()

        components = ObservabilityComponents(
            enabled=True,
            logger=mock_logger,
        )

        await components.close()
        mock_logger.close.assert_called_once()

    def test_create_observability_none_config(self):
        """Test creating observability with None config."""
        from llmcore.agents.observability_factory import create_observability_from_config

        # With None config, should use defaults
        obs = create_observability_from_config(None, session_id="test-session")

        # Should be enabled with defaults
        assert obs.enabled is True
        assert obs.logger is not None

    def test_create_observability_disabled(self):
        """Test creating observability with disabled setting."""
        from llmcore.agents.observability_factory import create_observability_from_config

        # Use override_enabled to disable
        obs = create_observability_from_config(
            None,
            session_id="test-session",
            override_enabled=False,
        )

        assert obs.enabled is False
        assert obs.logger is None
        assert obs.metrics is None

    def test_create_event_logger_simple_enabled(self):
        """Test create_event_logger_simple when enabled."""
        from llmcore.agents.observability_factory import create_event_logger_simple

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/test_events.jsonl"
            logger = create_event_logger_simple(
                session_id="test-123",
                log_path=log_path,
                enabled=True,
            )

            assert logger is not None
            assert logger.session_id == "test-123"

    def test_create_event_logger_simple_disabled(self):
        """Test create_event_logger_simple when disabled."""
        from llmcore.agents.observability_factory import create_event_logger_simple

        logger = create_event_logger_simple(
            session_id="test-123",
            enabled=False,
        )

        assert logger is None


class TestAgentManagerObservabilityIntegration:
    """Integration tests for AgentManager with observability."""

    @pytest.fixture
    def mock_provider_manager(self):
        """Create mock provider manager."""
        provider = MagicMock()
        provider.get_name.return_value = "mock_provider"
        provider.default_model = "test-model"
        provider.chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "I will use the finish tool.\n\nAction: finish\nAction Input: Task completed successfully",
                            "tool_calls": None,
                        }
                    }
                ]
            }
        )

        manager = MagicMock()
        manager.get_provider.return_value = provider
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        manager = MagicMock()
        manager.retrieve_relevant_context = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        manager = MagicMock()
        manager.add_episode = AsyncMock()
        return manager

    @requires_agent_manager
    def test_agent_manager_init_without_observability(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
    ):
        """Test AgentManager init without observability (backward compatible)."""
        from llmcore.agents import AgentManager

        manager = AgentManager(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
        )

        assert manager._observability is None
        assert manager.observability is None
        assert manager.event_logger is None

    @requires_agent_manager
    def test_agent_manager_init_with_observability(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
    ):
        """Test AgentManager init with observability enabled."""
        from llmcore.agents import AgentManager
        from llmcore.agents.observability_factory import (
            ObservabilityComponents,
            create_observability_from_config,
        )

        obs = create_observability_from_config(None, session_id="test-session")

        manager = AgentManager(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            observability=obs,
        )

        assert manager._observability is not None
        assert manager._observability.enabled is True
        assert manager.observability is obs
        assert manager.event_logger is not None

    @requires_agent_manager
    @pytest.mark.asyncio
    async def test_agent_manager_cleanup_with_observability(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
    ):
        """Test cleanup closes observability."""
        from llmcore.agents import AgentManager
        from llmcore.agents.observability_factory import ObservabilityComponents

        mock_logger = AsyncMock()
        mock_logger.close = AsyncMock()

        obs = ObservabilityComponents(
            enabled=True,
            logger=mock_logger,
        )

        manager = AgentManager(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
            observability=obs,
        )

        await manager.cleanup()

        mock_logger.close.assert_called_once()


class TestObservabilityEventLogging:
    """Tests for event logging during agent execution."""

    @pytest.fixture
    def memory_sink(self):
        """Create in-memory sink for testing."""
        from llmcore.agents.observability import InMemorySink

        return InMemorySink()

    @pytest.fixture
    def event_logger(self, memory_sink):
        """Create event logger with memory sink."""
        from llmcore.agents.observability import EventLogger

        logger = EventLogger(session_id="test-session")
        logger.add_sink(memory_sink)
        return logger

    @pytest.mark.asyncio
    async def test_lifecycle_events_logged(self, event_logger, memory_sink):
        """Test that lifecycle events are logged correctly."""
        from llmcore.agents.observability import EventCategory

        # Log lifecycle start
        await event_logger.log_lifecycle_start(goal="Test goal")

        # Log lifecycle end
        await event_logger.log_lifecycle_end(status="success")

        # Verify events
        events = memory_sink.get_events()
        lifecycle_events = [
            e for e in events if e.category == EventCategory.LIFECYCLE
        ]

        assert len(lifecycle_events) >= 2
        assert lifecycle_events[0].event_type == "agent_started"
        assert lifecycle_events[-1].event_type == "agent_completed"

    @pytest.mark.asyncio
    async def test_cognitive_phase_events_logged(self, event_logger, memory_sink):
        """Test that cognitive phase events are logged."""
        from llmcore.agents.observability import EventCategory

        # Log cognitive phases
        await event_logger.log_cognitive_phase(
            phase="plan",
            input_summary="Test goal",
            output_summary="Generated 3 steps",
        )

        await event_logger.log_cognitive_phase(
            phase="think",
            input_summary="Context provided",
            output_summary="Decided to use tool X",
        )

        await event_logger.log_cognitive_phase(
            phase="reflect",
            input_summary="Tool result",
            output_summary="Progress made",
        )

        # Verify events
        events = memory_sink.get_events()
        cognitive_events = [
            e for e in events if e.category == EventCategory.COGNITIVE
        ]

        assert len(cognitive_events) == 3
        phases = [e.phase for e in cognitive_events]
        assert "plan" in phases
        assert "think" in phases
        assert "reflect" in phases

    @pytest.mark.asyncio
    async def test_activity_events_logged(self, event_logger, memory_sink):
        """Test that activity events are logged."""
        from llmcore.agents.observability import EventCategory

        # Log activity
        await event_logger.log_activity(
            activity_name="execute_python",
            activity_input={"code": "print('hello')"},
            activity_output="hello",
            success=True,
            duration_ms=150.0,
        )

        # Verify events
        events = memory_sink.get_events()
        activity_events = [
            e for e in events if e.category == EventCategory.ACTIVITY
        ]

        assert len(activity_events) == 1
        assert activity_events[0].activity_name == "execute_python"

    @pytest.mark.asyncio
    async def test_error_events_logged(self, event_logger, memory_sink):
        """Test that error events are logged."""
        from llmcore.agents.observability import EventCategory, EventSeverity

        # Log error
        await event_logger.log_error(
            error_type="execution_error",
            error_message="Test error message",
            recoverable=False,
        )

        # Verify events
        events = memory_sink.get_events()
        error_events = [
            e for e in events if e.category == EventCategory.ERROR
        ]

        assert len(error_events) == 1
        assert error_events[0].error_type == "execution_error"
        assert error_events[0].severity == EventSeverity.ERROR

    @pytest.mark.asyncio
    async def test_iteration_events_logged(self, event_logger, memory_sink):
        """Test that iteration events are logged."""
        from llmcore.agents.observability import EventCategory

        # Log iteration start and end
        await event_logger.log_iteration_start(iteration=1)
        await event_logger.log_iteration_end(iteration=1, duration_ms=500.0)

        # Verify events
        events = memory_sink.get_events()
        lifecycle_events = [
            e for e in events if e.category == EventCategory.LIFECYCLE
        ]

        assert len(lifecycle_events) == 2
        event_types = [e.event_type for e in lifecycle_events]
        assert "iteration_started" in event_types
        assert "iteration_completed" in event_types


class TestObservabilityMetricsCollection:
    """Tests for metrics collection during agent execution."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector."""
        from llmcore.agents.observability import MetricsCollector

        return MetricsCollector()

    def test_metrics_collector_start_execution(self, metrics_collector):
        """Test starting execution metrics."""
        metrics = metrics_collector.start_execution(
            execution_id="exec-123",
            goal="Test goal",
        )

        assert metrics is not None
        assert metrics.execution_id == "exec-123"
        assert metrics.goal == "Test goal"

    def test_metrics_collector_record_iteration(self, metrics_collector):
        """Test recording iteration metrics."""
        metrics = metrics_collector.start_execution(
            execution_id="exec-123",
            goal="Test goal",
        )

        metrics.record_iteration(duration_ms=150.0)
        metrics.record_iteration(duration_ms=200.0)

        # Use total_iterations property (not iterations attribute)
        assert metrics.total_iterations == 2

    def test_metrics_collector_complete(self, metrics_collector):
        """Test completing execution metrics."""
        from llmcore.agents.observability import ExecutionStatus

        metrics = metrics_collector.start_execution(
            execution_id="exec-123",
            goal="Test goal",
        )

        metrics.record_iteration(duration_ms=150.0)
        metrics.complete(success=True)

        # Check status enum, not completed boolean
        assert metrics.status == ExecutionStatus.SUCCESS

    def test_metrics_collector_summary(self, metrics_collector):
        """Test getting metrics summary."""
        metrics = metrics_collector.start_execution(
            execution_id="exec-123",
            goal="Test goal",
        )

        metrics.record_iteration(duration_ms=150.0)
        metrics.record_iteration(duration_ms=200.0)

        # end_execution moves from _active_executions to _executions
        metrics_collector.end_execution("exec-123", success=True)

        summary = metrics_collector.get_summary()

        assert summary is not None
        # get_summary returns a dict, not an object
        assert summary["total_executions"] >= 1


class TestObservabilityBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    @pytest.fixture
    def mock_provider_manager(self):
        """Create mock provider manager."""
        provider = MagicMock()
        provider.get_name.return_value = "mock_provider"
        provider.default_model = "test-model"
        provider.chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "Task completed.",
                            "tool_calls": None,
                        }
                    }
                ]
            }
        )

        manager = MagicMock()
        manager.get_provider.return_value = provider
        return manager

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        manager = MagicMock()
        manager.retrieve_relevant_context = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        manager = MagicMock()
        manager.add_episode = AsyncMock()
        return manager

    @requires_agent_manager
    def test_agent_manager_works_without_observability(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
    ):
        """Test AgentManager works without observability parameter."""
        from llmcore.agents import AgentManager

        # This should work without observability parameter
        manager = AgentManager(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
        )

        assert manager is not None
        assert manager._observability is None

    @requires_enhanced_agent_manager
    def test_enhanced_agent_manager_works_without_observability(
        self,
        mock_provider_manager,
        mock_memory_manager,
        mock_storage_manager,
    ):
        """Test EnhancedAgentManager works without observability parameter."""
        from llmcore.agents import EnhancedAgentManager

        # This should work without observability parameter
        manager = EnhancedAgentManager(
            provider_manager=mock_provider_manager,
            memory_manager=mock_memory_manager,
            storage_manager=mock_storage_manager,
        )

        assert manager is not None
        capabilities = manager.get_capabilities()
        assert capabilities["observability_enabled"] is False


class TestObservabilityConfiguration:
    """Tests for observability configuration handling."""

    def test_extract_config_from_none(self):
        """Test extracting config from None."""
        from llmcore.agents.observability_factory import _extract_observability_config

        config = _extract_observability_config(None)

        assert config == {"enabled": True}

    def test_create_with_custom_log_path(self):
        """Test creating observability with custom log path."""
        from llmcore.agents.observability_factory import create_event_logger_simple

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = f"{tmpdir}/custom_events.jsonl"

            logger = create_event_logger_simple(
                session_id="test-123",
                log_path=custom_path,
            )

            assert logger is not None


class TestObservabilityPerformance:
    """Performance tests for observability overhead."""

    @pytest.mark.asyncio
    async def test_logging_overhead_acceptable(self):
        """Verify logging adds acceptable overhead (< 100ms for 100 events)."""
        import time
        from llmcore.agents.observability import EventLogger, InMemorySink

        sink = InMemorySink()
        logger = EventLogger(session_id="perf-test")
        logger.add_sink(sink)

        # Time 100 log operations
        start = time.monotonic()

        for i in range(100):
            await logger.log_cognitive_phase(
                phase="think",
                input_summary=f"Test input {i}",
                output_summary=f"Test output {i}",
            )

        duration_ms = (time.monotonic() - start) * 1000

        # Should complete in < 100ms (very generous threshold)
        assert duration_ms < 100, f"Logging 100 events took {duration_ms:.1f}ms"

        await logger.close()


class TestObservabilityReplay:
    """Tests for execution replay functionality."""

    @pytest.mark.asyncio
    async def test_replay_from_logged_events(self):
        """Test replaying execution from logged events."""
        from llmcore.agents.observability import (
            EventLogger,
            ExecutionReplay,
            JSONLFileSink,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_events.jsonl"

            # Log some events
            logger = EventLogger(session_id="test-replay", execution_id="exec-001")
            logger.add_sink(JSONLFileSink(log_path))

            await logger.log_lifecycle_start(goal="Test replay")
            await logger.log_cognitive_phase(phase="plan", output_summary="Planned")
            await logger.log_cognitive_phase(phase="think", output_summary="Thought")
            await logger.log_activity(
                activity_name="test_tool",
                activity_output="Result",
                success=True,
            )
            await logger.log_lifecycle_end(status="success")
            await logger.close()

            # Replay the execution
            replay = ExecutionReplay.from_file(log_path)
            executions = replay.list_executions()

            # list_executions returns ExecutionInfo objects, not strings
            exec_ids = [e.execution_id for e in executions]
            assert "exec-001" in exec_ids

            result = replay.replay("exec-001")

            assert result is not None
            assert len(result.timeline) >= 4  # At least 4 events


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TestObservabilityFactoryUnit",
    "TestAgentManagerObservabilityIntegration",
    "TestObservabilityEventLogging",
    "TestObservabilityMetricsCollection",
    "TestObservabilityBackwardCompatibility",
    "TestObservabilityConfiguration",
    "TestObservabilityPerformance",
    "TestObservabilityReplay",
]
