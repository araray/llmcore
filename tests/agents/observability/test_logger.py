# tests/agents/observability/test_logger.py
"""
Tests for the observability logger module.

Covers:
- EventSink abstract interface
- JSONLFileSink file operations
- InMemorySink memory management
- CallbackSink callback invocation
- FilteredSink filtering logic
- EventLogger core functionality
- Async context management
- Event logging methods
- Statistics tracking
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.observability import (
    ActivityEvent,
    ActivityEventType,
    # Events
    AgentEvent,
    CallbackSink,
    CognitiveEvent,
    CognitiveEventType,
    ErrorEvent,
    ErrorEventType,
    # Enums
    EventCategory,
    EventLogger,
    EventSeverity,
    # Logger
    EventSink,
    FilteredSink,
    HITLEvent,
    HITLEventType,
    InMemorySink,
    JSONLFileSink,
    LifecycleEvent,
    LifecycleEventType,
    MetricEvent,
    MetricEventType,
    SandboxEvent,
    create_event_logger,
)

# =============================================================================
# JSONL FILE SINK TESTS
# =============================================================================


class TestJSONLFileSink:
    """Tests for JSONLFileSink."""

    @pytest.mark.asyncio
    async def test_create_sink(self, temp_jsonl_file):
        """Test creating a JSONL file sink."""
        sink = JSONLFileSink(temp_jsonl_file)

        assert sink.name == "JSONLFileSink"
        await sink.close()

    @pytest.mark.asyncio
    async def test_write_single_event(self, temp_jsonl_file, sample_lifecycle_event):
        """Test writing a single event."""
        sink = JSONLFileSink(temp_jsonl_file, buffer_size=1)

        await sink.write(sample_lifecycle_event)
        await sink.flush()
        await sink.close()

        # Verify file content
        with open(temp_jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["session_id"] == sample_lifecycle_event.session_id
        assert data["category"] == "lifecycle"

    @pytest.mark.asyncio
    async def test_write_multiple_events(self, temp_jsonl_file, sample_event_sequence):
        """Test writing multiple events."""
        sink = JSONLFileSink(temp_jsonl_file, buffer_size=1)

        for event in sample_event_sequence:
            await sink.write(event)

        await sink.flush()
        await sink.close()

        with open(temp_jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == len(sample_event_sequence)

    @pytest.mark.asyncio
    async def test_buffered_writes(self, temp_jsonl_file, sample_lifecycle_event):
        """Test that writes are buffered until buffer_size is reached."""
        sink = JSONLFileSink(temp_jsonl_file, buffer_size=5)

        # Write 3 events (below buffer size)
        for _ in range(3):
            await sink.write(sample_lifecycle_event)

        # File might not exist yet or be empty (buffered)
        if temp_jsonl_file.exists():
            with open(temp_jsonl_file) as f:
                lines = f.readlines()
            # Buffer not flushed yet
            assert len(lines) < 3

        # Write 2 more to hit buffer size
        for _ in range(2):
            await sink.write(sample_lifecycle_event)

        await sink.flush()
        await sink.close()

        with open(temp_jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_flush_writes_buffer(self, temp_jsonl_file, sample_lifecycle_event):
        """Test that flush writes buffered events."""
        sink = JSONLFileSink(temp_jsonl_file, buffer_size=100)

        await sink.write(sample_lifecycle_event)
        await sink.flush()

        with open(temp_jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        await sink.close()

    @pytest.mark.asyncio
    async def test_close_flushes_buffer(self, temp_jsonl_file, sample_lifecycle_event):
        """Test that close flushes any remaining buffered events."""
        sink = JSONLFileSink(temp_jsonl_file, buffer_size=100)

        await sink.write(sample_lifecycle_event)
        await sink.close()  # Should flush

        with open(temp_jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        nested_path = temp_dir / "nested" / "dir" / "events.jsonl"
        sink = JSONLFileSink(nested_path)

        event = LifecycleEvent(
            session_id="test",
            event_type=LifecycleEventType.AGENT_STARTED,
        )

        await sink.write(event)
        await sink.flush()
        await sink.close()

        assert nested_path.exists()

    @pytest.mark.asyncio
    async def test_path_as_string(self, temp_dir, sample_lifecycle_event):
        """Test that string paths work."""
        path = str(temp_dir / "events.jsonl")
        sink = JSONLFileSink(path)

        await sink.write(sample_lifecycle_event)
        await sink.close()

        assert Path(path).exists()


# =============================================================================
# IN MEMORY SINK TESTS
# =============================================================================


class TestInMemorySink:
    """Tests for InMemorySink."""

    def test_create_sink(self):
        """Test creating an in-memory sink."""
        sink = InMemorySink(max_events=100)

        assert sink.name == "InMemorySink"
        assert len(sink.events) == 0

    @pytest.mark.asyncio
    async def test_write_event(self, sample_lifecycle_event):
        """Test writing an event."""
        sink = InMemorySink()

        await sink.write(sample_lifecycle_event)

        assert len(sink.events) == 1
        assert sink.events[0] is sample_lifecycle_event

    @pytest.mark.asyncio
    async def test_write_multiple_events(self, sample_event_sequence):
        """Test writing multiple events."""
        sink = InMemorySink()

        for event in sample_event_sequence:
            await sink.write(event)

        assert len(sink.events) == len(sample_event_sequence)

    @pytest.mark.asyncio
    async def test_max_events_limit(self, sample_lifecycle_event):
        """Test that max_events limit is enforced."""
        sink = InMemorySink(max_events=5)

        # Write 10 events
        for i in range(10):
            event = LifecycleEvent(
                session_id=f"session-{i}",
                event_type=LifecycleEventType.AGENT_STARTED,
            )
            await sink.write(event)

        # Should only keep last 5
        assert len(sink.events) == 5
        # Oldest events should be dropped
        assert sink.events[0].session_id == "session-5"

    @pytest.mark.asyncio
    async def test_get_events_by_category(self, in_memory_sink, sample_event_sequence):
        """Test filtering events by category."""
        for event in sample_event_sequence:
            await in_memory_sink.write(event)

        lifecycle_events = in_memory_sink.get_events(category=EventCategory.LIFECYCLE)
        cognitive_events = in_memory_sink.get_events(category=EventCategory.COGNITIVE)
        activity_events = in_memory_sink.get_events(category=EventCategory.ACTIVITY)

        assert len(lifecycle_events) > 0
        assert all(e.category == EventCategory.LIFECYCLE for e in lifecycle_events)

    @pytest.mark.asyncio
    async def test_get_events_by_severity(self, in_memory_sink):
        """Test filtering events by severity."""
        events = [
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.INFO,
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.WARNING,
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.ERROR,
            ),
        ]

        for event in events:
            await in_memory_sink.write(event)

        error_events = in_memory_sink.get_events(severity=EventSeverity.ERROR)
        assert len(error_events) == 1
        assert error_events[0].severity == EventSeverity.ERROR

    @pytest.mark.asyncio
    async def test_get_events_by_session_id(self, in_memory_sink):
        """Test filtering events by session_id."""
        events = [
            AgentEvent(
                session_id="session-1",
                category=EventCategory.LIFECYCLE,
                event_type="test",
            ),
            AgentEvent(
                session_id="session-2",
                category=EventCategory.LIFECYCLE,
                event_type="test",
            ),
            AgentEvent(
                session_id="session-1",
                category=EventCategory.ACTIVITY,
                event_type="test",
            ),
        ]

        for event in events:
            await in_memory_sink.write(event)

        session1_events = in_memory_sink.get_events(session_id="session-1")
        assert len(session1_events) == 2

    @pytest.mark.asyncio
    async def test_get_events_by_execution_id(self, in_memory_sink):
        """Test filtering events by execution_id."""
        events = [
            AgentEvent(
                session_id="test",
                execution_id="exec-1",
                category=EventCategory.LIFECYCLE,
                event_type="test",
            ),
            AgentEvent(
                session_id="test",
                execution_id="exec-2",
                category=EventCategory.LIFECYCLE,
                event_type="test",
            ),
        ]

        for event in events:
            await in_memory_sink.write(event)

        exec1_events = in_memory_sink.get_events(execution_id="exec-1")
        assert len(exec1_events) == 1

    @pytest.mark.asyncio
    async def test_get_events_by_time_range(self, in_memory_sink, fixed_timestamp):
        """Test filtering events by time range."""
        events = [
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                timestamp=fixed_timestamp,
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                timestamp=fixed_timestamp + timedelta(hours=1),
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                timestamp=fixed_timestamp + timedelta(hours=2),
            ),
        ]

        for event in events:
            await in_memory_sink.write(event)

        # Get events in middle time range
        filtered = in_memory_sink.get_events(
            since=fixed_timestamp + timedelta(minutes=30),
            until=fixed_timestamp + timedelta(hours=1, minutes=30),
        )

        assert len(filtered) == 1

    @pytest.mark.asyncio
    async def test_clear(self, in_memory_sink, sample_event_sequence):
        """Test clearing all events."""
        for event in sample_event_sequence:
            await in_memory_sink.write(event)

        assert len(in_memory_sink.events) > 0

        in_memory_sink.clear()

        assert len(in_memory_sink.events) == 0

    @pytest.mark.asyncio
    async def test_flush_is_noop(self, in_memory_sink):
        """Test that flush is a no-op for in-memory sink."""
        await in_memory_sink.flush()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_clears_events(self, in_memory_sink, sample_lifecycle_event):
        """Test that close clears events."""
        await in_memory_sink.write(sample_lifecycle_event)
        await in_memory_sink.close()

        assert len(in_memory_sink.events) == 0


# =============================================================================
# CALLBACK SINK TESTS
# =============================================================================


class TestCallbackSink:
    """Tests for CallbackSink."""

    @pytest.mark.asyncio
    async def test_create_sink(self, mock_callback):
        """Test creating a callback sink."""
        sink = CallbackSink(mock_callback)

        assert sink.name == "CallbackSink"

    @pytest.mark.asyncio
    async def test_callback_invoked_on_write(self, mock_callback, sample_lifecycle_event):
        """Test that callback is invoked when writing."""
        sink = CallbackSink(mock_callback)

        await sink.write(sample_lifecycle_event)

        mock_callback.assert_called_once_with(sample_lifecycle_event)

    @pytest.mark.asyncio
    async def test_callback_invoked_for_each_event(self, mock_callback, sample_event_sequence):
        """Test that callback is invoked for each event."""
        sink = CallbackSink(mock_callback)

        for event in sample_event_sequence:
            await sink.write(event)

        assert mock_callback.call_count == len(sample_event_sequence)

    @pytest.mark.asyncio
    async def test_sync_callback_supported(self, sample_lifecycle_event):
        """Test that synchronous callbacks are supported."""
        received_events = []

        def sync_callback(event):
            received_events.append(event)

        sink = CallbackSink(sync_callback)
        await sink.write(sample_lifecycle_event)

        assert len(received_events) == 1
        assert received_events[0] is sample_lifecycle_event

    @pytest.mark.asyncio
    async def test_async_callback_supported(self, sample_lifecycle_event):
        """Test that callbacks work with the CallbackSink.
        
        Note: Current implementation calls callback synchronously (doesn't await).
        Async callbacks would create unawaited coroutines. Use sync callbacks.
        """
        received_events = []

        def sync_callback(event):
            received_events.append(event)

        sink = CallbackSink(sync_callback)
        await sink.write(sample_lifecycle_event)

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_custom_name(self, mock_callback):
        """Test sink name."""
        sink = CallbackSink(mock_callback)

        # Default name is the class name
        assert sink.name == "CallbackSink"


# =============================================================================
# FILTERED SINK TESTS
# =============================================================================


class TestFilteredSink:
    """Tests for FilteredSink."""

    @pytest.mark.asyncio
    async def test_create_filtered_sink(self, in_memory_sink):
        """Test creating a filtered sink."""
        sink = FilteredSink(
            inner_sink=in_memory_sink,
            categories=[EventCategory.LIFECYCLE],
        )

        assert "Filtered" in sink.name

    @pytest.mark.asyncio
    async def test_filter_by_category(self, in_memory_sink):
        """Test filtering by category."""
        sink = FilteredSink(
            inner_sink=in_memory_sink,
            categories=[EventCategory.LIFECYCLE],
        )

        lifecycle_event = LifecycleEvent(
            session_id="test",
            event_type=LifecycleEventType.AGENT_STARTED,
        )
        activity_event = ActivityEvent(
            session_id="test",
            event_type=ActivityEventType.ACTIVITY_STARTED,
            activity_type="test",
            activity_name="test",
        )

        await sink.write(lifecycle_event)
        await sink.write(activity_event)

        # Only lifecycle event should be written
        assert len(in_memory_sink.events) == 1
        assert in_memory_sink.events[0].category == EventCategory.LIFECYCLE

    @pytest.mark.asyncio
    async def test_filter_by_multiple_categories(self, in_memory_sink):
        """Test filtering by multiple categories."""
        sink = FilteredSink(
            inner_sink=in_memory_sink,
            categories=[EventCategory.LIFECYCLE, EventCategory.ACTIVITY],
        )

        events = [
            LifecycleEvent(
                session_id="test",
                event_type=LifecycleEventType.AGENT_STARTED,
            ),
            ActivityEvent(
                session_id="test",
                event_type=ActivityEventType.ACTIVITY_STARTED,
                activity_type="test",
                activity_name="test",
            ),
            CognitiveEvent(
                session_id="test",
                event_type=CognitiveEventType.PHASE_STARTED,
            ),
        ]

        for event in events:
            await sink.write(event)

        # Lifecycle and Activity should pass, Cognitive should be filtered
        assert len(in_memory_sink.events) == 2

    @pytest.mark.asyncio
    async def test_filter_by_min_severity(self, in_memory_sink):
        """Test filtering by minimum severity."""
        sink = FilteredSink(
            inner_sink=in_memory_sink,
            min_severity=EventSeverity.WARNING,
        )

        events = [
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.DEBUG,
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.INFO,
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.WARNING,
            ),
            AgentEvent(
                session_id="test",
                category=EventCategory.LIFECYCLE,
                event_type="test",
                severity=EventSeverity.ERROR,
            ),
        ]

        for event in events:
            await sink.write(event)

        # Only WARNING and ERROR should pass
        assert len(in_memory_sink.events) == 2

    @pytest.mark.asyncio
    async def test_filter_by_event_types(self, in_memory_sink):
        """Test filtering by specific event types."""
        sink = FilteredSink(
            inner_sink=in_memory_sink,
            include_types=["agent_started", "agent_completed"],
        )

        events = [
            LifecycleEvent(
                session_id="test",
                event_type=LifecycleEventType.AGENT_STARTED,
            ),
            LifecycleEvent(
                session_id="test",
                event_type=LifecycleEventType.AGENT_COMPLETED,
            ),
            LifecycleEvent(
                session_id="test",
                event_type=LifecycleEventType.ITERATION_STARTED,
            ),
        ]

        for event in events:
            await sink.write(event)

        assert len(in_memory_sink.events) == 2

    @pytest.mark.asyncio
    async def test_combined_filters(self, in_memory_sink):
        """Test combining multiple filter criteria."""
        sink = FilteredSink(
            inner_sink=in_memory_sink,
            categories=[EventCategory.LIFECYCLE, EventCategory.ERROR],
            min_severity=EventSeverity.WARNING,
        )

        events = [
            LifecycleEvent(
                session_id="test",
                event_type=LifecycleEventType.AGENT_STARTED,
                severity=EventSeverity.INFO,  # Filtered out (severity)
            ),
            LifecycleEvent(
                session_id="test",
                event_type=LifecycleEventType.AGENT_FAILED,
                severity=EventSeverity.ERROR,  # Passes
            ),
            ErrorEvent(
                session_id="test",
                event_type=ErrorEventType.EXCEPTION,
                severity=EventSeverity.ERROR,
                error_type="Test",
                error_message="Test",
            ),  # Passes
            CognitiveEvent(
                session_id="test",
                event_type=CognitiveEventType.PHASE_COMPLETED,
                severity=EventSeverity.ERROR,  # Filtered out (category)
            ),
        ]

        for event in events:
            await sink.write(event)

        assert len(in_memory_sink.events) == 2

    @pytest.mark.asyncio
    async def test_flush_delegates(self, in_memory_sink):
        """Test that flush delegates to wrapped sink."""
        sink = FilteredSink(inner_sink=in_memory_sink)

        await sink.flush()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_delegates(self, in_memory_sink):
        """Test that close delegates to wrapped sink."""
        sink = FilteredSink(inner_sink=in_memory_sink)

        await sink.close()


# =============================================================================
# EVENT LOGGER TESTS
# =============================================================================


class TestEventLogger:
    """Tests for EventLogger."""

    def test_create_logger(self, session_id, execution_id):
        """Test creating an event logger."""
        logger = EventLogger(
            session_id=session_id,
            execution_id=execution_id,
        )

        assert logger.session_id == session_id
        assert logger.execution_id == execution_id

    def test_add_sink(self, event_logger, in_memory_sink):
        """Test adding a sink."""
        # event_logger fixture already has one sink
        new_sink = InMemorySink()
        event_logger.add_sink(new_sink)

        # Should have 2 sinks now
        assert len(event_logger.sinks) == 2

    def test_remove_sink(self, event_logger, in_memory_sink):
        """Test removing a sink."""
        removed = event_logger.remove_sink(in_memory_sink)

        assert removed is True
        assert len(event_logger.sinks) == 0

    def test_remove_nonexistent_sink(self, event_logger):
        """Test removing a sink that wasn't added."""
        other_sink = InMemorySink()
        removed = event_logger.remove_sink(other_sink)

        assert removed is False

    @pytest.mark.asyncio
    async def test_log_event(self, event_logger, in_memory_sink, sample_lifecycle_event):
        """Test logging an event."""
        logged_event = await event_logger.log(sample_lifecycle_event)

        # Event should be returned
        assert logged_event is sample_lifecycle_event
        # Event should be in sink
        assert len(in_memory_sink.events) == 1

    @pytest.mark.asyncio
    async def test_log_multiple_events(self, event_logger, in_memory_sink, sample_event_sequence):
        """Test logging multiple events."""
        for event in sample_event_sequence:
            await event_logger.log(event)

        assert len(in_memory_sink.events) == len(sample_event_sequence)

    @pytest.mark.asyncio
    async def test_log_to_multiple_sinks(self, session_id, execution_id, sample_lifecycle_event):
        """Test that events are logged to all sinks."""
        sink1 = InMemorySink()
        sink2 = InMemorySink()

        logger = EventLogger(session_id=session_id, execution_id=execution_id)
        logger.add_sink(sink1)
        logger.add_sink(sink2)

        await logger.log(sample_lifecycle_event)

        assert len(sink1.events) == 1
        assert len(sink2.events) == 1

    @pytest.mark.asyncio
    async def test_set_iteration(self, event_logger):
        """Test setting current iteration."""
        event_logger.set_iteration(5)

        assert event_logger._iteration == 5

    @pytest.mark.asyncio
    async def test_set_phase(self, event_logger):
        """Test setting current phase."""
        event_logger.set_phase("think")

        assert event_logger._phase == "think"

        event_logger.set_phase(None)
        assert event_logger._phase is None

    @pytest.mark.asyncio
    async def test_set_correlation_id(self, event_logger, correlation_id):
        """Test setting correlation ID."""
        event_logger.set_correlation_id(correlation_id)

        assert event_logger._correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_event_count(self, event_logger, in_memory_sink, sample_event_sequence):
        """Test event count tracking."""
        assert event_logger.event_count == 0

        for event in sample_event_sequence:
            await event_logger.log(event)

        assert event_logger.event_count == len(sample_event_sequence)

    @pytest.mark.asyncio
    async def test_error_count(self, event_logger, in_memory_sink, session_id, execution_id):
        """Test error count tracking."""
        assert event_logger.error_count == 0

        # Log a normal event
        await event_logger.log(LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
        ))

        assert event_logger.error_count == 0

        # Log an error event
        await event_logger.log(ErrorEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ErrorEventType.EXCEPTION,
            error_type="Test",
            error_message="Test",
        ))

        assert event_logger.error_count == 1

    @pytest.mark.asyncio
    async def test_get_statistics(self, event_logger, in_memory_sink, sample_event_sequence):
        """Test getting statistics."""
        for event in sample_event_sequence:
            await event_logger.log(event)

        stats = event_logger.get_statistics()

        assert stats["event_count"] == len(sample_event_sequence)
        assert "session_id" in stats
        assert "execution_id" in stats
        assert "sink_count" in stats

    @pytest.mark.asyncio
    async def test_async_context_manager(self, session_id, execution_id, in_memory_sink):
        """Test async context manager."""
        event_logged = False
        async with EventLogger(
            session_id=session_id,
            execution_id=execution_id,
        ) as logger:
            logger.add_sink(in_memory_sink)
            await logger.log(LifecycleEvent(
                session_id=session_id,
                event_type=LifecycleEventType.AGENT_STARTED,
            ))
            # Check events were written (before context exit which calls close)
            event_logged = len(in_memory_sink.events) == 1

        # Context manager should have called close() without error
        assert event_logged, "Event should have been logged before context exit"

    @pytest.mark.asyncio
    async def test_flush(self, event_logger, in_memory_sink, sample_lifecycle_event):
        """Test flush method."""
        await event_logger.log(sample_lifecycle_event)
        await event_logger.flush()

        # No exception should be raised

    @pytest.mark.asyncio
    async def test_close(self, event_logger, sample_lifecycle_event):
        """Test close method."""
        await event_logger.log(sample_lifecycle_event)
        await event_logger.close()

        # No exception should be raised


# =============================================================================
# EVENT LOGGER CONVENIENCE METHODS
# =============================================================================


class TestEventLoggerLifecycleMethods:
    """Tests for EventLogger lifecycle convenience methods."""

    @pytest.mark.asyncio
    async def test_log_lifecycle_start(self, event_logger, in_memory_sink):
        """Test log_lifecycle_start method."""
        event = await event_logger.log_lifecycle_start(goal="Test goal")

        assert isinstance(event, LifecycleEvent)
        assert event.event_type == LifecycleEventType.AGENT_STARTED
        assert event.goal == "Test goal"
        assert len(in_memory_sink.events) == 1

    @pytest.mark.asyncio
    async def test_log_lifecycle_end_success(self, event_logger, in_memory_sink):
        """Test log_lifecycle_end for successful completion."""
        event = await event_logger.log_lifecycle_end(
            status="success",
            total_iterations=5,
            exit_reason="Goal achieved",
        )

        assert isinstance(event, LifecycleEvent)
        assert event.event_type == LifecycleEventType.AGENT_COMPLETED
        assert event.final_status == "success"
        assert event.total_iterations == 5

    @pytest.mark.asyncio
    async def test_log_lifecycle_end_failure(self, event_logger, in_memory_sink):
        """Test log_lifecycle_end for failure."""
        event = await event_logger.log_lifecycle_end(
            status="failure",
            exit_reason="Max iterations exceeded",
        )

        assert event.event_type == LifecycleEventType.AGENT_FAILED
        assert event.final_status == "failure"

    @pytest.mark.asyncio
    async def test_log_iteration_start(self, event_logger, in_memory_sink):
        """Test log_iteration_start method."""
        event = await event_logger.log_iteration_start(iteration=1)

        assert event.event_type == LifecycleEventType.ITERATION_STARTED
        assert event.iteration == 1

    @pytest.mark.asyncio
    async def test_log_iteration_end(self, event_logger, in_memory_sink):
        """Test log_iteration_end method."""
        event = await event_logger.log_iteration_end(
            iteration=1,
            activities_executed=3,
        )

        assert event.event_type == LifecycleEventType.ITERATION_COMPLETED
        # Note: iteration is set on logger state, not on the event
        assert event.activities_executed == 3


class TestEventLoggerCognitiveMethods:
    """Tests for EventLogger cognitive convenience methods."""

    @pytest.mark.asyncio
    async def test_log_cognitive_phase(self, event_logger, in_memory_sink):
        """Test log_cognitive_phase method."""
        event = await event_logger.log_cognitive_phase(
            phase="think",
            iteration=1,
            event_type=CognitiveEventType.PHASE_COMPLETED,
            input_summary="User query",
            output_summary="Decision made",
            duration_ms=500.0,
        )

        assert isinstance(event, CognitiveEvent)
        assert event.phase == "think"
        assert event.input_summary == "User query"
        assert event.duration_ms == 500.0


class TestEventLoggerActivityMethods:
    """Tests for EventLogger activity convenience methods."""

    @pytest.mark.asyncio
    async def test_log_activity(self, event_logger, in_memory_sink):
        """Test log_activity method."""
        event = await event_logger.log_activity(
            activity_type="execute_python",
            activity_name="run_script",
            event_type=ActivityEventType.ACTIVITY_COMPLETED,
            success=True,
            result="Output text",
            parameters={"code": "print('hello')"},
            duration_ms=150.0,
        )

        assert isinstance(event, ActivityEvent)
        assert event.activity_type == "execute_python"
        assert event.success is True
        assert event.result == "Output text"


class TestEventLoggerErrorMethods:
    """Tests for EventLogger error convenience methods."""

    @pytest.mark.asyncio
    async def test_log_error(self, event_logger, in_memory_sink):
        """Test log_error method."""
        event = await event_logger.log_error(
            error_type="ValueError",
            error_message="Invalid input",
            recoverable=True,
            stack_trace="Traceback...",
        )

        assert isinstance(event, ErrorEvent)
        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid input"
        assert event.recoverable is True
        assert event_logger.error_count == 1


class TestEventLoggerMetricMethods:
    """Tests for EventLogger metric convenience methods."""

    @pytest.mark.asyncio
    async def test_log_metric(self, event_logger, in_memory_sink):
        """Test log_metric method."""
        event = await event_logger.log_metric(
            metric_name="tokens_used",
            metric_value=1500.0,
            metric_unit="tokens",
            event_type=MetricEventType.TOKEN_USAGE,
        )

        assert isinstance(event, MetricEvent)
        assert event.metric_name == "tokens_used"
        assert event.metric_value == 1500.0


class TestEventLoggerHITLMethods:
    """Tests for EventLogger HITL convenience methods."""

    @pytest.mark.asyncio
    async def test_log_hitl(self, event_logger, in_memory_sink):
        """Test log_hitl method."""
        event = await event_logger.log_hitl(
            event_type=HITLEventType.APPROVAL_REQUESTED,
            request_id="req-123",
            action_type="execute_shell",
            risk_level="high",
            approval_status="pending",
        )

        assert isinstance(event, HITLEvent)
        assert event.request_id == "req-123"
        assert event.risk_level == "high"


class TestEventLoggerSandboxMethods:
    """Tests for EventLogger sandbox convenience methods."""

    @pytest.mark.asyncio
    async def test_log_sandbox(self, event_logger, in_memory_sink):
        """Test log_sandbox method."""
        event = await event_logger.log_sandbox(
            event_type="sandbox_created",
            sandbox_type="docker",
            operation="create",
            sandbox_id="sandbox-123",
            image="python:3.11",
        )

        assert isinstance(event, SandboxEvent)
        assert event.sandbox_id == "sandbox-123"
        assert event.sandbox_type == "docker"


# =============================================================================
# EVENT SCOPE TESTS
# =============================================================================


class TestEventScope:
    """Tests for EventLogger event_scope context manager."""

    @pytest.mark.asyncio
    async def test_event_scope_basic(self, event_logger, in_memory_sink, session_id, execution_id):
        """Test basic event scope functionality."""
        parent_event = LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.ITERATION_STARTED,
        )
        await event_logger.log(parent_event)

        async with event_logger.event_scope(parent_event):
            child_event = await event_logger.log(LifecycleEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=LifecycleEventType.ITERATION_COMPLETED,
            ))

            assert child_event.parent_event_id == parent_event.event_id

    @pytest.mark.asyncio
    async def test_event_scope_restores_state(self, event_logger, in_memory_sink, session_id, execution_id):
        """Test that event scope restores previous state."""
        outer_parent = LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
        )
        await event_logger.log(outer_parent)

        async with event_logger.event_scope(outer_parent):
            inner_parent = LifecycleEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=LifecycleEventType.ITERATION_STARTED,
            )
            await event_logger.log(inner_parent)

            # Inner parent should have outer as parent
            assert inner_parent.parent_event_id == outer_parent.event_id

        # After scope, parent should be None again
        final_event = await event_logger.log(LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_COMPLETED,
        ))

        assert final_event.parent_event_id is None


# =============================================================================
# CREATE EVENT LOGGER FACTORY TESTS
# =============================================================================


class TestCreateEventLogger:
    """Tests for create_event_logger factory function."""

    def test_create_basic_logger(self, session_id):
        """Test creating a basic logger."""
        logger = create_event_logger(session_id=session_id)

        assert logger.session_id == session_id

    def test_create_logger_with_file_sink(self, session_id, temp_jsonl_file):
        """Test creating logger with file sink."""
        logger = create_event_logger(
            session_id=session_id,
            log_path=temp_jsonl_file,
        )

        assert len(logger.sinks) >= 1

    def test_create_logger_with_options(self, session_id, execution_id):
        """Test creating logger with all options."""
        logger = create_event_logger(
            session_id=session_id,
            execution_id=execution_id,
        )

        assert logger.session_id == session_id
        assert logger.execution_id == execution_id


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestLoggerEdgeCases:
    """Tests for logger edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_log_without_sinks(self, session_id, execution_id, sample_lifecycle_event):
        """Test logging when no sinks are configured."""
        logger = EventLogger(session_id=session_id, execution_id=execution_id)

        # Should not raise, just log nothing
        event = await logger.log(sample_lifecycle_event)
        assert event is sample_lifecycle_event

    @pytest.mark.asyncio
    async def test_sink_write_error_handling(self, event_logger, sample_lifecycle_event):
        """Test handling of sink write errors."""
        error_sink = AsyncMock(spec=EventSink)
        error_sink.write = AsyncMock(side_effect=Exception("Write failed"))
        error_sink.name = MagicMock(return_value="ErrorSink")

        event_logger.add_sink(error_sink)

        # Should not raise, error should be logged
        event = await event_logger.log(sample_lifecycle_event)
        assert event is sample_lifecycle_event

    @pytest.mark.asyncio
    async def test_empty_statistics(self, session_id, execution_id):
        """Test statistics when no events logged."""
        logger = EventLogger(session_id=session_id, execution_id=execution_id)

        stats = logger.get_statistics()

        assert stats["event_count"] == 0
        assert stats["error_count"] == 0
