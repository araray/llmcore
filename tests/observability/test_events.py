# tests/observability/test_events.py
"""Tests for the event logging system."""

import gzip
import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmcore.observability.events import (
    # Data models
    Event,
    # Core classes
    EventBuffer,
    EventBufferConfig,
    # Enums
    EventCategory,
    EventFileWriter,
    EventRotationConfig,
    ExecutionReplayer,
    # Replay
    ExecutionTrace,
    ObservabilityConfig,
    ObservabilityLogger,
    RotationStrategy,
    Severity,
    # Factories
    create_observability_logger,
    load_events_from_file,
)

# =============================================================================
# EVENT CATEGORY TESTS
# =============================================================================


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories are defined."""
        expected = [
            "lifecycle",
            "cognitive",
            "activity",
            "hitl",
            "error",
            "metric",
            "memory",
            "sandbox",
            "rag",
            "llm",
            "storage",
            "config",
            "custom",
        ]
        for cat in expected:
            assert EventCategory(cat) is not None

    def test_category_string_values(self):
        """Test category string values match enum."""
        assert EventCategory.LIFECYCLE.value == "lifecycle"
        assert EventCategory.COGNITIVE.value == "cognitive"
        assert EventCategory.ACTIVITY.value == "activity"


# =============================================================================
# SEVERITY TESTS
# =============================================================================


class TestSeverity:
    """Tests for Severity enum."""

    def test_from_string_valid(self):
        """Test parsing valid severity strings."""
        assert Severity.from_string("debug") == Severity.DEBUG
        assert Severity.from_string("INFO") == Severity.INFO
        assert Severity.from_string("Warning") == Severity.WARNING

    def test_from_string_invalid(self):
        """Test parsing invalid severity defaults to INFO."""
        assert Severity.from_string("unknown") == Severity.INFO
        assert Severity.from_string("") == Severity.INFO

    def test_comparison_operators(self):
        """Test severity comparison."""
        assert Severity.DEBUG < Severity.INFO
        assert Severity.ERROR > Severity.WARNING
        assert Severity.CRITICAL >= Severity.CRITICAL
        assert Severity.INFO <= Severity.WARNING


# =============================================================================
# EVENT MODEL TESTS
# =============================================================================


class TestEvent:
    """Tests for Event model."""

    def test_create_minimal_event(self):
        """Test creating event with minimal fields."""
        event = Event(
            category=EventCategory.LIFECYCLE,
            event_type="test_event",
        )

        assert event.id is not None
        assert event.timestamp is not None
        assert event.category == EventCategory.LIFECYCLE
        assert event.event_type == "test_event"
        assert event.severity == Severity.INFO
        assert event.data == {}

    def test_create_full_event(self):
        """Test creating event with all fields."""
        event = Event(
            category=EventCategory.COGNITIVE,
            event_type="phase_completed",
            severity=Severity.WARNING,
            execution_id="exec_123",
            iteration=5,
            session_id="sess_456",
            user_id="user_789",
            data={"phase": "THINK", "duration_ms": 1500},
            source="darwin_agent",
            tags=["agent", "cognitive"],
        )

        assert event.execution_id == "exec_123"
        assert event.iteration == 5
        assert event.data["phase"] == "THINK"
        assert "agent" in event.tags

    def test_to_jsonl(self):
        """Test JSONL serialization."""
        event = Event(
            category=EventCategory.ACTIVITY,
            event_type="tool_executed",
            data={"tool": "execute_python"},
        )

        jsonl = event.to_jsonl()
        assert isinstance(jsonl, str)

        # Verify it's valid JSON
        parsed = json.loads(jsonl)
        assert parsed["category"] == "activity"
        assert parsed["event_type"] == "tool_executed"

    def test_from_jsonl(self):
        """Test JSONL deserialization."""
        original = Event(
            category=EventCategory.ERROR,
            event_type="exception",
            severity=Severity.ERROR,
            data={"message": "Test error"},
        )

        jsonl = original.to_jsonl()
        restored = Event.from_jsonl(jsonl)

        assert restored.category == original.category
        assert restored.event_type == original.event_type
        assert restored.severity == original.severity
        assert restored.data["message"] == "Test error"

    def test_to_dict(self):
        """Test dictionary conversion."""
        event = Event(
            category=EventCategory.RAG,
            event_type="search",
            data={"query": "test query", "results": 5},
        )

        d = event.to_dict()

        assert isinstance(d, dict)
        assert d["category"] == "rag"
        assert d["event_type"] == "search"
        assert d["data"]["results"] == 5


# =============================================================================
# EVENT BUFFER TESTS
# =============================================================================


class TestEventBuffer:
    """Tests for EventBuffer class."""

    def test_add_event(self):
        """Test adding events to buffer."""
        buffer = EventBuffer(max_size=10)

        event = Event(
            category=EventCategory.LIFECYCLE,
            event_type="start",
        )

        result = buffer.add(event)

        assert result is True
        assert len(buffer) == 1

    def test_buffer_max_size(self):
        """Test buffer respects max size."""
        flushed = []
        buffer = EventBuffer(
            max_size=3,
            flush_callback=lambda events: flushed.extend(events),
        )

        for i in range(5):
            buffer.add(
                Event(
                    category=EventCategory.LIFECYCLE,
                    event_type=f"event_{i}",
                )
            )

        # Should have flushed at least once
        assert len(flushed) > 0

    def test_flush(self):
        """Test manual flush."""
        buffer = EventBuffer(max_size=100)

        for i in range(5):
            buffer.add(
                Event(
                    category=EventCategory.LIFECYCLE,
                    event_type=f"event_{i}",
                )
            )

        events = buffer.flush()

        assert len(events) == 5
        assert len(buffer) == 0

    def test_stats(self):
        """Test buffer statistics."""
        buffer = EventBuffer(max_size=100)

        for i in range(10):
            buffer.add(
                Event(
                    category=EventCategory.LIFECYCLE,
                    event_type=f"event_{i}",
                )
            )

        stats = buffer.stats

        assert stats["buffer_size"] == 10
        assert stats["total_events"] == 10
        assert stats["max_size"] == 100

    def test_thread_safety(self):
        """Test buffer is thread-safe."""
        buffer = EventBuffer(max_size=1000)
        errors = []

        def add_events(count):
            try:
                for i in range(count):
                    buffer.add(
                        Event(
                            category=EventCategory.LIFECYCLE,
                            event_type=f"thread_event_{i}",
                        )
                    )
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=add_events, args=(100,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# EVENT FILE WRITER TESTS
# =============================================================================


class TestEventFileWriter:
    """Tests for EventFileWriter class."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path):
        """Create temporary log path."""
        return tmp_path / "test_events.jsonl"

    def test_write_events(self, tmp_log_path):
        """Test writing events to file."""
        writer = EventFileWriter(tmp_log_path)

        events = [
            Event(category=EventCategory.LIFECYCLE, event_type="start"),
            Event(category=EventCategory.LIFECYCLE, event_type="stop"),
        ]

        written = writer.write(events)

        assert written == 2
        assert tmp_log_path.exists()

        # Verify content
        with open(tmp_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_append_mode(self, tmp_log_path):
        """Test events are appended to existing file."""
        writer = EventFileWriter(tmp_log_path)

        # First write
        writer.write([Event(category=EventCategory.LIFECYCLE, event_type="first")])

        # Second write
        writer.write([Event(category=EventCategory.LIFECYCLE, event_type="second")])

        with open(tmp_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_rotation_by_size(self, tmp_path):
        """Test rotation when file exceeds size limit."""
        log_path = tmp_path / "events.jsonl"

        writer = EventFileWriter(
            log_path,
            rotation_config=EventRotationConfig(
                strategy=RotationStrategy.SIZE,
                max_size_mb=1,  # 1MB limit
                compress=False,
            ),
        )

        # Write enough data to trigger rotation
        # Create large events to exceed 1MB
        large_data = {"content": "x" * 10000}
        for _ in range(150):  # ~1.5MB of data
            writer.write(
                [
                    Event(
                        category=EventCategory.METRIC,
                        event_type="large_event",
                        data=large_data,
                    )
                ]
            )

        # Check for rotated files
        rotated_files = list(tmp_path.glob("events_*.jsonl"))
        # Rotation should have occurred
        assert writer.current_size_bytes < 1024 * 1024  # Less than 1MB after rotation


# =============================================================================
# OBSERVABILITY LOGGER TESTS
# =============================================================================


class TestObservabilityLogger:
    """Tests for ObservabilityLogger class."""

    @pytest.fixture
    def logger_config(self, tmp_path):
        """Create test configuration."""
        return ObservabilityConfig(
            enabled=True,
            events_enabled=True,
            log_path=str(tmp_path / "events.jsonl"),
            min_severity="debug",
            buffer=EventBufferConfig(enabled=False),  # Direct write for testing
        )

    def test_log_event_basic(self, logger_config):
        """Test basic event logging."""
        obs_logger = ObservabilityLogger(logger_config)

        event = obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="test_event",
            data={"test": "data"},
        )

        assert event is not None
        assert event.category == EventCategory.LIFECYCLE
        assert event.data["test"] == "data"

    def test_log_event_with_context(self, logger_config):
        """Test logging with execution context."""
        obs_logger = ObservabilityLogger(logger_config)

        event = obs_logger.log_event(
            category=EventCategory.COGNITIVE,
            event_type="phase_start",
            execution_id="exec_123",
            iteration=5,
            session_id="sess_456",
            data={"phase": "THINK"},
        )

        assert event.execution_id == "exec_123"
        assert event.iteration == 5
        assert event.session_id == "sess_456"

    def test_severity_filtering(self, tmp_path):
        """Test events are filtered by severity."""
        config = ObservabilityConfig(
            log_path=str(tmp_path / "events.jsonl"),
            min_severity="warning",
            buffer=EventBufferConfig(enabled=False),
        )
        obs_logger = ObservabilityLogger(config)

        # Debug should be filtered
        event_debug = obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="debug_event",
            severity=Severity.DEBUG,
        )

        # Warning should pass
        event_warning = obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="warning_event",
            severity=Severity.WARNING,
        )

        assert event_debug is None
        assert event_warning is not None

    def test_category_filtering(self, tmp_path):
        """Test events are filtered by category."""
        config = ObservabilityConfig(
            log_path=str(tmp_path / "events.jsonl"),
            categories=["lifecycle", "error"],
            buffer=EventBufferConfig(enabled=False),
        )
        obs_logger = ObservabilityLogger(config)

        # Lifecycle should pass
        event_lifecycle = obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="start",
        )

        # Cognitive should be filtered
        event_cognitive = obs_logger.log_event(
            category=EventCategory.COGNITIVE,
            event_type="think",
        )

        assert event_lifecycle is not None
        assert event_cognitive is None

    def test_callbacks(self, logger_config):
        """Test event callbacks are called."""
        obs_logger = ObservabilityLogger(logger_config)
        received = []

        obs_logger.add_callback(lambda e: received.append(e))

        obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )

        assert len(received) == 1
        assert received[0].event_type == "test"

    def test_stats(self, logger_config):
        """Test logger statistics."""
        obs_logger = ObservabilityLogger(logger_config)

        for i in range(10):
            obs_logger.log_event(
                category=EventCategory.LIFECYCLE,
                event_type=f"event_{i}",
            )

        stats = obs_logger.stats

        assert stats["total_events"] == 10
        assert stats["enabled"] is True

    def test_disable_enable(self, logger_config):
        """Test disabling and enabling logger."""
        obs_logger = ObservabilityLogger(logger_config)

        obs_logger.disable()
        event = obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="disabled",
        )
        assert event is None

        obs_logger.enable()
        event = obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="enabled",
        )
        assert event is not None


# =============================================================================
# EXECUTION TRACE TESTS
# =============================================================================


class TestExecutionTrace:
    """Tests for ExecutionTrace class."""

    def test_add_event(self):
        """Test adding events to trace."""
        trace = ExecutionTrace(execution_id="test_exec")

        event = Event(
            category=EventCategory.COGNITIVE,
            event_type="phase_start",
            execution_id="test_exec",
        )
        trace.add_event(event)

        assert len(trace.events) == 1
        assert trace.start_time is not None
        assert trace.end_time is not None

    def test_filter_by_category(self):
        """Test filtering events by category."""
        trace = ExecutionTrace(execution_id="test")

        trace.add_event(
            Event(
                category=EventCategory.COGNITIVE,
                event_type="think",
            )
        )
        trace.add_event(
            Event(
                category=EventCategory.ACTIVITY,
                event_type="tool_call",
            )
        )
        trace.add_event(
            Event(
                category=EventCategory.COGNITIVE,
                event_type="act",
            )
        )

        cognitive_events = trace.filter_by_category(EventCategory.COGNITIVE)

        assert len(cognitive_events) == 2

    def test_duration(self):
        """Test duration calculation."""
        trace = ExecutionTrace(execution_id="test")

        t1 = datetime.now(tz=timezone.utc)
        trace.add_event(
            Event(
                category=EventCategory.LIFECYCLE,
                event_type="start",
                timestamp=t1,
            )
        )

        # Simulate time passing
        import time

        time.sleep(0.1)

        t2 = datetime.now(tz=timezone.utc)
        trace.add_event(
            Event(
                category=EventCategory.LIFECYCLE,
                event_type="end",
                timestamp=t2,
            )
        )

        assert trace.duration_seconds is not None
        assert trace.duration_seconds > 0


# =============================================================================
# EXECUTION REPLAYER TESTS
# =============================================================================


class TestExecutionReplayer:
    """Tests for ExecutionReplayer class."""

    @pytest.fixture
    def populated_log(self, tmp_path):
        """Create a log file with test events."""
        log_path = tmp_path / "events.jsonl"

        events = [
            Event(
                category=EventCategory.LIFECYCLE,
                event_type="start",
                execution_id="exec_123",
            ),
            Event(
                category=EventCategory.COGNITIVE,
                event_type="phase_start",
                execution_id="exec_123",
                data={"phase": "THINK"},
            ),
            Event(
                category=EventCategory.COGNITIVE,
                event_type="phase_end",
                execution_id="exec_123",
                data={"phase": "THINK"},
            ),
            Event(
                category=EventCategory.ACTIVITY,
                event_type="tool_call",
                execution_id="exec_123",
                data={"tool": "execute_python"},
            ),
            Event(
                category=EventCategory.LIFECYCLE,
                event_type="end",
                execution_id="exec_123",
            ),
            Event(
                category=EventCategory.LIFECYCLE,
                event_type="start",
                execution_id="exec_456",
            ),
        ]

        with open(log_path, "w") as f:
            for event in events:
                f.write(event.to_jsonl() + "\n")

        return log_path

    def test_load_execution(self, populated_log):
        """Test loading an execution trace."""
        replayer = ExecutionReplayer(populated_log)

        trace = replayer.load_execution("exec_123")

        assert trace is not None
        assert trace.execution_id == "exec_123"
        assert len(trace.events) == 5

    def test_load_nonexistent_execution(self, populated_log):
        """Test loading non-existent execution returns None."""
        replayer = ExecutionReplayer(populated_log)

        trace = replayer.load_execution("nonexistent")

        assert trace is None

    def test_list_executions(self, populated_log):
        """Test listing execution IDs."""
        replayer = ExecutionReplayer(populated_log)

        executions = replayer.list_executions()

        assert "exec_123" in executions
        assert "exec_456" in executions

    def test_replay_with_callbacks(self, populated_log):
        """Test replay with callbacks."""
        replayer = ExecutionReplayer(populated_log)
        trace = replayer.load_execution("exec_123")

        phase_starts = []
        phase_ends = []
        tool_calls = []
        all_events = []

        replayer.replay_step_by_step(
            trace,
            on_phase_start=lambda e: phase_starts.append(e),
            on_phase_end=lambda e: phase_ends.append(e),
            on_tool_call=lambda e: tool_calls.append(e),
            on_event=lambda e: all_events.append(e),
        )

        assert len(phase_starts) == 1
        assert len(phase_ends) == 1
        assert len(tool_calls) == 1
        assert len(all_events) == 5

    def test_caching(self, populated_log):
        """Test execution caching."""
        replayer = ExecutionReplayer(populated_log)

        # First load
        trace1 = replayer.load_execution("exec_123")
        assert replayer.cache_size == 1

        # Second load should use cache
        trace2 = replayer.load_execution("exec_123")
        assert replayer.cache_size == 1

        # Clear cache
        replayer.clear_cache()
        assert replayer.cache_size == 0


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_observability_logger(self, tmp_path):
        """Test creating logger with factory function."""
        obs_logger = create_observability_logger(
            log_path=str(tmp_path / "events.jsonl"),
            min_severity="warning",
            buffer_enabled=True,
            buffer_size=50,
        )

        assert obs_logger._min_severity == Severity.WARNING

    def test_load_events_from_file(self, tmp_path):
        """Test loading events from file."""
        log_path = tmp_path / "events.jsonl"

        # Create test file
        with open(log_path, "w") as f:
            for i in range(10):
                event = Event(
                    category=EventCategory.LIFECYCLE if i < 5 else EventCategory.COGNITIVE,
                    event_type=f"event_{i}",
                    execution_id="exec_test",
                )
                f.write(event.to_jsonl() + "\n")

        # Load all events
        events = load_events_from_file(log_path)
        assert len(events) == 10

        # Load with execution filter
        events = load_events_from_file(log_path, execution_id="exec_test")
        assert len(events) == 10

        # Load with category filter
        events = load_events_from_file(log_path, category=EventCategory.LIFECYCLE)
        assert len(events) == 5

        # Load with limit
        events = load_events_from_file(log_path, limit=3)
        assert len(events) == 3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the event system."""

    def test_full_event_lifecycle(self, tmp_path):
        """Test complete event lifecycle from logging to replay."""
        log_path = tmp_path / "events.jsonl"

        # Create logger
        config = ObservabilityConfig(
            log_path=str(log_path),
            buffer=EventBufferConfig(enabled=False),
        )
        obs_logger = ObservabilityLogger(config)

        # Log events
        exec_id = "integration_test_exec"

        obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="start",
            execution_id=exec_id,
        )

        for phase in ["OBSERVE", "ORIENT", "THINK", "DECIDE", "ACT"]:
            obs_logger.log_event(
                category=EventCategory.COGNITIVE,
                event_type="phase_start",
                execution_id=exec_id,
                data={"phase": phase},
            )
            obs_logger.log_event(
                category=EventCategory.COGNITIVE,
                event_type="phase_end",
                execution_id=exec_id,
                data={"phase": phase},
            )

        obs_logger.log_event(
            category=EventCategory.LIFECYCLE,
            event_type="end",
            execution_id=exec_id,
        )

        obs_logger.close()

        # Replay and verify
        replayer = ExecutionReplayer(log_path)
        trace = replayer.load_execution(exec_id)

        assert trace is not None
        assert len(trace.events) == 12  # 2 lifecycle + 5*2 cognitive

        # Check phases
        cognitive = trace.filter_by_category(EventCategory.COGNITIVE)
        assert len(cognitive) == 10
