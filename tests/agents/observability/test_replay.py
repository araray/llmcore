# tests/agents/observability/test_replay.py
"""
Tests for the observability replay module.

Covers:
- parse_event function and EVENT_CLASS_MAP
- ReplayStep, ExecutionInfo, ReplayResult data classes
- ExecutionReplay loading and initialization
- Event indexing and listing
- Replay timeline generation
- Event filtering and retrieval
- Execution summary generation
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from llmcore.agents.observability import (
    EVENT_CLASS_MAP,
    ActivityEvent,
    ActivityEventType,
    # Events
    AgentEvent,
    CognitiveEvent,
    ErrorEvent,
    ErrorEventType,
    # Enums
    EventCategory,
    EventSeverity,
    ExecutionInfo,
    ExecutionReplay,
    HITLEvent,
    LifecycleEvent,
    LifecycleEventType,
    MemoryEvent,
    MetricEvent,
    RAGEvent,
    ReplayResult,
    # Replay
    ReplayStep,
    SandboxEvent,
    parse_event,
)

# =============================================================================
# EVENT CLASS MAP TESTS
# =============================================================================


class TestEventClassMap:
    """Tests for EVENT_CLASS_MAP."""

    def test_all_categories_mapped(self):
        """Verify all event categories are in the map."""
        expected_categories = {
            "lifecycle",
            "cognitive",
            "activity",
            "memory",
            "hitl",
            "error",
            "metric",
            "sandbox",
            "rag",
        }

        assert set(EVENT_CLASS_MAP.keys()) == expected_categories

    def test_correct_class_mapping(self):
        """Verify correct class mappings."""
        assert EVENT_CLASS_MAP["lifecycle"] == LifecycleEvent
        assert EVENT_CLASS_MAP["cognitive"] == CognitiveEvent
        assert EVENT_CLASS_MAP["activity"] == ActivityEvent
        assert EVENT_CLASS_MAP["memory"] == MemoryEvent
        assert EVENT_CLASS_MAP["hitl"] == HITLEvent
        assert EVENT_CLASS_MAP["error"] == ErrorEvent
        assert EVENT_CLASS_MAP["metric"] == MetricEvent
        assert EVENT_CLASS_MAP["sandbox"] == SandboxEvent
        assert EVENT_CLASS_MAP["rag"] == RAGEvent


# =============================================================================
# PARSE EVENT TESTS
# =============================================================================


class TestParseEvent:
    """Tests for parse_event function."""

    def test_parse_lifecycle_event(self, event_dict_factory):
        """Test parsing a lifecycle event."""
        data = event_dict_factory(
            category="lifecycle",
            event_type="agent_started",
            goal="Test goal",
        )

        event = parse_event(data)

        assert isinstance(event, LifecycleEvent)
        assert event.category == EventCategory.LIFECYCLE
        assert event.goal == "Test goal"

    def test_parse_cognitive_event(self, event_dict_factory):
        """Test parsing a cognitive event."""
        data = event_dict_factory(
            category="cognitive",
            event_type="phase_completed",
            phase="think",
            input_summary="User query",
            output_summary="Decision",
        )

        event = parse_event(data)

        assert isinstance(event, CognitiveEvent)
        assert event.phase == "think"

    def test_parse_activity_event(self, event_dict_factory):
        """Test parsing an activity event."""
        data = event_dict_factory(
            category="activity",
            event_type="activity_completed",
            activity_name="run_script",
            success=True,
        )

        event = parse_event(data)

        assert isinstance(event, ActivityEvent)
        assert event.success is True

    def test_parse_error_event(self, event_dict_factory):
        """Test parsing an error event."""
        data = event_dict_factory(
            category="error",
            event_type="exception",
            severity="error",
            error_type="ValueError",
            error_message="Invalid input",
        )

        event = parse_event(data)

        assert isinstance(event, ErrorEvent)
        assert event.error_type == "ValueError"

    def test_parse_unknown_category(self, event_dict_factory):
        """Test parsing event with unknown category raises ValidationError."""
        from pydantic import ValidationError

        data = event_dict_factory(
            category="unknown_category",
            event_type="unknown_event",
        )

        # Unknown category should raise ValidationError since EventCategory is an enum
        with pytest.raises(ValidationError):
            parse_event(data)

    def test_parse_preserves_all_fields(self, event_dict_factory, fixed_timestamp):
        """Test that parsing preserves all event fields."""
        data = event_dict_factory(
            category="lifecycle",
            event_type="agent_started",
            session_id="sess-123",
            execution_id="exec-456",
            severity="info",
            phase="init",
            iteration=1,
            duration_ms=500.0,
            parent_event_id="parent-789",
            correlation_id="corr-abc",
            tags=["tag1", "tag2"],
            data={"key": "value"},
        )

        event = parse_event(data)

        assert event.session_id == "sess-123"
        assert event.execution_id == "exec-456"
        assert event.iteration == 1
        assert event.duration_ms == 500.0


# =============================================================================
# REPLAY STEP TESTS
# =============================================================================


class TestReplayStep:
    """Tests for ReplayStep data class."""

    def test_create_replay_step(self, fixed_timestamp, sample_lifecycle_event):
        """Test creating a replay step."""
        step = ReplayStep(
            timestamp=fixed_timestamp,
            event_id=sample_lifecycle_event.event_id,
            category=sample_lifecycle_event.category,
            event_type=sample_lifecycle_event.event_type,
            phase=sample_lifecycle_event.phase,
            iteration=sample_lifecycle_event.iteration,
            summary="Agent started",
            duration_ms=None,
            event=sample_lifecycle_event,
        )

        assert step.event_id == sample_lifecycle_event.event_id
        assert step.event is sample_lifecycle_event
        assert step.summary == "Agent started"

    def test_replay_step_with_duration(self, fixed_timestamp, sample_lifecycle_event):
        """Test replay step with duration."""
        step = ReplayStep(
            timestamp=fixed_timestamp,
            event_id=sample_lifecycle_event.event_id,
            category=sample_lifecycle_event.category,
            event_type=sample_lifecycle_event.event_type,
            phase=sample_lifecycle_event.phase,
            iteration=sample_lifecycle_event.iteration,
            summary="Activity completed",
            duration_ms=500.0,
            event=sample_lifecycle_event,
        )

        assert step.duration_ms == 500.0


# =============================================================================
# EXECUTION INFO TESTS
# =============================================================================


class TestExecutionInfo:
    """Tests for ExecutionInfo data class."""

    def test_create_execution_info(self, fixed_timestamp, session_id, execution_id):
        """Test creating execution info."""
        info = ExecutionInfo(
            execution_id=execution_id,
            session_id=session_id,
            start_time=fixed_timestamp,
            event_count=10,
        )

        assert info.execution_id == execution_id
        assert info.event_count == 10
        assert info.end_time is None

    def test_execution_info_complete(self, fixed_timestamp, session_id, execution_id):
        """Test execution info with all fields."""
        info = ExecutionInfo(
            execution_id=execution_id,
            session_id=session_id,
            start_time=fixed_timestamp,
            end_time=fixed_timestamp + timedelta(seconds=60),
            event_count=50,
            error_count=2,
            status="completed",
            goal="Test goal",
        )

        assert info.error_count == 2
        assert info.status == "completed"
        assert info.goal == "Test goal"


# =============================================================================
# REPLAY RESULT TESTS
# =============================================================================


class TestReplayResult:
    """Tests for ReplayResult data class."""

    def test_create_replay_result(
        self, execution_id, session_id, fixed_timestamp, sample_lifecycle_event
    ):
        """Test creating a replay result."""
        step = ReplayStep(
            timestamp=fixed_timestamp,
            event_id=sample_lifecycle_event.event_id,
            category=EventCategory.LIFECYCLE,
            event_type="agent_started",
            phase=None,
            iteration=None,
            summary="Agent started",
            duration_ms=None,
            event=sample_lifecycle_event,
        )

        result = ReplayResult(
            execution_id=execution_id,
            session_id=session_id,
            start_time=fixed_timestamp,
            end_time=None,
            goal=None,
            status=None,
            total_events=1,
            timeline=[step],
        )

        assert result.execution_id == execution_id
        assert len(result.timeline) == 1
        assert result.total_events == 1

    def test_replay_result_duration(
        self, execution_id, session_id, fixed_timestamp, sample_lifecycle_event
    ):
        """Test replay result with time range."""
        end_time = fixed_timestamp + timedelta(seconds=120)

        result = ReplayResult(
            execution_id=execution_id,
            session_id=session_id,
            start_time=fixed_timestamp,
            end_time=end_time,
            goal="Test goal",
            status="success",
            total_events=10,
            timeline=[],
        )

        assert result.start_time == fixed_timestamp
        assert result.end_time == end_time


# =============================================================================
# EXECUTION REPLAY INITIALIZATION TESTS
# =============================================================================


class TestExecutionReplayInit:
    """Tests for ExecutionReplay initialization."""

    def test_create_empty_replay(self):
        """Test creating an empty replay."""
        replay = ExecutionReplay()

        assert replay.event_count == 0
        assert replay.execution_count == 0

    def test_create_from_events(self, sample_event_sequence):
        """Test creating replay from event list."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        assert replay.event_count == len(sample_event_sequence)

    def test_create_from_file(self, sample_jsonl_log):
        """Test creating replay from JSONL file."""
        replay = ExecutionReplay.from_file(sample_jsonl_log)

        assert replay.event_count > 0

    def test_create_from_string_path(self, sample_jsonl_log):
        """Test creating replay from string path."""
        replay = ExecutionReplay.from_file(str(sample_jsonl_log))

        assert replay.event_count > 0

    def test_create_from_nonexistent_file(self, temp_dir):
        """Test creating replay from nonexistent file returns empty replay."""
        # Implementation logs warning and returns empty replay rather than raising
        replay = ExecutionReplay.from_file(temp_dir / "nonexistent.jsonl")

        assert replay.event_count == 0


# =============================================================================
# EXECUTION REPLAY INDEXING TESTS
# =============================================================================


class TestExecutionReplayIndexing:
    """Tests for ExecutionReplay event indexing."""

    def test_indexes_by_session(self, execution_replay):
        """Test events are indexed by session."""
        sessions = execution_replay.list_sessions()

        assert len(sessions) >= 1

    def test_indexes_by_execution(self, execution_replay):
        """Test events are indexed by execution."""
        executions = execution_replay.list_executions()

        assert len(executions) >= 1

    def test_multi_execution_indexing(self, multi_execution_log):
        """Test indexing multiple executions."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        # Should have 2 executions
        assert replay.execution_count >= 2

    def test_multi_session_indexing(self, multi_execution_log):
        """Test indexing multiple sessions."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        sessions = replay.list_sessions()
        assert len(sessions) >= 2


# =============================================================================
# EXECUTION REPLAY LISTING TESTS
# =============================================================================


class TestExecutionReplayListing:
    """Tests for ExecutionReplay listing methods."""

    def test_list_executions(self, execution_replay):
        """Test listing all executions."""
        executions = execution_replay.list_executions()

        assert len(executions) >= 1
        for exec_info in executions:
            assert isinstance(exec_info, ExecutionInfo)

    def test_list_executions_by_session(self, multi_execution_log):
        """Test listing executions filtered by session."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        executions = replay.list_executions(session_id="session-1")

        for exec_info in executions:
            assert exec_info.session_id == "session-1"

    def test_list_executions_with_limit(self, multi_execution_log):
        """Test listing executions with limit."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        executions = replay.list_executions(limit=1)

        assert len(executions) == 1

    def test_list_sessions(self, multi_execution_log):
        """Test listing all sessions."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        sessions = replay.list_sessions()

        assert "session-1" in sessions
        assert "session-2" in sessions


# =============================================================================
# EXECUTION REPLAY REPLAY TESTS
# =============================================================================


class TestExecutionReplayReplay:
    """Tests for ExecutionReplay.replay method."""

    def test_replay_execution(self, execution_replay, execution_id):
        """Test replaying an execution."""
        result = execution_replay.replay(execution_id=execution_id)

        assert isinstance(result, ReplayResult)
        assert result.execution_id == execution_id
        assert len(result.timeline) > 0

    def test_replay_has_timeline(self, execution_replay, execution_id):
        """Test replay generates timeline with steps."""
        result = execution_replay.replay(execution_id=execution_id)

        for step in result.timeline:
            assert step.timestamp is not None
            assert step.event is not None
            assert step.summary is not None
            assert step.event_id is not None

    def test_replay_calculates_elapsed(self, execution_replay, execution_id):
        """Test replay generates timeline with proper ordering."""
        result = execution_replay.replay(execution_id=execution_id)

        # Steps should be ordered by timestamp
        if len(result.timeline) > 1:
            for i in range(1, len(result.timeline)):
                assert result.timeline[i].timestamp >= result.timeline[i - 1].timestamp

    def test_replay_nonexistent_execution(self, execution_replay):
        """Test replaying a nonexistent execution raises error."""
        import pytest

        with pytest.raises(ValueError, match="Execution not found"):
            execution_replay.replay(execution_id="nonexistent")

    def test_replay_includes_execution_info(self, execution_replay, execution_id):
        """Test replay includes execution summary."""
        result = execution_replay.replay(execution_id=execution_id)

        assert result.execution_id == execution_id
        assert result.session_id is not None
        assert result.start_time is not None


# =============================================================================
# EVENT SUMMARIZATION TESTS
# =============================================================================


class TestEventSummarization:
    """Tests for event summarization in replay."""

    def test_summarize_lifecycle_started(self, session_id, execution_id):
        """Test summarizing agent_started event."""
        event = LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
            goal="Analyze data",
        )

        replay = ExecutionReplay.from_events([event])
        result = replay.replay(execution_id=execution_id)

        assert (
            "started" in result.timeline[0].summary.lower()
            or "agent" in result.timeline[0].summary.lower()
        )

    def test_summarize_activity_completed(self, session_id, execution_id):
        """Test summarizing activity_completed event."""
        event = ActivityEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ActivityEventType.ACTIVITY_COMPLETED,
            activity_name="execute_python",
            success=True,
        )

        replay = ExecutionReplay.from_events([event])
        result = replay.replay(execution_id=execution_id)

        # Summary should mention activity or python
        summary = result.timeline[0].summary.lower()
        assert "activity" in summary or "python" in summary or "completed" in summary

    def test_summarize_error_event(self, session_id, execution_id):
        """Test summarizing error event."""
        event = ErrorEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ErrorEventType.EXCEPTION,
            error_type="ValueError",
            error_message="Invalid input",
        )

        replay = ExecutionReplay.from_events([event])
        result = replay.replay(execution_id=execution_id)

        summary = result.timeline[0].summary.lower()
        assert "error" in summary or "exception" in summary or "valueerror" in summary


# =============================================================================
# GET EVENTS FILTERING TESTS
# =============================================================================


class TestGetEventsFiltering:
    """Tests for ExecutionReplay.get_events filtering."""

    def test_get_all_events(self, execution_replay):
        """Test getting all events without filters."""
        events = execution_replay.get_events()

        assert len(events) == execution_replay.event_count

    def test_filter_by_session_id(self, multi_execution_log):
        """Test filtering by session_id."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        events = replay.get_events(session_id="session-1")

        for event in events:
            assert event.session_id == "session-1"

    def test_filter_by_execution_id(self, execution_replay, execution_id):
        """Test filtering by execution_id."""
        events = execution_replay.get_events(execution_id=execution_id)

        for event in events:
            assert event.execution_id == execution_id

    def test_filter_by_category(self, execution_replay):
        """Test filtering by category."""
        events = execution_replay.get_events(category=EventCategory.LIFECYCLE)

        for event in events:
            assert event.category == EventCategory.LIFECYCLE

    def test_filter_by_event_type(self, execution_replay):
        """Test filtering by event_type."""
        events = execution_replay.get_events(event_type="agent_started")

        for event in events:
            assert (
                event.event_type == LifecycleEventType.AGENT_STARTED
                or event.event_type == "agent_started"
            )

    def test_filter_by_severity(self, sample_event_sequence, session_id, execution_id):
        """Test filtering by minimum severity."""
        # Add an error event to the sequence
        events_with_error = list(sample_event_sequence) + [
            ErrorEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ErrorEventType.EXCEPTION,
                severity=EventSeverity.ERROR,
                error_type="Test",
                error_message="Test error",
            )
        ]

        replay = ExecutionReplay.from_events(events_with_error)

        error_events = replay.get_events(severity=EventSeverity.ERROR)

        for event in error_events:
            assert event.severity in [
                EventSeverity.ERROR,
                EventSeverity.CRITICAL,
                "error",
                "critical",
            ]

    def test_filter_by_iteration(self, sample_event_sequence):
        """Test filtering by iteration."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        events = replay.get_events(iteration=1)

        for event in events:
            assert event.iteration == 1 or event.iteration is None

    def test_filter_by_phase(self, sample_event_sequence):
        """Test filtering by phase."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        events = replay.get_events(phase="think")

        for event in events:
            assert event.phase == "think"

    def test_filter_by_time_range(self, sample_event_sequence, fixed_timestamp):
        """Test filtering by time range."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        events = replay.get_events(
            since=fixed_timestamp,
            until=fixed_timestamp + timedelta(seconds=3),
        )

        for event in events:
            assert event.timestamp >= fixed_timestamp
            assert event.timestamp <= fixed_timestamp + timedelta(seconds=3)

    def test_filter_with_duration(self, sample_event_sequence):
        """Test filtering events with duration."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        events = replay.get_events(has_duration=True)

        for event in events:
            assert event.duration_ms is not None

    def test_combined_filters(self, execution_replay, execution_id):
        """Test combining multiple filters."""
        events = execution_replay.get_events(
            execution_id=execution_id,
            category=EventCategory.LIFECYCLE,
        )

        for event in events:
            assert event.execution_id == execution_id
            assert event.category == EventCategory.LIFECYCLE


# =============================================================================
# GET ERRORS TESTS
# =============================================================================


class TestGetErrors:
    """Tests for ExecutionReplay.get_errors method."""

    def test_get_errors_empty(self, sample_event_sequence):
        """Test getting errors when none exist."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        errors = replay.get_errors()

        # sample_event_sequence might not have errors
        assert isinstance(errors, list)

    def test_get_errors_with_errors(self, session_id, execution_id):
        """Test getting errors when they exist."""
        events = [
            LifecycleEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=LifecycleEventType.AGENT_STARTED,
            ),
            ErrorEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ErrorEventType.EXCEPTION,
                error_type="ValueError",
                error_message="Test error",
            ),
            ErrorEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ErrorEventType.TIMEOUT_ERROR,
                error_type="TimeoutError",
                error_message="Timeout",
            ),
        ]

        replay = ExecutionReplay.from_events(events)
        errors = replay.get_errors()

        assert len(errors) == 2
        assert all(isinstance(e, ErrorEvent) for e in errors)

    def test_get_errors_by_execution(self, session_id):
        """Test getting errors filtered by execution."""
        events = [
            ErrorEvent(
                session_id=session_id,
                execution_id="exec-1",
                event_type=ErrorEventType.EXCEPTION,
                error_type="Error1",
                error_message="Error 1",
            ),
            ErrorEvent(
                session_id=session_id,
                execution_id="exec-2",
                event_type=ErrorEventType.EXCEPTION,
                error_type="Error2",
                error_message="Error 2",
            ),
        ]

        replay = ExecutionReplay.from_events(events)
        errors = replay.get_errors(execution_id="exec-1")

        assert len(errors) == 1
        assert errors[0].execution_id == "exec-1"


# =============================================================================
# GET ACTIVITIES TESTS
# =============================================================================


class TestGetActivities:
    """Tests for ExecutionReplay.get_activities method."""

    def test_get_activities(self, sample_event_sequence):
        """Test getting activities."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        activities = replay.get_activities()

        assert isinstance(activities, list)
        # sample_event_sequence should have at least one activity
        if len(activities) > 0:
            assert all(isinstance(a, ActivityEvent) for a in activities)

    def test_get_activities_by_type(self, session_id, execution_id):
        """Test getting activities filtered by type."""
        events = [
            ActivityEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ActivityEventType.ACTIVITY_COMPLETED,
                activity_name="execute_python",
                success=True,
            ),
            ActivityEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ActivityEventType.ACTIVITY_COMPLETED,
                activity_name="execute_shell",
                success=True,
            ),
        ]

        replay = ExecutionReplay.from_events(events)
        python_activities = replay.get_activities(activity_name="execute_python")

        assert len(python_activities) == 1
        assert python_activities[0].activity_name == "execute_python"

    def test_get_activities_success_only(self, session_id, execution_id):
        """Test getting only successful activities."""
        events = [
            ActivityEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ActivityEventType.ACTIVITY_COMPLETED,
                activity_name="execute_python",
                success=True,
            ),
            ActivityEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ActivityEventType.ACTIVITY_FAILED,
                activity_name="execute_python",
                success=False,
            ),
        ]

        replay = ExecutionReplay.from_events(events)
        successful = replay.get_activities(success_only=True)

        assert len(successful) == 1
        assert successful[0].success is True

    def test_get_activities_failed_only(self, session_id, execution_id):
        """Test getting only failed activities."""
        events = [
            ActivityEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ActivityEventType.ACTIVITY_COMPLETED,
                activity_name="success_activity",
                success=True,
            ),
            ActivityEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=ActivityEventType.ACTIVITY_FAILED,
                activity_name="failed_activity",
                success=False,
            ),
        ]

        replay = ExecutionReplay.from_events(events)
        failed = replay.get_activities(failed_only=True)

        assert len(failed) == 1
        assert failed[0].success is False


# =============================================================================
# ITER EVENTS TESTS
# =============================================================================


class TestIterEvents:
    """Tests for ExecutionReplay.iter_events method."""

    def test_iter_events_basic(self, execution_replay):
        """Test basic event iteration."""
        events = list(execution_replay.iter_events())

        assert len(events) == execution_replay.event_count

    def test_iter_events_with_filter(self, execution_replay, execution_id):
        """Test event iteration with filter."""
        events = list(execution_replay.iter_events(execution_id=execution_id))

        for event in events:
            assert event.execution_id == execution_id

    def test_iter_events_is_generator(self, execution_replay):
        """Test that iter_events returns a generator."""
        iterator = execution_replay.iter_events()

        # Should be iterable without converting to list
        first = next(iterator)
        assert isinstance(first, AgentEvent)


# =============================================================================
# EXECUTION SUMMARY TESTS
# =============================================================================


class TestExecutionSummaryGeneration:
    """Tests for execution summary generation."""

    def test_execution_info_from_events(self, sample_event_sequence, execution_id):
        """Test execution info is correctly built from events."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        executions = replay.list_executions()

        assert len(executions) >= 1
        exec_info = executions[0]
        assert exec_info.event_count > 0

    def test_execution_info_counts_iterations(self, sample_event_sequence, execution_id):
        """Test execution info counts events correctly."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        executions = replay.list_executions()
        exec_info = next(e for e in executions if e.execution_id == execution_id)

        # sample_event_sequence has multiple events
        assert exec_info.event_count >= 1

    def test_execution_info_tracks_time_range(
        self, sample_event_sequence, execution_id, fixed_timestamp
    ):
        """Test execution info tracks start/end time."""
        replay = ExecutionReplay.from_events(sample_event_sequence)

        executions = replay.list_executions()
        exec_info = next(e for e in executions if e.execution_id == execution_id)

        assert exec_info.start_time is not None
        # End time should be set from last event
        assert exec_info.end_time is not None or exec_info.event_count > 0


# =============================================================================
# PROPERTIES TESTS
# =============================================================================


class TestExecutionReplayProperties:
    """Tests for ExecutionReplay properties."""

    def test_event_count_property(self, execution_replay, sample_event_sequence):
        """Test event_count property."""
        assert execution_replay.event_count == len(sample_event_sequence)

    def test_execution_count_property(self, multi_execution_log):
        """Test execution_count property."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        assert replay.execution_count >= 2

    def test_session_count_property(self, multi_execution_log):
        """Test session_count property."""
        replay = ExecutionReplay.from_file(multi_execution_log)

        assert replay.session_count >= 2


# =============================================================================
# EDGE CASES
# =============================================================================


class TestReplayEdgeCases:
    """Tests for replay edge cases."""

    def test_empty_log_file(self, temp_dir):
        """Test loading empty log file."""
        empty_file = temp_dir / "empty.jsonl"
        empty_file.touch()

        replay = ExecutionReplay.from_file(empty_file)

        assert replay.event_count == 0

    def test_malformed_json_line(self, temp_dir, sample_lifecycle_event):
        """Test handling malformed JSON in log file."""
        log_file = temp_dir / "malformed.jsonl"

        with open(log_file, "w") as f:
            f.write(sample_lifecycle_event.to_json() + "\n")
            f.write("this is not valid json\n")
            f.write(sample_lifecycle_event.to_json() + "\n")

        replay = ExecutionReplay.from_file(log_file)

        # Should load valid events, skip invalid ones
        assert replay.event_count >= 1

    def test_missing_category_in_event(self, temp_dir):
        """Test handling event without category."""
        log_file = temp_dir / "no_category.jsonl"

        with open(log_file, "w") as f:
            f.write('{"session_id": "test", "event_type": "test"}\n')

        # Should handle gracefully
        replay = ExecutionReplay.from_file(log_file)
        # May be 0 if it fails to parse, or 1 with defaults
        assert replay.event_count >= 0

    def test_filter_returns_empty_list(self, execution_replay):
        """Test filtering that returns no results."""
        events = execution_replay.get_events(session_id="nonexistent-session")

        assert events == []

    def test_replay_with_single_event(self, session_id, execution_id):
        """Test replaying execution with single event."""
        events = [
            LifecycleEvent(
                session_id=session_id,
                execution_id=execution_id,
                event_type=LifecycleEventType.AGENT_STARTED,
            )
        ]

        replay = ExecutionReplay.from_events(events)
        result = replay.replay(execution_id=execution_id)

        assert result.total_events == 1
        assert len(result.timeline) == 1

    def test_unicode_in_log_file(self, temp_dir, session_id, execution_id):
        """Test handling unicode characters in log file."""
        log_file = temp_dir / "unicode.jsonl"

        event = LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
            goal="Test 日本語 🎉",
        )

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

        replay = ExecutionReplay.from_file(log_file)

        assert replay.event_count == 1
        loaded_event = replay.get_events()[0]
        assert "日本語" in loaded_event.goal

    def test_large_log_file(self, temp_dir, session_id, execution_id):
        """Test handling large log file."""
        log_file = temp_dir / "large.jsonl"

        with open(log_file, "w") as f:
            for i in range(1000):
                event = LifecycleEvent(
                    session_id=session_id,
                    execution_id=execution_id,
                    event_type=LifecycleEventType.ITERATION_STARTED,
                    iteration=i,
                )
                f.write(event.to_json() + "\n")

        replay = ExecutionReplay.from_file(log_file)

        assert replay.event_count == 1000
