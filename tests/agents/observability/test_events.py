# tests/agents/observability/test_events.py
"""
Tests for the observability events module.

Covers:
- Enum validation and values
- Base AgentEvent model validation and methods
- All specialized event types
- Factory functions
- Serialization/deserialization
- Edge cases and validation errors
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from llmcore.agents.observability import (
    # Enums
    EventCategory,
    EventSeverity,
    LifecycleEventType,
    CognitiveEventType,
    ActivityEventType,
    HITLEventType,
    ErrorEventType,
    MetricEventType,
    MemoryEventType,
    SandboxEventType,
    RAGEventType,
    # Events
    AgentEvent,
    LifecycleEvent,
    CognitiveEvent,
    ActivityEvent,
    MemoryEvent,
    HITLEvent,
    ErrorEvent,
    MetricEvent,
    SandboxEvent,
    RAGEvent,
    # Factory functions
    create_lifecycle_event,
    create_cognitive_event,
    create_activity_event,
    create_error_event,
    create_metric_event,
    create_hitl_event,
    create_sandbox_event,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestEventCategory:
    """Tests for EventCategory enum."""
    
    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        expected = {
            "LIFECYCLE", "COGNITIVE", "ACTIVITY", "MEMORY",
            "HITL", "ERROR", "METRIC", "SANDBOX", "RAG"
        }
        actual = {cat.name for cat in EventCategory}
        assert actual == expected
    
    def test_category_values_are_lowercase(self):
        """All category values should be lowercase strings."""
        for cat in EventCategory:
            assert cat.value == cat.value.lower()
            assert cat.value == cat.name.lower()
    
    def test_category_is_string_enum(self):
        """EventCategory should be usable as a string."""
        assert EventCategory.LIFECYCLE == "lifecycle"
        assert EventCategory.COGNITIVE.value == "cognitive"


class TestEventSeverity:
    """Tests for EventSeverity enum."""
    
    def test_all_severities_exist(self):
        """Verify all expected severities are defined."""
        expected = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        actual = {sev.name for sev in EventSeverity}
        assert actual == expected
    
    def test_severity_values_are_lowercase(self):
        """All severity values should be lowercase."""
        for sev in EventSeverity:
            assert sev.value == sev.value.lower()
    
    def test_severity_is_string_enum(self):
        """EventSeverity should be usable as a string."""
        assert EventSeverity.INFO == "info"
        assert EventSeverity.ERROR.value == "error"


class TestLifecycleEventType:
    """Tests for LifecycleEventType enum."""
    
    def test_all_lifecycle_types_exist(self):
        """Verify all lifecycle event types."""
        expected = {
            "AGENT_STARTED", "AGENT_COMPLETED", "AGENT_FAILED",
            "AGENT_CANCELLED", "ITERATION_STARTED", "ITERATION_COMPLETED",
            "GOAL_CLASSIFIED", "FAST_PATH_TRIGGERED"
        }
        actual = {t.name for t in LifecycleEventType}
        assert actual == expected


class TestCognitiveEventType:
    """Tests for CognitiveEventType enum."""
    
    def test_all_cognitive_types_exist(self):
        """Verify all cognitive event types."""
        expected = {
            "PHASE_STARTED", "PHASE_COMPLETED", "PHASE_FAILED",
            "PHASE_SKIPPED", "DECISION_MADE", "REASONING_STEP"
        }
        actual = {t.name for t in CognitiveEventType}
        assert actual == expected


class TestActivityEventType:
    """Tests for ActivityEventType enum."""
    
    def test_all_activity_types_exist(self):
        """Verify all activity event types."""
        expected = {
            "ACTIVITY_STARTED", "ACTIVITY_COMPLETED", "ACTIVITY_FAILED",
            "ACTIVITY_TIMEOUT", "ACTIVITY_RETRIED", "ACTIVITY_CANCELLED"
        }
        actual = {t.name for t in ActivityEventType}
        assert actual == expected


class TestHITLEventType:
    """Tests for HITLEventType enum."""
    
    def test_all_hitl_types_exist(self):
        """Verify all HITL event types."""
        expected = {
            "APPROVAL_REQUESTED", "APPROVAL_GRANTED", "APPROVAL_DENIED",
            "APPROVAL_TIMEOUT", "APPROVAL_MODIFIED", "SCOPE_GRANTED",
            "SCOPE_REVOKED"
        }
        actual = {t.name for t in HITLEventType}
        assert actual == expected


class TestErrorEventType:
    """Tests for ErrorEventType enum."""
    
    def test_all_error_types_exist(self):
        """Verify all error event types."""
        expected = {
            "EXCEPTION", "VALIDATION_ERROR", "TIMEOUT_ERROR",
            "RATE_LIMIT_ERROR", "API_ERROR", "SANDBOX_ERROR",
            "RECOVERY_ATTEMPTED", "RECOVERY_FAILED"
        }
        actual = {t.name for t in ErrorEventType}
        assert actual == expected


class TestMetricEventType:
    """Tests for MetricEventType enum."""
    
    def test_all_metric_types_exist(self):
        """Verify all metric event types."""
        expected = {
            "LATENCY", "TOKEN_USAGE", "COST", "THROUGHPUT",
            "CACHE_HIT", "CACHE_MISS"
        }
        actual = {t.name for t in MetricEventType}
        assert actual == expected


class TestMemoryEventType:
    """Tests for MemoryEventType enum."""
    
    def test_all_memory_types_exist(self):
        """Verify all memory event types."""
        expected = {
            "MEMORY_READ", "MEMORY_WRITE", "MEMORY_UPDATE",
            "MEMORY_DELETE", "MEMORY_SEARCH"
        }
        actual = {t.name for t in MemoryEventType}
        assert actual == expected


class TestSandboxEventType:
    """Tests for SandboxEventType enum."""
    
    def test_all_sandbox_types_exist(self):
        """Verify all sandbox event types."""
        expected = {
            "SANDBOX_CREATED", "SANDBOX_STARTED", "SANDBOX_STOPPED",
            "SANDBOX_DESTROYED", "CODE_EXECUTED", "FILE_OPERATION"
        }
        actual = {t.name for t in SandboxEventType}
        assert actual == expected


class TestRAGEventType:
    """Tests for RAGEventType enum."""
    
    def test_all_rag_types_exist(self):
        """Verify all RAG event types."""
        expected = {
            "QUERY_STARTED", "QUERY_COMPLETED",
            "DOCUMENTS_RETRIEVED", "CONTEXT_ASSEMBLED"
        }
        actual = {t.name for t in RAGEventType}
        assert actual == expected


# =============================================================================
# BASE AGENT EVENT TESTS
# =============================================================================


class TestAgentEvent:
    """Tests for base AgentEvent model."""
    
    def test_create_minimal_event(self, session_id, execution_id):
        """Create event with only required fields."""
        event = AgentEvent(
            session_id=session_id,
            execution_id=execution_id,
            category=EventCategory.LIFECYCLE,
            event_type="test_event",
        )
        
        assert event.session_id == session_id
        assert event.execution_id == execution_id
        assert event.category == EventCategory.LIFECYCLE
        assert event.event_type == "test_event"
        assert event.severity == EventSeverity.INFO  # default
        assert event.event_id.startswith("evt-")
        assert event.timestamp is not None
        assert event.data == {}
        assert event.tags == []
    
    def test_event_id_auto_generated(self, session_id):
        """Event IDs should be auto-generated and unique."""
        event1 = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        event2 = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        assert event1.event_id.startswith("evt-")
        assert event2.event_id.startswith("evt-")
        assert event1.event_id != event2.event_id
    
    def test_timestamp_auto_generated(self, session_id):
        """Timestamps should be auto-generated in UTC."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        assert event.timestamp.tzinfo == timezone.utc
        # Should be within last few seconds
        delta = datetime.now(timezone.utc) - event.timestamp
        assert delta.total_seconds() < 5
    
    def test_custom_timestamp(self, session_id, fixed_timestamp):
        """Allow custom timestamp to be set."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            timestamp=fixed_timestamp,
        )
        
        assert event.timestamp == fixed_timestamp
    
    def test_optional_fields_none_by_default(self, session_id):
        """Optional fields should default to None."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        assert event.execution_id is None
        assert event.phase is None
        assert event.iteration is None
        assert event.duration_ms is None
        assert event.parent_event_id is None
        assert event.correlation_id is None
    
    def test_full_event_creation(self, session_id, execution_id, correlation_id):
        """Create event with all fields populated."""
        event = AgentEvent(
            session_id=session_id,
            execution_id=execution_id,
            category=EventCategory.ACTIVITY,
            event_type="activity_completed",
            severity=EventSeverity.WARNING,
            phase="act",
            iteration=3,
            data={"key": "value", "nested": {"a": 1}},
            duration_ms=150.5,
            parent_event_id="parent-123",
            correlation_id=correlation_id,
            tags=["tag1", "tag2"],
        )
        
        assert event.severity == EventSeverity.WARNING
        assert event.phase == "act"
        assert event.iteration == 3
        assert event.data == {"key": "value", "nested": {"a": 1}}
        assert event.duration_ms == 150.5
        assert event.parent_event_id == "parent-123"
        assert event.correlation_id == correlation_id
        assert event.tags == ["tag1", "tag2"]
    
    def test_with_duration(self, session_id):
        """Test with_duration method calculates duration correctly."""
        start_time = datetime.now(timezone.utc) - timedelta(milliseconds=500)
        
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        result = event.with_duration(start_time)
        
        # Should return self for chaining
        assert result is event
        # Duration should be approximately 500ms (allow some tolerance)
        assert event.duration_ms is not None
        assert 400 < event.duration_ms < 700
    
    def test_with_parent(self, session_id):
        """Test with_parent method sets parent_event_id."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        result = event.with_parent("parent-abc")
        
        assert result is event
        assert event.parent_event_id == "parent-abc"
    
    def test_with_correlation(self, session_id, correlation_id):
        """Test with_correlation method sets correlation_id."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        result = event.with_correlation(correlation_id)
        
        assert result is event
        assert event.correlation_id == correlation_id
    
    def test_add_tag(self, session_id):
        """Test add_tag method adds tags."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        )
        
        result = event.add_tag("tag1")
        assert result is event
        assert "tag1" in event.tags
        
        # Adding same tag again should not duplicate
        event.add_tag("tag1")
        assert event.tags.count("tag1") == 1
        
        # Adding different tag should work
        event.add_tag("tag2")
        assert "tag2" in event.tags
    
    def test_method_chaining(self, session_id, correlation_id):
        """Test that helper methods can be chained."""
        start_time = datetime.now(timezone.utc) - timedelta(milliseconds=100)
        
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
        ).with_parent("parent-1").with_correlation(correlation_id).with_duration(start_time).add_tag("chained")
        
        assert event.parent_event_id == "parent-1"
        assert event.correlation_id == correlation_id
        assert event.duration_ms is not None
        assert "chained" in event.tags
    
    def test_to_dict(self, sample_agent_event):
        """Test to_dict serialization."""
        result = sample_agent_event.to_dict()
        
        assert isinstance(result, dict)
        assert result["session_id"] == sample_agent_event.session_id
        assert result["category"] == "lifecycle"
        assert result["event_type"] == "test_event"
        assert "timestamp" in result
        assert "event_id" in result
    
    def test_to_json(self, sample_agent_event):
        """Test to_json serialization."""
        result = sample_agent_event.to_json()
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["session_id"] == sample_agent_event.session_id
        assert parsed["category"] == "lifecycle"
    
    def test_json_round_trip(self, sample_agent_event):
        """Test that JSON serialization round-trips correctly."""
        json_str = sample_agent_event.to_json()
        data = json.loads(json_str)
        
        # Should be able to recreate the event
        recreated = AgentEvent(**data)
        assert recreated.session_id == sample_agent_event.session_id
        assert recreated.category == sample_agent_event.category
        assert recreated.event_type == sample_agent_event.event_type
    
    def test_iteration_validation(self, session_id):
        """Test iteration must be non-negative."""
        # Valid iteration
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            iteration=0,
        )
        assert event.iteration == 0
        
        # Invalid iteration
        with pytest.raises(ValidationError):
            AgentEvent(
                session_id=session_id,
                category=EventCategory.LIFECYCLE,
                event_type="test",
                iteration=-1,
            )
    
    def test_duration_validation(self, session_id):
        """Test duration_ms must be non-negative."""
        # Valid duration
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            duration_ms=0.0,
        )
        assert event.duration_ms == 0.0
        
        # Invalid duration
        with pytest.raises(ValidationError):
            AgentEvent(
                session_id=session_id,
                category=EventCategory.LIFECYCLE,
                event_type="test",
                duration_ms=-1.0,
            )
    
    def test_extra_fields_allowed(self, session_id):
        """Test that extra fields are allowed (pydantic extra='allow')."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            custom_field="custom_value",
        )
        
        assert event.custom_field == "custom_value"


# =============================================================================
# LIFECYCLE EVENT TESTS
# =============================================================================


class TestLifecycleEvent:
    """Tests for LifecycleEvent model."""
    
    def test_create_lifecycle_event(self, session_id, execution_id):
        """Create basic lifecycle event."""
        event = LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
            goal="Test goal",
        )
        
        assert event.category == EventCategory.LIFECYCLE
        assert event.event_type == LifecycleEventType.AGENT_STARTED
        assert event.goal == "Test goal"
    
    def test_category_is_frozen(self, session_id):
        """Category should be immutable for LifecycleEvent."""
        event = LifecycleEvent(
            session_id=session_id,
            event_type=LifecycleEventType.AGENT_STARTED,
        )
        
        assert event.category == EventCategory.LIFECYCLE
        # Cannot change category via assignment after creation (frozen field)
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            event.category = EventCategory.COGNITIVE
    
    def test_completion_event_fields(self, session_id, execution_id):
        """Test completion event with all status fields."""
        event = LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_COMPLETED,
            final_status="success",
            total_iterations=5,
            total_tokens=15000,
            exit_reason="Goal achieved",
        )
        
        assert event.final_status == "success"
        assert event.total_iterations == 5
        assert event.total_tokens == 15000
        assert event.exit_reason == "Goal achieved"
    
    def test_goal_classification_fields(self, session_id):
        """Test goal classification event fields."""
        event = LifecycleEvent(
            session_id=session_id,
            event_type=LifecycleEventType.GOAL_CLASSIFIED,
            goal="Complex task",
            goal_complexity="complex",
            recommended_strategy="full_cycle",
        )
        
        assert event.goal_complexity == "complex"
        assert event.recommended_strategy == "full_cycle"
    
    def test_lifecycle_serialization(self, sample_lifecycle_event):
        """Test lifecycle event serialization."""
        data = sample_lifecycle_event.to_dict()
        
        assert data["category"] == "lifecycle"
        assert data["event_type"] == "agent_started"
        assert "goal" in data


# =============================================================================
# COGNITIVE EVENT TESTS
# =============================================================================


class TestCognitiveEvent:
    """Tests for CognitiveEvent model."""
    
    def test_create_cognitive_event(self, session_id, execution_id):
        """Create basic cognitive event."""
        event = CognitiveEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=CognitiveEventType.PHASE_COMPLETED,
            phase="think",
            iteration=1,
        )
        
        assert event.category == EventCategory.COGNITIVE
        assert event.event_type == CognitiveEventType.PHASE_COMPLETED
        assert event.phase == "think"
    
    def test_cognitive_phase_fields(self, session_id, execution_id):
        """Test cognitive phase with all fields."""
        event = CognitiveEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=CognitiveEventType.PHASE_COMPLETED,
            phase="think",
            iteration=2,
            input_summary="User asked about data processing",
            output_summary="Decided to use pandas library",
            tokens_used=500,
            reasoning="Analysis of user requirements",
            confidence=0.85,
            phase_name="thinking",
            phase_order=2,
            decisions=["use_pandas", "read_csv"],
        )
        
        assert event.input_summary == "User asked about data processing"
        assert event.output_summary == "Decided to use pandas library"
        assert event.tokens_used == 500
        assert event.reasoning == "Analysis of user requirements"
        assert event.confidence == 0.85
        assert event.decisions == ["use_pandas", "read_csv"]
    
    def test_category_is_frozen(self, session_id):
        """Category should be immutable for CognitiveEvent."""
        event = CognitiveEvent(
            session_id=session_id,
            event_type=CognitiveEventType.PHASE_STARTED,
        )
        
        assert event.category == EventCategory.COGNITIVE


# =============================================================================
# ACTIVITY EVENT TESTS
# =============================================================================


class TestActivityEvent:
    """Tests for ActivityEvent model."""
    
    def test_create_activity_event(self, session_id, execution_id):
        """Create basic activity event."""
        event = ActivityEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ActivityEventType.ACTIVITY_COMPLETED,
            activity_type="execute_python",
            activity_name="execute_python",
        )
        
        assert event.category == EventCategory.ACTIVITY
        assert event.activity_type == "execute_python"
    
    def test_activity_full_fields(self, session_id, execution_id):
        """Test activity event with all fields."""
        event = ActivityEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ActivityEventType.ACTIVITY_COMPLETED,
            activity_type="execute_python",
            activity_name="run_analysis",
            parameters={"code": "import pandas", "timeout": 30},
            result="Success: module imported",
            success=True,
            duration_ms=150.0,
            error_message=None,
            retry_count=0,
            sandbox_id="sandbox-123",
        )
        
        assert event.parameters == {"code": "import pandas", "timeout": 30}
        assert event.result == "Success: module imported"
        assert event.success is True
        assert event.retry_count == 0
        assert event.sandbox_id == "sandbox-123"
    
    def test_failed_activity(self, session_id, execution_id):
        """Test activity event for failed execution."""
        event = ActivityEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ActivityEventType.ACTIVITY_FAILED,
            activity_type="execute_shell",
            activity_name="execute_shell",
            success=False,
            error_message="Permission denied",
            retry_count=3,
        )
        
        assert event.success is False
        assert event.error_message == "Permission denied"


# =============================================================================
# MEMORY EVENT TESTS
# =============================================================================


class TestMemoryEvent:
    """Tests for MemoryEvent model."""
    
    def test_create_memory_event(self, session_id, execution_id):
        """Create basic memory event."""
        event = MemoryEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=MemoryEventType.MEMORY_WRITE,
            memory_type="short_term",
            operation="write",
        )
        
        assert event.category == EventCategory.MEMORY
        assert event.memory_type == "short_term"
    
    def test_memory_full_fields(self, session_id, execution_id):
        """Test memory event with all fields."""
        event = MemoryEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=MemoryEventType.MEMORY_WRITE,
            memory_type="long_term",
            operation="write",
            key="user_preferences",
            value_summary="User prefers Python",
            size_bytes=256,
            hit=True,
            ttl_seconds=3600,
        )
        
        assert event.key == "user_preferences"
        assert event.value_summary == "User prefers Python"
        assert event.size_bytes == 256
        assert event.hit is True
        assert event.ttl_seconds == 3600


# =============================================================================
# HITL EVENT TESTS
# =============================================================================


class TestHITLEvent:
    """Tests for HITLEvent model."""
    
    def test_create_hitl_event(self, session_id, execution_id):
        """Create basic HITL event."""
        event = HITLEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=HITLEventType.APPROVAL_REQUESTED,
            request_id="req-12345",
            action_type="execute_shell",
            risk_level="high",
            approval_status="pending",
        )
        
        assert event.category == EventCategory.HITL
        assert event.request_id == "req-12345"
        assert event.risk_level == "high"
        assert event.approval_status == "pending"
    
    def test_hitl_full_fields(self, session_id, execution_id):
        """Test HITL event with all fields."""
        event = HITLEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=HITLEventType.APPROVAL_GRANTED,
            request_id="req-12345",
            action_type="execute_shell",
            risk_level="high",
            risk_factors=["shell_access", "file_write"],
            approval_status="approved",
            responder_id="user@example.com",
            feedback="Needed for data processing",
            scope_granted="full_shell_access",
            timeout_seconds=300,
        )
        
        assert event.risk_level == "high"
        assert event.risk_factors == ["shell_access", "file_write"]
        assert event.approval_status == "approved"
        assert event.responder_id == "user@example.com"


# =============================================================================
# ERROR EVENT TESTS
# =============================================================================


class TestErrorEvent:
    """Tests for ErrorEvent model."""
    
    def test_create_error_event(self, session_id, execution_id):
        """Create basic error event."""
        event = ErrorEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ErrorEventType.EXCEPTION,
            severity=EventSeverity.ERROR,
            error_type="ValueError",
            error_message="Invalid input",
        )
        
        assert event.category == EventCategory.ERROR
        assert event.severity == EventSeverity.ERROR
        assert event.error_type == "ValueError"
    
    def test_error_full_fields(self, session_id, execution_id):
        """Test error event with all fields."""
        event = ErrorEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ErrorEventType.EXCEPTION,
            severity=EventSeverity.CRITICAL,
            error_type="RuntimeError",
            error_message="Process crashed",
            error_code="E500",
            stack_trace="Traceback...",
            recoverable=False,
            recovery_action="restart",
            context={"module": "sandbox", "operation": "execute"},
        )
        
        assert event.error_code == "E500"
        assert event.stack_trace == "Traceback..."
        assert event.recoverable is False
        assert event.context == {"module": "sandbox", "operation": "execute"}


# =============================================================================
# METRIC EVENT TESTS
# =============================================================================


class TestMetricEvent:
    """Tests for MetricEvent model."""
    
    def test_create_metric_event(self, session_id, execution_id):
        """Create basic metric event."""
        event = MetricEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=MetricEventType.TOKEN_USAGE,
            metric_name="input_tokens",
            metric_value=1500.0,
        )
        
        assert event.category == EventCategory.METRIC
        assert event.metric_name == "input_tokens"
        assert event.metric_value == 1500.0
    
    def test_metric_full_fields(self, session_id, execution_id):
        """Test metric event with all fields."""
        event = MetricEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=MetricEventType.LATENCY,
            metric_name="llm_call_duration",
            metric_value=1250.5,
            metric_unit="ms",
            metric_context="gpt-4 completion",
            dimensions={"model": "gpt-4", "operation": "completion"},
        )
        
        assert event.metric_unit == "ms"
        assert event.metric_context == "gpt-4 completion"
        assert event.dimensions == {"model": "gpt-4", "operation": "completion"}


# =============================================================================
# SANDBOX EVENT TESTS
# =============================================================================


class TestSandboxEvent:
    """Tests for SandboxEvent model."""
    
    def test_create_sandbox_event(self, session_id, execution_id):
        """Create basic sandbox event."""
        event = SandboxEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=SandboxEventType.SANDBOX_CREATED,
            sandbox_id="sandbox-abc123",
            sandbox_type="docker",
            operation="create",
        )
        
        assert event.category == EventCategory.SANDBOX
        assert event.sandbox_id == "sandbox-abc123"
        assert event.sandbox_type == "docker"
        assert event.operation == "create"
    
    def test_sandbox_full_fields(self, session_id, execution_id):
        """Test sandbox event with all fields."""
        event = SandboxEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=SandboxEventType.CODE_EXECUTED,
            sandbox_id="sandbox-abc123",
            sandbox_type="docker",
            operation="execute",
            image="python:3.11",
            access_mode="restricted",
            exit_code=0,
            stdout_length=1024,
            stderr_length=0,
            memory_used_mb=256.5,
            cpu_time_ms=150.0,
        )
        
        assert event.image == "python:3.11"
        assert event.operation == "execute"
        assert event.exit_code == 0
        assert event.memory_used_mb == 256.5


# =============================================================================
# RAG EVENT TESTS
# =============================================================================


class TestRAGEvent:
    """Tests for RAGEvent model."""
    
    def test_create_rag_event(self, session_id, execution_id):
        """Create basic RAG event."""
        event = RAGEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=RAGEventType.QUERY_STARTED,
            query="What is machine learning?",
            source="vector_store",
        )
        
        assert event.category == EventCategory.RAG
        assert event.query == "What is machine learning?"
        assert event.source == "vector_store"
    
    def test_rag_full_fields(self, session_id, execution_id):
        """Test RAG event with all fields."""
        event = RAGEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=RAGEventType.DOCUMENTS_RETRIEVED,
            query="What is machine learning?",
            query_type="semantic",
            source="vector_store",
            source_id="docs-collection",
            num_results=5,
            results_truncated=False,
            top_score=0.95,
            avg_score=0.82,
            threshold_score=0.7,
            documents_used=["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"],
        )
        
        assert event.source == "vector_store"
        assert event.num_results == 5
        assert event.top_score == 0.95
        assert len(event.documents_used) == 5


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateLifecycleEvent:
    """Tests for create_lifecycle_event factory function."""
    
    def test_create_agent_started(self, session_id, execution_id):
        """Test creating agent_started event."""
        event = create_lifecycle_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
            goal="Analyze data",
        )
        
        assert isinstance(event, LifecycleEvent)
        assert event.event_type == LifecycleEventType.AGENT_STARTED
        assert event.goal == "Analyze data"
    
    def test_create_agent_completed(self, session_id, execution_id):
        """Test creating agent_completed event."""
        event = create_lifecycle_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_COMPLETED,
            final_status="success",
            total_iterations=3,
        )
        
        assert event.event_type == LifecycleEventType.AGENT_COMPLETED
        assert event.final_status == "success"
        assert event.total_iterations == 3


class TestCreateCognitiveEvent:
    """Tests for create_cognitive_event factory function."""
    
    def test_create_phase_completed(self, session_id, execution_id):
        """Test creating phase_completed event."""
        event = create_cognitive_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=CognitiveEventType.PHASE_COMPLETED,
            phase="think",
            iteration=1,
            input_summary="User query",
            output_summary="Decision made",
        )
        
        assert isinstance(event, CognitiveEvent)
        assert event.phase == "think"
        assert event.input_summary == "User query"


class TestCreateActivityEvent:
    """Tests for create_activity_event factory function."""
    
    def test_create_activity_completed(self, session_id, execution_id):
        """Test creating activity_completed event."""
        event = create_activity_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ActivityEventType.ACTIVITY_COMPLETED,
            activity_type="execute_python",
            activity_name="execute_python",
            success=True,
            result="Output",
        )
        
        assert isinstance(event, ActivityEvent)
        assert event.activity_type == "execute_python"
        assert event.success is True


class TestCreateErrorEvent:
    """Tests for create_error_event factory function."""
    
    def test_create_exception_event(self, session_id, execution_id):
        """Test creating exception event."""
        event = create_error_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ErrorEventType.EXCEPTION,
            error_type="ValueError",
            error_message="Bad input",
            recoverable=True,
        )
        
        assert isinstance(event, ErrorEvent)
        assert event.error_type == "ValueError"
        assert event.recoverable is True


class TestCreateMetricEvent:
    """Tests for create_metric_event factory function."""
    
    def test_create_token_usage_event(self, session_id, execution_id):
        """Test creating token_usage event."""
        event = create_metric_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=MetricEventType.TOKEN_USAGE,
            metric_name="total_tokens",
            metric_value=5000.0,
            metric_unit="tokens",
        )
        
        assert isinstance(event, MetricEvent)
        assert event.metric_name == "total_tokens"
        assert event.metric_value == 5000.0


class TestCreateHITLEvent:
    """Tests for create_hitl_event factory function."""
    
    def test_create_approval_requested(self, session_id, execution_id):
        """Test creating approval_requested event."""
        event = create_hitl_event(
            session_id=session_id,
            event_type=HITLEventType.APPROVAL_REQUESTED,
            request_id="req-123",
            action_type="shell",
            risk_level="high",
            approval_status="pending",
            execution_id=execution_id,
        )
        
        assert isinstance(event, HITLEvent)
        assert event.request_id == "req-123"
        assert event.risk_level == "high"
        assert event.approval_status == "pending"


class TestCreateSandboxEvent:
    """Tests for create_sandbox_event factory function."""
    
    def test_create_sandbox_created(self, session_id, execution_id):
        """Test creating sandbox_created event."""
        event = create_sandbox_event(
            session_id=session_id,
            event_type=SandboxEventType.SANDBOX_CREATED,
            sandbox_type="docker",
            operation="create",
            execution_id=execution_id,
            sandbox_id="sandbox-123",
            image="python:3.11",
        )
        
        assert isinstance(event, SandboxEvent)
        assert event.sandbox_id == "sandbox-123"
        assert event.image == "python:3.11"
        assert event.operation == "create"


# =============================================================================
# SERIALIZATION EDGE CASES
# =============================================================================


class TestSerializationEdgeCases:
    """Tests for serialization edge cases."""
    
    def test_empty_data_dict(self, session_id):
        """Empty data dict should serialize correctly."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            data={},
        )
        
        result = event.to_dict()
        assert result["data"] == {}
    
    def test_nested_data(self, session_id):
        """Nested data should serialize correctly."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            data={
                "level1": {
                    "level2": {
                        "level3": "value"
                    }
                },
                "list": [1, 2, {"nested": True}],
            },
        )
        
        result = event.to_dict()
        assert result["data"]["level1"]["level2"]["level3"] == "value"
        assert result["data"]["list"][2]["nested"] is True
    
    def test_special_characters_in_strings(self, session_id):
        """Special characters should serialize correctly."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            data={"message": "Hello\n\"World\"\t\\path"},
        )
        
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"]["message"] == "Hello\n\"World\"\t\\path"
    
    def test_unicode_characters(self, session_id):
        """Unicode characters should serialize correctly."""
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            data={"emoji": "🎉", "chinese": "你好", "arabic": "مرحبا"},
        )
        
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"]["emoji"] == "🎉"
        assert parsed["data"]["chinese"] == "你好"
        assert parsed["data"]["arabic"] == "مرحبا"
    
    def test_large_data_payload(self, session_id):
        """Large data payloads should serialize correctly."""
        large_list = list(range(1000))
        event = AgentEvent(
            session_id=session_id,
            category=EventCategory.LIFECYCLE,
            event_type="test",
            data={"large_list": large_list},
        )
        
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["data"]["large_list"]) == 1000
