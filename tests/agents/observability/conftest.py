# tests/agents/observability/conftest.py
"""
Pytest configuration and fixtures for observability tests.

Provides shared fixtures for testing the observability module including:
- Event instances for all event types
- Logger configurations with various sinks
- Metrics collectors and execution contexts
- Replay fixtures with sample event logs
"""

from __future__ import annotations

import asyncio
import importlib.util

# =============================================================================
# IMPORT OBSERVABILITY MODULE
# =============================================================================
# Import directly from observability submodules to avoid full llmcore import chain
# This isolates the tests from unrelated dependency issues
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest

# Pre-register observability modules to bypass llmcore/__init__.py chain
_src_path = _Path(__file__).parent.parent.parent.parent / "src"


def _load_module(name: str, filepath: str):
    """Load module directly from file without triggering package __init__.py"""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _register_dummy_package(name: str):
    """Register a dummy package module to prevent import chain"""
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
    return sys.modules[name]


# Register parent packages as dummies to prevent cascade
_register_dummy_package("llmcore")
_register_dummy_package("llmcore.agents")
_register_dummy_package("llmcore.agents.observability")

# Load observability modules in order (events first, since others depend on it)
_events = _load_module(
    "llmcore.agents.observability.events",
    str(_src_path / "llmcore" / "agents" / "observability" / "events.py"),
)
_logger = _load_module(
    "llmcore.agents.observability.logger",
    str(_src_path / "llmcore" / "agents" / "observability" / "logger.py"),
)
_metrics = _load_module(
    "llmcore.agents.observability.metrics",
    str(_src_path / "llmcore" / "agents" / "observability" / "metrics.py"),
)
_replay = _load_module(
    "llmcore.agents.observability.replay",
    str(_src_path / "llmcore" / "agents" / "observability" / "replay.py"),
)

_observability_factory = _load_module(
    "llmcore.agents.observability_factory",
    str(_src_path / "llmcore" / "agents" / "observability_factory.py"),
)

# Link modules into the package namespace for standard imports
_obs_pkg = sys.modules["llmcore.agents.observability"]
_obs_pkg.events = _events
_obs_pkg.logger = _logger
_obs_pkg.metrics = _metrics
_obs_pkg.replay = _replay
_agents_pkg = sys.modules["llmcore.agents"]
_agents_pkg.observability_factory = _observability_factory

# Export all symbols at the observability package level
for _name in _events.__all__:
    setattr(_obs_pkg, _name, getattr(_events, _name))
for _name in _logger.__all__:
    setattr(_obs_pkg, _name, getattr(_logger, _name))
for _name in _metrics.__all__:
    setattr(_obs_pkg, _name, getattr(_metrics, _name))
for _name in _replay.__all__:
    setattr(_obs_pkg, _name, getattr(_replay, _name))

# Now use standard imports (they'll find the pre-registered modules)
from llmcore.agents.observability.events import (
    ActivityEvent,
    ActivityEventType,
    # Events
    AgentEvent,
    CognitiveEvent,
    CognitiveEventType,
    ErrorEvent,
    ErrorEventType,
    # Enums
    EventCategory,
    EventSeverity,
    HITLEvent,
    HITLEventType,
    LifecycleEvent,
    LifecycleEventType,
    MemoryEvent,
    MemoryEventType,
    MetricEvent,
    MetricEventType,
    RAGEvent,
    RAGEventType,
    SandboxEvent,
    SandboxEventType,
)
from llmcore.agents.observability.logger import (
    CallbackSink,
    EventLogger,
    # Logger
    EventSink,
    FilteredSink,
    InMemorySink,
    JSONLFileSink,
)
from llmcore.agents.observability.metrics import (
    # Enums
    ExecutionMetrics,
    MetricsCollector,
)
from llmcore.agents.observability.replay import (
    # Replay
    ExecutionReplay,
)

# =============================================================================
# EVENT LOOP CONFIGURATION
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# TIME HELPERS
# =============================================================================


@pytest.fixture
def fixed_timestamp():
    """Provide a fixed timestamp for deterministic testing."""
    return datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def timestamp_factory(fixed_timestamp):
    """Factory for generating sequential timestamps."""
    counter = [0]

    def make_timestamp(offset_seconds: int = 0):
        ts = fixed_timestamp + timedelta(seconds=counter[0] + offset_seconds)
        counter[0] += 1
        return ts

    return make_timestamp


# =============================================================================
# SESSION AND EXECUTION IDS
# =============================================================================


@pytest.fixture
def session_id():
    """Standard test session ID."""
    return "test-session-12345"


@pytest.fixture
def execution_id():
    """Standard test execution ID."""
    return "test-exec-67890"


@pytest.fixture
def correlation_id():
    """Standard test correlation ID."""
    return "test-corr-abcdef"


# =============================================================================
# TEMPORARY FILE FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_jsonl_file(temp_dir):
    """Create a temporary JSONL file path."""
    return temp_dir / "events.jsonl"


@pytest.fixture
def temp_log_dir(temp_dir):
    """Create a temporary log directory."""
    log_dir = temp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# =============================================================================
# BASE EVENT FIXTURES
# =============================================================================


@pytest.fixture
def sample_agent_event(session_id, execution_id):
    """Create a sample base agent event."""
    return AgentEvent(
        session_id=session_id,
        execution_id=execution_id,
        category=EventCategory.LIFECYCLE,
        event_type="test_event",
        severity=EventSeverity.INFO,
        phase="test",
        iteration=1,
        data={"key": "value"},
    )


@pytest.fixture
def sample_lifecycle_event(session_id, execution_id):
    """Create a sample lifecycle event."""
    return LifecycleEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=LifecycleEventType.AGENT_STARTED,
        goal="Test goal",
    )


@pytest.fixture
def sample_cognitive_event(session_id, execution_id):
    """Create a sample cognitive event."""
    return CognitiveEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=CognitiveEventType.PHASE_COMPLETED,
        phase="think",
        iteration=1,
        input_summary="User asked about data",
        output_summary="Decided to use pandas",
    )


@pytest.fixture
def sample_activity_event(session_id, execution_id):
    """Create a sample activity event."""
    return ActivityEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=ActivityEventType.ACTIVITY_COMPLETED,
        activity_type="execute_python",
        activity_name="execute_python",
        parameters={"code": "print('hello')"},
        result="hello",
        success=True,
        duration_ms=150.5,
    )


@pytest.fixture
def sample_memory_event(session_id, execution_id):
    """Create a sample memory event."""
    return MemoryEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=MemoryEventType.MEMORY_WRITE,
        memory_type="short_term",
        operation="write",
        key="test_key",
        value_summary="Test value summary",
    )


@pytest.fixture
def sample_hitl_event(session_id, execution_id):
    """Create a sample HITL event."""
    return HITLEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=HITLEventType.APPROVAL_REQUESTED,
        request_id="req-12345",
        action_type="execute_shell",
        risk_level="high",
        requires_approval=True,
    )


@pytest.fixture
def sample_error_event(session_id, execution_id):
    """Create a sample error event."""
    return ErrorEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=ErrorEventType.EXCEPTION,
        severity=EventSeverity.ERROR,
        error_type="ValueError",
        error_message="Invalid input provided",
        error_code="E001",
        recoverable=True,
    )


@pytest.fixture
def sample_metric_event(session_id, execution_id):
    """Create a sample metric event."""
    return MetricEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=MetricEventType.TOKEN_USAGE,
        metric_name="input_tokens",
        metric_value=1500.0,
        metric_unit="tokens",
    )


@pytest.fixture
def sample_sandbox_event(session_id, execution_id):
    """Create a sample sandbox event."""
    return SandboxEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=SandboxEventType.CODE_EXECUTED,
        sandbox_id="sandbox-12345",
        sandbox_type="docker",
        image="llmcore-sandbox-python:1.0.0",
        command="python -c 'print(1)'",
        exit_code=0,
    )


@pytest.fixture
def sample_rag_event(session_id, execution_id):
    """Create a sample RAG event."""
    return RAGEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=RAGEventType.DOCUMENTS_RETRIEVED,
        query="test query",
        num_documents=5,
        retrieval_time_ms=25.5,
    )


# =============================================================================
# EVENT COLLECTION FIXTURES
# =============================================================================


@pytest.fixture
def all_event_types(
    sample_lifecycle_event,
    sample_cognitive_event,
    sample_activity_event,
    sample_memory_event,
    sample_hitl_event,
    sample_error_event,
    sample_metric_event,
    sample_sandbox_event,
    sample_rag_event,
):
    """Collection of all event types for testing."""
    return [
        sample_lifecycle_event,
        sample_cognitive_event,
        sample_activity_event,
        sample_memory_event,
        sample_hitl_event,
        sample_error_event,
        sample_metric_event,
        sample_sandbox_event,
        sample_rag_event,
    ]


@pytest.fixture
def sample_event_sequence(session_id, execution_id, fixed_timestamp):
    """Create a realistic sequence of events for testing."""
    events = []

    # Agent start
    events.append(
        LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_STARTED,
            timestamp=fixed_timestamp,
            goal="Analyze data and generate report",
        )
    )

    # Iteration 1 start
    events.append(
        LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.ITERATION_STARTED,
            timestamp=fixed_timestamp + timedelta(seconds=1),
            iteration=1,
        )
    )

    # Cognitive phase - think
    events.append(
        CognitiveEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=CognitiveEventType.PHASE_COMPLETED,
            timestamp=fixed_timestamp + timedelta(seconds=2),
            phase="think",
            iteration=1,
            input_summary="User query",
            output_summary="Determined approach",
            duration_ms=500.0,
        )
    )

    # Activity execution
    events.append(
        ActivityEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ActivityEventType.ACTIVITY_COMPLETED,
            timestamp=fixed_timestamp + timedelta(seconds=3),
            activity_type="execute_python",
            activity_name="execute_python",
            parameters={"code": "import pandas"},
            result="Success",
            success=True,
            duration_ms=150.0,
            iteration=1,
        )
    )

    # Agent completion
    events.append(
        LifecycleEvent(
            session_id=session_id,
            execution_id=execution_id,
            event_type=LifecycleEventType.AGENT_COMPLETED,
            timestamp=fixed_timestamp + timedelta(seconds=5),
            final_status="success",
            total_iterations=1,
        )
    )

    return events


# =============================================================================
# SINK FIXTURES
# =============================================================================


@pytest.fixture
def in_memory_sink():
    """Create an in-memory sink for testing."""
    return InMemorySink(max_events=1000)


@pytest.fixture
def jsonl_sink(temp_jsonl_file):
    """Create a JSONL file sink."""
    return JSONLFileSink(temp_jsonl_file, buffer_size=1)


@pytest.fixture
def callback_sink():
    """Create a callback sink with mock callback."""
    callback = AsyncMock()
    return CallbackSink(callback), callback


@pytest.fixture
def filtered_sink(in_memory_sink):
    """Create a filtered sink wrapping in-memory sink."""
    return FilteredSink(
        inner_sink=in_memory_sink,
        categories=[EventCategory.LIFECYCLE, EventCategory.ACTIVITY],
        min_severity=EventSeverity.INFO,
    )


# =============================================================================
# LOGGER FIXTURES
# =============================================================================


@pytest.fixture
def event_logger(session_id, execution_id, in_memory_sink):
    """Create an event logger with in-memory sink."""
    logger = EventLogger(
        session_id=session_id,
        execution_id=execution_id,
    )
    logger.add_sink(in_memory_sink)
    return logger


@pytest.fixture
def file_logger(session_id, execution_id, temp_jsonl_file):
    """Create an event logger with file sink."""
    sink = JSONLFileSink(temp_jsonl_file, buffer_size=1)
    logger = EventLogger(
        session_id=session_id,
        execution_id=execution_id,
    )
    logger.add_sink(sink)
    return logger


# =============================================================================
# METRICS FIXTURES
# =============================================================================


@pytest.fixture
def execution_metrics(execution_id):
    """Create execution metrics instance."""
    return ExecutionMetrics(
        execution_id=execution_id,
        goal="Test goal",
    )


@pytest.fixture
def populated_metrics(execution_metrics):
    """Create execution metrics with sample data."""
    # Record iteration
    execution_metrics.start_iteration(1)
    execution_metrics.record_llm_call(
        model="gpt-4",
        input_tokens=500,
        output_tokens=200,
        duration_ms=1500.0,
        cost=0.05,
    )
    execution_metrics.record_activity(
        activity_type="execute_python",
        success=True,
        duration_ms=150.0,
    )
    execution_metrics.end_iteration(1)

    # Record second iteration
    execution_metrics.start_iteration(2)
    execution_metrics.record_llm_call(
        model="gpt-4",
        input_tokens=800,
        output_tokens=300,
        duration_ms=2000.0,
        cost=0.08,
    )
    execution_metrics.end_iteration(2)

    return execution_metrics


@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector(max_history=100)


@pytest.fixture
def populated_collector(metrics_collector):
    """Create a metrics collector with historical data."""
    # Add several executions
    for i in range(5):
        metrics = metrics_collector.start_execution(
            execution_id=f"exec-{i}",
            goal=f"Goal {i}",
        )
        metrics.start_iteration(1)
        metrics.record_llm_call(
            model="gpt-4",
            input_tokens=500 + i * 100,
            output_tokens=200 + i * 50,
            duration_ms=1000.0 + i * 200,
            cost=0.05 + i * 0.01,
        )
        metrics.end_iteration(1)
        metrics.complete(success=True if i % 2 == 0 else False)

    return metrics_collector


# =============================================================================
# REPLAY FIXTURES
# =============================================================================


@pytest.fixture
def sample_jsonl_log(temp_jsonl_file, sample_event_sequence):
    """Create a sample JSONL log file with events."""
    with open(temp_jsonl_file, "w") as f:
        for event in sample_event_sequence:
            f.write(event.to_json() + "\n")
    return temp_jsonl_file


@pytest.fixture
def execution_replay(sample_jsonl_log):
    """Create an execution replay from sample log."""
    return ExecutionReplay.from_file(sample_jsonl_log)


@pytest.fixture
def multi_execution_log(temp_dir, fixed_timestamp):
    """Create a log file with multiple executions."""
    log_file = temp_dir / "multi_exec.jsonl"

    events = []

    # Execution 1
    for i in range(3):
        events.append(
            LifecycleEvent(
                session_id="session-1",
                execution_id="exec-1",
                event_type=LifecycleEventType.ITERATION_STARTED
                if i < 2
                else LifecycleEventType.AGENT_COMPLETED,
                timestamp=fixed_timestamp + timedelta(seconds=i),
                iteration=i + 1 if i < 2 else None,
            )
        )

    # Execution 2
    for i in range(2):
        events.append(
            LifecycleEvent(
                session_id="session-2",
                execution_id="exec-2",
                event_type=LifecycleEventType.AGENT_STARTED
                if i == 0
                else LifecycleEventType.AGENT_FAILED,
                timestamp=fixed_timestamp + timedelta(seconds=10 + i),
            )
        )

    with open(log_file, "w") as f:
        for event in events:
            f.write(event.to_json() + "\n")

    return log_file


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_sink():
    """Create a mock event sink."""
    sink = AsyncMock(spec=EventSink)
    sink.write = AsyncMock()
    sink.flush = AsyncMock()
    sink.close = AsyncMock()
    return sink


@pytest.fixture
def mock_callback():
    """Create a mock callback function."""
    return Mock()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_event_dict(
    session_id: str = "test-session",
    execution_id: str = "test-exec",
    category: str = "lifecycle",
    event_type: str = "agent_started",
    **kwargs,
) -> Dict[str, Any]:
    """Create an event dictionary for testing parsing."""
    return {
        "event_id": f"evt-{id(kwargs) % 100000:05d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "execution_id": execution_id,
        "category": category,
        "event_type": event_type,
        "severity": "info",
        "data": {},
        "tags": [],
        **kwargs,
    }


# Make utility available to tests
@pytest.fixture
def event_dict_factory():
    """Factory for creating event dictionaries."""
    return create_event_dict
