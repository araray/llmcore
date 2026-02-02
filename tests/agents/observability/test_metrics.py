# tests/agents/observability/test_metrics.py
"""
Tests for the observability metrics module.

Covers:
- MetricType and ExecutionStatus enums
- Data classes (IterationMetrics, LLMCallMetrics, ActivityMetrics, HITLMetrics)
- ExecutionMetrics tracking and recording
- MetricsCollector aggregation
- Summary generation and percentile calculations
- Edge cases and validation
"""

from __future__ import annotations

import pytest

from llmcore.agents.observability import (
    ActivityMetrics,
    # Main classes
    ExecutionMetrics,
    ExecutionStatus,
    ExecutionSummary,
    HITLMetrics,
    # Data classes
    IterationMetrics,
    LLMCallMetrics,
    MetricsCollector,
    # Pydantic models
    MetricsSummary,
    # Enums
    MetricType,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_metric_types_exist(self):
        """Verify all expected metric types are defined."""
        expected = {
            "COUNTER", "GAUGE", "HISTOGRAM", "SUMMARY"
        }
        actual = {t.name for t in MetricType}
        assert actual == expected

    def test_metric_type_is_string_enum(self):
        """MetricType should be usable as a string."""
        assert MetricType.COUNTER == "counter"
        # StrEnum value is the string itself
        assert MetricType.SUMMARY.value == "summary"


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses are defined."""
        expected = {
            "RUNNING", "SUCCESS", "FAILURE", "TIMEOUT", "CANCELLED"
        }
        actual = {s.name for s in ExecutionStatus}
        assert actual == expected

    def test_status_is_string_enum(self):
        """ExecutionStatus should be usable as a string."""
        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.FAILURE == "failure"


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestIterationMetrics:
    """Tests for IterationMetrics data class."""

    def test_create_iteration_metrics(self):
        """Test creating iteration metrics."""
        metrics = IterationMetrics(
            iteration=1,
            duration_ms=150.0,
        )

        assert metrics.iteration == 1
        assert metrics.duration_ms == 150.0
        assert metrics.tokens_used == 0
        assert metrics.activities_executed == 0
        assert metrics.errors_occurred == 0

    def test_iteration_with_all_fields(self):
        """Test iteration metrics with all fields populated."""
        metrics = IterationMetrics(
            iteration=2,
            duration_ms=500.0,
            phase_durations={"think": 200.0, "act": 300.0},
            tokens_used=1000,
            activities_executed=2,
            errors_occurred=1,
        )

        assert metrics.duration_ms == 500.0
        assert metrics.tokens_used == 1000
        assert metrics.activities_executed == 2
        assert metrics.errors_occurred == 1
        assert "think" in metrics.phase_durations


class TestLLMCallMetrics:
    """Tests for LLMCallMetrics data class."""

    def test_create_llm_call_metrics(self):
        """Test creating LLM call metrics."""
        metrics = LLMCallMetrics(
            model="gpt-4",
            tokens_input=100,
            tokens_output=50,
            duration_ms=500.0,
            cost=0.01,
        )

        assert metrics.model == "gpt-4"
        assert metrics.tokens_input == 100
        assert metrics.tokens_output == 50
        assert metrics.cache_hit is False

    def test_llm_call_with_cost_and_cache(self):
        """Test LLM call with cost and cache fields."""
        metrics = LLMCallMetrics(
            model="claude-3",
            tokens_input=200,
            tokens_output=100,
            duration_ms=300.0,
            cost=0.02,
            cache_hit=True,
        )

        assert metrics.cost == 0.02
        assert metrics.cache_hit is True


class TestActivityMetrics:
    """Tests for ActivityMetrics data class."""

    def test_create_activity_metrics(self):
        """Test creating activity metrics."""
        metrics = ActivityMetrics(
            activity_name="execute_python",
            success=True,
            duration_ms=150.0,
        )

        assert metrics.activity_name == "execute_python"
        assert metrics.success is True
        assert metrics.retry_count == 0

    def test_failed_activity_metrics(self):
        """Test failed activity with retry count."""
        metrics = ActivityMetrics(
            activity_name="web_request",
            success=False,
            duration_ms=5000.0,
            retry_count=3,
        )

        assert metrics.success is False
        assert metrics.retry_count == 3


class TestHITLMetrics:
    """Tests for HITLMetrics data class."""

    def test_create_hitl_metrics(self):
        """Test creating HITL metrics."""
        metrics = HITLMetrics(
            request_id="req-123",
            action_type="execute_shell",
            risk_level="high",
            approved=True,
            wait_time_ms=5000.0,
        )

        assert metrics.request_id == "req-123"
        assert metrics.approved is True
        assert metrics.timeout_occurred is False

    def test_hitl_metrics_denied(self):
        """Test HITL metrics for denied action."""
        metrics = HITLMetrics(
            request_id="req-456",
            action_type="delete_file",
            risk_level="critical",
            approved=False,
            wait_time_ms=10000.0,
            timeout_occurred=True,
        )

        assert metrics.approved is False
        assert metrics.timeout_occurred is True


# =============================================================================
# EXECUTION METRICS TESTS
# =============================================================================


class TestExecutionMetrics:
    """Tests for ExecutionMetrics class."""

    def test_create_execution_metrics(self):
        """Test creating execution metrics."""
        metrics = ExecutionMetrics(
            execution_id="exec-123",
            session_id="sess-456",
            goal="Test goal",
        )

        assert metrics.execution_id == "exec-123"
        assert metrics.session_id == "sess-456"
        assert metrics.goal == "Test goal"

    def test_start_iteration(self):
        """Test starting an iteration."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        iteration = metrics.start_iteration()
        assert iteration == 1

        iteration2 = metrics.start_iteration()
        assert iteration2 == 2

    def test_start_multiple_iterations(self):
        """Test starting multiple iterations."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        for i in range(5):
            iteration = metrics.start_iteration()
            assert iteration == i + 1

    def test_end_iteration(self):
        """Test ending an iteration."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.start_iteration()

        result = metrics.end_iteration(
            tokens_used=100,
            activities_executed=2,
        )

        assert result.iteration == 1
        assert result.tokens_used == 100
        assert result.activities_executed == 2

    def test_end_iteration_with_stats(self):
        """Test ending iteration with all stats."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.start_iteration()

        result = metrics.end_iteration(
            phase_durations={"think": 100.0, "act": 200.0},
            tokens_used=500,
            activities_executed=3,
            errors_occurred=1,
        )

        assert "think" in result.phase_durations
        assert result.errors_occurred == 1

    def test_record_iteration(self):
        """Test recording a completed iteration."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        result = metrics.record_iteration(
            duration_ms=250.0,
            tokens_used=200,
        )

        assert result.iteration == 1
        assert result.duration_ms == 250.0

    def test_record_llm_call(self):
        """Test recording an LLM call."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        result = metrics.record_llm_call(
            model="gpt-4",
            tokens_input=100,
            tokens_output=50,
            duration_ms=500.0,
            cost=0.01,
        )

        assert result.model == "gpt-4"
        assert result.tokens_input == 100

    def test_record_multiple_llm_calls(self):
        """Test recording multiple LLM calls."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.record_llm_call("gpt-4", 100, 50, 500.0, 0.01)
        metrics.record_llm_call("claude-3", 200, 100, 300.0, 0.02)

        assert metrics.total_tokens == 450
        assert metrics.total_cost == pytest.approx(0.03)

    def test_record_tokens(self):
        """Test recording token usage."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.record_tokens(input=100, output=50, cost=0.01)

        assert metrics.total_tokens == 150

    def test_record_activity(self):
        """Test recording an activity."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        result = metrics.record_activity(
            activity_name="execute_python",
            success=True,
            duration_ms=150.0,
        )

        assert result.activity_name == "execute_python"
        assert result.success is True

    def test_record_failed_activity(self):
        """Test recording a failed activity."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        result = metrics.record_activity(
            activity_name="web_request",
            success=False,
            duration_ms=5000.0,
            retry_count=3,
        )

        assert result.success is False
        assert result.retry_count == 3

    def test_record_hitl(self):
        """Test recording HITL interaction."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        result = metrics.record_hitl(
            request_id="req-123",
            action_type="shell",
            risk_level="high",
            approved=True,
            wait_time_ms=5000.0,
        )

        assert result.request_id == "req-123"
        assert result.approved is True

    def test_record_error(self):
        """Test recording an error."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.record_error()
        metrics.record_error()

        # Error count is tracked internally
        summary = metrics.to_summary()
        assert summary["errors"]["total"] == 2

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.record_cache_hit()

        assert metrics.cache_hit_rate == 1.0

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.record_cache_miss()

        assert metrics.cache_hit_rate == 0.0

    def test_complete_success(self):
        """Test completing execution successfully."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_iteration(100.0)

        metrics.complete(success=True)

        assert metrics.status == ExecutionStatus.SUCCESS

    def test_complete_failure(self):
        """Test completing execution with failure."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.complete(success=False, exit_reason="Max iterations exceeded")

        assert metrics.status == ExecutionStatus.FAILURE

    def test_timeout(self):
        """Test timeout completion."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.timeout()

        assert metrics.status == ExecutionStatus.TIMEOUT

    def test_cancel(self):
        """Test cancellation."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.cancel()

        assert metrics.status == ExecutionStatus.CANCELLED


class TestExecutionMetricsProperties:
    """Tests for ExecutionMetrics computed properties."""

    def test_total_duration_ms(self):
        """Test total_duration_ms property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.complete(success=True)

        # Duration is computed from start/end time
        assert metrics.total_duration_ms >= 0

    def test_total_iterations(self):
        """Test total_iterations property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_iteration(100.0)
        metrics.record_iteration(150.0)

        assert metrics.total_iterations == 2

    def test_total_tokens(self):
        """Test total_tokens property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_tokens(input=100, output=50)
        metrics.record_tokens(input=200, output=100)

        assert metrics.total_tokens == 450

    def test_total_cost(self):
        """Test total_cost property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_llm_call("gpt-4", 100, 50, 500.0, 0.01)
        metrics.record_llm_call("claude-3", 200, 100, 300.0, 0.02)

        assert metrics.total_cost == pytest.approx(0.03)

    def test_total_activities(self):
        """Test total_activities property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_activity("act1", True, 100.0)
        metrics.record_activity("act2", False, 200.0)

        assert metrics.total_activities == 2

    def test_activity_success_rate(self):
        """Test activity_success_rate property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_activity("act1", True, 100.0)
        metrics.record_activity("act2", True, 100.0)
        metrics.record_activity("act3", False, 100.0)

        assert metrics.activity_success_rate == pytest.approx(2/3)

    def test_cache_hit_rate(self):
        """Test cache_hit_rate property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()

        assert metrics.cache_hit_rate == pytest.approx(2/3)

    def test_avg_iteration_duration_ms(self):
        """Test avg_iteration_duration_ms property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_iteration(100.0)
        metrics.record_iteration(200.0)
        metrics.record_iteration(300.0)

        assert metrics.avg_iteration_duration_ms == pytest.approx(200.0)

    def test_avg_llm_call_duration_ms(self):
        """Test avg_llm_call_duration_ms property."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_llm_call("gpt-4", 100, 50, 100.0, 0.01)
        metrics.record_llm_call("gpt-4", 100, 50, 200.0, 0.01)

        assert metrics.avg_llm_call_duration_ms == pytest.approx(150.0)


class TestExecutionMetricsSummary:
    """Tests for ExecutionMetrics summary generation."""

    def test_basic_summary(self):
        """Test basic summary generation."""
        metrics = ExecutionMetrics("exec-1", "sess-1", goal="Test")
        metrics.complete(success=True)

        summary = metrics.to_summary()

        assert "execution_id" in summary
        assert "status" in summary
        assert summary["goal"] == "Test"

    def test_summary_with_iterations(self):
        """Test summary with iterations."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_iteration(100.0, tokens_used=50)
        metrics.record_iteration(200.0, tokens_used=100)
        metrics.complete(success=True)

        summary = metrics.to_summary()

        assert summary["iterations"]["total"] == 2

    def test_summary_with_llm_calls(self):
        """Test summary with LLM calls."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_llm_call("gpt-4", 100, 50, 500.0, 0.01)
        metrics.complete(success=True)

        summary = metrics.to_summary()

        assert summary["llm_calls"]["total"] == 1

    def test_summary_with_activities(self):
        """Test summary with activities."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_activity("execute_python", True, 100.0)
        metrics.complete(success=True)

        summary = metrics.to_summary()

        assert summary["activities"]["total"] == 1

    def test_summary_model_breakdown(self):
        """Test summary includes model breakdown."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_llm_call("gpt-4", 100, 50, 500.0, 0.01)
        metrics.record_llm_call("gpt-4", 100, 50, 500.0, 0.01)
        metrics.record_llm_call("claude-3", 200, 100, 300.0, 0.02)
        metrics.complete(success=True)

        summary = metrics.to_summary()

        assert "by_model" in summary["llm_calls"]

    def test_summary_activity_breakdown(self):
        """Test summary includes activity breakdown."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.record_activity("execute_python", True, 100.0)
        metrics.record_activity("execute_python", True, 100.0)
        metrics.record_activity("web_request", False, 200.0)
        metrics.complete(success=True)

        summary = metrics.to_summary()

        assert "breakdown" in summary["activities"]


# =============================================================================
# METRICS COLLECTOR TESTS
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_create_collector(self):
        """Test creating a metrics collector."""
        collector = MetricsCollector()

        assert collector is not None

    def test_start_execution(self):
        """Test starting an execution."""
        collector = MetricsCollector()

        metrics = collector.start_execution("exec-1", goal="Test goal")

        assert metrics.execution_id == "exec-1"
        assert metrics.goal == "Test goal"

    def test_end_execution(self):
        """Test ending an execution."""
        collector = MetricsCollector()
        metrics = collector.start_execution("exec-1")

        collector.end_execution("exec-1", success=True)

        # Execution should be stored
        executions = collector.list_executions()
        assert len(executions) == 1

    def test_end_nonexistent_execution(self):
        """Test ending a nonexistent execution."""
        collector = MetricsCollector()

        # Should return None, not raise
        result = collector.end_execution("nonexistent", success=True)
        assert result is None

    def test_get_execution(self):
        """Test getting an execution."""
        collector = MetricsCollector()
        metrics = collector.start_execution("exec-1")

        result = collector.get_execution("exec-1")

        assert result is metrics

    def test_get_nonexistent_execution(self):
        """Test getting a nonexistent execution."""
        collector = MetricsCollector()

        result = collector.get_execution("nonexistent")

        assert result is None

    def test_list_executions(self):
        """Test listing executions."""
        collector = MetricsCollector()
        collector.start_execution("exec-1")
        collector.end_execution("exec-1", success=True)
        collector.start_execution("exec-2")
        collector.end_execution("exec-2", success=True)

        executions = collector.list_executions()

        assert len(executions) == 2

    def test_list_executions_by_status(self):
        """Test listing executions by status."""
        collector = MetricsCollector()
        collector.start_execution("exec-1")
        collector.end_execution("exec-1", success=True)
        collector.start_execution("exec-2")
        collector.end_execution("exec-2", success=False)

        successful = collector.list_executions(status=ExecutionStatus.SUCCESS)

        assert len(successful) == 1

    def test_list_executions_with_limit(self):
        """Test listing executions with limit."""
        collector = MetricsCollector()
        for i in range(5):
            collector.start_execution(f"exec-{i}")
            collector.end_execution(f"exec-{i}", success=True)

        executions = collector.list_executions(limit=3)

        assert len(executions) == 3

    def test_max_history_limit(self):
        """Test max history limit."""
        collector = MetricsCollector(max_history=2)

        for i in range(5):
            collector.start_execution(f"exec-{i}")
            collector.end_execution(f"exec-{i}", success=True)

        # Only the most recent 2 should be stored
        assert len(collector.list_executions()) == 2

    def test_clear(self):
        """Test clearing history."""
        collector = MetricsCollector()
        collector.start_execution("exec-1")
        collector.end_execution("exec-1", success=True)

        collector.clear()

        assert len(collector.list_executions()) == 0

    def test_reset(self):
        """Test resetting collector."""
        collector = MetricsCollector()
        collector.start_execution("exec-1")

        collector.reset()

        # Active executions should also be cleared
        assert collector.get_execution("exec-1") is None


class TestMetricsCollectorSummary:
    """Tests for MetricsCollector summary generation."""

    def test_empty_summary(self):
        """Test summary with no executions."""
        collector = MetricsCollector()

        summary = collector.get_summary()

        assert summary["total_executions"] == 0

    def test_basic_summary(self):
        """Test basic summary generation."""
        collector = MetricsCollector()
        collector.start_execution("exec-1")
        collector.end_execution("exec-1", success=True)

        summary = collector.get_summary()

        assert summary["total_executions"] == 1

    def test_summary_with_percentiles(self):
        """Test summary includes percentile calculations."""
        collector = MetricsCollector()
        for i in range(10):
            m = collector.start_execution(f"exec-{i}")
            m.record_iteration(float(i * 100))
            collector.end_execution(f"exec-{i}", success=True)

        summary = collector.get_summary()

        assert "latency" in summary

    def test_summary_token_statistics(self):
        """Test summary includes token statistics."""
        collector = MetricsCollector()
        m = collector.start_execution("exec-1")
        m.record_tokens(input=100, output=50)
        collector.end_execution("exec-1", success=True)

        summary = collector.get_summary()

        assert "tokens" in summary

    def test_summary_cost_statistics(self):
        """Test summary includes cost statistics."""
        collector = MetricsCollector()
        m = collector.start_execution("exec-1")
        m.record_llm_call("gpt-4", 100, 50, 500.0, 0.01)
        collector.end_execution("exec-1", success=True)

        summary = collector.get_summary()

        assert "cost" in summary

    def test_summary_status_breakdown(self):
        """Test summary includes status breakdown."""
        collector = MetricsCollector()
        collector.start_execution("exec-1")
        collector.end_execution("exec-1", success=True)
        collector.start_execution("exec-2")
        collector.end_execution("exec-2", success=False)

        summary = collector.get_summary()

        assert "status_breakdown" in summary


# =============================================================================
# PYDANTIC MODEL TESTS
# =============================================================================


class TestMetricsSummaryModel:
    """Tests for MetricsSummary Pydantic model."""

    def test_create_metrics_summary(self):
        """Test creating a metrics summary model."""
        summary = MetricsSummary(
            total_executions=10,
            active_executions=0,
            completed_executions=10,
            success_rate=0.8,
            status_breakdown={"success": 8, "failure": 2},
            latency={"p50": 100.0, "p95": 500.0, "p99": 1000.0},
            iterations={"total": 50, "avg": 5.0},
            tokens={"total": 10000, "avg": 1000.0},
            cost={"total": 0.5, "avg": 0.05},
            activity_success_rate=0.9,
            cache_hit_rate=0.3,
        )

        assert summary.total_executions == 10
        assert summary.success_rate == 0.8

    def test_metrics_summary_json(self):
        """Test metrics summary JSON serialization."""
        summary = MetricsSummary(
            total_executions=5,
            active_executions=0,
            completed_executions=5,
            success_rate=1.0,
            status_breakdown={"success": 5},
            iterations={"total": 10, "avg": 2.0},
            tokens={"total": 5000, "avg": 1000.0},
            cost={"total": 0.2, "avg": 0.04},
            activity_success_rate=1.0,
            cache_hit_rate=0.5,
        )

        json_str = summary.model_dump_json()
        assert "total_executions" in json_str


class TestExecutionSummaryModel:
    """Tests for ExecutionSummary Pydantic model."""

    def test_create_execution_summary(self):
        """Test creating an execution summary model."""
        summary = ExecutionSummary(
            execution_id="exec-123",
            session_id="sess-456",
            goal="Test goal",
            status="success",
            exit_reason="Completed",
            timing={"start": "2024-01-01T00:00:00Z", "duration_ms": 1500.0},
            iterations={"count": 5, "avg_duration_ms": 300.0},
            tokens={"input": 1000, "output": 500},
            cost={"total": 0.05},
            activities={"count": 3, "success_rate": 1.0},
            llm_calls={"count": 5, "avg_duration_ms": 200.0},
            cache={"hits": 2, "misses": 3},
            hitl={"requests": 0},
            errors={"count": 0},
        )

        assert summary.execution_id == "exec-123"
        assert summary.status == "success"

    def test_execution_summary_validation(self):
        """Test execution summary validates required fields."""
        # Should work with all required fields
        summary = ExecutionSummary(
            execution_id="exec-1",
            session_id="sess-1",
            goal=None,
            status="running",
            exit_reason=None,
            timing={},
            iterations={},
            tokens={},
            cost={},
            activities={},
            llm_calls={},
            cache={},
            hitl={},
            errors={},
        )
        assert summary is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestMetricsEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_duration_iteration(self):
        """Test iteration with zero duration."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        result = metrics.record_iteration(0.0)

        assert result.duration_ms == 0.0

    def test_very_large_token_counts(self):
        """Test very large token counts."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.record_tokens(input=1_000_000, output=500_000)

        assert metrics.total_tokens == 1_500_000

    def test_negative_cost_prevented(self):
        """Test that negative costs don't break calculations."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        # Recording negative cost - implementation should handle gracefully
        metrics.record_llm_call("gpt-4", 100, 50, 500.0, -0.01)

        # Should still work
        summary = metrics.to_summary()
        assert summary is not None

    def test_concurrent_iterations_warning(self):
        """Test that concurrent iterations work (start without end)."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        # Start without ending
        metrics.start_iteration()
        metrics.start_iteration()  # Should still work

        assert metrics.total_iterations == 0  # Not recorded until end_iteration

    def test_end_wrong_iteration(self):
        """Test ending iteration without starting."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        # End without start - should handle gracefully
        result = metrics.end_iteration()

        assert result.duration_ms == 0.0  # No start time

    def test_multiple_completions(self):
        """Test that multiple completions don't break."""
        metrics = ExecutionMetrics("exec-1", "sess-1")

        metrics.complete(success=True)
        metrics.complete(success=False)  # Second completion

        # Should still have a status
        assert metrics.status is not None

    def test_operations_after_completion(self):
        """Test operations after completion."""
        metrics = ExecutionMetrics("exec-1", "sess-1")
        metrics.complete(success=True)

        # Should still be able to record (for edge cases)
        metrics.record_iteration(100.0)

        assert metrics.total_iterations == 1
