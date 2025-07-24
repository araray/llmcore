# src/llmcore/api_server/metrics.py
"""
Custom Prometheus metrics for the llmcore platform.

This module defines and registers application-specific metrics for monitoring
LLM usage, performance, costs, and system health. These metrics complement
the standard HTTP metrics provided by prometheus-fastapi-instrumentator.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Define no-op classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def time(self):
            from contextlib import nullcontext
            return nullcontext()

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Provider Metrics
# ============================================================================

# Counter for tracking total LLM requests
llm_requests_total = Counter(
    'llmcore_llm_requests_total',
    'Total number of LLM API requests',
    ['provider', 'model', 'tenant_id', 'endpoint']
)

# Counter for tracking token usage (input/output separately for cost analysis)
llm_tokens_total = Counter(
    'llmcore_llm_tokens_total',
    'Total number of tokens processed by LLM providers',
    ['provider', 'model', 'tenant_id', 'token_type']  # token_type: input|output
)

# Histogram for LLM request latency
llm_request_latency_seconds = Histogram(
    'llmcore_llm_request_latency_seconds',
    'Latency of LLM API requests in seconds',
    ['provider', 'model', 'tenant_id'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')]
)

# Counter for LLM request errors
llm_request_errors_total = Counter(
    'llmcore_llm_request_errors_total',
    'Total number of failed LLM API requests',
    ['provider', 'model', 'tenant_id', 'error_type']
)

# ============================================================================
# Task Queue Metrics
# ============================================================================

# Gauge for task queue depth
task_queue_depth = Gauge(
    'llmcore_task_queue_depth',
    'Number of pending jobs in the task queue',
    ['queue_name']
)

# Counter for task completions
task_completions_total = Counter(
    'llmcore_task_completions_total',
    'Total number of completed tasks',
    ['task_type', 'status']  # status: success|failure
)

# Histogram for task execution duration
task_duration_seconds = Histogram(
    'llmcore_task_duration_seconds',
    'Duration of task execution in seconds',
    ['task_type'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0, float('inf')]
)

# ============================================================================
# Agent Metrics
# ============================================================================

# Counter for agent loops/sessions
agent_loops_total = Counter(
    'llmcore_agent_loops_total',
    'Total number of agent execution loops',
    ['tenant_id', 'status']  # status: completed|timeout|error
)

# Histogram for agent loop iterations
agent_loop_iterations = Histogram(
    'llmcore_agent_loop_iterations',
    'Number of iterations in agent execution loops',
    ['tenant_id'],
    buckets=[1, 3, 5, 10, 15, 20, 30, 50, float('inf')]
)

# Counter for tool executions
agent_tool_executions_total = Counter(
    'llmcore_agent_tool_executions_total',
    'Total number of tool executions by agents',
    ['tenant_id', 'tool_name', 'status']  # status: success|error
)

# ============================================================================
# Memory System Metrics
# ============================================================================

# Counter for memory operations
memory_operations_total = Counter(
    'llmcore_memory_operations_total',
    'Total number of memory system operations',
    ['tenant_id', 'operation_type', 'memory_type']  # operation_type: search|store|delete, memory_type: semantic|episodic
)

# Histogram for memory search latency
memory_search_latency_seconds = Histogram(
    'llmcore_memory_search_latency_seconds',
    'Latency of memory search operations in seconds',
    ['tenant_id', 'memory_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
)

# ============================================================================
# System Health Metrics
# ============================================================================

# Info metric for system version and configuration
system_info = Info(
    'llmcore_system_info',
    'System information and configuration'
)

# Gauge for active tenant count
active_tenants_total = Gauge(
    'llmcore_active_tenants_total',
    'Number of active tenants in the system'
)

# Gauge for database connection pool status
db_connection_pool_active = Gauge(
    'llmcore_db_connection_pool_active',
    'Number of active database connections',
    ['pool_type']  # pool_type: auth|tenant|vector
)

# ============================================================================
# Metric Helper Functions
# ============================================================================

def record_llm_request(
    provider: str,
    model: str,
    tenant_id: str,
    duration: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    endpoint: str = "chat_completion",
    error: Optional[str] = None
) -> None:
    """
    Record metrics for an LLM API request.

    Args:
        provider: LLM provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3")
        tenant_id: Tenant identifier
        duration: Request duration in seconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        endpoint: API endpoint called
        error: Error type if request failed
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        # Record request count
        llm_requests_total.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id,
            endpoint=endpoint
        ).inc()

        # Record latency
        llm_request_latency_seconds.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id
        ).observe(duration)

        # Record token usage
        if input_tokens is not None:
            llm_tokens_total.labels(
                provider=provider,
                model=model,
                tenant_id=tenant_id,
                token_type="input"
            ).inc(input_tokens)

        if output_tokens is not None:
            llm_tokens_total.labels(
                provider=provider,
                model=model,
                tenant_id=tenant_id,
                token_type="output"
            ).inc(output_tokens)

        # Record errors
        if error:
            llm_request_errors_total.labels(
                provider=provider,
                model=model,
                tenant_id=tenant_id,
                error_type=error
            ).inc()

    except Exception as e:
        logger.warning(f"Failed to record LLM request metrics: {e}")


def record_task_completion(
    task_type: str,
    duration: float,
    status: str = "success"
) -> None:
    """
    Record metrics for a completed task.

    Args:
        task_type: Type of task (e.g., "agent", "ingestion")
        duration: Task duration in seconds
        status: Task completion status ("success" or "failure")
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        task_completions_total.labels(
            task_type=task_type,
            status=status
        ).inc()

        task_duration_seconds.labels(
            task_type=task_type
        ).observe(duration)

    except Exception as e:
        logger.warning(f"Failed to record task completion metrics: {e}")


def record_agent_execution(
    tenant_id: str,
    iterations: int,
    status: str = "completed"
) -> None:
    """
    Record metrics for an agent execution loop.

    Args:
        tenant_id: Tenant identifier
        iterations: Number of iterations in the loop
        status: Execution status ("completed", "timeout", "error")
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        agent_loops_total.labels(
            tenant_id=tenant_id,
            status=status
        ).inc()

        agent_loop_iterations.labels(
            tenant_id=tenant_id
        ).observe(iterations)

    except Exception as e:
        logger.warning(f"Failed to record agent execution metrics: {e}")


def record_tool_execution(
    tenant_id: str,
    tool_name: str,
    status: str = "success"
) -> None:
    """
    Record metrics for a tool execution.

    Args:
        tenant_id: Tenant identifier
        tool_name: Name of the executed tool
        status: Execution status ("success" or "error")
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        agent_tool_executions_total.labels(
            tenant_id=tenant_id,
            tool_name=tool_name,
            status=status
        ).inc()

    except Exception as e:
        logger.warning(f"Failed to record tool execution metrics: {e}")


def record_memory_operation(
    tenant_id: str,
    operation_type: str,
    memory_type: str,
    duration: Optional[float] = None
) -> None:
    """
    Record metrics for memory system operations.

    Args:
        tenant_id: Tenant identifier
        operation_type: Type of operation ("search", "store", "delete")
        memory_type: Type of memory ("semantic", "episodic")
        duration: Operation duration in seconds (for search operations)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        memory_operations_total.labels(
            tenant_id=tenant_id,
            operation_type=operation_type,
            memory_type=memory_type
        ).inc()

        if duration is not None and operation_type == "search":
            memory_search_latency_seconds.labels(
                tenant_id=tenant_id,
                memory_type=memory_type
            ).observe(duration)

    except Exception as e:
        logger.warning(f"Failed to record memory operation metrics: {e}")


async def update_queue_depth_metrics():
    """
    Background task to periodically update task queue depth metrics.

    This function should be called periodically (e.g., every 30 seconds)
    to update the task queue depth gauge by querying Redis.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        from ..services.redis_client import get_redis_pool

        # Get the Redis pool
        redis_pool = get_redis_pool()

        # Query queue length
        queue_length = await redis_pool.llen('arq:queue')

        # Update metric
        task_queue_depth.labels(queue_name='default').set(queue_length)

    except Exception as e:
        logger.debug(f"Failed to update queue depth metrics: {e}")


def initialize_system_info():
    """
    Initialize the system info metric with current configuration.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        import llmcore

        system_info.info({
            'version': getattr(llmcore, '__version__', 'unknown'),
            'service': 'llmcore-api',
            'observability_enabled': 'true'
        })

    except Exception as e:
        logger.warning(f"Failed to initialize system info metrics: {e}")


class MetricsTimer:
    """
    Context manager for timing operations and recording metrics.

    Usage:
        with MetricsTimer(metric_histogram, labels={'operation': 'search'}):
            # perform operation
            pass
    """

    def __init__(self, histogram, labels: Optional[Dict[str, str]] = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            try:
                if self.labels:
                    self.histogram.labels(**self.labels).observe(duration)
                else:
                    self.histogram.observe(duration)
            except Exception as e:
                logger.debug(f"Failed to record timing metric: {e}")


def create_metrics_timer(histogram, **labels):
    """
    Create a metrics timer for the given histogram and labels.

    Args:
        histogram: Prometheus histogram metric
        **labels: Label values for the metric

    Returns:
        MetricsTimer context manager
    """
    return MetricsTimer(histogram, labels)
