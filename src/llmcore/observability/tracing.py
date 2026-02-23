# src/llmcore/observability/tracing.py
"""
Distributed Tracing for LLMCore.

This module consolidates tracing configuration from:

- ``llmcore/tracing.py`` — Core OpenTelemetry setup
- ``llmcore/agents/tracing.py`` — Agent-specific span instrumentation

It provides the spec-mandated ``observability/tracing.py`` entry-point
while delegating to the existing implementations to avoid code duplication.

Usage::

    from llmcore.observability.tracing import (
        configure_tracer,
        get_tracer,
        trace_llm_call,
        trace_agent_phase,
    )

    # Application startup
    configure_tracer(service_name="my-agent")

    # In code
    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation"):
        ...

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §13 (Observability System)
    - OpenTelemetry Python SDK documentation
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Re-export the core tracer configuration from the existing module
from ..tracing import configure_tracer

# Try to import OpenTelemetry for span creation
try:
    from opentelemetry import trace

    _otel_available = True
except ImportError:
    _otel_available = False
    trace = None  # type: ignore[assignment]


def get_tracer(name: str = "llmcore") -> Any:
    """Get an OpenTelemetry tracer instance.

    Returns a real tracer if OpenTelemetry is available, or a no-op
    stub otherwise.

    Args:
        name: Tracer name (usually the module or service name).

    Returns:
        An OpenTelemetry Tracer or a NoOpTracer.
    """
    if _otel_available and trace is not None:
        return trace.get_tracer(name)
    return _NoOpTracer()


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "llmcore",
) -> Generator[Any, None, None]:
    """Context manager for creating a traced span.

    Falls back to a no-op if OpenTelemetry is not available.

    Args:
        name: Span name.
        attributes: Optional span attributes.
        tracer_name: Tracer name.

    Yields:
        The span object (real or no-op).
    """
    tracer = get_tracer(tracer_name)
    if _otel_available:
        with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span
    else:
        yield _NoOpSpan()


def trace_llm_call(
    provider: str,
    model: str,
    operation: str = "chat",
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: float = 0.0,
    error: str | None = None,
) -> None:
    """Record a traced LLM API call.

    Creates a span with standard LLM call attributes.  No-op if
    OpenTelemetry is not configured.

    Args:
        provider: LLM provider name.
        model: Model identifier.
        operation: Operation type (chat, embed, etc.).
        input_tokens: Input token count.
        output_tokens: Output token count.
        duration_ms: Call duration in milliseconds.
        error: Error message if the call failed.
    """
    if not _otel_available:
        return

    tracer = get_tracer("llmcore.llm")
    with tracer.start_as_current_span(f"llm.{operation}") as span:
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.operation", operation)
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("llm.duration_ms", duration_ms)
        if error:
            span.set_attribute("llm.error", error)
            span.set_status(trace.StatusCode.ERROR, error)


def trace_agent_phase(
    phase: str,
    agent_id: str = "",
    iteration: int = 0,
    duration_ms: float = 0.0,
    status: str = "ok",
) -> None:
    """Record a traced cognitive phase execution.

    Args:
        phase: Phase name (PERCEIVE, PLAN, THINK, etc.).
        agent_id: Agent identifier.
        iteration: Cycle iteration number.
        duration_ms: Phase duration in milliseconds.
        status: Execution status.
    """
    if not _otel_available:
        return

    tracer = get_tracer("llmcore.agent")
    with tracer.start_as_current_span(f"agent.phase.{phase.lower()}") as span:
        span.set_attribute("agent.phase", phase)
        span.set_attribute("agent.id", agent_id)
        span.set_attribute("agent.iteration", iteration)
        span.set_attribute("agent.phase_duration_ms", duration_ms)
        span.set_attribute("agent.status", status)


# ---------------------------------------------------------------------------
# No-op stubs for when OpenTelemetry is not installed
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Minimal span stub that accepts arbitrary attribute/status calls."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Minimal tracer stub."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


__all__ = [
    "configure_tracer",
    "get_tracer",
    "trace_agent_phase",
    "trace_llm_call",
    "trace_span",
]
