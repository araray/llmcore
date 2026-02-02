# src/llmcore/tracing.py
"""
OpenTelemetry configuration and initialization for the llmcore platform.

This module provides centralized configuration for distributed tracing across
both the FastAPI server and the arq TaskMaster worker, enabling end-to-end
trace correlation for complex agentic workflows.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Global flag to track if tracing has been configured
_tracing_configured = False


def configure_tracer(service_name: str = "llmcore") -> None:
    """
    Configure the OpenTelemetry SDK for distributed tracing.

    This function sets up the OpenTelemetry tracer provider, processors, and exporters
    based on environment configuration. It should be called once at application startup
    for both the API server and TaskMaster worker.

    Args:
        service_name: The name of the service for trace identification
                     (e.g., "llmcore-api", "llmcore-worker")
    """
    global _tracing_configured

    if _tracing_configured:
        logger.debug(f"OpenTelemetry already configured for service: {service_name}")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        # Create resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "2.0.0-dev",
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
        })

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Configure exporters based on environment
        exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "console").lower()

        if exporter_type == "otlp":
            # Production OTLP exporter (e.g., for Jaeger, Zipkin via OTEL Collector)
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            logger.info(f"Configured OTLP span exporter for endpoint: {otlp_endpoint}")
        else:
            # Development console exporter
            exporter = ConsoleSpanExporter()
            logger.info("Configured console span exporter for development")

        # Add the exporter to a BatchSpanProcessor
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)

        # Auto-instrument common libraries
        _configure_auto_instrumentation()

        _tracing_configured = True
        logger.info(f"OpenTelemetry tracing configured successfully for service: {service_name}")

    except ImportError as e:
        logger.warning(f"OpenTelemetry dependencies not available: {e}")
        logger.warning("Distributed tracing will be disabled")
    except Exception as e:
        logger.error(f"Failed to configure OpenTelemetry tracing: {e}", exc_info=True)


def _configure_auto_instrumentation() -> None:
    """
    Configure automatic instrumentation for common libraries.

    This sets up auto-instrumentation for HTTP clients, Redis, and database
    connections to automatically create spans for external service calls.
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        # Instrument HTTP clients (for LLM provider calls)
        HTTPXClientInstrumentor().instrument()
        logger.debug("Configured HTTPX client instrumentation")

        # Instrument Redis (for task queue operations)
        RedisInstrumentor().instrument()
        logger.debug("Configured Redis instrumentation")

        # Instrument PostgreSQL connections
        PsycopgInstrumentor().instrument()
        logger.debug("Configured PostgreSQL (psycopg) instrumentation")

    except ImportError as e:
        logger.debug(f"Some auto-instrumentation libraries not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to configure auto-instrumentation: {e}")


def get_tracer(name: str) -> Optional[object]:
    """
    Get a tracer instance for creating manual spans.

    Args:
        name: The name of the tracer (typically the module name)

    Returns:
        OpenTelemetry tracer instance if available, None otherwise
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None


def create_span(tracer, name: str, **kwargs):
    """
    Create a new span with error handling.

    Args:
        tracer: OpenTelemetry tracer instance
        name: Name of the span
        **kwargs: Additional span attributes

    Returns:
        Span context manager or no-op context manager
    """
    if tracer is None:
        # Return a no-op context manager if tracing is not available
        from contextlib import nullcontext
        return nullcontext()

    try:
        return tracer.start_as_current_span(name, attributes=kwargs)
    except Exception as e:
        logger.debug(f"Failed to create span '{name}': {e}")
        from contextlib import nullcontext
        return nullcontext()


def inject_trace_context() -> dict:
    """
    Inject the current trace context into a dictionary for propagation.

    This is used to serialize trace context when enqueuing jobs in arq,
    allowing traces to span across the API server and TaskMaster worker.

    Returns:
        Dictionary containing serialized trace context
    """
    try:
        from opentelemetry import trace
        from opentelemetry.propagate import inject

        # Create a carrier dictionary to hold the trace context
        carrier = {}
        inject(carrier)
        return carrier

    except ImportError:
        return {}
    except Exception as e:
        logger.debug(f"Failed to inject trace context: {e}")
        return {}


def extract_and_set_trace_context(carrier: dict) -> None:
    """
    Extract trace context from a dictionary and set it as the current context.

    This is used in the TaskMaster worker to deserialize trace context from
    job payloads, linking worker spans to the originating API request trace.

    Args:
        carrier: Dictionary containing serialized trace context
    """
    if not carrier:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.context import attach
        from opentelemetry.propagate import extract

        # Extract the trace context from the carrier
        context = extract(carrier)

        # Attach the context to the current execution
        attach(context)

        logger.debug("Successfully extracted and set trace context")

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to extract trace context: {e}")


def add_span_attributes(span, attributes: dict) -> None:
    """
    Safely add attributes to a span.

    Args:
        span: OpenTelemetry span instance
        attributes: Dictionary of attributes to add
    """
    if span is None:
        return

    try:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value))
    except Exception as e:
        logger.debug(f"Failed to add span attributes: {e}")


def record_span_exception(span, exception: Exception) -> None:
    """
    Record an exception in a span.

    Args:
        span: OpenTelemetry span instance
        exception: Exception to record
    """
    if span is None:
        return

    try:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
    except Exception as e:
        logger.debug(f"Failed to record span exception: {e}")
