# src/llmcore/api_server/middleware/observability.py
"""
Observability middleware for the llmcore API server.

This middleware handles structured logging context injection, ensuring that
all log messages within a request lifecycle are enriched with request_id,
tenant_id, and other contextual information for enhanced debugging and monitoring.
"""

import logging
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structured logging with structlog
try:
    import structlog
    from structlog.contextvars import bind_contextvars, clear_contextvars
    STRUCTLOG_AVAILABLE = True

    # Configure structlog processors and formatting
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if logging.getLogger().isEnabledFor(logging.DEBUG)
            else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Get structured logger
    logger = structlog.get_logger(__name__)

except ImportError:
    STRUCTLOG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("structlog not available, falling back to standard logging")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for injecting observability context into requests.

    This middleware:
    1. Generates a unique request_id for each incoming request
    2. Extracts the tenant_id from the authenticated tenant (if available)
    3. Binds this context to structlog's context-local storage
    4. Ensures context is available throughout the request lifecycle
    5. Clears context after request completion
    """

    def __init__(self, app, enable_request_logging: bool = True):
        """
        Initialize the observability middleware.

        Args:
            app: FastAPI application instance
            enable_request_logging: Whether to log request start/end events
        """
        super().__init__(app)
        self.enable_request_logging = enable_request_logging

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request with observability context injection.

        Args:
            request: The incoming FastAPI request
            call_next: The next middleware or route handler

        Returns:
            The response from the downstream handler
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Initialize context variables
        context_vars = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent", "unknown")
        }

        # Extract tenant information if available (after auth middleware has run)
        tenant_id = None
        tenant_name = None

        if hasattr(request.state, 'tenant') and request.state.tenant:
            tenant_id = str(request.state.tenant.id)
            tenant_name = request.state.tenant.name
            context_vars.update({
                "tenant_id": tenant_id,
                "tenant_name": tenant_name
            })

        # Bind context variables for structured logging
        if STRUCTLOG_AVAILABLE:
            clear_contextvars()
            bind_contextvars(**context_vars)

        # Add request_id to request state for access by route handlers
        request.state.request_id = request_id

        # Log request start
        if self.enable_request_logging:
            if STRUCTLOG_AVAILABLE:
                logger.info("Request started",
                           extra={"event": "request_start"})
            else:
                logger.info(f"Request started: {request.method} {request.url.path} "
                           f"[request_id={request_id}] [tenant_id={tenant_id}]")

        try:
            # Process the request
            response = await call_next(request)

            # Add observability headers to response
            response.headers["X-Request-ID"] = request_id
            if tenant_id:
                response.headers["X-Tenant-ID"] = tenant_id

            # Log successful request completion
            if self.enable_request_logging:
                if STRUCTLOG_AVAILABLE:
                    logger.info("Request completed",
                               status_code=response.status_code,
                               extra={"event": "request_completed"})
                else:
                    logger.info(f"Request completed: {response.status_code} "
                               f"[request_id={request_id}] [tenant_id={tenant_id}]")

            return response

        except Exception as e:
            # Log request error
            if STRUCTLOG_AVAILABLE:
                logger.error("Request failed",
                            error=str(e),
                            error_type=type(e).__name__,
                            extra={"event": "request_error"})
            else:
                logger.error(f"Request failed: {str(e)} "
                           f"[request_id={request_id}] [tenant_id={tenant_id}]")

            # Re-raise the exception to be handled by FastAPI's exception handlers
            raise

        finally:
            # Clear context variables after request completion
            if STRUCTLOG_AVAILABLE:
                clear_contextvars()


def get_current_request_context() -> dict:
    """
    Get the current request context variables.

    This function allows route handlers and other code to access the
    current request's context information (request_id, tenant_id, etc.)
    that was set by the ObservabilityMiddleware.

    Returns:
        Dictionary containing current context variables
    """
    if STRUCTLOG_AVAILABLE:
        try:
            # Extract context from structlog's context variables
            from structlog.contextvars import get_contextvars
            return get_contextvars()
        except Exception:
            return {}
    else:
        # Fallback: return empty dict if structlog not available
        return {}


def log_with_context(message: str, level: str = "info", **extra_context):
    """
    Log a message with the current request context.

    This is a convenience function for logging messages with automatic
    inclusion of the current request context (request_id, tenant_id, etc.).

    Args:
        message: The log message
        level: Log level (debug, info, warning, error, critical)
        **extra_context: Additional context to include in the log
    """
    if STRUCTLOG_AVAILABLE:
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message, **extra_context)
    else:
        # Fallback to standard logging
        log_func = getattr(logger, level.lower(), logger.info)
        context = get_current_request_context()
        context.update(extra_context)

        # Format context as string for standard logger
        context_str = " ".join([f"{k}={v}" for k, v in context.items() if v is not None])
        log_func(f"{message} [{context_str}]")


def create_child_logger(name: str):
    """
    Create a child logger that inherits the current request context.

    Args:
        name: Name for the child logger (typically module name)

    Returns:
        Configured logger instance
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)
