# src/llmcore/exceptions.py
"""
Custom exceptions for the LLMCore library.

This module defines a hierarchy of custom exception classes to provide
more specific error information and allow for targeted error handling
by applications using LLMCore.

Exception Hierarchy:
    LLMCoreError (base)
    ├── ConfigError - Configuration loading or validation errors
    ├── ProviderError - LLM provider API errors
    ├── StorageError - Storage operation errors
    │   ├── SessionStorageError - Session storage specific
    │   ├── VectorStorageError - Vector storage specific
    │   └── SessionNotFoundError - Session lookup failure
    ├── ContextError - Context management errors
    │   └── ContextLengthError - Context exceeds model limits
    ├── EmbeddingError - Embedding generation errors
    └── SandboxError - Sandbox execution errors (NEW in v0.26.0)
        ├── SandboxInitializationError - Failed to create/start sandbox
        ├── SandboxExecutionError - Command/code execution failed
        ├── SandboxTimeoutError - Operation exceeded time limit
        ├── SandboxAccessDenied - Security policy violation
        ├── SandboxResourceError - Resource limits exceeded
        ├── SandboxConnectionError - Failed to connect (VM/remote Docker)
        └── SandboxCleanupError - Failed to cleanup resources
"""

from typing import Any, Dict, Optional

# =============================================================================
# BASE EXCEPTION
# =============================================================================

class LLMCoreError(Exception):
    """Base class for all LLMCore specific errors."""
    def __init__(self, message: str = "An unspecified error occurred in LLMCore."):
        super().__init__(message)


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigError(LLMCoreError):
    """Raised for errors related to configuration loading or validation."""
    def __init__(self, message: str = "Configuration error."):
        super().__init__(message)


# =============================================================================
# PROVIDER EXCEPTIONS
# =============================================================================

class ProviderError(LLMCoreError):
    """Raised for errors originating from an LLM provider (e.g., API errors, connection issues)."""
    def __init__(self, provider_name: str = "Unknown", message: str = "Provider error."):
        self.provider_name = provider_name
        super().__init__(f"Error with provider '{provider_name}': {message}")


# =============================================================================
# STORAGE EXCEPTIONS
# =============================================================================

class StorageError(LLMCoreError):
    """Base class for errors related to storage operations."""
    def __init__(self, message: str = "Storage error."):
        super().__init__(message)


class SessionStorageError(StorageError):
    """Raised for errors specific to session storage operations."""
    def __init__(self, message: str = "Session storage error."):
        super().__init__(message)


class VectorStorageError(StorageError):
    """Raised for errors specific to vector storage operations."""
    def __init__(self, message: str = "Vector storage error."):
        super().__init__(message)


class SessionNotFoundError(StorageError):
    """
    Raised when a specified session ID is not found in storage.
    Inherits from StorageError as it's a storage-related lookup failure.
    """
    def __init__(self, session_id: str, message: str = "Session not found."):
        self.session_id = session_id
        super().__init__(f"{message} Session ID: '{session_id}'")


class StorageUnavailableError(StorageError):
    """
    Raised when storage backend is unavailable (e.g., circuit breaker is open).

    This indicates a transient failure condition where the backend may recover.
    Applications can catch this to implement fallback behavior.
    """
    def __init__(
        self,
        backend_name: str = "unknown",
        message: str = "Storage backend is unavailable.",
        retry_after_seconds: Optional[int] = None
    ):
        self.backend_name = backend_name
        self.retry_after_seconds = retry_after_seconds
        detail = f" Retry after {retry_after_seconds}s." if retry_after_seconds else ""
        super().__init__(f"{message} Backend: '{backend_name}'.{detail}")


class StorageHealthError(StorageError):
    """
    Raised when storage health check fails.

    Contains diagnostic information to help identify the issue.
    """
    def __init__(
        self,
        backend_name: str = "unknown",
        check_type: str = "connectivity",
        message: str = "Storage health check failed.",
        latency_ms: Optional[float] = None
    ):
        self.backend_name = backend_name
        self.check_type = check_type
        self.latency_ms = latency_ms

        detail = f" (check={check_type}"
        if latency_ms is not None:
            detail += f", latency={latency_ms:.0f}ms"
        detail += ")"

        super().__init__(f"{message} Backend: '{backend_name}'{detail}")


class SchemaError(StorageError):
    """
    Raised when schema management operations fail.

    This includes schema version mismatches, migration failures,
    and DDL execution errors.
    """
    def __init__(
        self,
        message: str = "Schema operation failed.",
        current_version: Optional[int] = None,
        target_version: Optional[int] = None
    ):
        self.current_version = current_version
        self.target_version = target_version

        detail = ""
        if current_version is not None or target_version is not None:
            detail = f" (current=v{current_version}, target=v{target_version})"

        super().__init__(f"{message}{detail}")


# =============================================================================
# CONTEXT EXCEPTIONS
# =============================================================================

class ContextError(LLMCoreError):
    """Base class for errors related to context management."""
    def __init__(self, message: str = "Context management error."):
        super().__init__(message)


class ContextLengthError(ContextError):
    """Raised when the context length exceeds the model's maximum limit, even after truncation attempts."""
    def __init__(
        self,
        model_name: str = "Unknown",
        limit: int = 0,
        actual: int = 0,
        message: str = "Context length exceeded."
    ):
        self.model_name = model_name
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"{message} Model: '{model_name}', Limit: {limit} tokens, Actual: {actual} tokens."
        )


# =============================================================================
# EMBEDDING EXCEPTIONS
# =============================================================================

class EmbeddingError(LLMCoreError):
    """Raised for errors related to embedding generation."""
    def __init__(self, model_name: str = "Unknown", message: str = "Embedding generation error."):
        self.model_name = model_name
        super().__init__(f"Error with embedding model '{model_name}': {message}")


# =============================================================================
# SANDBOX EXCEPTIONS (NEW in v0.26.0)
# =============================================================================
# These exceptions are re-exported from llmcore.agents.sandbox.exceptions
# for convenience. The canonical definitions are in the sandbox module.

class SandboxError(LLMCoreError):
    """
    Base exception for all sandbox-related errors.

    All sandbox exceptions inherit from this class, enabling
    catch-all handling when specific error types don't matter.

    Attributes:
        message: Human-readable error description
        details: Optional dictionary with additional context
        sandbox_id: ID of the affected sandbox (if known)
    """

    def __init__(
        self,
        message: str = "Sandbox error.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.sandbox_id = sandbox_id

        # Build formatted message
        formatted = message
        if sandbox_id:
            formatted = f"[Sandbox {sandbox_id[:8]}] {formatted}"
        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            formatted = f"{formatted} ({detail_str})"

        super().__init__(formatted)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "sandbox_id": self.sandbox_id
        }


class SandboxInitializationError(SandboxError):
    """
    Raised when sandbox creation or initialization fails.

    This can occur due to:
    - Docker daemon not running
    - Image not found or pull failed
    - SSH connection failed
    - Resource allocation failed
    - Configuration errors
    """

    def __init__(
        self,
        message: str = "Failed to initialize sandbox.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        provider: Optional[str] = None
    ):
        self.provider = provider
        if provider and "provider" not in (details or {}):
            details = details or {}
            details["provider"] = provider
        super().__init__(message, details, sandbox_id)


class SandboxExecutionError(SandboxError):
    """
    Raised when command or code execution fails within the sandbox.

    This indicates the command ran but returned a non-zero exit code
    or produced an error output.
    """

    def __init__(
        self,
        message: str = "Execution failed in sandbox.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        exit_code: Optional[int] = None,
        stderr: Optional[str] = None
    ):
        self.exit_code = exit_code
        self.stderr = stderr
        details = details or {}
        if exit_code is not None:
            details["exit_code"] = exit_code
        if stderr:
            details["stderr"] = stderr[:200]  # Truncate for readability
        super().__init__(message, details, sandbox_id)


class SandboxTimeoutError(SandboxError):
    """
    Raised when a sandbox operation exceeds its time limit.

    Timeouts are enforced to prevent runaway processes and
    ensure resource availability.
    """

    def __init__(
        self,
        message: str = "Sandbox operation timed out.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        operation: Optional[str] = None
    ):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(message, details, sandbox_id)


class SandboxAccessDenied(SandboxError):
    """
    Raised when an operation is denied due to security policy.

    This occurs when:
    - A tool is not in the allowed list for RESTRICTED access
    - An image/host is not whitelisted
    - A network operation is attempted with network disabled
    - A privileged operation is attempted without FULL access
    """

    def __init__(
        self,
        message: str = "Access denied by sandbox security policy.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        required_level: Optional[str] = None,
        current_level: Optional[str] = None,
        denied_operation: Optional[str] = None
    ):
        self.required_level = required_level
        self.current_level = current_level
        self.denied_operation = denied_operation
        details = details or {}
        if required_level:
            details["required_level"] = required_level
        if current_level:
            details["current_level"] = current_level
        if denied_operation:
            details["denied_operation"] = denied_operation
        super().__init__(message, details, sandbox_id)


class SandboxResourceError(SandboxError):
    """
    Raised when sandbox resource limits are exceeded.

    This includes:
    - Memory limit exceeded
    - CPU quota exceeded
    - Disk space exhausted
    - Too many concurrent sandboxes
    """

    def __init__(
        self,
        message: str = "Sandbox resource limit exceeded.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: Optional[str] = None,
        actual: Optional[str] = None
    ):
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if limit:
            details["limit"] = limit
        if actual:
            details["actual"] = actual
        super().__init__(message, details, sandbox_id)


class SandboxConnectionError(SandboxError):
    """
    Raised when connection to sandbox backend fails.

    This can occur with:
    - Remote Docker daemons
    - SSH connections to VMs
    - Network issues
    """

    def __init__(
        self,
        message: str = "Failed to connect to sandbox backend.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        self.host = host
        self.port = port
        details = details or {}
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        super().__init__(message, details, sandbox_id)


class SandboxCleanupError(SandboxError):
    """
    Raised when sandbox cleanup fails.

    This is typically non-fatal but indicates orphaned resources
    that may need manual cleanup.
    """

    def __init__(
        self,
        message: str = "Failed to cleanup sandbox resources.",
        details: Optional[Dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        partial_cleanup: bool = False
    ):
        self.partial_cleanup = partial_cleanup
        details = details or {}
        details["partial_cleanup"] = partial_cleanup
        super().__init__(message, details, sandbox_id)


# =============================================================================
# FUTURE EXCEPTIONS (placeholders)
# =============================================================================
# These can be uncommented and implemented as needed:
#
# class AuthenticationError(ProviderError):
#     """Raised for authentication failures with an LLM provider."""
#     def __init__(self, provider_name: str, message: str = "Authentication failed."):
#         super().__init__(provider_name, message)
#
# class RateLimitError(ProviderError):
#     """Raised when an LLM provider's rate limit is exceeded."""
#     def __init__(self, provider_name: str, message: str = "Rate limit exceeded."):
#         super().__init__(provider_name, message)
