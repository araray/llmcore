# src/llmcore/agents/sandbox/exceptions.py
"""
Sandbox-specific exceptions for the LLMCore agent system.

This module defines a hierarchy of exceptions that can occur during
sandbox operations, enabling precise error handling and informative
error messages for debugging and user feedback.

Exception Hierarchy:
    SandboxError (base)
    ├── SandboxInitializationError - Failed to create/start sandbox
    ├── SandboxExecutionError - Command/code execution failed
    ├── SandboxTimeoutError - Operation exceeded time limit
    ├── SandboxAccessDenied - Security policy violation
    ├── SandboxResourceError - Resource limits exceeded
    ├── SandboxConnectionError - Failed to connect (VM/remote Docker)
    └── SandboxCleanupError - Failed to cleanup resources
"""

from typing import Any, Dict, Optional


class SandboxError(Exception):
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
        message: str,
        details: dict[str, Any] | None = None,
        sandbox_id: str | None = None,
    ):
        """
        Initialize the sandbox error.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
            sandbox_id: ID of the affected sandbox
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.sandbox_id = sandbox_id

    def __str__(self) -> str:
        """Return formatted error message."""
        base_msg = self.message
        if self.sandbox_id:
            base_msg = f"[Sandbox {self.sandbox_id[:8]}] {base_msg}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} ({detail_str})"
        return base_msg

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "sandbox_id": self.sandbox_id,
        }


class SandboxInitializationError(SandboxError):
    """
    Raised when sandbox creation or initialization fails.

    This can occur due to:
    - Docker daemon not running
    - Image pull failure
    - SSH connection failure
    - Insufficient permissions
    - Resource allocation failure

    Example:
        >>> raise SandboxInitializationError(
        ...     "Failed to pull Docker image",
        ...     details={"image": "python:3.11-slim", "reason": "network timeout"}
        ... )
    """

    pass


class SandboxExecutionError(SandboxError):
    """
    Raised when command or code execution fails within the sandbox.

    This is for execution-level failures, not for commands that
    return non-zero exit codes (those are captured in ExecutionResult).

    This can occur due to:
    - Sandbox not initialized
    - Internal sandbox errors
    - Communication failures

    Attributes:
        command: The command/code that failed to execute
        exit_code: Exit code if available
        stdout: Standard output if available
        stderr: Standard error if available
    """

    def __init__(
        self,
        message: str,
        command: str | None = None,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        **kwargs,
    ):
        """
        Initialize the execution error.

        Args:
            message: Human-readable error description
            command: The command/code that failed
            exit_code: Exit code if available
            stdout: Standard output if available
            stderr: Standard error if available
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "command": self.command[:200] if self.command else None,
                "exit_code": self.exit_code,
                "stdout": self.stdout[:500] if self.stdout else None,
                "stderr": self.stderr[:500] if self.stderr else None,
            }
        )
        return result


class SandboxTimeoutError(SandboxError):
    """
    Raised when a sandbox operation exceeds its time limit.

    This is a specific type of execution error that indicates
    the operation was forcibly terminated due to timeout.

    Attributes:
        timeout_seconds: The timeout that was exceeded
        operation: Description of the operation that timed out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        **kwargs,
    ):
        """
        Initialize the timeout error.

        Args:
            message: Human-readable error description
            timeout_seconds: The timeout that was exceeded
            operation: Description of the operation
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update({"timeout_seconds": self.timeout_seconds, "operation": self.operation})
        return result


class SandboxAccessDenied(SandboxError):
    """
    Raised when a security policy prevents an operation.

    This occurs when:
    - Image is not in whitelist
    - Tool is not allowed for access level
    - Attempting to access restricted resources
    - Host is not in allowed list

    Attributes:
        resource: The resource that was denied
        reason: Why access was denied
        policy: The policy that was violated
    """

    def __init__(
        self,
        message: str,
        resource: str | None = None,
        reason: str | None = None,
        policy: str | None = None,
        **kwargs,
    ):
        """
        Initialize the access denied error.

        Args:
            message: Human-readable error description
            resource: The resource that was denied
            reason: Why access was denied
            policy: The policy that was violated
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.resource = resource
        self.reason = reason
        self.policy = policy

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update({"resource": self.resource, "reason": self.reason, "policy": self.policy})
        return result


class SandboxResourceError(SandboxError):
    """
    Raised when sandbox resource limits are exceeded.

    This occurs when:
    - Memory limit exceeded (OOM)
    - CPU quota exceeded
    - Disk space exhausted
    - Too many processes/files

    Attributes:
        resource_type: Type of resource (memory, cpu, disk, etc.)
        limit: The limit that was set
        actual: The actual value that exceeded the limit
    """

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        limit: str | None = None,
        actual: str | None = None,
        **kwargs,
    ):
        """
        Initialize the resource error.

        Args:
            message: Human-readable error description
            resource_type: Type of resource
            limit: The limit that was set
            actual: The actual value
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update(
            {"resource_type": self.resource_type, "limit": self.limit, "actual": self.actual}
        )
        return result


class SandboxConnectionError(SandboxError):
    """
    Raised when connection to a remote sandbox fails.

    This is specific to VM sandboxes and remote Docker hosts.

    This occurs when:
    - SSH connection fails
    - Remote Docker API unreachable
    - Authentication failure
    - Network timeout

    Attributes:
        host: The host that couldn't be reached
        port: The port used
        connection_type: Type of connection (ssh, docker_api, etc.)
    """

    def __init__(
        self,
        message: str,
        host: str | None = None,
        port: int | None = None,
        connection_type: str | None = None,
        **kwargs,
    ):
        """
        Initialize the connection error.

        Args:
            message: Human-readable error description
            host: The host that couldn't be reached
            port: The port used
            connection_type: Type of connection
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port
        self.connection_type = connection_type

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update(
            {"host": self.host, "port": self.port, "connection_type": self.connection_type}
        )
        return result


class SandboxCleanupError(SandboxError):
    """
    Raised when sandbox cleanup fails.

    This is non-fatal but indicates resources may be leaked.

    This occurs when:
    - Container removal fails
    - Temporary files can't be deleted
    - SSH session can't be closed
    - Volume unmount fails

    Attributes:
        resources_leaked: List of resources that weren't cleaned
        partial_cleanup: Whether some cleanup succeeded
    """

    def __init__(
        self,
        message: str,
        resources_leaked: list | None = None,
        partial_cleanup: bool = False,
        **kwargs,
    ):
        """
        Initialize the cleanup error.

        Args:
            message: Human-readable error description
            resources_leaked: List of resources not cleaned
            partial_cleanup: Whether some cleanup succeeded
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.resources_leaked = resources_leaked or []
        self.partial_cleanup = partial_cleanup

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update(
            {"resources_leaked": self.resources_leaked, "partial_cleanup": self.partial_cleanup}
        )
        return result


class SandboxNotInitializedError(SandboxError):
    """
    Raised when operations are attempted on an uninitialized sandbox.

    This is a programming error that indicates the sandbox was used
    before calling initialize() or after calling cleanup().
    """

    def __init__(self, message: str = "Sandbox not initialized", **kwargs):
        """
        Initialize the error.

        Args:
            message: Human-readable error description
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)


class SandboxImageNotFoundError(SandboxError):
    """
    Raised when a Docker image cannot be found or pulled.

    Attributes:
        image: The image that wasn't found
        registry: The registry that was searched
    """

    def __init__(
        self, message: str, image: str | None = None, registry: str | None = None, **kwargs
    ):
        """
        Initialize the image not found error.

        Args:
            message: Human-readable error description
            image: The image that wasn't found
            registry: The registry searched
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.image = image
        self.registry = registry

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary."""
        result = super().to_dict()
        result.update({"image": self.image, "registry": self.registry})
        return result
