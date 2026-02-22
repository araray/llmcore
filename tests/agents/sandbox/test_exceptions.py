# tests/agents/sandbox/test_exceptions.py
# tests/sandbox/test_exceptions.py
"""
Unit tests for sandbox exception classes.

Tests:
    - Exception inheritance hierarchy
    - Error message formatting
    - Serialization to dictionary
    - Custom attributes
"""

from llmcore.agents.sandbox.exceptions import (
    SandboxAccessDenied,
    SandboxCleanupError,
    SandboxConnectionError,
    SandboxError,
    SandboxExecutionError,
    SandboxImageNotFoundError,
    SandboxInitializationError,
    SandboxNotInitializedError,
    SandboxResourceError,
    SandboxTimeoutError,
)


class TestSandboxError:
    """Tests for SandboxError base class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = SandboxError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
        assert error.sandbox_id is None

    def test_error_with_details(self):
        """Test error with details dictionary."""
        error = SandboxError("Test error", details={"key": "value", "count": 42})

        assert error.details == {"key": "value", "count": 42}
        assert "key=value" in str(error)

    def test_error_with_sandbox_id(self):
        """Test error with sandbox ID."""
        error = SandboxError("Test error", sandbox_id="abc12345-def6-7890-ghij-klmnopqrstuv")

        assert error.sandbox_id == "abc12345-def6-7890-ghij-klmnopqrstuv"
        # First 8 chars of sandbox_id in message
        assert "[Sandbox abc12345]" in str(error)

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        error = SandboxError("Test error", details={"reason": "test"}, sandbox_id="test-id")

        data = error.to_dict()

        assert data["error_type"] == "SandboxError"
        assert data["message"] == "Test error"
        assert data["details"]["reason"] == "test"
        assert data["sandbox_id"] == "test-id"


class TestSandboxInitializationError:
    """Tests for SandboxInitializationError."""

    def test_inheritance(self):
        """Test exception inherits from SandboxError."""
        error = SandboxInitializationError("Init failed")

        assert isinstance(error, SandboxError)
        assert isinstance(error, Exception)

    def test_creation(self):
        """Test error creation."""
        error = SandboxInitializationError(
            "Failed to pull Docker image",
            details={"image": "python:3.11", "reason": "network timeout"},
        )

        assert "Failed to pull Docker image" in str(error)
        assert error.details["image"] == "python:3.11"


class TestSandboxExecutionError:
    """Tests for SandboxExecutionError."""

    def test_execution_error_attributes(self):
        """Test execution error custom attributes."""
        error = SandboxExecutionError(
            "Command failed", command="rm -rf /", exit_code=1, stdout="", stderr="Permission denied"
        )

        assert error.command == "rm -rf /"
        assert error.exit_code == 1
        assert error.stdout == ""
        assert error.stderr == "Permission denied"

    def test_to_dict_with_execution_details(self):
        """Test serialization includes execution details."""
        error = SandboxExecutionError(
            "Failed", command="test command", exit_code=127, stdout="out", stderr="err"
        )

        data = error.to_dict()

        assert data["error_type"] == "SandboxExecutionError"
        assert data["exit_code"] == 127
        assert data["stdout"] == "out"
        assert data["stderr"] == "err"

    def test_command_truncation(self):
        """Test that long commands are truncated in serialization."""
        long_command = "x" * 500
        error = SandboxExecutionError("Failed", command=long_command)

        data = error.to_dict()

        assert len(data["command"]) == 200  # Truncated


class TestSandboxTimeoutError:
    """Tests for SandboxTimeoutError."""

    def test_timeout_attributes(self):
        """Test timeout error custom attributes."""
        error = SandboxTimeoutError(
            "Command timed out", timeout_seconds=60, operation="execute_shell"
        )

        assert error.timeout_seconds == 60
        assert error.operation == "execute_shell"

    def test_to_dict_with_timeout_details(self):
        """Test serialization includes timeout details."""
        error = SandboxTimeoutError("Timeout", timeout_seconds=30.5, operation="python execution")

        data = error.to_dict()

        assert data["timeout_seconds"] == 30.5
        assert data["operation"] == "python execution"


class TestSandboxAccessDenied:
    """Tests for SandboxAccessDenied."""

    def test_access_denied_attributes(self):
        """Test access denied error attributes."""
        error = SandboxAccessDenied(
            "Image not whitelisted",
            resource="python:latest",
            reason="Not in allowed list",
            policy="image_whitelist",
        )

        assert error.resource == "python:latest"
        assert error.reason == "Not in allowed list"
        assert error.policy == "image_whitelist"

    def test_to_dict_serialization(self):
        """Test serialization includes access details."""
        error = SandboxAccessDenied(
            "Denied", resource="/etc/passwd", reason="Outside sandbox", policy="filesystem_boundary"
        )

        data = error.to_dict()

        assert data["resource"] == "/etc/passwd"
        assert data["reason"] == "Outside sandbox"
        assert data["policy"] == "filesystem_boundary"


class TestSandboxResourceError:
    """Tests for SandboxResourceError."""

    def test_resource_error_attributes(self):
        """Test resource error attributes."""
        error = SandboxResourceError(
            "Out of memory", resource_type="memory", limit="1g", actual="1.5g"
        )

        assert error.resource_type == "memory"
        assert error.limit == "1g"
        assert error.actual == "1.5g"

    def test_to_dict_serialization(self):
        """Test serialization includes resource details."""
        error = SandboxResourceError(
            "CPU limit exceeded", resource_type="cpu", limit="2.0", actual="2.5"
        )

        data = error.to_dict()

        assert data["resource_type"] == "cpu"
        assert data["limit"] == "2.0"
        assert data["actual"] == "2.5"


class TestSandboxConnectionError:
    """Tests for SandboxConnectionError."""

    def test_connection_error_attributes(self):
        """Test connection error attributes."""
        error = SandboxConnectionError(
            "SSH connection failed", host="192.168.1.100", port=22, connection_type="ssh"
        )

        assert error.host == "192.168.1.100"
        assert error.port == 22
        assert error.connection_type == "ssh"

    def test_to_dict_serialization(self):
        """Test serialization includes connection details."""
        error = SandboxConnectionError(
            "Docker unreachable", host="docker.local", port=2375, connection_type="docker_api"
        )

        data = error.to_dict()

        assert data["host"] == "docker.local"
        assert data["port"] == 2375
        assert data["connection_type"] == "docker_api"


class TestSandboxCleanupError:
    """Tests for SandboxCleanupError."""

    def test_cleanup_error_attributes(self):
        """Test cleanup error attributes."""
        error = SandboxCleanupError(
            "Cleanup failed",
            resources_leaked=["container:abc123", "volume:data"],
            partial_cleanup=True,
        )

        assert error.resources_leaked == ["container:abc123", "volume:data"]
        assert error.partial_cleanup is True

    def test_to_dict_serialization(self):
        """Test serialization includes cleanup details."""
        error = SandboxCleanupError(
            "Partial cleanup", resources_leaked=["temp_file"], partial_cleanup=True
        )

        data = error.to_dict()

        assert data["resources_leaked"] == ["temp_file"]
        assert data["partial_cleanup"] is True


class TestSandboxNotInitializedError:
    """Tests for SandboxNotInitializedError."""

    def test_default_message(self):
        """Test default error message."""
        error = SandboxNotInitializedError()

        assert "not initialized" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = SandboxNotInitializedError("Custom message")

        assert str(error) == "Custom message"


class TestSandboxImageNotFoundError:
    """Tests for SandboxImageNotFoundError."""

    def test_image_not_found_attributes(self):
        """Test image not found error attributes."""
        error = SandboxImageNotFoundError(
            "Image not found", image="myapp:v1.0", registry="docker.io"
        )

        assert error.image == "myapp:v1.0"
        assert error.registry == "docker.io"

    def test_to_dict_serialization(self):
        """Test serialization includes image details."""
        error = SandboxImageNotFoundError(
            "Pull failed", image="private/repo:tag", registry="gcr.io"
        )

        data = error.to_dict()

        assert data["image"] == "private/repo:tag"
        assert data["registry"] == "gcr.io"


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_sandbox_error(self):
        """Test that all sandbox exceptions inherit from SandboxError."""
        exception_classes = [
            SandboxInitializationError,
            SandboxExecutionError,
            SandboxTimeoutError,
            SandboxAccessDenied,
            SandboxResourceError,
            SandboxConnectionError,
            SandboxCleanupError,
            SandboxNotInitializedError,
            SandboxImageNotFoundError,
        ]

        for exc_class in exception_classes:
            error = exc_class("test")
            assert isinstance(error, SandboxError)
            assert isinstance(error, Exception)

    def test_can_catch_all_with_sandbox_error(self):
        """Test that all sandbox exceptions can be caught with SandboxError."""
        exceptions_to_test = [
            SandboxInitializationError("test"),
            SandboxExecutionError("test"),
            SandboxTimeoutError("test"),
            SandboxAccessDenied("test"),
            SandboxResourceError("test"),
            SandboxConnectionError("test"),
            SandboxCleanupError("test"),
            SandboxNotInitializedError("test"),
            SandboxImageNotFoundError("test"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except SandboxError as e:
                assert e is exc
