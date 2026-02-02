# tests/test_exceptions.py
"""
Comprehensive tests for the llmcore.exceptions module.

Tests all exception classes, their inheritance, attributes,
message formatting, and serialization.
"""

import pytest

from llmcore.exceptions import (
    ConfigError,
    ContextError,
    ContextLengthError,
    EmbeddingError,
    LLMCoreError,
    ProviderError,
    SandboxAccessDenied,
    SandboxCleanupError,
    SandboxConnectionError,
    SandboxError,
    SandboxExecutionError,
    SandboxInitializationError,
    SandboxResourceError,
    SandboxTimeoutError,
    SessionNotFoundError,
    SessionStorageError,
    StorageError,
    VectorStorageError,
)


class TestLLMCoreError:
    """Tests for the base LLMCoreError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = LLMCoreError()
        assert "unspecified error" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = LLMCoreError("Custom error message")
        assert str(error) == "Custom error message"

    def test_is_exception(self):
        """Test that it inherits from Exception."""
        error = LLMCoreError()
        assert isinstance(error, Exception)

    def test_can_be_raised(self):
        """Test that it can be raised and caught."""
        with pytest.raises(LLMCoreError):
            raise LLMCoreError("Test error")


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = ConfigError()
        assert "configuration error" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = ConfigError("Invalid API key format")
        assert "Invalid API key format" in str(error)

    def test_inherits_llmcore_error(self):
        """Test inheritance from LLMCoreError."""
        error = ConfigError()
        assert isinstance(error, LLMCoreError)


class TestProviderError:
    """Tests for ProviderError exception."""

    def test_default_values(self):
        """Test default provider name and message."""
        error = ProviderError()
        assert "Unknown" in str(error)
        assert "Provider error" in str(error)

    def test_provider_name_attribute(self):
        """Test provider_name attribute is stored."""
        error = ProviderError(provider_name="openai")
        assert error.provider_name == "openai"

    def test_custom_message(self):
        """Test custom message with provider."""
        error = ProviderError(provider_name="anthropic", message="Rate limited")
        assert "anthropic" in str(error)
        assert "Rate limited" in str(error)


class TestStorageError:
    """Tests for StorageError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = StorageError()
        assert "storage error" in str(error).lower()

    def test_inherits_llmcore_error(self):
        """Test inheritance from LLMCoreError."""
        error = StorageError()
        assert isinstance(error, LLMCoreError)


class TestSessionStorageError:
    """Tests for SessionStorageError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = SessionStorageError()
        assert "session storage error" in str(error).lower()

    def test_inherits_storage_error(self):
        """Test inheritance from StorageError."""
        error = SessionStorageError()
        assert isinstance(error, StorageError)
        assert isinstance(error, LLMCoreError)


class TestVectorStorageError:
    """Tests for VectorStorageError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = VectorStorageError()
        assert "vector storage error" in str(error).lower()

    def test_inherits_storage_error(self):
        """Test inheritance from StorageError."""
        error = VectorStorageError()
        assert isinstance(error, StorageError)


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError exception."""

    def test_session_id_in_message(self):
        """Test session ID appears in message."""
        error = SessionNotFoundError(session_id="test-session-123")
        assert "test-session-123" in str(error)

    def test_session_id_attribute(self):
        """Test session_id attribute is stored."""
        error = SessionNotFoundError(session_id="abc-123")
        assert error.session_id == "abc-123"

    def test_inherits_storage_error(self):
        """Test inheritance from StorageError."""
        error = SessionNotFoundError(session_id="test")
        assert isinstance(error, StorageError)


class TestContextError:
    """Tests for ContextError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = ContextError()
        assert "context" in str(error).lower()

    def test_inherits_llmcore_error(self):
        """Test inheritance from LLMCoreError."""
        error = ContextError()
        assert isinstance(error, LLMCoreError)


class TestContextLengthError:
    """Tests for ContextLengthError exception."""

    def test_attributes_stored(self):
        """Test all attributes are stored."""
        error = ContextLengthError(model_name="gpt-4", limit=8192, actual=10000)
        assert error.model_name == "gpt-4"
        assert error.limit == 8192
        assert error.actual == 10000

    def test_message_includes_details(self):
        """Test message includes model, limit, and actual."""
        error = ContextLengthError(model_name="claude-3", limit=200000, actual=250000)
        msg = str(error)
        assert "claude-3" in msg
        assert "200000" in msg
        assert "250000" in msg

    def test_inherits_context_error(self):
        """Test inheritance from ContextError."""
        error = ContextLengthError()
        assert isinstance(error, ContextError)


class TestEmbeddingError:
    """Tests for EmbeddingError exception."""

    def test_model_name_attribute(self):
        """Test model_name attribute is stored."""
        error = EmbeddingError(model_name="text-embedding-3-small")
        assert error.model_name == "text-embedding-3-small"

    def test_message_includes_model(self):
        """Test message includes model name."""
        error = EmbeddingError(model_name="ada-002", message="Dimension mismatch")
        msg = str(error)
        assert "ada-002" in msg
        assert "Dimension mismatch" in msg


class TestSandboxError:
    """Tests for SandboxError base exception."""

    def test_default_message(self):
        """Test default error message."""
        error = SandboxError()
        assert "sandbox error" in str(error).lower()

    def test_details_attribute(self):
        """Test details dictionary is stored."""
        details = {"key": "value", "number": 42}
        error = SandboxError(details=details)
        assert error.details == details

    def test_sandbox_id_in_message(self):
        """Test sandbox_id appears in formatted message."""
        error = SandboxError(message="Test error", sandbox_id="abcdef123456")
        assert "abcdef12" in str(error)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        error = SandboxError(
            message="Test error",
            details={"key": "value"},
            sandbox_id="sandbox-123"
        )
        d = error.to_dict()
        assert d["error_type"] == "SandboxError"
        assert d["message"] == "Test error"
        assert d["details"] == {"key": "value"}
        assert d["sandbox_id"] == "sandbox-123"


class TestSandboxInitializationError:
    """Tests for SandboxInitializationError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = SandboxInitializationError()
        assert "initialize" in str(error).lower()

    def test_provider_attribute(self):
        """Test provider attribute is stored."""
        error = SandboxInitializationError(provider="docker")
        assert error.provider == "docker"

    def test_inherits_sandbox_error(self):
        """Test inheritance from SandboxError."""
        error = SandboxInitializationError()
        assert isinstance(error, SandboxError)


class TestSandboxExecutionError:
    """Tests for SandboxExecutionError exception."""

    def test_exit_code_attribute(self):
        """Test exit_code attribute is stored."""
        error = SandboxExecutionError(exit_code=1)
        assert error.exit_code == 1

    def test_stderr_attribute(self):
        """Test stderr attribute is stored."""
        error = SandboxExecutionError(stderr="Error output")
        assert error.stderr == "Error output"

    def test_stderr_truncated(self):
        """Test long stderr is truncated in details."""
        long_stderr = "x" * 500
        error = SandboxExecutionError(stderr=long_stderr)
        assert len(error.details.get("stderr", "")) <= 200


class TestSandboxTimeoutError:
    """Tests for SandboxTimeoutError exception."""

    def test_timeout_seconds_attribute(self):
        """Test timeout_seconds attribute is stored."""
        error = SandboxTimeoutError(timeout_seconds=30)
        assert error.timeout_seconds == 30

    def test_operation_attribute(self):
        """Test operation attribute is stored."""
        error = SandboxTimeoutError(operation="execute_python")
        assert error.operation == "execute_python"


class TestSandboxAccessDenied:
    """Tests for SandboxAccessDenied exception."""

    def test_levels_attributes(self):
        """Test access level attributes."""
        error = SandboxAccessDenied(required_level="FULL", current_level="RESTRICTED")
        assert error.required_level == "FULL"
        assert error.current_level == "RESTRICTED"


class TestSandboxResourceError:
    """Tests for SandboxResourceError exception."""

    def test_resource_attributes(self):
        """Test resource attributes."""
        error = SandboxResourceError(resource_type="memory", limit="512MB", actual="1GB")
        assert error.resource_type == "memory"
        assert error.limit == "512MB"
        assert error.actual == "1GB"


class TestSandboxConnectionError:
    """Tests for SandboxConnectionError exception."""

    def test_host_port_attributes(self):
        """Test host and port attributes."""
        error = SandboxConnectionError(host="localhost", port=22)
        assert error.host == "localhost"
        assert error.port == 22


class TestSandboxCleanupError:
    """Tests for SandboxCleanupError exception."""

    def test_partial_cleanup_attribute(self):
        """Test partial_cleanup attribute."""
        error = SandboxCleanupError(partial_cleanup=True)
        assert error.partial_cleanup is True


class TestExceptionHierarchy:
    """Tests for the exception inheritance hierarchy."""

    def test_all_inherit_from_llmcore_error(self):
        """Test all exceptions inherit from LLMCoreError."""
        exceptions = [
            ConfigError(),
            ProviderError(),
            StorageError(),
            SessionStorageError(),
            VectorStorageError(),
            SessionNotFoundError(session_id="test"),
            ContextError(),
            ContextLengthError(),
            EmbeddingError(),
            SandboxError(),
            SandboxInitializationError(),
            SandboxExecutionError(),
            SandboxTimeoutError(),
            SandboxAccessDenied(),
            SandboxResourceError(),
            SandboxConnectionError(),
            SandboxCleanupError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, LLMCoreError), f"{type(exc).__name__} should inherit from LLMCoreError"

    def test_sandbox_hierarchy(self):
        """Test sandbox exception hierarchy."""
        sandbox_exceptions = [
            SandboxInitializationError,
            SandboxExecutionError,
            SandboxTimeoutError,
            SandboxAccessDenied,
            SandboxResourceError,
            SandboxConnectionError,
            SandboxCleanupError,
        ]
        for exc_class in sandbox_exceptions:
            assert issubclass(exc_class, SandboxError)


class TestExceptionCatching:
    """Tests for exception catching patterns."""

    def test_catch_sandbox_catches_all_sandbox(self):
        """Test SandboxError catches all sandbox exceptions."""
        sandbox_exceptions = [
            SandboxInitializationError(),
            SandboxExecutionError(),
            SandboxTimeoutError(),
            SandboxAccessDenied(),
            SandboxResourceError(),
            SandboxConnectionError(),
            SandboxCleanupError(),
        ]

        for exc in sandbox_exceptions:
            caught = False
            try:
                raise exc
            except SandboxError:
                caught = True
            assert caught, f"SandboxError should catch {type(exc).__name__}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
