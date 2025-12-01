# tests/sandbox/test_base.py
"""
Unit tests for sandbox base classes and data models.

Tests:
    - SandboxConfig serialization and validation
    - ExecutionResult formatting
    - FileInfo data model
    - SandboxAccessLevel enum
    - SandboxStatus enum
"""

import pytest
from datetime import datetime

from llmcore.agents.sandbox.base import (
    SandboxProvider,
    SandboxConfig,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo
)


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.access_level == SandboxAccessLevel.RESTRICTED
        assert config.timeout_seconds == 600
        assert config.memory_limit == "1g"
        assert config.cpu_limit == 2.0
        assert config.network_enabled is False
        assert config.working_directory == "/workspace"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            access_level=SandboxAccessLevel.FULL,
            timeout_seconds=300,
            memory_limit="2g",
            cpu_limit=4.0,
            network_enabled=True
        )

        assert config.access_level == SandboxAccessLevel.FULL
        assert config.timeout_seconds == 300
        assert config.memory_limit == "2g"
        assert config.cpu_limit == 4.0
        assert config.network_enabled is True

    def test_unique_sandbox_id(self):
        """Test that each config gets a unique sandbox ID."""
        config1 = SandboxConfig()
        config2 = SandboxConfig()

        assert config1.sandbox_id != config2.sandbox_id
        assert len(config1.sandbox_id) == 36  # UUID format

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        config = SandboxConfig(
            timeout_seconds=120,
            labels={"env": "test"}
        )

        data = config.to_dict()

        assert data["sandbox_id"] == config.sandbox_id
        assert data["access_level"] == "restricted"
        assert data["timeout_seconds"] == 120
        assert data["labels"] == {"env": "test"}
        assert "created_at" in data

    def test_path_string_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = SandboxConfig(
            share_mount_host="~/test_share",
            output_mount_host="/tmp/test_output"
        )

        from pathlib import Path
        assert isinstance(config.share_mount_host, Path)
        assert isinstance(config.output_mount_host, Path)


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_property(self):
        """Test success property for various results."""
        # Successful execution
        success = ExecutionResult(exit_code=0, stdout="output")
        assert success.success is True

        # Failed execution
        failure = ExecutionResult(exit_code=1, stderr="error")
        assert failure.success is False

        # Timed out
        timeout = ExecutionResult(exit_code=0, timed_out=True)
        assert timeout.success is False

    def test_to_tool_output_success(self):
        """Test tool output formatting for successful execution."""
        result = ExecutionResult(exit_code=0, stdout="Hello, World!")

        output = result.to_tool_output()

        assert "Hello, World!" in output
        assert "EXIT CODE" not in output

    def test_to_tool_output_failure(self):
        """Test tool output formatting for failed execution."""
        result = ExecutionResult(exit_code=1, stdout="", stderr="Error occurred")

        output = result.to_tool_output()

        assert "EXIT CODE: 1" in output
        assert "Error occurred" in output

    def test_to_tool_output_timeout(self):
        """Test tool output formatting for timed out execution."""
        result = ExecutionResult(
            exit_code=-1,
            timed_out=True,
            execution_time_seconds=60.0
        )

        output = result.to_tool_output()

        assert "TIMED OUT" in output

    def test_to_tool_output_truncated(self):
        """Test tool output formatting with truncation notice."""
        result = ExecutionResult(exit_code=0, stdout="output", truncated=True)

        output = result.to_tool_output()

        assert "truncated" in output.lower()

    def test_to_tool_output_no_output(self):
        """Test tool output formatting when command produces no output."""
        result = ExecutionResult(exit_code=0, stdout="", stderr="")

        output = result.to_tool_output()

        assert "no output" in output.lower()

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        result = ExecutionResult(
            exit_code=0,
            stdout="output",
            stderr="",
            execution_time_seconds=1.5,
            metadata={"test": True}
        )

        data = result.to_dict()

        assert data["exit_code"] == 0
        assert data["stdout"] == "output"
        assert data["success"] is True
        assert data["execution_time_seconds"] == 1.5
        assert data["metadata"]["test"] is True


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self):
        """Test FileInfo creation."""
        info = FileInfo(
            path="/workspace/test.py",
            name="test.py",
            is_directory=False,
            size_bytes=1024
        )

        assert info.path == "/workspace/test.py"
        assert info.name == "test.py"
        assert info.is_directory is False
        assert info.size_bytes == 1024

    def test_directory_info(self):
        """Test FileInfo for directories."""
        info = FileInfo(
            path="/workspace/data",
            name="data",
            is_directory=True
        )

        assert info.is_directory is True
        assert info.size_bytes == 0  # Default

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        info = FileInfo(
            path="/test.txt",
            name="test.txt",
            is_directory=False,
            size_bytes=500,
            permissions="rw-r--r--",
            modified_at=datetime(2024, 1, 1, 12, 0, 0)
        )

        data = info.to_dict()

        assert data["path"] == "/test.txt"
        assert data["name"] == "test.txt"
        assert data["is_directory"] is False
        assert data["size_bytes"] == 500
        assert data["permissions"] == "rw-r--r--"
        assert "2024-01-01" in data["modified_at"]


class TestSandboxAccessLevel:
    """Tests for SandboxAccessLevel enum."""

    def test_restricted_value(self):
        """Test RESTRICTED enum value."""
        assert SandboxAccessLevel.RESTRICTED.value == "restricted"

    def test_full_value(self):
        """Test FULL enum value."""
        assert SandboxAccessLevel.FULL.value == "full"

    def test_comparison(self):
        """Test enum comparisons."""
        assert SandboxAccessLevel.RESTRICTED != SandboxAccessLevel.FULL
        assert SandboxAccessLevel.RESTRICTED == SandboxAccessLevel.RESTRICTED


class TestSandboxStatus:
    """Tests for SandboxStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        statuses = [
            SandboxStatus.CREATED,
            SandboxStatus.INITIALIZING,
            SandboxStatus.READY,
            SandboxStatus.EXECUTING,
            SandboxStatus.PAUSED,
            SandboxStatus.ERROR,
            SandboxStatus.CLEANING_UP,
            SandboxStatus.TERMINATED
        ]

        for status in statuses:
            assert status is not None
            assert isinstance(status.value, str)

    def test_lifecycle_transitions(self):
        """Test typical lifecycle status transitions."""
        lifecycle = [
            SandboxStatus.CREATED,
            SandboxStatus.INITIALIZING,
            SandboxStatus.READY,
            SandboxStatus.EXECUTING,
            SandboxStatus.READY,
            SandboxStatus.CLEANING_UP,
            SandboxStatus.TERMINATED
        ]

        # Just verify all statuses are valid
        for status in lifecycle:
            assert status in SandboxStatus


class TestSandboxProviderAbstract:
    """Tests for SandboxProvider abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that SandboxProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SandboxProvider()

    def test_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        abstract_methods = [
            "initialize",
            "execute_shell",
            "execute_python",
            "write_file",
            "read_file",
            "write_file_binary",
            "read_file_binary",
            "list_files",
            "file_exists",
            "delete_file",
            "create_directory",
            "cleanup",
            "is_healthy",
            "get_access_level",
            "get_status",
            "get_config",
            "get_info"
        ]

        for method_name in abstract_methods:
            assert hasattr(SandboxProvider, method_name)
