# tests/agents/sandbox/test_ephemeral.py
# tests/sandbox/test_ephemeral.py
"""
Unit tests for EphemeralResourceManager.

These tests verify the ephemeral state management, logging,
and file tracking functionality.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.sandbox.base import ExecutionResult, SandboxConfig

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.ephemeral import AgentLogEntry, EphemeralResourceManager, FileRecord

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox provider."""
    sandbox = MagicMock()
    config = SandboxConfig()
    config.ephemeral_db_path = "/tmp/test_agent.db"
    sandbox.get_config.return_value = config

    # Default successful execution
    sandbox.execute_shell = AsyncMock(
        return_value=ExecutionResult(exit_code=0, stdout="", stderr="")
    )

    return sandbox


@pytest.fixture
def ephemeral_manager(mock_sandbox):
    """Create an EphemeralResourceManager with mock sandbox."""
    return EphemeralResourceManager(mock_sandbox)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestEphemeralResourceManagerInit:
    """Tests for EphemeralResourceManager initialization."""

    def test_init_with_sandbox(self, mock_sandbox):
        """Test initialization with sandbox."""
        manager = EphemeralResourceManager(mock_sandbox)

        assert manager._sandbox is mock_sandbox
        assert manager._db_path == "/tmp/test_agent.db"

    def test_init_with_custom_db_path(self, mock_sandbox):
        """Test initialization with custom database path."""
        manager = EphemeralResourceManager(mock_sandbox, db_path="/custom/path/db.sqlite")

        assert manager._db_path == "/custom/path/db.sqlite"

    def test_init_with_no_config(self):
        """Test initialization when sandbox has no config."""
        sandbox = MagicMock()
        sandbox.get_config.return_value = None

        manager = EphemeralResourceManager(sandbox)

        assert manager._db_path == "/tmp/agent_task.db"


class TestDatabaseInitialization:
    """Tests for database initialization."""

    @pytest.mark.asyncio
    async def test_init_database_success(self, ephemeral_manager, mock_sandbox):
        """Test successful database initialization."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.init_database()

        assert result is True
        mock_sandbox.execute_shell.assert_called_once()

        # Verify SQL contains expected tables
        call_args = mock_sandbox.execute_shell.call_args[0][0]
        assert "agent_state" in call_args
        assert "agent_logs" in call_args
        assert "agent_files" in call_args

    @pytest.mark.asyncio
    async def test_init_database_failure(self, ephemeral_manager, mock_sandbox):
        """Test database initialization failure."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=1, stdout="", stderr="sqlite3: command not found"
        )

        result = await ephemeral_manager.init_database()

        assert result is False


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================


class TestStateManagement:
    """Tests for state management."""

    @pytest.mark.asyncio
    async def test_set_state_string(self, ephemeral_manager, mock_sandbox):
        """Test setting string state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.set_state("my_key", "my_value")

        assert result is True

    @pytest.mark.asyncio
    async def test_set_state_number(self, ephemeral_manager, mock_sandbox):
        """Test setting number state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.set_state("counter", 42)

        assert result is True

    @pytest.mark.asyncio
    async def test_set_state_json(self, ephemeral_manager, mock_sandbox):
        """Test setting JSON state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        data = {"key": "value", "nested": {"a": 1}}
        result = await ephemeral_manager.set_state("complex", data)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_state_string(self, ephemeral_manager, mock_sandbox):
        """Test getting string state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout="stored_value|string\n", stderr=""
        )

        value = await ephemeral_manager.get_state("my_key")

        assert value == "stored_value"

    @pytest.mark.asyncio
    async def test_get_state_number(self, ephemeral_manager, mock_sandbox):
        """Test getting number state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout="42|number\n", stderr=""
        )

        value = await ephemeral_manager.get_state("counter")

        assert value == 42

    @pytest.mark.asyncio
    async def test_get_state_json(self, ephemeral_manager, mock_sandbox):
        """Test getting JSON state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout='{"key": "value"}|json\n', stderr=""
        )

        value = await ephemeral_manager.get_state("complex")

        assert value == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, ephemeral_manager, mock_sandbox):
        """Test getting nonexistent state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        value = await ephemeral_manager.get_state("nonexistent")

        assert value is None

    @pytest.mark.asyncio
    async def test_get_state_with_default(self, ephemeral_manager, mock_sandbox):
        """Test getting state with default value."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        value = await ephemeral_manager.get_state("nonexistent", default="default_value")

        assert value == "default_value"

    @pytest.mark.asyncio
    async def test_delete_state(self, ephemeral_manager, mock_sandbox):
        """Test deleting state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.delete_state("my_key")

        assert result is True

    @pytest.mark.asyncio
    async def test_list_state_keys(self, ephemeral_manager, mock_sandbox):
        """Test listing state keys."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout="key1\nkey2\nkey3\n", stderr=""
        )

        keys = await ephemeral_manager.list_state_keys()

        assert keys == ["key1", "key2", "key3"]

    @pytest.mark.asyncio
    async def test_list_state_keys_empty(self, ephemeral_manager, mock_sandbox):
        """Test listing state keys when empty."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        keys = await ephemeral_manager.list_state_keys()

        assert keys == []

    @pytest.mark.asyncio
    async def test_get_all_state(self, ephemeral_manager, mock_sandbox):
        """Test getting all state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout="key1|value1|string\nkey2|42|number\n", stderr=""
        )

        state = await ephemeral_manager.get_all_state()

        assert "key1" in state
        assert state["key1"] == "value1"
        assert state["key2"] == 42


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Tests for event logging."""

    @pytest.mark.asyncio
    async def test_log_event(self, ephemeral_manager, mock_sandbox):
        """Test logging an event."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.log_event("INFO", "Test message")

        assert result is True

    @pytest.mark.asyncio
    async def test_log_event_with_special_chars(self, ephemeral_manager, mock_sandbox):
        """Test logging event with special characters."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.log_event(
            "WARNING", "Message with 'quotes' and \"double quotes\""
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_logs(self, ephemeral_manager, mock_sandbox):
        """Test retrieving logs."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0,
            stdout="1|2024-01-15T10:30:00|INFO|Test message 1\n2|2024-01-15T10:31:00|WARNING|Test message 2\n",
            stderr="",
        )

        logs = await ephemeral_manager.get_logs()

        assert len(logs) == 2
        assert logs[0].level == "INFO"
        assert logs[0].message == "Test message 1"

    @pytest.mark.asyncio
    async def test_get_logs_with_filter(self, ephemeral_manager, mock_sandbox):
        """Test retrieving logs with level filter."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout="1|2024-01-15T10:30:00|ERROR|Error message\n", stderr=""
        )

        logs = await ephemeral_manager.get_logs(level="ERROR")

        assert len(logs) == 1
        assert logs[0].level == "ERROR"

    @pytest.mark.asyncio
    async def test_get_logs_empty(self, ephemeral_manager, mock_sandbox):
        """Test retrieving logs when empty."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        logs = await ephemeral_manager.get_logs()

        assert logs == []

    @pytest.mark.asyncio
    async def test_clear_logs(self, ephemeral_manager, mock_sandbox):
        """Test clearing logs."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.clear_logs()

        assert result is True


# =============================================================================
# FILE TRACKING TESTS
# =============================================================================


class TestFileTracking:
    """Tests for file tracking."""

    @pytest.mark.asyncio
    async def test_record_file(self, ephemeral_manager, mock_sandbox):
        """Test recording a file."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.record_file(
            "/workspace/output.py", size_bytes=1024, description="Generated code"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_list_recorded_files(self, ephemeral_manager, mock_sandbox):
        """Test listing recorded files."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0,
            stdout="/workspace/output.py|2024-01-15T10:30:00|1024|Generated code\n/workspace/data.json|2024-01-15T10:31:00|512|Data file\n",
            stderr="",
        )

        files = await ephemeral_manager.list_recorded_files()

        assert len(files) == 2
        assert files[0].path == "/workspace/output.py"
        assert files[0].size_bytes == 1024
        assert files[0].description == "Generated code"

    @pytest.mark.asyncio
    async def test_list_recorded_files_empty(self, ephemeral_manager, mock_sandbox):
        """Test listing recorded files when empty."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        files = await ephemeral_manager.list_recorded_files()

        assert files == []


# =============================================================================
# UTILITY TESTS
# =============================================================================


class TestUtilities:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_clear_state(self, ephemeral_manager, mock_sandbox):
        """Test clearing all state."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(exit_code=0, stdout="", stderr="")

        result = await ephemeral_manager.clear_state()

        assert result is True

    @pytest.mark.asyncio
    async def test_get_database_size(self, ephemeral_manager, mock_sandbox):
        """Test getting database size."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=0, stdout="4096\n", stderr=""
        )

        size = await ephemeral_manager.get_database_size()

        assert size == 4096

    @pytest.mark.asyncio
    async def test_get_database_size_error(self, ephemeral_manager, mock_sandbox):
        """Test getting database size on error."""
        mock_sandbox.execute_shell.return_value = ExecutionResult(
            exit_code=1, stdout="", stderr="No such file"
        )

        size = await ephemeral_manager.get_database_size()

        assert size == 0

    @pytest.mark.asyncio
    async def test_export_state(self, ephemeral_manager, mock_sandbox):
        """Test exporting all state."""
        # Mock state query
        mock_sandbox.execute_shell.side_effect = [
            ExecutionResult(exit_code=0, stdout="key1|value1|string\n", stderr=""),  # get_all_state
            ExecutionResult(
                exit_code=0, stdout="1|2024-01-15T10:30:00|INFO|Log entry\n", stderr=""
            ),  # get_logs
            ExecutionResult(
                exit_code=0, stdout="/file.py|2024-01-15T10:30:00|100|Desc\n", stderr=""
            ),  # list_recorded_files
        ]

        export = await ephemeral_manager.export_state()

        assert "state" in export
        assert "logs" in export
        assert "files" in export


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestDataClasses:
    """Tests for data classes."""

    def test_agent_log_entry_to_dict(self):
        """Test AgentLogEntry serialization."""
        entry = AgentLogEntry(
            id=1, timestamp=datetime(2024, 1, 15, 10, 30, 0), level="INFO", message="Test message"
        )

        d = entry.to_dict()

        assert d["id"] == 1
        assert d["level"] == "INFO"
        assert d["message"] == "Test message"
        assert "2024-01-15" in d["timestamp"]

    def test_file_record_to_dict(self):
        """Test FileRecord serialization."""
        record = FileRecord(
            path="/workspace/file.py",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            size_bytes=1024,
            description="Test file",
        )

        d = record.to_dict()

        assert d["path"] == "/workspace/file.py"
        assert d["size_bytes"] == 1024
        assert d["description"] == "Test file"
        assert "2024-01-15" in d["created_at"]
