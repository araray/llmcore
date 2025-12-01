# tests/sandbox/test_tools.py
"""
Unit tests for sandbox tools.

These tests verify the tool implementations work correctly
when the sandbox is properly configured.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
# Assumes llmcore is installed or in PYTHONPATH

from llmcore.agents.sandbox.tools import (
    set_active_sandbox,
    clear_active_sandbox,
    get_active_sandbox,
    execute_shell,
    execute_python,
    save_file,
    load_file,
    replace_in_file,
    append_to_file,
    list_files,
    file_exists,
    delete_file,
    create_directory,
    get_state,
    set_state,
    list_state,
    get_sandbox_info,
    get_recorded_files,
    _check_tool_access,
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS
)
from llmcore.agents.sandbox.base import (
    SandboxConfig,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo
)
from llmcore.agents.sandbox.registry import SandboxRegistry, SandboxRegistryConfig, SandboxMode


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox provider."""
    sandbox = MagicMock()
    sandbox.get_access_level.return_value = SandboxAccessLevel.RESTRICTED
    sandbox.get_status.return_value = SandboxStatus.READY
    sandbox.get_config.return_value = SandboxConfig()
    sandbox.get_info.return_value = {
        "provider": "docker",
        "status": "ready",
        "container_id": "abc123"
    }
    sandbox.is_healthy = AsyncMock(return_value=True)
    return sandbox


@pytest.fixture
def mock_registry():
    """Create a mock sandbox registry."""
    registry = MagicMock()
    registry.is_tool_allowed.return_value = True
    return registry


@pytest.fixture
def mock_ephemeral():
    """Create a mock ephemeral resource manager."""
    ephemeral = MagicMock()
    ephemeral.get_state = AsyncMock(return_value=None)
    ephemeral.set_state = AsyncMock(return_value=True)
    ephemeral.list_state_keys = AsyncMock(return_value=[])
    ephemeral.record_file = AsyncMock(return_value=True)
    ephemeral.list_recorded_files = AsyncMock(return_value=[])
    return ephemeral


@pytest.fixture
def setup_sandbox(mock_sandbox, mock_registry, mock_ephemeral):
    """Setup the active sandbox for testing."""
    with patch('sandbox.tools.EphemeralResourceManager') as MockEphemeral:
        MockEphemeral.return_value = mock_ephemeral
        set_active_sandbox(mock_sandbox, mock_registry)
        yield mock_sandbox, mock_registry, mock_ephemeral
        clear_active_sandbox()


# =============================================================================
# SANDBOX MANAGEMENT TESTS
# =============================================================================

class TestSandboxManagement:
    """Tests for sandbox management functions."""

    def test_set_active_sandbox(self, mock_sandbox, mock_registry):
        """Test setting active sandbox."""
        with patch('sandbox.tools.EphemeralResourceManager'):
            set_active_sandbox(mock_sandbox, mock_registry)

            assert get_active_sandbox() is mock_sandbox

            clear_active_sandbox()

    def test_clear_active_sandbox(self, mock_sandbox, mock_registry):
        """Test clearing active sandbox."""
        with patch('sandbox.tools.EphemeralResourceManager'):
            set_active_sandbox(mock_sandbox, mock_registry)
            clear_active_sandbox()

            assert get_active_sandbox() is None

    def test_get_active_sandbox_none(self):
        """Test getting active sandbox when none set."""
        clear_active_sandbox()
        assert get_active_sandbox() is None


# =============================================================================
# TOOL ACCESS CONTROL TESTS
# =============================================================================

class TestToolAccessControl:
    """Tests for tool access control."""

    def test_tool_allowed(self, setup_sandbox):
        """Test tool is allowed."""
        mock_sandbox, mock_registry, _ = setup_sandbox
        mock_registry.is_tool_allowed.return_value = True

        error = _check_tool_access("execute_shell")

        assert error is None

    def test_tool_denied(self, setup_sandbox):
        """Test tool is denied."""
        mock_sandbox, mock_registry, _ = setup_sandbox
        mock_registry.is_tool_allowed.return_value = False

        error = _check_tool_access("dangerous_tool")

        assert error is not None
        assert "not allowed" in error

    def test_no_sandbox_error(self):
        """Test error when no sandbox active."""
        clear_active_sandbox()

        error = _check_tool_access("execute_shell")

        assert error is not None
        assert "No active sandbox" in error


# =============================================================================
# EXECUTION TOOL TESTS
# =============================================================================

class TestExecutionTools:
    """Tests for execution tools."""

    @pytest.mark.asyncio
    async def test_execute_shell_success(self, setup_sandbox):
        """Test successful shell execution."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_shell = AsyncMock(return_value=ExecutionResult(
            exit_code=0,
            stdout="Hello, World!\n",
            stderr=""
        ))

        result = await execute_shell("echo 'Hello, World!'")

        assert "Hello, World!" in result
        mock_sandbox.execute_shell.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_shell_with_timeout(self, setup_sandbox):
        """Test shell execution with timeout."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_shell = AsyncMock(return_value=ExecutionResult(
            exit_code=0,
            stdout="done",
            stderr=""
        ))

        await execute_shell("sleep 1", timeout=30)

        mock_sandbox.execute_shell.assert_called_with("sleep 1", 30, None)

    @pytest.mark.asyncio
    async def test_execute_shell_error(self, setup_sandbox):
        """Test shell execution with error."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_shell = AsyncMock(return_value=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="command not found"
        ))

        result = await execute_shell("nonexistent")

        assert "EXIT CODE: 1" in result
        assert "command not found" in result

    @pytest.mark.asyncio
    async def test_execute_shell_no_sandbox(self):
        """Test shell execution without active sandbox."""
        clear_active_sandbox()

        result = await execute_shell("echo test")

        assert "ERROR" in result
        assert "No active sandbox" in result

    @pytest.mark.asyncio
    async def test_execute_python_success(self, setup_sandbox):
        """Test successful Python execution."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_python = AsyncMock(return_value=ExecutionResult(
            exit_code=0,
            stdout="42\n",
            stderr=""
        ))

        result = await execute_python("print(6 * 7)")

        assert "42" in result

    @pytest.mark.asyncio
    async def test_execute_python_syntax_error(self, setup_sandbox):
        """Test Python execution with syntax error."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_python = AsyncMock(return_value=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="SyntaxError: invalid syntax"
        ))

        result = await execute_python("def broken(")

        assert "EXIT CODE: 1" in result
        assert "SyntaxError" in result


# =============================================================================
# FILE OPERATION TOOL TESTS
# =============================================================================

class TestFileOperationTools:
    """Tests for file operation tools."""

    @pytest.mark.asyncio
    async def test_save_file_success(self, setup_sandbox):
        """Test successful file save."""
        mock_sandbox, _, mock_ephemeral = setup_sandbox
        mock_sandbox.write_file = AsyncMock(return_value=True)

        result = await save_file("test.txt", "Hello, World!")

        assert "Successfully saved" in result
        mock_sandbox.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_file_failure(self, setup_sandbox):
        """Test file save failure."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.write_file = AsyncMock(return_value=False)

        result = await save_file("test.txt", "content")

        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_load_file_success(self, setup_sandbox):
        """Test successful file load."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value="file content")

        result = await load_file("test.txt")

        assert result == "file content"

    @pytest.mark.asyncio
    async def test_load_file_not_found(self, setup_sandbox):
        """Test loading nonexistent file."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value=None)

        result = await load_file("nonexistent.txt")

        assert "ERROR" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_replace_in_file_success(self, setup_sandbox):
        """Test successful text replacement."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value="old value here")
        mock_sandbox.write_file = AsyncMock(return_value=True)

        result = await replace_in_file("test.txt", "old value", "new value")

        assert "Successfully replaced" in result
        mock_sandbox.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_replace_in_file_not_found(self, setup_sandbox):
        """Test replacement in nonexistent file."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value=None)

        result = await replace_in_file("nonexistent.txt", "old", "new")

        assert "ERROR" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_replace_in_file_value_not_found(self, setup_sandbox):
        """Test replacement when value not in file."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value="some other content")

        result = await replace_in_file("test.txt", "nonexistent", "new")

        assert "ERROR" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_append_to_file(self, setup_sandbox):
        """Test appending to file."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.write_file = AsyncMock(return_value=True)

        result = await append_to_file("test.txt", "\nnew line")

        assert "Successfully appended" in result
        mock_sandbox.write_file.assert_called_with("test.txt", "\nnew line", mode="a")

    @pytest.mark.asyncio
    async def test_list_files(self, setup_sandbox):
        """Test listing files."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.list_files = AsyncMock(return_value=[
            FileInfo(path="test.txt", name="test.txt", is_directory=False, size_bytes=100),
            FileInfo(path="subdir", name="subdir", is_directory=True)
        ])

        result = await list_files()

        assert "test.txt" in result
        assert "subdir/" in result

    @pytest.mark.asyncio
    async def test_list_files_empty(self, setup_sandbox):
        """Test listing empty directory."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.list_files = AsyncMock(return_value=[])

        result = await list_files()

        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_file_exists_true(self, setup_sandbox):
        """Test file exists check - true."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.file_exists = AsyncMock(return_value=True)

        result = await file_exists("test.txt")

        assert result == "true"

    @pytest.mark.asyncio
    async def test_file_exists_false(self, setup_sandbox):
        """Test file exists check - false."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.file_exists = AsyncMock(return_value=False)

        result = await file_exists("nonexistent.txt")

        assert result == "false"

    @pytest.mark.asyncio
    async def test_delete_file(self, setup_sandbox):
        """Test file deletion."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.delete_file = AsyncMock(return_value=True)

        result = await delete_file("test.txt")

        assert "Successfully deleted" in result

    @pytest.mark.asyncio
    async def test_create_directory(self, setup_sandbox):
        """Test directory creation."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.create_directory = AsyncMock(return_value=True)

        result = await create_directory("subdir/nested")

        assert "Successfully created" in result


# =============================================================================
# STATE MANAGEMENT TOOL TESTS
# =============================================================================

class TestStateManagementTools:
    """Tests for state management tools."""

    @pytest.mark.asyncio
    async def test_get_state_exists(self, setup_sandbox):
        """Test getting existing state."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.get_state.return_value = "stored_value"

        with patch('sandbox.tools._ephemeral_manager', mock_ephemeral):
            result = await get_state("my_key")

        assert result == "stored_value"

    @pytest.mark.asyncio
    async def test_get_state_not_exists(self, setup_sandbox):
        """Test getting nonexistent state."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.get_state.return_value = None

        with patch('sandbox.tools._ephemeral_manager', mock_ephemeral):
            result = await get_state("nonexistent")

        assert result == "(not set)"

    @pytest.mark.asyncio
    async def test_set_state(self, setup_sandbox):
        """Test setting state."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.set_state.return_value = True

        with patch('sandbox.tools._ephemeral_manager', mock_ephemeral):
            result = await set_state("my_key", "my_value")

        assert "updated" in result.lower()

    @pytest.mark.asyncio
    async def test_list_state(self, setup_sandbox):
        """Test listing state keys."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.list_state_keys.return_value = ["key1", "key2", "key3"]

        with patch('sandbox.tools._ephemeral_manager', mock_ephemeral):
            result = await list_state()

        assert "key1" in result
        assert "key2" in result
        assert "key3" in result


# =============================================================================
# INFORMATION TOOL TESTS
# =============================================================================

class TestInformationTools:
    """Tests for information tools."""

    @pytest.mark.asyncio
    async def test_get_sandbox_info(self, setup_sandbox):
        """Test getting sandbox info."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.get_info.return_value = {
            "provider": "docker",
            "status": "ready",
            "container_id": "abc123"
        }

        result = await get_sandbox_info()

        assert "docker" in result.lower()
        assert "restricted" in result.lower()

    @pytest.mark.asyncio
    async def test_get_sandbox_info_full_access(self, setup_sandbox):
        """Test getting sandbox info for full access."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.get_access_level.return_value = SandboxAccessLevel.FULL
        mock_sandbox.get_info.return_value = {"provider": "docker"}

        result = await get_sandbox_info()

        assert "full" in result.lower()
        assert "network: enabled" in result.lower()

    @pytest.mark.asyncio
    async def test_get_recorded_files(self, setup_sandbox):
        """Test getting recorded files."""
        _, _, mock_ephemeral = setup_sandbox

        from llmcore.agents.sandbox.ephemeral import FileRecord
        from datetime import datetime

        mock_ephemeral.list_recorded_files.return_value = [
            FileRecord(
                path="output.py",
                created_at=datetime.now(),
                size_bytes=1024,
                description="Generated code"
            )
        ]

        with patch('sandbox.tools._ephemeral_manager', mock_ephemeral):
            result = await get_recorded_files()

        assert "output.py" in result
        assert "1024 bytes" in result


# =============================================================================
# TOOL REGISTRY TESTS
# =============================================================================

class TestToolRegistry:
    """Tests for tool registries."""

    def test_implementations_registry(self):
        """Test all implementations are registered."""
        assert "llmcore.tools.sandbox.execute_shell" in SANDBOX_TOOL_IMPLEMENTATIONS
        assert "llmcore.tools.sandbox.execute_python" in SANDBOX_TOOL_IMPLEMENTATIONS
        assert "llmcore.tools.sandbox.save_file" in SANDBOX_TOOL_IMPLEMENTATIONS
        assert "llmcore.tools.sandbox.load_file" in SANDBOX_TOOL_IMPLEMENTATIONS

    def test_schemas_registry(self):
        """Test tool schemas are defined."""
        assert "execute_shell" in SANDBOX_TOOL_SCHEMAS
        assert "execute_python" in SANDBOX_TOOL_SCHEMAS
        assert "save_file" in SANDBOX_TOOL_SCHEMAS

        # Verify schema structure
        shell_schema = SANDBOX_TOOL_SCHEMAS["execute_shell"]
        assert shell_schema["type"] == "function"
        assert "parameters" in shell_schema["function"]

    def test_implementations_are_callable(self):
        """Test all implementations are callable."""
        for name, impl in SANDBOX_TOOL_IMPLEMENTATIONS.items():
            assert callable(impl), f"{name} is not callable"
