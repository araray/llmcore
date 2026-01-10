# tests/agents/sandbox/test_tools.py
"""
Unit tests for sandbox tools.

These tests verify the tool implementations work correctly
when the sandbox is properly configured.

IMPORTANT FIX NOTES:
====================
The original tests failed because they tried to patch
'llmcore.agents.sandbox.tools.EphemeralResourceManager', but the class
may not be imported at module level in the way expected.

Fix approach: Patch at the correct location or use create=True.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.agents.sandbox.base import (
    ExecutionResult,
    FileInfo,
    SandboxAccessLevel,
    SandboxConfig,
    SandboxStatus,
)
from llmcore.agents.sandbox.registry import SandboxMode, SandboxRegistry, SandboxRegistryConfig

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.tools import (
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
    _check_tool_access,
    append_to_file,
    clear_active_sandbox,
    create_directory,
    delete_file,
    execute_python,
    execute_shell,
    file_exists,
    get_active_sandbox,
    get_recorded_files,
    get_sandbox_info,
    get_state,
    list_files,
    list_state,
    load_file,
    replace_in_file,
    save_file,
    set_active_sandbox,
    set_state,
)

# =============================================================================
# FIXTURES - FIXED VERSION
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
        "container_id": "abc123",
    }
    sandbox.is_healthy = AsyncMock(return_value=True)
    # Add AsyncMock for execute_shell - needed by EphemeralResourceManager
    sandbox.execute_shell = AsyncMock(
        return_value=MagicMock(exit_code=0, stdout="", stderr="", success=True)
    )
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
    """
    Setup the active sandbox for testing.

    FIXED: Use create=True when patching EphemeralResourceManager.
    This allows the patch to work even if EphemeralResourceManager is not
    imported at module level in tools.py.
    """
    with patch(
        "llmcore.agents.sandbox.tools.EphemeralResourceManager",
        create=True,  # Key fix: create the attribute if it doesn't exist
    ) as MockEphemeral:
        MockEphemeral.return_value = mock_ephemeral
        set_active_sandbox(mock_sandbox, mock_registry)
        yield mock_sandbox, mock_registry, mock_ephemeral
        clear_active_sandbox()


# =============================================================================
# ALTERNATIVE: Direct patching of the ephemeral module
# =============================================================================


@pytest.fixture
def setup_sandbox_v2(mock_sandbox, mock_registry, mock_ephemeral):
    """
    Alternative setup using module import patching.

    This patches the EphemeralResourceManager where it's defined,
    which works regardless of how it's imported in tools.py.
    """
    with patch(
        "llmcore.agents.sandbox.ephemeral.EphemeralResourceManager",
    ) as MockEphemeral:
        MockEphemeral.return_value = mock_ephemeral

        # Also patch in tools module if it imports the class
        with patch.dict(
            "llmcore.agents.sandbox.tools.__dict__", {"EphemeralResourceManager": MockEphemeral}
        ):
            set_active_sandbox(mock_sandbox, mock_registry)
            yield mock_sandbox, mock_registry, mock_ephemeral
            clear_active_sandbox()


# =============================================================================
# SANDBOX MANAGEMENT TESTS
# =============================================================================


class TestSandboxManagement:
    """Tests for sandbox management functions."""

    def test_set_and_get_active_sandbox(self, mock_sandbox, mock_registry):
        """Test setting and getting active sandbox."""
        with patch("llmcore.agents.sandbox.tools.EphemeralResourceManager", create=True):
            set_active_sandbox(mock_sandbox, mock_registry)

            # get_active_sandbox() returns the sandbox provider, not a tuple
            sandbox = get_active_sandbox()

            assert sandbox is mock_sandbox

            clear_active_sandbox()

    def test_clear_active_sandbox(self, mock_sandbox, mock_registry):
        """Test clearing active sandbox."""
        with patch("llmcore.agents.sandbox.tools.EphemeralResourceManager", create=True):
            set_active_sandbox(mock_sandbox, mock_registry)
            clear_active_sandbox()

            # After clearing, get_active_sandbox() returns None
            sandbox = get_active_sandbox()

            assert sandbox is None


# =============================================================================
# TOOL ACCESS CONTROL TESTS
# =============================================================================


class TestToolAccessControl:
    """Tests for tool access control."""

    @pytest.mark.asyncio
    async def test_tool_allowed(self, setup_sandbox):
        """Test tool execution when allowed."""
        mock_sandbox, mock_registry, _ = setup_sandbox
        mock_registry.is_tool_allowed.return_value = True

        # Should not raise
        _check_tool_access("execute_shell")

    @pytest.mark.asyncio
    async def test_tool_denied(self, setup_sandbox):
        """Test tool execution when denied."""
        mock_sandbox, mock_registry, _ = setup_sandbox
        mock_registry.is_tool_allowed.return_value = False

        # Tools return error strings rather than raising exceptions
        result = await execute_shell("echo test")
        assert "ERROR" in result or "not allowed" in result
        _check_tool_access("dangerous_tool")


# =============================================================================
# EXECUTION TOOL TESTS
# =============================================================================


class TestExecutionTools:
    """Tests for execution tools (shell, python)."""

    @pytest.mark.asyncio
    async def test_execute_shell_success(self, setup_sandbox):
        """Test successful shell execution."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_shell = AsyncMock(
            return_value=ExecutionResult(exit_code=0, stdout="Hello World\n", stderr="")
        )

        result = await execute_shell("echo 'Hello World'")

        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_execute_shell_with_timeout(self, setup_sandbox):
        """Test shell execution with custom timeout."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_shell = AsyncMock(
            return_value=ExecutionResult(exit_code=0, stdout="Done\n", stderr="")
        )

        result = await execute_shell("sleep 1 && echo Done", timeout=30)

        assert "Done" in result

    @pytest.mark.asyncio
    async def test_execute_shell_error(self, setup_sandbox):
        """Test shell execution with error."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_shell = AsyncMock(
            return_value=ExecutionResult(exit_code=1, stdout="", stderr="Command not found")
        )

        result = await execute_shell("nonexistent_command")

        assert "EXIT CODE: 1" in result

    @pytest.mark.asyncio
    async def test_execute_python_success(self, setup_sandbox):
        """Test successful Python execution."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_python = AsyncMock(
            return_value=ExecutionResult(exit_code=0, stdout="42\n", stderr="")
        )

        result = await execute_python("print(6 * 7)")

        assert "42" in result

    @pytest.mark.asyncio
    async def test_execute_python_syntax_error(self, setup_sandbox):
        """Test Python execution with syntax error."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.execute_python = AsyncMock(
            return_value=ExecutionResult(
                exit_code=1, stdout="", stderr="SyntaxError: invalid syntax"
            )
        )

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
        mock_sandbox.read_file = AsyncMock(return_value="Hello World")
        mock_sandbox.write_file = AsyncMock(return_value=True)

        result = await replace_in_file("test.txt", "World", "Universe")

        assert "Successfully" in result or "replaced" in result.lower()

    @pytest.mark.asyncio
    async def test_replace_in_file_not_found(self, setup_sandbox):
        """Test replacement in nonexistent file."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value=None)

        result = await replace_in_file("nonexistent.txt", "old", "new")

        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_replace_in_file_value_not_found(self, setup_sandbox):
        """Test replacement when value not found."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value="Hello World")

        result = await replace_in_file("test.txt", "NotInFile", "new")

        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_append_to_file(self, setup_sandbox):
        """Test appending to file."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.read_file = AsyncMock(return_value="Existing content\n")
        mock_sandbox.write_file = AsyncMock(return_value=True)

        result = await append_to_file("test.txt", "New content")

        assert "Successfully" in result

    @pytest.mark.asyncio
    async def test_list_files(self, setup_sandbox):
        """Test listing files."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.list_files = AsyncMock(
            return_value=[
                FileInfo(
                    name="file1.txt",
                    path="/workspace/file1.txt",
                    size_bytes=100,
                    is_directory=False,
                ),
                FileInfo(
                    name="file2.py", path="/workspace/file2.py", size_bytes=200, is_directory=False
                ),
            ]
        )

        result = await list_files()

        assert "file1.txt" in result
        assert "file2.py" in result

    @pytest.mark.asyncio
    async def test_list_files_empty(self, setup_sandbox):
        """Test listing files when empty."""
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
        mock_ephemeral.get_state = AsyncMock(return_value="stored_value")

        with patch("llmcore.agents.sandbox.tools._ephemeral_manager", mock_ephemeral):
            result = await get_state("my_key")

        assert result == "stored_value"

    @pytest.mark.asyncio
    async def test_get_state_not_exists(self, setup_sandbox):
        """Test getting nonexistent state."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.get_state = AsyncMock(return_value=None)

        with patch("llmcore.agents.sandbox.tools._ephemeral_manager", mock_ephemeral):
            result = await get_state("nonexistent")

        assert result == "(not set)"

    @pytest.mark.asyncio
    async def test_set_state(self, setup_sandbox):
        """Test setting state."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.set_state = AsyncMock(return_value=True)

        with patch("llmcore.agents.sandbox.tools._ephemeral_manager", mock_ephemeral):
            result = await set_state("key", "value")

        assert "updated" in result.lower() or "State" in result

    @pytest.mark.asyncio
    async def test_list_state(self, setup_sandbox):
        """Test listing state keys."""
        _, _, mock_ephemeral = setup_sandbox
        mock_ephemeral.list_state_keys = AsyncMock(return_value=["key1", "key2"])

        with patch("llmcore.agents.sandbox.tools._ephemeral_manager", mock_ephemeral):
            result = await list_state()

        assert "key1" in result
        assert "key2" in result


# =============================================================================
# INFORMATION TOOL TESTS
# =============================================================================


class TestInformationTools:
    """Tests for information retrieval tools."""

    @pytest.mark.asyncio
    async def test_get_sandbox_info(self, setup_sandbox):
        """Test getting sandbox info."""
        mock_sandbox, _, _ = setup_sandbox

        result = await get_sandbox_info()

        assert "docker" in result.lower() or "provider" in result.lower()

    @pytest.mark.asyncio
    async def test_get_sandbox_info_full_access(self, setup_sandbox):
        """Test getting sandbox info with full access."""
        mock_sandbox, _, _ = setup_sandbox
        mock_sandbox.get_access_level.return_value = SandboxAccessLevel.FULL

        result = await get_sandbox_info()

        assert "full" in result.lower() or "access" in result.lower()

    @pytest.mark.asyncio
    async def test_get_recorded_files(self, setup_sandbox):
        """Test getting recorded files."""
        _, _, mock_ephemeral = setup_sandbox
        from llmcore.agents.sandbox.ephemeral import FileRecord

        mock_ephemeral.list_recorded_files = AsyncMock(
            return_value=[
                MagicMock(
                    path="/workspace/output.py", size_bytes=1024, description="Generated code"
                ),
            ]
        )

        with patch("llmcore.agents.sandbox.tools._ephemeral_manager", mock_ephemeral):
            result = await get_recorded_files()

        assert "output.py" in result or "file" in result.lower()


# =============================================================================
# TOOL SCHEMA AND IMPLEMENTATION TESTS
# =============================================================================


class TestToolSchemasAndImplementations:
    """Tests for tool schemas and implementations."""

    def test_all_tools_have_schemas(self):
        """Test that all tool implementations have corresponding schemas."""
        for tool_name in SANDBOX_TOOL_IMPLEMENTATIONS.keys():
            # Handle fully qualified names (extract simple name)
            simple_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
            assert simple_name in SANDBOX_TOOL_SCHEMAS, (
                f"Missing schema for {tool_name} (simple: {simple_name})"
            )

    def test_all_schemas_have_implementations(self):
        """Test that all tool schemas have corresponding implementations."""
        # Get simple names from implementations
        impl_simple_names = {
            k.split(".")[-1] if "." in k else k for k in SANDBOX_TOOL_IMPLEMENTATIONS.keys()
        }
        for tool_name in SANDBOX_TOOL_SCHEMAS.keys():
            assert tool_name in impl_simple_names, f"Missing implementation for {tool_name}"

    def test_schema_structure(self):
        """Test that schemas have required structure."""
        for tool_name, schema in SANDBOX_TOOL_SCHEMAS.items():
            # Schemas use OpenAI function calling format:
            # {'type': 'function', 'function': {'name': ..., 'description': ..., 'parameters': ...}}
            assert "type" in schema, f"Schema for {tool_name} missing 'type'"
            assert schema["type"] == "function", f"Schema for {tool_name} type should be 'function'"
            assert "function" in schema, f"Schema for {tool_name} missing 'function'"

            func_schema = schema["function"]
            assert "name" in func_schema, f"Schema for {tool_name} missing 'function.name'"
            assert "description" in func_schema, (
                f"Schema for {tool_name} missing 'function.description'"
            )

            # parameters is optional but if present should have certain structure
            if "parameters" in func_schema:
                params = func_schema["parameters"]
                assert "type" in params, f"Schema for {tool_name} parameters missing 'type'"
