# tests/sandbox/test_integration.py
"""
Integration tests for the sandbox system.

These tests verify end-to-end workflows using the mock provider.
Real Docker/VM tests are in separate files and require actual infrastructure.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.agents.sandbox import (
    EphemeralResourceManager,
    ExecutionResult,
    OutputTracker,
    SandboxAccessLevel,
    SandboxMode,
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxStatus,
    clear_active_sandbox,
    create_registry_config,
    execute_python,
    execute_shell,
    load_sandbox_config,
    set_active_sandbox,
)

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.registry import SandboxMode, SandboxRegistry, SandboxRegistryConfig

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as share:
        with tempfile.TemporaryDirectory() as outputs:
            yield Path(share), Path(outputs)


@pytest.fixture
def mock_docker_client():
    """Create a fully configured mock Docker client."""
    mock_client = MagicMock()

    # Version check
    mock_client.version.return_value = {"Version": "24.0.0"}
    mock_client.ping.return_value = True

    # Image handling
    mock_image = MagicMock()
    mock_image.labels = {}
    mock_client.images.get.return_value = mock_image
    mock_client.images.pull.return_value = mock_image

    # Container
    mock_container = MagicMock()
    mock_container.short_id = "abc12345"
    mock_container.name = "test-sandbox"
    mock_container.status = "running"

    # Default exec_run behavior
    def exec_run_side_effect(cmd, **kwargs):
        if "echo " in cmd:
            msg = cmd.split("echo ", 1)[1].strip("'\"")
            return (0, (msg.encode() + b"\n", b""))
        elif "python3" in cmd:
            return (0, (b"42\n", b""))
        elif "cat " in cmd:
            return (0, (b"file content", b""))
        elif "test -e" in cmd:
            return (0, (b"yes\n", b""))
        elif "mkdir" in cmd:
            return (0, (b"", b""))
        elif "sqlite3" in cmd:
            return (0, (b"", b""))
        elif "ls" in cmd:
            return (0, (b"file1.txt\nfile2.py\n", b""))
        else:
            return (0, (b"", b""))

    mock_container.exec_run.side_effect = exec_run_side_effect
    mock_container.reload = MagicMock()
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()

    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container

    return mock_client


# =============================================================================
# CONFIGURATION INTEGRATION TESTS
# =============================================================================


class TestConfigurationIntegration:
    """Tests for configuration loading and registry creation."""

    def test_load_and_create_registry_config(self, temp_dirs):
        """Test loading config and creating registry config."""
        share_path, outputs_path = temp_dirs

        # Load with overrides
        config = load_sandbox_config(
            overrides={
                "mode": "docker",
                "docker": {"image": "python:3.11-slim"},
                "volumes": {"share_path": str(share_path), "outputs_path": str(outputs_path)},
            }
        )

        # Convert to registry config
        registry_config = create_registry_config(config)

        assert registry_config.mode == SandboxMode.DOCKER
        assert registry_config.docker_image == "python:3.11-slim"
        assert registry_config.share_path == str(share_path)

    def test_full_config_workflow(self, temp_dirs):
        """Test complete configuration workflow."""
        share_path, outputs_path = temp_dirs

        # Create TOML config
        toml_content = f"""
[agents.sandbox]
mode = "docker"

[agents.sandbox.docker]
image = "python:3.11-slim"
memory_limit = "1g"

[agents.sandbox.volumes]
share_path = "{share_path}"
outputs_path = "{outputs_path}"
"""
        config_path = share_path / "config.toml"
        config_path.write_text(toml_content)

        # Load and create registry
        config = load_sandbox_config(config_path=config_path)
        registry_config = create_registry_config(config)

        assert registry_config.docker_memory_limit == "1g"


# =============================================================================
# REGISTRY INTEGRATION TESTS
# =============================================================================


class TestRegistryIntegration:
    """Tests for sandbox registry workflows."""

    def test_registry_creation(self, temp_dirs):
        """Test registry creation with valid config."""
        share_path, outputs_path = temp_dirs

        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            docker_image="python:3.11-slim",
            docker_image_whitelist=["python:3.*-slim"],
            share_path=str(share_path),
            outputs_path=str(outputs_path),
        )

        registry = SandboxRegistry(config)

        assert registry is not None
        assert registry.get_active_count() == 0

    def test_tool_access_control(self, temp_dirs):
        """Test tool access control in registry."""
        share_path, outputs_path = temp_dirs

        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            allowed_tools=["execute_shell", "save_file"],
            denied_tools=["dangerous_tool"],
            share_path=str(share_path),
            outputs_path=str(outputs_path),
        )

        registry = SandboxRegistry(config)

        # Check restricted access
        assert registry.is_tool_allowed("execute_shell", SandboxAccessLevel.RESTRICTED)
        assert not registry.is_tool_allowed("dangerous_tool", SandboxAccessLevel.RESTRICTED)

        # Full access bypasses restrictions
        assert registry.is_tool_allowed("dangerous_tool", SandboxAccessLevel.FULL)

    def test_get_allowed_tools(self, temp_dirs):
        """Test getting list of allowed tools."""
        share_path, outputs_path = temp_dirs

        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            allowed_tools=["tool1", "tool2", "tool3"],
            denied_tools=["bad_tool"],
            share_path=str(share_path),
            outputs_path=str(outputs_path),
        )

        registry = SandboxRegistry(config)

        allowed = registry.get_allowed_tools(SandboxAccessLevel.RESTRICTED)
        assert "tool1" in allowed
        assert "bad_tool" not in allowed


# =============================================================================
# MOCK SANDBOX INTEGRATION TESTS
# =============================================================================


class TestMockSandboxIntegration:
    """Tests for sandbox operations using mock provider."""

    @pytest.mark.asyncio
    async def test_complete_sandbox_lifecycle(self, mock_sandbox_provider, sandbox_config):
        """Test complete sandbox lifecycle: create, use, cleanup."""
        from tests.agents.sandbox.conftest import MockSandboxProvider

        provider = MockSandboxProvider()

        # Initialize
        await provider.initialize(sandbox_config)
        assert provider.get_status() == SandboxStatus.READY

        # Execute commands
        result = await provider.execute_shell("echo 'Hello'")
        assert result.success

        result = await provider.execute_python("print('World')")
        assert result.success

        # File operations
        await provider.write_file("test.txt", "content")
        content = await provider.read_file("test.txt")
        assert "content" in content

        # Cleanup
        await provider.cleanup()
        assert provider.get_status() == SandboxStatus.TERMINATED

    @pytest.mark.asyncio
    async def test_file_operations_workflow(self, initialized_mock_provider):
        """Test file operation workflow."""
        provider = initialized_mock_provider

        # Create file
        await provider.write_file("output.py", "print('hello')")

        # Read file
        content = await provider.read_file("output.py")
        assert content == "print('hello')"

        # Check existence
        exists = await provider.file_exists("output.py")
        assert exists

        # List files
        files = await provider.list_files("/workspace")
        assert len(files) > 0

        # Delete file
        await provider.delete_file("output.py")
        exists = await provider.file_exists("output.py")
        assert not exists


# =============================================================================
# TOOLS INTEGRATION TESTS
# =============================================================================


class TestToolsIntegrationFixed:
    """
    Fixed version of TestToolsIntegration with proper mock patching.
    """

    @pytest.fixture
    def setup_tools_environment(self, temp_dirs, initialized_mock_provider):
        """
        Setup tools environment with mock sandbox.

        FIXED: Uses create=True for EphemeralResourceManager patching
        to handle lazy imports.
        """
        from llmcore.agents.sandbox.registry import (
            SandboxMode,
            SandboxRegistry,
            SandboxRegistryConfig,
        )
        from llmcore.agents.sandbox.tools import clear_active_sandbox, set_active_sandbox

        share_path, outputs_path = temp_dirs

        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            share_path=str(share_path),
            outputs_path=str(outputs_path),
        )
        registry = SandboxRegistry(config)

        # FIXED: Use create=True to handle lazy imports
        # This creates the attribute at the module level even though
        # it's normally imported inside methods
        with patch(
            "llmcore.agents.sandbox.tools.EphemeralResourceManager",
            create=True,  # <-- KEY FIX
        ) as MockEphemeral:
            mock_ephemeral_instance = MagicMock()
            MockEphemeral.return_value = mock_ephemeral_instance

            # Setup the mock provider as active sandbox
            set_active_sandbox(
                sandbox=initialized_mock_provider,
                registry=registry,
            )

            yield {
                "registry": registry,
                "sandbox": initialized_mock_provider,
                "ephemeral": mock_ephemeral_instance,
                "share_path": share_path,
                "outputs_path": outputs_path,
            }

            # Cleanup
            clear_active_sandbox()

    @pytest.mark.asyncio
    async def test_execute_shell_tool(self, setup_tools_environment):
        """Test shell command execution through tool interface."""
        from llmcore.agents.sandbox.base import ExecutionResult
        from llmcore.agents.sandbox.tools import execute_shell

        env = setup_tools_environment
        sandbox = env["sandbox"]

        # Mock the sandbox's execute_shell method (not execute_command!)
        sandbox.execute_shell = AsyncMock(
            return_value=ExecutionResult(exit_code=0, stdout="Hello World\n", stderr="")
        )

        result = await execute_shell(command="echo 'Hello World'")

        # Verify
        assert result is not None
        assert "Hello World" in result
        sandbox.execute_shell.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_and_load_file_tools(self, setup_tools_environment):
        """Test file save and load operations through tool interface."""
        from llmcore.agents.sandbox.tools import load_file, save_file

        env = setup_tools_environment
        sandbox = env["sandbox"]

        # Mock file operations
        sandbox.write_file = AsyncMock(return_value=True)
        sandbox.read_file = AsyncMock(return_value="test content")

        # Test save
        save_result = await save_file(path="/tmp/test.txt", content="test content")
        assert save_result is not None

        # Test load
        load_result = await load_file(path="/tmp/test.txt")
        assert load_result is not None

    @pytest.mark.asyncio
    async def test_get_sandbox_info_tool(self, setup_tools_environment):
        """Test sandbox info retrieval through tool interface."""
        from llmcore.agents.sandbox.tools import get_sandbox_info

        env = setup_tools_environment
        sandbox = env["sandbox"]

        # Mock info retrieval
        sandbox.get_info = MagicMock(
            return_value={"sandbox_id": "test-sandbox", "provider": "mock", "status": "healthy"}
        )

        result = await get_sandbox_info()

        assert result is not None


# =============================================================================
# OUTPUT TRACKER INTEGRATION TESTS
# =============================================================================


class TestOutputTrackerIntegration:
    """Tests for output tracker integration."""

    @pytest.mark.asyncio
    async def test_complete_tracking_workflow(self, temp_dirs, initialized_mock_provider):
        """Test complete output tracking workflow."""
        share_path, outputs_path = temp_dirs

        tracker = OutputTracker(base_path=str(outputs_path), max_log_entries=100)

        # Create run
        sandbox_config = initialized_mock_provider.get_config()
        run_id = await tracker.create_run(
            sandbox_id=sandbox_config.sandbox_id,
            sandbox_type="mock",
            access_level="restricted",
            task_description="Integration test",
            metadata={"test": True},
        )

        # Log executions
        result = ExecutionResult(exit_code=0, stdout="output", execution_time_seconds=0.1)
        await tracker.log_execution(run_id, "execute_shell", "echo test", result)

        # Track files
        await tracker.track_file(run_id, "/workspace/output.py", 100, "Test file")

        # Finalize
        await tracker.finalize_run(run_id, success=True)

        # Verify
        metadata = await tracker.get_run_metadata(run_id)
        assert metadata["status"] == "completed"
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_run_listing_and_filtering(self, temp_dirs):
        """Test run listing and filtering."""
        share_path, outputs_path = temp_dirs

        tracker = OutputTracker(base_path=str(outputs_path))

        # Create multiple runs
        run1 = await tracker.create_run(sandbox_id="run-1", task_description="Task 1")
        run2 = await tracker.create_run(sandbox_id="run-2", task_description="Task 2")

        await tracker.finalize_run(run1, success=True)
        await tracker.finalize_run(run2, success=False, error_message="Test error")

        # List all
        all_runs = await tracker.list_runs()
        assert len(all_runs) == 2

        # Filter by status
        completed = await tracker.list_runs(status_filter="completed")
        failed = await tracker.list_runs(status_filter="failed")

        assert len(completed) == 1
        assert len(failed) == 1


# =============================================================================
# EPHEMERAL RESOURCE INTEGRATION TESTS
# =============================================================================


class TestEphemeralIntegration:
    """Tests for ephemeral resource integration."""

    @pytest.mark.asyncio
    async def test_state_persistence(self, initialized_mock_provider):
        """Test state persistence across operations."""
        ephemeral = EphemeralResourceManager(initialized_mock_provider)

        # Initialize database
        await ephemeral.init_database()

        # Set and get state
        await ephemeral.set_state("counter", 1)
        await ephemeral.set_state("config", {"key": "value"})

        # List keys
        keys = await ephemeral.list_state_keys()
        # Keys may or may not include our items depending on mock

        # Log events
        await ephemeral.log_event("INFO", "Test message")
        await ephemeral.log_event("WARNING", "Warning message")

    @pytest.mark.asyncio
    async def test_file_tracking(self, initialized_mock_provider):
        """Test file tracking in ephemeral storage."""
        ephemeral = EphemeralResourceManager(initialized_mock_provider)

        await ephemeral.init_database()

        # Record files
        await ephemeral.record_file("/workspace/output.py", 100, "Generated code")
        await ephemeral.record_file("/workspace/data.json", 50, "Data file")


# =============================================================================
# ERROR HANDLING INTEGRATION TESTS
# =============================================================================


class TestErrorHandlingIntegration:
    """Tests for error handling across components."""

    @pytest.mark.asyncio
    async def test_tool_access_denied(self, temp_dirs, initialized_mock_provider):
        """Test tool access denied handling."""
        share_path, outputs_path = temp_dirs

        # Configure registry with restricted tools
        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            allowed_tools=["execute_shell"],  # Only shell allowed
            denied_tools=["execute_python"],  # Python denied
            share_path=str(share_path),
            outputs_path=str(outputs_path),
        )
        registry = SandboxRegistry(config)

        with patch("llmcore.agents.sandbox.ephemeral.EphemeralResourceManager"):
            set_active_sandbox(initialized_mock_provider, registry)

            try:
                # Shell should work
                result = await execute_shell("echo test")
                assert "ERROR" not in result or "test" in result

                # Python should be denied
                result = await execute_python("print(1)")
                # Result should indicate denial or error
                assert "ERROR" in result or "not allowed" in result.lower()
            finally:
                clear_active_sandbox()

    @pytest.mark.asyncio
    async def test_sandbox_not_initialized(self):
        """Test operations without initialized sandbox."""
        clear_active_sandbox()

        result = await execute_shell("echo test")
        assert "ERROR" in result
        assert "No active sandbox" in result


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================


class TestEndToEndWorkflows:
    """Tests for complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_agent_simulation_workflow(self, temp_dirs, initialized_mock_provider):
        """Simulate a complete agent workflow."""
        share_path, outputs_path = temp_dirs

        # Setup
        config = SandboxRegistryConfig(
            mode=SandboxMode.DOCKER,
            docker_enabled=True,
            share_path=str(share_path),
            outputs_path=str(outputs_path),
        )
        registry = SandboxRegistry(config)
        tracker = OutputTracker(base_path=str(outputs_path))

        # Create sandbox and tracking
        sandbox = initialized_mock_provider
        run_id = await tracker.create_run(
            sandbox_id=sandbox.get_config().sandbox_id,
            sandbox_type="mock",
            task_description="Agent simulation",
        )

        with patch("llmcore.agents.sandbox.tools.EphemeralResourceManager", create=True):
            set_active_sandbox(sandbox, registry)

            try:
                # Simulate agent iterations
                # Iteration 1: Explore
                result = await sandbox.execute_shell("ls -la")
                await tracker.log_execution(run_id, "execute_shell", "ls -la", result)

                # Iteration 2: Create file
                await sandbox.write_file("solution.py", "print('solution')")
                await tracker.track_file(run_id, "solution.py", 20, "Solution code")

                # Iteration 3: Test
                result = await sandbox.execute_python("print('test passed')")
                await tracker.log_execution(run_id, "execute_python", "print test", result)

                # Finalize
                await tracker.finalize_run(run_id, success=True)

            finally:
                clear_active_sandbox()

        # Verify results
        metadata = await tracker.get_run_metadata(run_id)
        assert metadata["success"] is True

        files = await tracker.get_tracked_files(run_id)
        assert len(files) >= 1
