# tests/agents/sandbox/test_vm_provider.py
"""
Unit tests for VMSandboxProvider.

These tests use mocking to test the VM provider without requiring
actual SSH connections. Integration tests with real VMs are in
a separate integration test file.

IMPORTANT FIX NOTES:
====================
The original tests failed because they tried to patch
'llmcore.agents.sandbox.vm_provider.paramiko', but the paramiko module
is imported LAZILY inside methods, not at module level.

Fix approach: Use sys.modules patching with proper MagicMock instances.

CRITICAL FIX (2026-01-10):
==========================
- mock_paramiko_module must use `SSHClient = MagicMock()` (instance),
  NOT `SSHClient = MagicMock` (class). The class form creates NEW mocks
  on each call instead of returning mock_client.
- All exec_command side_effects must use factory functions that create
  fresh mocks with properly configured channel attributes.
"""

import sys
from unittest.mock import MagicMock

import pytest

from llmcore.agents.sandbox.base import (
    SandboxAccessLevel,
    SandboxConfig,
    SandboxStatus,
)
from llmcore.agents.sandbox.exceptions import (
    SandboxInitializationError,
    SandboxNotInitializedError,
)

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.vm_provider import VMSandboxProvider

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_exec_command_result(*args, **kwargs):
    """
    Factory function that creates fresh mocks for each exec_command call.

    This prevents state pollution (like .read() returning empty after first call)
    and ensures proper mock configuration for each invocation.

    Returns:
        Tuple of (mock_stdin, mock_stdout, mock_stderr) with proper configuration.
    """
    mock_stdin = MagicMock()
    mock_stdout = MagicMock()
    mock_stderr = MagicMock()

    # Default successful execution
    mock_stdout.read.return_value = b""
    mock_stderr.read.return_value = b""

    # CRITICAL: Must explicitly set channel as MagicMock before accessing attributes
    mock_stdout.channel = MagicMock()
    mock_stdout.channel.recv_exit_status.return_value = 0

    return (mock_stdin, mock_stdout, mock_stderr)


# =============================================================================
# FIXTURES - FIXED VERSION
# =============================================================================


@pytest.fixture
def mock_paramiko_module():
    """
    Create a mock paramiko module.

    This fixture patches the paramiko module using sys.modules, which works
    regardless of where/when the import happens (lazy or eager).

    CRITICAL FIX: Use MagicMock() (instances) for classes, not MagicMock (class).
    Using the class causes SSHClient() to create NEW mocks instead of returning
    the configured mock_client.
    """
    mock_paramiko = MagicMock()

    # FIXED: Use MagicMock() instances for classes, not the MagicMock class itself
    # This ensures that calling e.g. paramiko.SSHClient() returns the configured return_value
    mock_paramiko.SSHClient = MagicMock()  # Instance, not class!
    mock_paramiko.AutoAddPolicy = MagicMock()  # Instance, not class!

    # Setup exceptions (these should be actual exception classes)
    mock_paramiko.AuthenticationException = type("AuthenticationException", (Exception,), {})
    mock_paramiko.SSHException = type("SSHException", (Exception,), {})

    # Mock key classes - these need to be callable to create key instances
    mock_paramiko.Ed25519Key = MagicMock()
    mock_paramiko.RSAKey = MagicMock()
    mock_paramiko.ECDSAKey = MagicMock()
    mock_paramiko.DSSKey = MagicMock()

    # Store original if exists
    original_paramiko = sys.modules.get("paramiko")

    # Patch sys.modules
    sys.modules["paramiko"] = mock_paramiko

    yield mock_paramiko

    # Restore original
    if original_paramiko is not None:
        sys.modules["paramiko"] = original_paramiko
    else:
        sys.modules.pop("paramiko", None)


@pytest.fixture
def mock_paramiko(mock_paramiko_module):
    """
    Create mock paramiko module with pre-configured SSH client.

    This fixture configures the mock SSH client with proper exec_command
    behavior using a factory function to create fresh mocks each call.
    """
    mock_client = MagicMock()
    mock_sftp = MagicMock()
    mock_transport = MagicMock()

    # Configure client methods
    mock_client.connect = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    mock_client.get_transport.return_value = mock_transport
    mock_client.close = MagicMock()
    mock_transport.is_active.return_value = True

    # Use the factory function for exec_command
    mock_client.exec_command.side_effect = create_exec_command_result

    # Make SSHClient() return our mock client
    mock_paramiko_module.SSHClient.return_value = mock_client

    yield mock_paramiko_module, mock_client, mock_sftp


@pytest.fixture
def vm_provider(mock_paramiko):
    """Create a VMSandboxProvider with mocked paramiko."""
    mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

    provider = VMSandboxProvider(
        host="test-vm.example.com",
        username="testuser",
        private_key_path="~/.ssh/test_key",
        full_access_hosts=["trusted-vm.example.com"],
    )

    return provider


@pytest.fixture
def sandbox_config():
    """Create a basic sandbox configuration."""
    return SandboxConfig(
        timeout_seconds=60, memory_limit="512m", cpu_limit=1.0, network_enabled=False
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestVMSandboxProviderInit:
    """Tests for VMSandboxProvider initialization."""

    def test_init_with_private_key(self, mock_paramiko):
        """Test initialization with private key."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )

        assert provider._host == "test-vm.example.com"
        assert provider._username == "testuser"
        assert provider._private_key_path == "~/.ssh/id_rsa"
        assert provider._status == SandboxStatus.CREATED

    def test_init_with_ssh_agent(self, mock_paramiko):
        """Test initialization with SSH agent."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", use_ssh_agent=True
        )

        assert provider._use_ssh_agent is True
        assert provider._private_key_path is None

    def test_init_fails_without_auth_method(self, mock_paramiko):
        """Test initialization fails without auth method.

        NOTE: Must explicitly pass use_ssh_agent=False because the default is True.
        The test validates that VMSandboxProvider raises SandboxInitializationError
        when neither private_key_path nor use_ssh_agent is enabled.
        """
        # FIXED: Must explicitly disable ssh_agent since default is True
        with pytest.raises(SandboxInitializationError):
            VMSandboxProvider(
                host="test-vm.example.com",
                username="testuser",
                use_ssh_agent=False,  # FIXED: Explicitly disable
                # No private_key_path
            )

    def test_init_with_custom_port(self, mock_paramiko):
        """Test initialization with custom port."""
        provider = VMSandboxProvider(
            host="test-vm.example.com",
            port=2222,
            username="testuser",
            private_key_path="~/.ssh/id_rsa",
        )

        assert provider._port == 2222


class TestAccessLevelDetermination:
    """Tests for access level determination."""

    def test_full_access_for_whitelisted_host(self, mock_paramiko):
        """Test full access for whitelisted host."""
        provider = VMSandboxProvider(
            host="trusted-vm.example.com",
            username="testuser",
            private_key_path="~/.ssh/id_rsa",
            full_access_hosts=["trusted-vm.example.com", "other-trusted.com"],
        )

        level = provider._determine_access_level()

        assert level == SandboxAccessLevel.FULL

    def test_restricted_access_for_unknown_host(self, mock_paramiko):
        """Test restricted access for unknown host."""
        provider = VMSandboxProvider(
            host="unknown-vm.example.com",
            username="testuser",
            private_key_path="~/.ssh/id_rsa",
            full_access_hosts=["trusted-vm.example.com"],
        )

        level = provider._determine_access_level()

        assert level == SandboxAccessLevel.RESTRICTED

    def test_restricted_access_with_empty_whitelist(self, mock_paramiko):
        """Test restricted access when whitelist is empty."""
        provider = VMSandboxProvider(
            host="any-vm.example.com",
            username="testuser",
            private_key_path="~/.ssh/id_rsa",
            full_access_hosts=[],
        )

        level = provider._determine_access_level()

        assert level == SandboxAccessLevel.RESTRICTED


class TestSSHConnection:
    """Tests for SSH connection establishment."""

    @pytest.mark.asyncio
    async def test_successful_connection(self, mock_paramiko, sandbox_config, tmp_path):
        """Test successful SSH connection."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        # Create a temporary key file
        key_file = tmp_path / "test_key"
        key_file.write_text("fake key content")

        # Mock key loading
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com",
            username="testuser",
            private_key_path=str(key_file),
            full_access_hosts=[],
        )

        # The fixture already sets up exec_command.side_effect properly
        # No need to override it here

        await provider.initialize(sandbox_config)

        assert provider._status == SandboxStatus.READY

    @pytest.mark.asyncio
    async def test_connection_with_key_file_not_found(self, mock_paramiko, sandbox_config):
        """Test connection failure when key file not found."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="/nonexistent/key"
        )

        with pytest.raises(SandboxInitializationError):
            await provider.initialize(sandbox_config)


class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_successful_command(self, mock_paramiko, sandbox_config, tmp_path):
        """Test successful command execution."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        # Create temp key
        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        # Initialize with default fixture behavior
        await provider.initialize(sandbox_config)

        # Now override for specific test command - use factory pattern
        def create_hello_world_result(*args, **kwargs):
            mock_stdin = MagicMock()
            mock_stdout = MagicMock()
            mock_stderr = MagicMock()
            mock_stdout.read.return_value = b"Hello World\n"
            mock_stderr.read.return_value = b""
            mock_stdout.channel = MagicMock()
            mock_stdout.channel.recv_exit_status.return_value = 0
            return (mock_stdin, mock_stdout, mock_stderr)

        mock_client.exec_command.side_effect = create_hello_world_result

        result = await provider.execute_shell("echo 'Hello World'")

        assert result.exit_code == 0
        assert "Hello World" in result.stdout

    @pytest.mark.asyncio
    async def test_command_with_nonzero_exit(self, mock_paramiko, sandbox_config, tmp_path):
        """Test command with non-zero exit code."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        # Initialize with default fixture behavior
        await provider.initialize(sandbox_config)

        # Now override for failing command - use factory pattern
        def create_error_result(*args, **kwargs):
            mock_stdin = MagicMock()
            mock_stdout = MagicMock()
            mock_stderr = MagicMock()
            mock_stdout.read.return_value = b""
            mock_stderr.read.return_value = b"ls: cannot access 'nonexistent': No such file"
            mock_stdout.channel = MagicMock()
            mock_stdout.channel.recv_exit_status.return_value = 1
            return (mock_stdin, mock_stdout, mock_stderr)

        mock_client.exec_command.side_effect = create_error_result

        result = await provider.execute_shell("ls nonexistent")

        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_command_not_initialized(self, mock_paramiko):
        """Test command execution without initialization raises error."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )

        with pytest.raises(SandboxNotInitializedError):
            await provider.execute_shell("echo test")

    @pytest.mark.asyncio
    async def test_output_truncation(self, mock_paramiko, sandbox_config, tmp_path):
        """Test output truncation for large outputs."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        # Initialize with default fixture behavior
        await provider.initialize(sandbox_config)

        # Large output - use factory pattern
        large_output = b"x" * 200000

        def create_large_output_result(*args, **kwargs):
            mock_stdin = MagicMock()
            mock_stdout = MagicMock()
            mock_stderr = MagicMock()
            mock_stdout.read.return_value = large_output
            mock_stderr.read.return_value = b""
            mock_stdout.channel = MagicMock()
            mock_stdout.channel.recv_exit_status.return_value = 0
            return (mock_stdin, mock_stdout, mock_stderr)

        mock_client.exec_command.side_effect = create_large_output_result

        result = await provider.execute_shell("cat largefile")

        # Should be truncated
        assert len(result.stdout) <= 110000  # MAX_OUTPUT_SIZE + some buffer


class TestPythonExecution:
    """Tests for Python code execution."""

    @pytest.mark.asyncio
    async def test_successful_python_code(self, mock_paramiko, sandbox_config, tmp_path):
        """Test successful Python code execution."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        # Initialize with default fixture behavior
        await provider.initialize(sandbox_config)

        # Python output - use factory pattern
        def create_python_result(*args, **kwargs):
            mock_stdin = MagicMock()
            mock_stdout = MagicMock()
            mock_stderr = MagicMock()
            mock_stdout.read.return_value = b"42\n"
            mock_stderr.read.return_value = b""
            mock_stdout.channel = MagicMock()
            mock_stdout.channel.recv_exit_status.return_value = 0
            return (mock_stdin, mock_stdout, mock_stderr)

        mock_client.exec_command.side_effect = create_python_result

        result = await provider.execute_python("print(6 * 7)")

        assert result.exit_code == 0
        assert "42" in result.stdout


class TestFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_write_file(self, mock_paramiko, sandbox_config, tmp_path):
        """Test writing a file."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Mock file handle for writing
        mock_file = MagicMock()
        mock_sftp.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_sftp.open.return_value.__exit__ = MagicMock(return_value=False)

        success = await provider.write_file("/workspace/test.txt", "Hello, World!")

        assert success is True

    @pytest.mark.asyncio
    async def test_read_file(self, mock_paramiko, sandbox_config, tmp_path):
        """Test reading a file."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Mock file handle for reading
        mock_file = MagicMock()
        mock_file.read.return_value = b"Hello, World!"
        mock_sftp.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_sftp.open.return_value.__exit__ = MagicMock(return_value=False)

        content = await provider.read_file("/workspace/test.txt")

        assert content == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, mock_paramiko, sandbox_config, tmp_path):
        """Test reading a nonexistent file."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Mock SFTP to raise FileNotFoundError
        mock_sftp.open.side_effect = FileNotFoundError("No such file")

        content = await provider.read_file("/workspace/nonexistent.txt")

        assert content is None

    @pytest.mark.asyncio
    async def test_file_exists(self, mock_paramiko, sandbox_config, tmp_path):
        """Test checking if file exists."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Mock stat for existing file
        mock_sftp.stat.return_value = MagicMock()

        exists = await provider.file_exists("/workspace/test.txt")

        assert exists is True

    @pytest.mark.asyncio
    async def test_file_not_exists(self, mock_paramiko, sandbox_config, tmp_path):
        """Test checking if file does not exist."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Mock stat to raise FileNotFoundError
        mock_sftp.stat.side_effect = FileNotFoundError("No such file")

        exists = await provider.file_exists("/workspace/nonexistent.txt")

        assert exists is False


class TestCleanup:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_successful_cleanup(self, mock_paramiko, sandbox_config, tmp_path):
        """Test successful cleanup."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        await provider.cleanup()

        assert provider._status == SandboxStatus.TERMINATED
        mock_client.close.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_no_connection(self, mock_paramiko):
        """Test cleanup when no connection exists."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )

        # Should not raise
        await provider.cleanup()

        assert provider._status == SandboxStatus.TERMINATED


class TestStatusAndInfo:
    """Tests for status and info methods."""

    def test_get_status(self, mock_paramiko):
        """Test status retrieval."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )
        provider._status = SandboxStatus.READY

        assert provider.get_status() == SandboxStatus.READY

    def test_get_access_level(self, mock_paramiko):
        """Test access level retrieval."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )
        provider._access_level = SandboxAccessLevel.FULL

        assert provider.get_access_level() == SandboxAccessLevel.FULL

    def test_get_info(self, mock_paramiko):
        """Test info retrieval."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )
        provider._status = SandboxStatus.READY
        provider._access_level = SandboxAccessLevel.RESTRICTED
        provider._workspace = "/home/testuser/workspace_abc123"

        info = provider.get_info()

        assert info["provider"] == "vm"
        assert info["host"] == "test-vm.example.com"

    @pytest.mark.asyncio
    async def test_is_healthy(self, mock_paramiko, sandbox_config, tmp_path):
        """Test health check - healthy."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Configure mock transport for health check
        mock_transport = mock_client.get_transport.return_value
        mock_transport.is_active.return_value = True

        # CRITICAL FIX: is_healthy() also runs execute_shell("echo 'health_check'")
        # and checks that "health_check" is in stdout. Must configure this!
        def create_health_check_result(*args, **kwargs):
            mock_stdin = MagicMock()
            mock_stdout = MagicMock()
            mock_stderr = MagicMock()
            mock_stdout.read.return_value = b"health_check\n"
            mock_stderr.read.return_value = b""
            mock_stdout.channel = MagicMock()
            mock_stdout.channel.recv_exit_status.return_value = 0
            return (mock_stdin, mock_stdout, mock_stderr)

        mock_client.exec_command.side_effect = create_health_check_result

        is_healthy = await provider.is_healthy()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_is_not_healthy_no_client(self, mock_paramiko):
        """Test health check - no client."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )

        is_healthy = await provider.is_healthy()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_is_not_healthy_inactive_transport(self, mock_paramiko, sandbox_config, tmp_path):
        """Test health check - inactive transport."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        await provider.initialize(sandbox_config)

        # Now set transport to inactive
        mock_transport = mock_client.get_transport.return_value
        mock_transport.is_active.return_value = False

        is_healthy = await provider.is_healthy()

        assert is_healthy is False
