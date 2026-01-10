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

Fix approach: Use sys.modules patching OR create=True.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from llmcore.agents.sandbox.base import (
    ExecutionResult,
    SandboxAccessLevel,
    SandboxConfig,
    SandboxStatus,
)
from llmcore.agents.sandbox.exceptions import (
    SandboxAccessDenied,
    SandboxConnectionError,
    SandboxInitializationError,
    SandboxNotInitializedError,
)

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.vm_provider import VMSandboxProvider

# =============================================================================
# FIXTURES - FIXED VERSION
# =============================================================================


@pytest.fixture
def mock_paramiko_module():
    """
    Create a mock paramiko module.

    This fixture patches the paramiko module using sys.modules, which works
    regardless of where/when the import happens (lazy or eager).
    """
    mock_paramiko = MagicMock()

    # Setup mock classes and exceptions
    mock_paramiko.SSHClient = MagicMock
    mock_paramiko.AutoAddPolicy = MagicMock
    mock_paramiko.AuthenticationException = Exception
    mock_paramiko.SSHException = Exception

    # Mock key classes
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

    FIXED: This fixture now uses sys.modules patching instead of
    trying to patch a non-existent module-level name.
    """
    mock_client = MagicMock()
    mock_sftp = MagicMock()
    mock_transport = MagicMock()

    # Configure client methods
    mock_client.connect = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    mock_client.get_transport.return_value = mock_transport
    mock_transport.is_active.return_value = True

    # Configure exec_command
    mock_stdin = MagicMock()
    mock_stdout = MagicMock()
    mock_stderr = MagicMock()

    mock_stdout.read.return_value = b""
    mock_stderr.read.return_value = b""
    mock_stdout.channel.recv_exit_status.return_value = 0

    mock_client.exec_command.side_effect = lambda *args, **kwargs: (
        mock_stdin,
        mock_stdout,
        mock_stderr,
    )

    # Make SSHClient() return our mock
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
# ALTERNATIVE: patch with create=True
# =============================================================================


@pytest.fixture
def mock_paramiko_v2():
    """
    Alternative approach using create=True.
    """
    with patch(
        "llmcore.agents.sandbox.vm_provider.paramiko",
        create=True,  # Key fix
    ) as mock_paramiko:
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_transport = MagicMock()

        mock_paramiko.SSHClient.return_value = mock_client
        mock_paramiko.AutoAddPolicy.return_value = MagicMock()
        mock_paramiko.AuthenticationException = Exception
        mock_paramiko.SSHException = Exception

        mock_client.connect = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_client.get_transport.return_value = mock_transport
        mock_transport.is_active.return_value = True

        yield mock_paramiko, mock_client


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
        """Test initialization fails without auth method."""
        with pytest.raises(SandboxInitializationError):
            VMSandboxProvider(
                host="test-vm.example.com",
                username="testuser",
                # No private_key_path or use_ssh_agent
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

        # Setup mock exec_command for workspace setup
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

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

        # Setup for init
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        # Now test actual command
        mock_stdout.read.return_value = b"Hello World\n"
        mock_stdout.channel.recv_exit_status.return_value = 0

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

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        # Now test failing command
        mock_stdout.read.return_value = b""
        mock_stderr.read.return_value = b"ls: cannot access 'nonexistent': No such file"
        mock_stdout.channel.recv_exit_status.return_value = 1

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

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        # Large output
        large_output = b"x" * 200000
        mock_stdout.read.return_value = large_output

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

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        mock_stdout.read.return_value = b"42\n"

        result = await provider.execute_python("print(6 * 7)")

        assert result.exit_code == 0
        assert "42" in result.stdout


class TestFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_write_file(self, mock_paramiko, sandbox_config, tmp_path):
        """Test file writing."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        result = await provider.write_file("/workspace/test.txt", "Hello")

        assert result is True

    @pytest.mark.asyncio
    async def test_read_file(self, mock_paramiko, sandbox_config, tmp_path):
        """Test file reading."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        mock_stdout.read.return_value = b"File content"

        content = await provider.read_file("/workspace/test.txt")

        assert content == "File content"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, mock_paramiko, sandbox_config, tmp_path):
        """Test reading nonexistent file."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        mock_stdout.channel.recv_exit_status.return_value = 1
        mock_stderr.read.return_value = b"No such file"

        content = await provider.read_file("nonexistent.txt")

        assert content is None

    @pytest.mark.asyncio
    async def test_file_exists(self, mock_paramiko, sandbox_config, tmp_path):
        """Test file existence check."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        exists = await provider.file_exists("/workspace/test.txt")

        assert exists is True

    @pytest.mark.asyncio
    async def test_file_not_exists(self, mock_paramiko, sandbox_config, tmp_path):
        """Test file existence check - not exists."""
        mock_paramiko_module, mock_client, mock_sftp = mock_paramiko

        key_file = tmp_path / "test_key"
        key_file.write_text("fake")
        mock_paramiko_module.Ed25519Key.from_private_key_file.return_value = MagicMock()

        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path=str(key_file)
        )

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        mock_stdout.channel.recv_exit_status.return_value = 1

        exists = await provider.file_exists("/nonexistent.txt")

        assert exists is False


class TestCleanup:
    """Tests for sandbox cleanup."""

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

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

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

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        mock_transport = MagicMock()
        mock_transport.is_active.return_value = True
        mock_client.get_transport.return_value = mock_transport

        await provider.initialize(sandbox_config)

        healthy = await provider.is_healthy()

        assert healthy is True

    @pytest.mark.asyncio
    async def test_is_not_healthy_no_client(self, mock_paramiko):
        """Test health check - no client."""
        provider = VMSandboxProvider(
            host="test-vm.example.com", username="testuser", private_key_path="~/.ssh/id_rsa"
        )

        healthy = await provider.is_healthy()

        assert healthy is False

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

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.side_effect = lambda *a, **k: (MagicMock(), mock_stdout, mock_stderr)

        await provider.initialize(sandbox_config)

        # Now make transport inactive
        mock_transport = MagicMock()
        mock_transport.is_active.return_value = False
        mock_client.get_transport.return_value = mock_transport

        healthy = await provider.is_healthy()

        assert healthy is False
