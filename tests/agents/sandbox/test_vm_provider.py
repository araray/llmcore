# tests/sandbox/test_vm_provider.py
"""
Unit tests for VMSandboxProvider.

These tests use mocking to test the VM provider without requiring
an actual VM or SSH connection.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime
from pathlib import Path

import sys
# Assumes llmcore is installed or in PYTHONPATH

from llmcore.agents.sandbox.vm_provider import VMSandboxProvider, MAX_OUTPUT_SIZE
from llmcore.agents.sandbox.base import (
    SandboxConfig,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult
)
from llmcore.agents.sandbox.exceptions import (
    SandboxInitializationError,
    SandboxConnectionError,
    SandboxNotInitializedError
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_paramiko():
    """Create mock paramiko module."""
    with patch('llmcore.agents.sandbox.vm_provider.paramiko') as mock:
        # Mock key types
        mock.Ed25519Key = MagicMock()
        mock.RSAKey = MagicMock()
        mock.ECDSAKey = MagicMock()
        mock.DSSKey = MagicMock()
        mock.SSHClient = MagicMock
        mock.AutoAddPolicy = MagicMock()
        mock.AuthenticationException = Exception
        mock.SSHException = Exception
        yield mock


@pytest.fixture
def mock_ssh_client():
    """Create a mock SSH client."""
    client = MagicMock()

    # Mock transport
    transport = MagicMock()
    transport.is_active.return_value = True
    client.get_transport.return_value = transport

    # Mock SFTP
    sftp = MagicMock()
    client.open_sftp.return_value = sftp

    return client


@pytest.fixture
def vm_provider(mock_paramiko, mock_ssh_client):
    """Create a VMSandboxProvider with mocked SSH."""
    with patch.object(Path, 'exists', return_value=True):
        provider = VMSandboxProvider(
            host="192.168.1.100",
            port=22,
            username="agent",
            private_key_path="~/.ssh/test_key",
            full_access_hosts=["trusted-host-1"]
        )

    provider._client = mock_ssh_client
    provider._sftp = mock_ssh_client.open_sftp()

    return provider


@pytest.fixture
def sandbox_config():
    """Create a basic sandbox configuration."""
    return SandboxConfig(
        timeout_seconds=60,
        network_enabled=False
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestVMSandboxProviderInit:
    """Tests for VMSandboxProvider initialization."""

    def test_init_with_private_key(self, mock_paramiko):
        """Test initialization with private key."""
        with patch.object(Path, 'exists', return_value=True):
            provider = VMSandboxProvider(
                host="192.168.1.100",
                username="agent",
                private_key_path="~/.ssh/test_key"
            )

        assert provider._host == "192.168.1.100"
        assert provider._username == "agent"
        assert provider._status == SandboxStatus.CREATED

    def test_init_with_ssh_agent(self, mock_paramiko):
        """Test initialization with SSH agent."""
        provider = VMSandboxProvider(
            host="192.168.1.100",
            username="agent",
            use_ssh_agent=True
        )

        assert provider._use_ssh_agent

    def test_init_fails_without_auth_method(self, mock_paramiko):
        """Test initialization fails without authentication method."""
        with pytest.raises(SandboxInitializationError):
            VMSandboxProvider(
                host="192.168.1.100",
                private_key_path=None,
                use_ssh_agent=False
            )

    def test_init_with_custom_port(self, mock_paramiko):
        """Test initialization with custom SSH port."""
        provider = VMSandboxProvider(
            host="192.168.1.100",
            port=2222,
            use_ssh_agent=True
        )

        assert provider._port == 2222


class TestAccessLevelDetermination:
    """Tests for access level determination."""

    def test_full_access_for_whitelisted_host(self, mock_paramiko):
        """Test full access for host in whitelist."""
        provider = VMSandboxProvider(
            host="trusted-host",
            full_access_hosts=["trusted-host", "trusted-host-2"],
            use_ssh_agent=True
        )

        access_level = provider._determine_access_level()

        assert access_level == SandboxAccessLevel.FULL

    def test_restricted_access_for_unknown_host(self, mock_paramiko):
        """Test restricted access for host not in whitelist."""
        provider = VMSandboxProvider(
            host="unknown-host",
            full_access_hosts=["trusted-host"],
            use_ssh_agent=True
        )

        access_level = provider._determine_access_level()

        assert access_level == SandboxAccessLevel.RESTRICTED

    def test_restricted_access_with_empty_whitelist(self, mock_paramiko):
        """Test restricted access when whitelist is empty."""
        provider = VMSandboxProvider(
            host="any-host",
            full_access_hosts=[],
            use_ssh_agent=True
        )

        access_level = provider._determine_access_level()

        assert access_level == SandboxAccessLevel.RESTRICTED


# =============================================================================
# CONNECTION TESTS
# =============================================================================

class TestSSHConnection:
    """Tests for SSH connection handling."""

    @pytest.mark.asyncio
    async def test_successful_connection(self, vm_provider, mock_ssh_client, sandbox_config, mock_paramiko):
        """Test successful SSH connection."""
        # Setup
        vm_provider._status = SandboxStatus.INITIALIZING
        vm_provider._config = sandbox_config
        vm_provider._workspace = "/home/agent/workspace_test"

        # Mock exec_command for workspace setup
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b""
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        # Verify connection setup
        assert vm_provider._client is not None

    @pytest.mark.asyncio
    async def test_connection_with_key_file_not_found(self, mock_paramiko):
        """Test connection fails when key file not found."""
        with patch.object(Path, 'exists', return_value=False):
            provider = VMSandboxProvider(
                host="192.168.1.100",
                private_key_path="~/.ssh/nonexistent",
                use_ssh_agent=False
            )

        with pytest.raises(SandboxInitializationError):
            provider._load_private_key()


# =============================================================================
# EXECUTION TESTS
# =============================================================================

class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_successful_command(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test successful shell command execution."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock exec_command
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b"Hello, World!\n"
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        result = await vm_provider.execute_shell("echo 'Hello, World!'")

        assert result.success
        assert result.exit_code == 0
        assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_command_with_nonzero_exit(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test command with non-zero exit code."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b""
        stderr.read.return_value = b"command not found"
        stdout.channel.recv_exit_status.return_value = 127

        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        result = await vm_provider.execute_shell("nonexistent_command")

        assert not result.success
        assert result.exit_code == 127
        assert "command not found" in result.stderr

    @pytest.mark.asyncio
    async def test_command_not_initialized(self, vm_provider):
        """Test command execution before initialization."""
        vm_provider._client = None
        vm_provider._status = SandboxStatus.CREATED

        with pytest.raises(SandboxNotInitializedError):
            await vm_provider.execute_shell("echo test")

    @pytest.mark.asyncio
    async def test_output_truncation(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test output truncation for large outputs."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        large_output = b"x" * (MAX_OUTPUT_SIZE + 1000)

        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = large_output
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        result = await vm_provider.execute_shell("cat large_file")

        assert result.truncated
        assert len(result.stdout) <= MAX_OUTPUT_SIZE + 100


class TestPythonExecution:
    """Tests for Python code execution."""

    @pytest.mark.asyncio
    async def test_successful_python_code(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test successful Python code execution."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock file write and execution
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()

        # Return different outputs for different commands
        call_count = [0]
        def mock_read():
            call_count[0] += 1
            if call_count[0] <= 2:  # mkdir calls
                return b""
            return b"42\n"

        stdout.read.side_effect = mock_read
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        # Mock SFTP write
        vm_provider._sftp.open = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        result = await vm_provider.execute_python("print(6 * 7)")

        # Verify execution happened
        assert mock_ssh_client.exec_command.called


# =============================================================================
# FILE OPERATION TESTS
# =============================================================================

class TestFileOperations:
    """Tests for file operations via SFTP."""

    @pytest.mark.asyncio
    async def test_write_file(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test writing a file via SFTP."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock SFTP
        mock_file = MagicMock()
        vm_provider._sftp.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        vm_provider._sftp.open.return_value.__exit__ = MagicMock(return_value=False)

        # Mock mkdir command
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b""
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        success = await vm_provider.write_file("test.txt", "Hello, World!")

        assert success

    @pytest.mark.asyncio
    async def test_read_file(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test reading a file via SFTP."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock SFTP read
        mock_file = MagicMock()
        mock_file.read.return_value = b"file content"
        vm_provider._sftp.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        vm_provider._sftp.open.return_value.__exit__ = MagicMock(return_value=False)

        content = await vm_provider.read_file("test.txt")

        assert content == "file content"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test reading a nonexistent file."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock SFTP to raise FileNotFoundError
        vm_provider._sftp.open.side_effect = FileNotFoundError()

        content = await vm_provider.read_file("nonexistent.txt")

        assert content is None

    @pytest.mark.asyncio
    async def test_file_exists(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test checking if file exists."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock SFTP stat
        vm_provider._sftp.stat.return_value = MagicMock()

        exists = await vm_provider.file_exists("test.txt")

        assert exists

    @pytest.mark.asyncio
    async def test_file_not_exists(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test checking if file doesn't exist."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock SFTP stat to raise FileNotFoundError
        vm_provider._sftp.stat.side_effect = FileNotFoundError()

        # Mock shell fallback
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b"no\n"
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        exists = await vm_provider.file_exists("test.txt")

        assert not exists


# =============================================================================
# CLEANUP TESTS
# =============================================================================

class TestCleanup:
    """Tests for sandbox cleanup."""

    @pytest.mark.asyncio
    async def test_successful_cleanup(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test successful cleanup."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._workspace = "/home/agent/workspace_test"

        # Mock cleanup commands
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b""
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        await vm_provider.cleanup()

        assert vm_provider._status == SandboxStatus.TERMINATED
        assert vm_provider._client is None
        assert vm_provider._sftp is None

    @pytest.mark.asyncio
    async def test_cleanup_no_connection(self, vm_provider):
        """Test cleanup when no connection exists."""
        vm_provider._client = None
        vm_provider._sftp = None

        await vm_provider.cleanup()  # Should not raise

        assert vm_provider._status == SandboxStatus.TERMINATED


# =============================================================================
# STATUS AND INFO TESTS
# =============================================================================

class TestStatusAndInfo:
    """Tests for status and info methods."""

    def test_get_status(self, vm_provider):
        """Test getting sandbox status."""
        vm_provider._status = SandboxStatus.READY

        assert vm_provider.get_status() == SandboxStatus.READY

    def test_get_access_level(self, vm_provider):
        """Test getting access level."""
        vm_provider._access_level = SandboxAccessLevel.FULL

        assert vm_provider.get_access_level() == SandboxAccessLevel.FULL

    def test_get_info(self, vm_provider, sandbox_config):
        """Test getting sandbox info."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace_test"

        info = vm_provider.get_info()

        assert info["provider"] == "vm"
        assert info["host"] == "192.168.1.100"
        assert info["port"] == 22
        assert info["username"] == "agent"
        assert info["status"] == "ready"

    @pytest.mark.asyncio
    async def test_is_healthy(self, vm_provider, mock_ssh_client, sandbox_config):
        """Test health check."""
        vm_provider._config = sandbox_config
        vm_provider._status = SandboxStatus.READY
        vm_provider._access_level = SandboxAccessLevel.RESTRICTED
        vm_provider._workspace = "/home/agent/workspace"

        # Mock health check command
        stdin = MagicMock()
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b"health_check\n"
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        healthy = await vm_provider.is_healthy()

        assert healthy

    @pytest.mark.asyncio
    async def test_is_not_healthy_no_client(self, vm_provider):
        """Test unhealthy status when no client."""
        vm_provider._client = None

        healthy = await vm_provider.is_healthy()

        assert not healthy

    @pytest.mark.asyncio
    async def test_is_not_healthy_inactive_transport(self, vm_provider, mock_ssh_client):
        """Test unhealthy status when transport is inactive."""
        transport = MagicMock()
        transport.is_active.return_value = False
        mock_ssh_client.get_transport.return_value = transport

        healthy = await vm_provider.is_healthy()

        assert not healthy
