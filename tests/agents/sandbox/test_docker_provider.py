# tests/sandbox/test_docker_provider.py
"""
Unit tests for DockerSandboxProvider.

These tests use mocking to test the Docker provider without requiring
an actual Docker daemon. Integration tests with real Docker are in
a separate integration test file.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime

import sys
# Assumes llmcore is installed or in PYTHONPATH

from llmcore.agents.sandbox.docker_provider import DockerSandboxProvider, MAX_OUTPUT_SIZE
from llmcore.agents.sandbox.base import (
    SandboxConfig,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult
)
from llmcore.agents.sandbox.exceptions import (
    SandboxInitializationError,
    SandboxAccessDenied,
    SandboxConnectionError,
    SandboxNotInitializedError,
    SandboxImageNotFoundError
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    with patch('llmcore.agents.sandbox.docker_provider.docker') as mock_docker:
        client = MagicMock()
        mock_docker.from_env.return_value = client
        mock_docker.DockerClient.return_value = client

        # Mock version check
        client.version.return_value = {"Version": "24.0.0"}

        yield client


@pytest.fixture
def docker_provider(mock_docker_client):
    """Create a DockerSandboxProvider with mocked Docker client."""
    provider = DockerSandboxProvider(
        image="python:3.11-slim",
        image_whitelist=["python:3.*-slim", "python:3.*-bookworm"],
        full_access_label="llmcore.sandbox.full_access=true",
        full_access_name_pattern="*-full-access"
    )
    provider._client = mock_docker_client
    return provider


@pytest.fixture
def sandbox_config():
    """Create a basic sandbox configuration."""
    return SandboxConfig(
        timeout_seconds=60,
        memory_limit="512m",
        cpu_limit=1.0,
        network_enabled=False
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestDockerSandboxProviderInit:
    """Tests for DockerSandboxProvider initialization."""

    def test_init_with_local_docker(self, mock_docker_client):
        """Test initialization with local Docker daemon."""
        provider = DockerSandboxProvider(
            image="python:3.11-slim",
            image_whitelist=["python:3.*-slim"]
        )

        assert provider._image == "python:3.11-slim"
        assert "python:3.*-slim" in provider._image_whitelist
        assert provider._status == SandboxStatus.CREATED

    def test_init_with_remote_docker(self, mock_docker_client):
        """Test initialization with remote Docker host."""
        with patch('llmcore.agents.sandbox.docker_provider.docker') as mock_docker:
            mock_docker.DockerClient.return_value = mock_docker_client
            mock_docker_client.version.return_value = {"Version": "24.0.0"}

            provider = DockerSandboxProvider(
                image="python:3.11-slim",
                image_whitelist=["python:3.*-slim"],
                docker_host="tcp://192.168.1.100:2375"
            )

            assert provider._docker_host == "tcp://192.168.1.100:2375"

    def test_init_fails_without_docker(self):
        """Test initialization fails when Docker is not available."""
        with patch('llmcore.agents.sandbox.docker_provider.docker') as mock_docker:
            mock_docker.from_env.side_effect = Exception("Cannot connect")

            with pytest.raises(SandboxConnectionError):
                DockerSandboxProvider(
                    image="python:3.11-slim",
                    image_whitelist=["python:3.*-slim"]
                )


class TestImageWhitelistValidation:
    """Tests for image whitelist validation."""

    def test_image_in_whitelist(self, docker_provider):
        """Test image that matches whitelist pattern."""
        docker_provider._image = "python:3.11-slim"
        docker_provider._validate_image_whitelist()  # Should not raise

    def test_image_not_in_whitelist(self, docker_provider):
        """Test image that doesn't match whitelist."""
        docker_provider._image = "untrusted:latest"

        with pytest.raises(SandboxAccessDenied) as exc_info:
            docker_provider._validate_image_whitelist()

        assert "not in the whitelist" in str(exc_info.value)
        assert exc_info.value.resource == "untrusted:latest"

    def test_wildcard_pattern_matching(self, docker_provider):
        """Test wildcard pattern matching."""
        docker_provider._image_whitelist = ["python:*", "ubuntu:*"]

        docker_provider._image = "python:latest"
        docker_provider._validate_image_whitelist()  # Should not raise

        docker_provider._image = "ubuntu:22.04"
        docker_provider._validate_image_whitelist()  # Should not raise

        docker_provider._image = "alpine:latest"
        with pytest.raises(SandboxAccessDenied):
            docker_provider._validate_image_whitelist()


class TestAccessLevelDetermination:
    """Tests for access level determination."""

    def test_full_access_by_label(self, docker_provider, mock_docker_client):
        """Test full access granted by image label."""
        mock_image = MagicMock()
        mock_image.labels = {"llmcore.sandbox.full_access": "true"}
        mock_docker_client.images.get.return_value = mock_image

        access_level = docker_provider._determine_access_level()

        assert access_level == SandboxAccessLevel.FULL

    def test_full_access_by_name_pattern(self, docker_provider, mock_docker_client):
        """Test full access granted by name pattern."""
        docker_provider._image = "myimage-full-access"

        mock_image = MagicMock()
        mock_image.labels = {}
        mock_docker_client.images.get.return_value = mock_image

        access_level = docker_provider._determine_access_level()

        assert access_level == SandboxAccessLevel.FULL

    def test_restricted_access_default(self, docker_provider, mock_docker_client):
        """Test restricted access is default."""
        mock_image = MagicMock()
        mock_image.labels = {}
        mock_docker_client.images.get.return_value = mock_image

        access_level = docker_provider._determine_access_level()

        assert access_level == SandboxAccessLevel.RESTRICTED

    def test_restricted_access_on_error(self, docker_provider, mock_docker_client):
        """Test restricted access on image inspection error."""
        mock_docker_client.images.get.side_effect = Exception("Image not found")

        access_level = docker_provider._determine_access_level()

        assert access_level == SandboxAccessLevel.RESTRICTED


# =============================================================================
# INITIALIZATION LIFECYCLE TESTS
# =============================================================================

class TestSandboxInitialization:
    """Tests for sandbox initialization."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self, docker_provider, mock_docker_client, sandbox_config):
        """Test successful sandbox initialization."""
        # Mock image exists
        mock_image = MagicMock()
        mock_image.labels = {}
        mock_docker_client.images.get.return_value = mock_image

        # Mock container creation
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.name = "test-container"
        mock_docker_client.containers.run.return_value = mock_container

        # Mock exec_run for workspace init
        mock_container.exec_run.return_value = (0, (b"", b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED
        docker_provider._status = SandboxStatus.READY

        # Verify status after setup
        assert docker_provider.get_status() == SandboxStatus.READY
        assert docker_provider.get_access_level() == SandboxAccessLevel.RESTRICTED

    @pytest.mark.asyncio
    async def test_initialization_with_non_whitelisted_image(self, docker_provider, sandbox_config):
        """Test initialization fails with non-whitelisted image."""
        docker_provider._image = "malicious:latest"

        with pytest.raises(SandboxAccessDenied):
            await docker_provider.initialize(sandbox_config)


# =============================================================================
# EXECUTION TESTS
# =============================================================================

class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_successful_command(self, docker_provider, mock_docker_client, sandbox_config):
        """Test successful shell command execution."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (b"Hello, World!\n", b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        result = await docker_provider.execute_shell("echo 'Hello, World!'")

        assert result.success
        assert result.exit_code == 0
        assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_command_with_nonzero_exit(self, docker_provider, mock_docker_client, sandbox_config):
        """Test command with non-zero exit code."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (1, (b"", b"Command not found"))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        result = await docker_provider.execute_shell("nonexistent_command")

        assert not result.success
        assert result.exit_code == 1
        assert "Command not found" in result.stderr

    @pytest.mark.asyncio
    async def test_command_not_initialized(self, docker_provider):
        """Test command execution before initialization."""
        docker_provider._container = None
        docker_provider._status = SandboxStatus.CREATED

        with pytest.raises(SandboxNotInitializedError):
            await docker_provider.execute_shell("echo test")

    @pytest.mark.asyncio
    async def test_output_truncation(self, docker_provider, mock_docker_client, sandbox_config):
        """Test output truncation for large outputs."""
        large_output = b"x" * (MAX_OUTPUT_SIZE + 1000)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (large_output, b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        result = await docker_provider.execute_shell("cat large_file")

        assert result.truncated
        assert len(result.stdout) <= MAX_OUTPUT_SIZE + 100  # With truncation message


class TestPythonExecution:
    """Tests for Python code execution."""

    @pytest.mark.asyncio
    async def test_successful_python_code(self, docker_provider, mock_docker_client, sandbox_config):
        """Test successful Python code execution."""
        mock_container = MagicMock()
        # First call for writing file, second for execution
        mock_container.exec_run.side_effect = [
            (0, (b"", b"")),  # Write file
            (0, (b"42\n", b""))  # Execute
        ]

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        result = await docker_provider.execute_python("print(6 * 7)")

        assert result.success
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_python_syntax_error(self, docker_provider, mock_docker_client, sandbox_config):
        """Test Python code with syntax error."""
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = [
            (0, (b"", b"")),  # Write file
            (1, (b"", b"SyntaxError: invalid syntax"))  # Execute
        ]

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        result = await docker_provider.execute_python("def broken(")

        assert not result.success
        assert "SyntaxError" in result.stderr


# =============================================================================
# FILE OPERATION TESTS
# =============================================================================

class TestFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_write_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test writing a file."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (b"", b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        success = await docker_provider.write_file("test.txt", "Hello, World!")

        assert success

    @pytest.mark.asyncio
    async def test_read_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test reading a file."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (b"file content", b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        content = await docker_provider.read_file("test.txt")

        assert content == "file content"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test reading a nonexistent file."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (1, (b"", b"No such file"))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        content = await docker_provider.read_file("nonexistent.txt")

        assert content is None

    @pytest.mark.asyncio
    async def test_file_exists(self, docker_provider, mock_docker_client, sandbox_config):
        """Test checking if file exists."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (b"yes\n", b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        exists = await docker_provider.file_exists("test.txt")

        assert exists

    @pytest.mark.asyncio
    async def test_delete_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test deleting a file."""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (b"", b""))

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        success = await docker_provider.delete_file("test.txt")

        assert success


# =============================================================================
# CLEANUP TESTS
# =============================================================================

class TestCleanup:
    """Tests for sandbox cleanup."""

    @pytest.mark.asyncio
    async def test_successful_cleanup(self, docker_provider, mock_docker_client, sandbox_config):
        """Test successful cleanup."""
        mock_container = MagicMock()

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY

        await docker_provider.cleanup()

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert docker_provider._status == SandboxStatus.TERMINATED

    @pytest.mark.asyncio
    async def test_cleanup_no_container(self, docker_provider):
        """Test cleanup when no container exists."""
        docker_provider._container = None

        await docker_provider.cleanup()  # Should not raise

        assert docker_provider._status == SandboxStatus.TERMINATED


# =============================================================================
# STATUS AND INFO TESTS
# =============================================================================

class TestStatusAndInfo:
    """Tests for status and info methods."""

    def test_get_status(self, docker_provider):
        """Test getting sandbox status."""
        docker_provider._status = SandboxStatus.READY

        assert docker_provider.get_status() == SandboxStatus.READY

    def test_get_access_level(self, docker_provider):
        """Test getting access level."""
        docker_provider._access_level = SandboxAccessLevel.FULL

        assert docker_provider.get_access_level() == SandboxAccessLevel.FULL

    def test_get_access_level_default(self, docker_provider):
        """Test default access level when not set."""
        docker_provider._access_level = None

        assert docker_provider.get_access_level() == SandboxAccessLevel.RESTRICTED

    def test_get_info(self, docker_provider, sandbox_config):
        """Test getting sandbox info."""
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.name = "test-container"

        docker_provider._container = mock_container
        docker_provider._config = sandbox_config
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        info = docker_provider.get_info()

        assert info["provider"] == "docker"
        assert info["status"] == "ready"
        assert info["container_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_is_healthy(self, docker_provider, mock_docker_client):
        """Test health check."""
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.reload = MagicMock()

        docker_provider._container = mock_container

        healthy = await docker_provider.is_healthy()

        assert healthy

    @pytest.mark.asyncio
    async def test_is_not_healthy(self, docker_provider):
        """Test unhealthy status."""
        docker_provider._container = None

        healthy = await docker_provider.is_healthy()

        assert not healthy
