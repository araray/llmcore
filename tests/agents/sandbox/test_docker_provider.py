# tests/agents/sandbox/test_docker_provider.py
"""
Unit tests for DockerSandboxProvider.

These tests use mocking to test the Docker provider without requiring
an actual Docker daemon. Integration tests with real Docker are in
a separate integration test file.

IMPORTANT FIX NOTES:
====================
The original tests failed because they tried to patch
'llmcore.agents.sandbox.docker_provider.docker', but the docker module
is imported LAZILY inside the _connect_docker() method, not at module level.

Fix approach: Use create=True when patching lazy imports, which creates
the attribute if it doesn't exist at the module level.

Alternative approach: Patch at the point of use (inside the class) or
use sys.modules patching.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llmcore.agents.sandbox.base import (
    SandboxAccessLevel,
    SandboxConfig,
    SandboxStatus,
)

# Assumes llmcore is installed or in PYTHONPATH
from llmcore.agents.sandbox.docker_provider import MAX_OUTPUT_SIZE, DockerSandboxProvider
from llmcore.agents.sandbox.exceptions import (
    SandboxAccessDenied,
    SandboxConnectionError,
    SandboxNotInitializedError,
)

# =============================================================================
# FIXTURES - FIXED VERSION
# =============================================================================


@pytest.fixture
def mock_docker_module():
    """
    Create a mock Docker module.

    This fixture patches the docker module using sys.modules, which works
    regardless of where/when the import happens (lazy or eager).
    """
    mock_docker = MagicMock()
    mock_client = MagicMock()

    # Setup the mock client
    mock_docker.from_env.return_value = mock_client
    mock_docker.DockerClient.return_value = mock_client
    mock_docker.errors = MagicMock()
    mock_docker.errors.ImageNotFound = Exception
    mock_docker.errors.DockerException = Exception

    # Mock version check
    mock_client.version.return_value = {"Version": "24.0.0"}
    mock_client.ping.return_value = True

    # Store original if exists
    original_docker = sys.modules.get("docker")

    # Patch sys.modules
    sys.modules["docker"] = mock_docker

    yield mock_docker, mock_client

    # Restore original
    if original_docker is not None:
        sys.modules["docker"] = original_docker
    else:
        sys.modules.pop("docker", None)


@pytest.fixture
def mock_docker_client(mock_docker_module):
    """
    Create a mock Docker client.

    FIXED: This fixture now uses sys.modules patching instead of
    trying to patch a non-existent module-level name.
    """
    mock_docker, mock_client = mock_docker_module
    return mock_client


@pytest.fixture
def docker_provider(mock_docker_client):
    """Create a DockerSandboxProvider with mocked Docker client."""
    # The mock is already in place via sys.modules
    provider = DockerSandboxProvider(
        image="python:3.11-slim",
        image_whitelist=["python:3.*-slim", "python:3.*-bookworm"],
        full_access_label="llmcore.sandbox.full_access=true",
        full_access_name_pattern="*-full-access",
    )
    provider._client = mock_docker_client
    return provider


@pytest.fixture
def sandbox_config():
    """Create a basic sandbox configuration."""
    return SandboxConfig(
        timeout_seconds=60, memory_limit="512m", cpu_limit=1.0, network_enabled=False
    )


# =============================================================================
# ALTERNATIVE APPROACH: patch with create=True
# =============================================================================


@pytest.fixture
def mock_docker_client_v2():
    """
    Alternative approach using create=True.

    This creates the 'docker' attribute on the module if it doesn't exist,
    allowing the patch to work even with lazy imports.
    """
    with patch(
        "llmcore.agents.sandbox.docker_provider.docker",
        create=True,  # Key fix: create the attribute if it doesn't exist
    ) as mock_docker:
        client = MagicMock()
        mock_docker.from_env.return_value = client
        mock_docker.DockerClient.return_value = client
        mock_docker.errors = MagicMock()
        mock_docker.errors.ImageNotFound = Exception
        mock_docker.errors.DockerException = Exception

        # Mock version check
        client.version.return_value = {"Version": "24.0.0"}
        client.ping.return_value = True

        yield client


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestDockerSandboxProviderInit:
    """Tests for DockerSandboxProvider initialization."""

    def test_init_with_local_docker(self, mock_docker_module):
        """Test initialization with local Docker daemon."""
        mock_docker, mock_client = mock_docker_module

        provider = DockerSandboxProvider(
            image="python:3.11-slim", image_whitelist=["python:3.*-slim"]
        )

        assert provider._image == "python:3.11-slim"
        assert "python:3.*-slim" in provider._image_whitelist
        assert provider._status == SandboxStatus.CREATED

    def test_init_with_remote_docker(self, mock_docker_module):
        """Test initialization with remote Docker host."""
        mock_docker, mock_client = mock_docker_module

        provider = DockerSandboxProvider(
            image="python:3.11-slim",
            image_whitelist=["python:3.*-slim"],
            docker_host="tcp://192.168.1.100:2375",
        )

        assert provider._docker_host == "tcp://192.168.1.100:2375"

    def test_init_fails_without_docker(self):
        """Test initialization fails when Docker is not available."""
        # Create a mock that raises on from_env
        mock_docker = MagicMock()
        mock_docker.from_env.side_effect = Exception("Cannot connect to Docker daemon")

        original_docker = sys.modules.get("docker")
        sys.modules["docker"] = mock_docker

        try:
            with pytest.raises(SandboxConnectionError):
                DockerSandboxProvider(image="python:3.11-slim", image_whitelist=["python:3.*-slim"])
        finally:
            if original_docker is not None:
                sys.modules["docker"] = original_docker
            else:
                sys.modules.pop("docker", None)


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
        # Mock image with full access label
        mock_image = MagicMock()
        mock_image.labels = {"llmcore.sandbox.full_access": "true"}
        mock_docker_client.images.get.return_value = mock_image

        level = docker_provider._determine_access_level()

        assert level == SandboxAccessLevel.FULL

    def test_full_access_by_name_pattern(self, docker_provider, mock_docker_client):
        """Test full access granted by image name pattern."""
        docker_provider._image = "python:3.11-slim-full-access"

        # Mock image without full access label
        mock_image = MagicMock()
        mock_image.labels = {}
        mock_docker_client.images.get.return_value = mock_image

        level = docker_provider._determine_access_level()

        assert level == SandboxAccessLevel.FULL

    def test_restricted_access_default(self, docker_provider, mock_docker_client):
        """Test restricted access is default."""
        # Mock image without full access label
        mock_image = MagicMock()
        mock_image.labels = {}
        mock_docker_client.images.get.return_value = mock_image

        docker_provider._image = "python:3.11-slim"  # Doesn't match full access pattern

        level = docker_provider._determine_access_level()

        assert level == SandboxAccessLevel.RESTRICTED

    def test_restricted_access_on_error(self, docker_provider, mock_docker_client):
        """Test restricted access on image inspection error."""
        mock_docker_client.images.get.side_effect = Exception("Image not found")

        level = docker_provider._determine_access_level()

        assert level == SandboxAccessLevel.RESTRICTED


class TestSandboxInitialization:
    """Tests for sandbox initialization."""

    @pytest.mark.asyncio
    async def test_successful_initialization(
        self, docker_provider, mock_docker_client, sandbox_config
    ):
        """Test successful sandbox initialization."""
        # Mock container creation
        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        # Mock image without full access label
        mock_image = MagicMock()
        mock_image.labels = {}
        mock_docker_client.images.get.return_value = mock_image

        await docker_provider.initialize(sandbox_config)

        assert docker_provider._status == SandboxStatus.READY
        assert docker_provider._container is mock_container

    @pytest.mark.asyncio
    async def test_initialization_with_non_whitelisted_image(
        self, mock_docker_module, sandbox_config
    ):
        """Test initialization fails with non-whitelisted image."""
        mock_docker, mock_client = mock_docker_module

        provider = DockerSandboxProvider(
            image="evil:malware",  # Not in whitelist
            image_whitelist=["python:*"],  # Only python images allowed
        )

        with pytest.raises(SandboxAccessDenied):
            await provider.initialize(sandbox_config)


class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_successful_command(self, docker_provider, mock_docker_client, sandbox_config):
        """Test successful command execution."""
        # Setup
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        # Mock exec_run
        mock_container.exec_run.return_value = (0, b"Hello World\n")

        result = await docker_provider.execute_shell("echo 'Hello World'")

        assert result.exit_code == 0
        assert "Hello World" in result.stdout

    @pytest.mark.asyncio
    async def test_command_with_nonzero_exit(
        self, docker_provider, mock_docker_client, sandbox_config
    ):
        """Test command with non-zero exit code."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (1, b"ls: cannot access 'nonexistent': No such file")

        result = await docker_provider.execute_shell("ls nonexistent")

        assert result.exit_code == 1
        assert "nonexistent" in result.stdout or "nonexistent" in result.stderr

    @pytest.mark.asyncio
    async def test_command_not_initialized(self, docker_provider):
        """Test command execution without initialization raises error."""
        with pytest.raises(SandboxNotInitializedError):
            await docker_provider.execute_shell("echo test")

    @pytest.mark.asyncio
    async def test_output_truncation(self, docker_provider, mock_docker_client, sandbox_config):
        """Test output truncation for large outputs."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        # Generate output larger than MAX_OUTPUT_SIZE
        large_output = b"x" * (MAX_OUTPUT_SIZE + 1000)
        mock_container.exec_run.return_value = (0, large_output)

        result = await docker_provider.execute_shell("cat largefile")

        # Output should be truncated
        assert len(result.stdout) <= MAX_OUTPUT_SIZE + 100  # Allow for truncation message


class TestPythonExecution:
    """Tests for Python code execution."""

    @pytest.mark.asyncio
    async def test_successful_python_code(
        self, docker_provider, mock_docker_client, sandbox_config
    ):
        """Test successful Python code execution."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (0, b"42\n")

        result = await docker_provider.execute_python("print(6 * 7)")

        assert result.exit_code == 0
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_python_syntax_error(self, docker_provider, mock_docker_client, sandbox_config):
        """Test Python syntax error handling."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        # write_file makes 2 calls (mkdir + base64 write), then python execution
        mock_container.exec_run.side_effect = [
            (0, b""),  # mkdir -p for parent directory
            (0, b""),  # base64 echo to write file content
            (
                1,
                b'  File "<stdin>", line 1\n    def broken(\n              ^\nSyntaxError: unexpected EOF',
            ),
        ]

        result = await docker_provider.execute_python("def broken(")

        assert result.exit_code == 1
        assert "SyntaxError" in result.stdout or "SyntaxError" in result.stderr


class TestFileOperations:
    """Tests for file operations."""

    @pytest.mark.asyncio
    async def test_write_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test file writing."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (0, b"")

        result = await docker_provider.write_file("/workspace/test.txt", "Hello")

        assert result is True

    @pytest.mark.asyncio
    async def test_read_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test file reading."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (0, b"File content here")

        content = await docker_provider.read_file("/workspace/test.txt")

        assert content == "File content here"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test reading nonexistent file."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (1, b"cat: nonexistent.txt: No such file")

        content = await docker_provider.read_file("nonexistent.txt")

        assert content is None

    @pytest.mark.asyncio
    async def test_file_exists(self, docker_provider, mock_docker_client, sandbox_config):
        """Test file existence check."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (0, b"yes")  # file_exists expects "yes" or "no"

        exists = await docker_provider.file_exists("/workspace/test.txt")

        assert exists is True

    @pytest.mark.asyncio
    async def test_delete_file(self, docker_provider, mock_docker_client, sandbox_config):
        """Test file deletion."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._config = sandbox_config
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        mock_container.exec_run.return_value = (0, b"")

        result = await docker_provider.delete_file("/workspace/test.txt")

        assert result is True


class TestCleanup:
    """Tests for sandbox cleanup."""

    @pytest.mark.asyncio
    async def test_successful_cleanup(self, docker_provider, mock_docker_client):
        """Test successful cleanup."""
        mock_container = MagicMock()
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY

        await docker_provider.cleanup()

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert docker_provider._status == SandboxStatus.TERMINATED

    @pytest.mark.asyncio
    async def test_cleanup_no_container(self, docker_provider):
        """Test cleanup when no container exists."""
        docker_provider._container = None

        # Should not raise
        await docker_provider.cleanup()

        assert docker_provider._status == SandboxStatus.TERMINATED


class TestStatusAndInfo:
    """Tests for status and info methods."""

    def test_get_status(self, docker_provider):
        """Test status retrieval."""
        docker_provider._status = SandboxStatus.READY
        assert docker_provider.get_status() == SandboxStatus.READY

    def test_get_access_level(self, docker_provider):
        """Test access level retrieval."""
        docker_provider._access_level = SandboxAccessLevel.FULL
        assert docker_provider.get_access_level() == SandboxAccessLevel.FULL

    def test_get_access_level_default(self, docker_provider):
        """Test access level default."""
        docker_provider._access_level = None
        assert docker_provider.get_access_level() == SandboxAccessLevel.RESTRICTED

    def test_get_info(self, docker_provider, mock_docker_client):
        """Test info retrieval."""
        mock_container = MagicMock()
        mock_container.id = "abc123456789"
        mock_container.short_id = "abc123456789"  # Docker uses short_id in get_info
        mock_container.name = "test-container"
        mock_container.status = "running"
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY
        docker_provider._access_level = SandboxAccessLevel.RESTRICTED

        info = docker_provider.get_info()

        assert info["provider"] == "docker"
        assert info["status"] == "ready"
        assert info["container_id"] == "abc123456789"

    @pytest.mark.asyncio
    async def test_is_healthy(self, docker_provider, mock_docker_client):
        """Test health check - healthy."""
        mock_container = MagicMock()
        mock_container.status = "running"
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY

        healthy = await docker_provider.is_healthy()

        assert healthy is True

    @pytest.mark.asyncio
    async def test_is_not_healthy(self, docker_provider, mock_docker_client):
        """Test health check - not healthy."""
        mock_container = MagicMock()
        mock_container.status = "exited"
        docker_provider._container = mock_container
        docker_provider._status = SandboxStatus.READY

        healthy = await docker_provider.is_healthy()

        assert healthy is False
