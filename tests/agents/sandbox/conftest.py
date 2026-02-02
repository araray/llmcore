# tests/agents/sandbox/conftest.py
"""
Pytest fixtures and configuration for sandbox tests.

This module provides fixtures for:
    - Mock sandbox providers
    - Test configurations
    - Docker availability detection
    - Temporary directories
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest

# Assumes llmcore is installed or PYTHONPATH includes src/
from llmcore.agents.sandbox.base import (
    ExecutionResult,
    FileInfo,
    SandboxAccessLevel,
    SandboxConfig,
    SandboxProvider,
    SandboxStatus,
)
from llmcore.agents.sandbox.registry import SandboxMode, SandboxRegistryConfig

# ==============================================================================
# Event Loop Configuration
# ==============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# Docker Availability
# ==============================================================================

def is_docker_available() -> bool:
    """Check if Docker is available for testing."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# Skip marker for tests requiring Docker
requires_docker = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker not available"
)


def is_paramiko_available() -> bool:
    """Check if Paramiko is available for testing."""
    try:
        import paramiko
        return True
    except ImportError:
        return False


# Skip marker for tests requiring Paramiko
requires_paramiko = pytest.mark.skipif(
    not is_paramiko_available(),
    reason="Paramiko not installed"
)


# ==============================================================================
# Configuration Fixtures
# ==============================================================================

@pytest.fixture
def sandbox_config() -> SandboxConfig:
    """Create a default SandboxConfig for testing."""
    return SandboxConfig(
        timeout_seconds=60,
        memory_limit="512m",
        cpu_limit=1.0,
        network_enabled=False
    )


@pytest.fixture
def full_access_sandbox_config() -> SandboxConfig:
    """Create a full-access SandboxConfig for testing."""
    return SandboxConfig(
        access_level=SandboxAccessLevel.FULL,
        timeout_seconds=60,
        memory_limit="1g",
        cpu_limit=2.0,
        network_enabled=True
    )


@pytest.fixture
def docker_registry_config() -> SandboxRegistryConfig:
    """Create a Docker-based registry config for testing."""
    return SandboxRegistryConfig(
        mode=SandboxMode.DOCKER,
        docker_enabled=True,
        docker_image="python:3.11-slim",
        docker_image_whitelist=["python:3.*-slim", "python:3.*-bookworm"],
        docker_full_access_label="llmcore.sandbox.full_access=true",
        docker_timeout_seconds=60,
        docker_memory_limit="512m",
        vm_enabled=False,
        share_path=tempfile.mkdtemp(prefix="llmcore_share_"),
        outputs_path=tempfile.mkdtemp(prefix="llmcore_outputs_")
    )


@pytest.fixture
def vm_registry_config() -> SandboxRegistryConfig:
    """Create a VM-based registry config for testing."""
    return SandboxRegistryConfig(
        mode=SandboxMode.VM,
        docker_enabled=False,
        vm_enabled=True,
        vm_host="localhost",
        vm_port=22,
        vm_username="test",
        vm_private_key_path=None,
        vm_use_ssh_agent=True,
        share_path=tempfile.mkdtemp(prefix="llmcore_share_"),
        outputs_path=tempfile.mkdtemp(prefix="llmcore_outputs_")
    )


# ==============================================================================
# Mock Sandbox Provider
# ==============================================================================

class MockSandboxProvider(SandboxProvider):
    """
    Mock sandbox provider for testing without Docker/VM.

    This provider simulates sandbox behavior in memory, useful for
    unit testing tools and registry logic.
    """

    def __init__(self, access_level: SandboxAccessLevel = SandboxAccessLevel.RESTRICTED):
        self._access_level = access_level
        self._status = SandboxStatus.CREATED
        self._config: SandboxConfig = None
        self._files: dict = {}  # path -> content
        self._initialized = False

    async def initialize(self, config: SandboxConfig) -> None:
        self._config = config
        self._config.access_level = self._access_level
        self._status = SandboxStatus.READY
        self._initialized = True
        self._files = {
            "/workspace": None,  # Directory marker
            "/workspace/output": None,
        }

    async def execute_shell(
        self,
        command: str,
        timeout: int = None,
        working_dir: str = None
    ) -> ExecutionResult:
        if not self._initialized:
            return ExecutionResult(exit_code=-1, stderr="Not initialized")

        # Simulate some basic commands
        self._status = SandboxStatus.EXECUTING

        if command.startswith("echo "):
            output = command[5:].strip("'\"")
            result = ExecutionResult(exit_code=0, stdout=output + "\n")
        elif command == "ls -la":
            files = "\n".join(f for f in self._files.keys() if self._files[f] is not None)
            result = ExecutionResult(exit_code=0, stdout=files)
        elif command.startswith("cat "):
            path = command[4:].strip("'\"")
            content = self._files.get(path)
            if content is not None:
                result = ExecutionResult(exit_code=0, stdout=content)
            else:
                result = ExecutionResult(exit_code=1, stderr=f"File not found: {path}")
        elif command.startswith("mkdir -p"):
            result = ExecutionResult(exit_code=0, stdout="")
        elif command.startswith("test -e"):
            path = command.split()[-1].strip("'\"")
            exists = "yes" if path in self._files else "no"
            result = ExecutionResult(exit_code=0, stdout=exists)
        elif command.startswith("python"):
            result = ExecutionResult(exit_code=0, stdout="Python 3.11.4")
        elif command.startswith("sqlite3"):
            result = ExecutionResult(exit_code=0, stdout="")
        elif command.startswith("rm -rf"):
            result = ExecutionResult(exit_code=0, stdout="")
        elif "health_check" in command:
            result = ExecutionResult(exit_code=0, stdout="health_check")
        else:
            result = ExecutionResult(exit_code=0, stdout=f"Executed: {command}")

        self._status = SandboxStatus.READY
        return result

    async def execute_python(
        self,
        code: str,
        timeout: int = None,
        working_dir: str = None
    ) -> ExecutionResult:
        if not self._initialized:
            return ExecutionResult(exit_code=-1, stderr="Not initialized")

        # Simple Python execution simulation
        self._status = SandboxStatus.EXECUTING

        try:
            # Capture basic print statements
            if "print(" in code:
                import re
                matches = re.findall(r"print\(['\"](.+?)['\"]\)", code)
                output = "\n".join(matches)
                result = ExecutionResult(exit_code=0, stdout=output + "\n")
            else:
                result = ExecutionResult(exit_code=0, stdout="")
        except Exception as e:
            result = ExecutionResult(exit_code=1, stderr=str(e))

        self._status = SandboxStatus.READY
        return result

    async def write_file(self, path: str, content: str, mode: str = "w") -> bool:
        if not path.startswith("/"):
            path = f"/workspace/{path}"

        if mode == "a" and path in self._files and self._files[path]:
            self._files[path] += content
        else:
            self._files[path] = content
        return True

    async def read_file(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        return self._files.get(path)

    async def write_file_binary(self, path: str, content: bytes) -> bool:
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        self._files[path] = content
        return True

    async def read_file_binary(self, path: str) -> bytes:
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        content = self._files.get(path)
        if isinstance(content, bytes):
            return content
        elif isinstance(content, str):
            return content.encode()
        return None

    async def list_files(self, path: str = ".", recursive: bool = False) -> list:
        if not path.startswith("/"):
            path = f"/workspace/{path}"

        files = []
        for file_path, content in self._files.items():
            if file_path.startswith(path):
                name = file_path.split("/")[-1]
                is_dir = content is None
                files.append(FileInfo(
                    path=file_path,
                    name=name,
                    is_directory=is_dir,
                    size_bytes=len(content) if content else 0
                ))
        return files

    async def file_exists(self, path: str) -> bool:
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        return path in self._files

    async def delete_file(self, path: str) -> bool:
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        if path in self._files:
            del self._files[path]
        return True

    async def create_directory(self, path: str) -> bool:
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        self._files[path] = None  # None = directory
        return True

    async def cleanup(self) -> None:
        self._files.clear()
        self._status = SandboxStatus.TERMINATED
        self._initialized = False

    async def is_healthy(self) -> bool:
        return self._initialized and self._status == SandboxStatus.READY

    def get_access_level(self) -> SandboxAccessLevel:
        return self._access_level

    def get_status(self) -> SandboxStatus:
        return self._status

    def get_config(self) -> SandboxConfig:
        return self._config

    def get_info(self) -> dict:
        return {
            "provider": "mock",
            "status": self._status.value,
            "access_level": self._access_level.value,
            "sandbox_id": self._config.sandbox_id if self._config else None
        }


@pytest.fixture
def mock_sandbox_provider() -> MockSandboxProvider:
    """Create a mock sandbox provider."""
    return MockSandboxProvider()


@pytest.fixture
def mock_full_access_provider() -> MockSandboxProvider:
    """Create a mock sandbox provider with full access."""
    return MockSandboxProvider(access_level=SandboxAccessLevel.FULL)


@pytest.fixture
async def initialized_mock_provider(
    mock_sandbox_provider: MockSandboxProvider,
    sandbox_config: SandboxConfig
) -> AsyncGenerator[MockSandboxProvider, None]:
    """Create an initialized mock provider."""
    await mock_sandbox_provider.initialize(sandbox_config)
    yield mock_sandbox_provider
    await mock_sandbox_provider.cleanup()


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory(prefix="llmcore_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Create an outputs directory for tests."""
    output_path = temp_dir / "outputs"
    output_path.mkdir()
    return output_path


@pytest.fixture
def share_dir(temp_dir: Path) -> Path:
    """Create a share directory for tests."""
    share_path = temp_dir / "share"
    share_path.mkdir()
    return share_path


# ==============================================================================
# Mock Docker Client
# ==============================================================================

@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    mock_client = MagicMock()

    # Mock version
    mock_client.version.return_value = {"Version": "24.0.0"}

    # Mock images
    mock_image = MagicMock()
    mock_image.labels = {}
    mock_client.images.get.return_value = mock_image
    mock_client.images.pull.return_value = mock_image

    # Mock container
    mock_container = MagicMock()
    mock_container.short_id = "abc12345"
    mock_container.name = "test-container"
    mock_container.status = "running"
    mock_container.exec_run.return_value = (0, (b"output\n", b""))
    mock_client.containers.run.return_value = mock_container

    return mock_client


# ==============================================================================
# Mock SSH Client (Paramiko)
# ==============================================================================

@pytest.fixture
def mock_ssh_client():
    """Create a mock SSH client."""
    mock_client = MagicMock()

    # Mock transport
    mock_transport = MagicMock()
    mock_transport.is_active.return_value = True
    mock_client.get_transport.return_value = mock_transport

    # Mock exec_command
    mock_stdin = MagicMock()
    mock_stdout = MagicMock()
    mock_stdout.read.return_value = b"output\n"
    mock_stdout.channel.recv_exit_status.return_value = 0
    mock_stderr = MagicMock()
    mock_stderr.read.return_value = b""
    mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

    # Mock SFTP
    mock_sftp = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp

    return mock_client


# ==============================================================================
# TOML Config Fixture
# ==============================================================================

@pytest.fixture
def sample_toml_config(temp_dir: Path) -> Path:
    """Create a sample TOML config file for testing."""
    config_path = temp_dir / "config.toml"

    config_content = '''
[agents.sandbox]
mode = "docker"
fallback_enabled = true

[agents.sandbox.docker]
enabled = true
image = "python:3.11-slim"
image_whitelist = ["python:3.*-slim"]
memory_limit = "512m"
timeout_seconds = 60

[agents.sandbox.vm]
enabled = false

[agents.sandbox.volumes]
share_path = "{share}"
outputs_path = "{outputs}"

[agents.sandbox.tools]
allowed = ["execute_shell", "execute_python"]
denied = ["sudo_execute"]
'''.format(share=str(temp_dir / "share"), outputs=str(temp_dir / "outputs"))

    config_path.write_text(config_content)
    return config_path
