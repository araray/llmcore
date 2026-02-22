# src/llmcore/agents/sandbox/docker_provider.py
"""
Docker-based sandbox execution using docker-py SDK.

This provider manages Docker containers for isolated code execution.
It supports both restricted and full-access modes based on image labels/names.

Security Model:
    - Images must be explicitly whitelisted to be used
    - Full access requires either:
        - A specific label (e.g., llmcore.sandbox.full_access=true)
        - Image name matching a full-access pattern
    - Restricted mode disables network and enforces tool whitelist

Usage:
    >>> provider = DockerSandboxProvider(
    ...     image="python:3.11-slim",
    ...     image_whitelist=["python:3.*-slim"]
    ... )
    >>> await provider.initialize(SandboxConfig())
    >>> result = await provider.execute_shell("python --version")
    >>> await provider.cleanup()

Requirements:
    - docker-py package (pip install docker)
    - Docker daemon running and accessible
"""

import asyncio
import base64
import fnmatch
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import (
    ExecutionResult,
    FileInfo,
    SandboxAccessLevel,
    SandboxConfig,
    SandboxProvider,
    SandboxStatus,
)
from .exceptions import (
    SandboxAccessDenied,
    SandboxCleanupError,
    SandboxConnectionError,
    SandboxImageNotFoundError,
    SandboxInitializationError,
    SandboxNotInitializedError,
)

logger = logging.getLogger(__name__)

# Maximum output size before truncation (100KB)
MAX_OUTPUT_SIZE = 100_000

# Sentinel value for heredoc EOF marker
HEREDOC_EOF = "LLMCORE_HEREDOC_EOF_MARKER"


class DockerSandboxProvider(SandboxProvider):
    """
    Docker-based sandbox using docker-py SDK.

    This provider creates and manages Docker containers for isolated
    code execution. It ensures that code NEVER runs on the host system.

    Access Level Determination:
        1. If image has the full_access_label -> FULL
        2. If image name matches full_access_name_pattern -> FULL
        3. Otherwise -> RESTRICTED

    Security Features:
        - Image whitelist validation (required)
        - Resource limits (memory, CPU)
        - Network isolation (disabled by default for restricted)
        - Automatic cleanup on errors

    Attributes:
        _image: Docker image to use
        _image_whitelist: List of allowed image patterns
        _full_access_label: Label that grants full access
        _full_access_name_pattern: Name pattern for full access
        _docker_host: Optional remote Docker host URL
        _client: Docker client instance
        _container: Running container instance
        _config: Active sandbox configuration
        _access_level: Determined access level
        _status: Current sandbox status
    """

    def __init__(
        self,
        image: str,
        image_whitelist: list[str],
        full_access_label: str = "llmcore.sandbox.full_access=true",
        full_access_name_pattern: str | None = None,
        docker_host: str | None = None,
        auto_pull: bool = True,
    ):
        """
        Initialize Docker sandbox provider.

        Args:
            image: Docker image to use (e.g., "python:3.11-slim")
            image_whitelist: List of allowed image patterns (glob-style)
            full_access_label: Label that grants full access (format: key=value)
            full_access_name_pattern: Image name pattern for full access (glob-style)
            docker_host: Optional Docker host URL for remote Docker
            auto_pull: Whether to automatically pull images if not found locally

        Raises:
            SandboxInitializationError: If Docker client cannot be created
        """
        self._image = image
        self._image_whitelist = image_whitelist
        self._full_access_label = full_access_label
        self._full_access_name_pattern = full_access_name_pattern
        self._docker_host = docker_host
        self._auto_pull = auto_pull

        # Runtime state (set during initialize)
        self._client: Any | None = None  # docker.DockerClient
        self._container: Any | None = None  # docker.Container
        self._config: SandboxConfig | None = None
        self._access_level: SandboxAccessLevel | None = None
        self._status: SandboxStatus = SandboxStatus.CREATED

        # Connect to Docker daemon
        self._connect_docker()

    def _connect_docker(self) -> None:
        """
        Connect to Docker daemon.

        Raises:
            SandboxInitializationError: If connection fails
        """
        try:
            import docker

            if self._docker_host:
                self._client = docker.DockerClient(base_url=self._docker_host)
                logger.info(f"Connected to remote Docker: {self._docker_host}")
            else:
                self._client = docker.from_env()
                logger.debug("Connected to local Docker daemon")

            # Verify connection
            version = self._client.version()
            logger.info(f"Docker version: {version.get('Version', 'unknown')}")

        except ImportError:
            raise SandboxInitializationError(
                "docker-py package not installed. Install with: pip install docker"
            )
        except Exception as e:
            raise SandboxConnectionError(
                f"Failed to connect to Docker daemon: {e}",
                host=self._docker_host or "local",
                connection_type="docker",
            )

    def _validate_image_whitelist(self) -> None:
        """
        Validate that the image is in the whitelist.

        Raises:
            SandboxAccessDenied: If image is not whitelisted
        """
        for pattern in self._image_whitelist:
            if fnmatch.fnmatch(self._image, pattern):
                logger.debug(f"Image '{self._image}' matches whitelist pattern '{pattern}'")
                return

        raise SandboxAccessDenied(
            f"Docker image '{self._image}' is not in the whitelist",
            resource=self._image,
            reason="Image not in whitelist",
            policy=f"Allowed patterns: {self._image_whitelist}",
        )

    def _determine_access_level(self) -> SandboxAccessLevel:
        """
        Determine access level based on image labels and name patterns.

        Returns:
            SandboxAccessLevel.FULL or SandboxAccessLevel.RESTRICTED
        """
        try:
            image_obj = self._client.images.get(self._image)
            labels = image_obj.labels or {}

            # Parse full_access_label (format: key=value)
            if "=" in self._full_access_label:
                label_key, label_value = self._full_access_label.split("=", 1)
                if labels.get(label_key) == label_value:
                    logger.info(
                        f"Image '{self._image}' has full access label '{self._full_access_label}'"
                    )
                    return SandboxAccessLevel.FULL

            # Check name pattern
            if self._full_access_name_pattern:
                if fnmatch.fnmatch(self._image, self._full_access_name_pattern):
                    logger.info(
                        f"Image '{self._image}' matches full access pattern '{self._full_access_name_pattern}'"
                    )
                    return SandboxAccessLevel.FULL

            logger.info(f"Image '{self._image}' will run in RESTRICTED mode")
            return SandboxAccessLevel.RESTRICTED

        except Exception as e:
            logger.warning(f"Could not determine access level, defaulting to RESTRICTED: {e}")
            return SandboxAccessLevel.RESTRICTED

    async def _pull_image(self) -> None:
        """
        Pull the Docker image if not available locally.

        Raises:
            SandboxImageNotFoundError: If image cannot be pulled
        """
        try:
            self._client.images.get(self._image)
            logger.debug(f"Image '{self._image}' found locally")
        except Exception:
            if not self._auto_pull:
                raise SandboxImageNotFoundError(
                    f"Image '{self._image}' not found locally and auto_pull is disabled",
                    image=self._image,
                )

            logger.info(f"Pulling Docker image '{self._image}'...")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.images.pull(self._image)
                )
                logger.info(f"Successfully pulled image '{self._image}'")
            except Exception as e:
                raise SandboxImageNotFoundError(
                    f"Failed to pull image '{self._image}': {e}", image=self._image
                )

    async def initialize(self, config: SandboxConfig) -> None:
        """
        Initialize the Docker container sandbox.

        This method:
            1. Validates the image against the whitelist
            2. Pulls the image if needed
            3. Determines access level
            4. Creates and starts the container
            5. Initializes workspace directories
            6. Creates ephemeral SQLite database

        Args:
            config: Sandbox configuration

        Raises:
            SandboxInitializationError: If container creation fails
            SandboxAccessDenied: If image is not whitelisted
        """
        self._status = SandboxStatus.INITIALIZING
        self._config = config

        try:
            # Validate image whitelist
            self._validate_image_whitelist()

            # Pull image if needed
            await self._pull_image()

            # Determine access level
            self._access_level = self._determine_access_level()

            # Update config access level
            config.access_level = self._access_level

            # Prepare volume mounts
            volumes = self._prepare_volumes(config)

            # Determine network mode
            network_mode = self._determine_network_mode(config)

            # Create container
            await self._create_container(config, volumes, network_mode)

            # Initialize workspace
            await self._init_workspace()

            # Initialize ephemeral SQLite
            await self._init_ephemeral_db()

            self._status = SandboxStatus.READY
            logger.info(
                f"Docker sandbox initialized: {self._container.short_id} "
                f"(access: {self._access_level.value}, network: {network_mode or 'default'})"
            )

        except (SandboxAccessDenied, SandboxImageNotFoundError):
            self._status = SandboxStatus.ERROR
            raise
        except Exception as e:
            self._status = SandboxStatus.ERROR
            logger.error(f"Failed to initialize Docker sandbox: {e}")
            # Cleanup any partial resources
            await self._cleanup_partial()
            raise SandboxInitializationError(f"Failed to create Docker container: {e}")

    def _prepare_volumes(self, config: SandboxConfig) -> dict[str, dict[str, str]]:
        """
        Prepare volume mount specifications.

        Args:
            config: Sandbox configuration

        Returns:
            Docker volume specification dictionary
        """
        volumes = {}

        if config.share_mount_host:
            host_share = Path(config.share_mount_host).expanduser().resolve()
            host_share.mkdir(parents=True, exist_ok=True)
            volumes[str(host_share)] = {"bind": config.share_mount_container, "mode": "rw"}
            logger.debug(f"Share volume: {host_share} -> {config.share_mount_container}")

        if config.output_mount_host:
            host_output = Path(config.output_mount_host).expanduser().resolve()
            host_output.mkdir(parents=True, exist_ok=True)
            volumes[str(host_output)] = {"bind": config.output_mount_container, "mode": "rw"}
            logger.debug(f"Output volume: {host_output} -> {config.output_mount_container}")

        return volumes

    def _determine_network_mode(self, config: SandboxConfig) -> str | None:
        """
        Determine Docker network mode based on config and access level.

        Args:
            config: Sandbox configuration

        Returns:
            Network mode string or None for default
        """
        # Full access gets network by default
        if self._access_level == SandboxAccessLevel.FULL:
            return None  # Use default Docker networking

        # Restricted: honor config, default to disabled
        if not config.network_enabled:
            return "none"

        return None

    async def _create_container(
        self, config: SandboxConfig, volumes: dict, network_mode: str | None
    ) -> None:
        """
        Create and start the Docker container.

        Args:
            config: Sandbox configuration
            volumes: Volume mount specifications
            network_mode: Network mode
        """
        container_name = f"llmcore-sandbox-{config.sandbox_id[:8]}"

        # Prepare container labels
        labels = {
            "llmcore.sandbox.id": config.sandbox_id,
            "llmcore.sandbox.access_level": self._access_level.value,
            "llmcore.sandbox.created_at": datetime.utcnow().isoformat(),
        }
        labels.update(config.labels)

        # Create container
        self._container = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._client.containers.run(
                self._image,
                command="sleep infinity",  # Keep container running
                detach=True,
                remove=False,  # We'll remove in cleanup
                name=container_name,
                volumes=volumes if volumes else None,
                environment=config.environment_vars,
                mem_limit=config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(config.cpu_limit * 100000),
                network_mode=network_mode,
                working_dir=config.working_directory,
                labels=labels,
                stdin_open=True,
                tty=False,
            ),
        )

    async def _init_workspace(self) -> None:
        """Initialize workspace directories in the container."""
        commands = [
            f"mkdir -p {self._config.working_directory}",
            "mkdir -p /tmp",
            f"mkdir -p {self._config.output_mount_container}",
        ]

        for cmd in commands:
            result = await self.execute_shell(cmd)
            if not result.success:
                logger.warning(f"Workspace init command failed: {cmd}")

    async def _init_ephemeral_db(self) -> None:
        """Initialize the ephemeral SQLite database."""
        init_sql = """
CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS agent_files (
    path TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    size_bytes INTEGER,
    description TEXT
);
"""
        db_path = self._config.ephemeral_db_path

        # Write SQL to temp file and execute
        result = await self.execute_shell(f"sqlite3 {db_path} << 'EOF'\n{init_sql}\nEOF")

        if result.success:
            logger.debug(f"Initialized ephemeral SQLite at {db_path}")
        else:
            logger.warning(f"Failed to init ephemeral DB: {result.stderr}")

    async def _cleanup_partial(self) -> None:
        """Clean up partial resources after failed initialization."""
        if self._container:
            try:
                self._container.stop(timeout=5)
                self._container.remove(force=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup partial container: {e}")
            self._container = None

    def _ensure_initialized(self) -> None:
        """
        Ensure sandbox is initialized before operations.

        Raises:
            SandboxNotInitializedError: If not initialized
        """
        if not self._container or self._status not in (
            SandboxStatus.READY,
            SandboxStatus.EXECUTING,
            SandboxStatus.INITIALIZING,
        ):
            raise SandboxNotInitializedError(
                "Sandbox not initialized or not ready",
                sandbox_id=self._config.sandbox_id if self._config else None,
            )

    async def execute_shell(
        self, command: str, timeout: int | None = None, working_dir: str | None = None
    ) -> ExecutionResult:
        """
        Execute a shell command in the Docker container.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (None = use config default)
            working_dir: Working directory (None = use default)

        Returns:
            ExecutionResult with command output
        """
        self._ensure_initialized()

        effective_timeout = timeout or self._config.timeout_seconds
        effective_workdir = working_dir or self._config.working_directory
        start_time = time.time()

        self._status = SandboxStatus.EXECUTING

        try:
            # Execute command using exec_run
            exit_code, output = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._container.exec_run(
                        cmd=["bash", "-c", command],
                        stdout=True,
                        stderr=True,
                        demux=True,
                        workdir=effective_workdir,
                    ),
                ),
                timeout=effective_timeout,
            )

            execution_time = time.time() - start_time

            # Handle output
            if isinstance(output, tuple):
                stdout_bytes, stderr_bytes = output
            else:
                stdout_bytes, stderr_bytes = output, b""

            stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
            stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")

            # Truncate if needed
            truncated = False
            if len(stdout) > MAX_OUTPUT_SIZE:
                stdout = stdout[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                truncated = True
            if len(stderr) > MAX_OUTPUT_SIZE:
                stderr = stderr[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                truncated = True

            self._status = SandboxStatus.READY

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time_seconds=execution_time,
                truncated=truncated,
                timed_out=False,
            )

        except TimeoutError:
            self._status = SandboxStatus.READY
            execution_time = time.time() - start_time

            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {effective_timeout} seconds",
                execution_time_seconds=execution_time,
                timed_out=True,
            )

        except Exception as e:
            self._status = SandboxStatus.READY
            logger.error(f"Shell execution failed: {e}")

            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time_seconds=time.time() - start_time,
            )

    async def execute_python(
        self, code: str, timeout: int | None = None, working_dir: str | None = None
    ) -> ExecutionResult:
        """
        Execute Python code in the Docker container.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            working_dir: Working directory

        Returns:
            ExecutionResult with execution output
        """
        self._ensure_initialized()

        # Write code to temp file
        temp_file = f"/tmp/agent_code_{self._config.sandbox_id[:8]}.py"
        write_success = await self.write_file(temp_file, code)

        if not write_success:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="Failed to write Python code to temp file"
            )

        # Execute the Python file
        return await self.execute_shell(
            f"python3 {temp_file}", timeout=timeout, working_dir=working_dir
        )

    async def write_file(self, path: str, content: str, mode: str = "w") -> bool:
        """
        Write content to a file in the container.

        Args:
            path: File path
            content: Content to write
            mode: Write mode ("w" or "a")

        Returns:
            True if successful
        """
        self._ensure_initialized()

        # Normalize path
        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        # Ensure parent directory exists
        parent_dir = str(Path(path).parent)
        await self.execute_shell(f"mkdir -p '{parent_dir}'")

        # Use heredoc for safe file writing with arbitrary content
        # Escape single quotes in content for shell safety
        redirect = ">" if mode == "w" else ">>"

        # Use base64 encoding for truly safe transfer of any content
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")

        result = await self.execute_shell(
            f"echo '{encoded_content}' | base64 -d {redirect} '{path}'"
        )

        return result.success

    async def read_file(self, path: str) -> str | None:
        """
        Read content from a file in the container.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        self._ensure_initialized()

        # Normalize path
        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        result = await self.execute_shell(f"cat '{path}' 2>/dev/null")

        if result.success:
            return result.stdout
        return None

    async def write_file_binary(self, path: str, content: bytes) -> bool:
        """
        Write binary content to a file in the container.

        Args:
            path: File path
            content: Binary content

        Returns:
            True if successful
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        parent_dir = str(Path(path).parent)
        await self.execute_shell(f"mkdir -p '{parent_dir}'")

        # Use base64 encoding for binary transfer
        encoded_content = base64.b64encode(content).decode("ascii")

        result = await self.execute_shell(f"echo '{encoded_content}' | base64 -d > '{path}'")

        return result.success

    async def read_file_binary(self, path: str) -> bytes | None:
        """
        Read binary content from a file in the container.

        Args:
            path: File path

        Returns:
            Binary content or None if not found
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        result = await self.execute_shell(f"base64 '{path}' 2>/dev/null")

        if result.success and result.stdout.strip():
            try:
                return base64.b64decode(result.stdout.strip())
            except Exception:
                return None
        return None

    async def list_files(self, path: str = ".", recursive: bool = False) -> list[FileInfo]:
        """
        List files in a directory in the container.

        Args:
            path: Directory path
            recursive: If True, list recursively

        Returns:
            List of FileInfo objects
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        # Use ls with specific format for parsing
        if recursive:
            cmd = f"find '{path}' -type f -o -type d 2>/dev/null | head -1000"
        else:
            cmd = f"ls -la '{path}' 2>/dev/null"

        result = await self.execute_shell(cmd)

        if not result.success:
            return []

        files = []
        for line in result.stdout.strip().split("\n"):
            if not line or line.startswith("total"):
                continue

            if recursive:
                # Simple path listing from find
                item_path = line.strip()
                if item_path:
                    name = Path(item_path).name
                    files.append(
                        FileInfo(
                            path=item_path,
                            name=name,
                            is_directory=False,  # Would need additional stat call
                        )
                    )
            else:
                # Parse ls -la output
                parts = line.split()
                if len(parts) >= 9:
                    permissions = parts[0]
                    size = int(parts[4]) if parts[4].isdigit() else 0
                    name = " ".join(parts[8:])

                    if name in (".", ".."):
                        continue

                    files.append(
                        FileInfo(
                            path=f"{path}/{name}",
                            name=name,
                            is_directory=permissions.startswith("d"),
                            size_bytes=size,
                            permissions=permissions,
                        )
                    )

        return files

    async def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in the container.

        Args:
            path: File path

        Returns:
            True if exists
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        result = await self.execute_shell(f"test -e '{path}' && echo 'yes' || echo 'no'")

        return result.stdout.strip() == "yes"

    async def delete_file(self, path: str) -> bool:
        """
        Delete a file in the container.

        Args:
            path: File path

        Returns:
            True if deleted successfully
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        result = await self.execute_shell(f"rm -f '{path}'")
        return result.success

    async def create_directory(self, path: str) -> bool:
        """
        Create a directory in the container.

        Args:
            path: Directory path

        Returns:
            True if created successfully
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._config.working_directory}/{path}"

        result = await self.execute_shell(f"mkdir -p '{path}'")
        return result.success

    async def cleanup(self) -> None:
        """
        Clean up the Docker container and resources.
        """
        self._status = SandboxStatus.CLEANING_UP
        errors = []

        if self._container:
            try:
                logger.info(f"Cleaning up Docker sandbox: {self._container.short_id}")

                # Stop container
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._container.stop(timeout=10)
                )

                # Remove container
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._container.remove(force=True)
                )

                logger.debug("Docker container stopped and removed")

            except Exception as e:
                errors.append(f"container cleanup: {e}")
                logger.warning(f"Error during Docker cleanup: {e}")
            finally:
                self._container = None

        self._status = SandboxStatus.TERMINATED

        if errors:
            raise SandboxCleanupError(
                f"Cleanup completed with errors: {'; '.join(errors)}",
                resources_leaked=errors,
                partial_cleanup=True,
                sandbox_id=self._config.sandbox_id if self._config else None,
            )

    async def is_healthy(self) -> bool:
        """
        Check if the Docker container is running and healthy.

        Returns:
            True if container is running
        """
        if not self._container:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(None, self._container.reload)
            return self._container.status == "running"
        except Exception:
            return False

    def get_access_level(self) -> SandboxAccessLevel:
        """Get the access level of this sandbox."""
        return self._access_level or SandboxAccessLevel.RESTRICTED

    def get_status(self) -> SandboxStatus:
        """Get the current status of the sandbox."""
        return self._status

    def get_config(self) -> SandboxConfig | None:
        """Get the configuration used for this sandbox."""
        return self._config

    def get_info(self) -> dict[str, Any]:
        """
        Get detailed information about the Docker sandbox.

        Returns:
            Dictionary with sandbox details
        """
        info = {
            "provider": "docker",
            "image": self._image,
            "status": self._status.value,
            "access_level": self._access_level.value if self._access_level else None,
            "container_id": self._container.short_id if self._container else None,
            "container_name": self._container.name if self._container else None,
        }

        if self._config:
            info.update(
                {
                    "sandbox_id": self._config.sandbox_id,
                    "memory_limit": self._config.memory_limit,
                    "cpu_limit": self._config.cpu_limit,
                    "network_enabled": self._config.network_enabled,
                    "working_directory": self._config.working_directory,
                }
            )

        return info
