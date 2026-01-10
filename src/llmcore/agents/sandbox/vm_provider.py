# src/llmcore/agents/sandbox/vm_provider.py
"""
VM-based sandbox execution using Paramiko SSH with PKI authentication.

This provider connects to a remote Linux VM via SSH for isolated code execution.
It supports both restricted and full-access modes based on host configuration.

Security Model:
    - Uses PKI authentication exclusively (no passwords)
    - Full access requires host to be in full_access_hosts list
    - Restricted mode blocks network and enforces tool whitelist

Usage:
    >>> provider = VMSandboxProvider(
    ...     host="192.168.1.100",
    ...     username="agent",
    ...     private_key_path="~/.ssh/llmcore_agent_key"
    ... )
    >>> await provider.initialize(SandboxConfig())
    >>> result = await provider.execute_shell("python --version")
    >>> await provider.cleanup()

Requirements:
    - paramiko package (pip install paramiko)
    - SSH access to the VM with PKI authentication
    - Linux-based VM (required for shell compatibility)
"""

import asyncio
import base64
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    SandboxExecutionError,
    SandboxInitializationError,
    SandboxNotInitializedError,
    SandboxTimeoutError,
)

logger = logging.getLogger(__name__)

# Maximum output size before truncation (100KB)
MAX_OUTPUT_SIZE = 100_000


class VMSandboxProvider(SandboxProvider):
    """
    VM-based sandbox using SSH/PKI authentication.

    This provider connects to a remote Linux VM for isolated code execution.
    It ensures that code NEVER runs on the host system.

    Access Level Determination:
        1. If host is in full_access_hosts list -> FULL
        2. Otherwise -> RESTRICTED

    Security Features:
        - PKI authentication only (no passwords)
        - Isolated workspace per sandbox
        - Automatic cleanup of workspace on termination

    Attributes:
        _host: VM hostname or IP address
        _port: SSH port
        _username: SSH username
        _private_key_path: Path to private key file
        _full_access_hosts: List of hosts with full access
        _client: Paramiko SSH client
        _sftp: SFTP client for file operations
        _config: Active sandbox configuration
        _access_level: Determined access level
        _status: Current sandbox status
        _workspace: Remote workspace directory path
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "agent",
        private_key_path: Optional[str] = None,
        full_access_hosts: Optional[List[str]] = None,
        use_ssh_agent: bool = True,
        connection_timeout: int = 30,
    ):
        """
        Initialize VM sandbox provider.

        Args:
            host: VM hostname or IP address
            port: SSH port (default: 22)
            username: SSH username (default: "agent")
            private_key_path: Path to private key file
            full_access_hosts: List of hosts with full access
            use_ssh_agent: Whether to use SSH agent for key lookup
            connection_timeout: Connection timeout in seconds

        Note:
            At least one of private_key_path or use_ssh_agent must be enabled
            for authentication.
        """
        self._host = host
        self._port = port
        self._username = username
        self._private_key_path = private_key_path
        self._full_access_hosts = full_access_hosts or []
        self._use_ssh_agent = use_ssh_agent
        self._connection_timeout = connection_timeout

        # Runtime state
        self._client: Optional[Any] = None  # paramiko.SSHClient
        self._sftp: Optional[Any] = None  # paramiko.SFTPClient
        self._config: Optional[SandboxConfig] = None
        self._access_level: Optional[SandboxAccessLevel] = None
        self._status: SandboxStatus = SandboxStatus.CREATED
        self._workspace: str = ""

        # Validate configuration
        if not private_key_path and not use_ssh_agent:
            raise SandboxInitializationError(
                "Either private_key_path or use_ssh_agent must be enabled"
            )

    def _determine_access_level(self) -> SandboxAccessLevel:
        """
        Determine access level based on host configuration.

        Returns:
            SandboxAccessLevel.FULL if host is in full_access_hosts,
            otherwise SandboxAccessLevel.RESTRICTED
        """
        # Check if host is in full access list
        if self._host in self._full_access_hosts:
            logger.info(f"VM '{self._host}' is in full access hosts list")
            return SandboxAccessLevel.FULL

        # Check by hostname if we connected by IP
        # (Could extend this to check cloud provider tags)

        logger.info(f"VM '{self._host}' will run in RESTRICTED mode")
        return SandboxAccessLevel.RESTRICTED

    def _load_private_key(self) -> Optional[Any]:
        """
        Load the private key for authentication.

        Returns:
            Paramiko key object or None

        Raises:
            SandboxInitializationError: If key loading fails
        """
        import paramiko

        if not self._private_key_path:
            return None

        key_path = Path(self._private_key_path).expanduser()

        if not key_path.exists():
            raise SandboxInitializationError(f"Private key file not found: {key_path}")

        # Try different key types
        key_types = [paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey, paramiko.DSSKey]

        for key_class in key_types:
            try:
                return key_class.from_private_key_file(str(key_path))
            except Exception:
                continue

        raise SandboxInitializationError(
            f"Could not load private key from {key_path}. Supported types: Ed25519, RSA, ECDSA, DSS"
        )

    async def initialize(self, config: SandboxConfig) -> None:
        """
        Initialize SSH connection to the VM.

        This method:
            1. Establishes SSH connection using PKI
            2. Determines access level based on host
            3. Creates isolated workspace directory
            4. Initializes ephemeral SQLite database

        Args:
            config: Sandbox configuration

        Raises:
            SandboxInitializationError: If connection fails
            SandboxConnectionError: If SSH connection fails
        """
        import paramiko

        self._status = SandboxStatus.INITIALIZING
        self._config = config

        try:
            # Determine access level first
            self._access_level = self._determine_access_level()
            config.access_level = self._access_level

            # Create SSH client
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Load private key
            pkey = self._load_private_key()

            # Connection parameters
            connect_kwargs = {
                "hostname": self._host,
                "port": self._port,
                "username": self._username,
                "timeout": self._connection_timeout,
                "allow_agent": self._use_ssh_agent,
                "look_for_keys": True,
            }

            if pkey:
                connect_kwargs["pkey"] = pkey

            logger.info(f"Connecting to VM sandbox: {self._username}@{self._host}:{self._port}")

            # Connect (run in executor to avoid blocking)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.connect(**connect_kwargs)
            )

            # Setup SFTP
            self._sftp = await asyncio.get_event_loop().run_in_executor(
                None, self._client.open_sftp
            )

            logger.info(f"Connected to VM sandbox (access: {self._access_level.value})")

            # Setup workspace
            await self._setup_workspace()

            # Initialize ephemeral SQLite
            await self._init_ephemeral_db()

            # Setup volume mounts if configured
            await self._setup_volume_mounts()

            self._status = SandboxStatus.READY
            logger.info(f"VM sandbox initialized: {self._workspace}")

        except paramiko.AuthenticationException as e:
            self._status = SandboxStatus.ERROR
            raise SandboxConnectionError(
                f"SSH authentication failed: {e}",
                host=self._host,
                port=self._port,
                connection_type="ssh",
            )
        except paramiko.SSHException as e:
            self._status = SandboxStatus.ERROR
            raise SandboxConnectionError(
                f"SSH connection error: {e}",
                host=self._host,
                port=self._port,
                connection_type="ssh",
            )
        except Exception as e:
            self._status = SandboxStatus.ERROR
            await self._cleanup_partial()
            raise SandboxInitializationError(f"Failed to initialize VM sandbox: {e}")

    async def _setup_workspace(self) -> None:
        """Create isolated workspace directory on the VM."""
        workspace_id = self._config.sandbox_id[:8]
        self._workspace = f"/home/{self._username}/workspace_{workspace_id}"

        setup_commands = f"""
mkdir -p {self._workspace}
mkdir -p {self._workspace}/output
mkdir -p {self._workspace}/tmp
mkdir -p {self._workspace}/share
chmod 700 {self._workspace}
cd {self._workspace}
"""

        result = await self.execute_shell(setup_commands)
        if not result.success:
            raise SandboxInitializationError(f"Failed to setup workspace: {result.stderr}")

        # Update config with workspace paths
        self._config.working_directory = self._workspace
        self._config.output_mount_container = f"{self._workspace}/output"
        self._config.share_mount_container = f"{self._workspace}/share"

        logger.debug(f"VM workspace created: {self._workspace}")

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
        db_path = f"{self._workspace}/tmp/agent_task.db"
        self._config.ephemeral_db_path = db_path

        result = await self.execute_shell(f"sqlite3 {db_path} << 'EOF'\n{init_sql}\nEOF")

        if result.success:
            logger.debug(f"Initialized ephemeral SQLite at {db_path}")
        else:
            logger.warning(f"Failed to init ephemeral DB: {result.stderr}")

    async def _setup_volume_mounts(self) -> None:
        """
        Setup volume mount directories.

        Note: For VMs, we create local directories that can be synced
        via SFTP. True volume mounting would require NFS or similar.
        """
        # Ensure output and share directories exist
        await self.execute_shell(f"""
mkdir -p {self._workspace}/output
mkdir -p {self._workspace}/share
""")

    async def _cleanup_partial(self) -> None:
        """Clean up partial resources after failed initialization."""
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None

        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _ensure_initialized(self) -> None:
        """
        Ensure sandbox is initialized before operations.

        Raises:
            SandboxNotInitializedError: If not initialized
        """
        if not self._client or self._status not in (SandboxStatus.READY, SandboxStatus.EXECUTING):
            raise SandboxNotInitializedError(
                "Sandbox not initialized or not ready",
                sandbox_id=self._config.sandbox_id if self._config else None,
            )

    async def execute_shell(
        self, command: str, timeout: Optional[int] = None, working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a shell command on the VM.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            working_dir: Working directory

        Returns:
            ExecutionResult with command output
        """
        # For workspace setup or cleanup, we may be called outside normal ready state
        # INITIALIZING: workspace creation during init
        # CLEANING_UP: workspace removal during cleanup
        if self._status in (SandboxStatus.INITIALIZING, SandboxStatus.CLEANING_UP):
            pass  # Allow during init and cleanup
        else:
            self._ensure_initialized()

        effective_timeout = timeout
        if effective_timeout is None and self._config:
            effective_timeout = self._config.timeout_seconds
        effective_timeout = effective_timeout or 600

        effective_workdir = working_dir or (self._workspace if self._workspace else None)

        start_time = time.time()

        if self._status == SandboxStatus.READY:
            self._status = SandboxStatus.EXECUTING

        try:
            # Wrap command to run in workspace
            if effective_workdir:
                full_command = f"cd {effective_workdir} && {command}"
            else:
                full_command = command

            # Execute via SSH
            stdin, stdout, stderr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.exec_command(full_command, timeout=effective_timeout)
                ),
                timeout=effective_timeout + 5,  # Extra buffer for network latency
            )

            # Read outputs
            stdout_str = await asyncio.get_event_loop().run_in_executor(
                None, lambda: stdout.read().decode("utf-8", errors="replace")
            )

            stderr_str = await asyncio.get_event_loop().run_in_executor(
                None, lambda: stderr.read().decode("utf-8", errors="replace")
            )

            exit_code = stdout.channel.recv_exit_status()

            execution_time = time.time() - start_time

            # Truncate if needed
            truncated = False
            if len(stdout_str) > MAX_OUTPUT_SIZE:
                stdout_str = stdout_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                truncated = True
            if len(stderr_str) > MAX_OUTPUT_SIZE:
                stderr_str = stderr_str[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                truncated = True

            if self._status == SandboxStatus.EXECUTING:
                self._status = SandboxStatus.READY

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time_seconds=execution_time,
                truncated=truncated,
                timed_out=False,
            )

        except asyncio.TimeoutError:
            if self._status == SandboxStatus.EXECUTING:
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
            if self._status == SandboxStatus.EXECUTING:
                self._status = SandboxStatus.READY

            logger.error(f"VM execution failed: {e}")

            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time_seconds=time.time() - start_time,
            )

    async def execute_python(
        self, code: str, timeout: Optional[int] = None, working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute Python code on the VM.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            working_dir: Working directory

        Returns:
            ExecutionResult with execution output
        """
        self._ensure_initialized()

        # Write code to temp file
        temp_file = f"{self._workspace}/tmp/agent_code.py"
        write_success = await self.write_file(temp_file, code)

        if not write_success:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="Failed to write Python code to temp file"
            )

        return await self.execute_shell(
            f"python3 {temp_file}", timeout=timeout, working_dir=working_dir
        )

    async def write_file(self, path: str, content: str, mode: str = "w") -> bool:
        """
        Write content to a file on the VM.

        Uses SFTP for reliable file transfer.

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
            path = f"{self._workspace}/{path}"

        try:
            # Ensure parent directory exists
            parent_dir = str(Path(path).parent)
            await self.execute_shell(f"mkdir -p '{parent_dir}'")

            # Write via SFTP
            sftp_mode = "w" if mode == "w" else "a"

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._write_file_sftp(path, content, sftp_mode)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return False

    def _write_file_sftp(self, path: str, content: str, mode: str) -> None:
        """Synchronous SFTP write helper."""
        with self._sftp.open(path, mode) as f:
            f.write(content)

    async def read_file(self, path: str) -> Optional[str]:
        """
        Read content from a file on the VM.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        try:
            content = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._read_file_sftp(path)
            )
            return content
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.debug(f"Failed to read file {path}: {e}")
            return None

    def _read_file_sftp(self, path: str) -> str:
        """Synchronous SFTP read helper."""
        with self._sftp.open(path, "r") as f:
            return f.read().decode("utf-8", errors="replace")

    async def write_file_binary(self, path: str, content: bytes) -> bool:
        """
        Write binary content to a file on the VM.

        Args:
            path: File path
            content: Binary content

        Returns:
            True if successful
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        try:
            parent_dir = str(Path(path).parent)
            await self.execute_shell(f"mkdir -p '{parent_dir}'")

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._write_file_binary_sftp(path, content)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to write binary file {path}: {e}")
            return False

    def _write_file_binary_sftp(self, path: str, content: bytes) -> None:
        """Synchronous SFTP binary write helper."""
        with self._sftp.open(path, "wb") as f:
            f.write(content)

    async def read_file_binary(self, path: str) -> Optional[bytes]:
        """
        Read binary content from a file on the VM.

        Args:
            path: File path

        Returns:
            Binary content or None if not found
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        try:
            content = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._read_file_binary_sftp(path)
            )
            return content
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.debug(f"Failed to read binary file {path}: {e}")
            return None

    def _read_file_binary_sftp(self, path: str) -> bytes:
        """Synchronous SFTP binary read helper."""
        with self._sftp.open(path, "rb") as f:
            return f.read()

    async def list_files(self, path: str = ".", recursive: bool = False) -> List[FileInfo]:
        """
        List files in a directory on the VM.

        Args:
            path: Directory path
            recursive: If True, list recursively

        Returns:
            List of FileInfo objects
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        try:
            if recursive:
                result = await self.execute_shell(
                    f"find '{path}' -maxdepth 10 -type f -o -type d 2>/dev/null | head -1000"
                )

                files = []
                for line in result.stdout.strip().split("\n"):
                    item_path = line.strip()
                    if item_path:
                        name = Path(item_path).name
                        files.append(FileInfo(path=item_path, name=name, is_directory=False))
                return files
            else:
                # Use SFTP for more accurate listing
                items = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._sftp.listdir_attr(path)
                )

                import stat

                files = []
                for item in items:
                    files.append(
                        FileInfo(
                            path=f"{path}/{item.filename}",
                            name=item.filename,
                            is_directory=stat.S_ISDIR(item.st_mode),
                            size_bytes=item.st_size,
                            modified_at=datetime.fromtimestamp(item.st_mtime)
                            if item.st_mtime
                            else None,
                        )
                    )

                return files

        except Exception as e:
            logger.debug(f"Failed to list files in {path}: {e}")
            return []

    async def file_exists(self, path: str) -> bool:
        """
        Check if a file exists on the VM.

        Args:
            path: File path

        Returns:
            True if exists
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self._sftp.stat(path))
            return True
        except FileNotFoundError:
            return False
        except Exception:
            # Fall back to shell check
            result = await self.execute_shell(f"test -e '{path}' && echo 'yes' || echo 'no'")
            return result.stdout.strip() == "yes"

    async def delete_file(self, path: str) -> bool:
        """
        Delete a file on the VM.

        Args:
            path: File path

        Returns:
            True if deleted successfully
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self._sftp.remove(path))
            return True
        except FileNotFoundError:
            return True  # Already doesn't exist
        except Exception:
            # Try shell fallback
            result = await self.execute_shell(f"rm -f '{path}'")
            return result.success

    async def create_directory(self, path: str) -> bool:
        """
        Create a directory on the VM.

        Args:
            path: Directory path

        Returns:
            True if created successfully
        """
        self._ensure_initialized()

        if not path.startswith("/"):
            path = f"{self._workspace}/{path}"

        result = await self.execute_shell(f"mkdir -p '{path}'")
        return result.success

    async def cleanup(self) -> None:
        """
        Clean up VM sandbox resources.

        This removes the workspace directory and closes connections.
        """
        self._status = SandboxStatus.CLEANING_UP
        errors = []

        try:
            if self._client and self._workspace:
                logger.info(f"Cleaning up VM sandbox workspace: {self._workspace}")

                # Remove workspace directory
                result = await self.execute_shell(f"rm -rf {self._workspace}")
                if not result.success:
                    errors.append(f"workspace removal: {result.stderr}")
        except Exception as e:
            errors.append(f"workspace cleanup: {e}")

        # Close SFTP
        if self._sftp:
            try:
                self._sftp.close()
            except Exception as e:
                errors.append(f"SFTP close: {e}")
            self._sftp = None

        # Close SSH client
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                errors.append(f"SSH close: {e}")
            self._client = None

        self._status = SandboxStatus.TERMINATED
        logger.debug("VM sandbox cleaned up")

        if errors:
            raise SandboxCleanupError(
                f"Cleanup completed with errors: {'; '.join(errors)}",
                resources_leaked=errors,
                partial_cleanup=True,
                sandbox_id=self._config.sandbox_id if self._config else None,
            )

    async def is_healthy(self) -> bool:
        """
        Check if SSH connection is alive.

        Returns:
            True if connection is active
        """
        if not self._client:
            return False

        try:
            transport = self._client.get_transport()
            if transport is None or not transport.is_active():
                return False

            # Try a simple command to verify
            result = await self.execute_shell("echo 'health_check'", timeout=10)
            return result.success and "health_check" in result.stdout

        except Exception:
            return False

    def get_access_level(self) -> SandboxAccessLevel:
        """Get the access level of this sandbox."""
        return self._access_level or SandboxAccessLevel.RESTRICTED

    def get_status(self) -> SandboxStatus:
        """Get the current status of the sandbox."""
        return self._status

    def get_config(self) -> Optional[SandboxConfig]:
        """Get the configuration used for this sandbox."""
        return self._config

    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the VM sandbox.

        Returns:
            Dictionary with sandbox details
        """
        info = {
            "provider": "vm",
            "host": self._host,
            "port": self._port,
            "username": self._username,
            "status": self._status.value,
            "access_level": self._access_level.value if self._access_level else None,
            "workspace": self._workspace,
        }

        if self._config:
            info.update(
                {
                    "sandbox_id": self._config.sandbox_id,
                    "network_enabled": self._config.network_enabled,
                    "timeout_seconds": self._config.timeout_seconds,
                }
            )

        return info
