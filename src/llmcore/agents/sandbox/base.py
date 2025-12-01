# src/llmcore/agents/sandbox/base.py
"""
Abstract base class and core data models for sandbox providers.

This module defines the contract that all sandbox providers must implement,
ensuring consistent behavior across Docker, VM, and any future sandbox types.

CRITICAL SECURITY INVARIANT:
    All implementations MUST ensure that code NEVER executes on the host system.
    This is a non-negotiable security requirement.

Classes:
    SandboxAccessLevel: Enum defining access levels (RESTRICTED, FULL)
    SandboxConfig: Configuration dataclass for sandbox instances
    ExecutionResult: Result of command/code execution
    SandboxProvider: Abstract base class for all providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid


class SandboxAccessLevel(Enum):
    """
    Access level determines what operations the sandbox can perform.

    RESTRICTED:
        - Tool whitelist is enforced
        - Network is disabled by default
        - Package installation is blocked
        - Used for untrusted or unverified sandbox configurations

    FULL:
        - All tools are available
        - Network access is enabled
        - Package installation is allowed
        - Only granted to whitelisted images/hosts with proper labels/tags

    The access level is determined at sandbox creation time based on:
        - Docker: Image labels and name patterns
        - VM: Host being in full_access_hosts list or having proper tags
    """
    RESTRICTED = "restricted"
    FULL = "full"


class SandboxStatus(Enum):
    """
    Current status of a sandbox instance.

    Used for lifecycle management and health monitoring.
    """
    CREATED = "created"          # Config created, not yet initialized
    INITIALIZING = "initializing"  # Being set up
    READY = "ready"              # Ready to accept commands
    EXECUTING = "executing"      # Currently running a command
    PAUSED = "paused"           # Temporarily suspended
    ERROR = "error"             # In error state
    CLEANING_UP = "cleaning_up"  # Being torn down
    TERMINATED = "terminated"    # Fully cleaned up


@dataclass
class SandboxConfig:
    """
    Configuration for a sandbox instance.

    This dataclass contains all settings needed to create and manage
    a sandbox. It is provider-agnostic; specific providers may use
    only a subset of these settings.

    Attributes:
        sandbox_id: Unique identifier for this sandbox instance
        access_level: RESTRICTED or FULL access
        timeout_seconds: Default timeout for operations
        memory_limit: Memory limit (e.g., "1g", "512m")
        cpu_limit: CPU limit as fraction of cores (e.g., 2.0 = 2 cores)
        network_enabled: Whether network access is allowed
        environment_vars: Environment variables to set in sandbox
        share_mount_host: Host path for shared persistent data
        share_mount_container: Container path for shared mount
        output_mount_host: Host path for output files
        output_mount_container: Container path for output mount
        ephemeral_db_path: Path for ephemeral SQLite database
        working_directory: Default working directory in sandbox
        labels: Additional labels/tags for the sandbox

    Example:
        >>> config = SandboxConfig(
        ...     memory_limit="2g",
        ...     timeout_seconds=300,
        ...     network_enabled=True
        ... )
    """
    # Identity
    sandbox_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Security
    access_level: SandboxAccessLevel = SandboxAccessLevel.RESTRICTED

    # Resource limits
    timeout_seconds: int = 600
    memory_limit: str = "1g"
    cpu_limit: float = 2.0

    # Network
    network_enabled: bool = False

    # Environment
    environment_vars: Dict[str, str] = field(default_factory=dict)

    # Volume mounts - will be set by SandboxRegistry based on global config
    share_mount_host: Optional[Path] = None
    share_mount_container: str = "/workspace/share"
    output_mount_host: Optional[Path] = None
    output_mount_container: str = "/workspace/output"

    # Ephemeral resources
    ephemeral_db_path: str = "/tmp/agent_task.db"

    # Working directory
    working_directory: str = "/workspace"

    # Additional metadata
    labels: Dict[str, str] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths are Path objects if provided as strings
        if isinstance(self.share_mount_host, str):
            self.share_mount_host = Path(self.share_mount_host).expanduser()
        if isinstance(self.output_mount_host, str):
            self.output_mount_host = Path(self.output_mount_host).expanduser()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "access_level": self.access_level.value,
            "timeout_seconds": self.timeout_seconds,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "network_enabled": self.network_enabled,
            "environment_vars": self.environment_vars,
            "share_mount_host": str(self.share_mount_host) if self.share_mount_host else None,
            "share_mount_container": self.share_mount_container,
            "output_mount_host": str(self.output_mount_host) if self.output_mount_host else None,
            "output_mount_container": self.output_mount_container,
            "ephemeral_db_path": self.ephemeral_db_path,
            "working_directory": self.working_directory,
            "labels": self.labels,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ExecutionResult:
    """
    Result of executing a command or code in a sandbox.

    This dataclass captures all output from an execution, including
    exit codes, stdout/stderr, timing information, and truncation status.

    Attributes:
        exit_code: Process exit code (0 = success)
        stdout: Standard output captured from the execution
        stderr: Standard error captured from the execution
        execution_time_seconds: Wall-clock time for execution
        truncated: True if output was truncated due to size limits
        timed_out: True if execution was terminated due to timeout
        metadata: Additional execution metadata

    Example:
        >>> result = ExecutionResult(exit_code=0, stdout="Hello, World!")
        >>> print(result.success)
        True
        >>> print(result.to_tool_output())
        Hello, World!
    """
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    execution_time_seconds: float = 0.0
    truncated: bool = False
    timed_out: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """
        Check if execution was successful.

        Returns:
            True if exit_code is 0 and execution didn't time out
        """
        return self.exit_code == 0 and not self.timed_out

    def to_tool_output(self) -> str:
        """
        Format result for agent tool consumption.

        This produces a human/agent-readable string that can be
        included in the agent's observation.

        Returns:
            Formatted string with execution results
        """
        parts = []

        if self.timed_out:
            parts.append("⏰ EXECUTION TIMED OUT")

        if self.success:
            if self.stdout.strip():
                parts.append(self.stdout)
            else:
                parts.append("(command completed successfully with no output)")
        else:
            parts.append(f"❌ EXIT CODE: {self.exit_code}")
            if self.stdout.strip():
                parts.append(f"STDOUT:\n{self.stdout}")
            if self.stderr.strip():
                parts.append(f"STDERR:\n{self.stderr}")

        if self.truncated:
            parts.append("\n⚠️ (output was truncated due to size limits)")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time_seconds": self.execution_time_seconds,
            "truncated": self.truncated,
            "timed_out": self.timed_out,
            "success": self.success,
            "metadata": self.metadata
        }


@dataclass
class FileInfo:
    """
    Information about a file in the sandbox.

    Used for file listing and inspection operations.
    """
    path: str
    name: str
    is_directory: bool
    size_bytes: int = 0
    modified_at: Optional[datetime] = None
    permissions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "is_directory": self.is_directory,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "permissions": self.permissions
        }


class SandboxProvider(ABC):
    """
    Abstract base class for sandbox execution environments.

    All sandbox providers must implement this interface to ensure
    consistent behavior across different execution backends (Docker, VM, etc.).

    CRITICAL SECURITY INVARIANT:
        All implementations MUST ensure that code NEVER executes on the
        host system. This is a non-negotiable security requirement.
        Violations of this invariant are considered critical security bugs.

    Lifecycle:
        1. Create provider instance with backend-specific configuration
        2. Call initialize() with SandboxConfig to set up the sandbox
        3. Use execute_shell(), execute_python(), file operations as needed
        4. Call cleanup() to tear down the sandbox

    Thread Safety:
        Implementations should be safe to use from a single async context.
        Concurrent operations within the same sandbox are not guaranteed
        to be safe and should be serialized by the caller.

    Example:
        >>> provider = DockerSandboxProvider(image="python:3.11-slim", ...)
        >>> config = SandboxConfig()
        >>> await provider.initialize(config)
        >>> result = await provider.execute_shell("echo 'Hello, World!'")
        >>> print(result.stdout)
        Hello, World!
        >>> await provider.cleanup()
    """

    @abstractmethod
    async def initialize(self, config: SandboxConfig) -> None:
        """
        Initialize the sandbox environment.

        This method sets up the sandbox execution environment, which may
        involve starting a container, connecting to a VM, or other
        backend-specific setup.

        Args:
            config: Configuration for the sandbox instance

        Raises:
            SandboxInitializationError: If setup fails
            SandboxAccessDenied: If security checks fail

        Post-conditions:
            - Sandbox is ready to accept execute_* and file operations
            - Volume mounts are configured
            - Ephemeral resources are initialized
        """
        pass

    @abstractmethod
    async def execute_shell(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute (passed to bash -c)
            timeout: Timeout in seconds (None = use config default)
            working_dir: Working directory (None = use sandbox default)

        Returns:
            ExecutionResult with stdout, stderr, exit code, timing

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
            SandboxExecutionError: If execution setup fails
            SandboxTimeoutError: If command exceeds timeout

        Note:
            Commands that return non-zero exit codes do NOT raise exceptions.
            The exit code is captured in ExecutionResult for the agent to handle.
        """
        pass

    @abstractmethod
    async def execute_python(
        self,
        code: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds (None = use config default)
            working_dir: Working directory (None = use sandbox default)

        Returns:
            ExecutionResult with stdout, stderr, exit code, timing

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
            SandboxExecutionError: If execution setup fails
            SandboxTimeoutError: If code exceeds timeout

        Note:
            Syntax errors and runtime exceptions are captured in stderr
            with appropriate exit codes, not raised as Python exceptions.
        """
        pass

    @abstractmethod
    async def write_file(
        self,
        path: str,
        content: str,
        mode: str = "w"
    ) -> bool:
        """
        Write content to a file in the sandbox.

        Args:
            path: File path (relative to workspace or absolute)
            content: Content to write
            mode: Write mode ("w" for write, "a" for append)

        Returns:
            True if write was successful

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
            SandboxExecutionError: If write fails

        Note:
            Paths outside the sandbox filesystem will fail.
        """
        pass

    @abstractmethod
    async def read_file(self, path: str) -> Optional[str]:
        """
        Read content from a file in the sandbox.

        Args:
            path: File path (relative to workspace or absolute)

        Returns:
            File content as string, or None if file doesn't exist

        Raises:
            SandboxNotInitializedError: If sandbox not initialized

        Note:
            Binary files may not be read correctly; this method
            assumes text content with UTF-8 encoding.
        """
        pass

    @abstractmethod
    async def write_file_binary(self, path: str, content: bytes) -> bool:
        """
        Write binary content to a file in the sandbox.

        Args:
            path: File path (relative to workspace or absolute)
            content: Binary content to write

        Returns:
            True if write was successful

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
            SandboxExecutionError: If write fails
        """
        pass

    @abstractmethod
    async def read_file_binary(self, path: str) -> Optional[bytes]:
        """
        Read binary content from a file in the sandbox.

        Args:
            path: File path (relative to workspace or absolute)

        Returns:
            File content as bytes, or None if file doesn't exist

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
        """
        pass

    @abstractmethod
    async def list_files(
        self,
        path: str = ".",
        recursive: bool = False
    ) -> List[FileInfo]:
        """
        List files in a directory within the sandbox.

        Args:
            path: Directory path (relative to workspace or absolute)
            recursive: If True, list files recursively

        Returns:
            List of FileInfo objects for files in the directory

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
        """
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """
        Check if a file or directory exists in the sandbox.

        Args:
            path: Path to check (relative to workspace or absolute)

        Returns:
            True if the path exists

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
        """
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """
        Delete a file in the sandbox.

        Args:
            path: File path to delete

        Returns:
            True if deletion was successful, False if file didn't exist

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
            SandboxExecutionError: If deletion fails (e.g., permission denied)
        """
        pass

    @abstractmethod
    async def create_directory(self, path: str) -> bool:
        """
        Create a directory in the sandbox.

        Args:
            path: Directory path to create (creates parents if needed)

        Returns:
            True if creation was successful

        Raises:
            SandboxNotInitializedError: If sandbox not initialized
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up the sandbox environment.

        This method tears down the sandbox, releasing all resources.
        After calling cleanup(), the sandbox cannot be used again.

        Actions performed:
            - Stop and remove containers (Docker)
            - Close SSH connections (VM)
            - Destroy ephemeral resources
            - Preserve output files (copy to output mount)

        Raises:
            SandboxCleanupError: If cleanup fails (non-fatal, resources may leak)

        Post-conditions:
            - All sandbox resources are released (best effort)
            - Output files are preserved
            - Sandbox is no longer usable
        """
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """
        Check if the sandbox is still responsive and healthy.

        Returns:
            True if sandbox is operational, False otherwise
        """
        pass

    @abstractmethod
    def get_access_level(self) -> SandboxAccessLevel:
        """
        Get the access level of this sandbox.

        Returns:
            SandboxAccessLevel.RESTRICTED or SandboxAccessLevel.FULL
        """
        pass

    @abstractmethod
    def get_status(self) -> SandboxStatus:
        """
        Get the current status of the sandbox.

        Returns:
            Current SandboxStatus
        """
        pass

    @abstractmethod
    def get_config(self) -> Optional[SandboxConfig]:
        """
        Get the configuration used to initialize this sandbox.

        Returns:
            SandboxConfig or None if not initialized
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the sandbox.

        Returns:
            Dictionary with sandbox details (provider-specific)
        """
        pass
