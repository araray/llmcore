# src/llmcore/agents/sandbox/registry.py
"""
Sandbox Registry: Routes execution requests to appropriate sandbox providers
and enforces access control policies.

The registry is the central management point for all sandbox operations:
    - Creates sandboxes using the configured provider (Docker, VM, or hybrid)
    - Enforces tool access policies based on sandbox access level
    - Manages sandbox lifecycle and tracking
    - Handles fallback between providers

Usage:
    >>> config = SandboxRegistryConfig(mode=SandboxMode.DOCKER)
    >>> registry = SandboxRegistry(config)
    >>> sandbox = await registry.create_sandbox(SandboxConfig())
    >>> result = await sandbox.execute_shell("echo 'Hello'")
    >>> await registry.cleanup_sandbox(sandbox.get_config().sandbox_id)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import SandboxAccessLevel, SandboxConfig, SandboxProvider, SandboxStatus
from .docker_provider import DockerSandboxProvider
from .exceptions import SandboxInitializationError
from .vm_provider import VMSandboxProvider

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """
    Sandbox mode preference.

    DOCKER: Use Docker containers for sandboxing
    VM: Use remote VMs via SSH for sandboxing
    HYBRID: Try Docker first, fall back to VM if unavailable
    """

    DOCKER = "docker"
    VM = "vm"
    HYBRID = "hybrid"


@dataclass
class SandboxRegistryConfig:
    """
    Configuration for the sandbox registry.

    This dataclass contains all settings for sandbox creation and management,
    including provider-specific configurations and access control rules.

    Attributes:
        mode: Primary sandbox mode (docker, vm, hybrid)
        fallback_enabled: Whether to fall back to secondary provider

        # Docker settings
        docker_enabled: Whether Docker provider is available
        docker_image: Default Docker image to use
        docker_image_whitelist: List of allowed image patterns
        docker_full_access_label: Label that grants full access
        docker_full_access_name_pattern: Name pattern for full access
        docker_host: Optional remote Docker host URL
        docker_auto_pull: Whether to auto-pull missing images

        # VM settings
        vm_enabled: Whether VM provider is available
        vm_host: VM hostname or IP
        vm_port: SSH port
        vm_username: SSH username
        vm_private_key_path: Path to private key
        vm_full_access_hosts: List of hosts with full access
        vm_use_ssh_agent: Whether to use SSH agent

        # Volume settings
        share_path: Host path for shared persistent data
        outputs_path: Host path for agent output files

        # Tool access control
        allowed_tools: Tools allowed in restricted sandboxes
        denied_tools: Tools denied in restricted sandboxes
    """

    # Mode selection
    mode: SandboxMode = SandboxMode.DOCKER
    fallback_enabled: bool = True

    # Docker configuration
    docker_enabled: bool = True
    docker_image: str = "python:3.11-slim"
    docker_image_whitelist: list[str] = field(
        default_factory=lambda: ["python:3.*-slim", "python:3.*-bookworm", "llmcore-sandbox:*"]
    )
    docker_full_access_label: str = "llmcore.sandbox.full_access=true"
    docker_full_access_name_pattern: str | None = "*-full-access"
    docker_host: str | None = None
    docker_auto_pull: bool = True
    docker_memory_limit: str = "1g"
    docker_cpu_limit: float = 2.0
    docker_timeout_seconds: int = 600

    # VM configuration
    vm_enabled: bool = False
    vm_host: str | None = None
    vm_port: int = 22
    vm_username: str = "agent"
    vm_private_key_path: str | None = None
    vm_full_access_hosts: list[str] = field(default_factory=list)
    vm_use_ssh_agent: bool = True
    vm_connection_timeout: int = 30

    # Volume configuration
    share_path: str = "~/.llmcore/agent_share"
    outputs_path: str = "~/.llmcore/agent_outputs"

    # Tool access control (for RESTRICTED sandboxes)
    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "execute_shell",
            "execute_python",
            "save_file",
            "load_file",
            "replace_in_file",
            "list_files",
            "delete_file",
            "create_directory",
            "file_exists",
            "get_sandbox_info",
            "calculator",
            "semantic_search",
            "episodic_search",
            "finish",
            "human_approval",
        ]
    )
    denied_tools: list[str] = field(
        default_factory=lambda: [
            "install_system_package",
            "sudo_execute",
            "network_request",
            "raw_socket",
        ]
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode == SandboxMode.DOCKER and not self.docker_enabled:
            raise ValueError("Mode is DOCKER but docker_enabled is False")
        if self.mode == SandboxMode.VM and not self.vm_enabled:
            raise ValueError("Mode is VM but vm_enabled is False")
        if self.mode == SandboxMode.VM and not self.vm_host:
            raise ValueError("Mode is VM but vm_host is not set")


class SandboxRegistry:
    """
    Central registry for managing sandbox providers.

    The registry is responsible for:
        - Creating sandboxes using the appropriate provider
        - Enforcing tool access policies
        - Tracking active sandboxes
        - Managing sandbox lifecycle (creation, tracking, cleanup)
        - Handling provider fallback in hybrid mode

    Thread Safety:
        The registry is designed to be used from a single async context.
        For multi-threaded use, external synchronization is required.

    Example:
        >>> config = SandboxRegistryConfig(mode=SandboxMode.DOCKER)
        >>> registry = SandboxRegistry(config)
        >>>
        >>> sandbox_config = SandboxConfig()
        >>> sandbox = await registry.create_sandbox(sandbox_config)
        >>>
        >>> if registry.is_tool_allowed("execute_shell", sandbox.get_access_level()):
        ...     result = await sandbox.execute_shell("echo 'Hello'")
        >>>
        >>> await registry.cleanup_sandbox(sandbox_config.sandbox_id)
    """

    def __init__(self, config: SandboxRegistryConfig):
        """
        Initialize the sandbox registry.

        Args:
            config: Registry configuration
        """
        self._config = config
        self._active_sandboxes: dict[str, SandboxProvider] = {}

        # Convert tool lists to sets for faster lookup
        self._allowed_tools: set[str] = set(config.allowed_tools)
        self._denied_tools: set[str] = set(config.denied_tools)

        logger.info(
            f"SandboxRegistry initialized: mode={config.mode.value}, "
            f"docker={config.docker_enabled}, vm={config.vm_enabled}"
        )

    async def create_sandbox(
        self,
        sandbox_config: SandboxConfig,
        prefer_mode: SandboxMode | None = None,
        docker_image: str | None = None,
    ) -> SandboxProvider:
        """
        Create and initialize a sandbox.

        This method creates a sandbox using the configured provider,
        setting up volume mounts and initializing the environment.

        Args:
            sandbox_config: Configuration for the sandbox instance
            prefer_mode: Optional override for sandbox mode
            docker_image: Optional override for Docker image

        Returns:
            Initialized SandboxProvider ready for use

        Raises:
            SandboxInitializationError: If sandbox creation fails
            SandboxAccessDenied: If security checks fail
        """
        mode = prefer_mode or self._config.mode

        # Setup volume paths
        sandbox_config.share_mount_host = Path(self._config.share_path).expanduser().resolve()
        sandbox_config.share_mount_host.mkdir(parents=True, exist_ok=True)

        sandbox_config.output_mount_host = (
            Path(self._config.outputs_path).expanduser().resolve() / sandbox_config.sandbox_id
        )
        sandbox_config.output_mount_host.mkdir(parents=True, exist_ok=True)

        # Set default resource limits from config
        if sandbox_config.memory_limit == "1g":  # default
            sandbox_config.memory_limit = self._config.docker_memory_limit
        if sandbox_config.cpu_limit == 2.0:  # default
            sandbox_config.cpu_limit = self._config.docker_cpu_limit
        if sandbox_config.timeout_seconds == 600:  # default
            sandbox_config.timeout_seconds = self._config.docker_timeout_seconds

        provider = None
        errors = []

        # Try Docker first if mode is DOCKER or HYBRID
        if mode in (SandboxMode.DOCKER, SandboxMode.HYBRID):
            if self._config.docker_enabled:
                try:
                    provider = await self._create_docker_sandbox(sandbox_config, docker_image)
                except Exception as e:
                    error_msg = f"Docker sandbox creation failed: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

                    if mode == SandboxMode.HYBRID and self._config.fallback_enabled:
                        logger.info("Falling back to VM sandbox")
                    elif mode == SandboxMode.DOCKER:
                        raise SandboxInitializationError(error_msg)

        # Try VM if mode is VM, or if HYBRID and Docker failed
        if provider is None and mode in (SandboxMode.VM, SandboxMode.HYBRID):
            if self._config.vm_enabled:
                try:
                    provider = await self._create_vm_sandbox(sandbox_config)
                except Exception as e:
                    error_msg = f"VM sandbox creation failed: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

        if provider is None:
            error_details = "; ".join(errors) if errors else "No providers available"
            raise SandboxInitializationError(
                f"Failed to create sandbox. Mode: {mode.value}. {error_details}"
            )

        # Track the sandbox
        self._active_sandboxes[sandbox_config.sandbox_id] = provider

        logger.info(
            f"Created sandbox {sandbox_config.sandbox_id[:8]} "
            f"(provider: {provider.get_info().get('provider')}, "
            f"access: {provider.get_access_level().value})"
        )

        return provider

    async def _create_docker_sandbox(
        self, config: SandboxConfig, image: str | None = None
    ) -> DockerSandboxProvider:
        """
        Create a Docker sandbox.

        Args:
            config: Sandbox configuration
            image: Optional image override

        Returns:
            Initialized DockerSandboxProvider
        """
        effective_image = image or self._config.docker_image

        provider = DockerSandboxProvider(
            image=effective_image,
            image_whitelist=self._config.docker_image_whitelist,
            full_access_label=self._config.docker_full_access_label,
            full_access_name_pattern=self._config.docker_full_access_name_pattern,
            docker_host=self._config.docker_host,
            auto_pull=self._config.docker_auto_pull,
        )

        await provider.initialize(config)
        return provider

    async def _create_vm_sandbox(self, config: SandboxConfig) -> VMSandboxProvider:
        """
        Create a VM sandbox.

        Args:
            config: Sandbox configuration

        Returns:
            Initialized VMSandboxProvider
        """
        if not self._config.vm_host:
            raise SandboxInitializationError("VM host not configured")

        provider = VMSandboxProvider(
            host=self._config.vm_host,
            port=self._config.vm_port,
            username=self._config.vm_username,
            private_key_path=self._config.vm_private_key_path,
            full_access_hosts=self._config.vm_full_access_hosts,
            use_ssh_agent=self._config.vm_use_ssh_agent,
            connection_timeout=self._config.vm_connection_timeout,
        )

        await provider.initialize(config)
        return provider

    def is_tool_allowed(self, tool_name: str, access_level: SandboxAccessLevel) -> bool:
        """
        Check if a tool is allowed for the given access level.

        Full access sandboxes bypass all tool restrictions.
        Restricted sandboxes must pass both allow and deny checks.

        Args:
            tool_name: Name of the tool to check
            access_level: Access level of the sandbox

        Returns:
            True if the tool is allowed, False otherwise
        """
        # Full access bypasses all restrictions
        if access_level == SandboxAccessLevel.FULL:
            return True

        # Check deny list first (takes priority)
        if tool_name in self._denied_tools:
            logger.debug(f"Tool '{tool_name}' denied by deny list")
            return False

        # If allow list is specified, tool must be in it
        if self._allowed_tools:
            allowed = tool_name in self._allowed_tools
            if not allowed:
                logger.debug(f"Tool '{tool_name}' not in allow list")
            return allowed

        # Default: allow if not denied
        return True

    def get_allowed_tools(self, access_level: SandboxAccessLevel) -> list[str]:
        """
        Get list of allowed tools for the given access level.

        Args:
            access_level: Access level to check

        Returns:
            List of allowed tool names
        """
        if access_level == SandboxAccessLevel.FULL:
            # Return all tools except explicitly denied ones
            # In full access mode, basically everything is allowed
            return list(self._allowed_tools | {"*"})  # * indicates all allowed

        # For restricted mode, return only explicitly allowed tools
        return list(self._allowed_tools - self._denied_tools)

    async def get_sandbox(self, sandbox_id: str) -> SandboxProvider | None:
        """
        Get an active sandbox by ID.

        Args:
            sandbox_id: Sandbox ID to look up

        Returns:
            SandboxProvider or None if not found
        """
        return self._active_sandboxes.get(sandbox_id)

    async def get_sandbox_status(self, sandbox_id: str) -> SandboxStatus | None:
        """
        Get the status of a sandbox.

        Args:
            sandbox_id: Sandbox ID to check

        Returns:
            SandboxStatus or None if not found
        """
        sandbox = self._active_sandboxes.get(sandbox_id)
        if sandbox:
            return sandbox.get_status()
        return None

    async def cleanup_sandbox(self, sandbox_id: str) -> bool:
        """
        Clean up and remove a sandbox.

        Args:
            sandbox_id: ID of sandbox to clean up

        Returns:
            True if sandbox was found and cleaned up
        """
        provider = self._active_sandboxes.pop(sandbox_id, None)

        if provider:
            try:
                await provider.cleanup()
                logger.info(f"Cleaned up sandbox {sandbox_id[:8]}")
                return True
            except Exception as e:
                logger.warning(f"Error cleaning up sandbox {sandbox_id[:8]}: {e}")
                return True  # Still consider it cleaned up

        return False

    async def cleanup_all(self) -> dict[str, bool]:
        """
        Clean up all active sandboxes.

        Returns:
            Dictionary mapping sandbox_id to cleanup success
        """
        results = {}

        for sandbox_id in list(self._active_sandboxes.keys()):
            results[sandbox_id] = await self.cleanup_sandbox(sandbox_id)

        return results

    def list_active_sandboxes(self) -> list[dict[str, Any]]:
        """
        List all active sandboxes.

        Returns:
            List of sandbox info dictionaries
        """
        return [
            {"sandbox_id": sandbox_id, **provider.get_info()}
            for sandbox_id, provider in self._active_sandboxes.items()
        ]

    def get_active_count(self) -> int:
        """
        Get count of active sandboxes.

        Returns:
            Number of active sandboxes
        """
        return len(self._active_sandboxes)

    async def health_check(self, sandbox_id: str) -> bool:
        """
        Check if a sandbox is healthy.

        Args:
            sandbox_id: Sandbox ID to check

        Returns:
            True if sandbox is healthy
        """
        sandbox = self._active_sandboxes.get(sandbox_id)
        if sandbox:
            return await sandbox.is_healthy()
        return False

    async def health_check_all(self) -> dict[str, bool]:
        """
        Health check all active sandboxes.

        Returns:
            Dictionary mapping sandbox_id to health status
        """
        results = {}

        for sandbox_id, provider in self._active_sandboxes.items():
            try:
                results[sandbox_id] = await provider.is_healthy()
            except Exception:
                results[sandbox_id] = False

        return results

    def get_config(self) -> SandboxRegistryConfig:
        """
        Get the registry configuration.

        Returns:
            SandboxRegistryConfig
        """
        return self._config
