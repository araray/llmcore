# src/llmcore/agents/sandbox/config.py
"""
Configuration management for the sandbox system.

This module handles loading, validation, and management of sandbox
configuration. It integrates with llmcore's configuration system
and supports loading from TOML files.

Configuration Hierarchy:
    1. Default values (defined in this module)
    2. System config file (~/.llmcore/config.toml)
    3. User config file (specified via environment)
    4. Environment variables (LLMCORE_SANDBOX_*)
    5. Runtime overrides (passed to functions)

Example TOML configuration:
    [agents.sandbox]
    mode = "docker"  # docker, vm, hybrid
    fallback_enabled = true

    [agents.sandbox.docker]
    image = "python:3.11-slim"
    image_whitelist = ["python:3.*-slim", "llmcore-sandbox:*"]
    full_access_label = "llmcore.sandbox.full_access=true"
    memory_limit = "1g"
    cpu_limit = 2.0
    timeout_seconds = 600

    [agents.sandbox.vm]
    host = "192.168.1.100"
    port = 22
    username = "agent"
    private_key_path = "~/.ssh/llmcore_agent_key"
    full_access_hosts = ["trusted-vm-1", "trusted-vm-2"]

    [agents.sandbox.volumes]
    share_path = "~/.llmcore/agent_share"
    outputs_path = "~/.llmcore/agent_outputs"

    [agents.sandbox.tools]
    allowed = ["execute_shell", "execute_python", "save_file", ...]
    denied = ["sudo_execute", "network_request", ...]
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "mode": "docker",
    "fallback_enabled": True,
    # Docker defaults
    "docker": {
        "enabled": True,
        "image": "python:3.11-slim",
        "image_whitelist": ["python:3.*-slim", "python:3.*-bookworm", "llmcore-sandbox:*"],
        "full_access_label": "llmcore.sandbox.full_access=true",
        "full_access_name_pattern": "*-full-access",
        "host": None,
        "auto_pull": True,
        "memory_limit": "1g",
        "cpu_limit": 2.0,
        "timeout_seconds": 600,
    },
    # VM defaults
    "vm": {
        "enabled": False,
        "host": None,
        "port": 22,
        "username": "agent",
        "private_key_path": None,
        "full_access_hosts": [],
        "use_ssh_agent": True,
        "connection_timeout": 30,
    },
    # Volume defaults
    "volumes": {"share_path": "~/.llmcore/agent_share", "outputs_path": "~/.llmcore/agent_outputs"},
    # Tool access control defaults
    "tools": {
        "allowed": [
            "execute_shell",
            "execute_python",
            "save_file",
            "load_file",
            "replace_in_file",
            "append_to_file",
            "list_files",
            "file_exists",
            "delete_file",
            "create_directory",
            "get_state",
            "set_state",
            "list_state",
            "get_sandbox_info",
            "get_recorded_files",
            "calculator",
            "semantic_search",
            "episodic_search",
            "finish",
            "human_approval",
        ],
        "denied": ["install_system_package", "sudo_execute", "network_request", "raw_socket"],
    },
    # Output tracking defaults
    "output_tracking": {
        "enabled": True,
        "max_log_entries": 10000,
        "log_input_preview_length": 200,
        "log_output_preview_length": 500,
        "cleanup_max_age_days": 30,
        "cleanup_keep_min_runs": 10,
    },
}


@dataclass
class DockerConfig:
    """Docker sandbox configuration."""

    enabled: bool = True
    image: str = "python:3.11-slim"
    image_whitelist: list[str] = field(
        default_factory=lambda: ["python:3.*-slim", "python:3.*-bookworm", "llmcore-sandbox:*"]
    )
    full_access_label: str = "llmcore.sandbox.full_access=true"
    full_access_name_pattern: str | None = "*-full-access"
    host: str | None = None
    auto_pull: bool = True
    memory_limit: str = "1g"
    cpu_limit: float = 2.0
    timeout_seconds: int = 600


@dataclass
class VMConfig:
    """VM sandbox configuration."""

    enabled: bool = False
    host: str | None = None
    port: int = 22
    username: str = "agent"
    private_key_path: str | None = None
    full_access_hosts: list[str] = field(default_factory=list)
    use_ssh_agent: bool = True
    connection_timeout: int = 30


@dataclass
class VolumeConfig:
    """Volume mount configuration."""

    share_path: str = "~/.llmcore/agent_share"
    outputs_path: str = "~/.llmcore/agent_outputs"


@dataclass
class ToolsConfig:
    """Tool access control configuration."""

    allowed: list[str] = field(default_factory=lambda: DEFAULT_CONFIG["tools"]["allowed"])
    denied: list[str] = field(default_factory=lambda: DEFAULT_CONFIG["tools"]["denied"])


@dataclass
class OutputTrackingConfig:
    """Output tracking configuration."""

    enabled: bool = True
    max_log_entries: int = 10000
    log_input_preview_length: int = 200
    log_output_preview_length: int = 500
    cleanup_max_age_days: int = 30
    cleanup_keep_min_runs: int = 10


@dataclass
class SandboxSystemConfig:
    """
    Complete sandbox system configuration.

    This class holds all configuration for the sandbox system,
    including Docker, VM, volume, and tool settings.
    """

    mode: str = "docker"
    fallback_enabled: bool = True
    docker: DockerConfig = field(default_factory=DockerConfig)
    vm: VMConfig = field(default_factory=VMConfig)
    volumes: VolumeConfig = field(default_factory=VolumeConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    output_tracking: OutputTrackingConfig = field(default_factory=OutputTrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mode": self.mode,
            "fallback_enabled": self.fallback_enabled,
            "docker": {
                "enabled": self.docker.enabled,
                "image": self.docker.image,
                "image_whitelist": self.docker.image_whitelist,
                "full_access_label": self.docker.full_access_label,
                "full_access_name_pattern": self.docker.full_access_name_pattern,
                "host": self.docker.host,
                "auto_pull": self.docker.auto_pull,
                "memory_limit": self.docker.memory_limit,
                "cpu_limit": self.docker.cpu_limit,
                "timeout_seconds": self.docker.timeout_seconds,
            },
            "vm": {
                "enabled": self.vm.enabled,
                "host": self.vm.host,
                "port": self.vm.port,
                "username": self.vm.username,
                "private_key_path": self.vm.private_key_path,
                "full_access_hosts": self.vm.full_access_hosts,
                "use_ssh_agent": self.vm.use_ssh_agent,
                "connection_timeout": self.vm.connection_timeout,
            },
            "volumes": {
                "share_path": self.volumes.share_path,
                "outputs_path": self.volumes.outputs_path,
            },
            "tools": {"allowed": self.tools.allowed, "denied": self.tools.denied},
            "output_tracking": {
                "enabled": self.output_tracking.enabled,
                "max_log_entries": self.output_tracking.max_log_entries,
                "log_input_preview_length": self.output_tracking.log_input_preview_length,
                "log_output_preview_length": self.output_tracking.log_output_preview_length,
                "cleanup_max_age_days": self.output_tracking.cleanup_max_age_days,
                "cleanup_keep_min_runs": self.output_tracking.cleanup_keep_min_runs,
            },
        }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Values in override take precedence. Nested dictionaries are merged recursively.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables follow the pattern:
        LLMCORE_SANDBOX_<SECTION>_<KEY>=value

    Examples:
        LLMCORE_SANDBOX_MODE=vm
        LLMCORE_SANDBOX_DOCKER_IMAGE=python:3.12-slim
        LLMCORE_SANDBOX_VM_HOST=192.168.1.100

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with environment overrides applied
    """
    prefix = "LLMCORE_SANDBOX_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Parse key: LLMCORE_SANDBOX_DOCKER_IMAGE -> ["docker", "image"]
        parts = key[len(prefix) :].lower().split("_")

        if len(parts) == 1:
            # Top-level setting
            config_key = parts[0]
            if config_key in config:
                config[config_key] = _parse_env_value(value)
        elif len(parts) >= 2:
            # Nested setting
            section = parts[0]
            nested_key = "_".join(parts[1:])

            if section in config and isinstance(config[section], dict):
                if nested_key in config[section]:
                    config[section][nested_key] = _parse_env_value(value)

    return config


def _parse_env_value(value: str) -> Any:
    """
    Parse environment variable value to appropriate type.

    Args:
        value: String value from environment

    Returns:
        Parsed value (bool, int, float, list, or string)
    """
    # Boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        return [v.strip() for v in value.split(",")]

    # String
    return value


def load_toml_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to TOML file (default: ~/.llmcore/config.toml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path.home() / ".llmcore" / "config.toml"

    if not config_path.exists():
        logger.debug(f"Config file not found: {config_path}")
        return {}

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logger.warning("No TOML parser available, using defaults")
            return {}

    try:
        with open(config_path, "rb") as f:
            full_config = tomllib.load(f)

        # Extract sandbox section
        agents_config = full_config.get("agents", {})
        sandbox_config = agents_config.get("sandbox", {})

        logger.debug(f"Loaded sandbox config from {config_path}")
        return sandbox_config

    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def load_sandbox_config(
    config_path: Path | None = None, overrides: dict[str, Any] | None = None
) -> SandboxSystemConfig:
    """
    Load complete sandbox system configuration.

    Configuration is loaded and merged in order:
        1. Default values
        2. TOML config file
        3. Environment variables
        4. Runtime overrides

    Args:
        config_path: Optional path to TOML config file
        overrides: Optional runtime overrides

    Returns:
        SandboxSystemConfig instance
    """
    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Merge TOML config
    toml_config = load_toml_config(config_path)
    if toml_config:
        config = _deep_merge(config, toml_config)

    # Apply environment overrides
    config = _apply_env_overrides(config)

    # Apply runtime overrides
    if overrides:
        config = _deep_merge(config, overrides)

    # Build typed config objects
    docker_config = DockerConfig(
        enabled=config["docker"].get("enabled", True),
        image=config["docker"].get("image", "python:3.11-slim"),
        image_whitelist=config["docker"].get("image_whitelist", []),
        full_access_label=config["docker"].get("full_access_label", ""),
        full_access_name_pattern=config["docker"].get("full_access_name_pattern"),
        host=config["docker"].get("host"),
        auto_pull=config["docker"].get("auto_pull", True),
        memory_limit=config["docker"].get("memory_limit", "1g"),
        cpu_limit=config["docker"].get("cpu_limit", 2.0),
        timeout_seconds=config["docker"].get("timeout_seconds", 600),
    )

    vm_config = VMConfig(
        enabled=config["vm"].get("enabled", False),
        host=config["vm"].get("host"),
        port=config["vm"].get("port", 22),
        username=config["vm"].get("username", "agent"),
        private_key_path=config["vm"].get("private_key_path"),
        full_access_hosts=config["vm"].get("full_access_hosts", []),
        use_ssh_agent=config["vm"].get("use_ssh_agent", True),
        connection_timeout=config["vm"].get("connection_timeout", 30),
    )

    volumes_config = VolumeConfig(
        share_path=config["volumes"].get("share_path", "~/.llmcore/agent_share"),
        outputs_path=config["volumes"].get("outputs_path", "~/.llmcore/agent_outputs"),
    )

    tools_config = ToolsConfig(
        allowed=config["tools"].get("allowed", []), denied=config["tools"].get("denied", [])
    )

    output_config = OutputTrackingConfig(
        enabled=config["output_tracking"].get("enabled", True),
        max_log_entries=config["output_tracking"].get("max_log_entries", 10000),
        log_input_preview_length=config["output_tracking"].get("log_input_preview_length", 200),
        log_output_preview_length=config["output_tracking"].get("log_output_preview_length", 500),
        cleanup_max_age_days=config["output_tracking"].get("cleanup_max_age_days", 30),
        cleanup_keep_min_runs=config["output_tracking"].get("cleanup_keep_min_runs", 10),
    )

    return SandboxSystemConfig(
        mode=config.get("mode", "docker"),
        fallback_enabled=config.get("fallback_enabled", True),
        docker=docker_config,
        vm=vm_config,
        volumes=volumes_config,
        tools=tools_config,
        output_tracking=output_config,
    )


def create_registry_config(sandbox_config: SandboxSystemConfig) -> "SandboxRegistryConfig":
    """
    Create a SandboxRegistryConfig from SandboxSystemConfig.

    This converts the typed configuration into the format expected
    by SandboxRegistry.

    Args:
        sandbox_config: System configuration

    Returns:
        SandboxRegistryConfig instance
    """
    from .registry import SandboxMode, SandboxRegistryConfig

    mode_map = {"docker": SandboxMode.DOCKER, "vm": SandboxMode.VM, "hybrid": SandboxMode.HYBRID}

    return SandboxRegistryConfig(
        mode=mode_map.get(sandbox_config.mode, SandboxMode.DOCKER),
        fallback_enabled=sandbox_config.fallback_enabled,
        # Docker
        docker_enabled=sandbox_config.docker.enabled,
        docker_image=sandbox_config.docker.image,
        docker_image_whitelist=sandbox_config.docker.image_whitelist,
        docker_full_access_label=sandbox_config.docker.full_access_label,
        docker_full_access_name_pattern=sandbox_config.docker.full_access_name_pattern,
        docker_host=sandbox_config.docker.host,
        docker_auto_pull=sandbox_config.docker.auto_pull,
        docker_memory_limit=sandbox_config.docker.memory_limit,
        docker_cpu_limit=sandbox_config.docker.cpu_limit,
        docker_timeout_seconds=sandbox_config.docker.timeout_seconds,
        # VM
        vm_enabled=sandbox_config.vm.enabled or sandbox_config.mode == "vm",
        vm_host=sandbox_config.vm.host,
        vm_port=sandbox_config.vm.port,
        vm_username=sandbox_config.vm.username,
        vm_private_key_path=sandbox_config.vm.private_key_path,
        vm_full_access_hosts=sandbox_config.vm.full_access_hosts,
        vm_use_ssh_agent=sandbox_config.vm.use_ssh_agent,
        vm_connection_timeout=sandbox_config.vm.connection_timeout,
        # Volumes
        share_path=sandbox_config.volumes.share_path,
        outputs_path=sandbox_config.volumes.outputs_path,
        # Tools
        allowed_tools=sandbox_config.tools.allowed,
        denied_tools=sandbox_config.tools.denied,
    )


def generate_sample_config() -> str:
    """
    Generate a sample TOML configuration file content.

    Returns:
        TOML configuration string
    """
    return """# LLMCore Sandbox Configuration
# Place this in ~/.llmcore/config.toml or specify via LLMCORE_CONFIG_PATH

[agents.sandbox]
# Primary sandbox mode: "docker", "vm", or "hybrid"
# hybrid tries docker first, falls back to vm
mode = "docker"

# Enable fallback to secondary provider in hybrid mode
fallback_enabled = true

[agents.sandbox.docker]
# Enable Docker sandboxing
enabled = true

# Default Docker image
image = "python:3.11-slim"

# Whitelist of allowed image patterns (glob-style)
image_whitelist = [
    "python:3.*-slim",
    "python:3.*-bookworm",
    "llmcore-sandbox:*"
]

# Label that grants full access (key=value format)
full_access_label = "llmcore.sandbox.full_access=true"

# Image name pattern that grants full access
full_access_name_pattern = "*-full-access"

# Remote Docker host (optional, e.g., "tcp://192.168.1.100:2375")
# host = "tcp://docker-host:2375"

# Auto-pull images if not found locally
auto_pull = true

# Resource limits
memory_limit = "1g"
cpu_limit = 2.0
timeout_seconds = 600

[agents.sandbox.vm]
# Enable VM sandboxing (via SSH)
enabled = false

# VM host (required if enabled)
# host = "192.168.1.100"

# SSH port
port = 22

# SSH username
username = "agent"

# Path to private key file for PKI authentication
# private_key_path = "~/.ssh/llmcore_agent_key"

# Hosts that get full access (others are restricted)
full_access_hosts = []

# Use SSH agent for key lookup
use_ssh_agent = true

# Connection timeout in seconds
connection_timeout = 30

[agents.sandbox.volumes]
# Persistent shared data directory
share_path = "~/.llmcore/agent_share"

# Output files directory
outputs_path = "~/.llmcore/agent_outputs"

[agents.sandbox.tools]
# Tools allowed in restricted sandboxes
allowed = [
    "execute_shell",
    "execute_python",
    "save_file",
    "load_file",
    "replace_in_file",
    "list_files",
    "file_exists",
    "delete_file",
    "create_directory",
    "get_state",
    "set_state",
    "list_state",
    "get_sandbox_info",
    "get_recorded_files",
    "calculator",
    "semantic_search",
    "episodic_search",
    "finish",
    "human_approval"
]

# Tools denied even if in allowed list (deny takes priority)
denied = [
    "install_system_package",
    "sudo_execute",
    "network_request",
    "raw_socket"
]

[agents.sandbox.output_tracking]
# Enable output tracking
enabled = true

# Maximum execution log entries per run
max_log_entries = 10000

# Preview lengths for logs
log_input_preview_length = 200
log_output_preview_length = 500

# Cleanup settings
cleanup_max_age_days = 30
cleanup_keep_min_runs = 10
"""


def write_sample_config(path: Path | None = None) -> Path:
    """
    Write a sample configuration file.

    Args:
        path: Path to write to (default: ~/.llmcore/config.toml.sample)

    Returns:
        Path where config was written
    """
    if path is None:
        path = Path.home() / ".llmcore" / "config.toml.sample"

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(generate_sample_config())

    logger.info(f"Wrote sample config to {path}")
    return path
