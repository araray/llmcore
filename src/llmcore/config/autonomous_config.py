# src/llmcore/config/autonomous_config.py
"""
Autonomous operation configuration models.

This module defines Pydantic models for all autonomous operation
configuration sections. These models are used for:
1. Type-safe configuration loading
2. Validation with sensible defaults
3. Documentation generation
4. Runtime configuration updates

The configuration hierarchy:
    AutonomousConfig (root)
    ├── GoalsAutonomousConfig    - Goal management settings
    ├── HeartbeatConfig          - Heartbeat/periodic task settings
    ├── EscalationConfig         - Human escalation settings
    │   ├── WebhookConfig        - Webhook notification settings
    │   └── FileNotificationConfig - File-based notification settings
    ├── ResourcesConfig          - Resource monitoring settings
    ├── SkillsConfig             - Skill loading settings
    └── ContextConfig            - Context synthesis settings

Usage:
    >>> from llmcore.config.autonomous_config import AutonomousConfig
    >>> config = AutonomousConfig()  # All defaults
    >>> config.goals.max_sub_goals
    10

    >>> # Override specific settings
    >>> config = AutonomousConfig(
    ...     resources=ResourcesConfig(max_cpu_percent=70.0)
    ... )

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §7 (Autonomous Operation)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §5.3 ([autonomous] config)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# GOALS CONFIGURATION
# =============================================================================


class GoalsAutonomousConfig(BaseModel):
    """
    Configuration for autonomous goal management.

    Controls goal persistence, decomposition behavior, and retry
    settings for the GoalManager.

    Examples:
        >>> config = GoalsAutonomousConfig()
        >>> config.persist_goals
        True
        >>> config.max_attempts_per_goal
        10
    """

    persist_goals: bool = Field(
        default=True,
        description="Enable persistent goal storage across sessions",
    )
    storage_path: str = Field(
        default="~/.local/share/llmcore/goals.json",
        description=(
            "Path to goals storage file (JSON). "
            "Tilde and environment variable expansion is applied."
        ),
    )

    # --- Decomposition settings ---
    auto_decompose: bool = Field(
        default=True,
        description="Automatically decompose goals into sub-goals via LLM",
    )
    max_sub_goals: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of sub-goals per parent goal",
    )
    max_goal_depth: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum nesting depth for goal hierarchies",
    )

    # --- Retry settings ---
    max_attempts_per_goal: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum retry attempts before marking a goal as failed",
    )
    base_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Base cooldown between retry attempts (exponential backoff)",
    )
    max_cooldown_seconds: float = Field(
        default=3600.0,
        ge=0.0,
        description="Maximum cooldown cap for exponential backoff",
    )

    @field_validator("storage_path")
    @classmethod
    def expand_storage_path(cls, v: str) -> str:
        """Expand ~ and environment variables in storage_path."""
        return os.path.expanduser(os.path.expandvars(v))


# =============================================================================
# HEARTBEAT CONFIGURATION
# =============================================================================


class HeartbeatConfig(BaseModel):
    """
    Configuration for the heartbeat / periodic task scheduler.

    The heartbeat system drives periodic tasks like resource checks,
    goal re-evaluation, and status reporting.

    Examples:
        >>> config = HeartbeatConfig()
        >>> config.base_interval
        60.0
        >>> config.max_concurrent_tasks
        3
    """

    enabled: bool = Field(
        default=True,
        description="Enable the heartbeat task scheduler",
    )
    base_interval: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Base heartbeat interval in seconds",
    )
    max_concurrent_tasks: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of heartbeat tasks running concurrently",
    )


# =============================================================================
# ESCALATION CONFIGURATION
# =============================================================================


class WebhookConfig(BaseModel):
    """
    Webhook notification channel configuration.

    Sends escalation notifications to an HTTP webhook endpoint.

    Examples:
        >>> config = WebhookConfig(
        ...     enabled=True,
        ...     url="https://hooks.example.com/notify"
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable webhook notifications",
    )
    url: str = Field(
        default="",
        description=(
            "Webhook URL. Supports ${ENV_VAR} substitution. Example: '${ESCALATION_WEBHOOK_URL}'"
        ),
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers for the webhook request",
    )

    @field_validator("url")
    @classmethod
    def expand_url_env_vars(cls, v: str) -> str:
        """Expand environment variables in URL."""
        return os.path.expandvars(v)


class FileNotificationConfig(BaseModel):
    """
    File-based notification channel configuration.

    Appends escalation notifications to a log file.

    Examples:
        >>> config = FileNotificationConfig()
        >>> config.path
        '~/.local/share/llmcore/escalations.log'
    """

    enabled: bool = Field(
        default=True,
        description="Enable file-based escalation logging",
    )
    path: str = Field(
        default="~/.local/share/llmcore/escalations.log",
        description="Path to escalation log file",
    )

    @field_validator("path")
    @classmethod
    def expand_file_path(cls, v: str) -> str:
        """Expand ~ and environment variables in file path."""
        return os.path.expanduser(os.path.expandvars(v))


class EscalationConfig(BaseModel):
    """
    Configuration for the human escalation framework.

    Controls when and how the system notifies humans about
    issues requiring attention.

    Examples:
        >>> config = EscalationConfig()
        >>> config.auto_resolve_below
        'advisory'
        >>> config.dedup_window
        300
    """

    enabled: bool = Field(
        default=True,
        description="Enable the escalation framework",
    )
    auto_resolve_below: str = Field(
        default="advisory",
        description=(
            "Auto-resolve escalations below this level. "
            "Levels: debug, info, advisory, action, urgent, critical"
        ),
    )
    dedup_window: int = Field(
        default=300,
        ge=0,
        le=86400,
        description="Deduplication window in seconds (0 = no dedup)",
    )

    # --- Notification channels ---
    webhook: WebhookConfig = Field(
        default_factory=WebhookConfig,
        description="Webhook notification configuration",
    )
    file: FileNotificationConfig = Field(
        default_factory=FileNotificationConfig,
        description="File-based notification configuration",
    )

    @field_validator("auto_resolve_below")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate escalation level string."""
        valid = {"debug", "info", "advisory", "action", "urgent", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid escalation level: {v!r}. Valid levels: {sorted(valid)}")
        return v.lower()


# =============================================================================
# RESOURCE MONITORING CONFIGURATION
# =============================================================================


class ResourcesConfig(BaseModel):
    """
    Configuration for system resource monitoring and constraints.

    Defines hardware and API cost limits that the agent must respect
    during autonomous operation. Critical for Raspberry Pi deployments.

    Examples:
        >>> config = ResourcesConfig()
        >>> config.max_cpu_percent
        80.0
        >>> config.max_daily_cost_usd
        10.0

        >>> # Constrained Pi deployment
        >>> config = ResourcesConfig(
        ...     max_cpu_percent=60.0,
        ...     max_memory_percent=70.0,
        ...     max_temperature_c=65.0,
        ... )
    """

    enabled: bool = Field(
        default=True,
        description="Enable resource monitoring",
    )
    check_interval: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="Resource check interval in seconds",
    )

    # --- Hardware limits ---
    max_cpu_percent: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Maximum CPU utilization percentage (soft limit)",
    )
    max_memory_percent: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Maximum memory utilization percentage (soft limit)",
    )
    max_temperature_c: float = Field(
        default=75.0,
        ge=0.0,
        le=110.0,
        description="Maximum CPU temperature in Celsius (hard limit)",
    )
    min_disk_free_gb: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum free disk space in GB (hard limit)",
    )

    # --- API cost limits ---
    max_hourly_cost_usd: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum API cost per hour in USD (hard limit)",
    )
    max_daily_cost_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum API cost per day in USD (hard limit)",
    )

    # --- Token limits ---
    max_hourly_tokens: int = Field(
        default=100_000,
        ge=0,
        description="Maximum tokens consumed per hour (soft limit)",
    )
    max_daily_tokens: int = Field(
        default=1_000_000,
        ge=0,
        description="Maximum tokens consumed per day (soft limit)",
    )


# =============================================================================
# SKILL LOADING CONFIGURATION
# =============================================================================


class SkillsConfig(BaseModel):
    """
    Configuration for the Skill Loading System.

    Controls which directories are scanned for skill files, caching
    behavior, and per-task loading limits.

    Skill files are markdown (``.md``) documents with optional YAML
    frontmatter.  The loader scans all registered directories
    recursively for ``.md`` files on first use.

    Examples:
        >>> config = SkillsConfig()
        >>> config.skill_directories
        ['~/.local/share/llmcore/skills']
        >>> config.cache_size
        50

    References:
        - UNIFIED_ECOSYSTEM_SPECIFICATION.md §13 (Skill Loading System)
    """

    skill_directories: list[str] = Field(
        default=["~/.local/share/llmcore/skills"],
        description=(
            "Directories to scan for skill markdown files. "
            "Supports ~ expansion. Searched recursively."
        ),
    )
    cache_size: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Maximum number of skill file contents to cache in memory",
    )
    max_skills_per_task: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of skills loaded per task",
    )
    max_skill_tokens: int = Field(
        default=20_000,
        ge=1000,
        description="Maximum total tokens for all skills loaded per task",
    )


# =============================================================================
# CONTEXT SYNTHESIS CONFIGURATION
# =============================================================================


class ContextConfig(BaseModel):
    """
    Configuration for Adaptive Context Synthesis.

    Controls how the system reconstructs context for each
    autonomous iteration.

    Examples:
        >>> config = ContextConfig()
        >>> config.max_context_tokens
        100000
        >>> config.prioritization_strategy
        'recency_relevance'
    """

    max_context_tokens: int = Field(
        default=100_000,
        ge=1000,
        description="Maximum tokens for synthesized context window",
    )
    compression_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Compression triggers when context usage exceeds this fraction of max_context_tokens"
        ),
    )
    prioritization_strategy: str = Field(
        default="recency_relevance",
        description=(
            "Context prioritization strategy. "
            "Options: 'recency_relevance', 'relevance_only', 'recency_only'"
        ),
    )

    @field_validator("prioritization_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate prioritization strategy string."""
        valid = {"recency_relevance", "relevance_only", "recency_only"}
        if v not in valid:
            raise ValueError(f"Invalid strategy: {v!r}. Valid: {sorted(valid)}")
        return v


# =============================================================================
# ROOT AUTONOMOUS CONFIGURATION
# =============================================================================


class AutonomousConfig(BaseModel):
    """
    Root configuration for the autonomous operation module.

    Aggregates all sub-configurations for goals, heartbeat,
    escalation, resources, and context synthesis.

    The autonomous module is the core infrastructure enabling
    agents to operate without continuous human interaction.

    Usage:
        >>> config = AutonomousConfig()
        >>> config.enabled
        True

        >>> # From TOML dict
        >>> config = AutonomousConfig(**toml_dict["autonomous"])

        >>> # Constrained deployment
        >>> config = AutonomousConfig(
        ...     resources=ResourcesConfig(
        ...         max_cpu_percent=60.0,
        ...         max_temperature_c=65.0,
        ...     ),
        ...     heartbeat=HeartbeatConfig(base_interval=120.0),
        ... )

    References:
        - UNIFIED_ECOSYSTEM_SPECIFICATION.md §7 (Autonomous Operation)
        - UNIFIED_ECOSYSTEM_SPECIFICATION.md §5.3 ([autonomous] config)
    """

    enabled: bool = Field(
        default=True,
        description="Master switch for autonomous operation capabilities",
    )
    goals: GoalsAutonomousConfig = Field(
        default_factory=GoalsAutonomousConfig,
        description="Goal management configuration",
    )
    heartbeat: HeartbeatConfig = Field(
        default_factory=HeartbeatConfig,
        description="Heartbeat/periodic task scheduler configuration",
    )
    escalation: EscalationConfig = Field(
        default_factory=EscalationConfig,
        description="Human escalation framework configuration",
    )
    resources: ResourcesConfig = Field(
        default_factory=ResourcesConfig,
        description="Resource monitoring and constraint configuration",
    )
    skills: SkillsConfig = Field(
        default_factory=SkillsConfig,
        description="Skill loading system configuration",
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Adaptive Context Synthesis configuration",
    )


# =============================================================================
# HELPER: LOAD FROM TOML DICT
# =============================================================================


def load_autonomous_config(
    config_dict: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> AutonomousConfig:
    """
    Load autonomous configuration from a dictionary or TOML file.

    Args:
        config_dict: Pre-parsed configuration dictionary. If provided,
            extracts the ``"autonomous"`` key if present.
        config_path: Path to a TOML file. If provided, reads and parses
            it, then extracts the ``"autonomous"`` section.

    Returns:
        Validated AutonomousConfig instance with defaults for
        any unspecified settings.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If validation fails on any config value.

    Examples:
        >>> config = load_autonomous_config()
        >>> config.enabled
        True

        >>> config = load_autonomous_config(config_dict={
        ...     "autonomous": {"resources": {"max_cpu_percent": 60.0}}
        ... })
        >>> config.resources.max_cpu_percent
        60.0
    """
    data: dict[str, Any] = {}

    if config_path is not None:
        import tomllib

        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        data = raw.get("autonomous", {})

    if config_dict is not None:
        # If the dict has an "autonomous" key, extract it
        if "autonomous" in config_dict:
            data = config_dict["autonomous"]
        else:
            data = config_dict

    return AutonomousConfig(**data)
