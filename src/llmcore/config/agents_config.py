# src/llmcore/config/agents_config.py
"""
Agent system configuration models.

This module defines Pydantic models for all G3 agent configuration
sections. These models are used for:
1. Type-safe configuration loading
2. Validation with sensible defaults
3. Documentation generation
4. Runtime configuration updates

The configuration hierarchy:
    AgentsConfig (root)
    ├── GoalsConfig          - Goal classification settings
    ├── FastPathConfig       - Fast-path execution settings
    ├── CircuitBreakerConfig - Circuit breaker settings
    ├── ActivitiesConfig     - Activity system settings
    ├── CapabilityCheckConfig- Model capability checking
    ├── HITLConfig           - Human-in-the-loop settings
    ├── RoutingConfig        - Model routing settings
    └── DarwinConfig         - Darwin agent enhancements (Phase 6)

Usage:
    >>> from llmcore.config.agents_config import AgentsConfig, load_agents_config
    >>> config = AgentsConfig()  # All defaults
    >>> config.goals.classifier_enabled
    True

    >>> # Load from TOML
    >>> config = load_agents_config(config_path=Path("config.toml"))

    >>> # Load with overrides
    >>> config = load_agents_config(
    ...     config_dict={"agents": {"goals": {"classifier_enabled": False}}}
    ... )

References:
    - G3_COMPLETE_IMPLEMENTATION_PLAN.md
    - LLMCORE_AGENTIC_SYSTEM_MASTER_PLAN_G3.md
    - UNIFIED_IMPLEMENTATION_PLAN.md (Phase 6)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .darwin_config import DarwinConfig

if TYPE_CHECKING:
    from confy.loader import Config as ConfyConfig

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class TimeoutPolicy(str, Enum):
    """Policy for handling HITL approval timeouts."""

    APPROVE = "approve"  # Auto-approve on timeout
    REJECT = "reject"  # Auto-reject on timeout
    ESCALATE = "escalate"  # Escalate to higher authority


class RoutingStrategy(str, Enum):
    """Model routing optimization strategy."""

    COST_OPTIMIZED = "cost_optimized"  # Minimize cost
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize latency


class RiskLevel(str, Enum):
    """Risk levels for HITL classification."""

    NONE = "none"  # No risk, auto-approve
    LOW = "low"  # Low risk, may auto-approve
    MEDIUM = "medium"  # Medium risk, approval recommended
    HIGH = "high"  # High risk, approval required
    CRITICAL = "critical"  # Critical risk, mandatory approval


# =============================================================================
# GOAL CLASSIFICATION CONFIG
# =============================================================================


class GoalsConfig(BaseModel):
    """
    Configuration for goal classification and fast-path routing.

    Goal classification analyzes user goals to determine complexity and
    appropriate execution strategy. This enables fast-path routing for
    trivial goals (e.g., "hello" → <5s response).
    """

    # Classification settings
    classifier_enabled: bool = Field(
        default=True, description="Enable goal complexity classification"
    )
    use_llm_fallback: bool = Field(
        default=False,
        description="Use LLM for uncertain classifications (slower but more accurate)",
    )
    heuristic_confidence_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for heuristic classification",
    )

    # Iteration limits by complexity
    trivial_max_iterations: int = Field(
        default=1, ge=1, le=10, description="Max iterations for trivial goals"
    )
    simple_max_iterations: int = Field(
        default=5, ge=1, le=20, description="Max iterations for simple goals"
    )
    moderate_max_iterations: int = Field(
        default=10, ge=1, le=50, description="Max iterations for moderate goals"
    )
    complex_max_iterations: int = Field(
        default=15, ge=1, le=100, description="Max iterations for complex goals"
    )


# =============================================================================
# FAST PATH CONFIG
# =============================================================================


class FastPathConfig(BaseModel):
    """
    Configuration for fast-path execution of trivial goals.

    Fast-path bypasses the full cognitive cycle for trivial goals,
    achieving <5 second response times for greetings, simple questions, etc.
    """

    enabled: bool = Field(default=True, description="Enable fast-path for trivial goals")

    # Caching
    cache_enabled: bool = Field(default=True, description="Use cached responses")
    cache_max_entries: int = Field(
        default=100, ge=10, le=10000, description="Maximum cache entries"
    )
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache entry TTL in seconds")

    # Templates
    templates_enabled: bool = Field(
        default=True, description="Use template responses for common intents"
    )

    # Performance
    max_response_time_ms: int = Field(
        default=5000,
        ge=100,
        le=30000,
        description="Maximum response time before timeout (ms)",
    )

    # LLM parameters
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature for fast-path"
    )
    max_tokens: int = Field(
        default=500, ge=50, le=4000, description="Maximum tokens for fast-path response"
    )

    fallback_on_timeout: bool = Field(
        default=True, description="Fall back to template response on timeout"
    )


# =============================================================================
# CIRCUIT BREAKER CONFIG
# =============================================================================


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for the agent circuit breaker.

    The circuit breaker prevents runaway agent execution by detecting:
    - Too many iterations
    - Repeated identical errors
    - Time limits exceeded
    - Cost limits exceeded
    - Progress stalls
    """

    enabled: bool = Field(default=True, description="Enable circuit breaker")
    max_iterations: int = Field(
        default=15, ge=1, le=1000, description="Maximum iterations before tripping"
    )
    max_same_errors: int = Field(
        default=3, ge=1, le=100, description="Trip after N identical errors"
    )
    max_execution_time_seconds: int = Field(default=300, ge=1, description="Trip after N seconds")
    max_total_cost: float = Field(
        default=1.0, ge=0.0, description="Trip after spending more than $X"
    )
    progress_stall_threshold: int = Field(
        default=5, ge=1, le=50, description="Trip if progress stalls for N iterations"
    )
    progress_stall_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum progress change to count as progress",
    )


# =============================================================================
# ACTIVITIES CONFIG
# =============================================================================


class ActivitiesConfig(BaseModel):
    """
    Configuration for the activity system (model-agnostic tool execution).

    The activity system provides tool execution for models without native
    function calling support. Activities are parsed from LLM output and
    executed through the activity registry.
    """

    enabled: bool = Field(default=True, description="Enable activity system as fallback")
    fallback_to_native_tools: bool = Field(
        default=True, description="Prefer native tools when model supports them"
    )

    # Limits
    max_per_iteration: int = Field(
        default=10, ge=1, le=100, description="Maximum activities per iteration"
    )
    max_total: int = Field(
        default=100, ge=1, le=1000, description="Maximum total activities per session"
    )

    # Timeouts
    default_timeout_seconds: int = Field(default=60, ge=1, description="Default activity timeout")
    total_timeout_seconds: int = Field(default=300, ge=1, description="Total session timeout")

    # Behavior
    stop_on_error: bool = Field(default=False, description="Stop processing on first error")
    parallel_execution: bool = Field(default=False, description="Execute activities in parallel")
    max_observation_length: int = Field(
        default=4000, ge=100, description="Maximum observation length per activity"
    )
    include_reasoning: bool = Field(
        default=True, description="Include reasoning in activity output"
    )


# =============================================================================
# CAPABILITY CHECK CONFIG
# =============================================================================


class CapabilityCheckConfig(BaseModel):
    """
    Configuration for model capability pre-flight checks.

    Capability checking ensures the selected model supports required
    capabilities (tools, vision, etc.) before execution begins.
    """

    enabled: bool = Field(default=True, description="Enable pre-flight capability checking")
    use_model_cards: bool = Field(
        default=True, description="Consult model card registry for capability info"
    )
    use_runtime_query: bool = Field(
        default=True,
        description="Query provider at runtime (e.g., Ollama /api/show)",
    )
    strict_mode: bool = Field(
        default=True,
        description="Fail immediately if model doesn't support required capabilities",
    )
    suggest_alternatives: bool = Field(
        default=True, description="Suggest alternative models when check fails"
    )


# =============================================================================
# HITL CONFIG
# =============================================================================


class HITLConfig(BaseModel):
    """
    Configuration for Human-In-The-Loop approval system.

    HITL provides human oversight for risky operations by requiring
    approval before execution of high-risk actions.
    """

    enabled: bool = Field(default=True, description="Enable HITL system")
    global_risk_threshold: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Minimum risk level requiring approval",
    )

    # Timeouts
    default_timeout_seconds: int = Field(
        default=300, ge=1, description="Default approval timeout (seconds)"
    )
    timeout_policy: TimeoutPolicy = Field(
        default=TimeoutPolicy.REJECT, description="Default timeout handling policy"
    )
    timeout_policies_by_risk: dict[str, TimeoutPolicy] = Field(
        default_factory=lambda: {
            "low": TimeoutPolicy.APPROVE,
            "medium": TimeoutPolicy.REJECT,
            "high": TimeoutPolicy.REJECT,
            "critical": TimeoutPolicy.REJECT,
        },
        description="Timeout policy per risk level",
    )

    # Tool classifications
    safe_tools: list[str] = Field(
        default_factory=lambda: ["final_answer", "respond_to_user", "think_aloud"],
        description="Tools that never require approval",
    )
    low_risk_tools: list[str] = Field(
        default_factory=lambda: ["file_read", "file_search", "list_directory"],
        description="Low risk tools",
    )
    high_risk_tools: list[str] = Field(
        default_factory=lambda: ["bash_exec", "python_exec", "file_delete"],
        description="High risk tools",
    )
    critical_tools: list[str] = Field(
        default_factory=lambda: ["execute_sudo", "drop_database"],
        description="Critical risk tools (always require approval)",
    )

    # Batch approval
    batch_similar_requests: bool = Field(
        default=True, description="Batch similar requests together"
    )
    batch_window_seconds: int = Field(
        default=5, ge=1, description="Window for batching similar requests"
    )

    # Audit
    audit_logging_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_path: str | None = Field(default=None, description="Path for audit log file")


# =============================================================================
# ROUTING CONFIG
# =============================================================================


class RoutingTiersConfig(BaseModel):
    """Model tiers for routing."""

    fast: list[str] = Field(
        default_factory=lambda: ["gpt-4o-mini", "claude-3-haiku", "gemma3:1b"],
        description="Fast/cheap models",
    )
    balanced: list[str] = Field(
        default_factory=lambda: ["gpt-4o", "claude-3-5-sonnet", "llama3.3:70b"],
        description="Balanced models",
    )
    capable: list[str] = Field(
        default_factory=lambda: ["gpt-4-turbo", "claude-3-opus"],
        description="Most capable models",
    )


class RoutingConfig(BaseModel):
    """
    Configuration for model routing.

    Model routing selects the appropriate model based on task complexity
    and configured optimization strategy.
    """

    enabled: bool = Field(default=True, description="Enable model routing")
    strategy: RoutingStrategy = Field(
        default=RoutingStrategy.COST_OPTIMIZED,
        description="Routing optimization strategy",
    )
    fallback_enabled: bool = Field(
        default=True, description="Enable fallback to alternative models"
    )
    tiers: RoutingTiersConfig = Field(
        default_factory=RoutingTiersConfig, description="Model tier configuration"
    )


# =============================================================================
# ROOT AGENTS CONFIG
# =============================================================================


class AgentsConfig(BaseModel):
    """
    Root configuration model for the agent system.

    This is the top-level configuration for all G3 agent components.
    It corresponds to the [agents] section in TOML configuration.

    Usage:
        >>> config = AgentsConfig()  # All defaults
        >>> config.goals.classifier_enabled
        True

        >>> config = AgentsConfig(
        ...     max_iterations=20,
        ...     goals=GoalsConfig(classifier_enabled=False)
        ... )
    """

    # Global settings
    max_iterations: int = Field(default=10, ge=1, le=1000, description="Default maximum iterations")
    default_timeout: int = Field(default=600, ge=1, description="Default session timeout (seconds)")
    default_persona: str = Field(default="assistant", description="Default persona for agents")
    memory_enabled: bool = Field(default=True, description="Enable agent memory system")

    # Component configs
    goals: GoalsConfig = Field(
        default_factory=GoalsConfig, description="Goal classification settings"
    )
    fast_path: FastPathConfig = Field(
        default_factory=FastPathConfig, description="Fast-path execution settings"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig, description="Circuit breaker settings"
    )
    activities: ActivitiesConfig = Field(
        default_factory=ActivitiesConfig, description="Activity system settings"
    )
    capability_check: CapabilityCheckConfig = Field(
        default_factory=CapabilityCheckConfig, description="Capability checking settings"
    )
    hitl: HITLConfig = Field(default_factory=HITLConfig, description="HITL system settings")
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig, description="Model routing settings"
    )
    darwin: DarwinConfig = Field(
        default_factory=DarwinConfig,
        description="Darwin agent enhancement settings (Phase 6)",
    )


# =============================================================================
# CONFIG LOADING
# =============================================================================


def load_agents_config(
    config: "ConfyConfig | None" = None,
    overrides: dict[str, Any] | None = None,
    *,
    # --- Backward-compatible parameters (deprecated) ---
    config_path: Path | None = None,
    config_dict: dict[str, Any] | None = None,
) -> AgentsConfig:
    """
    Load agent configuration from a unified confy Config or legacy sources.

    **Preferred (new API)**::

        # From unified confy Config (reads the "agents" section):
        config = load_agents_config(config=llmcore.config)

        # With runtime overrides:
        config = load_agents_config(config=llmcore.config, overrides={"max_iterations": 20})

    **Legacy (backward-compatible, deprecated)**::

        # From TOML file:
        config = load_agents_config(config_path=Path("config.toml"))

        # From dict:
        config = load_agents_config(config_dict={"agents": {"max_iterations": 20}})

    When ``config`` is provided, the "agents" section is extracted from it.
    When legacy parameters are used, a temporary confy Config is built internally
    to perform the same merge (TOML → env → overrides).

    Args:
        config: Unified confy Config object (from ``LLMCore.config``).
            Reads the ``"agents"`` section.
        overrides: Optional runtime overrides (merged last, highest precedence).
        config_path: *Deprecated.* Path to a TOML config file.
        config_dict: *Deprecated.* Configuration dictionary with an
            ``"agents"`` top-level key.

    Returns:
        Validated ``AgentsConfig`` instance.  Falls back to defaults on
        validation errors.
    """
    import warnings

    agents_dict: dict[str, Any] = {}

    # --- New path: extract from unified confy Config ---
    if config is not None:
        agents_section = config.get("agents", {})
        if hasattr(agents_section, "as_dict"):
            agents_dict = agents_section.as_dict()
        elif isinstance(agents_section, dict):
            agents_dict = dict(agents_section)
        # Note: env vars already merged by confy when building `config`.

    # --- Legacy path: build from config_path / config_dict ---
    elif config_path is not None or config_dict is not None:
        warnings.warn(
            "load_agents_config(config_path=..., config_dict=...) is deprecated. "
            "Pass the unified confy Config via config= instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        agents_dict = _load_agents_legacy(config_path, config_dict)

    else:
        # No arguments: still apply env var overrides (LLMCORE_AGENTS__*).
        agents_dict = _apply_env_overrides_legacy(agents_dict)

    # --- Apply runtime overrides ---
    if overrides:
        try:
            from confy.loader import deep_merge
        except ImportError:
            deep_merge = _deep_merge_fallback  # type: ignore[assignment]
        agents_dict = deep_merge(agents_dict, overrides)

    # --- Validate and return ---
    try:
        return AgentsConfig(**agents_dict)
    except Exception as e:
        logger.error(f"Invalid agents configuration: {e}")
        logger.warning("Using default configuration")
        return AgentsConfig()


# ---------------------------------------------------------------------------
# Legacy helpers (backward compat)
# ---------------------------------------------------------------------------


def _load_agents_legacy(
    config_path: Path | None,
    config_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Load agents dict from legacy TOML-path / dict sources.

    Delegates to confy when available, falls back to manual TOML loading.
    """
    merged: dict[str, Any] = {}

    # Load from TOML file
    if config_path is not None:
        try:
            from confy.loader import Config as ActualConfyConfig

            cfg = ActualConfyConfig(
                file_path=str(config_path),
                load_dotenv_file=False,
                prefix=None,
            )
            agents_section = cfg.get("agents", {})
            if hasattr(agents_section, "as_dict"):
                merged = agents_section.as_dict()
            elif isinstance(agents_section, dict):
                merged = dict(agents_section)
        except ImportError:
            # confy not available — manual TOML loading
            merged = _load_toml_agents_section(config_path)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load agents config from {config_path}: {e}")

    # Merge config dictionary
    if config_dict is not None:
        agents_section = config_dict.get("agents", {})
        try:
            from confy.loader import deep_merge
        except ImportError:
            deep_merge = _deep_merge_fallback  # type: ignore[assignment]
        merged = deep_merge(merged, agents_section)

    # Apply environment variable overrides (confy handles this when available,
    # but in legacy mode without a full Config we do it manually)
    merged = _apply_env_overrides_legacy(merged)

    return merged


def _load_toml_agents_section(config_path: Path) -> dict[str, Any]:
    """Load the ``[agents]`` section from a TOML file (no confy)."""
    try:
        import sys

        if sys.version_info >= (3, 11):
            import tomllib as _tomllib
        else:
            try:
                import tomli as _tomllib
            except ImportError:
                logger.warning("No TOML parser available (install tomli for Python <3.11)")
                return {}

        with open(config_path, "rb") as f:
            full_config = _tomllib.load(f)
            agents_section = full_config.get("agents", {})
            logger.debug(f"Loaded agents config from {config_path}")
            return agents_section
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
    except Exception as e:
        logger.warning(f"Failed to load agents config from {config_path}: {e}")
    return {}


def _apply_env_overrides_legacy(config: dict[str, Any]) -> dict[str, Any]:
    """Apply ``LLMCORE_AGENTS__*`` environment variable overrides (legacy path).

    This is the fallback when the caller does not pass a unified confy Config.
    When confy is used, its own env-var collection handles this automatically.
    """
    prefix = "LLMCORE_AGENTS__"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Parse the key path
        path_parts = key[len(prefix) :].lower().split("__")
        if len(path_parts) < 2:
            continue

        # Navigate to the right place in config
        current = config
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value (with type conversion)
        final_key = path_parts[-1]
        current[final_key] = _convert_env_value_legacy(value)

    return config


def _convert_env_value_legacy(value: str) -> Any:
    """Convert environment variable string to appropriate type (legacy path)."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

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

    # String
    return value


def _deep_merge_fallback(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Minimal deep merge fallback when confy is not installed.

    This preserves backward compatibility for code that imports ``_deep_merge``
    from this module (e.g. tests).  Prefer ``confy.loader.deep_merge`` instead.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_fallback(result[key], value)
        else:
            result[key] = value
    return result


# Backward-compatible alias — some tests import ``_deep_merge`` directly.
_deep_merge = _deep_merge_fallback


# =============================================================================
# EXPORTS
# =============================================================================

# Rebuild models to resolve forward references
GoalsConfig.model_rebuild()
FastPathConfig.model_rebuild()
CircuitBreakerConfig.model_rebuild()
ActivitiesConfig.model_rebuild()
CapabilityCheckConfig.model_rebuild()
HITLConfig.model_rebuild()
RoutingTiersConfig.model_rebuild()
RoutingConfig.model_rebuild()
DarwinConfig.model_rebuild()
AgentsConfig.model_rebuild()


__all__ = [
    # Enums
    "TimeoutPolicy",
    "RoutingStrategy",
    "RiskLevel",
    # Config classes
    "GoalsConfig",
    "FastPathConfig",
    "CircuitBreakerConfig",
    "ActivitiesConfig",
    "CapabilityCheckConfig",
    "HITLConfig",
    "RoutingConfig",
    "RoutingTiersConfig",
    "DarwinConfig",
    "AgentsConfig",
    # Functions
    "load_agents_config",
    # Backward-compat aliases (prefer confy.loader.deep_merge)
    "_deep_merge",
]
