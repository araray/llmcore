# src/llmcore/config/darwin_config.py
"""
Darwin agent enhancement configuration models.

This module defines Pydantic models for Darwin agent configuration,
including failure learning, TDD support, and multi-attempt arbiter.

These models integrate with the main AgentsConfig via:
    src/llmcore/config/agents_config.py (add DarwinConfig field)

Usage:
    >>> from llmcore.config.darwin_config import DarwinConfig
    >>> config = DarwinConfig()
    >>> config.failure_learning.enabled
    True
    >>> config.failure_learning.backend
    'sqlite'

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 6
    - Phase 6.1: Failure Learning System
    - Phase 6.2: TDD Support (planned)
    - Phase 6.3: Multi-Attempt Arbiter (planned)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# FAILURE LEARNING CONFIGURATION
# =============================================================================


class FailureLearningPostgresConfig(BaseModel):
    """PostgreSQL-specific configuration for failure learning."""

    min_pool_size: int = Field(
        default=2, ge=1, le=50, description="Minimum number of connections in pool"
    )
    max_pool_size: int = Field(
        default=10, ge=1, le=100, description="Maximum number of connections in pool"
    )
    table_prefix: str = Field(
        default="", max_length=50, description="Prefix for table names (useful for multi-tenant)"
    )


class FailureLearningConfig(BaseModel):
    """
    Configuration for Darwin failure learning system.

    Enables agents to learn from past failures and avoid repeating mistakes.
    Supports both SQLite (development) and PostgreSQL (production) backends.

    Examples:
        >>> config = FailureLearningConfig()
        >>> config.enabled
        True
        >>> config.backend
        'sqlite'

        >>> # Production PostgreSQL config
        >>> config = FailureLearningConfig(
        ...     backend="postgres",
        ...     db_url="postgresql://user:pass@localhost/llmcore"
        ... )
    """

    enabled: bool = Field(default=True, description="Enable failure learning system")
    backend: Literal["sqlite", "postgres"] = Field(
        default="sqlite", description="Storage backend: 'sqlite' for dev, 'postgres' for production"
    )
    db_path: str = Field(
        default="~/.local/share/llmcore/failures.db",
        description="SQLite database path (used when backend='sqlite')",
    )
    db_url: str = Field(
        default="", description="PostgreSQL connection URL (used when backend='postgres')"
    )
    max_failures_to_retrieve: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of similar failures to retrieve before planning",
    )
    postgres: FailureLearningPostgresConfig = Field(
        default_factory=FailureLearningPostgresConfig, description="PostgreSQL-specific settings"
    )

    @field_validator("db_url")
    @classmethod
    def validate_postgres_url(cls, v: str, info) -> str:
        """Validate PostgreSQL URL is provided when backend is postgres."""
        backend = info.data.get("backend")
        if backend == "postgres" and not v:
            # Don't raise error - allow env var to provide it
            pass
        return v

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        """Expand user home directory in path."""
        return str(Path(v).expanduser())


# =============================================================================
# TDD CONFIGURATION (Phase 6.2 - PLANNED)
# =============================================================================


class TDDConfig(BaseModel):
    """
    Configuration for Test-Driven Development support.

    Enables test-first workflow for code generation tasks.
    This will be activated in Phase 6.2.

    Examples:
        >>> config = TDDConfig(enabled=True)
        >>> config.default_framework
        'pytest'
    """

    enabled: bool = Field(
        default=False, description="Enable TDD workflow (will be true after Phase 6.2)"
    )
    default_framework: Literal["pytest", "unittest", "jest"] = Field(
        default="pytest", description="Default test framework to use"
    )
    min_tests: int = Field(
        default=5, ge=1, le=100, description="Minimum number of tests to generate"
    )
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum TDD iteration cycles")


# =============================================================================
# ARBITER CONFIGURATION (Phase 6.3 - PLANNED)
# =============================================================================


class ArbiterScoringConfig(BaseModel):
    """Scoring criteria weights for multi-attempt arbiter."""

    correctness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for correctness criteria"
    )
    completeness: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for completeness criteria"
    )
    style: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Weight for style/quality criteria"
    )

    @field_validator("style")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:
        """Validate that all weights sum to 1.0."""
        correctness = info.data.get("correctness", 0.5)
        completeness = info.data.get("completeness", 0.3)
        total = correctness + completeness + v
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")
        return v


class ArbiterConfig(BaseModel):
    """
    Configuration for multi-attempt arbiter.

    Enables generating multiple solution candidates and selecting the best.
    This will be activated in Phase 6.3.

    Examples:
        >>> config = ArbiterConfig(enabled=True)
        >>> config.num_candidates
        3
        >>> config.temperatures
        [0.3, 0.7, 1.0]
    """

    enabled: bool = Field(
        default=False, description="Enable multi-attempt generation (will be true after Phase 6.3)"
    )
    num_candidates: int = Field(
        default=3, ge=2, le=10, description="Number of candidates to generate"
    )
    temperatures: list[float] = Field(
        default=[0.3, 0.7, 1.0],
        min_length=1,
        max_length=10,
        description="Temperature values for candidate generation",
    )
    use_llm_arbiter: bool = Field(
        default=True, description="Use LLM-based arbiter for selection vs simple heuristics"
    )
    scoring: ArbiterScoringConfig = Field(
        default_factory=ArbiterScoringConfig, description="Scoring criteria weights"
    )

    @field_validator("temperatures")
    @classmethod
    def validate_temperatures(cls, v: list[float]) -> list[float]:
        """Validate temperature values are in valid range."""
        for temp in v:
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"Temperature {temp} outside valid range [0.0, 2.0]")
        return v

    @field_validator("num_candidates")
    @classmethod
    def validate_candidate_count(cls, v: int, info) -> int:
        """Validate num_candidates matches temperatures list length."""
        temps = info.data.get("temperatures", [])
        if temps and len(temps) != v:
            raise ValueError(
                f"num_candidates ({v}) must match temperatures list length ({len(temps)})"
            )
        return v


# =============================================================================
# ROOT DARWIN CONFIGURATION
# =============================================================================


class DarwinConfig(BaseModel):
    """
    Root configuration for Darwin agent enhancements.

    This is the top-level configuration for all Darwin agent capabilities,
    including failure learning, TDD support, and multi-attempt arbiter.

    It corresponds to the [agents.darwin] section in TOML configuration.

    Usage:
        >>> config = DarwinConfig()
        >>> config.enabled
        True
        >>> config.failure_learning.backend
        'sqlite'

        >>> # Custom configuration
        >>> config = DarwinConfig(
        ...     enabled=True,
        ...     failure_learning=FailureLearningConfig(
        ...         backend="postgres",
        ...         db_url="postgresql://..."
        ...     )
        ... )
    """

    enabled: bool = Field(default=True, description="Master switch for all Darwin enhancements")
    failure_learning: FailureLearningConfig = Field(
        default_factory=FailureLearningConfig,
        description="Failure learning system settings (Phase 6.1)",
    )
    tdd: TDDConfig = Field(
        default_factory=TDDConfig, description="TDD support settings (Phase 6.2 - planned)"
    )
    arbiter: ArbiterConfig = Field(
        default_factory=ArbiterConfig,
        description="Multi-attempt arbiter settings (Phase 6.3 - planned)",
    )


# =============================================================================
# ENVIRONMENT VARIABLE LOADING
# =============================================================================


def load_darwin_config_from_env() -> DarwinConfig:
    """
    Load Darwin configuration from environment variables.

    Environment variables follow the pattern:
        LLMCORE_AGENTS__DARWIN__<SECTION>__<KEY>

    Examples:
        export LLMCORE_AGENTS__DARWIN__ENABLED=true
        export LLMCORE_AGENTS__DARWIN__FAILURE_LEARNING__ENABLED=true
        export LLMCORE_AGENTS__DARWIN__FAILURE_LEARNING__BACKEND=postgres
        export LLMCORE_AGENTS__DARWIN__FAILURE_LEARNING__DB_URL="postgresql://..."

    Returns:
        DarwinConfig instance with environment variable overrides applied
    """
    # Start with defaults
    config_dict = {}

    # Parse environment variables
    prefix = "LLMCORE_AGENTS__DARWIN__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix and split into path
        path = key[len(prefix) :].lower().split("__")

        # Build nested dict
        current = config_dict
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value (attempt type conversion)
        final_key = path[-1]
        if value.lower() in ("true", "false"):
            current[final_key] = value.lower() == "true"
        elif value.isdigit():
            current[final_key] = int(value)
        else:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value

    return DarwinConfig(**config_dict)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ArbiterConfig",
    "ArbiterScoringConfig",
    "DarwinConfig",
    "FailureLearningConfig",
    "FailureLearningPostgresConfig",
    "TDDConfig",
    "load_darwin_config_from_env",
]
