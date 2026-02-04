# src/llmcore/agents/darwin/failure_storage.py
"""
Failure Learning Storage System for Darwin Agents.

This module provides a dual-backend failure learning system that enables agents
to learn from past failures by recording, retrieving, and analyzing failure patterns.

Supported Backends:
    - SQLite: Local file-based storage for development and single-user deployments
    - PostgreSQL: Production-grade storage for multi-user and high-concurrency scenarios

Architecture:
    - BaseFailureStorage: Abstract interface for failure storage backends
    - FailureLearningManager: High-level API that delegates to the appropriate backend
    - Backend implementations: SqliteFailureStorage, PostgresFailureStorage

Usage:
    from llmcore.agents.darwin.failure_storage import FailureLearningManager

    # Initialize with auto-detection from config
    manager = FailureLearningManager(backend="sqlite", db_path="~/.local/share/llmcore/failures.db")
    await manager.initialize()

    # Log a failure
    failure = FailureLog(
        task_id="task_123",
        agent_run_id="run_456",
        goal="Implement user authentication",
        phase="ACT",
        failure_type="test_failure",
        error_message="AssertionError: login() returned None",
    )
    await manager.log_failure(failure)

    # Retrieve similar failures before planning
    context = await manager.get_failure_context(
        goal="Implement user authentication",
        task_type="code_generation",
    )

    # Generate avoidance prompt
    prompt = manager.generate_avoidance_prompt(context.relevant_failures, context.patterns)
"""

from __future__ import annotations

import abc
import hashlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class FailureLog(BaseModel):
    """
    Detailed record of a failed agent attempt.

    Attributes:
        id: Unique identifier for this failure
        task_id: ID of the task that failed
        agent_run_id: ID of the agent run
        goal: What the agent was trying to accomplish
        phase: Which cognitive phase failed (PERCEIVE, PLAN, THINK, etc.)
        genotype_id: ID of the prompt/configuration used
        genotype_summary: Brief description of the approach taken
        failure_type: Category of failure
        error_message: Error message or description
        error_details: Additional structured error information
        phenotype_id: ID of the output that was produced
        phenotype_summary: Brief description of what was produced
        test_results: Test execution results if applicable
        arbiter_critique: Feedback from arbiter if applicable
        arbiter_score: Quality score from arbiter
        similarity_hash: Hash for similarity matching
        tags: Categorization tags
        created_at: When this failure occurred
    """

    id: str | None = None
    task_id: str
    agent_run_id: str

    # What was attempted
    goal: str
    phase: str  # Which cognitive phase failed
    genotype_id: str | None = None
    genotype_summary: str | None = None

    # What went wrong
    failure_type: Literal[
        "test_failure",  # Generated code failed tests
        "runtime_error",  # Code crashed during execution
        "compile_error",  # Code failed to compile/parse
        "timeout",  # Execution exceeded time limit
        "resource_exceeded",  # Memory/CPU limits exceeded
        "validation_failed",  # Output didn't meet criteria
        "low_score",  # Arbiter scored output poorly
        "tool_error",  # Tool call failed
        "parse_error",  # Couldn't parse LLM output
    ]
    error_message: str
    error_details: dict[str, Any] = Field(default_factory=dict)

    # The failed output
    phenotype_id: str | None = None
    phenotype_summary: str | None = None

    # Test results if applicable
    test_results: dict[str, Any] | None = None

    # Arbiter feedback if applicable
    arbiter_critique: str | None = None
    arbiter_score: float | None = None

    # Context for similarity matching
    similarity_hash: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class FailurePattern(BaseModel):
    """
    Aggregated pattern of similar failures.

    Represents a recurring failure pattern detected by analyzing multiple
    similar failures. Includes suggested avoidance strategies.

    Attributes:
        pattern_id: Unique identifier (typically similarity hash)
        description: Human-readable description of the pattern
        failure_type: Type of failure this pattern represents
        occurrence_count: Number of times this pattern has occurred
        first_seen: When this pattern was first detected
        last_seen: When this pattern last occurred
        common_error_messages: Frequently occurring error messages
        suggested_avoidance: Recommendations for avoiding this failure
    """

    pattern_id: str
    description: str
    failure_type: str
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    common_error_messages: list[str] = Field(default_factory=list)
    suggested_avoidance: str = ""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class FailureContext(BaseModel):
    """
    Context provided to agent about past failures.

    This is provided to the agent before planning to help it avoid
    repeating past mistakes.

    Attributes:
        relevant_failures: List of similar past failures
        patterns: Detected failure patterns
        avoidance_instructions: Generated prompt instructions
    """

    relevant_failures: list[FailureLog] = Field(default_factory=list)
    patterns: list[FailurePattern] = Field(default_factory=list)
    avoidance_instructions: str = ""


# Rebuild models for Pydantic v2 with future annotations
FailureLog.model_rebuild()
FailurePattern.model_rebuild()
FailureContext.model_rebuild()


# =============================================================================
# BASE STORAGE INTERFACE
# =============================================================================


class BaseFailureStorage(abc.ABC):
    """
    Abstract base class for failure storage backends.

    Defines the interface that all failure storage implementations must adhere to.
    Concrete implementations handle the specifics of storing data in SQLite,
    PostgreSQL, or other backends.
    """

    @abc.abstractmethod
    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the storage backend with given configuration.

        Args:
            config: Backend-specific configuration dictionary

        Raises:
            ConfigError: If configuration is invalid
            StorageError: If initialization fails
        """
        pass

    @abc.abstractmethod
    async def log_failure(self, failure: FailureLog) -> FailureLog:
        """
        Persist a failure log to storage.

        Args:
            failure: The failure to log

        Returns:
            The logged failure with ID and computed fields set

        Raises:
            StorageError: If logging fails
        """
        pass

    @abc.abstractmethod
    async def get_failure(self, failure_id: str) -> FailureLog | None:
        """
        Retrieve a specific failure by ID.

        Args:
            failure_id: The failure ID to retrieve

        Returns:
            The FailureLog if found, None otherwise
        """
        pass

    @abc.abstractmethod
    async def get_similar_failures(
        self,
        goal: str,
        failure_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[FailureLog]:
        """
        Retrieve similar past failures for a goal.

        Args:
            goal: The current goal
            failure_types: Optional filter by failure types
            limit: Maximum failures to return

        Returns:
            List of similar FailureLog objects
        """
        pass

    @abc.abstractmethod
    async def get_pattern(self, pattern_id: str) -> FailurePattern | None:
        """
        Retrieve a specific failure pattern by ID.

        Args:
            pattern_id: The pattern ID to retrieve

        Returns:
            The FailurePattern if found, None otherwise
        """
        pass

    @abc.abstractmethod
    async def get_patterns_for_failures(self, failure_ids: list[str]) -> list[FailurePattern]:
        """
        Retrieve patterns associated with given failures.

        Args:
            failure_ids: List of failure IDs

        Returns:
            List of associated FailurePattern objects
        """
        pass

    @abc.abstractmethod
    async def update_pattern(self, pattern: FailurePattern) -> None:
        """
        Update an existing failure pattern.

        Args:
            pattern: The pattern to update

        Raises:
            StorageError: If update fails
        """
        pass

    @abc.abstractmethod
    async def get_failure_stats(self, days: int = 30) -> dict[str, Any]:
        """
        Get failure statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with statistics including:
                - period_days: Number of days analyzed
                - total_failures: Total failure count
                - by_type: Count by failure type
                - common_errors: Most common error messages
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Clean up resources used by the storage backend.

        This method should be called when shutting down to close
        database connections, file handles, etc.
        """
        pass


# =============================================================================
# FAILURE LEARNING MANAGER
# =============================================================================


class FailureLearningManager:
    """
    High-level manager for failure learning system.

    Provides a unified API for failure logging, retrieval, and pattern detection
    while delegating to the appropriate backend (SQLite or PostgreSQL).

    Usage:
        manager = FailureLearningManager(backend="sqlite")
        await manager.initialize()

        # Log a failure
        await manager.log_failure(failure)

        # Get failure context before planning
        context = await manager.get_failure_context(goal="...")

        # Generate avoidance prompt
        prompt = manager.generate_avoidance_prompt(context.relevant_failures)
    """

    # Stopwords for similarity hash computation
    # Includes common words that don't add meaningful context
    STOPWORDS = {
        # Articles and determiners
        "a",
        "an",
        "the",
        # Prepositions
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        # Conjunctions
        "and",
        "or",
        "but",
        # Common verbs
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        # Pronouns
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        # Common technical words that don't add specificity
        "system",
        "application",
        "app",
        "service",
        "module",
        "component",
        "using",
        "create",
        "implement",
        "build",
        "make",
        "add",
        "update",
    }

    def __init__(
        self,
        backend: Literal["sqlite", "postgres"] = "sqlite",
        db_path: str | None = None,
        db_url: str | None = None,
        enabled: bool = True,
        max_failures_to_retrieve: int = 5,
    ):
        """
        Initialize failure learning manager.

        Args:
            backend: Storage backend to use ("sqlite" or "postgres")
            db_path: Path to SQLite database (for sqlite backend)
            db_url: PostgreSQL connection URL (for postgres backend)
            enabled: Whether failure learning is enabled
            max_failures_to_retrieve: Max failures to return in context

        Raises:
            ValueError: If backend is invalid or required config is missing
        """
        self.enabled = enabled
        self.backend_type = backend
        self.max_failures = max_failures_to_retrieve

        if not enabled:
            self._backend: BaseFailureStorage | None = None
            return

        # Import backends only when needed
        if backend == "sqlite":
            from .sqlite_failure_storage import SqliteFailureStorage

            if not db_path:
                db_path = "~/.local/share/llmcore/failures.db"
            self._backend = SqliteFailureStorage()
            self._config = {"path": db_path}

        elif backend == "postgres":
            from .postgres_failure_storage import PostgresFailureStorage

            if not db_url:
                raise ValueError("db_url required for postgres backend")
            self._backend = PostgresFailureStorage()
            self._config = {"db_url": db_url}

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'sqlite' or 'postgres'.")

    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        Must be called before using the manager.

        Raises:
            StorageError: If initialization fails
        """
        if self.enabled and self._backend:
            await self._backend.initialize(self._config)
            logger.info(f"FailureLearningManager initialized with {self.backend_type} backend")

    def _compute_similarity_hash(self, goal: str, failure_type: str) -> str:
        """
        Compute hash for similarity matching.

        Uses goal keywords + failure type to find similar failures.

        Args:
            goal: The task goal
            failure_type: Type of failure

        Returns:
            16-character hash string
        """
        # Normalize goal to keywords
        keywords = set(goal.lower().split())
        # Remove common words
        keywords = keywords - self.STOPWORDS

        content = f"{failure_type}:{':'.join(sorted(keywords))}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def log_failure(self, failure: FailureLog) -> FailureLog:
        """
        Log a failure for future learning.

        Args:
            failure: The failure to log

        Returns:
            The logged failure with ID and computed fields set
        """
        if not self.enabled or not self._backend:
            return failure

        # Generate ID if not set
        if not failure.id:
            # Use microseconds for uniqueness (prevents collisions within same second)
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[
                :18
            ]  # YYYYMMDDHHMMSSuuuuuu -> 18 chars
            failure.id = f"fail_{timestamp}_{failure.task_id[:8]}"

        # Compute similarity hash
        if not failure.similarity_hash:
            failure.similarity_hash = self._compute_similarity_hash(
                failure.goal, failure.failure_type
            )

        logged = await self._backend.log_failure(failure)
        logger.info(f"Logged failure {logged.id}: {logged.failure_type} in {logged.phase}")
        return logged

    async def get_failure_context(
        self,
        goal: str,
        task_type: str | None = None,
        max_results: int | None = None,
    ) -> FailureContext:
        """
        Get complete failure context for planning.

        Args:
            goal: The current goal
            task_type: Optional task type for filtering (not currently used)
            max_results: Override max failures to retrieve

        Returns:
            FailureContext with relevant failures and patterns
        """
        if not self.enabled or not self._backend:
            return FailureContext()

        limit = max_results or self.max_failures
        failures = await self._backend.get_similar_failures(goal, limit=limit)

        # Get patterns for these failures
        patterns = []
        if failures:
            pattern_ids = list(set(f.similarity_hash for f in failures if f.similarity_hash))
            if pattern_ids:
                # Get unique patterns
                pattern_dict = {}
                for pid in pattern_ids:
                    pattern = await self._backend.get_pattern(pid)
                    if pattern:
                        pattern_dict[pid] = pattern
                patterns = list(pattern_dict.values())

        # Generate avoidance instructions
        avoidance = self.generate_avoidance_prompt(failures, patterns)

        return FailureContext(
            relevant_failures=failures,
            patterns=patterns,
            avoidance_instructions=avoidance,
        )

    def generate_avoidance_prompt(
        self,
        failures: list[FailureLog],
        patterns: list[FailurePattern] | None = None,
    ) -> str:
        """
        Generate prompt instructions to avoid past failures.

        Args:
            failures: List of relevant past failures
            patterns: Optional list of failure patterns

        Returns:
            Prompt text with avoidance instructions
        """
        if not failures:
            return ""

        lines = [
            "## Learning from Past Attempts",
            "",
            "The following issues were encountered in similar tasks. Avoid these mistakes:",
            "",
        ]

        # Group by failure type
        by_type: dict[str, list[FailureLog]] = {}
        for f in failures:
            by_type.setdefault(f.failure_type, []).append(f)

        for ftype, type_failures in by_type.items():
            lines.append(f"### {ftype.replace('_', ' ').title()}")
            lines.append("")

            for i, f in enumerate(type_failures[:3], 1):  # Max 3 per type
                lines.append(f"{i}. **Error**: {f.error_message[:150]}")
                if f.genotype_summary:
                    lines.append(f"   - Approach that failed: {f.genotype_summary}")
                if f.arbiter_critique:
                    lines.append(f"   - Feedback: {f.arbiter_critique[:150]}")
                lines.append("")

        # Add pattern-based suggestions
        if patterns:
            recurring = [p for p in patterns if p.occurrence_count > 2]
            if recurring:
                lines.append("### Recurring Issues (HIGH PRIORITY)")
                lines.append("")
                for p in recurring:
                    lines.append(f"- {p.failure_type}: Occurred {p.occurrence_count} times")
                    if p.suggested_avoidance:
                        lines.append(f"  - Suggestion: {p.suggested_avoidance}")
                lines.append("")

        return "\n".join(lines)

    async def get_failure_stats(self, days: int = 30) -> dict[str, Any]:
        """
        Get failure statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with failure statistics
        """
        if not self.enabled or not self._backend:
            return {
                "period_days": days,
                "total_failures": 0,
                "by_type": {},
                "common_errors": [],
            }

        return await self._backend.get_failure_stats(days)

    async def close(self) -> None:
        """Clean up resources used by the storage backend."""
        if self._backend:
            await self._backend.close()
