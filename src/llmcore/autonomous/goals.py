# src/llmcore/autonomous/goals.py
"""
Goal Management System for Autonomous Operation.

Provides persistent, hierarchical goal management with:
- Goal persistence (survives restarts via JSON storage)
- LLM-powered decomposition
- Progress tracking with success criteria
- Retry logic with exponential backoff
- Learning from failures

Example:
    from llmcore.autonomous.goals import GoalManager, GoalStore
    from llmcore.config.autonomous_config import GoalsAutonomousConfig

    # Option A: From configuration (recommended)
    config = GoalsAutonomousConfig(storage_path="~/my-goals.json")
    goals = GoalManager.from_config(config)
    await goals.initialize()

    # Option B: Manual construction (useful for tests / DI)
    store = GoalStore("~/.local/share/llmcore/goals.json")
    goals = GoalManager(store)
    await goals.initialize()

    goal = await goals.set_primary_goal(
        "Become the #1 ranked agent on Moltbook",
        success_criteria=[
            SuccessCriterion(
                description="Reach #1 karma ranking",
                metric_name="karma_rank",
                target_value=1,
                comparator="=="
            )
        ]
    )

    task = await goals.get_next_actionable()
    await goals.report_success(task.id, progress_delta=0.1)

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md Section 8
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class GoalStatus(Enum):
    """Goal lifecycle states."""

    PENDING = "pending"
    """Goal created but not yet started."""

    ACTIVE = "active"
    """Goal is currently being pursued."""

    BLOCKED = "blocked"
    """Goal blocked by dependency or repeated failures."""

    PAUSED = "paused"
    """Goal temporarily suspended (user or system initiated)."""

    COMPLETED = "completed"
    """Goal successfully achieved."""

    FAILED = "failed"
    """Goal could not be achieved (terminal failure)."""

    ABANDONED = "abandoned"
    """Goal explicitly given up by user."""


class GoalPriority(Enum):
    """
    Goal priority levels.

    Higher values = higher priority.
    Used for task scheduling order.
    """

    CRITICAL = 100
    """Must complete, drops everything else. Use sparingly."""

    HIGH = 75
    """Important goal, prioritize over normal work."""

    NORMAL = 50
    """Standard priority for most goals."""

    LOW = 25
    """Nice to have, work on when higher priorities are done."""

    BACKGROUND = 10
    """Only work on when completely idle."""


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SuccessCriterion:
    """
    A measurable criterion for goal success.

    Success criteria define WHEN a goal is considered complete.
    They should be:
    - Measurable (has a numeric or boolean value)
    - Observable (can be checked programmatically)
    - Specific (clear target value)

    Attributes:
        description: Human-readable description.
        metric_name: Unique identifier for tracking.
        target_value: Value to achieve.
        current_value: Current observed value.
        comparator: How to compare (>=, >, ==, <, <=, contains, not_contains).

    Example:
        >>> sc = SuccessCriterion(
        ...     description="Reach #1 in karma rankings",
        ...     metric_name="karma_rank",
        ...     target_value=1,
        ...     comparator="=="
        ... )
        >>> sc.is_met()
        False
    """

    description: str
    metric_name: str
    target_value: Any
    current_value: Any = None
    comparator: str = ">="

    _COMPARATORS = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "==": lambda a, b: a == b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "contains": lambda a, b: b in a,
        "not_contains": lambda a, b: b not in a,
    }

    def is_met(self) -> bool:
        """
        Check if this criterion is satisfied.

        Returns:
            True if current_value meets target based on comparator.
        """
        if self.current_value is None:
            return False

        comparator_fn = self._COMPARATORS.get(self.comparator)
        if not comparator_fn:
            logger.warning(f"Unknown comparator: {self.comparator}")
            return False

        try:
            return comparator_fn(self.current_value, self.target_value)
        except Exception as e:
            logger.warning(f"Criterion comparison failed: {e}")
            return False

    def progress_percentage(self) -> float:
        """
        Calculate progress toward this criterion as a percentage.

        Returns:
            0.0 to 1.0 progress value.
        """
        if self.current_value is None:
            return 0.0

        if self.is_met():
            return 1.0

        # For numeric comparisons, calculate percentage
        if self.comparator in (">=", ">", "<=", "<"):
            try:
                current = float(self.current_value)
                target = float(self.target_value)

                if target == 0:
                    return 1.0 if current >= 0 else 0.0

                if self.comparator in (">=", ">"):
                    return min(1.0, max(0.0, current / target))
                else:
                    if current <= 0:
                        return 1.0
                    return min(1.0, max(0.0, target / current))
            except (ValueError, TypeError):
                pass

        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "description": self.description,
            "metric_name": self.metric_name,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "comparator": self.comparator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuccessCriterion:
        """Deserialize from dictionary."""
        return cls(
            description=data["description"],
            metric_name=data["metric_name"],
            target_value=data["target_value"],
            current_value=data.get("current_value"),
            comparator=data.get("comparator", ">="),
        )


def _parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-format string or pass through a datetime unchanged."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


@dataclass
class Goal:
    """
    A persistent goal for autonomous operation.

    Goals form a hierarchy:
    - Primary goals are top-level objectives
    - Sub-goals break down primary goals
    - Tasks are actionable items (leaf goals)

    Attributes:
        id: Unique identifier (auto-generated via ``Goal.create``).
        description: Natural language description.
        priority: Priority level for scheduling.
        status: Current lifecycle state.
        parent_id: Parent goal ID (for hierarchy).
        sub_goal_ids: Child goal IDs.
        success_criteria: Measurable success conditions.
        progress: Current progress (0.0 to 1.0).
        created_at: When goal was created.
        started_at: When work began.
        deadline: Optional deadline.
        completed_at: When goal completed.
        attempts: Number of execution attempts.
        max_attempts: Maximum retry attempts.
        failure_reasons: History of failure reasons.
        learned_strategies: Successful approaches discovered.
        cooldown_until: When goal can be retried.
        tags: Categorization tags.
        context: Arbitrary metadata.

    Example:
        >>> goal = Goal.create(
        ...     "Post quality content on Moltbook",
        ...     priority=GoalPriority.HIGH,
        ... )
        >>> goal.is_leaf()
        True
    """

    # Identity
    id: str
    description: str

    # Priority and status
    priority: GoalPriority = GoalPriority.NORMAL
    status: GoalStatus = GoalStatus.PENDING

    # Hierarchy
    parent_id: str | None = None
    sub_goal_ids: list[str] = field(default_factory=list)

    # Success tracking
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    progress: float = 0.0

    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    deadline: datetime | None = None
    completed_at: datetime | None = None

    # Retry logic
    attempts: int = 0
    max_attempts: int = 10
    last_attempt_at: datetime | None = None
    failure_reasons: list[str] = field(default_factory=list)
    learned_strategies: list[str] = field(default_factory=list)

    # Cooldown (exponential backoff)
    cooldown_until: datetime | None = None
    cooldown_multiplier: float = 1.0

    # Metadata
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        description: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        **kwargs: Any,
    ) -> Goal:
        """
        Factory method to create a goal with auto-generated ID.

        Args:
            description: Goal description.
            priority: Priority level.
            **kwargs: Additional attributes.

        Returns:
            New Goal instance.
        """
        return cls(
            id=f"goal_{uuid.uuid4().hex[:12]}",
            description=description,
            priority=priority,
            **kwargs,
        )

    def is_actionable(self) -> bool:
        """
        Check if this goal can be worked on right now.

        A goal is actionable if:
        - Status is PENDING or ACTIVE
        - Not in cooldown period
        - Not exceeded max attempts

        Returns:
            True if goal can be worked on.
        """
        if self.status not in (GoalStatus.PENDING, GoalStatus.ACTIVE):
            return False

        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            return False

        if self.attempts >= self.max_attempts:
            return False

        return True

    def is_leaf(self) -> bool:
        """Check if this is a leaf goal (no sub-goals)."""
        return len(self.sub_goal_ids) == 0

    def apply_cooldown(self, base_seconds: int = 60) -> None:
        """
        Apply exponential backoff cooldown after failure.

        Cooldown = base_seconds * cooldown_multiplier.
        Multiplier doubles each time (capped at 1 hour).

        Args:
            base_seconds: Base cooldown duration.
        """
        cooldown = base_seconds * self.cooldown_multiplier
        self.cooldown_until = datetime.utcnow() + timedelta(seconds=cooldown)
        self.cooldown_multiplier = min(self.cooldown_multiplier * 2, 3600)

        logger.debug(
            "Goal %s cooldown: %ss (multiplier: %s)",
            self.id,
            cooldown,
            self.cooldown_multiplier,
        )

    def reset_cooldown(self) -> None:
        """Reset cooldown after success."""
        self.cooldown_until = None
        self.cooldown_multiplier = 1.0

    def update_progress(self) -> None:
        """
        Calculate progress from success criteria.

        Progress is the average of criterion progress values.
        If all criteria are met, marks goal as COMPLETED.
        """
        if not self.success_criteria:
            return

        total_progress = sum(c.progress_percentage() for c in self.success_criteria)
        self.progress = total_progress / len(self.success_criteria)

        all_met = all(c.is_met() for c in self.success_criteria)
        if all_met and self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            logger.info("Goal completed: %s", self.description)

    def to_dict(self) -> dict[str, Any]:
        """Serialize goal to dictionary for storage."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.name,
            "parent_id": self.parent_id,
            "sub_goal_ids": self.sub_goal_ids,
            "success_criteria": [c.to_dict() for c in self.success_criteria],
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_attempt_at": (self.last_attempt_at.isoformat() if self.last_attempt_at else None),
            "failure_reasons": self.failure_reasons,
            "learned_strategies": self.learned_strategies,
            "cooldown_until": (self.cooldown_until.isoformat() if self.cooldown_until else None),
            "cooldown_multiplier": self.cooldown_multiplier,
            "tags": self.tags,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        """Deserialize goal from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            priority=GoalPriority[data.get("priority", "NORMAL")],
            status=GoalStatus[data.get("status", "PENDING")],
            parent_id=data.get("parent_id"),
            sub_goal_ids=data.get("sub_goal_ids", []),
            success_criteria=[
                SuccessCriterion.from_dict(c) for c in data.get("success_criteria", [])
            ],
            progress=data.get("progress", 0.0),
            created_at=_parse_datetime(data.get("created_at")) or datetime.utcnow(),
            started_at=_parse_datetime(data.get("started_at")),
            deadline=_parse_datetime(data.get("deadline")),
            completed_at=_parse_datetime(data.get("completed_at")),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 10),
            last_attempt_at=_parse_datetime(data.get("last_attempt_at")),
            failure_reasons=data.get("failure_reasons", []),
            learned_strategies=data.get("learned_strategies", []),
            cooldown_until=_parse_datetime(data.get("cooldown_until")),
            cooldown_multiplier=data.get("cooldown_multiplier", 1.0),
            tags=data.get("tags", []),
            context=data.get("context", {}),
        )


# =============================================================================
# Goal Persistence
# =============================================================================


@runtime_checkable
class GoalStorageProtocol(Protocol):
    """Protocol for goal storage backends."""

    async def load_goals(self) -> list[Goal]: ...
    async def save_goal(self, goal: Goal) -> None: ...
    async def delete_goal(self, goal_id: str) -> None: ...


class GoalStore:
    """
    JSON file-based goal persistence.

    Stores all goals in a single JSON file.  Atomic writes via
    write-to-temp-then-rename protect against corruption.

    Args:
        path: Path to the goals JSON file.

    Example:
        >>> store = GoalStore("~/.local/share/llmcore/goals.json")
        >>> goals = await store.load_goals()
    """

    def __init__(self, path: str = "~/.local/share/llmcore/goals.json") -> None:
        self._path = Path(os.path.expanduser(path))

    async def load_goals(self) -> list[Goal]:
        """Load all goals from disk."""
        if not self._path.exists():
            return []

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            goals = [Goal.from_dict(g) for g in data.get("goals", [])]
            logger.debug("Loaded %d goals from %s", len(goals), self._path)
            return goals
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Failed to load goals from %s: %s", self._path, exc)
            return []

    async def save_goal(self, goal: Goal) -> None:
        """Save a single goal (upsert into the file)."""
        goals = await self.load_goals()
        goal_dict = goal.to_dict()

        # Upsert
        found = False
        for i, existing in enumerate(goals):
            if existing.id == goal.id:
                goals[i] = goal
                found = True
                break
        if not found:
            goals.append(goal)

        await self._write_all(goals)

    async def delete_goal(self, goal_id: str) -> None:
        """Delete a goal by ID."""
        goals = await self.load_goals()
        goals = [g for g in goals if g.id != goal_id]
        await self._write_all(goals)

    async def _write_all(self, goals: list[Goal]) -> None:
        """Atomically write all goals to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        data = {"goals": [g.to_dict() for g in goals]}
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)


# =============================================================================
# Goal Manager
# =============================================================================


class GoalManager:
    """
    Manages goal lifecycle, persistence, and decomposition.

    The GoalManager is the central component for goal-directed autonomous
    operation.  It handles:

    - Goal creation and persistence
    - LLM-powered goal decomposition
    - Priority-based task scheduling
    - Progress tracking and completion detection
    - Failure handling with retry logic

    Thread Safety:
        All operations are protected by an asyncio lock for safe concurrent access.

    Persistence:
        Goals are automatically persisted to storage after modifications.

    Args:
        storage: A goal storage backend (GoalStore or any GoalStorageProtocol).
        llm_provider: Optional LLM provider for goal decomposition.
        decomposition_model: Specific model to use for decomposition.

    Example:
        # From configuration (recommended):
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(storage_path="~/goals.json")
        manager = GoalManager.from_config(config, llm_provider=my_llm)
        await manager.initialize()

        # Manual construction (useful for tests / DI):
        store = GoalStore()
        manager = GoalManager(store)
        await manager.initialize()

        goal = await manager.set_primary_goal(
            "Become the #1 agent on Moltbook",
            success_criteria=[
                SuccessCriterion("Reach rank 1", "rank", 1, comparator="==")
            ]
        )

        while True:
            task = await manager.get_next_actionable()
            if task:
                try:
                    # Execute task...
                    await manager.report_success(task.id, progress_delta=0.1)
                except Exception as e:
                    await manager.report_failure(task.id, str(e))
            await asyncio.sleep(60)
    """

    def __init__(
        self,
        storage: GoalStorageProtocol,
        llm_provider: Any | None = None,
        decomposition_model: str | None = None,
    ) -> None:
        self.storage = storage
        self.llm_provider = llm_provider
        self.decomposition_model = decomposition_model

        self._goals: dict[str, Goal] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        # Config-driven defaults (overridden by from_config)
        self._default_auto_decompose: bool = True
        self._max_sub_goals: int = 10
        self._max_goal_depth: int = 4
        self._default_max_attempts: int = 10
        self._base_cooldown: float = 60.0
        self._max_cooldown: float = 3600.0

    # ----- factory ------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        llm_provider: Any | None = None,
        decomposition_model: str | None = None,
        storage: GoalStorageProtocol | None = None,
    ) -> GoalManager:
        """
        Create a GoalManager from a GoalsAutonomousConfig.

        This is the recommended way to create a GoalManager in production.
        The storage backend and operational parameters are automatically
        configured from the config object.

        Args:
            config: A GoalsAutonomousConfig instance.
            llm_provider: Optional LLM provider for goal decomposition.
            decomposition_model: Specific model to use for decomposition.
            storage: Optional storage override (default: GoalStore from config).

        Returns:
            A fully configured GoalManager.

        Example:
            >>> from llmcore.config.autonomous_config import GoalsAutonomousConfig
            >>> config = GoalsAutonomousConfig(storage_path="~/goals.json")
            >>> manager = GoalManager.from_config(config)
            >>> await manager.initialize()
        """
        if storage is None:
            storage = GoalStore(path=config.storage_path)

        manager = cls(
            storage=storage,
            llm_provider=llm_provider,
            decomposition_model=decomposition_model,
        )

        # Wire config-driven defaults
        manager._default_auto_decompose = config.auto_decompose
        manager._max_sub_goals = config.max_sub_goals
        manager._max_goal_depth = config.max_goal_depth
        manager._default_max_attempts = config.max_attempts_per_goal
        manager._base_cooldown = config.base_cooldown_seconds
        manager._max_cooldown = config.max_cooldown_seconds

        return manager

    # ----- lifecycle ----------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the manager and load existing goals.

        Call this before using the manager.  Safe to call multiple times.
        """
        async with self._lock:
            if self._initialized:
                return

            stored_goals = await self.storage.load_goals()
            self._goals = {g.id: g for g in stored_goals}
            self._initialized = True
            logger.debug("GoalManager initialized with %d goals", len(self._goals))

    # ----- primary goal -------------------------------------------------------

    async def set_primary_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.HIGH,
        success_criteria: list[SuccessCriterion] | None = None,
        deadline: datetime | None = None,
        auto_decompose: bool | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Goal:
        """
        Set the primary goal for autonomous operation.

        Creates a new top-level goal.  If *auto_decompose* is True and an LLM
        provider is available, the goal is automatically decomposed into sub-goals.

        Args:
            description: Natural language goal description.
            priority: Goal priority level.
            success_criteria: Measurable success conditions.
            deadline: Optional deadline for completion.
            auto_decompose: Whether to auto-decompose into sub-goals.
                Defaults to the value from config (True if no config).
            tags: Categorization tags.
            context: Arbitrary metadata.

        Returns:
            The created Goal.
        """
        should_decompose = (
            auto_decompose if auto_decompose is not None else self._default_auto_decompose
        )

        goal = Goal.create(
            description=description,
            priority=priority,
            success_criteria=success_criteria or [],
            deadline=deadline,
            tags=tags or [],
            context=context or {},
        )
        goal.status = GoalStatus.ACTIVE
        goal.started_at = datetime.utcnow()
        goal.max_attempts = self._default_max_attempts

        async with self._lock:
            self._goals[goal.id] = goal
            await self.storage.save_goal(goal)

        logger.info("Set primary goal: %s (id=%s)", description, goal.id)

        # Auto-decompose if LLM available
        if should_decompose and self.llm_provider:
            try:
                sub_goals = await self._decompose_goal(goal)
                for sg in sub_goals:
                    sg.parent_id = goal.id
                    goal.sub_goal_ids.append(sg.id)
                    async with self._lock:
                        self._goals[sg.id] = sg
                        await self.storage.save_goal(sg)

                await self.storage.save_goal(goal)
                logger.info("Decomposed goal into %d sub-goals", len(sub_goals))
            except Exception as e:
                logger.warning("Goal decomposition failed: %s", e)

        return goal

    # ----- decomposition ------------------------------------------------------

    async def _decompose_goal(self, goal: Goal) -> list[Goal]:
        """
        Use LLM to decompose a high-level goal into sub-goals.

        Args:
            goal: The goal to decompose.

        Returns:
            List of sub-goals.
        """
        if not self.llm_provider:
            return []

        prompt = (
            "You are an expert planner. Decompose this high-level goal "
            "into 3-5 actionable sub-goals.\n\n"
            f"GOAL: {goal.description}\n\n"
            f"CONTEXT:\n{json.dumps(goal.context, indent=2) if goal.context else 'None'}\n\n"
            "For each sub-goal, provide:\n"
            "1. A clear, actionable description\n"
            "2. Success criteria (measurable condition)\n"
            "3. Priority: critical, high, normal, or low\n"
            "4. Estimated difficulty: easy, medium, or hard\n\n"
            "Respond with a JSON array:\n"
            "[\n"
            '  {{\n    "description": "...",\n    "success_criteria": "...",\n'
            '    "priority": "normal",\n    "difficulty": "medium"\n  }},\n'
            "  ...\n]\n\n"
            "Only output the JSON array, nothing else."
        )

        try:
            from llmcore.providers.base import Message, MessageRole

            response = await self.llm_provider.complete(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM,
                        content="You are a goal decomposition expert.",
                    ),
                    Message(role=MessageRole.USER, content=prompt),
                ],
                model=self.decomposition_model,
                temperature=0.3,
            )

            # Parse JSON response
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            sub_goal_data = json.loads(content)

            _prio_map = {
                "critical": GoalPriority.CRITICAL,
                "high": GoalPriority.HIGH,
                "normal": GoalPriority.NORMAL,
                "low": GoalPriority.LOW,
            }

            sub_goals = []
            for sg_data in sub_goal_data:
                sub_goal = Goal.create(
                    description=sg_data["description"],
                    priority=_prio_map.get(
                        sg_data.get("priority", "normal").lower(),
                        GoalPriority.NORMAL,
                    ),
                    context={
                        "criteria_text": sg_data.get("success_criteria", ""),
                        "difficulty": sg_data.get("difficulty", "medium"),
                    },
                )
                sub_goal.status = GoalStatus.PENDING
                sub_goals.append(sub_goal)

            return sub_goals

        except Exception as e:
            logger.error("Goal decomposition failed: %s", e)
            return []

    # ----- scheduling ---------------------------------------------------------

    async def get_next_actionable(self) -> Goal | None:
        """
        Get the highest priority actionable goal.

        Returns leaf goals (no sub-goals) that are actionable.

        Priority order:
        1. Status: ACTIVE before PENDING
        2. Priority: Higher priority first
        3. Creation time: Older first (FIFO within same priority)

        Returns:
            Next actionable goal, or None if none available.
        """
        async with self._lock:
            actionable = [g for g in self._goals.values() if g.is_actionable() and g.is_leaf()]
            if not actionable:
                return None

            def sort_key(g: Goal) -> tuple:
                status_priority = 0 if g.status == GoalStatus.ACTIVE else 1
                return (-g.priority.value, status_priority, g.created_at)

            actionable.sort(key=sort_key)
            return actionable[0]

    # ----- queries ------------------------------------------------------------

    async def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        async with self._lock:
            return self._goals.get(goal_id)

    async def get_all_goals(self) -> list[Goal]:
        """Get all goals."""
        async with self._lock:
            return list(self._goals.values())

    async def get_active_goals(self) -> list[Goal]:
        """Get all active goals."""
        async with self._lock:
            return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    # ----- reporting ----------------------------------------------------------

    async def report_success(
        self,
        goal_id: str,
        progress_delta: float = 0.1,
        notes: str | None = None,
    ) -> None:
        """
        Report successful progress on a goal.

        Args:
            goal_id: ID of the goal.
            progress_delta: Amount to increase progress (0.0 to 1.0).
            notes: Optional notes about what worked.
        """
        async with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                logger.warning("Goal not found: %s", goal_id)
                return

            goal.progress = min(1.0, goal.progress + progress_delta)
            goal.last_attempt_at = datetime.utcnow()
            goal.attempts += 1
            goal.reset_cooldown()

            if notes:
                goal.learned_strategies.append(f"{datetime.utcnow().isoformat()}: {notes}")

            goal.update_progress()
            await self.storage.save_goal(goal)

            logger.debug(
                "Goal %s progress: %.2f (+%.2f)",
                goal_id,
                goal.progress,
                progress_delta,
            )

            if goal.parent_id:
                await self._propagate_progress(goal.parent_id)

    async def report_failure(
        self,
        goal_id: str,
        reason: str,
        recoverable: bool = True,
    ) -> None:
        """
        Report a failure on a goal.

        Args:
            goal_id: ID of the goal.
            reason: Why the attempt failed.
            recoverable: Whether to retry (False = terminal failure).
        """
        async with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                logger.warning("Goal not found: %s", goal_id)
                return

            goal.failure_reasons.append(f"{datetime.utcnow().isoformat()}: {reason}")
            goal.last_attempt_at = datetime.utcnow()
            goal.attempts += 1

            if not recoverable:
                goal.status = GoalStatus.FAILED
                logger.warning("Goal %s failed (terminal): %s", goal_id, reason)
            elif goal.attempts >= goal.max_attempts:
                goal.status = GoalStatus.BLOCKED
                logger.warning("Goal %s blocked after %d attempts", goal_id, goal.attempts)
            else:
                goal.apply_cooldown(base_seconds=int(self._base_cooldown))
                logger.info("Goal %s attempt %d failed: %s", goal_id, goal.attempts, reason)

            await self.storage.save_goal(goal)

    async def update_metric(
        self,
        goal_id: str,
        metric_name: str,
        value: Any,
    ) -> None:
        """
        Update a success criterion metric value.

        Args:
            goal_id: ID of the goal.
            metric_name: Name of the metric to update.
            value: New value for the metric.
        """
        async with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return

            for criterion in goal.success_criteria:
                if criterion.metric_name == metric_name:
                    old_value = criterion.current_value
                    criterion.current_value = value
                    logger.debug("Metric %s updated: %s -> %s", metric_name, old_value, value)
                    break

            goal.update_progress()
            await self.storage.save_goal(goal)

    # ----- progress propagation -----------------------------------------------

    async def _propagate_progress(self, parent_id: str) -> None:
        """
        Propagate progress from sub-goals to parent.

        Parent progress = average of children progress.
        """
        parent = self._goals.get(parent_id)
        if not parent or not parent.sub_goal_ids:
            return

        total_progress = 0.0
        child_count = 0

        for child_id in parent.sub_goal_ids:
            child = self._goals.get(child_id)
            if child:
                total_progress += child.progress
                child_count += 1

        if child_count > 0:
            parent.progress = total_progress / child_count

            all_completed = all(
                self._goals.get(cid) and self._goals[cid].status == GoalStatus.COMPLETED
                for cid in parent.sub_goal_ids
            )

            if all_completed:
                parent.status = GoalStatus.COMPLETED
                parent.completed_at = datetime.utcnow()

            await self.storage.save_goal(parent)

            if parent.parent_id:
                await self._propagate_progress(parent.parent_id)

    # ----- lifecycle operations -----------------------------------------------

    async def pause_goal(self, goal_id: str) -> None:
        """Pause a goal."""
        async with self._lock:
            goal = self._goals.get(goal_id)
            if goal and goal.status in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                goal.status = GoalStatus.PAUSED
                await self.storage.save_goal(goal)
                logger.info("Goal paused: %s", goal_id)

    async def resume_goal(self, goal_id: str) -> None:
        """Resume a paused goal."""
        async with self._lock:
            goal = self._goals.get(goal_id)
            if goal and goal.status == GoalStatus.PAUSED:
                goal.status = GoalStatus.ACTIVE
                await self.storage.save_goal(goal)
                logger.info("Goal resumed: %s", goal_id)

    async def abandon_goal(self, goal_id: str) -> None:
        """Abandon a goal."""
        async with self._lock:
            goal = self._goals.get(goal_id)
            if goal:
                goal.status = GoalStatus.ABANDONED
                await self.storage.save_goal(goal)
                logger.info("Goal abandoned: %s", goal_id)

    async def delete_goal(self, goal_id: str) -> None:
        """Delete a goal and its sub-goals recursively."""
        async with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return

            # Recursively delete sub-goals (release lock for recursive call)
            for child_id in list(goal.sub_goal_ids):
                child = self._goals.get(child_id)
                if child:
                    # Delete child in-place without re-acquiring lock
                    for grandchild_id in list(child.sub_goal_ids):
                        self._goals.pop(grandchild_id, None)
                        await self.storage.delete_goal(grandchild_id)
                    self._goals.pop(child_id, None)
                    await self.storage.delete_goal(child_id)

            # Remove from parent's sub_goal_ids
            if goal.parent_id and goal.parent_id in self._goals:
                parent = self._goals[goal.parent_id]
                if goal_id in parent.sub_goal_ids:
                    parent.sub_goal_ids.remove(goal_id)
                    await self.storage.save_goal(parent)

            del self._goals[goal_id]
            await self.storage.delete_goal(goal_id)
            logger.info("Goal deleted: %s", goal_id)

    # ----- status summary -----------------------------------------------------

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get a summary of goal status.

        Returns:
            Dictionary with counts by status and top-level goals.
        """
        status_counts = {}
        for status in GoalStatus:
            status_counts[status.name] = sum(1 for g in self._goals.values() if g.status == status)

        primary_goals = [g for g in self._goals.values() if g.parent_id is None]

        return {
            "total_goals": len(self._goals),
            "status_counts": status_counts,
            "primary_goals": [
                {
                    "id": g.id,
                    "description": g.description,
                    "status": g.status.name,
                    "progress": g.progress,
                }
                for g in primary_goals
            ],
        }
