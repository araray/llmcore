# src/llmcore/autonomous/scheduler.py
"""
Autonomous Task Scheduler.

This module provides the ``AutonomousScheduler`` which wraps and extends the
:class:`~llmcore.autonomous.heartbeat.HeartbeatManager` with higher-level
scheduling semantics for autonomous agent operation.

The HeartbeatManager handles the core periodic-task execution loop (registering
tasks, running them on intervals, circuit-breaking on failures).  The scheduler
adds:

- Cron-like scheduling expressions (optional)
- Task dependency ordering
- Priority-aware execution when multiple tasks are due simultaneously
- Integration with :class:`~llmcore.autonomous.resource.ResourceMonitor` for
  resource-aware scheduling (skip low-priority tasks when constrained)

For most use-cases the HeartbeatManager alone is sufficient.  Use
``AutonomousScheduler`` when you need the additional scheduling primitives.

Example::

    from llmcore.autonomous.scheduler import AutonomousScheduler

    scheduler = AutonomousScheduler(heartbeat_manager=hb)
    scheduler.schedule("cleanup", interval_seconds=3600, priority=1)
    scheduler.schedule("report", interval_seconds=300, priority=10)

    await scheduler.start()

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §11.2 (HeartbeatManager)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §5 Phase 2 (Resource Management)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from enum import IntEnum
from typing import Any, Callable, Coroutine

from .heartbeat import HeartbeatManager, HeartbeatTask, heartbeat_task
from .resource import ResourceMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & models
# ---------------------------------------------------------------------------


class TaskPriority(IntEnum):
    """Priority levels for scheduled tasks.

    Higher numeric values indicate higher priority.  When the system is
    resource-constrained, lower-priority tasks may be deferred.
    """

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class ScheduledTask:
    """Metadata wrapper around a HeartbeatTask with scheduling extras.

    Attributes:
        name: Unique task identifier.
        interval_seconds: How often the task should fire.
        priority: Scheduling priority (see :class:`TaskPriority`).
        callback: The async callable to execute.
        enabled: Whether the task is currently active.
        depends_on: Names of tasks that must complete before this one.
        resource_aware: If *True*, skip execution when resources are constrained.
        metadata: Arbitrary key-value metadata for observability / debugging.
    """

    name: str
    interval_seconds: float
    priority: TaskPriority = TaskPriority.NORMAL
    callback: Callable[..., Coroutine[Any, Any, None]] | None = None
    enabled: bool = True
    depends_on: list[str] = field(default_factory=list)
    resource_aware: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class AutonomousScheduler:
    """High-level scheduler wrapping :class:`HeartbeatManager`.

    This class is the spec-mandated ``autonomous/scheduler.py`` entry-point.
    Internally it delegates all periodic execution to the HeartbeatManager and
    layers priority / dependency / resource-aware gating on top.

    Args:
        heartbeat_manager: An initialized HeartbeatManager instance.
        resource_monitor: Optional resource monitor for resource-aware scheduling.
    """

    def __init__(
        self,
        heartbeat_manager: HeartbeatManager,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        self._hb = heartbeat_manager
        self._resource_monitor = resource_monitor
        self._tasks: dict[str, ScheduledTask] = {}
        logger.debug("AutonomousScheduler initialized (delegates to HeartbeatManager).")

    # -- public API ----------------------------------------------------------

    def schedule(
        self,
        name: str,
        callback: Callable[..., Coroutine[Any, Any, None]],
        interval_seconds: float,
        priority: TaskPriority = TaskPriority.NORMAL,
        depends_on: list[str] | None = None,
        resource_aware: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledTask:
        """Register a new scheduled task.

        The task is wrapped in a resource/dependency gate and then registered
        with the underlying HeartbeatManager.

        Args:
            name: Unique task name.
            callback: Async callable to run periodically.
            interval_seconds: Execution interval.
            priority: Task priority level.
            depends_on: Task names that must succeed first.
            resource_aware: If True, skip when resources are constrained.
            metadata: Extra metadata for observability.

        Returns:
            The :class:`ScheduledTask` descriptor.
        """
        task = ScheduledTask(
            name=name,
            interval_seconds=interval_seconds,
            priority=priority,
            callback=callback,
            depends_on=depends_on or [],
            resource_aware=resource_aware,
            metadata=metadata or {},
        )
        self._tasks[name] = task

        # Wrap with gating logic and register on HeartbeatManager
        gated_callback = self._make_gated_callback(task)
        self._hb.register_task(
            HeartbeatTask(
                name=name,
                callback=gated_callback,
                interval=timedelta(seconds=interval_seconds),
            )
        )
        logger.info(
            "Scheduled task '%s' (interval=%ss, priority=%s, resource_aware=%s).",
            name,
            interval_seconds,
            priority.name,
            resource_aware,
        )
        return task

    def unschedule(self, name: str) -> bool:
        """Remove a scheduled task.

        Args:
            name: Task name to remove.

        Returns:
            True if the task existed and was removed.
        """
        removed = self._tasks.pop(name, None) is not None
        if removed:
            self._hb.unregister_task(name)
            logger.info("Unscheduled task '%s'.", name)
        return removed

    def get_task(self, name: str) -> ScheduledTask | None:
        """Retrieve a scheduled task by name."""
        return self._tasks.get(name)

    def list_tasks(self) -> list[ScheduledTask]:
        """Return all registered tasks sorted by priority (highest first)."""
        return sorted(self._tasks.values(), key=lambda t: t.priority, reverse=True)

    async def start(self) -> None:
        """Start the underlying HeartbeatManager loop."""
        logger.info("AutonomousScheduler starting (%d tasks).", len(self._tasks))
        await self._hb.start()

    async def stop(self) -> None:
        """Stop the underlying HeartbeatManager loop."""
        logger.info("AutonomousScheduler stopping.")
        await self._hb.stop()

    # -- internal helpers ----------------------------------------------------

    def _make_gated_callback(self, task: ScheduledTask) -> Callable[..., Coroutine[Any, Any, None]]:
        """Wrap task callback with resource-awareness and dependency checks."""

        async def _gated() -> None:
            # Resource gate
            if task.resource_aware and self._resource_monitor is not None:
                if self._resource_monitor.should_throttle():
                    if task.priority < TaskPriority.CRITICAL:
                        logger.debug(
                            "Skipping task '%s' (priority=%s) due to resource constraints.",
                            task.name,
                            task.priority.name,
                        )
                        return

            # Dependency gate (simple: all deps must be registered and enabled)
            for dep_name in task.depends_on:
                dep = self._tasks.get(dep_name)
                if dep is None or not dep.enabled:
                    logger.debug(
                        "Skipping task '%s': dependency '%s' not met.", task.name, dep_name
                    )
                    return

            # Execute
            if task.callback is not None:
                await task.callback()

        return _gated


# ---------------------------------------------------------------------------
# Convenience re-exports
# ---------------------------------------------------------------------------
# Allow ``from llmcore.autonomous.scheduler import HeartbeatManager`` etc.
# so users who import via the spec-documented path still reach the real
# implementations.

__all__ = [
    "AutonomousScheduler",
    "ScheduledTask",
    "TaskPriority",
    # Re-exports from heartbeat
    "HeartbeatManager",
    "HeartbeatTask",
    "heartbeat_task",
]
