# src/llmcore/autonomous/heartbeat.py
"""
Heartbeat System for Periodic Task Scheduling.

Provides a lightweight, async-native scheduling system for autonomous operation.
Each registered task runs at its specified interval.

Features:
    - Fixed interval scheduling
    - Async callback support
    - Error handling with circuit breakers
    - Pause/resume capability
    - Task statistics

Example:
    heartbeat = HeartbeatManager(base_interval=timedelta(seconds=60))

    # Register tasks
    heartbeat.register(HeartbeatTask(
        name="moltbook_check",
        callback=check_moltbook_feed,
        interval=timedelta(hours=4),
        description="Check Moltbook feed and engage"
    ))

    heartbeat.register(HeartbeatTask(
        name="goal_review",
        callback=review_goals,
        interval=timedelta(hours=1),
        description="Review goal progress"
    ))

    # Start heartbeat loop
    await heartbeat.start()

    # Check status
    print(heartbeat.get_status())

    # Later...
    await heartbeat.stop()
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# HeartbeatTask
# =============================================================================


@dataclass
class HeartbeatTask:
    """
    A task to run periodically.

    Attributes:
        name: Unique task identifier
        callback: Async function to call
        interval: Time between runs
        enabled: Whether task is active
        last_run: When task last ran
        next_run: When task should run next
        run_count: Total successful runs
        error_count: Total errors
        consecutive_errors: Errors since last success
        max_consecutive_errors: Circuit breaker threshold
        last_error: Most recent error message
        description: Human-readable description
        tags: Categorization tags

    Example:
        task = HeartbeatTask(
            name="check_feed",
            callback=my_async_function,
            interval=timedelta(hours=4),
            description="Check social feed for new posts"
        )
    """

    name: str
    callback: Callable[[], Awaitable[Any]]
    interval: timedelta

    # State
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0

    # Error handling
    error_count: int = 0
    consecutive_errors: int = 0
    max_consecutive_errors: int = 5
    last_error: Optional[str] = None

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def should_run(self, now: datetime) -> bool:
        """
        Check if task should run now.

        A task should run if:
        - It's enabled
        - Not circuit-broken (too many consecutive errors)
        - Never run before OR next_run time has passed

        Args:
            now: Current timestamp

        Returns:
            True if task should execute
        """
        if not self.enabled:
            return False

        if self.consecutive_errors >= self.max_consecutive_errors:
            return False  # Circuit breaker open

        if self.next_run is None:
            return True  # Never run, run now

        return now >= self.next_run

    def schedule_next(self, now: datetime) -> None:
        """Schedule the next run time."""
        self.last_run = now
        self.next_run = now + self.interval
        self.run_count += 1

    def record_success(self) -> None:
        """Record successful execution (reset circuit breaker)."""
        self.consecutive_errors = 0
        self.last_error = None

    def record_error(self, error: str) -> None:
        """Record execution error."""
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error = error

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self.consecutive_errors = 0
        self.last_error = None
        logger.info(f"Circuit breaker reset for task: {self.name}")

    @property
    def is_circuit_broken(self) -> bool:
        """Check if circuit breaker is open."""
        return self.consecutive_errors >= self.max_consecutive_errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "interval_seconds": self.interval.total_seconds(),
            "enabled": self.enabled,
            "last_run": (
                self.last_run.isoformat() if self.last_run else None
            ),
            "next_run": (
                self.next_run.isoformat() if self.next_run else None
            ),
            "run_count": self.run_count,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "last_error": self.last_error,
            "is_circuit_broken": self.is_circuit_broken,
            "description": self.description,
            "tags": self.tags,
        }


# =============================================================================
# HeartbeatManager
# =============================================================================


class HeartbeatManager:
    """
    Manages periodic task execution.

    The HeartbeatManager is the "pulse" of autonomous operation, running
    registered tasks at their specified intervals.

    Features:
        - Async task execution
        - Error isolation (one task failure doesn't affect others)
        - Circuit breakers prevent runaway failures
        - Pause/resume support
        - Callbacks for monitoring

    Example:
        manager = HeartbeatManager(base_interval=timedelta(seconds=60))

        # Register tasks
        manager.register(HeartbeatTask(
            name="check_status",
            callback=check_status,
            interval=timedelta(minutes=5)
        ))

        # Add error callback
        manager.on_error(lambda task, e: print(f"Task {task} failed: {e}"))

        # Start
        await manager.start()

        # Check status
        print(manager.get_status())

        # Stop
        await manager.stop()
    """

    def __init__(
        self,
        base_interval: timedelta = timedelta(seconds=60),
    ):
        """
        Initialize the HeartbeatManager.

        Args:
            base_interval: How often to check for due tasks (the "tick" rate)
        """
        self.base_interval = base_interval

        self._tasks: Dict[str, HeartbeatTask] = {}
        self._running = False
        self._paused = False
        self._loop_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_heartbeat: List[Callable[[], Awaitable[None]]] = []
        self._on_error: List[
            Callable[[str, Exception], Awaitable[None]]
        ] = []
        self._on_task_complete: List[
            Callable[[str, Any], Awaitable[None]]
        ] = []

    def register(self, task: HeartbeatTask) -> None:
        """
        Register a periodic task.

        Args:
            task: HeartbeatTask to register
        """
        self._tasks[task.name] = task
        logger.info(
            f"Registered heartbeat task: {task.name} "
            f"(every {task.interval.total_seconds()}s)"
        )

    def unregister(self, name: str) -> None:
        """
        Unregister a task.

        Args:
            name: Task name to remove
        """
        if name in self._tasks:
            del self._tasks[name]
            logger.info(f"Unregistered heartbeat task: {name}")

    def get_task(self, name: str) -> Optional[HeartbeatTask]:
        """Get a task by name."""
        return self._tasks.get(name)

    def list_tasks(self) -> List[str]:
        """
        Get names of all registered tasks.

        Returns:
            List of task name strings.
        """
        return list(self._tasks.keys())

    def enable_task(self, name: str) -> None:
        """
        Enable a disabled task.

        Also resets circuit breaker.
        """
        if name in self._tasks:
            self._tasks[name].enabled = True
            self._tasks[name].reset_circuit_breaker()
            logger.info(f"Enabled task: {name}")

    def disable_task(self, name: str) -> None:
        """Disable a task (won't run until enabled)."""
        if name in self._tasks:
            self._tasks[name].enabled = False
            logger.info(f"Disabled task: {name}")

    def on_heartbeat(
        self,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Register callback for every heartbeat tick.

        Called once per base_interval, regardless of tasks.
        """
        self._on_heartbeat.append(callback)

    def on_error(
        self,
        callback: Callable[[str, Exception], Awaitable[None]],
    ) -> None:
        """
        Register callback for task errors.

        Args:
            callback: Async function(task_name, exception)
        """
        self._on_error.append(callback)

    def on_task_complete(
        self,
        callback: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        """
        Register callback for task completion.

        Args:
            callback: Async function(task_name, result)
        """
        self._on_task_complete.append(callback)

    async def start(self) -> None:
        """
        Start the heartbeat loop.

        Idempotent — calling multiple times is safe.
        """
        if self._running:
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            f"Heartbeat started "
            f"(interval: {self.base_interval.total_seconds()}s)"
        )

    async def stop(self) -> None:
        """
        Stop the heartbeat loop.

        Waits for current task to complete before stopping.
        """
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat stopped")

    def pause(self) -> None:
        """Pause heartbeat (tasks won't run but loop continues)."""
        self._paused = True
        logger.info("Heartbeat paused")

    def resume(self) -> None:
        """Resume heartbeat after pause."""
        self._paused = False
        logger.info("Heartbeat resumed")

    @property
    def is_running(self) -> bool:
        """Check if heartbeat is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if heartbeat is paused."""
        return self._paused

    async def tick(self) -> None:
        """
        Manually trigger a single heartbeat tick.

        Useful for testing or when you want to run due tasks
        without waiting for the automatic loop.
        """
        now = datetime.utcnow()

        # Run general heartbeat callbacks
        for callback in self._on_heartbeat:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Heartbeat callback error: {e}")

        # Run due tasks
        for task in list(self._tasks.values()):
            if task.should_run(now):
                await self._run_task(task, now)

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(1)
                    continue

                await self.tick()

                # Wait for next tick
                await asyncio.sleep(
                    self.base_interval.total_seconds()
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _run_task(
        self, task: HeartbeatTask, now: datetime
    ) -> None:
        """
        Run a single task with error handling.

        Args:
            task: Task to run
            now: Current timestamp
        """
        try:
            logger.debug(f"Running heartbeat task: {task.name}")
            result = await task.callback()
            task.schedule_next(now)
            task.record_success()

            # Notify completion callbacks
            for callback in self._on_task_complete:
                try:
                    await callback(task.name, result)
                except Exception:
                    pass

        except Exception as e:
            task.record_error(str(e))
            task.schedule_next(now)  # Still schedule next attempt
            logger.error(f"Task {task.name} failed: {e}")

            # Notify error callbacks
            for callback in self._on_error:
                try:
                    await callback(task.name, e)
                except Exception:
                    pass

            # Check circuit breaker
            if task.consecutive_errors >= task.max_consecutive_errors:
                logger.warning(
                    f"Circuit breaker opened for task: {task.name} "
                    f"after {task.consecutive_errors} consecutive errors"
                )

    async def run_task_now(self, name: str) -> Any:
        """
        Run a specific task immediately (bypass schedule).

        Args:
            name: Task name

        Returns:
            Task result

        Raises:
            ValueError: If task not found
        """
        if name not in self._tasks:
            raise ValueError(f"Task not found: {name}")

        task = self._tasks[name]
        now = datetime.utcnow()

        logger.info(f"Running task immediately: {name}")
        await self._run_task(task, now)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current heartbeat status.

        Returns:
            Status dictionary with running state and task info
        """
        return {
            "running": self._running,
            "paused": self._paused,
            "base_interval_seconds": self.base_interval.total_seconds(),
            "task_count": len(self._tasks),
            "tasks": {
                name: task.to_dict()
                for name, task in self._tasks.items()
            },
        }

    def get_due_tasks(self) -> List[str]:
        """Get names of tasks that are due to run."""
        now = datetime.utcnow()
        return [
            name
            for name, task in self._tasks.items()
            if task.should_run(now)
        ]


# =============================================================================
# Convenience Decorator
# =============================================================================


def heartbeat_task(
    interval: timedelta,
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Callable[[Callable[[], Awaitable[Any]]], HeartbeatTask]:
    """
    Decorator to create a HeartbeatTask from an async function.

    Args:
        interval: Time between runs
        name: Task name (defaults to function name)
        description: Task description
        tags: Task tags

    Returns:
        HeartbeatTask instance

    Example:
        @heartbeat_task(
            interval=timedelta(hours=4),
            description="Check feed"
        )
        async def check_moltbook():
            # ... do check ...
            pass

        heartbeat.register(check_moltbook)
    """

    def decorator(
        func: Callable[[], Awaitable[Any]],
    ) -> HeartbeatTask:
        return HeartbeatTask(
            name=name or func.__name__,
            callback=func,
            interval=interval,
            description=description or func.__doc__ or "",
            tags=tags or [],
        )

    return decorator
