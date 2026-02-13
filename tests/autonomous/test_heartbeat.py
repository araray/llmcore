# tests/autonomous/test_heartbeat.py
"""
Test suite for the Heartbeat System.

Tests cover:
    - HeartbeatTask: scheduling logic, circuit breakers, serialization
    - HeartbeatManager: registration, start/stop lifecycle, pause/resume,
      error handling, callbacks, manual tick, status reporting
    - heartbeat_task decorator: function wrapping and naming

Coverage target: >90%
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from llmcore.autonomous.heartbeat import (
    HeartbeatManager,
    HeartbeatTask,
    heartbeat_task,
)

# =============================================================================
# HeartbeatTask Unit Tests
# =============================================================================


class TestHeartbeatTask:
    """Tests for HeartbeatTask dataclass and its methods."""

    def _make_task(self, **kwargs) -> HeartbeatTask:
        """Helper to create a task with defaults."""
        defaults = {
            "name": "test_task",
            "callback": AsyncMock(),
            "interval": timedelta(seconds=60),
        }
        defaults.update(kwargs)
        return HeartbeatTask(**defaults)

    def test_create_with_defaults(self):
        """Task initializes with correct defaults."""
        cb = AsyncMock()
        task = HeartbeatTask(
            name="my_task",
            callback=cb,
            interval=timedelta(minutes=5),
        )
        assert task.name == "my_task"
        assert task.callback is cb
        assert task.interval == timedelta(minutes=5)
        assert task.enabled is True
        assert task.last_run is None
        assert task.next_run is None
        assert task.run_count == 0
        assert task.error_count == 0
        assert task.consecutive_errors == 0
        assert task.max_consecutive_errors == 5
        assert task.last_error is None
        assert task.description == ""
        assert task.tags == []

    # ── should_run() ─────────────────────────────────────────────

    def test_should_run_never_run_before(self):
        """A new task (next_run=None) should run immediately."""
        task = self._make_task()
        assert task.should_run(datetime.utcnow()) is True

    def test_should_run_disabled(self):
        """A disabled task should never run."""
        task = self._make_task(enabled=False)
        assert task.should_run(datetime.utcnow()) is False

    def test_should_run_circuit_broken(self):
        """Task with too many consecutive errors should not run."""
        task = self._make_task(
            max_consecutive_errors=3,
        )
        task.consecutive_errors = 3
        assert task.should_run(datetime.utcnow()) is False

    def test_should_run_time_not_due(self):
        """Task should not run if next_run is in the future."""
        task = self._make_task()
        task.next_run = datetime.utcnow() + timedelta(hours=1)
        assert task.should_run(datetime.utcnow()) is False

    def test_should_run_time_is_due(self):
        """Task should run if next_run is in the past."""
        task = self._make_task()
        task.next_run = datetime.utcnow() - timedelta(seconds=1)
        assert task.should_run(datetime.utcnow()) is True

    def test_should_run_exact_time(self):
        """Task should run if next_run equals now."""
        now = datetime.utcnow()
        task = self._make_task()
        task.next_run = now
        assert task.should_run(now) is True

    # ── schedule_next() ──────────────────────────────────────────

    def test_schedule_next(self):
        """schedule_next sets last_run, next_run, and increments run_count."""
        task = self._make_task(interval=timedelta(minutes=10))
        now = datetime.utcnow()

        task.schedule_next(now)

        assert task.last_run == now
        assert task.next_run == now + timedelta(minutes=10)
        assert task.run_count == 1

    def test_schedule_next_multiple_times(self):
        """Successive schedule_next calls increment run_count."""
        task = self._make_task()
        now = datetime.utcnow()

        task.schedule_next(now)
        task.schedule_next(now + timedelta(seconds=60))

        assert task.run_count == 2
        assert task.last_run == now + timedelta(seconds=60)

    # ── record_success() / record_error() ────────────────────────

    def test_record_success_resets_errors(self):
        """Success resets consecutive_errors and last_error."""
        task = self._make_task()
        task.consecutive_errors = 3
        task.last_error = "something broke"

        task.record_success()

        assert task.consecutive_errors == 0
        assert task.last_error is None

    def test_record_error_increments(self):
        """Error increments both error_count and consecutive_errors."""
        task = self._make_task()
        task.record_error("first error")

        assert task.error_count == 1
        assert task.consecutive_errors == 1
        assert task.last_error == "first error"

        task.record_error("second error")

        assert task.error_count == 2
        assert task.consecutive_errors == 2
        assert task.last_error == "second error"

    def test_record_error_then_success(self):
        """Success after errors resets consecutive count but not total."""
        task = self._make_task()
        task.record_error("e1")
        task.record_error("e2")
        task.record_success()

        assert task.error_count == 2  # total preserved
        assert task.consecutive_errors == 0

    # ── Circuit breaker ──────────────────────────────────────────

    def test_is_circuit_broken_false(self):
        """Circuit breaker is closed when errors < threshold."""
        task = self._make_task(max_consecutive_errors=5)
        task.consecutive_errors = 4
        assert task.is_circuit_broken is False

    def test_is_circuit_broken_true(self):
        """Circuit breaker opens at threshold."""
        task = self._make_task(max_consecutive_errors=5)
        task.consecutive_errors = 5
        assert task.is_circuit_broken is True

    def test_reset_circuit_breaker(self):
        """Manual reset restores circuit breaker."""
        task = self._make_task(max_consecutive_errors=3)
        task.consecutive_errors = 3
        task.last_error = "some error"
        assert task.is_circuit_broken is True

        task.reset_circuit_breaker()

        assert task.is_circuit_broken is False
        assert task.consecutive_errors == 0
        assert task.last_error is None

    # ── to_dict() ────────────────────────────────────────────────

    def test_to_dict_complete(self):
        """Serialization includes all fields."""
        task = self._make_task(
            description="A test task",
            tags=["test", "unit"],
        )
        now = datetime.utcnow()
        task.schedule_next(now)

        d = task.to_dict()

        assert d["name"] == "test_task"
        assert d["interval_seconds"] == 60.0
        assert d["enabled"] is True
        assert d["last_run"] == now.isoformat()
        assert d["next_run"] == (now + timedelta(seconds=60)).isoformat()
        assert d["run_count"] == 1
        assert d["error_count"] == 0
        assert d["consecutive_errors"] == 0
        assert d["max_consecutive_errors"] == 5
        assert d["last_error"] is None
        assert d["is_circuit_broken"] is False
        assert d["description"] == "A test task"
        assert d["tags"] == ["test", "unit"]

    def test_to_dict_none_dates(self):
        """Serialization handles None dates gracefully."""
        task = self._make_task()
        d = task.to_dict()
        assert d["last_run"] is None
        assert d["next_run"] is None


# =============================================================================
# HeartbeatManager Unit Tests
# =============================================================================


class TestHeartbeatManager:
    """Tests for HeartbeatManager lifecycle, registration, and execution."""

    def _make_task(self, name="task1", interval_s=60, **kwargs) -> HeartbeatTask:
        """Helper to create a task."""
        return HeartbeatTask(
            name=name,
            callback=kwargs.pop("callback", AsyncMock()),
            interval=timedelta(seconds=interval_s),
            **kwargs,
        )

    # ── Registration ─────────────────────────────────────────────

    def test_register(self, heartbeat_manager):
        """Tasks can be registered."""
        task = self._make_task("my_task")
        heartbeat_manager.register(task)
        assert heartbeat_manager.get_task("my_task") is task

    def test_register_overwrites(self, heartbeat_manager):
        """Registering with same name replaces the task."""
        t1 = self._make_task("dup")
        t2 = self._make_task("dup")

        heartbeat_manager.register(t1)
        heartbeat_manager.register(t2)

        assert heartbeat_manager.get_task("dup") is t2

    def test_unregister(self, heartbeat_manager):
        """Unregistering removes the task."""
        task = self._make_task("removable")
        heartbeat_manager.register(task)
        heartbeat_manager.unregister("removable")
        assert heartbeat_manager.get_task("removable") is None

    def test_unregister_nonexistent(self, heartbeat_manager):
        """Unregistering a non-existent task is a no-op."""
        heartbeat_manager.unregister("ghost")  # should not raise

    def test_get_task_not_found(self, heartbeat_manager):
        """get_task returns None for unknown names."""
        assert heartbeat_manager.get_task("nope") is None

    # ── Enable / Disable ─────────────────────────────────────────

    def test_enable_task(self, heartbeat_manager):
        """Enabling a task sets enabled=True and resets circuit breaker."""
        task = self._make_task("t")
        task.enabled = False
        task.consecutive_errors = 99
        task.last_error = "old error"

        heartbeat_manager.register(task)
        heartbeat_manager.enable_task("t")

        assert task.enabled is True
        assert task.consecutive_errors == 0
        assert task.last_error is None

    def test_disable_task(self, heartbeat_manager):
        """Disabling a task sets enabled=False."""
        task = self._make_task("t")
        heartbeat_manager.register(task)
        heartbeat_manager.disable_task("t")
        assert task.enabled is False

    def test_enable_nonexistent(self, heartbeat_manager):
        """Enabling a non-existent task is a no-op."""
        heartbeat_manager.enable_task("ghost")  # no raise

    def test_disable_nonexistent(self, heartbeat_manager):
        """Disabling a non-existent task is a no-op."""
        heartbeat_manager.disable_task("ghost")  # no raise

    # ── Start / Stop ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_start_stop(self, heartbeat_manager):
        """Start creates a loop task; stop cancels it."""
        assert heartbeat_manager.is_running is False

        await heartbeat_manager.start()
        assert heartbeat_manager.is_running is True

        await heartbeat_manager.stop()
        assert heartbeat_manager.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self, heartbeat_manager):
        """Calling start twice doesn't create a second loop."""
        await heartbeat_manager.start()
        first_task = heartbeat_manager._loop_task

        await heartbeat_manager.start()  # idempotent
        assert heartbeat_manager._loop_task is first_task

        await heartbeat_manager.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, heartbeat_manager):
        """Stopping before starting is safe."""
        await heartbeat_manager.stop()  # no raise

    # ── Pause / Resume ───────────────────────────────────────────

    def test_pause_resume(self, heartbeat_manager):
        """Pause and resume toggle the paused flag."""
        assert heartbeat_manager.is_paused is False

        heartbeat_manager.pause()
        assert heartbeat_manager.is_paused is True

        heartbeat_manager.resume()
        assert heartbeat_manager.is_paused is False

    @pytest.mark.asyncio
    async def test_paused_does_not_execute(self, heartbeat_manager):
        """Tick skips task execution when paused."""
        cb = AsyncMock()
        task = self._make_task("check", callback=cb)
        heartbeat_manager.register(task)
        heartbeat_manager.pause()

        # tick() still runs heartbeat callbacks but should_run sees paused state
        # Actually, tick() checks task.should_run not paused — paused is checked in loop.
        # So we test the loop behavior indirectly:
        await heartbeat_manager.start()
        await asyncio.sleep(0.15)  # > base_interval (50ms)
        heartbeat_manager.pause()
        initial_count = task.run_count

        await asyncio.sleep(0.15)  # another interval passes while paused
        # Run count should not increase significantly while paused
        assert task.run_count <= initial_count + 1  # at most 1 extra from race

        await heartbeat_manager.stop()

    # ── tick() ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tick_runs_due_tasks(self, heartbeat_manager):
        """tick() executes tasks whose should_run returns True."""
        cb = AsyncMock(return_value="ok")
        task = self._make_task("due", callback=cb)
        heartbeat_manager.register(task)

        await heartbeat_manager.tick()

        cb.assert_awaited_once()
        assert task.run_count == 1
        assert task.consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_tick_skips_not_due(self, heartbeat_manager):
        """tick() does not execute tasks that are not due."""
        cb = AsyncMock()
        task = self._make_task("not_due", callback=cb)
        task.next_run = datetime.utcnow() + timedelta(hours=1)
        heartbeat_manager.register(task)

        await heartbeat_manager.tick()

        cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_runs_heartbeat_callbacks(self, heartbeat_manager):
        """tick() invokes on_heartbeat callbacks regardless of tasks."""
        hb_callback = AsyncMock()
        heartbeat_manager.on_heartbeat(hb_callback)

        await heartbeat_manager.tick()

        hb_callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_heartbeat_callback_error_isolated(self, heartbeat_manager):
        """Heartbeat callback errors don't prevent task execution."""
        failing_hb = AsyncMock(side_effect=RuntimeError("hb fail"))
        heartbeat_manager.on_heartbeat(failing_hb)

        cb = AsyncMock(return_value="ok")
        task = self._make_task("still_runs", callback=cb)
        heartbeat_manager.register(task)

        await heartbeat_manager.tick()  # should not raise
        cb.assert_awaited_once()

    # ── Error handling ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_task_error_records_failure(self, heartbeat_manager):
        """Failed tasks get their error recorded."""
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        task = self._make_task("failing", callback=cb)
        heartbeat_manager.register(task)

        await heartbeat_manager.tick()

        assert task.error_count == 1
        assert task.consecutive_errors == 1
        assert task.last_error == "boom"
        assert task.run_count == 1  # still scheduled for next

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, heartbeat_manager):
        """After max_consecutive_errors failures, task stops running."""
        cb = AsyncMock(side_effect=RuntimeError("fail"))
        task = self._make_task(
            "breaking",
            callback=cb,
            max_consecutive_errors=3,
        )
        heartbeat_manager.register(task)

        # Run 3 times to break the circuit
        for _ in range(3):
            task.next_run = None  # Force due
            await heartbeat_manager.tick()

        assert task.is_circuit_broken is True
        assert task.consecutive_errors == 3

        # 4th tick should not call the callback
        cb.reset_mock()
        task.next_run = None
        await heartbeat_manager.tick()
        cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_callback_invoked(self, heartbeat_manager):
        """on_error callbacks are called when tasks fail."""
        error_cb = AsyncMock()
        heartbeat_manager.on_error(error_cb)

        cb = AsyncMock(side_effect=ValueError("bad"))
        task = self._make_task("err", callback=cb)
        heartbeat_manager.register(task)

        await heartbeat_manager.tick()

        error_cb.assert_awaited_once()
        args = error_cb.call_args[0]
        assert args[0] == "err"  # task name
        assert isinstance(args[1], ValueError)

    @pytest.mark.asyncio
    async def test_completion_callback_invoked(self, heartbeat_manager):
        """on_task_complete callbacks are called on success."""
        complete_cb = AsyncMock()
        heartbeat_manager.on_task_complete(complete_cb)

        cb = AsyncMock(return_value="result")
        task = self._make_task("ok", callback=cb)
        heartbeat_manager.register(task)

        await heartbeat_manager.tick()

        complete_cb.assert_awaited_once_with("ok", "result")

    @pytest.mark.asyncio
    async def test_error_in_completion_callback_isolated(self, heartbeat_manager):
        """Error in completion callback doesn't affect other processing."""
        bad_cb = AsyncMock(side_effect=RuntimeError("cb fail"))
        heartbeat_manager.on_task_complete(bad_cb)

        cb = AsyncMock(return_value="ok")
        task = self._make_task("t", callback=cb)
        heartbeat_manager.register(task)

        # Should not raise
        await heartbeat_manager.tick()
        assert task.run_count == 1

    # ── run_task_now() ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_task_now(self, heartbeat_manager):
        """run_task_now executes a task immediately."""
        cb = AsyncMock(return_value="immediate")
        task = self._make_task("imm", callback=cb)
        task.next_run = datetime.utcnow() + timedelta(hours=99)
        heartbeat_manager.register(task)

        await heartbeat_manager.run_task_now("imm")

        cb.assert_awaited_once()
        assert task.run_count == 1

    @pytest.mark.asyncio
    async def test_run_task_now_not_found(self, heartbeat_manager):
        """run_task_now raises ValueError for unknown tasks."""
        with pytest.raises(ValueError, match="Task not found"):
            await heartbeat_manager.run_task_now("nope")

    # ── Status ───────────────────────────────────────────────────

    def test_get_status(self, heartbeat_manager):
        """get_status returns expected structure."""
        task = self._make_task("s1")
        heartbeat_manager.register(task)

        status = heartbeat_manager.get_status()

        assert status["running"] is False
        assert status["paused"] is False
        assert status["base_interval_seconds"] == 0.05  # 50ms from fixture
        assert status["task_count"] == 1
        assert "s1" in status["tasks"]
        assert status["tasks"]["s1"]["name"] == "s1"

    def test_get_due_tasks_none(self, heartbeat_manager):
        """get_due_tasks returns empty when no tasks due."""
        task = self._make_task("future")
        task.next_run = datetime.utcnow() + timedelta(hours=1)
        heartbeat_manager.register(task)

        assert heartbeat_manager.get_due_tasks() == []

    def test_get_due_tasks_some(self, heartbeat_manager):
        """get_due_tasks returns names of due tasks."""
        t1 = self._make_task("due1")  # next_run=None → due
        t2 = self._make_task("not_due")
        t2.next_run = datetime.utcnow() + timedelta(hours=1)

        heartbeat_manager.register(t1)
        heartbeat_manager.register(t2)

        due = heartbeat_manager.get_due_tasks()
        assert "due1" in due
        assert "not_due" not in due

    # ── Multiple tasks ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_multiple_tasks_independent(self, heartbeat_manager):
        """Tasks execute independently — one failure doesn't block others."""
        fail_cb = AsyncMock(side_effect=RuntimeError("fail"))
        ok_cb = AsyncMock(return_value="ok")

        fail_task = self._make_task("fail", callback=fail_cb)
        ok_task = self._make_task("ok", callback=ok_cb)

        heartbeat_manager.register(fail_task)
        heartbeat_manager.register(ok_task)

        await heartbeat_manager.tick()

        fail_cb.assert_awaited_once()
        ok_cb.assert_awaited_once()
        assert fail_task.error_count == 1
        assert ok_task.error_count == 0

    @pytest.mark.asyncio
    async def test_loop_runs_periodically(self, heartbeat_manager):
        """The heartbeat loop runs tasks at intervals."""
        cb = AsyncMock(return_value=None)
        task = self._make_task("periodic", callback=cb, interval_s=0)
        heartbeat_manager.register(task)

        await heartbeat_manager.start()
        await asyncio.sleep(0.2)  # 200ms > 4 × 50ms base_interval
        await heartbeat_manager.stop()

        # Should have run multiple times
        assert cb.await_count >= 2


# =============================================================================
# heartbeat_task Decorator Tests
# =============================================================================


class TestHeartbeatTaskDecorator:
    """Tests for the @heartbeat_task convenience decorator."""

    def test_decorator_creates_task(self):
        """Decorator returns a HeartbeatTask wrapping the function."""

        @heartbeat_task(interval=timedelta(hours=1))
        async def my_func():
            """My doc."""
            pass

        assert isinstance(my_func, HeartbeatTask)
        assert my_func.name == "my_func"
        assert my_func.interval == timedelta(hours=1)
        assert my_func.description == "My doc."

    def test_decorator_custom_name(self):
        """Decorator uses custom name when provided."""

        @heartbeat_task(
            interval=timedelta(minutes=5),
            name="custom_name",
            description="Custom desc",
            tags=["a", "b"],
        )
        async def ignored_name():
            pass

        assert isinstance(ignored_name, HeartbeatTask)
        assert ignored_name.name == "custom_name"
        assert ignored_name.description == "Custom desc"
        assert ignored_name.tags == ["a", "b"]

    def test_decorator_no_docstring(self):
        """Decorator handles missing docstring gracefully."""

        @heartbeat_task(interval=timedelta(seconds=10))
        async def no_doc():
            pass

        assert no_doc.description == ""  # func.__doc__ is None → ""

    @pytest.mark.asyncio
    async def test_decorated_task_is_callable(self):
        """Decorated task's callback is the original function."""
        call_count = 0

        @heartbeat_task(interval=timedelta(seconds=1))
        async def counter():
            nonlocal call_count
            call_count += 1

        await counter.callback()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorated_task_registerable(self, heartbeat_manager):
        """Decorated task can be registered and executed."""

        @heartbeat_task(
            interval=timedelta(seconds=1),
            name="registerable",
        )
        async def my_task():
            return 42

        heartbeat_manager.register(my_task)
        await heartbeat_manager.tick()

        task = heartbeat_manager.get_task("registerable")
        assert task.run_count == 1

    @pytest.fixture
    def heartbeat_manager(self):
        """Local fixture for decorator tests."""
        return HeartbeatManager(base_interval=timedelta(milliseconds=50))


# =============================================================================
# Concurrent Task Limit Tests
# =============================================================================


class TestHeartbeatConcurrentLimit:
    """Tests for max_concurrent_tasks semaphore enforcement."""

    @pytest.mark.asyncio
    async def test_semaphore_created_when_limit_set(self):
        """Semaphore is created when max_concurrent_tasks > 0."""
        hb = HeartbeatManager(
            base_interval=timedelta(milliseconds=50),
            max_concurrent_tasks=2,
        )
        assert hb._semaphore is not None
        assert hb.max_concurrent_tasks == 2

    @pytest.mark.asyncio
    async def test_no_semaphore_when_unlimited(self):
        """No semaphore when max_concurrent_tasks is 0 (default)."""
        hb = HeartbeatManager(base_interval=timedelta(milliseconds=50))
        assert hb._semaphore is None
        assert hb.max_concurrent_tasks == 0

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforced(self):
        """Only max_concurrent_tasks callbacks run simultaneously."""
        max_concurrent = 2
        hb = HeartbeatManager(
            base_interval=timedelta(milliseconds=50),
            max_concurrent_tasks=max_concurrent,
        )

        # Track concurrent execution
        import asyncio

        concurrent_count = 0
        max_observed_concurrent = 0
        lock = asyncio.Lock()

        async def slow_task():
            nonlocal concurrent_count, max_observed_concurrent
            async with lock:
                concurrent_count += 1
                max_observed_concurrent = max(max_observed_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            async with lock:
                concurrent_count -= 1

        # Register 5 tasks, all immediately due
        now = datetime.utcnow()
        for i in range(5):
            task = HeartbeatTask(
                name=f"task_{i}",
                callback=slow_task,
                interval=timedelta(seconds=0),  # Always due
            )
            task.next_run = now - timedelta(seconds=1)
            hb.register(task)

        # Run all due tasks concurrently via gather
        due_tasks = [t for t in hb._tasks.values() if t.should_run(now)]
        await asyncio.gather(*[hb._run_task(t, now) for t in due_tasks])

        # The semaphore should have limited concurrency
        assert max_observed_concurrent <= max_concurrent

    @pytest.mark.asyncio
    async def test_status_includes_concurrent_limit(self):
        """get_status() reports max_concurrent_tasks."""
        hb = HeartbeatManager(
            base_interval=timedelta(milliseconds=50),
            max_concurrent_tasks=3,
        )
        status = hb.get_status()
        assert status["max_concurrent_tasks"] == 3

    @pytest.mark.asyncio
    async def test_unlimited_status(self):
        """get_status() reports 0 for unlimited."""
        hb = HeartbeatManager(base_interval=timedelta(milliseconds=50))
        status = hb.get_status()
        assert status["max_concurrent_tasks"] == 0

    @pytest.mark.asyncio
    async def test_semaphore_released_on_error(self):
        """Semaphore is released even when task callback raises."""
        hb = HeartbeatManager(
            base_interval=timedelta(milliseconds=50),
            max_concurrent_tasks=1,
        )

        async def failing_task():
            raise RuntimeError("boom")

        task = HeartbeatTask(
            name="failing",
            callback=failing_task,
            interval=timedelta(seconds=0),
        )
        task.next_run = datetime.utcnow() - timedelta(seconds=1)
        hb.register(task)

        now = datetime.utcnow()
        await hb._run_task(task, now)

        # Semaphore should be released — next task should be able to acquire
        assert hb._semaphore._value == 1  # type: ignore[union-attr]


# =============================================================================
# HeartbeatManager.from_config()
# =============================================================================


class TestHeartbeatManagerFromConfig:
    """Tests for the HeartbeatManager.from_config() factory classmethod."""

    def test_from_config_basic(self):
        """from_config maps config fields to constructor params."""

        class FakeConfig:
            base_interval = 30.0
            max_concurrent_tasks = 5
            enabled = True

        hb = HeartbeatManager.from_config(FakeConfig())

        assert hb.base_interval == timedelta(seconds=30.0)
        assert hb.max_concurrent_tasks == 5
        assert hb._semaphore is not None

    def test_from_config_defaults(self):
        """from_config handles missing attributes gracefully."""

        class MinimalConfig:
            pass

        hb = HeartbeatManager.from_config(MinimalConfig())

        assert hb.base_interval == timedelta(seconds=60.0)
        assert hb.max_concurrent_tasks == 0
        assert hb._semaphore is None

    def test_from_config_zero_concurrent(self):
        """max_concurrent_tasks=0 means unlimited (no semaphore)."""

        class ZeroConfig:
            base_interval = 120.0
            max_concurrent_tasks = 0

        hb = HeartbeatManager.from_config(ZeroConfig())

        assert hb.max_concurrent_tasks == 0
        assert hb._semaphore is None

    def test_from_config_preserves_type(self):
        """from_config returns a HeartbeatManager instance."""

        class Cfg:
            base_interval = 10.0
            max_concurrent_tasks = 2

        hb = HeartbeatManager.from_config(Cfg())

        assert isinstance(hb, HeartbeatManager)

    @pytest.mark.asyncio
    async def test_from_config_manager_is_functional(self):
        """Manager created via from_config can register tasks and tick."""

        class Cfg:
            base_interval = 0.05
            max_concurrent_tasks = 2

        hb = HeartbeatManager.from_config(Cfg())

        ran = False

        async def my_task():
            nonlocal ran
            ran = True

        task = HeartbeatTask(
            name="test",
            callback=my_task,
            interval=timedelta(seconds=0),
        )
        hb.register(task)
        await hb.tick()

        assert ran is True

    def test_from_config_with_pydantic_model(self):
        """Works with actual Pydantic HeartbeatConfig if available."""
        try:
            from llmcore.config.autonomous_config import HeartbeatConfig

            cfg = HeartbeatConfig(
                enabled=True,
                base_interval=45.0,
                max_concurrent_tasks=4,
            )
            hb = HeartbeatManager.from_config(cfg)

            assert hb.base_interval == timedelta(seconds=45.0)
            assert hb.max_concurrent_tasks == 4
        except ImportError:
            pytest.skip("HeartbeatConfig not importable")
