# tests/autonomous/test_integration.py
"""
Integration tests for the autonomous module.

These tests verify that the four core subsystems (GoalManager,
EscalationManager, HeartbeatManager, ResourceMonitor) compose
correctly for a full autonomous operation lifecycle.

The integration scenario simulates:
1. Setting a primary goal with success criteria
2. Registering heartbeat tasks for periodic checks
3. Resource monitoring with constraint enforcement
4. Escalation when goals fail or resources are constrained
5. Full lifecycle: start → execute → monitor → escalate → complete

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §7.3 (Module Structure)
    - autonomous/__init__.py (Module exports)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.autonomous import (
    ConstraintViolation,
    Escalation,
    EscalationLevel,
    EscalationManager,
    EscalationReason,
    Goal,
    GoalManager,
    GoalPriority,
    GoalStatus,
    GoalStore,
    HeartbeatManager,
    HeartbeatTask,
    ResourceConstraints,
    ResourceMonitor,
    ResourceStatus,
    ResourceUsage,
    SuccessCriterion,
    heartbeat_task,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def goal_store(tmp_path):
    """Create a GoalStore backed by a temporary file."""
    store_path = str(tmp_path / "test_goals.json")
    return GoalStore(store_path)


@pytest.fixture
def goal_manager(goal_store):
    """Create a GoalManager with mock LLM for decomposition."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="Sub-goal decomposition response")
    return GoalManager(storage=goal_store, llm_provider=llm)


@pytest.fixture
def escalation_manager():
    """Create an EscalationManager with defaults."""
    return EscalationManager()


@pytest.fixture
def heartbeat_manager():
    """Create a HeartbeatManager."""
    return HeartbeatManager()


@pytest.fixture
def resource_constraints():
    """Create moderate resource constraints for testing."""
    return ResourceConstraints(
        max_cpu_percent=80.0,
        max_memory_percent=80.0,
        max_temperature_c=75.0,
        max_hourly_cost_usd=1.0,
        max_daily_cost_usd=10.0,
        max_hourly_tokens=100_000,
        max_daily_tokens=1_000_000,
        min_disk_free_gb=1.0,
    )


@pytest.fixture
def resource_monitor(resource_constraints):
    """Create a ResourceMonitor with given constraints."""
    return ResourceMonitor(constraints=resource_constraints)


# =============================================================================
# TEST: FULL AUTONOMOUS LIFECYCLE
# =============================================================================


class TestAutonomousLifecycle:
    """
    End-to-end integration test simulating an autonomous operation cycle.

    Flow:
        1. Initialize all subsystems
        2. Set a primary goal
        3. Register resource-check heartbeat
        4. Simulate work iterations
        5. Handle resource constraints via escalation
        6. Complete the goal
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self,
        goal_manager,
        escalation_manager,
        heartbeat_manager,
        resource_monitor,
    ):
        """
        Simulate a complete autonomous operation cycle.

        This is the canonical integration test: start → work → monitor → complete.
        """
        # --- Phase 1: Set primary goal ---
        goal = await goal_manager.set_primary_goal(
            description="Complete all integration tests",
            priority=GoalPriority.HIGH,
            success_criteria=[
                SuccessCriterion(
                    description="All tests pass",
                    metric_name="test_pass_rate",
                    target_value=100.0,
                    comparator=">=",
                ),
            ],
        )

        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority == GoalPriority.HIGH
        assert len(goal.success_criteria) == 1

        # --- Phase 2: Register heartbeat tasks ---
        check_count = {"value": 0}

        async def resource_check():
            """Heartbeat task: check resources."""
            check_count["value"] += 1
            return {"checked": True}

        task = HeartbeatTask(
            name="resource_check",
            callback=resource_check,
            interval=timedelta(seconds=5),
            description="Periodic resource check",
        )
        heartbeat_manager.register(task)

        # Verify registration
        assert "resource_check" in heartbeat_manager.list_tasks()

        # --- Phase 3: Simulate work with resource monitoring ---

        # Mock healthy system state
        with patch("psutil.cpu_percent", return_value=45.0), \
             patch("psutil.virtual_memory") as mock_mem, \
             patch("psutil.disk_usage") as mock_disk, \
             patch("psutil.sensors_temperatures", return_value={}):

            mock_mem.return_value = MagicMock(
                percent=50.0, used=4 * 1024**3, available=4 * 1024**3,
            )
            mock_disk.return_value = MagicMock(free=10 * 1024**3)  # 10 GB free

            status = await resource_monitor._check_resources()

        assert isinstance(status, ResourceStatus)
        assert status.can_proceed
        assert not status.is_constrained

        # --- Phase 4: Report goal progress ---
        goal.success_criteria[0].current_value = 75.0
        goal.update_progress()

        assert goal.progress > 0.0
        assert goal.status == GoalStatus.ACTIVE

        # --- Phase 5: Simulate resource constraint and escalation ---
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.RESOURCE_EXHAUSTED,
            title="CPU throttle detected",
            message="CPU usage exceeded 80% for 5 minutes",
        )

        # Verify escalation is pending
        pending = escalation_manager.get_pending()
        assert len(pending) == 1
        escalation = pending[0]
        assert escalation.level == EscalationLevel.ACTION
        assert escalation.reason == EscalationReason.RESOURCE_EXHAUSTED

        # Resolve escalation (simulating human response)
        escalation_manager.respond(
            escalation.id,
            response="Acknowledged - reducing concurrent tasks",
        )

        pending_after = escalation_manager.get_pending()
        assert len(pending_after) == 0

        # --- Phase 6: Complete the goal ---
        goal.success_criteria[0].current_value = 100.0
        goal.update_progress()

        assert goal.progress == 1.0
        assert goal.status == GoalStatus.COMPLETED

        # Report success through the manager
        await goal_manager.report_success(goal.id)

        # Verify final state
        retrieved = await goal_manager.get_goal(goal.id)
        assert retrieved is not None
        assert retrieved.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_goal_failure_triggers_escalation(
        self,
        goal_manager,
        escalation_manager,
    ):
        """
        When a goal fails terminally, the system should create
        an escalation for human review.
        """
        goal = await goal_manager.set_primary_goal(
            description="Deploy production release",
            priority=GoalPriority.CRITICAL,
            success_criteria=[
                SuccessCriterion(
                    description="Deployment successful",
                    metric_name="deploy_status",
                    target_value=1.0,
                    comparator="==",
                ),
            ],
        )

        # Simulate terminal failure
        await goal_manager.report_failure(
            goal.id,
            reason="Deployment failed: connection timeout",
            recoverable=False,
        )

        retrieved = await goal_manager.get_goal(goal.id)
        assert retrieved.status == GoalStatus.FAILED

        # Agent should escalate the failure
        await escalation_manager.escalate(
            level=EscalationLevel.URGENT,
            reason=EscalationReason.REPEATED_FAILURE,
            title=f"Critical goal failed: {goal.description}",
            message="Deployment failed: connection timeout",
            details={"goal_id": goal.id},
        )

        pending = escalation_manager.get_pending()
        assert len(pending) == 1
        escalation = pending[0]
        assert escalation.level == EscalationLevel.URGENT
        assert escalation.details["goal_id"] == goal.id

    @pytest.mark.asyncio
    async def test_heartbeat_drives_resource_checks(
        self,
        heartbeat_manager,
        resource_monitor,
    ):
        """
        Heartbeat system drives periodic resource monitoring.
        Resource results feed back into the operation loop.
        """
        results: List[ResourceStatus] = []

        async def monitor_resources():
            """Heartbeat-driven resource check."""
            with patch("psutil.cpu_percent", return_value=30.0), \
                 patch("psutil.virtual_memory") as mm, \
                 patch("psutil.disk_usage") as md, \
                 patch("psutil.sensors_temperatures", return_value={}):
                mm.return_value = MagicMock(
                    percent=40.0, used=3 * 1024**3, available=5 * 1024**3,
                )
                md.return_value = MagicMock(free=20 * 1024**3)
                status = await resource_monitor._check_resources()
                results.append(status)
                return status

        task = HeartbeatTask(
            name="resource_monitor",
            callback=monitor_resources,
            interval=timedelta(seconds=1),
        )
        heartbeat_manager.register(task)

        # Start heartbeat manager
        await heartbeat_manager.start()

        # Manually tick a few times (rather than waiting real-time)
        for _ in range(3):
            task.last_run = None  # Force it to be "due"
            task.next_run = None  # Clear scheduled next-run
            await heartbeat_manager.tick()

        await heartbeat_manager.stop()

        assert len(results) >= 3
        for status in results:
            assert status.can_proceed

    @pytest.mark.asyncio
    async def test_resource_violation_blocks_goal_execution(
        self,
        goal_manager,
        resource_monitor,
    ):
        """
        When resources are constrained, the system should throttle
        goal execution rather than proceeding.
        """
        goal = await goal_manager.set_primary_goal(
            description="Process large dataset",
            priority=GoalPriority.NORMAL,
        )

        # Simulate over-heated CPU (hard limit)
        with patch("psutil.cpu_percent", return_value=95.0), \
             patch("psutil.virtual_memory") as mm, \
             patch("psutil.disk_usage") as md, \
             patch("psutil.sensors_temperatures", return_value={
                 "coretemp": [MagicMock(current=82.0)]
             }):
            mm.return_value = MagicMock(
                percent=90.0, used=7 * 1024**3, available=1 * 1024**3,
            )
            md.return_value = MagicMock(free=0.5 * 1024**3)  # 500 MB

            status = await resource_monitor._check_resources()

        # System should be constrained
        assert status.is_constrained

        # In an autonomous loop, this would skip execution:
        # if resources.can_proceed:
        #     execute(goal)
        # else:
        #     wait / throttle
        assert not status.can_proceed or len(status.violations) > 0


# =============================================================================
# TEST: COMPONENT INTEROP
# =============================================================================


class TestComponentInterop:
    """Tests verifying clean interoperability between autonomous components."""

    @pytest.mark.asyncio
    async def test_escalation_dedup_across_subsystems(
        self,
        escalation_manager,
    ):
        """
        Multiple subsystems raising the same escalation should be
        deduplicated within the configured window.
        """
        # First escalation from resource monitor
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.RESOURCE_EXHAUSTED,
            title="High memory usage",
            message="Memory at 85%",
        )

        # Duplicate from heartbeat check (same reason+title+message = dedup)
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.RESOURCE_EXHAUSTED,
            title="High memory usage",
            message="Memory at 85%",
        )

        # Should be deduplicated — only one pending escalation
        pending = escalation_manager.get_pending()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_goal_store_persistence_round_trip(self, tmp_path):
        """
        Goals persisted to disk can be loaded in a new GoalManager.
        """
        store_path = str(tmp_path / "persist_test.json")

        # Session 1: Create goal (auto-persisted via set_primary_goal)
        store1 = GoalStore(store_path)
        mgr1 = GoalManager(storage=store1)
        goal = await mgr1.set_primary_goal(
            description="Persistent goal test",
            priority=GoalPriority.HIGH,
            success_criteria=[
                SuccessCriterion(
                    description="Metric reached",
                    metric_name="metric",
                    target_value=1.0,
                    comparator=">=",
                ),
            ],
        )
        goal_id = goal.id

        # Session 2: Load and verify via initialize()
        store2 = GoalStore(store_path)
        mgr2 = GoalManager(storage=store2)
        await mgr2.initialize()

        loaded = await mgr2.get_goal(goal_id)
        assert loaded is not None
        assert loaded.description == "Persistent goal test"
        assert loaded.priority == GoalPriority.HIGH
        assert len(loaded.success_criteria) == 1
        assert loaded.success_criteria[0].metric_name == "metric"

    @pytest.mark.asyncio
    async def test_heartbeat_task_decorator_in_integration(
        self,
        heartbeat_manager,
    ):
        """
        The @heartbeat_task decorator should work seamlessly with
        HeartbeatManager registration.
        """
        invocations = []

        @heartbeat_task(interval=timedelta(seconds=2), name="decorated_check")
        async def decorated_check():
            invocations.append(datetime.utcnow())
            return "ok"

        heartbeat_manager.register(decorated_check)
        assert "decorated_check" in heartbeat_manager.list_tasks()

        # Force run
        decorated_check.last_run = None
        await heartbeat_manager.tick()

        assert len(invocations) == 1

    @pytest.mark.asyncio
    async def test_resource_usage_recording(self, resource_monitor):
        """
        ResourceMonitor should track API token/cost usage and
        enforce limits.
        """
        # Record some token usage
        resource_monitor.record_usage(tokens=5_000)
        resource_monitor.record_usage(tokens=3_000, cost_usd=0.15)

        # Verify internal counters accumulated correctly
        assert resource_monitor._hourly_tokens >= 8_000
        assert resource_monitor._hourly_cost >= 0.15

        # Verify usage appears in resource snapshot (mock _get_usage
        # to avoid psutil dependency in this focused test)
        resource_monitor._get_usage = AsyncMock(
            return_value=ResourceUsage(
                tokens_this_hour=resource_monitor._hourly_tokens,
                cost_this_hour_usd=resource_monitor._hourly_cost,
            )
        )
        status = await resource_monitor._check_resources()
        assert status.usage.tokens_this_hour >= 8_000
        assert status.usage.cost_this_hour_usd >= 0.15

    @pytest.mark.asyncio
    async def test_multi_goal_priority_scheduling(self, goal_manager):
        """
        GoalManager should schedule goals by priority and status.
        """
        # Create goals at different priorities (set_primary_goal creates
        # top-level active goals; multiple calls are fine for this test)
        await goal_manager.set_primary_goal(
            description="Low priority task",
            priority=GoalPriority.LOW,
        )
        await goal_manager.set_primary_goal(
            description="High priority task",
            priority=GoalPriority.HIGH,
        )
        await goal_manager.set_primary_goal(
            description="Critical priority task",
            priority=GoalPriority.CRITICAL,
        )

        # Get next actionable should favor highest priority
        next_goal = await goal_manager.get_next_actionable()
        assert next_goal is not None
        assert next_goal.priority == GoalPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_escalation_lifecycle_states(self, escalation_manager):
        """
        Escalation should transition through states:
        pending → acknowledged → resolved
        """
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.RESOURCE_EXHAUSTED,
            title="Disk space warning",
            message="Less than 2GB free",
        )

        pending = escalation_manager.get_pending()
        assert len(pending) == 1
        esc = pending[0]

        # Initially pending
        assert esc.acknowledged_at is None
        assert esc.resolved_at is None

        # Acknowledge
        escalation_manager.acknowledge(esc.id)
        esc_ack = escalation_manager.get_escalation(esc.id)
        assert esc_ack is not None
        assert esc_ack.acknowledged_at is not None
        assert esc_ack.resolved_at is None

        # Resolve via respond (sets human_response + resolved_at)
        escalation_manager.respond(esc.id, response="Cleaned up temp files")
        esc_resolved = escalation_manager.get_escalation(esc.id)
        assert esc_resolved is not None
        assert esc_resolved.resolved_at is not None
        assert esc_resolved.human_response == "Cleaned up temp files"

        # Should no longer be pending
        assert len(escalation_manager.get_pending()) == 0


# =============================================================================
# TEST: CONFIGURATION INTEGRATION
# =============================================================================


class TestConfigIntegration:
    """Tests for autonomous configuration loading and application."""

    def test_default_config_loads(self):
        """Default AutonomousConfig should load without errors."""
        from llmcore.config.autonomous_config import AutonomousConfig

        config = AutonomousConfig()
        assert config.enabled is True
        assert config.goals.max_sub_goals == 10
        assert config.heartbeat.base_interval == 60.0
        assert config.resources.max_cpu_percent == 80.0
        assert config.escalation.auto_resolve_below == "advisory"

    def test_config_override(self):
        """Config values can be overridden."""
        from llmcore.config.autonomous_config import (
            AutonomousConfig,
            ResourcesConfig,
        )

        config = AutonomousConfig(
            resources=ResourcesConfig(
                max_cpu_percent=60.0,
                max_temperature_c=65.0,
            ),
        )
        assert config.resources.max_cpu_percent == 60.0
        assert config.resources.max_temperature_c == 65.0
        # Other defaults preserved
        assert config.resources.max_memory_percent == 80.0

    def test_config_validation_rejects_invalid(self):
        """Config validation should reject invalid values."""
        from llmcore.config.autonomous_config import EscalationConfig

        with pytest.raises(ValueError, match="Invalid escalation level"):
            EscalationConfig(auto_resolve_below="nonexistent")

    def test_config_from_dict(self):
        """Config can be loaded from a TOML-like dictionary."""
        from llmcore.config.autonomous_config import load_autonomous_config

        data = {
            "autonomous": {
                "enabled": True,
                "goals": {"max_sub_goals": 5},
                "resources": {"max_cpu_percent": 50.0},
            }
        }
        config = load_autonomous_config(config_dict=data)
        assert config.goals.max_sub_goals == 5
        assert config.resources.max_cpu_percent == 50.0

    def test_config_applies_to_resource_constraints(self):
        """
        Config values should translate to ResourceConstraints
        used by ResourceMonitor.
        """
        from llmcore.config.autonomous_config import ResourcesConfig

        cfg = ResourcesConfig(
            max_cpu_percent=70.0,
            max_memory_percent=75.0,
            max_hourly_tokens=50_000,
        )

        constraints = ResourceConstraints(
            max_cpu_percent=cfg.max_cpu_percent,
            max_memory_percent=cfg.max_memory_percent,
            max_hourly_tokens=cfg.max_hourly_tokens,
        )

        assert constraints.max_cpu_percent == 70.0
        assert constraints.max_memory_percent == 75.0
        assert constraints.max_hourly_tokens == 50_000

    def test_config_toml_file_load(self, tmp_path):
        """Config can be loaded from a TOML file."""
        from llmcore.config.autonomous_config import load_autonomous_config

        toml_content = """
[autonomous]
enabled = true

[autonomous.goals]
max_sub_goals = 7
max_goal_depth = 3

[autonomous.resources]
max_cpu_percent = 65.0
max_daily_cost_usd = 5.0
"""
        toml_file = tmp_path / "test_config.toml"
        toml_file.write_text(toml_content)

        config = load_autonomous_config(config_path=toml_file)
        assert config.goals.max_sub_goals == 7
        assert config.goals.max_goal_depth == 3
        assert config.resources.max_cpu_percent == 65.0
        assert config.resources.max_daily_cost_usd == 5.0
        # Defaults for unset values
        assert config.heartbeat.enabled is True

    def test_config_file_not_found(self, tmp_path):
        """Missing config file should raise FileNotFoundError."""
        from llmcore.config.autonomous_config import load_autonomous_config

        with pytest.raises(FileNotFoundError):
            load_autonomous_config(config_path=tmp_path / "nonexistent.toml")

    def test_context_config_validation(self):
        """Context config validates strategy values."""
        from llmcore.config.autonomous_config import ContextConfig

        # Valid strategies
        for strategy in ("recency_relevance", "relevance_only", "recency_only"):
            config = ContextConfig(prioritization_strategy=strategy)
            assert config.prioritization_strategy == strategy

        # Invalid strategy
        with pytest.raises(ValueError, match="Invalid strategy"):
            ContextConfig(prioritization_strategy="invalid_strategy")
