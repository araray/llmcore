# tests/autonomous/test_goals.py
"""
Tests for the Goal Management System.

Covers:
- GoalStatus and GoalPriority enums
- SuccessCriterion data model and comparators
- Goal data model (creation, serialization, actionability)
- GoalStore persistence (JSON file-based)
- GoalManager lifecycle (create, decompose, progress, failure)
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.autonomous.goals import (
    Goal,
    GoalManager,
    GoalPriority,
    GoalStatus,
    GoalStore,
    SuccessCriterion,
)

# =============================================================================
# SuccessCriterion Tests
# =============================================================================


class TestSuccessCriterion:
    """Tests for SuccessCriterion data model."""

    def test_create_criterion(self):
        """Test basic creation."""
        criterion = SuccessCriterion(
            description="Reach rank #1",
            metric_name="rank",
            target_value=1,
            comparator="==",
        )
        assert criterion.description == "Reach rank #1"
        assert criterion.metric_name == "rank"
        assert criterion.target_value == 1
        assert criterion.current_value is None

    def test_is_met_equal(self):
        """Test == comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=10,
            comparator="==",
            current_value=10,
        )
        assert c.is_met() is True
        c.current_value = 9
        assert c.is_met() is False

    def test_is_met_gte(self):
        """Test >= comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=100,
            comparator=">=",
            current_value=100,
        )
        assert c.is_met() is True
        c.current_value = 101
        assert c.is_met() is True
        c.current_value = 99
        assert c.is_met() is False

    def test_is_met_gt(self):
        """Test > comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=50,
            comparator=">",
            current_value=51,
        )
        assert c.is_met() is True
        c.current_value = 50
        assert c.is_met() is False

    def test_is_met_lte(self):
        """Test <= comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=5,
            comparator="<=",
            current_value=5,
        )
        assert c.is_met() is True
        c.current_value = 4
        assert c.is_met() is True
        c.current_value = 6
        assert c.is_met() is False

    def test_is_met_lt(self):
        """Test < comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=10,
            comparator="<",
            current_value=9,
        )
        assert c.is_met() is True
        c.current_value = 10
        assert c.is_met() is False

    def test_is_met_contains(self):
        """Test contains comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="tags",
            target_value="important",
            comparator="contains",
            current_value=["important", "urgent"],
        )
        assert c.is_met() is True
        c.current_value = ["normal"]
        assert c.is_met() is False

    def test_is_met_not_contains(self):
        """Test not_contains comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="errors",
            target_value="critical",
            comparator="not_contains",
            current_value=["minor"],
        )
        assert c.is_met() is True
        c.current_value = ["critical"]
        assert c.is_met() is False

    def test_is_met_none_current(self):
        """Test that None current_value means not met."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=10,
            comparator=">=",
        )
        assert c.is_met() is False

    def test_progress_percentage_gte(self):
        """Test progress for >= comparator."""
        c = SuccessCriterion(
            description="test",
            metric_name="score",
            target_value=100,
            comparator=">=",
            current_value=50,
        )
        assert c.progress_percentage() == pytest.approx(0.5)
        c.current_value = 100
        assert c.progress_percentage() == pytest.approx(1.0)
        c.current_value = 150
        assert c.progress_percentage() == pytest.approx(1.0)  # capped

    def test_progress_percentage_none(self):
        """Test progress with None current."""
        c = SuccessCriterion(
            description="test",
            metric_name="x",
            target_value=10,
            comparator=">=",
        )
        assert c.progress_percentage() == 0.0

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = SuccessCriterion(
            description="Reach rank #1",
            metric_name="rank",
            target_value=1,
            comparator="==",
            current_value=5,
        )
        data = original.to_dict()
        restored = SuccessCriterion.from_dict(data)
        assert restored.description == original.description
        assert restored.metric_name == original.metric_name
        assert restored.target_value == original.target_value
        assert restored.current_value == original.current_value
        assert restored.comparator == original.comparator


# =============================================================================
# Goal Tests
# =============================================================================


class TestGoal:
    """Tests for Goal data model."""

    def test_create_goal(self):
        """Test goal factory method."""
        goal = Goal.create(
            description="Test goal",
            priority=GoalPriority.HIGH,
        )
        assert goal.description == "Test goal"
        assert goal.priority == GoalPriority.HIGH
        assert goal.status == GoalStatus.PENDING
        assert goal.id.startswith("goal_")
        assert goal.progress == 0.0

    def test_is_actionable_pending(self):
        """Test actionability of pending goal."""
        goal = Goal.create("test")
        assert goal.is_actionable() is True

    def test_is_actionable_completed(self):
        """Test that completed goals are not actionable."""
        goal = Goal.create("test")
        goal.status = GoalStatus.COMPLETED
        assert goal.is_actionable() is False

    def test_is_actionable_max_attempts(self):
        """Test that max-attempts goals are not actionable."""
        goal = Goal.create("test")
        goal.max_attempts = 3
        goal.attempts = 3
        assert goal.is_actionable() is False

    def test_is_actionable_cooldown(self):
        """Test that goals in cooldown are not actionable."""
        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        goal.cooldown_until = datetime.utcnow() + timedelta(minutes=5)
        assert goal.is_actionable() is False

    def test_is_leaf(self):
        """Test leaf detection."""
        goal = Goal.create("test")
        assert goal.is_leaf() is True
        goal.sub_goal_ids.append("child_1")
        assert goal.is_leaf() is False

    def test_apply_cooldown(self):
        """Test exponential backoff cooldown."""
        goal = Goal.create("test")
        goal.apply_cooldown(base_seconds=10)
        assert goal.cooldown_until is not None
        assert goal.cooldown_multiplier == 2.0  # doubled

        # Apply again — multiplier should double
        goal.apply_cooldown(base_seconds=10)
        assert goal.cooldown_multiplier == 4.0

    def test_reset_cooldown(self):
        """Test cooldown reset after success."""
        goal = Goal.create("test")
        goal.apply_cooldown(base_seconds=10)
        goal.reset_cooldown()
        assert goal.cooldown_until is None
        assert goal.cooldown_multiplier == 1.0

    def test_update_progress_all_met(self):
        """Test that goal completes when all criteria met."""
        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        goal.success_criteria = [
            SuccessCriterion(
                description="a",
                metric_name="x",
                target_value=10,
                comparator=">=",
                current_value=10,
            ),
            SuccessCriterion(
                description="b",
                metric_name="y",
                target_value=5,
                comparator=">=",
                current_value=5,
            ),
        ]
        goal.update_progress()
        assert goal.progress == pytest.approx(1.0)
        assert goal.status == GoalStatus.COMPLETED
        assert goal.completed_at is not None

    def test_update_progress_partial(self):
        """Test partial progress calculation."""
        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        goal.success_criteria = [
            SuccessCriterion(
                description="a",
                metric_name="x",
                target_value=100,
                comparator=">=",
                current_value=50,
            ),
        ]
        goal.update_progress()
        assert goal.progress == pytest.approx(0.5)
        assert goal.status == GoalStatus.ACTIVE

    def test_serialization_roundtrip(self):
        """Test full Goal to_dict/from_dict roundtrip."""
        original = Goal.create(
            description="Test roundtrip",
            priority=GoalPriority.CRITICAL,
        )
        original.status = GoalStatus.ACTIVE
        original.progress = 0.5
        original.tags = ["test", "roundtrip"]
        original.context = {"key": "value"}
        original.sub_goal_ids = ["child_1"]
        original.success_criteria = [
            SuccessCriterion(
                description="metric",
                metric_name="x",
                target_value=10,
                comparator=">=",
                current_value=5,
            )
        ]

        data = original.to_dict()
        restored = Goal.from_dict(data)

        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.priority == original.priority
        assert restored.status == original.status
        assert restored.progress == original.progress
        assert restored.tags == original.tags
        assert restored.context == original.context
        assert restored.sub_goal_ids == original.sub_goal_ids
        assert len(restored.success_criteria) == 1
        assert restored.success_criteria[0].metric_name == "x"


# =============================================================================
# GoalStore Tests
# =============================================================================


class TestGoalStore:
    """Tests for JSON-based goal persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_goals_path):
        """Test basic save and load."""
        store = GoalStore(tmp_goals_path)
        goal = Goal.create("test persistence")

        await store.save_goal(goal)
        goals = await store.load_goals()

        assert len(goals) == 1
        assert goals[0].id == goal.id
        assert goals[0].description == "test persistence"

    @pytest.mark.asyncio
    async def test_load_empty(self, tmp_goals_path):
        """Test loading when no file exists."""
        store = GoalStore(tmp_goals_path)
        goals = await store.load_goals()
        assert goals == []

    @pytest.mark.asyncio
    async def test_upsert(self, tmp_goals_path):
        """Test that saving same goal twice updates it."""
        store = GoalStore(tmp_goals_path)
        goal = Goal.create("test upsert")

        await store.save_goal(goal)
        goal.progress = 0.5
        await store.save_goal(goal)

        goals = await store.load_goals()
        assert len(goals) == 1
        assert goals[0].progress == 0.5

    @pytest.mark.asyncio
    async def test_delete(self, tmp_goals_path):
        """Test goal deletion."""
        store = GoalStore(tmp_goals_path)
        goal1 = Goal.create("goal 1")
        goal2 = Goal.create("goal 2")

        await store.save_goal(goal1)
        await store.save_goal(goal2)
        await store.delete_goal(goal1.id)

        goals = await store.load_goals()
        assert len(goals) == 1
        assert goals[0].id == goal2.id

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_goals_path):
        """Test deleting a non-existent goal (should be safe)."""
        store = GoalStore(tmp_goals_path)
        await store.delete_goal("nonexistent_id")  # Should not raise

    @pytest.mark.asyncio
    async def test_atomic_write(self, tmp_goals_path):
        """Test that writes are atomic (temp file then rename)."""
        store = GoalStore(tmp_goals_path)
        goal = Goal.create("test atomic")
        await store.save_goal(goal)

        # File should exist and be valid JSON
        path = Path(tmp_goals_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "goals" in data
        assert len(data["goals"]) == 1


# =============================================================================
# GoalManager Tests
# =============================================================================


class TestGoalManager:
    """Tests for GoalManager lifecycle and operations."""

    @pytest.mark.asyncio
    async def test_initialize_empty(self, goal_store):
        """Test initialization with empty store."""
        manager = GoalManager(goal_store)
        await manager.initialize()
        goals = await manager.get_all_goals()
        assert goals == []

    @pytest.mark.asyncio
    async def test_set_primary_goal(self, goal_store):
        """Test creating a primary goal."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = await manager.set_primary_goal(
            description="Be the best",
            success_criteria=[
                SuccessCriterion(
                    description="Rank #1",
                    metric_name="rank",
                    target_value=1,
                    comparator="==",
                )
            ],
        )

        assert goal.description == "Be the best"
        assert goal.status == GoalStatus.ACTIVE
        assert goal.parent_id is None
        assert len(goal.success_criteria) == 1

        # Should be persisted
        loaded = await goal_store.load_goals()
        assert len(loaded) >= 1

    @pytest.mark.asyncio
    async def test_get_next_actionable(self, goal_store):
        """Test priority-based task selection."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        # Create goals with different priorities
        low = Goal.create("low priority", priority=GoalPriority.LOW)
        low.status = GoalStatus.ACTIVE
        high = Goal.create("high priority", priority=GoalPriority.HIGH)
        high.status = GoalStatus.ACTIVE

        manager._goals[low.id] = low
        manager._goals[high.id] = high

        next_goal = await manager.get_next_actionable()
        assert next_goal is not None
        assert next_goal.id == high.id  # Higher priority first

    @pytest.mark.asyncio
    async def test_get_next_actionable_active_before_pending(self, goal_store):
        """Test that ACTIVE goals come before PENDING at same priority."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        pending = Goal.create("pending", priority=GoalPriority.NORMAL)
        pending.status = GoalStatus.PENDING
        active = Goal.create("active", priority=GoalPriority.NORMAL)
        active.status = GoalStatus.ACTIVE

        manager._goals[pending.id] = pending
        manager._goals[active.id] = active

        next_goal = await manager.get_next_actionable()
        assert next_goal is not None
        assert next_goal.id == active.id

    @pytest.mark.asyncio
    async def test_get_next_actionable_skips_parent(self, goal_store):
        """Test that non-leaf goals are skipped."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        parent = Goal.create("parent", priority=GoalPriority.HIGH)
        parent.status = GoalStatus.ACTIVE
        parent.sub_goal_ids = ["child_1"]

        child = Goal.create("child", priority=GoalPriority.NORMAL)
        child.id = "child_1"
        child.status = GoalStatus.ACTIVE
        child.parent_id = parent.id

        manager._goals[parent.id] = parent
        manager._goals[child.id] = child

        next_goal = await manager.get_next_actionable()
        assert next_goal is not None
        assert next_goal.id == child.id  # Leaf goal

    @pytest.mark.asyncio
    async def test_report_success(self, goal_store):
        """Test success reporting and progress update."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.report_success(goal.id, progress_delta=0.3)

        updated = manager._goals[goal.id]
        assert updated.progress == pytest.approx(0.3)
        assert updated.attempts == 1
        assert updated.cooldown_until is None

    @pytest.mark.asyncio
    async def test_report_failure_recoverable(self, goal_store):
        """Test recoverable failure with cooldown."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.report_failure(goal.id, "transient error")

        updated = manager._goals[goal.id]
        assert updated.attempts == 1
        assert len(updated.failure_reasons) == 1
        assert "transient error" in updated.failure_reasons[0]
        assert updated.cooldown_until is not None

    @pytest.mark.asyncio
    async def test_report_failure_terminal(self, goal_store):
        """Test terminal failure sets FAILED status."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.report_failure(goal.id, "permanent error", recoverable=False)

        updated = manager._goals[goal.id]
        assert updated.status == GoalStatus.FAILED

    @pytest.mark.asyncio
    async def test_report_failure_max_attempts(self, goal_store):
        """Test that exceeding max attempts blocks goal."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        goal.max_attempts = 3
        goal.attempts = 2
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.report_failure(goal.id, "attempt 3")

        updated = manager._goals[goal.id]
        assert updated.status == GoalStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_update_metric(self, goal_store):
        """Test updating a success criterion metric."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        goal.success_criteria = [
            SuccessCriterion(
                description="reach rank",
                metric_name="rank",
                target_value=1,
                comparator="<=",
            )
        ]
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.update_metric(goal.id, "rank", 5)

        updated = manager._goals[goal.id]
        assert updated.success_criteria[0].current_value == 5

    @pytest.mark.asyncio
    async def test_pause_resume(self, goal_store):
        """Test pause and resume lifecycle."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.pause_goal(goal.id)
        assert manager._goals[goal.id].status == GoalStatus.PAUSED

        await manager.resume_goal(goal.id)
        assert manager._goals[goal.id].status == GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_abandon_goal(self, goal_store):
        """Test goal abandonment."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.abandon_goal(goal.id)
        assert manager._goals[goal.id].status == GoalStatus.ABANDONED

    @pytest.mark.asyncio
    async def test_delete_goal(self, goal_store):
        """Test goal deletion."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        manager._goals[goal.id] = goal
        await goal_store.save_goal(goal)

        await manager.delete_goal(goal.id)
        assert goal.id not in manager._goals

    @pytest.mark.asyncio
    async def test_progress_propagation(self, goal_store):
        """Test that child progress propagates to parent."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        parent = Goal.create("parent", priority=GoalPriority.HIGH)
        parent.status = GoalStatus.ACTIVE

        child1 = Goal.create("child1")
        child1.status = GoalStatus.ACTIVE
        child1.parent_id = parent.id

        child2 = Goal.create("child2")
        child2.status = GoalStatus.ACTIVE
        child2.parent_id = parent.id

        parent.sub_goal_ids = [child1.id, child2.id]

        manager._goals[parent.id] = parent
        manager._goals[child1.id] = child1
        manager._goals[child2.id] = child2
        await goal_store.save_goal(parent)
        await goal_store.save_goal(child1)
        await goal_store.save_goal(child2)

        # Report progress on child1
        await manager.report_success(child1.id, progress_delta=1.0)

        # Parent should have ~0.5 progress (average of children)
        assert manager._goals[parent.id].progress == pytest.approx(0.5, abs=0.1)

    @pytest.mark.asyncio
    async def test_get_status_summary(self, goal_store):
        """Test status summary generation."""
        manager = GoalManager(goal_store)
        await manager.initialize()

        goal = Goal.create("test")
        goal.status = GoalStatus.ACTIVE
        manager._goals[goal.id] = goal

        summary = manager.get_status_summary()
        assert "total_goals" in summary
        assert summary["total_goals"] == 1
        assert "status_counts" in summary
        assert summary["status_counts"]["ACTIVE"] == 1


# =============================================================================
# GoalManager.from_config Tests
# =============================================================================


class TestGoalManagerFromConfig:
    """Tests for GoalManager.from_config() factory method."""

    @pytest.mark.asyncio
    async def test_from_config_creates_manager(self, goals_config):
        """Test that from_config creates a working GoalManager."""
        manager = GoalManager.from_config(goals_config)
        await manager.initialize()

        goals = await manager.get_all_goals()
        assert goals == []

    @pytest.mark.asyncio
    async def test_from_config_uses_storage_path(self, tmp_path):
        """Test that from_config creates GoalStore at configured path."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        custom_path = str(tmp_path / "custom" / "goals.json")
        config = GoalsAutonomousConfig(storage_path=custom_path)
        manager = GoalManager.from_config(config)
        await manager.initialize()

        # Create and persist a goal
        goal = await manager.set_primary_goal("test persistence", auto_decompose=False)

        # Verify file was written at the configured path
        assert Path(custom_path).exists()

        # Verify goal survives round-trip
        manager2 = GoalManager.from_config(config)
        await manager2.initialize()
        loaded = await manager2.get_all_goals()
        assert len(loaded) == 1
        assert loaded[0].id == goal.id

    @pytest.mark.asyncio
    async def test_from_config_wires_auto_decompose(self, tmp_goals_path):
        """Test that auto_decompose config is respected."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(storage_path=tmp_goals_path, auto_decompose=False)
        manager = GoalManager.from_config(config)
        await manager.initialize()

        assert manager._default_auto_decompose is False

    @pytest.mark.asyncio
    async def test_from_config_wires_max_attempts(self, tmp_goals_path):
        """Test that max_attempts config flows into created goals."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(
            storage_path=tmp_goals_path,
            max_attempts_per_goal=5,
            auto_decompose=False,
        )
        manager = GoalManager.from_config(config)
        await manager.initialize()

        goal = await manager.set_primary_goal("test max attempts")
        assert goal.max_attempts == 5

    @pytest.mark.asyncio
    async def test_from_config_wires_cooldown(self, tmp_goals_path):
        """Test that base_cooldown config flows into failure handling."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(
            storage_path=tmp_goals_path,
            base_cooldown_seconds=120.0,
            auto_decompose=False,
        )
        manager = GoalManager.from_config(config)
        await manager.initialize()

        assert manager._base_cooldown == 120.0

    @pytest.mark.asyncio
    async def test_from_config_wires_sub_goal_limits(self, tmp_goals_path):
        """Test that sub-goal limits are wired from config."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(
            storage_path=tmp_goals_path,
            max_sub_goals=3,
            max_goal_depth=2,
        )
        manager = GoalManager.from_config(config)

        assert manager._max_sub_goals == 3
        assert manager._max_goal_depth == 2

    @pytest.mark.asyncio
    async def test_from_config_with_custom_storage(self, tmp_goals_path):
        """Test that storage override is respected."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(storage_path=tmp_goals_path)

        custom_store = GoalStore(tmp_goals_path)
        manager = GoalManager.from_config(config, storage=custom_store)

        assert manager.storage is custom_store

    @pytest.mark.asyncio
    async def test_from_config_with_llm_provider(self, goals_config):
        """Test that llm_provider and model are passed through."""

        mock_llm = MagicMock()
        manager = GoalManager.from_config(
            goals_config,
            llm_provider=mock_llm,
            decomposition_model="gpt-4",
        )

        assert manager.llm_provider is mock_llm
        assert manager.decomposition_model == "gpt-4"

    @pytest.mark.asyncio
    async def test_auto_decompose_explicit_overrides_config(self, tmp_goals_path):
        """Test that explicit auto_decompose=True overrides config False."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(storage_path=tmp_goals_path, auto_decompose=False)
        mock_llm = MagicMock()

        manager = GoalManager.from_config(config, llm_provider=mock_llm)
        await manager.initialize()

        # Mock the internal decomposition to avoid provider import
        manager._decompose_goal = AsyncMock(return_value=[])

        # Config says no auto-decompose, but explicit True overrides
        await manager.set_primary_goal("test override", auto_decompose=True)

        # Decomposition should have been attempted because explicit True won
        manager._decompose_goal.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_decompose_none_uses_config(self, tmp_goals_path):
        """Test that omitting auto_decompose uses config default."""
        from llmcore.config.autonomous_config import GoalsAutonomousConfig

        config = GoalsAutonomousConfig(storage_path=tmp_goals_path, auto_decompose=False)
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        manager = GoalManager.from_config(config, llm_provider=mock_llm)
        await manager.initialize()

        # Don't pass auto_decompose — should use config (False)
        await manager.set_primary_goal("test config default")

        # LLM should NOT have been called because config says False
        mock_llm.complete.assert_not_called()
