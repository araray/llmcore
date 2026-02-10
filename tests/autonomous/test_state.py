# tests/autonomous/test_state.py
"""
Tests for the Autonomous State Persistence System.

Covers:
- AutonomousState dataclass: creation, defaults, serialization roundtrip
- StateManager lifecycle: load, save, checkpoint, reset, get_summary
- Atomic file writes: crash safety via tmp→rename
- Checkpoint accumulation and rolling window (max 100)
- Cost and token tracking across iterations
- Edge cases: missing files, corrupt JSON, empty files
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.autonomous.state import AutonomousState, StateManager

# =============================================================================
# AutonomousState Data Model Tests
# =============================================================================


class TestAutonomousState:
    """Tests for the AutonomousState dataclass."""

    def test_default_creation(self):
        """All fields have sensible defaults."""
        state = AutonomousState()
        assert state.session_id == ""
        assert state.iteration == 0
        assert state.phase == ""
        assert state.solver == ""
        assert state.goal_snapshot == {}
        assert state.extra == {}
        assert state.error_count == 0
        assert state.total_tokens_used == 0
        assert state.total_cost_usd == 0.0
        assert state.checkpoints == []

    def test_custom_creation(self):
        """All custom field values are preserved."""
        state = AutonomousState(
            session_id="sess-42",
            iteration=10,
            phase="act",
            solver="hybrid",
            goal_snapshot={"id": "g-1", "progress": 0.5},
            error_count=3,
            total_tokens_used=50_000,
            total_cost_usd=0.75,
        )
        assert state.iteration == 10
        assert state.phase == "act"
        assert state.solver == "hybrid"
        assert state.goal_snapshot["id"] == "g-1"
        assert state.error_count == 3
        assert state.total_tokens_used == 50_000
        assert state.total_cost_usd == 0.75

    def test_to_dict(self):
        """to_dict() returns JSON-safe plain dict."""
        state = AutonomousState(session_id="s1", iteration=5, phase="reflect")
        d = state.to_dict()
        assert d["session_id"] == "s1"
        assert d["iteration"] == 5
        assert d["phase"] == "reflect"
        # Ensure JSON-serializable
        json.dumps(d)

    def test_from_dict(self):
        """from_dict() reconstructs state from plain dict."""
        d = {"session_id": "s2", "iteration": 7, "phase": "act", "solver": "cautious"}
        state = AutonomousState.from_dict(d)
        assert state.session_id == "s2"
        assert state.iteration == 7
        assert state.solver == "cautious"

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict() silently ignores unexpected keys."""
        d = {"session_id": "s3", "unknown_field": "ignored", "iteration": 1}
        state = AutonomousState.from_dict(d)
        assert state.session_id == "s3"
        assert state.iteration == 1

    def test_is_empty(self):
        """is_empty property checks for fresh state."""
        assert AutonomousState().is_empty is True
        assert AutonomousState(session_id="s1").is_empty is False

    def test_roundtrip(self):
        """to_dict → from_dict preserves all fields."""
        original = AutonomousState(
            session_id="round",
            iteration=42,
            phase="reflect",
            solver="hybrid",
            goal_snapshot={"id": "g-99", "progress": 0.8},
            extra={"custom": True},
            error_count=2,
            total_tokens_used=10000,
            total_cost_usd=0.15,
            checkpoints=["2026-01-01T00:00:00Z"],
        )
        restored = AutonomousState.from_dict(original.to_dict())
        assert restored.session_id == original.session_id
        assert restored.iteration == original.iteration
        assert restored.goal_snapshot == original.goal_snapshot
        assert restored.extra == original.extra
        assert restored.checkpoints == original.checkpoints


# =============================================================================
# StateManager Tests
# =============================================================================


class TestStateManager:
    """Tests for StateManager persistence operations."""

    @pytest.fixture
    def state_path(self, tmp_path):
        """Temporary path for state file."""
        return str(tmp_path / "session_state.json")

    @pytest.fixture
    def mgr(self, state_path):
        """StateManager with temporary path."""
        return StateManager(state_path=state_path)

    # ---- Construction ----

    def test_construction(self, mgr, state_path):
        """StateManager initializes with resolved path."""
        assert str(mgr.path) == state_path
        assert mgr.state.is_empty

    def test_path_expansion(self, tmp_path):
        """StateManager expands ~ and $VARS in path."""
        with patch.dict(os.environ, {"TEST_DIR": str(tmp_path)}):
            mgr = StateManager(state_path="$TEST_DIR/state.json")
            assert str(tmp_path) in str(mgr.path)

    def test_default_path(self):
        """Omitting state_path uses a default."""
        mgr = StateManager()
        assert "autonomous_state.json" in str(mgr.path)

    # ---- Save and Load ----

    @pytest.mark.asyncio
    async def test_save_creates_file(self, mgr, state_path):
        """save() creates a new JSON file."""
        await mgr.save()
        assert Path(state_path).exists()

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, state_path):
        """Saved state can be loaded back identically."""
        mgr = StateManager(state_path=state_path)
        await mgr.checkpoint(
            session_id="test-session",
            iteration=42,
            phase="reflect",
            solver="cautious",
            tokens_used=10_000,
            cost_usd=0.15,
        )

        mgr2 = StateManager(state_path=state_path)
        await mgr2.load()
        assert mgr2.state.session_id == "test-session"
        assert mgr2.state.iteration == 42
        assert mgr2.state.phase == "reflect"
        assert mgr2.state.solver == "cautious"
        assert mgr2.state.total_tokens_used == 10_000
        assert mgr2.state.total_cost_usd == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, mgr):
        """load() on missing file returns empty state (no crash)."""
        state = await mgr.load()
        assert state.iteration == 0
        assert state.phase == ""

    @pytest.mark.asyncio
    async def test_load_corrupt_json(self, state_path):
        """load() handles corrupt JSON gracefully."""
        Path(state_path).write_text("{not valid json")
        mgr = StateManager(state_path=state_path)
        state = await mgr.load()
        assert state.iteration == 0

    @pytest.mark.asyncio
    async def test_load_empty_file(self, state_path):
        """load() handles empty file gracefully."""
        Path(state_path).write_text("")
        mgr = StateManager(state_path=state_path)
        state = await mgr.load()
        assert state.iteration == 0

    @pytest.mark.asyncio
    async def test_save_creates_parent_dirs(self, tmp_path):
        """save() creates intermediate directories if needed."""
        deep_path = str(tmp_path / "a" / "b" / "c" / "state.json")
        mgr = StateManager(state_path=deep_path)
        await mgr.save()
        assert Path(deep_path).exists()

    # ---- Checkpoint ----

    @pytest.mark.asyncio
    async def test_checkpoint(self, mgr):
        """checkpoint() records iteration, phase, and solver."""
        state = await mgr.checkpoint(
            session_id="s1",
            iteration=1,
            phase="act",
            solver="confident",
            goal_snapshot={"progress": 0.3},
            tokens_used=2000,
            cost_usd=0.03,
        )
        assert state.session_id == "s1"
        assert state.iteration == 1
        assert state.phase == "act"
        assert state.solver == "confident"
        assert state.total_tokens_used == 2000
        assert state.total_cost_usd == pytest.approx(0.03)
        assert len(state.checkpoints) == 1
        assert state.started_at != ""

    @pytest.mark.asyncio
    async def test_checkpoint_accumulates_tokens_and_cost(self, mgr):
        """Multiple checkpoints sum tokens and costs."""
        await mgr.checkpoint(iteration=1, phase="act", tokens_used=1000, cost_usd=0.01)
        await mgr.checkpoint(iteration=2, phase="reflect", tokens_used=2000, cost_usd=0.02)
        await mgr.checkpoint(iteration=3, phase="act", tokens_used=500, cost_usd=0.005)

        assert mgr.state.total_tokens_used == 3500
        assert mgr.state.total_cost_usd == pytest.approx(0.035)
        assert mgr.state.iteration == 3
        assert len(mgr.state.checkpoints) == 3

    @pytest.mark.asyncio
    async def test_checkpoint_error_tracking(self, mgr):
        """error=True increments error_count."""
        await mgr.checkpoint(iteration=1, phase="act", error=True)
        await mgr.checkpoint(iteration=2, phase="act", error=True)
        await mgr.checkpoint(iteration=3, phase="act", error=False)
        assert mgr.state.error_count == 2

    @pytest.mark.asyncio
    async def test_checkpoint_rolling_window(self, mgr):
        """Checkpoints are capped at most recent 100."""
        for i in range(120):
            await mgr.checkpoint(iteration=i, phase="act", tokens_used=10, cost_usd=0.001)

        assert len(mgr.state.checkpoints) == 100
        assert mgr.state.total_tokens_used == 1200

    @pytest.mark.asyncio
    async def test_checkpoint_persists(self, mgr, state_path):
        """checkpoint() auto-saves to disk."""
        await mgr.checkpoint(
            iteration=5, phase="reflect", solver="cautious", tokens_used=500, cost_usd=0.01
        )
        data = json.loads(Path(state_path).read_text())
        assert data["iteration"] == 5
        assert data["phase"] == "reflect"

    @pytest.mark.asyncio
    async def test_checkpoint_extra_merges(self, mgr):
        """Extra metadata merges across checkpoints."""
        await mgr.checkpoint(iteration=1, extra={"key1": "val1"})
        await mgr.checkpoint(iteration=2, extra={"key2": "val2"})
        assert mgr.state.extra == {"key1": "val1", "key2": "val2"}

    @pytest.mark.asyncio
    async def test_checkpoint_partial_update(self, mgr):
        """Only non-None fields are updated."""
        await mgr.checkpoint(iteration=1, phase="act", solver="hybrid")
        await mgr.checkpoint(iteration=2)
        assert mgr.state.phase == "act"
        assert mgr.state.solver == "hybrid"
        assert mgr.state.iteration == 2

    # ---- Reset ----

    @pytest.mark.asyncio
    async def test_reset(self, mgr, state_path):
        """reset() clears all state and removes file."""
        await mgr.checkpoint(iteration=10, phase="act", tokens_used=5000, cost_usd=0.1)
        assert Path(state_path).exists()

        state = await mgr.reset()
        assert state.iteration == 0
        assert state.phase == ""
        assert state.total_tokens_used == 0
        assert state.total_cost_usd == 0.0
        assert state.checkpoints == []
        assert not Path(state_path).exists()

    @pytest.mark.asyncio
    async def test_reset_nonexistent_file(self, mgr):
        """reset() doesn't crash if file doesn't exist."""
        state = await mgr.reset()
        assert state.iteration == 0

    # ---- Summary ----

    @pytest.mark.asyncio
    async def test_get_summary(self, mgr):
        """get_summary() returns structured dict."""
        await mgr.checkpoint(
            session_id="test-session",
            iteration=3,
            phase="act",
            solver="hybrid",
            tokens_used=5000,
            cost_usd=0.05,
            goal_snapshot={"progress": 0.6},
        )
        summary = await mgr.get_summary()
        assert summary["session_id"] == "test-session"
        assert summary["iteration"] == 3
        assert summary["phase"] == "act"
        assert summary["solver"] == "hybrid"
        assert summary["total_tokens_used"] == 5000
        assert summary["total_cost_usd"] == pytest.approx(0.05, abs=0.001)
        assert summary["goal_progress"] == 0.6

    # ---- Goal Snapshot ----

    @pytest.mark.asyncio
    async def test_goal_snapshot_roundtrip(self, state_path):
        """Goal snapshots persist through save/load."""
        snapshot = {
            "id": "g-42",
            "description": "Complete Phase 3",
            "progress": 0.75,
            "sub_goals": ["g-43", "g-44"],
        }
        mgr = StateManager(state_path=state_path)
        await mgr.checkpoint(iteration=1, phase="act", goal_snapshot=snapshot)

        mgr2 = StateManager(state_path=state_path)
        await mgr2.load()
        assert mgr2.state.goal_snapshot == snapshot


# =============================================================================
# Atomic Write Safety
# =============================================================================


class TestAtomicWrites:
    """Verify atomic write semantics (tmp→rename)."""

    @pytest.mark.asyncio
    async def test_no_partial_writes_on_error(self, tmp_path):
        """If write fails, original file is preserved."""
        state_path = str(tmp_path / "state.json")
        mgr = StateManager(state_path=state_path)
        await mgr.checkpoint(iteration=1, phase="act", tokens_used=100)

        original = Path(state_path).read_text()

        # Simulate write failure by making tmp file creation fail
        from unittest.mock import patch as _p

        def _fail_write(*args, **kwargs):
            raise OSError("Disk full")

        with _p.object(Path, "write_text", _fail_write):
            with pytest.raises(OSError):
                await mgr.save()

        # Original file should still be intact
        assert Path(state_path).read_text() == original
