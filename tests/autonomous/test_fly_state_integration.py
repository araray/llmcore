# tests/autonomous/test_fly_state_integration.py
# tests/autonomous/test_state_fly_integration.py
"""
Integration tests for StateManager checkpoint API.

Verifies the StateManager API surface that consumer code (fly.py, daemon)
relies on: checkpoint with keyword args, load/recovery, reset, accumulation.
These are pure llmcore tests — no wairu imports.

Covers:
- G10 prerequisite: StateManager.checkpoint() keyword interface
- G10 prerequisite: StateManager.load() recovery path
- G10 prerequisite: StateManager.reset() cleanup path
"""

import json
from pathlib import Path

import pytest

from llmcore.autonomous.state import StateManager

# =============================================================================
# StateManager Checkpoint Tests (verifying the consumer-facing API)
# =============================================================================


class TestStateManagerCheckpoint:
    """Tests for StateManager.checkpoint() — the core wiring mechanism."""

    @pytest.mark.asyncio
    async def test_checkpoint_creates_file(self, tmp_path):
        """Checkpoint creates the state file atomically."""
        state_path = tmp_path / "state.json"
        mgr = StateManager(state_path=str(state_path))

        await mgr.checkpoint(
            session_id="sess-001",
            iteration=1,
            phase="act",
            solver="hybrid",
        )

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["session_id"] == "sess-001"
        assert data["iteration"] == 1
        assert data["phase"] == "act"
        assert data["solver"] == "hybrid"

    @pytest.mark.asyncio
    async def test_checkpoint_accumulates_tokens_and_cost(self, tmp_path):
        """Tokens and cost are additive across checkpoints."""
        mgr = StateManager(state_path=str(tmp_path / "state.json"))

        await mgr.checkpoint(iteration=1, tokens_used=100, cost_usd=0.01)
        await mgr.checkpoint(iteration=2, tokens_used=200, cost_usd=0.02)

        state = mgr.state
        assert state.total_tokens_used == 300
        assert state.total_cost_usd == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_checkpoint_increments_error_count(self, tmp_path):
        """Error flag increments the error counter."""
        mgr = StateManager(state_path=str(tmp_path / "state.json"))

        await mgr.checkpoint(iteration=1, error=True)
        await mgr.checkpoint(iteration=2, error=True)
        await mgr.checkpoint(iteration=3, error=False)

        assert mgr.state.error_count == 2

    @pytest.mark.asyncio
    async def test_checkpoint_stores_goal_snapshot(self, tmp_path):
        """Goal snapshot is stored and retrievable."""
        mgr = StateManager(state_path=str(tmp_path / "state.json"))

        await mgr.checkpoint(
            iteration=1,
            goal_snapshot={"description": "Fix all tests", "progress": 0.5},
        )

        assert mgr.state.goal_snapshot["description"] == "Fix all tests"
        assert mgr.state.goal_snapshot["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_checkpoint_timestamps(self, tmp_path):
        """started_at is set once, updated_at changes each checkpoint."""
        mgr = StateManager(state_path=str(tmp_path / "state.json"))

        await mgr.checkpoint(iteration=1)
        first_started = mgr.state.started_at
        first_updated = mgr.state.updated_at

        await mgr.checkpoint(iteration=2)
        assert mgr.state.started_at == first_started  # Unchanged
        assert mgr.state.updated_at != first_updated  # Updated


# =============================================================================
# StateManager Load/Recovery Tests
# =============================================================================


class TestStateManagerLoadRecover:
    """Tests for state load/recovery — the resume path."""

    @pytest.mark.asyncio
    async def test_load_returns_empty_when_no_file(self, tmp_path):
        """Loading from non-existent file returns empty state."""
        mgr = StateManager(state_path=str(tmp_path / "missing.json"))
        state = await mgr.load()
        assert state.is_empty

    @pytest.mark.asyncio
    async def test_load_recovers_checkpointed_state(self, tmp_path):
        """Loading reads back previously checkpointed state."""
        state_path = tmp_path / "state.json"
        mgr1 = StateManager(state_path=str(state_path))
        await mgr1.checkpoint(
            session_id="sess-recover",
            iteration=5,
            phase="reflect",
            solver="proactive",
        )

        # New manager loads from disk
        mgr2 = StateManager(state_path=str(state_path))
        state = await mgr2.load()
        assert state.session_id == "sess-recover"
        assert state.iteration == 5
        assert state.phase == "reflect"

    @pytest.mark.asyncio
    async def test_load_handles_corrupt_file(self, tmp_path):
        """Corrupt state file returns empty state instead of crashing."""
        state_path = tmp_path / "state.json"
        state_path.write_text("not valid json {{{")

        mgr = StateManager(state_path=str(state_path))
        state = await mgr.load()
        assert state.is_empty


# =============================================================================
# StateManager Reset Tests
# =============================================================================


class TestStateManagerReset:
    """Tests for state reset — the clean completion path."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, tmp_path):
        """Reset returns empty state and removes file."""
        state_path = tmp_path / "state.json"
        mgr = StateManager(state_path=str(state_path))
        await mgr.checkpoint(iteration=10, session_id="to-clear")

        assert state_path.exists()
        state = await mgr.reset()
        assert state.is_empty
        assert not state_path.exists()


__all__ = []
