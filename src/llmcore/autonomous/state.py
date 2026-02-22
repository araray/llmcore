# src/llmcore/autonomous/state.py
"""
Autonomous State Persistence for llmcore.

Provides a unified state object and manager that persists the running
state of an autonomous session — checkpoints, iteration count, solver
state, active goal snapshot, etc.  This allows an agent to:

- Resume exactly where it left off after a restart or crash.
- Inspect its own history for meta-reasoning.
- Expose state via the introspection API.

The state is deliberately decoupled from GoalManager (which owns goal
persistence) and SessionManager (which owns conversation turns).
AutonomousState captures the *orchestration* metadata that neither of
those systems track.

Example::

    from llmcore.autonomous.state import StateManager, AutonomousState

    # Create manager (file-backed)
    mgr = StateManager(state_path="~/.local/share/llmcore/autonomous_state.json")

    # Checkpoint after each iteration
    await mgr.checkpoint(
        iteration=42,
        phase="act",
        solver="hybrid",
        goal_snapshot={"id": "g-123", "progress": 0.6},
    )

    # Recover after restart
    state = await mgr.load()
    print(state.iteration)  # 42

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §7.3 (autonomous/state.py)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Sentinel for UTC-aware "never".
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class AutonomousState:
    """
    Snapshot of the autonomous orchestration state.

    This is serialized to JSON on every checkpoint and reloaded on
    recovery.  Keep fields JSON-serializable (strings, numbers, lists,
    dicts, datetimes as ISO strings).

    Attributes:
        session_id: Identifier of the owning AutonomousSession.
        iteration: Current iteration number (0-based).
        phase: Last cognitive-cycle phase completed.
        solver: Active solver name (``"reactive"``, ``"proactive"``,
            ``"hybrid"``).
        goal_snapshot: Lightweight copy of active goal state (id,
            description, progress, status).
        extra: Arbitrary solver- or plugin-specific metadata.
        started_at: When the autonomous run started.
        updated_at: When this state was last checkpointed.
        error_count: Cumulative error count across iterations.
        total_tokens_used: Cumulative token usage.
        total_cost_usd: Cumulative cost estimate (USD).
        checkpoints: List of checkpoint timestamps (ISO strings).
    """

    session_id: str = ""
    iteration: int = 0
    phase: str = ""
    solver: str = ""
    goal_snapshot: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    started_at: str = ""
    updated_at: str = ""
    error_count: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0

    checkpoints: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "phase": self.phase,
            "solver": self.solver,
            "goal_snapshot": self.goal_snapshot,
            "extra": self.extra,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error_count": self.error_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": self.total_cost_usd,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AutonomousState:
        """Deserialize from a plain dict."""
        known_fields = {
            "session_id",
            "iteration",
            "phase",
            "solver",
            "goal_snapshot",
            "extra",
            "started_at",
            "updated_at",
            "error_count",
            "total_tokens_used",
            "total_cost_usd",
            "checkpoints",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @property
    def is_empty(self) -> bool:
        """True if this state has never been checkpointed."""
        return self.iteration == 0 and not self.session_id


# =============================================================================
# State Manager
# =============================================================================


class StateManager:
    """
    Manages loading and saving ``AutonomousState`` to a JSON file.

    Thread/coroutine safety: writing is atomic (write-then-rename) so
    concurrent reads will see either the previous or the new state,
    never a partially-written file.

    Args:
        state_path: Path to the JSON state file.  Tilde and environment
            variables are expanded.

    Example::

        mgr = StateManager("~/.local/share/llmcore/autonomous_state.json")
        await mgr.checkpoint(iteration=1, phase="act", solver="hybrid")
        state = await mgr.load()
        assert state.iteration == 1
    """

    def __init__(self, state_path: str | Path = "") -> None:
        if not state_path:
            state_path = "~/.local/share/llmcore/autonomous_state.json"
        self._path = Path(os.path.expandvars(os.path.expanduser(str(state_path))))
        self._state = AutonomousState()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Resolved path to the state file."""
        return self._path

    @property
    def state(self) -> AutonomousState:
        """Current in-memory state (may be stale; call ``load`` to refresh)."""
        return self._state

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def load(self) -> AutonomousState:
        """
        Load state from disk.

        Returns:
            The loaded ``AutonomousState``, or a fresh empty one if the
            file doesn't exist or is corrupt.
        """
        if not self._path.exists():
            logger.debug("State file does not exist: %s", self._path)
            self._state = AutonomousState()
            return self._state

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._state = AutonomousState.from_dict(data)
            logger.info(
                "Loaded autonomous state: session=%s iter=%d",
                self._state.session_id,
                self._state.iteration,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt state file %s: %s — starting fresh", self._path, exc)
            self._state = AutonomousState()

        return self._state

    async def save(self) -> None:
        """
        Persist current in-memory state to disk.

        Uses atomic write (tmp → rename) to avoid partial writes.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            data = json.dumps(self._state.to_dict(), indent=2, default=str)
            tmp_path.write_text(data, encoding="utf-8")
            tmp_path.replace(self._path)
            logger.debug("Saved autonomous state to %s", self._path)
        except OSError as exc:
            logger.error("Failed to save state: %s", exc)
            # Clean up partial tmp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    async def checkpoint(
        self,
        *,
        iteration: int | None = None,
        phase: str | None = None,
        solver: str | None = None,
        goal_snapshot: Dict[str, Any] | None = None,
        extra: Dict[str, Any] | None = None,
        session_id: str | None = None,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        error: bool = False,
    ) -> AutonomousState:
        """
        Update state fields and persist to disk.

        Only non-``None`` fields are updated.  ``tokens_used`` and
        ``cost_usd`` are *added* to the running totals.

        Args:
            iteration: Current iteration number.
            phase: Last completed cognitive phase.
            solver: Active solver name.
            goal_snapshot: Lightweight goal state dict.
            extra: Arbitrary metadata to merge.
            session_id: Session identifier (usually set once).
            tokens_used: Tokens consumed in this iteration (additive).
            cost_usd: Cost incurred in this iteration (additive).
            error: Whether this iteration resulted in an error.

        Returns:
            Updated ``AutonomousState``.
        """
        now = datetime.now(timezone.utc).isoformat()

        if session_id is not None:
            self._state.session_id = session_id
        if iteration is not None:
            self._state.iteration = iteration
        if phase is not None:
            self._state.phase = phase
        if solver is not None:
            self._state.solver = solver
        if goal_snapshot is not None:
            self._state.goal_snapshot = goal_snapshot
        if extra is not None:
            self._state.extra = {**self._state.extra, **extra}

        # Set started_at on first checkpoint
        if not self._state.started_at:
            self._state.started_at = now

        self._state.updated_at = now
        self._state.total_tokens_used += tokens_used
        self._state.total_cost_usd += cost_usd

        if error:
            self._state.error_count += 1

        # Keep last 100 checkpoint timestamps
        self._state.checkpoints.append(now)
        if len(self._state.checkpoints) > 100:
            self._state.checkpoints = self._state.checkpoints[-100:]

        await self.save()
        return self._state

    async def reset(self) -> AutonomousState:
        """
        Reset state to empty and remove the state file.

        Returns:
            Fresh empty ``AutonomousState``.
        """
        self._state = AutonomousState()
        if self._path.exists():
            try:
                self._path.unlink()
                logger.info("State file removed: %s", self._path)
            except OSError as exc:
                logger.warning("Failed to remove state file: %s", exc)
        return self._state

    async def get_summary(self) -> Dict[str, Any]:
        """
        Return a concise summary for introspection/status display.

        Returns:
            Dict with key state fields.
        """
        return {
            "session_id": self._state.session_id,
            "iteration": self._state.iteration,
            "phase": self._state.phase,
            "solver": self._state.solver,
            "error_count": self._state.error_count,
            "total_tokens_used": self._state.total_tokens_used,
            "total_cost_usd": round(self._state.total_cost_usd, 4),
            "started_at": self._state.started_at,
            "updated_at": self._state.updated_at,
            "goal_progress": self._state.goal_snapshot.get("progress", 0.0),
        }
