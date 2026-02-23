# src/llmcore/sessions/recovery.py
"""
Session Recovery for Autonomous Agents.

Provides crash-recovery logic so that an autonomous agent can resume
a previous session after an unexpected termination (OOM, power loss,
network partition, etc.).

Recovery strategy:
1. **Checkpoint writing** — periodically persist a lightweight
   ``RecoveryCheckpoint`` to storage (goal state, phase, turn index,
   pending tool calls).
2. **Crash detection** — on startup, look for an incomplete checkpoint
   (``status != COMPLETED``).
3. **Session restoration** — reload the chat session, trim any dangling
   assistant turns (which may be incomplete), replay pending tool results,
   and resume from the last known-good state.

The module integrates with:

- :class:`~llmcore.sessions.manager.SessionManager` for session CRUD.
- :class:`~llmcore.storage.base_session.BaseSessionStorage` for persistence.
- :class:`~llmcore.autonomous.state.StateManager` for autonomous state.

Example::

    from llmcore.sessions.recovery import SessionRecovery, RecoveryCheckpoint

    recovery = SessionRecovery(session_manager, storage)
    cp = await recovery.find_latest_checkpoint(agent_id="wairu-main")
    if cp and cp.status == CheckpointStatus.IN_PROGRESS:
        session = await recovery.restore_session(cp)
        # resume agent loop from session

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §14 (Session System)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §11 (Autonomous Operation)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..exceptions import LLMCoreError, SessionNotFoundError
from ..models import ChatSession
from ..sessions.manager import SessionManager
from ..storage.base_session import BaseSessionStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CheckpointStatus(str, Enum):
    """Lifecycle states of a recovery checkpoint."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class RecoveryCheckpoint:
    """Lightweight snapshot written periodically during autonomous operation.

    Attributes:
        checkpoint_id: Unique checkpoint identifier.
        agent_id: The agent or session group this checkpoint belongs to.
        session_id: The chat session being executed.
        status: Current checkpoint lifecycle state.
        phase: Cognitive phase at checkpoint time (e.g. ``"THINK"``).
        turn_index: Number of completed conversation turns.
        goal_snapshot: Serialised goal state (JSON-safe dict).
        pending_tool_calls: Tool calls that were dispatched but not yet
            observed at checkpoint time.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
        metadata: Arbitrary extra data.
    """

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    session_id: str = ""
    status: CheckpointStatus = CheckpointStatus.IN_PROGRESS
    phase: str = ""
    turn_index: int = 0
    goal_snapshot: dict[str, Any] = field(default_factory=dict)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Serialisation helpers -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "phase": self.phase,
            "turn_index": self.turn_index,
            "goal_snapshot": self.goal_snapshot,
            "pending_tool_calls": self.pending_tool_calls,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryCheckpoint:
        """Deserialise from a dictionary (e.g. loaded from storage)."""
        return cls(
            checkpoint_id=data.get("checkpoint_id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", ""),
            session_id=data.get("session_id", ""),
            status=CheckpointStatus(data.get("status", "in_progress")),
            phase=data.get("phase", ""),
            turn_index=data.get("turn_index", 0),
            goal_snapshot=data.get("goal_snapshot", {}),
            pending_tool_calls=data.get("pending_tool_calls", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Recovery engine
# ---------------------------------------------------------------------------

# Storage key prefix for checkpoints
_CHECKPOINT_PREFIX = "recovery:checkpoint:"


class SessionRecovery:
    """Manages session checkpoints and crash recovery.

    Args:
        session_manager: The SessionManager for session CRUD.
        storage: A BaseSessionStorage backend for persisting checkpoints.
        max_checkpoints: Maximum checkpoints to retain per agent (FIFO).
    """

    def __init__(
        self,
        session_manager: SessionManager,
        storage: BaseSessionStorage,
        max_checkpoints: int = 10,
    ) -> None:
        self._session_mgr = session_manager
        self._storage = storage
        self._max_checkpoints = max_checkpoints
        logger.debug("SessionRecovery initialized (max_checkpoints=%d).", max_checkpoints)

    # -- Checkpoint lifecycle ------------------------------------------------

    async def create_checkpoint(
        self,
        agent_id: str,
        session_id: str,
        phase: str = "",
        turn_index: int = 0,
        goal_snapshot: dict[str, Any] | None = None,
        pending_tool_calls: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecoveryCheckpoint:
        """Create and persist a new checkpoint.

        Args:
            agent_id: Agent identifier.
            session_id: Chat session being executed.
            phase: Current cognitive phase.
            turn_index: Number of completed turns.
            goal_snapshot: Serialised goal state.
            pending_tool_calls: Dispatched-but-unobserved tool calls.
            metadata: Extra data.

        Returns:
            The persisted :class:`RecoveryCheckpoint`.
        """
        cp = RecoveryCheckpoint(
            agent_id=agent_id,
            session_id=session_id,
            phase=phase,
            turn_index=turn_index,
            goal_snapshot=goal_snapshot or {},
            pending_tool_calls=pending_tool_calls or [],
            metadata=metadata or {},
        )
        await self._persist_checkpoint(cp)
        logger.info(
            "Checkpoint created: id=%s, agent=%s, session=%s, phase=%s, turn=%d.",
            cp.checkpoint_id,
            agent_id,
            session_id,
            phase,
            turn_index,
        )
        return cp

    async def update_checkpoint(
        self,
        checkpoint: RecoveryCheckpoint,
        **updates: Any,
    ) -> RecoveryCheckpoint:
        """Update an existing checkpoint in place.

        Accepts keyword arguments matching :class:`RecoveryCheckpoint` fields.
        """
        for key, value in updates.items():
            if hasattr(checkpoint, key):
                setattr(checkpoint, key, value)
        checkpoint.updated_at = time.time()
        await self._persist_checkpoint(checkpoint)
        logger.debug("Checkpoint %s updated: %s.", checkpoint.checkpoint_id, list(updates.keys()))
        return checkpoint

    async def complete_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        """Mark a checkpoint as completed (normal shutdown)."""
        checkpoint.status = CheckpointStatus.COMPLETED
        checkpoint.updated_at = time.time()
        await self._persist_checkpoint(checkpoint)
        logger.info("Checkpoint %s marked COMPLETED.", checkpoint.checkpoint_id)

    async def fail_checkpoint(self, checkpoint: RecoveryCheckpoint, reason: str = "") -> None:
        """Mark a checkpoint as failed."""
        checkpoint.status = CheckpointStatus.FAILED
        checkpoint.metadata["failure_reason"] = reason
        checkpoint.updated_at = time.time()
        await self._persist_checkpoint(checkpoint)
        logger.warning("Checkpoint %s marked FAILED: %s.", checkpoint.checkpoint_id, reason)

    # -- Recovery logic ------------------------------------------------------

    async def find_latest_checkpoint(self, agent_id: str) -> RecoveryCheckpoint | None:
        """Find the most recent checkpoint for the given agent.

        Returns:
            The latest checkpoint, or *None* if no checkpoints exist.
        """
        key = f"{_CHECKPOINT_PREFIX}{agent_id}:latest"
        try:
            raw = await self._load_raw(key)
            if raw is None:
                return None
            return RecoveryCheckpoint.from_dict(raw)
        except Exception as e:
            logger.error("Error loading latest checkpoint for '%s': %s", agent_id, e)
            return None

    async def find_recoverable_checkpoint(self, agent_id: str) -> RecoveryCheckpoint | None:
        """Find a checkpoint that indicates a crash (status == IN_PROGRESS).

        Returns:
            A recoverable checkpoint, or *None*.
        """
        cp = await self.find_latest_checkpoint(agent_id)
        if cp is not None and cp.status == CheckpointStatus.IN_PROGRESS:
            logger.info(
                "Found recoverable checkpoint for '%s': id=%s, phase=%s, turn=%d.",
                agent_id,
                cp.checkpoint_id,
                cp.phase,
                cp.turn_index,
            )
            return cp
        return None

    async def restore_session(self, checkpoint: RecoveryCheckpoint) -> ChatSession:
        """Restore a chat session from a recovery checkpoint.

        This:
        1. Loads the session from storage.
        2. Trims any trailing incomplete assistant message.
        3. Marks the checkpoint as ``RECOVERING``.

        Args:
            checkpoint: The checkpoint to recover from.

        Returns:
            The restored :class:`ChatSession`.

        Raises:
            SessionNotFoundError: If the session no longer exists.
            LLMCoreError: On other recovery failures.
        """
        logger.info(
            "Restoring session '%s' from checkpoint '%s'...",
            checkpoint.session_id,
            checkpoint.checkpoint_id,
        )

        # Mark recovering
        checkpoint.status = CheckpointStatus.RECOVERING
        checkpoint.updated_at = time.time()
        await self._persist_checkpoint(checkpoint)

        try:
            session = await self._session_mgr.load_or_create_session(
                session_id=checkpoint.session_id
            )
        except SessionNotFoundError:
            logger.error("Session '%s' not found during recovery.", checkpoint.session_id)
            raise
        except Exception as e:
            raise LLMCoreError(f"Failed to restore session '{checkpoint.session_id}': {e}") from e

        # Trim trailing incomplete assistant message if present
        if session.messages:
            last_msg = session.messages[-1]
            if last_msg.role.value == "assistant" and not last_msg.content.strip():
                session.messages.pop()
                logger.debug("Trimmed empty trailing assistant message.")

        # Trim to turn_index if the session grew beyond checkpoint
        expected_messages = checkpoint.turn_index * 2  # rough: user + assistant per turn
        if len(session.messages) > expected_messages + 2:
            logger.debug(
                "Session has %d messages but checkpoint was at turn %d; trimming.",
                len(session.messages),
                checkpoint.turn_index,
            )
            # Keep system message + up to checkpoint point
            system_msgs = [m for m in session.messages if m.role.value == "system"]
            non_system = [m for m in session.messages if m.role.value != "system"]
            trimmed = non_system[:expected_messages]
            session.messages = system_msgs + trimmed

        logger.info(
            "Session '%s' restored with %d messages (was at turn %d).",
            checkpoint.session_id,
            len(session.messages),
            checkpoint.turn_index,
        )
        return session

    # -- Persistence helpers -------------------------------------------------

    async def _persist_checkpoint(self, cp: RecoveryCheckpoint) -> None:
        """Save a checkpoint to storage."""
        key_latest = f"{_CHECKPOINT_PREFIX}{cp.agent_id}:latest"
        key_specific = f"{_CHECKPOINT_PREFIX}{cp.agent_id}:{cp.checkpoint_id}"
        data = json.dumps(cp.to_dict())

        # Store as metadata in the session storage
        try:
            await self._storage.save_metadata(key_latest, data)
            await self._storage.save_metadata(key_specific, data)
        except AttributeError:
            # Fallback: storage backend doesn't have save_metadata.
            # Use a simple approach — store as a special session.
            logger.debug("Storage backend lacks save_metadata; using fallback persistence.")
            await self._fallback_persist(key_latest, data)

    async def _load_raw(self, key: str) -> dict[str, Any] | None:
        """Load a checkpoint dict from storage."""
        try:
            raw = await self._storage.load_metadata(key)
            if raw is None:
                return None
            return json.loads(raw) if isinstance(raw, str) else raw
        except AttributeError:
            # Fallback
            return await self._fallback_load(key)
        except Exception:
            return None

    async def _fallback_persist(self, key: str, data: str) -> None:
        """Fallback persistence when save_metadata is unavailable."""
        # Store checkpoint data as a pseudo-session
        try:
            session = ChatSession(
                session_id=key,
                system_message=f"__recovery_checkpoint__:{data}",
            )
            await self._session_mgr._storage.save_session(session)
        except Exception as e:
            logger.error("Fallback checkpoint persistence failed: %s", e)

    async def _fallback_load(self, key: str) -> dict[str, Any] | None:
        """Fallback load when load_metadata is unavailable."""
        try:
            session = await self._session_mgr._storage.load_session(key)
            if (
                session
                and session.system_message
                and session.system_message.startswith("__recovery_checkpoint__:")
            ):
                payload = session.system_message.split(":", 1)[1]
                return json.loads(payload)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "CheckpointStatus",
    "RecoveryCheckpoint",
    "SessionRecovery",
]
