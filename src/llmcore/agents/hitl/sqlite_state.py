# src/llmcore/agents/hitl/sqlite_state.py
"""
SQLite implementation of HITL state storage backend.

Provides persistent storage of HITL requests, responses, and scopes using SQLite.
Suitable for development and single-user deployments.

Architecture:
    - Uses aiosqlite for async database operations
    - WAL mode for better concurrency
    - JSON columns for complex nested structures
    - Indexes for common query patterns

Usage:
    from llmcore.agents.hitl.sqlite_state import SqliteHITLStore

    store = SqliteHITLStore()
    await store.initialize({"path": "~/.local/share/llmcore/hitl.db"})

    # Save a request
    await store.save_request(request)

    # Get pending requests
    pending = await store.get_pending_requests()

References:
    - Master Plan: Section 20.3 (State Persistence)
    - sqlite_failure_storage.py for implementation patterns
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import datetime, timezone, UTC
from typing import Any, Dict, List, Optional

try:
    import aiosqlite

    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from .models import (
    ActivityInfo,
    ApprovalStatus,
    HITLRequest,
    HITLResponse,
    PersistentScope,
    RiskAssessment,
    RiskFactor,
    SessionScope,
    ToolScope,
)
from .state import HITLStateStore

logger = logging.getLogger(__name__)


class SqliteHITLStore(HITLStateStore):
    """
    SQLite-based storage for HITL state.

    Uses aiosqlite for async database operations. Stores requests, responses,
    and scopes in a single SQLite database file with proper indexes.

    Schema:
        hitl_requests: Pending and completed approval requests
        hitl_responses: Responses to approval requests
        hitl_session_scopes: Session-bound approval scopes
        hitl_persistent_scopes: Cross-session approval scopes
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS hitl_requests (
        request_id TEXT PRIMARY KEY,
        activity_type TEXT NOT NULL,
        activity_parameters TEXT,
        activity_reason TEXT,
        risk_overall_level TEXT NOT NULL,
        risk_factors TEXT,
        risk_requires_approval INTEGER DEFAULT 1,
        risk_reason TEXT,
        risk_dangerous_patterns TEXT,
        context_summary TEXT,
        created_at TEXT NOT NULL,
        expires_at TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        session_id TEXT,
        user_id TEXT,
        priority INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_request_status ON hitl_requests(status);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_session ON hitl_requests(session_id);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_expires ON hitl_requests(expires_at);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_created ON hitl_requests(created_at);

    CREATE TABLE IF NOT EXISTS hitl_responses (
        request_id TEXT PRIMARY KEY,
        approved INTEGER NOT NULL DEFAULT 1,
        status TEXT,
        modified_parameters TEXT,
        feedback TEXT,
        responded_at TEXT NOT NULL,
        response_time_ms INTEGER DEFAULT 0,
        responder_id TEXT,
        scope_grant TEXT,
        FOREIGN KEY (request_id) REFERENCES hitl_requests(request_id)
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_response_request ON hitl_responses(request_id);

    CREATE TABLE IF NOT EXISTS hitl_session_scopes (
        session_id TEXT PRIMARY KEY,
        approved_tools TEXT,
        approved_patterns TEXT,
        session_approval INTEGER DEFAULT 0,
        expires_at TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_session_scope_id ON hitl_session_scopes(session_id);

    CREATE TABLE IF NOT EXISTS hitl_persistent_scopes (
        user_id TEXT PRIMARY KEY,
        approved_tools TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_persistent_user ON hitl_persistent_scopes(user_id);
    """

    def __init__(self):
        """Initialize SQLite HITL store."""
        self._db_path: pathlib.Path | None = None
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the SQLite database and create schema.

        Args:
            config: Configuration dictionary with 'path' key

        Raises:
            ImportError: If aiosqlite is not installed
            ValueError: If path is not provided
            RuntimeError: If database initialization fails
        """
        if not AIOSQLITE_AVAILABLE:
            raise ImportError(
                "aiosqlite library is not installed. Install with: pip install aiosqlite"
            )

        db_path_str = config.get("path")
        if not db_path_str:
            raise ValueError("SQLite HITL store 'path' not specified in config")

        self._db_path = pathlib.Path(os.path.expanduser(db_path_str))

        try:
            # Create parent directory if needed
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row

            # Enable WAL mode for better concurrency
            await self._conn.execute("PRAGMA journal_mode=WAL;")

            # Create schema
            await self._conn.executescript(self.SCHEMA)
            await self._conn.commit()

            logger.info(f"SQLite HITL store initialized at: {self._db_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite HITL store: {e}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # =========================================================================
    # REQUEST METHODS
    # =========================================================================

    async def save_request(self, request: HITLRequest) -> None:
        """
        Save a pending approval request.

        Args:
            request: The HITL request to save

        Raises:
            RuntimeError: If not initialized or save fails
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            await self._conn.execute(
                """INSERT OR REPLACE INTO hitl_requests
                   (request_id, activity_type, activity_parameters, activity_reason,
                    risk_overall_level, risk_factors, risk_requires_approval, risk_reason,
                    risk_dangerous_patterns, context_summary, created_at, expires_at,
                    status, session_id, user_id, priority)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.request_id,
                    request.activity.activity_type,
                    json.dumps(request.activity.parameters),
                    request.activity.reason,
                    request.risk_assessment.overall_level,
                    json.dumps(
                        [self._risk_factor_to_dict(f) for f in request.risk_assessment.factors]
                    ),
                    1 if request.risk_assessment.requires_approval else 0,
                    request.risk_assessment.reason,
                    json.dumps(request.risk_assessment.dangerous_patterns),
                    request.context_summary,
                    request.created_at.isoformat(),
                    request.expires_at.isoformat() if request.expires_at else None,
                    request.status.value,
                    request.session_id,
                    request.user_id,
                    request.priority,
                ),
            )
            await self._conn.commit()
            logger.debug(f"Saved HITL request {request.request_id}")

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save HITL request: {e}")

    async def get_request(self, request_id: str) -> HITLRequest | None:
        """
        Retrieve a request by ID.

        Args:
            request_id: The request ID

        Returns:
            HITLRequest or None if not found
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM hitl_requests WHERE request_id = ?",
            (request_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_request(row)

    async def update_request_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        response: HITLResponse | None = None,
    ) -> bool:
        """
        Update request status.

        Args:
            request_id: Request to update
            status: New status
            response: Optional response to save

        Returns:
            True if request was updated
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            cursor = await self._conn.execute(
                "UPDATE hitl_requests SET status = ? WHERE request_id = ?",
                (status.value, request_id),
            )
            updated = cursor.rowcount > 0

            if response and updated:
                await self.save_response(response)

            await self._conn.commit()
            return updated

        except Exception as e:
            await self._conn.rollback()
            logger.error(f"Failed to update request status: {e}")
            return False

    async def get_pending_requests(
        self,
        session_id: str | None = None,
    ) -> list[HITLRequest]:
        """
        Get all pending requests.

        Args:
            session_id: Optional session filter

        Returns:
            List of pending requests
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        if session_id:
            cursor = await self._conn.execute(
                "SELECT * FROM hitl_requests WHERE status = ? AND session_id = ? ORDER BY created_at",
                (ApprovalStatus.PENDING.value, session_id),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM hitl_requests WHERE status = ? ORDER BY created_at",
                (ApprovalStatus.PENDING.value,),
            )

        rows = await cursor.fetchall()
        return [self._row_to_request(row) for row in rows]

    async def delete_request(self, request_id: str) -> bool:
        """
        Delete a request and associated response.

        Args:
            request_id: Request to delete

        Returns:
            True if deleted
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            # Delete response first (foreign key)
            await self._conn.execute(
                "DELETE FROM hitl_responses WHERE request_id = ?",
                (request_id,),
            )
            cursor = await self._conn.execute(
                "DELETE FROM hitl_requests WHERE request_id = ?",
                (request_id,),
            )
            await self._conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            await self._conn.rollback()
            logger.error(f"Failed to delete request: {e}")
            return False

    # =========================================================================
    # RESPONSE METHODS
    # =========================================================================

    async def save_response(self, response: HITLResponse) -> None:
        """
        Save an approval response.

        Args:
            response: The response to save
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            scope_grant_json = None
            if response.scope_grant is not None:
                if hasattr(response.scope_grant, "to_dict"):
                    scope_grant_json = json.dumps(response.scope_grant.to_dict())
                elif hasattr(response.scope_grant, "value"):
                    scope_grant_json = json.dumps({"value": response.scope_grant.value})
                else:
                    scope_grant_json = json.dumps(str(response.scope_grant))

            await self._conn.execute(
                """INSERT OR REPLACE INTO hitl_responses
                   (request_id, approved, status, modified_parameters, feedback,
                    responded_at, response_time_ms, responder_id, scope_grant)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    response.request_id,
                    1 if response.approved else 0,
                    response.status.value if response.status else None,
                    json.dumps(response.modified_parameters)
                    if response.modified_parameters
                    else None,
                    response.feedback,
                    response.responded_at.isoformat(),
                    response.response_time_ms,
                    response.responder_id,
                    scope_grant_json,
                ),
            )
            await self._conn.commit()
            logger.debug(f"Saved HITL response for {response.request_id}")

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save HITL response: {e}")

    async def get_response(self, request_id: str) -> HITLResponse | None:
        """
        Get response for a request.

        Args:
            request_id: Request ID

        Returns:
            Response or None
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM hitl_responses WHERE request_id = ?",
            (request_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_response(row)

    # =========================================================================
    # SESSION SCOPE METHODS
    # =========================================================================

    async def save_session_scope(self, scope: SessionScope) -> None:
        """
        Save session scope.

        Args:
            scope: Session scope to save
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            await self._conn.execute(
                """INSERT OR REPLACE INTO hitl_session_scopes
                   (session_id, approved_tools, approved_patterns, session_approval,
                    expires_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    scope.session_id,
                    json.dumps([self._tool_scope_to_dict(t) for t in scope.approved_tools]),
                    json.dumps(scope.approved_patterns),
                    1 if scope.session_approval else 0,
                    scope.expires_at.isoformat() if scope.expires_at else None,
                    scope.created_at.isoformat(),
                ),
            )
            await self._conn.commit()
            logger.debug(f"Saved session scope for {scope.session_id}")

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save session scope: {e}")

    async def get_session_scope(self, session_id: str) -> SessionScope | None:
        """
        Get session scope.

        Args:
            session_id: Session ID

        Returns:
            SessionScope or None
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM hitl_session_scopes WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_session_scope(row)

    # =========================================================================
    # PERSISTENT SCOPE METHODS
    # =========================================================================

    async def save_persistent_scope(self, scope: PersistentScope) -> None:
        """
        Save persistent scope.

        Args:
            scope: Persistent scope to save
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            await self._conn.execute(
                """INSERT OR REPLACE INTO hitl_persistent_scopes
                   (user_id, approved_tools, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    scope.user_id,
                    json.dumps([self._tool_scope_to_dict(t) for t in scope.approved_tools]),
                    scope.created_at.isoformat(),
                    scope.updated_at.isoformat(),
                ),
            )
            await self._conn.commit()
            logger.debug(f"Saved persistent scope for {scope.user_id}")

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save persistent scope: {e}")

    async def get_persistent_scope(self, user_id: str) -> PersistentScope | None:
        """
        Get persistent scope.

        Args:
            user_id: User ID

        Returns:
            PersistentScope or None
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM hitl_persistent_scopes WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_persistent_scope(row)

    # =========================================================================
    # CLEANUP METHODS
    # =========================================================================

    async def cleanup_expired(self) -> int:
        """
        Remove expired requests.

        Returns:
            Number of requests cleaned up
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            now = datetime.now(UTC).isoformat()

            # Mark expired requests as TIMEOUT
            cursor = await self._conn.execute(
                """UPDATE hitl_requests
                   SET status = ?
                   WHERE status = ? AND expires_at IS NOT NULL AND expires_at < ?""",
                (ApprovalStatus.TIMEOUT.value, ApprovalStatus.PENDING.value, now),
            )
            count = cursor.rowcount
            await self._conn.commit()

            if count > 0:
                logger.info(f"Marked {count} HITL requests as expired")

            return count

        except Exception as e:
            await self._conn.rollback()
            logger.error(f"Failed to cleanup expired requests: {e}")
            return 0

    async def count_pending(self) -> int:
        """
        Count pending requests.

        Returns:
            Number of pending requests
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM hitl_requests WHERE status = ?",
            (ApprovalStatus.PENDING.value,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _risk_factor_to_dict(self, factor: RiskFactor) -> dict[str, Any]:
        """Convert RiskFactor to dictionary."""
        return {
            "name": factor.name,
            "level": factor.level,
            "reason": factor.reason,
            "weight": factor.weight,
        }

    def _tool_scope_to_dict(self, scope: ToolScope) -> dict[str, Any]:
        """Convert ToolScope to dictionary."""
        return {
            "tool_name": scope.tool_name,
            "approved": scope.approved,
            "conditions": scope.conditions,
            "granted_at": scope.granted_at.isoformat(),
            "granted_by": scope.granted_by,
            "max_risk_level": scope.max_risk_level,
        }

    def _parse_json_field(self, value: Any, default: Any = None) -> Any:
        """Safely parse JSON field that may already be deserialized."""
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value) if value else default
            except json.JSONDecodeError:
                return default
        return default

    def _row_to_request(self, row: aiosqlite.Row) -> HITLRequest:
        """Convert database row to HITLRequest."""
        factors_data = self._parse_json_field(row["risk_factors"], [])
        factors = [
            RiskFactor(
                name=f.get("name", ""),
                level=f.get("level", "low"),
                reason=f.get("reason", ""),
                weight=f.get("weight", 1.0),
            )
            for f in factors_data
        ]

        return HITLRequest(
            request_id=row["request_id"],
            activity=ActivityInfo(
                activity_type=row["activity_type"],
                parameters=self._parse_json_field(row["activity_parameters"], {}),
                reason=row["activity_reason"],
            ),
            risk_assessment=RiskAssessment(
                overall_level=row["risk_overall_level"],
                factors=factors,
                requires_approval=bool(row["risk_requires_approval"]),
                reason=row["risk_reason"] or "",
                dangerous_patterns=self._parse_json_field(row["risk_dangerous_patterns"], []),
            ),
            context_summary=row["context_summary"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            status=ApprovalStatus(row["status"]),
            session_id=row["session_id"],
            user_id=row["user_id"],
            priority=row["priority"] or 0,
        )

    def _row_to_response(self, row: aiosqlite.Row) -> HITLResponse:
        """Convert database row to HITLResponse."""
        status = ApprovalStatus(row["status"]) if row["status"] else None

        return HITLResponse(
            request_id=row["request_id"],
            approved=bool(row["approved"]),
            status=status,
            modified_parameters=self._parse_json_field(row["modified_parameters"]),
            feedback=row["feedback"],
            responded_at=datetime.fromisoformat(row["responded_at"]),
            response_time_ms=row["response_time_ms"] or 0,
            responder_id=row["responder_id"] or "",
            scope_grant=self._parse_json_field(row["scope_grant"]),
        )

    def _row_to_session_scope(self, row: aiosqlite.Row) -> SessionScope:
        """Convert database row to SessionScope."""
        tools_data = self._parse_json_field(row["approved_tools"], [])
        approved_tools = [
            ToolScope(
                tool_name=t["tool_name"],
                approved=t.get("approved", True),
                conditions=t.get("conditions"),
                granted_at=datetime.fromisoformat(t["granted_at"])
                if "granted_at" in t
                else datetime.now(),
                granted_by=t.get("granted_by", ""),
                max_risk_level=t.get("max_risk_level", "medium"),
            )
            for t in tools_data
        ]

        return SessionScope(
            session_id=row["session_id"],
            approved_tools=approved_tools,
            approved_patterns=self._parse_json_field(row["approved_patterns"], []),
            session_approval=bool(row["session_approval"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_persistent_scope(self, row: aiosqlite.Row) -> PersistentScope:
        """Convert database row to PersistentScope."""
        tools_data = self._parse_json_field(row["approved_tools"], [])
        approved_tools = [
            ToolScope(
                tool_name=t["tool_name"],
                approved=t.get("approved", True),
                conditions=t.get("conditions"),
                granted_at=datetime.fromisoformat(t["granted_at"])
                if "granted_at" in t
                else datetime.now(),
                granted_by=t.get("granted_by", ""),
                max_risk_level=t.get("max_risk_level", "medium"),
            )
            for t in tools_data
        ]

        return PersistentScope(
            user_id=row["user_id"],
            approved_tools=approved_tools,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["SqliteHITLStore"]
