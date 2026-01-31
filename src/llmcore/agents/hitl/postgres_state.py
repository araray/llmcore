# src/llmcore/agents/hitl/postgres_state.py
"""
PostgreSQL implementation of HITL state storage backend.

Provides persistent storage of HITL requests, responses, and scopes using PostgreSQL.
Suitable for production deployments with multi-user and high-concurrency requirements.

Architecture:
    - Uses psycopg with async connection pooling
    - JSONB columns for complex nested structures
    - GIN indexes for JSON querying
    - Proper transaction handling

Usage:
    from llmcore.agents.hitl.postgres_state import PostgresHITLStore

    store = PostgresHITLStore()
    await store.initialize({
        "db_url": "postgresql://user:pass@localhost:5432/llmcore",
        "min_pool_size": 2,
        "max_pool_size": 10,
    })

    # Save a request
    await store.save_request(request)

    # Get pending requests
    pending = await store.get_pending_requests()

References:
    - Master Plan: Section 20.3 (State Persistence)
    - postgres_failure_storage.py for implementation patterns
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    psycopg = None
    dict_row = None
    AsyncConnectionPool = None

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


class PostgresHITLStore(HITLStateStore):
    """
    PostgreSQL-based storage for HITL state.

    Uses psycopg with connection pooling for async database operations.
    Suitable for production deployments with high concurrency.

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
        activity_parameters JSONB,
        activity_reason TEXT,
        risk_overall_level TEXT NOT NULL,
        risk_factors JSONB,
        risk_requires_approval BOOLEAN DEFAULT TRUE,
        risk_reason TEXT,
        risk_dangerous_patterns JSONB,
        context_summary TEXT,
        created_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP,
        status TEXT NOT NULL DEFAULT 'pending',
        session_id TEXT,
        user_id TEXT,
        priority INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_request_status ON hitl_requests(status);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_session ON hitl_requests(session_id);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_expires ON hitl_requests(expires_at);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_created ON hitl_requests(created_at);
    CREATE INDEX IF NOT EXISTS idx_hitl_request_activity_gin ON hitl_requests USING gin(activity_parameters);

    CREATE TABLE IF NOT EXISTS hitl_responses (
        request_id TEXT PRIMARY KEY,
        approved BOOLEAN NOT NULL DEFAULT TRUE,
        status TEXT,
        modified_parameters JSONB,
        feedback TEXT,
        responded_at TIMESTAMP NOT NULL,
        response_time_ms INTEGER DEFAULT 0,
        responder_id TEXT,
        scope_grant JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_response_request ON hitl_responses(request_id);

    CREATE TABLE IF NOT EXISTS hitl_session_scopes (
        session_id TEXT PRIMARY KEY,
        approved_tools JSONB,
        approved_patterns JSONB,
        session_approval BOOLEAN DEFAULT FALSE,
        expires_at TIMESTAMP,
        created_at TIMESTAMP NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_session_scope_id ON hitl_session_scopes(session_id);

    CREATE TABLE IF NOT EXISTS hitl_persistent_scopes (
        user_id TEXT PRIMARY KEY,
        approved_tools JSONB,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hitl_persistent_user ON hitl_persistent_scopes(user_id);
    """

    def __init__(self):
        """Initialize PostgreSQL HITL store."""
        self._pool: Optional["AsyncConnectionPool"] = None
        self._table_prefix: str = ""

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PostgreSQL connection pool and create schema.

        Args:
            config: Configuration dictionary with 'db_url' key and optional
                   'min_pool_size', 'max_pool_size', 'table_prefix'

        Raises:
            ImportError: If psycopg is not installed
            ValueError: If db_url is not provided
            RuntimeError: If database initialization fails
        """
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg library is not installed. Install with: pip install 'psycopg[binary,pool]'"
            )

        db_url = config.get("db_url")
        if not db_url:
            raise ValueError("PostgreSQL HITL store 'db_url' not specified in config")

        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)
        self._table_prefix = config.get("table_prefix", "")

        try:
            logger.debug(
                f"Initializing PostgreSQL connection pool for HITL store "
                f"(min: {min_pool_size}, max: {max_pool_size})..."
            )

            # Create connection pool
            self._pool = AsyncConnectionPool(
                conninfo=db_url,
                min_size=min_pool_size,
                max_size=max_pool_size,
            )

            # Test connection
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;")
                    result = await cur.fetchone()
                    if not result:
                        raise RuntimeError("Database connection test failed")
                logger.debug("PostgreSQL connection test successful")

            # Create schema
            await self._ensure_tables_exist()

            logger.info("PostgreSQL HITL store initialized successfully")

        except Exception as e:
            if self._pool:
                await self._pool.close()
                self._pool = None
            raise RuntimeError(f"Failed to initialize PostgreSQL HITL store: {e}")

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""
        schema = self.SCHEMA
        if self._table_prefix:
            schema = schema.replace("hitl_requests", f"{self._table_prefix}hitl_requests")
            schema = schema.replace("hitl_responses", f"{self._table_prefix}hitl_responses")
            schema = schema.replace(
                "hitl_session_scopes", f"{self._table_prefix}hitl_session_scopes"
            )
            schema = schema.replace(
                "hitl_persistent_scopes", f"{self._table_prefix}hitl_persistent_scopes"
            )

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(schema)
            await conn.commit()

    def _get_table_name(self, base_name: str) -> str:
        """Get table name with prefix if configured."""
        return f"{self._table_prefix}{base_name}" if self._table_prefix else base_name

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

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
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_requests")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""INSERT INTO {table_name}
                           (request_id, activity_type, activity_parameters, activity_reason,
                            risk_overall_level, risk_factors, risk_requires_approval, risk_reason,
                            risk_dangerous_patterns, context_summary, created_at, expires_at,
                            status, session_id, user_id, priority)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (request_id) DO UPDATE SET
                            activity_type = EXCLUDED.activity_type,
                            activity_parameters = EXCLUDED.activity_parameters,
                            activity_reason = EXCLUDED.activity_reason,
                            risk_overall_level = EXCLUDED.risk_overall_level,
                            risk_factors = EXCLUDED.risk_factors,
                            risk_requires_approval = EXCLUDED.risk_requires_approval,
                            risk_reason = EXCLUDED.risk_reason,
                            risk_dangerous_patterns = EXCLUDED.risk_dangerous_patterns,
                            context_summary = EXCLUDED.context_summary,
                            expires_at = EXCLUDED.expires_at,
                            status = EXCLUDED.status,
                            priority = EXCLUDED.priority""",
                        (
                            request.request_id,
                            request.activity.activity_type,
                            json.dumps(request.activity.parameters),
                            request.activity.reason,
                            request.risk_assessment.overall_level,
                            json.dumps(
                                [
                                    self._risk_factor_to_dict(f)
                                    for f in request.risk_assessment.factors
                                ]
                            ),
                            request.risk_assessment.requires_approval,
                            request.risk_assessment.reason,
                            json.dumps(request.risk_assessment.dangerous_patterns),
                            request.context_summary,
                            request.created_at,
                            request.expires_at,
                            request.status.value,
                            request.session_id,
                            request.user_id,
                            request.priority,
                        ),
                    )
                await conn.commit()
            logger.debug(f"Saved HITL request {request.request_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to save HITL request: {e}")

    async def get_request(self, request_id: str) -> Optional[HITLRequest]:
        """
        Retrieve a request by ID.

        Args:
            request_id: The request ID

        Returns:
            HITLRequest or None if not found
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_requests")

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT * FROM {table_name} WHERE request_id = %s",
                    (request_id,),
                )
                row = await cur.fetchone()

        if not row:
            return None

        return self._row_to_request(row)

    async def update_request_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        response: Optional[HITLResponse] = None,
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
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_requests")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"UPDATE {table_name} SET status = %s WHERE request_id = %s",
                        (status.value, request_id),
                    )
                    updated = cur.rowcount > 0

                    if response and updated:
                        await self._save_response_internal(cur, response)

                await conn.commit()
            return updated

        except Exception as e:
            logger.error(f"Failed to update request status: {e}")
            return False

    async def get_pending_requests(
        self,
        session_id: Optional[str] = None,
    ) -> List[HITLRequest]:
        """
        Get all pending requests.

        Args:
            session_id: Optional session filter

        Returns:
            List of pending requests
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_requests")

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                if session_id:
                    await cur.execute(
                        f"SELECT * FROM {table_name} WHERE status = %s AND session_id = %s ORDER BY created_at",
                        (ApprovalStatus.PENDING.value, session_id),
                    )
                else:
                    await cur.execute(
                        f"SELECT * FROM {table_name} WHERE status = %s ORDER BY created_at",
                        (ApprovalStatus.PENDING.value,),
                    )
                rows = await cur.fetchall()

        return [self._row_to_request(row) for row in rows]

    async def delete_request(self, request_id: str) -> bool:
        """
        Delete a request and associated response.

        Args:
            request_id: Request to delete

        Returns:
            True if deleted
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        request_table = self._get_table_name("hitl_requests")
        response_table = self._get_table_name("hitl_responses")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Delete response first (foreign key)
                    await cur.execute(
                        f"DELETE FROM {response_table} WHERE request_id = %s",
                        (request_id,),
                    )
                    await cur.execute(
                        f"DELETE FROM {request_table} WHERE request_id = %s",
                        (request_id,),
                    )
                    deleted = cur.rowcount > 0
                await conn.commit()
            return deleted

        except Exception as e:
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
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await self._save_response_internal(cur, response)
                await conn.commit()
            logger.debug(f"Saved HITL response for {response.request_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to save HITL response: {e}")

    async def _save_response_internal(self, cur, response: HITLResponse) -> None:
        """Internal method to save response using existing cursor."""
        table_name = self._get_table_name("hitl_responses")

        scope_grant_json = None
        if response.scope_grant is not None:
            if hasattr(response.scope_grant, "to_dict"):
                scope_grant_json = json.dumps(response.scope_grant.to_dict())
            elif hasattr(response.scope_grant, "value"):
                scope_grant_json = json.dumps({"value": response.scope_grant.value})
            else:
                scope_grant_json = json.dumps(str(response.scope_grant))

        await cur.execute(
            f"""INSERT INTO {table_name}
               (request_id, approved, status, modified_parameters, feedback,
                responded_at, response_time_ms, responder_id, scope_grant)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (request_id) DO UPDATE SET
                approved = EXCLUDED.approved,
                status = EXCLUDED.status,
                modified_parameters = EXCLUDED.modified_parameters,
                feedback = EXCLUDED.feedback,
                responded_at = EXCLUDED.responded_at,
                response_time_ms = EXCLUDED.response_time_ms,
                responder_id = EXCLUDED.responder_id,
                scope_grant = EXCLUDED.scope_grant""",
            (
                response.request_id,
                response.approved,
                response.status.value if response.status else None,
                json.dumps(response.modified_parameters) if response.modified_parameters else None,
                response.feedback,
                response.responded_at,
                response.response_time_ms,
                response.responder_id,
                scope_grant_json,
            ),
        )

    async def get_response(self, request_id: str) -> Optional[HITLResponse]:
        """
        Get response for a request.

        Args:
            request_id: Request ID

        Returns:
            Response or None
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_responses")

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT * FROM {table_name} WHERE request_id = %s",
                    (request_id,),
                )
                row = await cur.fetchone()

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
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_session_scopes")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""INSERT INTO {table_name}
                           (session_id, approved_tools, approved_patterns, session_approval,
                            expires_at, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT (session_id) DO UPDATE SET
                            approved_tools = EXCLUDED.approved_tools,
                            approved_patterns = EXCLUDED.approved_patterns,
                            session_approval = EXCLUDED.session_approval,
                            expires_at = EXCLUDED.expires_at""",
                        (
                            scope.session_id,
                            json.dumps([self._tool_scope_to_dict(t) for t in scope.approved_tools]),
                            json.dumps(scope.approved_patterns),
                            scope.session_approval,
                            scope.expires_at,
                            scope.created_at,
                        ),
                    )
                await conn.commit()
            logger.debug(f"Saved session scope for {scope.session_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to save session scope: {e}")

    async def get_session_scope(self, session_id: str) -> Optional[SessionScope]:
        """
        Get session scope.

        Args:
            session_id: Session ID

        Returns:
            SessionScope or None
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_session_scopes")

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT * FROM {table_name} WHERE session_id = %s",
                    (session_id,),
                )
                row = await cur.fetchone()

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
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_persistent_scopes")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""INSERT INTO {table_name}
                           (user_id, approved_tools, created_at, updated_at)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (user_id) DO UPDATE SET
                            approved_tools = EXCLUDED.approved_tools,
                            updated_at = EXCLUDED.updated_at""",
                        (
                            scope.user_id,
                            json.dumps([self._tool_scope_to_dict(t) for t in scope.approved_tools]),
                            scope.created_at,
                            scope.updated_at,
                        ),
                    )
                await conn.commit()
            logger.debug(f"Saved persistent scope for {scope.user_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to save persistent scope: {e}")

    async def get_persistent_scope(self, user_id: str) -> Optional[PersistentScope]:
        """
        Get persistent scope.

        Args:
            user_id: User ID

        Returns:
            PersistentScope or None
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_persistent_scopes")

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT * FROM {table_name} WHERE user_id = %s",
                    (user_id,),
                )
                row = await cur.fetchone()

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
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_requests")

        try:
            now = datetime.now(timezone.utc)

            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""UPDATE {table_name}
                           SET status = %s
                           WHERE status = %s AND expires_at IS NOT NULL AND expires_at < %s""",
                        (ApprovalStatus.TIMEOUT.value, ApprovalStatus.PENDING.value, now),
                    )
                    count = cur.rowcount
                await conn.commit()

            if count > 0:
                logger.info(f"Marked {count} HITL requests as expired")

            return count

        except Exception as e:
            logger.error(f"Failed to cleanup expired requests: {e}")
            return 0

    async def count_pending(self) -> int:
        """
        Count pending requests.

        Returns:
            Number of pending requests
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("hitl_requests")

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE status = %s",
                    (ApprovalStatus.PENDING.value,),
                )
                row = await cur.fetchone()

        return row[0] if row else 0

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _risk_factor_to_dict(self, factor: RiskFactor) -> Dict[str, Any]:
        """Convert RiskFactor to dictionary."""
        return {
            "name": factor.name,
            "level": factor.level,
            "reason": factor.reason,
            "weight": factor.weight,
        }

    def _tool_scope_to_dict(self, scope: ToolScope) -> Dict[str, Any]:
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
        """Safely parse JSON field that may already be deserialized by psycopg."""
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value  # Already deserialized by psycopg
        if isinstance(value, str):
            try:
                return json.loads(value) if value else default
            except json.JSONDecodeError:
                return default
        return default

    def _row_to_request(self, row: Dict[str, Any]) -> HITLRequest:
        """Convert database row to HITLRequest."""
        factors_data = self._parse_json_field(row.get("risk_factors"), [])
        factors = [
            RiskFactor(
                name=f.get("name", ""),
                level=f.get("level", "low"),
                reason=f.get("reason", ""),
                weight=f.get("weight", 1.0),
            )
            for f in factors_data
        ]

        # Handle datetime parsing
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        expires_at = row.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return HITLRequest(
            request_id=row["request_id"],
            activity=ActivityInfo(
                activity_type=row["activity_type"],
                parameters=self._parse_json_field(row.get("activity_parameters"), {}),
                reason=row.get("activity_reason"),
            ),
            risk_assessment=RiskAssessment(
                overall_level=row["risk_overall_level"],
                factors=factors,
                requires_approval=bool(row.get("risk_requires_approval", True)),
                reason=row.get("risk_reason") or "",
                dangerous_patterns=self._parse_json_field(row.get("risk_dangerous_patterns"), []),
            ),
            context_summary=row.get("context_summary") or "",
            created_at=created_at,
            expires_at=expires_at,
            status=ApprovalStatus(row["status"]),
            session_id=row.get("session_id"),
            user_id=row.get("user_id"),
            priority=row.get("priority") or 0,
        )

    def _row_to_response(self, row: Dict[str, Any]) -> HITLResponse:
        """Convert database row to HITLResponse."""
        status = ApprovalStatus(row["status"]) if row.get("status") else None

        responded_at = row["responded_at"]
        if isinstance(responded_at, str):
            responded_at = datetime.fromisoformat(responded_at)

        return HITLResponse(
            request_id=row["request_id"],
            approved=bool(row["approved"]),
            status=status,
            modified_parameters=self._parse_json_field(row.get("modified_parameters")),
            feedback=row.get("feedback"),
            responded_at=responded_at,
            response_time_ms=row.get("response_time_ms") or 0,
            responder_id=row.get("responder_id") or "",
            scope_grant=self._parse_json_field(row.get("scope_grant")),
        )

    def _row_to_session_scope(self, row: Dict[str, Any]) -> SessionScope:
        """Convert database row to SessionScope."""
        tools_data = self._parse_json_field(row.get("approved_tools"), [])
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

        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        expires_at = row.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return SessionScope(
            session_id=row["session_id"],
            approved_tools=approved_tools,
            approved_patterns=self._parse_json_field(row.get("approved_patterns"), []),
            session_approval=bool(row.get("session_approval", False)),
            expires_at=expires_at,
            created_at=created_at,
        )

    def _row_to_persistent_scope(self, row: Dict[str, Any]) -> PersistentScope:
        """Convert database row to PersistentScope."""
        tools_data = self._parse_json_field(row.get("approved_tools"), [])
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

        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = row["updated_at"]
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return PersistentScope(
            user_id=row["user_id"],
            approved_tools=approved_tools,
            created_at=created_at,
            updated_at=updated_at,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["PostgresHITLStore"]
