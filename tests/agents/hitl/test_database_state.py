# tests/agents/hitl/test_database_state.py
"""
Tests for SQLite and PostgreSQL HITL State Storage backends.

Tests:
- SqliteHITLStore operations
- PostgresHITLStore operations
- Interface compliance
- Concurrent access
- Edge cases

Following patterns from:
- tests/agents/darwin/test_failure_learning.py
- tests/agents/hitl/test_state.py
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llmcore.agents.hitl import (
    ActivityInfo,
    ApprovalStatus,
    HITLRequest,
    HITLResponse,
    InMemoryHITLStore,
    PersistentScope,
    RiskAssessment,
    RiskFactor,
    SessionScope,
    ToolScope,
)

# Try to import database-backed stores
try:
    from llmcore.agents.hitl import SQLITE_HITL_AVAILABLE, SqliteHITLStore
except ImportError:
    SqliteHITLStore = None
    SQLITE_HITL_AVAILABLE = False

try:
    from llmcore.agents.hitl import POSTGRES_HITL_AVAILABLE, PostgresHITLStore
except ImportError:
    PostgresHITLStore = None
    POSTGRES_HITL_AVAILABLE = False

# Check for PostgreSQL test database URL
POSTGRES_TEST_URL = os.environ.get("TEST_POSTGRES_URL")
POSTGRES_TESTS_ENABLED = POSTGRES_TEST_URL is not None and POSTGRES_HITL_AVAILABLE


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for SQLite tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_hitl.db"


@pytest.fixture
def sample_request():
    """Create a sample HITL request."""
    activity = ActivityInfo(
        activity_type="bash_exec",
        parameters={"command": "ls -la", "cwd": "/workspace"},
        reason="List directory contents",
    )
    risk = RiskAssessment(
        overall_level="medium",
        factors=[
            RiskFactor(name="command_execution", level="medium", reason="Executing shell command"),
            RiskFactor(name="file_system_access", level="low", reason="Reading directory"),
        ],
        requires_approval=True,
        reason="Shell command execution requires human oversight",
        dangerous_patterns=["rm", "sudo"],
    )
    request = HITLRequest(
        activity=activity,
        risk_assessment=risk,
        session_id="test-session-123",
        user_id="test-user",
        context_summary="User wants to list files in workspace",
        priority=5,
    )
    request.set_expiration(300)  # 5 minutes
    return request


@pytest.fixture
def sample_response(sample_request):
    """Create a sample HITL response."""
    return HITLResponse(
        request_id=sample_request.request_id,
        approved=True,
        status=ApprovalStatus.APPROVED,
        feedback="Looks safe, approved",
        responder_id="test-user",
        response_time_ms=1500,
    )


@pytest.fixture
def sample_session_scope():
    """Create a sample session scope."""
    return SessionScope(
        session_id="test-session-123",
        approved_tools=[
            ToolScope(
                tool_name="bash_exec",
                approved=True,
                max_risk_level="medium",
                granted_by="test-user",
                conditions={"allowed_commands": ["ls", "pwd", "cat"]},
            ),
            ToolScope(
                tool_name="file_read",
                approved=True,
                max_risk_level="low",
                granted_by="test-user",
            ),
        ],
        approved_patterns=["read any file in /workspace"],
        session_approval=False,
    )


@pytest.fixture
def sample_persistent_scope():
    """Create a sample persistent scope."""
    return PersistentScope(
        user_id="test-user",
        approved_tools=[
            ToolScope(
                tool_name="file_search",
                approved=True,
                max_risk_level="low",
                granted_by="admin",
            ),
        ],
    )


# =============================================================================
# SQLITE STORE TESTS
# =============================================================================


@pytest.mark.skipif(not SQLITE_HITL_AVAILABLE, reason="aiosqlite not available")
class TestSqliteHITLStore:
    """Tests for SqliteHITLStore."""

    @pytest.fixture
    async def sqlite_store(self, temp_db_path):
        """Create and initialize SQLite store."""
        store = SqliteHITLStore()
        await store.initialize({"path": str(temp_db_path)})
        yield store
        await store.close()

    # -------------------------------------------------------------------------
    # Request Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_and_get_request(self, sqlite_store, sample_request):
        """Should save and retrieve a request."""
        await sqlite_store.save_request(sample_request)

        retrieved = await sqlite_store.get_request(sample_request.request_id)

        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id
        assert retrieved.activity.activity_type == "bash_exec"
        assert retrieved.activity.parameters["command"] == "ls -la"
        assert retrieved.risk_assessment.overall_level == "medium"
        assert len(retrieved.risk_assessment.factors) == 2
        assert retrieved.status == ApprovalStatus.PENDING
        assert retrieved.session_id == "test-session-123"
        assert retrieved.priority == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_request(self, sqlite_store):
        """Should return None for nonexistent request."""
        result = await sqlite_store.get_request("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_request_status(self, sqlite_store, sample_request, sample_response):
        """Should update request status."""
        await sqlite_store.save_request(sample_request)

        success = await sqlite_store.update_request_status(
            sample_request.request_id,
            ApprovalStatus.APPROVED,
            sample_response,
        )

        assert success
        retrieved = await sqlite_store.get_request(sample_request.request_id)
        assert retrieved.status == ApprovalStatus.APPROVED

        response = await sqlite_store.get_response(sample_request.request_id)
        assert response is not None
        assert response.approved

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, sqlite_store):
        """Should get pending requests."""
        # Create multiple requests with different sessions
        for i in range(3):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium", requires_approval=True)
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="session-1" if i < 2 else "session-2",
            )
            req.set_expiration(300)
            await sqlite_store.save_request(req)

        # Get all pending
        all_pending = await sqlite_store.get_pending_requests()
        assert len(all_pending) == 3

        # Get pending for session-1
        session_pending = await sqlite_store.get_pending_requests(session_id="session-1")
        assert len(session_pending) == 2

    @pytest.mark.asyncio
    async def test_delete_request(self, sqlite_store, sample_request, sample_response):
        """Should delete request and response."""
        await sqlite_store.save_request(sample_request)
        await sqlite_store.save_response(sample_response)

        deleted = await sqlite_store.delete_request(sample_request.request_id)

        assert deleted
        assert await sqlite_store.get_request(sample_request.request_id) is None
        assert await sqlite_store.get_response(sample_request.request_id) is None

    # -------------------------------------------------------------------------
    # Response Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_and_get_response(self, sqlite_store, sample_request, sample_response):
        """Should save and retrieve response."""
        await sqlite_store.save_request(sample_request)
        await sqlite_store.save_response(sample_response)

        retrieved = await sqlite_store.get_response(sample_request.request_id)

        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id
        assert retrieved.approved
        assert retrieved.feedback == "Looks safe, approved"
        assert retrieved.responder_id == "test-user"
        assert retrieved.response_time_ms == 1500

    @pytest.mark.asyncio
    async def test_response_with_modified_params(self, sqlite_store, sample_request):
        """Should save response with modified parameters."""
        await sqlite_store.save_request(sample_request)

        response = HITLResponse(
            request_id=sample_request.request_id,
            approved=True,
            status=ApprovalStatus.MODIFIED,
            modified_parameters={"command": "ls -l", "safe_mode": True},
            responder_id="test-user",
        )
        await sqlite_store.save_response(response)

        retrieved = await sqlite_store.get_response(sample_request.request_id)
        assert retrieved.modified_parameters["command"] == "ls -l"
        assert retrieved.modified_parameters["safe_mode"]

    # -------------------------------------------------------------------------
    # Scope Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_and_get_session_scope(self, sqlite_store, sample_session_scope):
        """Should save and retrieve session scope."""
        await sqlite_store.save_session_scope(sample_session_scope)

        retrieved = await sqlite_store.get_session_scope(sample_session_scope.session_id)

        assert retrieved is not None
        assert retrieved.session_id == "test-session-123"
        assert len(retrieved.approved_tools) == 2
        assert retrieved.approved_tools[0].tool_name == "bash_exec"
        assert retrieved.approved_tools[0].max_risk_level == "medium"
        assert "ls" in retrieved.approved_tools[0].conditions["allowed_commands"]
        assert len(retrieved.approved_patterns) == 1

    @pytest.mark.asyncio
    async def test_save_and_get_persistent_scope(self, sqlite_store, sample_persistent_scope):
        """Should save and retrieve persistent scope."""
        await sqlite_store.save_persistent_scope(sample_persistent_scope)

        retrieved = await sqlite_store.get_persistent_scope(sample_persistent_scope.user_id)

        assert retrieved is not None
        assert retrieved.user_id == "test-user"
        assert len(retrieved.approved_tools) == 1
        assert retrieved.approved_tools[0].tool_name == "file_search"

    # -------------------------------------------------------------------------
    # Cleanup Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, sqlite_store):
        """Should cleanup expired requests."""
        # Create expired request
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium", requires_approval=True)
        expired_req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session",
        )
        expired_req.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await sqlite_store.save_request(expired_req)

        # Create valid request
        valid_req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session",
        )
        valid_req.set_expiration(300)
        await sqlite_store.save_request(valid_req)

        # Cleanup
        cleaned = await sqlite_store.cleanup_expired()

        assert cleaned == 1
        pending = await sqlite_store.get_pending_requests()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_count_pending(self, sqlite_store, sample_request):
        """Should count pending requests."""
        assert await sqlite_store.count_pending() == 0

        await sqlite_store.save_request(sample_request)
        assert await sqlite_store.count_pending() == 1

        await sqlite_store.update_request_status(
            sample_request.request_id,
            ApprovalStatus.APPROVED,
        )
        assert await sqlite_store.count_pending() == 0

    # -------------------------------------------------------------------------
    # Persistence Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_persistence_across_connections(self, temp_db_path, sample_request):
        """Should persist data across connections."""
        # Save with first store
        store1 = SqliteHITLStore()
        await store1.initialize({"path": str(temp_db_path)})
        await store1.save_request(sample_request)
        await store1.close()

        # Retrieve with second store
        store2 = SqliteHITLStore()
        await store2.initialize({"path": str(temp_db_path)})
        retrieved = await store2.get_request(sample_request.request_id)
        await store2.close()

        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialize_without_aiosqlite(self, temp_db_path, monkeypatch):
        """Should raise ImportError if aiosqlite not available."""
        import llmcore.agents.hitl.sqlite_state as sqlite_module

        monkeypatch.setattr(sqlite_module, "AIOSQLITE_AVAILABLE", False)

        store = SqliteHITLStore()
        with pytest.raises(ImportError, match="aiosqlite"):
            await store.initialize({"path": str(temp_db_path)})

    @pytest.mark.asyncio
    async def test_initialize_without_path(self):
        """Should raise ValueError if path not provided."""
        store = SqliteHITLStore()
        with pytest.raises(ValueError, match="path"):
            await store.initialize({})


# =============================================================================
# POSTGRES STORE TESTS
# =============================================================================


@pytest.mark.skipif(not POSTGRES_TESTS_ENABLED, reason="PostgreSQL not configured")
class TestPostgresHITLStore:
    """Tests for PostgresHITLStore."""

    @pytest.fixture
    async def postgres_store(self):
        """Create and initialize PostgreSQL store with unique table prefix."""
        import time

        store = PostgresHITLStore()
        # Use unique prefix to avoid conflicts between test runs
        prefix = f"test_{int(time.time() * 1000000)}_"
        await store.initialize(
            {
                "db_url": POSTGRES_TEST_URL,
                "table_prefix": prefix,
            }
        )
        yield store
        # Cleanup: drop test tables
        try:
            async with store._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP TABLE IF EXISTS {prefix}hitl_responses CASCADE")
                    await cur.execute(f"DROP TABLE IF EXISTS {prefix}hitl_requests CASCADE")
                    await cur.execute(f"DROP TABLE IF EXISTS {prefix}hitl_session_scopes CASCADE")
                    await cur.execute(
                        f"DROP TABLE IF EXISTS {prefix}hitl_persistent_scopes CASCADE"
                    )
                await conn.commit()
        except Exception:
            pass
        await store.close()

    # -------------------------------------------------------------------------
    # Request Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_and_get_request(self, postgres_store, sample_request):
        """Should save and retrieve a request."""
        await postgres_store.save_request(sample_request)

        retrieved = await postgres_store.get_request(sample_request.request_id)

        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id
        assert retrieved.activity.activity_type == "bash_exec"
        assert retrieved.activity.parameters["command"] == "ls -la"
        assert retrieved.risk_assessment.overall_level == "medium"
        assert len(retrieved.risk_assessment.factors) == 2
        assert retrieved.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_update_request_status(self, postgres_store, sample_request, sample_response):
        """Should update request status."""
        await postgres_store.save_request(sample_request)

        success = await postgres_store.update_request_status(
            sample_request.request_id,
            ApprovalStatus.APPROVED,
            sample_response,
        )

        assert success
        retrieved = await postgres_store.get_request(sample_request.request_id)
        assert retrieved.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, postgres_store):
        """Should get pending requests."""
        for i in range(3):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium", requires_approval=True)
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="session-1" if i < 2 else "session-2",
            )
            req.set_expiration(300)
            await postgres_store.save_request(req)

        all_pending = await postgres_store.get_pending_requests()
        assert len(all_pending) == 3

        session_pending = await postgres_store.get_pending_requests(session_id="session-1")
        assert len(session_pending) == 2

    # -------------------------------------------------------------------------
    # Response Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_and_get_response(self, postgres_store, sample_request, sample_response):
        """Should save and retrieve response."""
        await postgres_store.save_request(sample_request)
        await postgres_store.save_response(sample_response)

        retrieved = await postgres_store.get_response(sample_request.request_id)

        assert retrieved is not None
        assert retrieved.approved
        assert retrieved.feedback == "Looks safe, approved"

    # -------------------------------------------------------------------------
    # Scope Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_and_get_session_scope(self, postgres_store, sample_session_scope):
        """Should save and retrieve session scope."""
        await postgres_store.save_session_scope(sample_session_scope)

        retrieved = await postgres_store.get_session_scope(sample_session_scope.session_id)

        assert retrieved is not None
        assert len(retrieved.approved_tools) == 2

    @pytest.mark.asyncio
    async def test_save_and_get_persistent_scope(self, postgres_store, sample_persistent_scope):
        """Should save and retrieve persistent scope."""
        await postgres_store.save_persistent_scope(sample_persistent_scope)

        retrieved = await postgres_store.get_persistent_scope(sample_persistent_scope.user_id)

        assert retrieved is not None
        assert len(retrieved.approved_tools) == 1

    # -------------------------------------------------------------------------
    # Cleanup Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, postgres_store):
        """Should cleanup expired requests."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium", requires_approval=True)

        # Create expired request
        expired_req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session",
        )
        expired_req.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await postgres_store.save_request(expired_req)

        # Create valid request
        valid_req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session",
        )
        valid_req.set_expiration(300)
        await postgres_store.save_request(valid_req)

        cleaned = await postgres_store.cleanup_expired()

        assert cleaned == 1
        pending = await postgres_store.get_pending_requests()
        assert len(pending) == 1


# =============================================================================
# INTERFACE COMPLIANCE TESTS
# =============================================================================


class TestInterfaceCompliance:
    """Test that all stores comply with HITLStateStore interface."""

    @pytest.fixture
    def stores(self, temp_db_path):
        """Get list of available stores to test."""
        stores = [InMemoryHITLStore()]

        if SQLITE_HITL_AVAILABLE:
            stores.append(("sqlite", temp_db_path))

        return stores

    @pytest.mark.asyncio
    async def test_interface_methods_memory(self, sample_request, sample_response):
        """InMemoryHITLStore should implement full interface."""
        store = InMemoryHITLStore()

        # Test all interface methods exist and work
        await store.save_request(sample_request)
        assert await store.get_request(sample_request.request_id) is not None
        await store.save_response(sample_response)
        assert await store.get_response(sample_request.request_id) is not None
        assert await store.update_request_status(sample_request.request_id, ApprovalStatus.APPROVED)
        pending = await store.get_pending_requests()
        assert isinstance(pending, list)
        assert await store.count_pending() >= 0
        cleaned = await store.cleanup_expired()
        assert cleaned >= 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SQLITE_HITL_AVAILABLE, reason="SQLite not available")
    async def test_interface_methods_sqlite(self, temp_db_path, sample_request, sample_response):
        """SqliteHITLStore should implement full interface."""
        store = SqliteHITLStore()
        await store.initialize({"path": str(temp_db_path)})

        try:
            await store.save_request(sample_request)
            assert await store.get_request(sample_request.request_id) is not None
            await store.save_response(sample_response)
            assert await store.get_response(sample_request.request_id) is not None
            assert await store.update_request_status(
                sample_request.request_id, ApprovalStatus.APPROVED
            )
            pending = await store.get_pending_requests()
            assert isinstance(pending, list)
            assert await store.count_pending() >= 0
            cleaned = await store.cleanup_expired()
            assert cleaned >= 0
        finally:
            await store.close()


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================


@pytest.mark.skipif(not SQLITE_HITL_AVAILABLE, reason="SQLite not available")
class TestConcurrentAccess:
    """Test concurrent access handling."""

    @pytest.fixture
    async def sqlite_store(self, temp_db_path):
        """Create SQLite store for concurrent tests."""
        store = SqliteHITLStore()
        await store.initialize({"path": str(temp_db_path)})
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, sqlite_store):
        """Should handle concurrent saves."""

        async def save_request(i):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={"index": i})
            risk = RiskAssessment(overall_level="medium", requires_approval=True)
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session",
            )
            req.set_expiration(300)
            await sqlite_store.save_request(req)

        # Concurrent saves
        await asyncio.gather(*[save_request(i) for i in range(10)])

        pending = await sqlite_store.get_pending_requests()
        assert len(pending) == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_writes(self, sqlite_store):
        """Should handle concurrent reads and writes."""
        # Pre-populate
        for i in range(5):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium", requires_approval=True)
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session",
            )
            req.set_expiration(300)
            await sqlite_store.save_request(req)

        async def read_pending():
            return await sqlite_store.get_pending_requests()

        async def save_new(i):
            activity = ActivityInfo(activity_type=f"new_tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium", requires_approval=True)
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session",
            )
            req.set_expiration(300)
            await sqlite_store.save_request(req)

        # Concurrent reads and writes
        tasks = [read_pending() for _ in range(5)] + [save_new(i) for i in range(5)]
        await asyncio.gather(*tasks)

        final = await sqlite_store.get_pending_requests()
        assert len(final) == 10


# =============================================================================
# EDGE CASES
# =============================================================================


@pytest.mark.skipif(not SQLITE_HITL_AVAILABLE, reason="SQLite not available")
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    async def sqlite_store(self, temp_db_path):
        """Create SQLite store."""
        store = SqliteHITLStore()
        await store.initialize({"path": str(temp_db_path)})
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_empty_store(self, sqlite_store):
        """Should handle empty store gracefully."""
        pending = await sqlite_store.get_pending_requests()
        assert pending == []

        cleaned = await sqlite_store.cleanup_expired()
        assert cleaned == 0

        count = await sqlite_store.count_pending()
        assert count == 0

    @pytest.mark.asyncio
    async def test_request_without_expiration(self, sqlite_store):
        """Should handle request without expiration."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium", requires_approval=True)
        req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session",
        )
        # No expiration set

        await sqlite_store.save_request(req)

        pending = await sqlite_store.get_pending_requests()
        assert len(pending) == 1

        # Should not be cleaned up
        cleaned = await sqlite_store.cleanup_expired()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_update_nonexistent_request(self, sqlite_store):
        """Should return False for nonexistent request."""
        result = await sqlite_store.update_request_status(
            "nonexistent-id",
            ApprovalStatus.APPROVED,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_request_with_empty_parameters(self, sqlite_store):
        """Should handle request with empty parameters."""
        activity = ActivityInfo(activity_type="minimal", parameters={})
        risk = RiskAssessment(overall_level="low", requires_approval=False)
        req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
        )
        req.set_expiration(300)

        await sqlite_store.save_request(req)

        retrieved = await sqlite_store.get_request(req.request_id)
        assert retrieved is not None
        assert retrieved.activity.parameters == {}

    @pytest.mark.asyncio
    async def test_request_with_complex_parameters(self, sqlite_store):
        """Should handle request with complex nested parameters."""
        activity = ActivityInfo(
            activity_type="complex",
            parameters={
                "nested": {"a": 1, "b": [1, 2, 3]},
                "list": [{"x": 1}, {"y": 2}],
                "unicode": "日本語テスト",
                "special": "quotes\"and'stuff",
            },
        )
        risk = RiskAssessment(
            overall_level="medium",
            factors=[
                RiskFactor(name="test", level="low", reason="Testing", weight=0.5),
            ],
            requires_approval=True,
            dangerous_patterns=["pattern1", "pattern2"],
        )
        req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
        )
        req.set_expiration(300)

        await sqlite_store.save_request(req)

        retrieved = await sqlite_store.get_request(req.request_id)
        assert retrieved.activity.parameters["nested"]["a"] == 1
        assert retrieved.activity.parameters["unicode"] == "日本語テスト"
        assert len(retrieved.risk_assessment.dangerous_patterns) == 2


# =============================================================================
# PARITY TESTS - Ensure SQLite and Postgres behave identically
# =============================================================================


@pytest.mark.skipif(
    not (SQLITE_HITL_AVAILABLE and POSTGRES_TESTS_ENABLED),
    reason="Both SQLite and PostgreSQL required",
)
class TestStoreParity:
    """Ensure SQLite and PostgreSQL stores behave identically."""

    @pytest.fixture
    async def stores(self, temp_db_path):
        """Create both stores."""
        import time

        sqlite = SqliteHITLStore()
        await sqlite.initialize({"path": str(temp_db_path)})

        postgres = PostgresHITLStore()
        prefix = f"parity_{int(time.time() * 1000000)}_"
        await postgres.initialize(
            {
                "db_url": POSTGRES_TEST_URL,
                "table_prefix": prefix,
            }
        )

        yield {"sqlite": sqlite, "postgres": postgres, "prefix": prefix}

        await sqlite.close()
        try:
            async with postgres._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP TABLE IF EXISTS {prefix}hitl_responses CASCADE")
                    await cur.execute(f"DROP TABLE IF EXISTS {prefix}hitl_requests CASCADE")
                    await cur.execute(f"DROP TABLE IF EXISTS {prefix}hitl_session_scopes CASCADE")
                    await cur.execute(
                        f"DROP TABLE IF EXISTS {prefix}hitl_persistent_scopes CASCADE"
                    )
                await conn.commit()
        except Exception:
            pass
        await postgres.close()

    @pytest.mark.asyncio
    async def test_request_parity(self, stores, sample_request):
        """Both stores should produce identical request roundtrips."""
        for name, store in [("sqlite", stores["sqlite"]), ("postgres", stores["postgres"])]:
            await store.save_request(sample_request)
            retrieved = await store.get_request(sample_request.request_id)

            assert retrieved is not None, f"{name} failed to retrieve"
            assert retrieved.activity.activity_type == sample_request.activity.activity_type
            assert retrieved.activity.parameters == sample_request.activity.parameters
            assert (
                retrieved.risk_assessment.overall_level
                == sample_request.risk_assessment.overall_level
            )
            assert len(retrieved.risk_assessment.factors) == len(
                sample_request.risk_assessment.factors
            )

    @pytest.mark.asyncio
    async def test_response_parity(self, stores, sample_request, sample_response):
        """Both stores should produce identical response roundtrips."""
        for name, store in [("sqlite", stores["sqlite"]), ("postgres", stores["postgres"])]:
            # Need unique request IDs for each store
            from copy import deepcopy

            req = deepcopy(sample_request)
            req.request_id = f"{req.request_id}_{name}"
            resp = deepcopy(sample_response)
            resp.request_id = req.request_id

            await store.save_request(req)
            await store.save_response(resp)
            retrieved = await store.get_response(req.request_id)

            assert retrieved is not None, f"{name} failed to retrieve response"
            assert retrieved.approved == resp.approved
            assert retrieved.feedback == resp.feedback
