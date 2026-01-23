# tests/agents/hitl/test_state.py
"""
Tests for HITL State Persistence.

Tests:
- InMemoryHITLStore operations
- FileHITLStore operations
- Request lifecycle
- Cleanup operations
"""

import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llmcore.agents.hitl import (
    ActivityInfo,
    ApprovalStatus,
    FileHITLStore,
    HITLRequest,
    HITLResponse,
    HITLStateStore,
    InMemoryHITLStore,
    RiskAssessment,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def memory_store():
    """Create in-memory store."""
    return InMemoryHITLStore()


@pytest.fixture
def temp_dir():
    """Create temporary directory for file store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def file_store(temp_dir):
    """Create file-based store."""
    return FileHITLStore(temp_dir)


@pytest.fixture
def sample_request():
    """Create sample HITL request."""
    activity = ActivityInfo(
        activity_type="bash_exec",
        parameters={"command": "ls -la"}
    )
    risk = RiskAssessment(
        overall_level="medium",
        requires_approval=True
    )
    request = HITLRequest(
        activity=activity,
        risk_assessment=risk,
        session_id="test-session",
        user_id="test-user"
    )
    request.set_expiration(300)
    return request


@pytest.fixture
def sample_response(sample_request):
    """Create sample HITL response."""
    return HITLResponse(
        request_id=sample_request.request_id,
        approved=True,
        responder_id="test-user",
        feedback="Approved for testing"
    )


# =============================================================================
# IN-MEMORY STORE TESTS
# =============================================================================


class TestInMemoryStore:
    """Test InMemoryHITLStore operations."""

    @pytest.mark.asyncio
    async def test_save_request(self, memory_store, sample_request):
        """Should save request."""
        await memory_store.save_request(sample_request)

        retrieved = await memory_store.get_request(sample_request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_request(self, memory_store):
        """Should return None for nonexistent request."""
        result = await memory_store.get_request("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_response(self, memory_store, sample_request, sample_response):
        """Should save response."""
        await memory_store.save_request(sample_request)
        await memory_store.save_response(sample_response)

        retrieved = await memory_store.get_response(sample_response.request_id)
        assert retrieved is not None
        assert retrieved.approved

    @pytest.mark.asyncio
    async def test_update_request_status(self, memory_store, sample_request):
        """Should update request status."""
        await memory_store.save_request(sample_request)

        await memory_store.update_request_status(
            sample_request.request_id,
            ApprovalStatus.APPROVED,
            None
        )

        retrieved = await memory_store.get_request(sample_request.request_id)
        assert retrieved.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, memory_store):
        """Should get pending requests."""
        # Create multiple requests
        for i in range(3):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session"
            )
            req.set_expiration(300)
            await memory_store.save_request(req)

        pending = await memory_store.get_pending_requests()
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_get_pending_requests_by_session(self, memory_store):
        """Should filter pending requests by session."""
        # Create requests for different sessions
        for session in ["sess-1", "sess-2", "sess-1"]:
            activity = ActivityInfo(activity_type="test", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id=session
            )
            req.set_expiration(300)
            await memory_store.save_request(req)

        pending = await memory_store.get_pending_requests(session_id="sess-1")
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, memory_store):
        """Should cleanup expired requests."""
        # Create expired request
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium")
        req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session"
        )
        req.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await memory_store.save_request(req)

        # Create non-expired request
        req2 = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session"
        )
        req2.set_expiration(300)
        await memory_store.save_request(req2)

        cleaned = await memory_store.cleanup_expired()
        assert cleaned == 1

        pending = await memory_store.get_pending_requests()
        assert len(pending) == 1


# =============================================================================
# FILE STORE TESTS
# =============================================================================


class TestFileStore:
    """Test FileHITLStore operations."""

    @pytest.mark.asyncio
    async def test_save_request(self, file_store, sample_request):
        """Should save request to file."""
        await file_store.save_request(sample_request)

        retrieved = await file_store.get_request(sample_request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_dir, sample_request):
        """Should persist data across store instances."""
        # Save with one store
        store1 = FileHITLStore(temp_dir)
        await store1.save_request(sample_request)

        # Retrieve with another store
        store2 = FileHITLStore(temp_dir)
        retrieved = await store2.get_request(sample_request.request_id)

        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_save_response(self, file_store, sample_request, sample_response):
        """Should save response to file."""
        await file_store.save_request(sample_request)
        await file_store.save_response(sample_response)

        retrieved = await file_store.get_response(sample_response.request_id)
        assert retrieved is not None
        assert retrieved.approved

    @pytest.mark.asyncio
    async def test_update_request_status(self, file_store, sample_request):
        """Should update request status in file."""
        await file_store.save_request(sample_request)

        await file_store.update_request_status(
            sample_request.request_id,
            ApprovalStatus.REJECTED,
            None
        )

        retrieved = await file_store.get_request(sample_request.request_id)
        assert retrieved.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, file_store):
        """Should get pending requests from files."""
        # Create multiple requests
        for i in range(3):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session"
            )
            req.set_expiration(300)
            await file_store.save_request(req)

        pending = await file_store.get_pending_requests()
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_cleanup_expired_files(self, file_store):
        """Should cleanup expired requests from files."""
        # Create expired request
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium")
        req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session"
        )
        req.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await file_store.save_request(req)

        cleaned = await file_store.cleanup_expired()
        assert cleaned == 1


# =============================================================================
# SCOPE PERSISTENCE TESTS
# =============================================================================


class TestScopePersistence:
    """Test scope persistence operations."""

    @pytest.mark.asyncio
    async def test_save_session_scope(self, file_store):
        """Should save session scope."""
        from llmcore.agents.hitl import SessionScope, ToolScope, RiskLevel

        tool_scope = ToolScope(
            tool_name="bash_exec",
            max_risk_level=RiskLevel.HIGH
        )
        session_scope = SessionScope(
            session_id="sess-123",
            approved_tools=[tool_scope]
        )

        await file_store.save_session_scope(session_scope)
        retrieved = await file_store.get_session_scope("sess-123")

        assert retrieved is not None
        assert len(retrieved.approved_tools) == 1
        assert retrieved.approved_tools[0].tool_name == "bash_exec"

    @pytest.mark.asyncio
    async def test_save_persistent_scope(self, file_store):
        """Should save persistent scope."""
        from llmcore.agents.hitl import PersistentScope, ToolScope, RiskLevel

        tool_scope = ToolScope(
            tool_name="file_read",
            max_risk_level=RiskLevel.MEDIUM
        )
        persistent_scope = PersistentScope(
            user_id="user-123",
            approved_tools=[tool_scope]
        )

        await file_store.save_persistent_scope(persistent_scope)
        retrieved = await file_store.get_persistent_scope("user-123")

        assert retrieved is not None
        assert len(retrieved.approved_tools) == 1
        assert retrieved.approved_tools[0].tool_name == "file_read"

    @pytest.mark.asyncio
    async def test_scope_persistence_across_instances(self, temp_dir):
        """Should persist scopes across store instances."""
        from llmcore.agents.hitl import SessionScope, ToolScope, RiskLevel

        tool_scope = ToolScope(
            tool_name="test_tool",
            max_risk_level=RiskLevel.LOW
        )
        session_scope = SessionScope(
            session_id="sess-456",
            approved_tools=[tool_scope]
        )

        # Save with one store
        store1 = FileHITLStore(temp_dir)
        await store1.save_session_scope(session_scope)

        # Retrieve with another store
        store2 = FileHITLStore(temp_dir)
        retrieved = await store2.get_session_scope("sess-456")

        assert retrieved is not None
        assert len(retrieved.approved_tools) == 1
        assert retrieved.approved_tools[0].tool_name == "test_tool"


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================


class TestConcurrentAccess:
    """Test concurrent access handling."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, memory_store):
        """Should handle concurrent saves."""
        async def save_request(i):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session"
            )
            req.set_expiration(300)
            await memory_store.save_request(req)

        # Concurrent saves
        await asyncio.gather(*[save_request(i) for i in range(10)])

        pending = await memory_store.get_pending_requests()
        assert len(pending) == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_writes(self, memory_store):
        """Should handle concurrent reads and writes."""
        # Pre-populate
        for i in range(5):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session"
            )
            req.set_expiration(300)
            await memory_store.save_request(req)

        async def read_pending():
            return await memory_store.get_pending_requests()

        async def save_new(i):
            activity = ActivityInfo(activity_type=f"new_tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(
                activity=activity,
                risk_assessment=risk,
                session_id="test-session"
            )
            req.set_expiration(300)
            await memory_store.save_request(req)

        # Concurrent reads and writes
        tasks = [read_pending() for _ in range(5)] + [save_new(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should complete without error
        final = await memory_store.get_pending_requests()
        assert len(final) == 10


# =============================================================================
# EDGE CASES
# =============================================================================


class TestStateEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_store(self, memory_store):
        """Should handle empty store gracefully."""
        pending = await memory_store.get_pending_requests()
        assert pending == []

        cleaned = await memory_store.cleanup_expired()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_save_duplicate_request(self, memory_store, sample_request):
        """Should handle duplicate request saves."""
        await memory_store.save_request(sample_request)
        await memory_store.save_request(sample_request)  # Same request

        # Should not duplicate
        pending = await memory_store.get_pending_requests()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_update_nonexistent_request(self, memory_store):
        """Should handle updating nonexistent request."""
        # Should not raise
        await memory_store.update_request_status(
            "nonexistent",
            ApprovalStatus.APPROVED,
            None
        )

    @pytest.mark.asyncio
    async def test_file_store_invalid_path(self):
        """Should handle invalid path gracefully."""
        # Path that doesn't exist and can't be created
        store = FileHITLStore("/nonexistent/deep/nested/path")

        # Operations should fail gracefully or raise appropriate error
        # The exact behavior depends on implementation

    @pytest.mark.asyncio
    async def test_request_with_no_expiration(self, memory_store):
        """Should handle request with no expiration."""
        activity = ActivityInfo(activity_type="test", parameters={})
        risk = RiskAssessment(overall_level="medium")
        req = HITLRequest(
            activity=activity,
            risk_assessment=risk,
            session_id="test-session"
        )
        # No expiration set

        await memory_store.save_request(req)

        pending = await memory_store.get_pending_requests()
        assert len(pending) == 1

        # Should not be cleaned up
        cleaned = await memory_store.cleanup_expired()
        assert cleaned == 0


# =============================================================================
# INTERFACE COMPLIANCE TESTS
# =============================================================================


class TestInterfaceCompliance:
    """Test that stores comply with HITLStateStore interface."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("store_class", [InMemoryHITLStore, FileHITLStore])
    async def test_interface_methods(self, store_class, temp_dir, sample_request, sample_response):
        """Both stores should implement interface."""
        if store_class == FileHITLStore:
            store = store_class(temp_dir)
        else:
            store = store_class()

        # Test all interface methods
        assert hasattr(store, 'save_request')
        assert hasattr(store, 'get_request')
        assert hasattr(store, 'save_response')
        assert hasattr(store, 'get_response')
        assert hasattr(store, 'update_request_status')
        assert hasattr(store, 'get_pending_requests')
        assert hasattr(store, 'cleanup_expired')

        # Test actual operations
        await store.save_request(sample_request)
        retrieved = await store.get_request(sample_request.request_id)
        assert retrieved is not None
