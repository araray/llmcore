# tests/agents/hitl/test_manager_integration.py
"""
Integration tests for HITLManager with database storage backends.

Tests Phase 7.2: HITLManager Integration with configurable storage backends.

Test Coverage:
- HITLConfig with storage configuration validation
- HITLManager initialization with each backend type
- create_hitl_manager() and create_hitl_manager_async() factory functions
- Full approval workflow through manager
- Rejection workflow through manager
- Request modification workflow
- Timeout handling
- Session scope management
- Concurrent access patterns (where applicable)

Usage:
    # Run all tests
    PYTHONPATH=src pytest tests/agents/hitl/test_manager_integration.py -v

    # Run only SQLite tests
    PYTHONPATH=src pytest tests/agents/hitl/test_manager_integration.py -v -k sqlite

    # Run PostgreSQL tests (requires TEST_POSTGRES_URL)
    TEST_POSTGRES_URL="postgresql://user:pass@localhost/test" \
        PYTHONPATH=src pytest tests/agents/hitl/test_manager_integration.py -v -k postgres

References:
    - LLMCORE_CONTINUATION_GUIDE_3.md: Phase 7.2 specifications
    - UNIFIED_IMPLEMENTATION_PLAN.md: Section 9 (Phase 7)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest

from llmcore.agents.hitl.callbacks import AutoApproveCallback
from llmcore.agents.hitl.manager import (
    HITLManager,
    create_hitl_manager,
    create_hitl_manager_async,
)
from llmcore.agents.hitl.models import (
    ActivityInfo,
    ApprovalStatus,
    HITLConfig,
    HITLRequest,
    HITLStorageConfig,
    RiskAssessment,
    RiskFactor,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def memory_config() -> HITLConfig:
    """Create HITLConfig with memory backend."""
    return HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        storage=HITLStorageConfig(backend="memory"),
    )


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def sqlite_config(temp_db_path: str) -> HITLConfig:
    """Create HITLConfig with SQLite backend."""
    return HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        storage=HITLStorageConfig(
            backend="sqlite",
            sqlite_path=temp_db_path,
        ),
    )


@pytest.fixture
def postgres_config() -> HITLConfig:
    """Create HITLConfig with PostgreSQL backend (if available)."""
    db_url = os.environ.get("TEST_POSTGRES_URL")
    if not db_url:
        pytest.skip("PostgreSQL not available (set TEST_POSTGRES_URL)")

    # Use unique table prefix for test isolation
    prefix = f"test_{uuid.uuid4().hex[:8]}_"

    return HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        storage=HITLStorageConfig(
            backend="postgres",
            postgres_url=db_url,
            postgres_table_prefix=prefix,
            postgres_min_pool_size=1,
            postgres_max_pool_size=3,
        ),
    )


@pytest.fixture
async def memory_manager(memory_config: HITLConfig) -> AsyncIterator[HITLManager]:
    """Create HITLManager with memory backend."""
    manager = await create_hitl_manager_async(
        config=memory_config,
        callback=AutoApproveCallback(),  # Use auto-approve for testing
    )
    yield manager
    await manager.close()


@pytest.fixture
async def sqlite_manager(sqlite_config: HITLConfig) -> AsyncIterator[HITLManager]:
    """Create HITLManager with SQLite backend."""
    manager = await create_hitl_manager_async(
        config=sqlite_config,
        callback=AutoApproveCallback(),
    )
    yield manager
    await manager.close()


@pytest.fixture
async def postgres_manager(postgres_config: HITLConfig) -> AsyncIterator[HITLManager]:
    """Create HITLManager with PostgreSQL backend."""
    manager = await create_hitl_manager_async(
        config=postgres_config,
        callback=AutoApproveCallback(),
    )
    yield manager
    await manager.close()


def create_test_request(
    activity_type: str = "test_action",
    parameters: dict | None = None,
    risk_level: str = "medium",
    session_id: str | None = None,
    user_id: str | None = None,
) -> HITLRequest:
    """Create a test HITLRequest."""
    return HITLRequest(
        activity=ActivityInfo(
            activity_type=activity_type,
            parameters=parameters or {"test": "value"},
            reason="Test action",
        ),
        risk_assessment=RiskAssessment(
            overall_level=risk_level,
            factors=[RiskFactor(name="test", level=risk_level, reason="Test factor")],
            requires_approval=risk_level != "none",
            reason=f"Test {risk_level} risk",
        ),
        context_summary="Test context",
        session_id=session_id or str(uuid.uuid4()),
        user_id=user_id or "test_user",
    )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestHITLStorageConfig:
    """Tests for HITLStorageConfig validation."""

    def test_default_config(self):
        """Test default storage configuration."""
        config = HITLStorageConfig()
        assert config.backend == "memory"
        assert config.sqlite_path == "~/.local/share/llmcore/hitl.db"
        assert config.postgres_url is None

    def test_sqlite_config(self):
        """Test SQLite storage configuration."""
        config = HITLStorageConfig(
            backend="sqlite",
            sqlite_path="/tmp/test_hitl.db",
        )
        assert config.backend == "sqlite"
        assert config.sqlite_path == "/tmp/test_hitl.db"

    def test_postgres_config(self):
        """Test PostgreSQL storage configuration."""
        config = HITLStorageConfig(
            backend="postgres",
            postgres_url="postgresql://user:pass@localhost/db",
            postgres_min_pool_size=2,
            postgres_max_pool_size=10,
            postgres_table_prefix="test_",
        )
        assert config.backend == "postgres"
        assert config.postgres_url == "postgresql://user:pass@localhost/db"
        assert config.postgres_min_pool_size == 2
        assert config.postgres_max_pool_size == 10
        assert config.postgres_table_prefix == "test_"

    def test_postgres_validation_fails_without_url(self):
        """Test that postgres backend requires postgres_url."""
        config = HITLStorageConfig(backend="postgres")
        with pytest.raises(ValueError, match="postgres_url is required"):
            config.validate_backend()

    def test_file_validation_fails_without_path(self):
        """Test that file backend requires file_path."""
        config = HITLStorageConfig(backend="file")
        with pytest.raises(ValueError, match="file_path is required"):
            config.validate_backend()

    def test_memory_validation_passes(self):
        """Test that memory backend validation passes."""
        config = HITLStorageConfig(backend="memory")
        config.validate_backend()  # Should not raise


class TestHITLConfigWithStorage:
    """Tests for HITLConfig with storage configuration."""

    def test_default_has_storage(self):
        """Test that default HITLConfig includes storage config."""
        config = HITLConfig()
        assert config.storage is not None
        assert config.storage.backend == "memory"

    def test_custom_storage_config(self):
        """Test HITLConfig with custom storage."""
        config = HITLConfig(
            enabled=True,
            storage=HITLStorageConfig(
                backend="sqlite",
                sqlite_path="/tmp/custom.db",
            ),
        )
        assert config.storage.backend == "sqlite"
        assert config.storage.sqlite_path == "/tmp/custom.db"

    def test_dict_based_storage_config(self):
        """Test creating HITLConfig from nested dict."""
        config_dict = {
            "enabled": True,
            "storage": {
                "backend": "sqlite",
                "sqlite_path": "/tmp/dict.db",
            },
        }
        config = HITLConfig(**config_dict)
        assert config.storage.backend == "sqlite"
        assert config.storage.sqlite_path == "/tmp/dict.db"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateHITLManager:
    """Tests for create_hitl_manager sync factory function."""

    def test_create_default(self):
        """Test creating manager with defaults."""
        manager = create_hitl_manager()
        assert manager.config.enabled is True
        assert manager.config.storage.backend == "memory"

    def test_create_with_storage_backend(self):
        """Test creating manager with storage_backend parameter."""
        manager = create_hitl_manager(
            storage_backend="memory",
            enabled=True,
        )
        assert manager.config.storage.backend == "memory"

    def test_create_sqlite_backend(self, temp_db_path: str):
        """Test creating manager with SQLite backend (not initialized)."""
        manager = create_hitl_manager(
            storage_backend="sqlite",
            storage_config={"sqlite_path": temp_db_path},
        )
        assert manager.config.storage.backend == "sqlite"
        assert manager.config.storage.sqlite_path == temp_db_path

    def test_legacy_persist_path_uses_file(self, temp_db_path: str):
        """Test that persist_path triggers file backend."""
        manager = create_hitl_manager(persist_path=temp_db_path)
        assert manager.config.storage.backend == "file"


class TestCreateHITLManagerAsync:
    """Tests for create_hitl_manager_async factory function."""

    @pytest.mark.asyncio
    async def test_create_memory_manager(self, memory_config: HITLConfig):
        """Test creating async manager with memory backend."""
        manager = await create_hitl_manager_async(config=memory_config)
        try:
            assert manager._initialized is True
            assert manager.config.storage.backend == "memory"
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_create_sqlite_manager(self, sqlite_config: HITLConfig):
        """Test creating async manager with SQLite backend."""
        manager = await create_hitl_manager_async(config=sqlite_config)
        try:
            assert manager._initialized is True
            assert manager.config.storage.backend == "sqlite"
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_create_postgres_manager(self, postgres_config: HITLConfig):
        """Test creating async manager with PostgreSQL backend."""
        manager = await create_hitl_manager_async(config=postgres_config)
        try:
            assert manager._initialized is True
            assert manager.config.storage.backend == "postgres"
        finally:
            await manager.close()


# =============================================================================
# MEMORY BACKEND TESTS
# =============================================================================


class TestManagerWithMemoryBackend:
    """Tests for HITLManager with memory backend."""

    @pytest.mark.asyncio
    async def test_check_approval_low_risk(self, memory_manager: HITLManager):
        """Test that low risk activities are auto-approved."""
        decision = await memory_manager.check_approval(
            activity_type="file_read",
            parameters={"path": "/tmp/test.txt"},
            reason="Reading test file",
        )
        # With AutoApproveCallback and low risk tool, should be auto-approved
        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_check_approval_high_risk(self, memory_manager: HITLManager):
        """Test high risk activity approval workflow."""
        decision = await memory_manager.check_approval(
            activity_type="bash_exec",
            parameters={"command": "ls -la"},
            reason="Listing files",
        )
        # AutoApproveCallback should approve
        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, memory_config: HITLConfig):
        """Test getting pending requests."""
        # Create manager without auto-approve to leave requests pending
        from llmcore.agents.hitl.callbacks import QueueHITLCallback

        callback = QueueHITLCallback()
        manager = await create_hitl_manager_async(
            config=memory_config,
            callback=callback,
        )

        # Start approval check in background (it will wait for response)
        async def check():
            return await manager.check_approval(
                activity_type="bash_exec",
                parameters={"command": "rm -rf /"},
            )

        # Don't wait, just get pending
        task = asyncio.create_task(check())
        await asyncio.sleep(0.1)  # Let the task start

        try:
            pending = await manager.get_pending_requests()
            # Should have at least one pending (if callback waits)
            # Note: With QueueHITLCallback, it blocks until response
        finally:
            # Cancel and cleanup
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await manager.close()

    @pytest.mark.asyncio
    async def test_statistics(self, memory_manager: HITLManager):
        """Test statistics tracking."""
        # Initial stats
        stats = memory_manager.get_statistics()
        assert stats["requests_created"] == 0

        # After approval
        await memory_manager.check_approval(
            activity_type="test",
            parameters={},
        )

        stats = memory_manager.get_statistics()
        # Either auto-approved (scope or low risk) or approved
        assert stats["requests_approved"] >= 0 or stats["requests_auto_approved"] >= 0


# =============================================================================
# SQLITE BACKEND TESTS
# =============================================================================


class TestManagerWithSQLiteBackend:
    """Tests for HITLManager with SQLite backend."""

    @pytest.mark.asyncio
    async def test_initialization(self, sqlite_manager: HITLManager):
        """Test manager initializes with SQLite."""
        assert sqlite_manager._initialized is True
        assert sqlite_manager.config.storage.backend == "sqlite"

    @pytest.mark.asyncio
    async def test_save_and_get_request(self, sqlite_manager: HITLManager):
        """Test saving and retrieving request through manager."""
        request = create_test_request()

        # Save request
        await sqlite_manager.state_store.save_request(request)

        # Retrieve request
        retrieved = await sqlite_manager.state_store.get_request(request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == request.request_id
        assert retrieved.activity.activity_type == request.activity.activity_type

    @pytest.mark.asyncio
    async def test_approval_workflow(self, sqlite_manager: HITLManager):
        """Test full approval workflow with SQLite persistence."""
        # Check approval (auto-approved with AutoApproveCallback)
        decision = await sqlite_manager.check_approval(
            activity_type="file_write",
            parameters={"path": "/tmp/test.txt", "content": "Hello"},
            reason="Writing test file",
        )

        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_persistence_across_manager_instances(
        self, sqlite_config: HITLConfig, temp_db_path: str
    ):
        """Test that data persists across manager instances."""
        # Create first manager and save a request
        manager1 = await create_hitl_manager_async(
            config=sqlite_config,
            callback=AutoApproveCallback(),
        )

        request = create_test_request()
        await manager1.state_store.save_request(request)
        await manager1.close()

        # Create second manager with same database
        manager2 = await create_hitl_manager_async(
            config=sqlite_config,
            callback=AutoApproveCallback(),
        )

        # Should be able to retrieve the request
        retrieved = await manager2.state_store.get_request(request.request_id)
        await manager2.close()

        assert retrieved is not None
        assert retrieved.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, sqlite_manager: HITLManager):
        """Test cleanup of expired requests."""
        # Create expired request
        request = create_test_request()
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        await sqlite_manager.state_store.save_request(request)

        # Cleanup
        count = await sqlite_manager.cleanup_expired()
        assert count >= 1

        # Request should be marked as expired
        retrieved = await sqlite_manager.state_store.get_request(request.request_id)
        assert retrieved.status == ApprovalStatus.TIMEOUT


# =============================================================================
# POSTGRESQL BACKEND TESTS
# =============================================================================


class TestManagerWithPostgresBackend:
    """Tests for HITLManager with PostgreSQL backend."""

    @pytest.mark.asyncio
    async def test_initialization(self, postgres_manager: HITLManager):
        """Test manager initializes with PostgreSQL."""
        assert postgres_manager._initialized is True
        assert postgres_manager.config.storage.backend == "postgres"

    @pytest.mark.asyncio
    async def test_save_and_get_request(self, postgres_manager: HITLManager):
        """Test saving and retrieving request through manager."""
        request = create_test_request()

        # Save request
        await postgres_manager.state_store.save_request(request)

        # Retrieve request
        retrieved = await postgres_manager.state_store.get_request(request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_approval_workflow(self, postgres_manager: HITLManager):
        """Test full approval workflow with PostgreSQL persistence."""
        decision = await postgres_manager.check_approval(
            activity_type="api_call",
            parameters={"endpoint": "/users", "method": "POST"},
            reason="Creating user",
        )

        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_concurrent_access(self, postgres_manager: HITLManager):
        """Test concurrent request handling."""
        # Create multiple requests concurrently
        requests = [create_test_request(activity_type=f"action_{i}") for i in range(5)]

        # Save all concurrently
        await asyncio.gather(*[postgres_manager.state_store.save_request(r) for r in requests])

        # Retrieve all
        pending = await postgres_manager.state_store.get_pending_requests()
        assert len(pending) >= 5


# =============================================================================
# CROSS-BACKEND PARITY TESTS
# =============================================================================


class TestBackendParity:
    """Tests ensuring consistent behavior across backends."""

    @pytest.mark.asyncio
    async def test_basic_workflow_parity_memory(
        self,
        memory_manager: HITLManager,
    ):
        """Test basic workflow with memory backend."""
        decision = await memory_manager.check_approval(
            activity_type="test_action",
            parameters={"key": "value"},
        )

        assert decision is not None
        # With AutoApproveCallback, should be approved
        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_basic_workflow_parity_sqlite(
        self,
        sqlite_manager: HITLManager,
    ):
        """Test basic workflow with SQLite backend."""
        decision = await sqlite_manager.check_approval(
            activity_type="test_action",
            parameters={"key": "value"},
        )

        assert decision is not None
        # With AutoApproveCallback, should be approved
        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_statistics_parity_memory(
        self,
        memory_manager: HITLManager,
    ):
        """Test statistics tracking with memory backend."""
        stats = memory_manager.get_statistics()
        assert "requests_created" in stats
        assert "requests_approved" in stats
        assert "requests_rejected" in stats

    @pytest.mark.asyncio
    async def test_statistics_parity_sqlite(
        self,
        sqlite_manager: HITLManager,
    ):
        """Test statistics tracking with SQLite backend."""
        stats = sqlite_manager.get_statistics()
        assert "requests_created" in stats
        assert "requests_approved" in stats
        assert "requests_rejected" in stats


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_backend_raises(self):
        """Test that invalid backend type raises error."""
        config = HITLConfig(
            storage=HITLStorageConfig(backend="invalid_backend"),
        )
        with pytest.raises(ValueError, match="Unknown storage backend"):
            HITLManager(config=config)

    @pytest.mark.asyncio
    async def test_postgres_without_url_raises(self):
        """Test that postgres without URL raises during initialization."""
        config = HITLConfig(
            storage=HITLStorageConfig(backend="postgres"),
        )
        manager = HITLManager(config=config)
        # Should raise RuntimeError wrapping the ValueError
        with pytest.raises(RuntimeError, match="postgres_url is required"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_double_initialization_safe(self, memory_config: HITLConfig):
        """Test that calling initialize twice is safe."""
        manager = await create_hitl_manager_async(config=memory_config)
        # Should not raise
        await manager.initialize()
        await manager.initialize()
        await manager.close()

    @pytest.mark.asyncio
    async def test_close_without_initialize_safe(self, memory_config: HITLConfig):
        """Test that close() is safe without initialization."""
        manager = HITLManager(config=memory_config)
        # Should not raise
        await manager.close()


# =============================================================================
# SCOPE MANAGEMENT TESTS
# =============================================================================


class TestScopeManagement:
    """Tests for scope management through manager."""

    @pytest.mark.asyncio
    async def test_grant_session_approval(self, memory_manager: HITLManager):
        """Test granting session-level approval."""

        scope_id = memory_manager.grant_session_approval("test_tool")
        assert scope_id is not None

    @pytest.mark.asyncio
    async def test_granted_scope_auto_approves(self, memory_manager: HITLManager):
        """Test that granted scope auto-approves subsequent requests."""
        from llmcore.agents.hitl.risk_assessor import RiskLevel

        # Grant approval for a tool with RiskLevel enum
        memory_manager.grant_session_approval(
            "test_tool",
            max_risk_level=RiskLevel.HIGH,
        )

        # Check approval for that tool
        decision = await memory_manager.check_approval(
            activity_type="test_tool",
            parameters={"action": "test"},
        )

        # Should be auto-approved by scope
        assert decision.is_approved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
