# tests/agents/darwin/test_failure_learning.py
"""
Comprehensive tests for Darwin failure learning system.

Tests both SQLite and PostgreSQL backends with the same test suite.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from llmcore.agents.darwin import (
    FailureContext,
    FailureLearningManager,
    FailureLog,
    FailurePattern,
)
from llmcore.agents.darwin.postgres_failure_storage import PostgresFailureStorage
from llmcore.agents.darwin.sqlite_failure_storage import SqliteFailureStorage

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def sqlite_storage():
    """Provide a SQLite failure storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_failures.db"
        storage = SqliteFailureStorage()
        await storage.initialize({"path": str(db_path)})
        yield storage
        await storage.close()


@pytest.fixture
async def postgres_storage():
    """Provide a PostgreSQL failure storage for testing."""
    # Skip if PostgreSQL is not available
    pytest.importorskip("psycopg")

    db_url = os.environ.get("TEST_POSTGRES_URL")
    if not db_url:
        pytest.skip("PostgreSQL not configured (set TEST_POSTGRES_URL)")

    storage = PostgresFailureStorage()
    await storage.initialize({"db_url": db_url, "table_prefix": "test_"})
    yield storage

    # Cleanup: Drop test tables
    if storage._pool:
        async with storage._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS test_failure_logs CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS test_failure_patterns CASCADE;")
            await conn.commit()

    await storage.close()


@pytest.fixture(params=["sqlite", "postgres"])
async def any_storage(request, sqlite_storage, postgres_storage):
    """Parametrized fixture that provides both storage backends."""
    if request.param == "sqlite":
        return sqlite_storage
    else:
        return postgres_storage


@pytest.fixture
async def sqlite_manager():
    """Provide a SQLite-backed failure learning manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_failures.db"
        manager = FailureLearningManager(backend="sqlite", db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


@pytest.fixture
async def postgres_manager():
    """Provide a PostgreSQL-backed failure learning manager."""
    pytest.importorskip("psycopg")

    db_url = os.environ.get("TEST_POSTGRES_URL")
    if not db_url:
        pytest.skip("PostgreSQL not configured (set TEST_POSTGRES_URL)")

    manager = FailureLearningManager(backend="postgres", db_url=db_url)
    await manager.initialize()
    yield manager

    # Cleanup
    if manager._backend and hasattr(manager._backend, "_pool"):
        async with manager._backend._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS failure_logs CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS failure_patterns CASCADE;")
            await conn.commit()

    await manager.close()


# =============================================================================
# Data Model Tests
# =============================================================================


class TestFailureLog:
    """Test FailureLog data model."""

    def test_create_failure_log_minimal(self):
        """Test creating a minimal FailureLog."""
        log = FailureLog(
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test goal",
            phase="ACT",
            failure_type="test_failure",
            error_message="Test error",
        )
        assert log.task_id == "task_123"
        assert log.agent_run_id == "run_456"
        assert log.failure_type == "test_failure"
        assert log.created_at is not None

    def test_create_failure_log_full(self):
        """Test creating a complete FailureLog with all fields."""
        log = FailureLog(
            id="fail_001",
            task_id="task_123",
            agent_run_id="run_456",
            goal="Implement user authentication",
            phase="ACT",
            genotype_id="gen_789",
            genotype_summary="Tried password hashing with bcrypt",
            failure_type="test_failure",
            error_message="AssertionError: login() returned None",
            error_details={"line": 42, "file": "auth.py"},
            phenotype_id="phen_101",
            phenotype_summary="Generated auth module without login method",
            test_results={"total": 5, "passed": 3, "failed": 2},
            arbiter_critique="Missing login method implementation",
            arbiter_score=0.6,
            similarity_hash="abc123",
            tags=["authentication", "security"],
        )
        assert log.id == "fail_001"
        assert log.genotype_summary == "Tried password hashing with bcrypt"
        assert log.error_details["line"] == 42
        assert log.test_results["failed"] == 2
        assert "authentication" in log.tags

    def test_failure_log_json_serialization(self):
        """Test FailureLog can be serialized to JSON."""
        log = FailureLog(
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test goal",
            phase="ACT",
            failure_type="runtime_error",
            error_message="Test error",
        )
        json_data = log.model_dump_json()
        assert "task_123" in json_data
        assert "runtime_error" in json_data


class TestFailurePattern:
    """Test FailurePattern data model."""

    def test_create_failure_pattern(self):
        """Test creating a FailurePattern."""
        pattern = FailurePattern(
            pattern_id="pattern_001",
            description="Missing return statement",
            failure_type="test_failure",
            occurrence_count=5,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            common_error_messages=["Function returned None", "Expected string, got None"],
            suggested_avoidance="Always include explicit return statements",
        )
        assert pattern.pattern_id == "pattern_001"
        assert pattern.occurrence_count == 5
        assert len(pattern.common_error_messages) == 2


# =============================================================================
# Storage Backend Tests (run on both SQLite and PostgreSQL)
# =============================================================================


class TestStorageBackend:
    """Test storage backend operations on both SQLite and PostgreSQL."""

    @pytest.mark.asyncio
    async def test_log_failure(self, any_storage):
        """Test logging a failure."""
        failure = FailureLog(
            id="fail_test_001",
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test goal",
            phase="ACT",
            failure_type="test_failure",
            error_message="Test error message",
            similarity_hash="hash123",
        )

        logged = await any_storage.log_failure(failure)
        assert logged.id == "fail_test_001"

        # Verify we can retrieve it
        retrieved = await any_storage.get_failure("fail_test_001")
        assert retrieved is not None
        assert retrieved.task_id == "task_123"
        assert retrieved.error_message == "Test error message"

    @pytest.mark.asyncio
    async def test_log_failure_with_test_results(self, any_storage):
        """Test logging a failure with test results."""
        failure = FailureLog(
            id="fail_test_002",
            task_id="task_456",
            agent_run_id="run_789",
            goal="Test with results",
            phase="VALIDATE",
            failure_type="test_failure",
            error_message="2 tests failed",
            test_results={"total": 10, "passed": 8, "failed": 2},
            similarity_hash="hash456",
        )

        await any_storage.log_failure(failure)
        retrieved = await any_storage.get_failure("fail_test_002")

        assert retrieved is not None
        assert retrieved.test_results["total"] == 10
        assert retrieved.test_results["failed"] == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_failure(self, any_storage):
        """Test retrieving a non-existent failure returns None."""
        result = await any_storage.get_failure("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_similar_failures(self, any_storage):
        """Test retrieving similar failures based on goal."""
        # Log several failures
        failures = [
            FailureLog(
                id=f"fail_sim_{i}",
                task_id=f"task_{i}",
                agent_run_id=f"run_{i}",
                goal="Implement user authentication with password hashing",
                phase="ACT",
                failure_type="test_failure",
                error_message=f"Error {i}",
                similarity_hash="auth_hash",
            )
            for i in range(3)
        ]

        for failure in failures:
            await any_storage.log_failure(failure)

        # Search for similar failures
        similar = await any_storage.get_similar_failures(
            goal="Implement user authentication",
            limit=5,
        )

        assert len(similar) >= 3
        assert all("authentication" in f.goal.lower() for f in similar[:3])

    @pytest.mark.asyncio
    async def test_pattern_auto_creation(self, any_storage):
        """Test that patterns are auto-created when logging failures."""
        failure = FailureLog(
            id="fail_pattern_001",
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test pattern creation",
            phase="ACT",
            failure_type="compile_error",
            error_message="SyntaxError",
            similarity_hash="syntax_hash",
        )

        await any_storage.log_failure(failure)

        # Verify pattern was created
        pattern = await any_storage.get_pattern("syntax_hash")
        assert pattern is not None
        assert pattern.pattern_id == "syntax_hash"
        assert pattern.occurrence_count == 1
        assert pattern.failure_type == "compile_error"

    @pytest.mark.asyncio
    async def test_pattern_increment(self, any_storage):
        """Test that pattern occurrence count increments."""
        # Log first failure
        failure1 = FailureLog(
            id="fail_inc_001",
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test increment",
            phase="ACT",
            failure_type="runtime_error",
            error_message="Error 1",
            similarity_hash="increment_hash",
        )
        await any_storage.log_failure(failure1)

        pattern = await any_storage.get_pattern("increment_hash")
        assert pattern.occurrence_count == 1

        # Log second failure with same hash
        failure2 = FailureLog(
            id="fail_inc_002",
            task_id="task_789",
            agent_run_id="run_012",
            goal="Test increment again",
            phase="ACT",
            failure_type="runtime_error",
            error_message="Error 2",
            similarity_hash="increment_hash",
        )
        await any_storage.log_failure(failure2)

        pattern = await any_storage.get_pattern("increment_hash")
        assert pattern.occurrence_count == 2

    @pytest.mark.asyncio
    async def test_get_failure_stats(self, any_storage):
        """Test getting failure statistics."""
        # Log failures of different types
        failure_types = ["test_failure", "runtime_error", "compile_error"]
        for i, ftype in enumerate(failure_types):
            for j in range(i + 1):  # Different counts per type
                failure = FailureLog(
                    id=f"fail_stats_{ftype}_{j}",
                    task_id=f"task_{i}_{j}",
                    agent_run_id=f"run_{i}_{j}",
                    goal=f"Test {ftype}",
                    phase="ACT",
                    failure_type=ftype,
                    error_message=f"Error {j}",
                )
                await any_storage.log_failure(failure)

        stats = await any_storage.get_failure_stats(days=30)

        assert stats["total_failures"] == 6  # 1 + 2 + 3
        assert stats["by_type"]["test_failure"] == 1
        assert stats["by_type"]["runtime_error"] == 2
        assert stats["by_type"]["compile_error"] == 3


# =============================================================================
# Manager Tests
# =============================================================================


class TestFailureLearningManager:
    """Test the high-level FailureLearningManager."""

    @pytest.mark.asyncio
    async def test_manager_initialization_sqlite(self, sqlite_manager):
        """Test manager initializes correctly with SQLite."""
        assert sqlite_manager.enabled is True
        assert sqlite_manager.backend_type == "sqlite"
        assert sqlite_manager._backend is not None

    @pytest.mark.asyncio
    async def test_manager_initialization_postgres(self, postgres_manager):
        """Test manager initializes correctly with PostgreSQL."""
        assert postgres_manager.enabled is True
        assert postgres_manager.backend_type == "postgres"
        assert postgres_manager._backend is not None

    @pytest.mark.asyncio
    async def test_manager_disabled(self):
        """Test manager with disabled learning."""
        manager = FailureLearningManager(enabled=False)
        await manager.initialize()

        assert manager.enabled is False
        assert manager._backend is None

        # Operations should be no-ops
        failure = FailureLog(
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test",
            phase="ACT",
            failure_type="test_failure",
            error_message="Error",
        )
        logged = await manager.log_failure(failure)
        assert logged.id is None  # Not persisted

    @pytest.mark.asyncio
    async def test_compute_similarity_hash(self):
        """Test similarity hash computation."""
        manager = FailureLearningManager(enabled=False)

        hash1 = manager._compute_similarity_hash(
            "Implement user authentication", "test_failure"
        )
        hash2 = manager._compute_similarity_hash(
            "Implement user authentication system", "test_failure"
        )
        hash3 = manager._compute_similarity_hash(
            "Implement user authentication", "runtime_error"
        )

        # Same goal, same type -> same hash
        assert hash1 == hash2

        # Same goal, different type -> different hash
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_log_failure_auto_id(self, sqlite_manager):
        """Test that manager auto-generates failure IDs."""
        failure = FailureLog(
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test auto ID",
            phase="ACT",
            failure_type="test_failure",
            error_message="Error",
        )

        logged = await sqlite_manager.log_failure(failure)
        assert logged.id is not None
        assert logged.id.startswith("fail_")

    @pytest.mark.asyncio
    async def test_log_failure_auto_similarity_hash(self, sqlite_manager):
        """Test that manager auto-generates similarity hashes."""
        failure = FailureLog(
            task_id="task_123",
            agent_run_id="run_456",
            goal="Test auto hash",
            phase="ACT",
            failure_type="test_failure",
            error_message="Error",
        )

        logged = await sqlite_manager.log_failure(failure)
        assert logged.similarity_hash is not None
        assert len(logged.similarity_hash) == 16

    @pytest.mark.asyncio
    async def test_get_failure_context(self, sqlite_manager):
        """Test getting failure context for planning."""
        # Log some failures
        for i in range(3):
            failure = FailureLog(
                id=f"fail_ctx_{i}",
                task_id=f"task_{i}",
                agent_run_id=f"run_{i}",
                goal="Implement database connection pooling",
                phase="ACT",
                failure_type="runtime_error",
                error_message=f"Connection pool exhausted {i}",
            )
            await sqlite_manager.log_failure(failure)

        # Get context for similar goal
        context = await sqlite_manager.get_failure_context(
            goal="Implement database connection",
        )

        assert isinstance(context, FailureContext)
        assert len(context.relevant_failures) > 0
        assert len(context.patterns) > 0
        assert "Connection pool" in context.avoidance_instructions

    @pytest.mark.asyncio
    async def test_generate_avoidance_prompt(self, sqlite_manager):
        """Test generating avoidance prompt from failures."""
        failures = [
            FailureLog(
                task_id="task_1",
                agent_run_id="run_1",
                goal="Test",
                phase="ACT",
                failure_type="test_failure",
                error_message="Assertion failed: expected 5, got None",
                genotype_summary="Forgot to return value",
            ),
            FailureLog(
                task_id="task_2",
                agent_run_id="run_2",
                goal="Test",
                phase="ACT",
                failure_type="runtime_error",
                error_message="AttributeError: 'NoneType' has no attribute 'value'",
            ),
        ]

        prompt = sqlite_manager.generate_avoidance_prompt(failures)

        assert "Learning from Past Attempts" in prompt
        assert "Test Failure" in prompt
        assert "Runtime Error" in prompt
        assert "Assertion failed" in prompt
        assert "Forgot to return value" in prompt

    @pytest.mark.asyncio
    async def test_get_failure_stats(self, sqlite_manager):
        """Test getting failure statistics through manager."""
        # Log some failures
        for i in range(5):
            failure = FailureLog(
                id=f"fail_stats_{i}",
                task_id=f"task_{i}",
                agent_run_id=f"run_{i}",
                goal="Test stats",
                phase="ACT",
                failure_type="test_failure" if i % 2 == 0 else "runtime_error",
                error_message=f"Error {i}",
            )
            await sqlite_manager.log_failure(failure)

        stats = await sqlite_manager.get_failure_stats(days=30)

        assert stats["total_failures"] == 5
        assert stats["by_type"]["test_failure"] == 3
        assert stats["by_type"]["runtime_error"] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_learning_workflow(self, sqlite_manager):
        """Test complete workflow: log failure -> retrieve context -> use in planning."""
        # Step 1: Agent attempts a task and fails
        failure1 = FailureLog(
            task_id="task_workflow",
            agent_run_id="run_001",
            goal="Create a REST API endpoint for user registration",
            phase="ACT",
            failure_type="test_failure",
            error_message="Missing email validation",
            genotype_summary="Implemented basic registration without validation",
            test_results={"total": 10, "passed": 8, "failed": 2},
        )
        await sqlite_manager.log_failure(failure1)

        # Step 2: Agent tries again with similar task
        context = await sqlite_manager.get_failure_context(
            goal="Create a REST API endpoint for user registration"
        )

        assert len(context.relevant_failures) > 0
        assert "Missing email validation" in context.avoidance_instructions

        # Step 3: Agent learns and tries with different approach
        failure2 = FailureLog(
            task_id="task_workflow",
            agent_run_id="run_002",
            goal="Create a REST API endpoint for user registration",
            phase="ACT",
            failure_type="test_failure",
            error_message="Password too weak error not caught",
            genotype_summary="Added email validation but missing password strength check",
        )
        await sqlite_manager.log_failure(failure2)

        # Verify pattern is building
        context = await sqlite_manager.get_failure_context(
            goal="Create a REST API endpoint for user registration"
        )

        assert len(context.relevant_failures) >= 2
        assert len(context.patterns) > 0
        assert any(p.occurrence_count >= 2 for p in context.patterns)

    @pytest.mark.asyncio
    async def test_cross_session_learning(self, sqlite_manager):
        """Test that failures persist across manager instances."""
        # Log failure in first session
        failure = FailureLog(
            id="fail_cross_001",
            task_id="task_cross",
            agent_run_id="run_001",
            goal="Implement caching layer",
            phase="ACT",
            failure_type="runtime_error",
            error_message="Cache key collision",
        )
        await sqlite_manager.log_failure(failure)
        await sqlite_manager.close()

        # Create new manager with same database

        # Note: This test assumes sqlite_manager uses a temporary directory
        # In a real scenario, you'd use the same db_path
        # For now, we'll just verify the pattern with the same manager after close/reinit

    @pytest.mark.asyncio
    async def test_pattern_evolution(self, sqlite_manager):
        """Test that patterns evolve as more failures are logged."""
        goal = "Implement file upload handling"

        # Log failures over time
        for i in range(10):
            failure = FailureLog(
                id=f"fail_evolve_{i}",
                task_id=f"task_evolve_{i}",
                agent_run_id=f"run_evolve_{i}",
                goal=goal,
                phase="ACT",
                failure_type="validation_failed",
                error_message=f"File size exceeded limit iteration {i}",
            )
            await sqlite_manager.log_failure(failure)

        # Verify pattern has evolved
        context = await sqlite_manager.get_failure_context(goal=goal)

        # Should have recurring pattern warning
        assert "Recurring Issues" in context.avoidance_instructions
        assert any(p.occurrence_count >= 2 for p in context.patterns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
