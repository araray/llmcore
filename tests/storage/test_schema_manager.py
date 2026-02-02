# tests/storage/test_schema_manager.py
"""
Tests for the Schema Manager (Phase 1 - PRIMORDIUM).

Tests cover:
- Schema version tracking
- Idempotent migrations
- PostgreSQL and SQLite backend support
- Concurrent startup safety (advisory locks)
"""

import sys
from pathlib import Path

# Add storage module to path for direct imports (avoids llmcore import chain issues)
_storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
if str(_storage_path) not in sys.path:
    sys.path.insert(0, str(_storage_path))

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from schema_manager import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    SCHEMA_TABLE_NAME,
    PostgresSchemaManager,
    SchemaBackend,
    SchemaMigration,
    SchemaVersion,
    SqliteSchemaManager,
    create_schema_manager,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_postgres_pool():
    """Create a mock PostgreSQL connection pool."""
    mock_cursor = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.fetchone = AsyncMock(return_value=None)

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.cursor = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_cursor), __aexit__=AsyncMock(return_value=None)
        )
    )
    mock_conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None), __aexit__=AsyncMock(return_value=None)
        )
    )

    mock_pool = AsyncMock()
    mock_pool.connection = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock(return_value=None)
        )
    )
    mock_pool.getconn = AsyncMock(return_value=mock_conn)
    mock_pool.putconn = AsyncMock()

    return mock_pool


@pytest.fixture
def mock_sqlite_conn():
    """Create a mock SQLite connection."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()

    # Mock fetchall to return empty list by default
    mock_cursor = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.description = [("version",), ("applied_at",), ("description",), ("checksum",)]

    mock_conn.execute = AsyncMock(return_value=mock_cursor)

    return mock_conn


# =============================================================================
# UNIT TESTS - SCHEMA VERSION
# =============================================================================


class TestSchemaVersion:
    """Tests for SchemaVersion dataclass."""

    def test_schema_version_creation(self):
        """Test creating a SchemaVersion instance."""
        now = datetime.now(timezone.utc)
        version = SchemaVersion(
            version=1, applied_at=now, description="Initial schema", checksum="abc123"
        )

        assert version.version == 1
        assert version.applied_at == now
        assert version.description == "Initial schema"
        assert version.checksum == "abc123"

    def test_schema_version_without_checksum(self):
        """Test SchemaVersion with default None checksum."""
        version = SchemaVersion(
            version=2, applied_at=datetime.now(timezone.utc), description="Add indexes"
        )

        assert version.checksum is None


class TestSchemaMigration:
    """Tests for SchemaMigration dataclass."""

    def test_migration_definition(self):
        """Test creating a SchemaMigration instance."""
        migration = SchemaMigration(
            from_version=0,
            to_version=1,
            description="Create base tables",
            sql_postgres="CREATE TABLE sessions ...",
            sql_sqlite="CREATE TABLE sessions ...",
        )

        assert migration.from_version == 0
        assert migration.to_version == 1
        assert "CREATE TABLE" in migration.sql_postgres
        assert "CREATE TABLE" in migration.sql_sqlite

    def test_migrations_are_sequential(self):
        """Test that defined migrations are sequential."""
        for i, migration in enumerate(MIGRATIONS):
            # Each migration should go to the next version
            assert migration.to_version == migration.from_version + 1, (
                f"Migration {i} is not sequential: {migration.from_version} -> {migration.to_version}"
            )

    def test_migrations_cover_all_versions(self):
        """Test that migrations cover from 0 to CURRENT_SCHEMA_VERSION."""
        covered_versions = set()
        for migration in MIGRATIONS:
            covered_versions.add(migration.to_version)

        expected_versions = set(range(1, CURRENT_SCHEMA_VERSION + 1))
        assert covered_versions == expected_versions, (
            f"Missing migrations for versions: {expected_versions - covered_versions}"
        )


# =============================================================================
# UNIT TESTS - SCHEMA BACKEND ENUM
# =============================================================================


class TestSchemaBackend:
    """Tests for SchemaBackend enum."""

    def test_backend_values(self):
        """Test that backend enum has expected values."""
        assert SchemaBackend.POSTGRES.value == "postgres"
        assert SchemaBackend.SQLITE.value == "sqlite"

    def test_backend_string_conversion(self):
        """Test that backend can be used as string."""
        assert str(SchemaBackend.POSTGRES) == "SchemaBackend.POSTGRES"
        assert SchemaBackend.POSTGRES == "postgres"


# =============================================================================
# UNIT TESTS - FACTORY FUNCTION
# =============================================================================


class TestCreateSchemaManager:
    """Tests for create_schema_manager factory function."""

    def test_create_postgres_manager(self, mock_postgres_pool):
        """Test creating PostgreSQL schema manager."""
        manager = create_schema_manager(SchemaBackend.POSTGRES, mock_postgres_pool)

        assert isinstance(manager, PostgresSchemaManager)
        assert manager.backend == SchemaBackend.POSTGRES

    def test_create_sqlite_manager(self, mock_sqlite_conn):
        """Test creating SQLite schema manager."""
        manager = create_schema_manager(SchemaBackend.SQLITE, mock_sqlite_conn)

        assert isinstance(manager, SqliteSchemaManager)
        assert manager.backend == SchemaBackend.SQLITE

    def test_create_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            create_schema_manager("invalid", None)


# =============================================================================
# UNIT TESTS - POSTGRES SCHEMA MANAGER
# =============================================================================


class TestPostgresSchemaManager:
    """Tests for PostgreSQL schema manager."""

    @pytest.mark.asyncio
    async def test_get_current_version_no_schema(self, mock_postgres_pool):
        """Test getting version when schema table is empty."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        # Mock empty result
        with patch.object(manager, "_execute_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [{"version": None}]

            with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                version = await manager.get_current_version()

        assert version == 0

    @pytest.mark.asyncio
    async def test_get_current_version_with_versions(self, mock_postgres_pool):
        """Test getting version when migrations have been applied."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        with patch.object(manager, "_execute_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [{"version": 3}]

            with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                version = await manager.get_current_version()

        assert version == 3

    @pytest.mark.asyncio
    async def test_ensure_schema_already_current(self, mock_postgres_pool):
        """Test ensure_schema when already at target version."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        with patch.object(manager, "get_current_version", new_callable=AsyncMock) as mock_version:
            mock_version.return_value = CURRENT_SCHEMA_VERSION

            with patch.object(manager, "_acquire_lock", new_callable=AsyncMock, return_value=True):
                with patch.object(manager, "_release_lock", new_callable=AsyncMock):
                    with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                        result = await manager.ensure_schema()

        assert result == CURRENT_SCHEMA_VERSION

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, mock_postgres_pool):
        """Test successful advisory lock acquisition."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        # Mock successful lock
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone = AsyncMock(return_value=(True,))
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_postgres_pool.getconn = AsyncMock(return_value=mock_conn)

        result = await manager._acquire_lock()

        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_failure(self, mock_postgres_pool):
        """Test failed advisory lock acquisition."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        # Mock failed lock (another process holds it)
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone = AsyncMock(return_value=(False,))
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_postgres_pool.getconn = AsyncMock(return_value=mock_conn)

        result = await manager._acquire_lock()

        assert result is False


# =============================================================================
# UNIT TESTS - SQLITE SCHEMA MANAGER
# =============================================================================


class TestSqliteSchemaManager:
    """Tests for SQLite schema manager."""

    @pytest.mark.asyncio
    async def test_get_current_version_no_schema(self, mock_sqlite_conn):
        """Test getting version when schema table is empty."""
        manager = SqliteSchemaManager(mock_sqlite_conn)

        with patch.object(manager, "_execute_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [{"version": None}]

            with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                version = await manager.get_current_version()

        assert version == 0

    @pytest.mark.asyncio
    async def test_acquire_lock_always_succeeds(self, mock_sqlite_conn):
        """Test that SQLite lock always succeeds (file-based locking)."""
        manager = SqliteSchemaManager(mock_sqlite_conn)

        result = await manager._acquire_lock()

        assert result is True

    @pytest.mark.asyncio
    async def test_release_lock_is_noop(self, mock_sqlite_conn):
        """Test that SQLite release lock is a no-op."""
        manager = SqliteSchemaManager(mock_sqlite_conn)

        # Should not raise
        await manager._release_lock()

    @pytest.mark.asyncio
    async def test_execute_ddl_splits_statements(self, mock_sqlite_conn):
        """Test that DDL execution splits multiple statements."""
        manager = SqliteSchemaManager(mock_sqlite_conn)

        ddl = "CREATE TABLE a (id INT); CREATE TABLE b (id INT);"

        await manager._execute_ddl(ddl)

        # Should have called execute twice (once per statement) plus commit
        assert mock_sqlite_conn.execute.call_count == 2
        mock_sqlite_conn.commit.assert_called_once()


# =============================================================================
# UNIT TESTS - VERSION HISTORY
# =============================================================================


class TestVersionHistory:
    """Tests for version history retrieval."""

    @pytest.mark.asyncio
    async def test_get_version_history_empty(self, mock_postgres_pool):
        """Test getting history when no versions applied."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        with patch.object(manager, "_execute_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []

            with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                history = await manager.get_version_history()

        assert history == []

    @pytest.mark.asyncio
    async def test_get_version_history_with_versions(self, mock_postgres_pool):
        """Test getting history with applied versions."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        now = datetime.now(timezone.utc)
        with patch.object(manager, "_execute_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {"version": 1, "applied_at": now, "description": "Initial", "checksum": None},
                {"version": 2, "applied_at": now, "description": "Add indexes", "checksum": "abc"},
            ]

            with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                history = await manager.get_version_history()

        assert len(history) == 2
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[1].checksum == "abc"


# =============================================================================
# INTEGRATION TESTS (with mocked database)
# =============================================================================


class TestSchemaManagerIntegration:
    """Integration tests for schema manager workflow."""

    @pytest.mark.asyncio
    async def test_full_migration_sequence(self, mock_postgres_pool):
        """Test applying migrations from v0 to current."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        applied_versions = []

        async def mock_execute_ddl(sql):
            # Track that DDL was executed
            pass

        async def mock_record_version(version, description):
            applied_versions.append(version)

        with patch.object(manager, "_execute_ddl", side_effect=mock_execute_ddl):
            with patch.object(manager, "_record_version", side_effect=mock_record_version):
                with patch.object(
                    manager, "_acquire_lock", new_callable=AsyncMock, return_value=True
                ):
                    with patch.object(manager, "_release_lock", new_callable=AsyncMock):
                        with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                            with patch.object(
                                manager,
                                "get_current_version",
                                new_callable=AsyncMock,
                                return_value=0,
                            ):
                                result = await manager.ensure_schema(
                                    target_version=CURRENT_SCHEMA_VERSION
                                )

        # Should have applied all migrations
        assert len(applied_versions) == CURRENT_SCHEMA_VERSION
        assert applied_versions == list(range(1, CURRENT_SCHEMA_VERSION + 1))

    @pytest.mark.asyncio
    async def test_partial_migration(self, mock_postgres_pool):
        """Test applying migrations from existing version."""
        manager = PostgresSchemaManager(mock_postgres_pool)

        applied_versions = []

        async def mock_record_version(version, description):
            applied_versions.append(version)

        with patch.object(manager, "_execute_ddl", new_callable=AsyncMock):
            with patch.object(manager, "_record_version", side_effect=mock_record_version):
                with patch.object(
                    manager, "_acquire_lock", new_callable=AsyncMock, return_value=True
                ):
                    with patch.object(manager, "_release_lock", new_callable=AsyncMock):
                        with patch.object(manager, "_ensure_schema_table", new_callable=AsyncMock):
                            # Start from version 2
                            with patch.object(
                                manager,
                                "get_current_version",
                                new_callable=AsyncMock,
                                return_value=2,
                            ):
                                result = await manager.ensure_schema(
                                    target_version=CURRENT_SCHEMA_VERSION
                                )

        # Should only apply migrations from v2 to current
        expected_versions = list(range(3, CURRENT_SCHEMA_VERSION + 1))
        assert applied_versions == expected_versions


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestSchemaConstants:
    """Tests for schema-related constants."""

    def test_current_schema_version_positive(self):
        """Test that current schema version is positive."""
        assert CURRENT_SCHEMA_VERSION > 0

    def test_schema_table_name(self):
        """Test schema table name is set correctly."""
        assert SCHEMA_TABLE_NAME == "_llmcore_schema"

    def test_migrations_not_empty(self):
        """Test that migrations list is not empty."""
        assert len(MIGRATIONS) > 0

    def test_migrations_have_both_sql(self):
        """Test that all migrations have both Postgres and SQLite SQL."""
        for migration in MIGRATIONS:
            assert migration.sql_postgres, f"Migration {migration.to_version} missing Postgres SQL"
            assert migration.sql_sqlite, f"Migration {migration.to_version} missing SQLite SQL"
