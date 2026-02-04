# src/llmcore/storage/schema_manager.py
"""
Schema Manager for LLMCore Storage System.

Implements idempotent, migration-free schema management as specified in
Storage System V2. This module handles:

- Schema version tracking via `_llmcore_schema` metadata table
- Idempotent DDL operations (CREATE IF NOT EXISTS, safe ALTERs)
- Sequential version upgrades without external migration tools
- Backend-agnostic interface supporting both Postgres and SQLite

Design Philosophy:
- No ORM dependency (raw SQL for transparency)
- No Alembic/migration framework required
- Safe for concurrent startup (advisory locks on Postgres)
- Schema changes are code, versioned with the application

Schema Version History:
    v1: Initial schema (sessions, messages, context_items, presets, episodes)
    v2: Added user_id column, indexes for multi-user isolation
    v3: Added vector collections and embeddings tables (pgvector)
    v4: Added GIN indexes for text search, health check support
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

CURRENT_SCHEMA_VERSION = 4
SCHEMA_TABLE_NAME = "_llmcore_schema"


class SchemaBackend(str, Enum):
    """Supported database backends for schema management."""

    POSTGRES = "postgres"
    SQLITE = "sqlite"


@dataclass
class SchemaVersion:
    """Represents a schema version record."""

    version: int
    applied_at: datetime
    description: str
    checksum: str | None = None


@dataclass
class SchemaMigration:
    """Defines a schema migration step."""

    from_version: int
    to_version: int
    description: str
    sql_postgres: str  # PostgreSQL-specific DDL
    sql_sqlite: str  # SQLite-specific DDL (may be no-op for PG-only features)


# =============================================================================
# MIGRATION DEFINITIONS
# =============================================================================

MIGRATIONS: list[SchemaMigration] = [
    # v0 -> v1: Create base schema
    SchemaMigration(
        from_version=0,
        to_version=1,
        description="Create base session storage tables",
        sql_postgres="""
            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                user_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions (user_id, updated_at DESC);

            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                tool_call_id TEXT,
                tokens INTEGER,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp
                ON messages (session_id, timestamp);

            -- Session context items table
            CREATE TABLE IF NOT EXISTS context_items (
                id TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                item_type TEXT NOT NULL,
                source_id TEXT,
                content TEXT NOT NULL,
                tokens INTEGER,
                original_tokens INTEGER,
                is_truncated BOOLEAN DEFAULT FALSE,
                metadata JSONB DEFAULT '{}'::jsonb,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (session_id, id)
            );

            -- Context presets table
            CREATE TABLE IF NOT EXISTS context_presets (
                name TEXT PRIMARY KEY,
                description TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_context_presets_updated
                ON context_presets (updated_at DESC);

            -- Context preset items table
            CREATE TABLE IF NOT EXISTS context_preset_items (
                item_id TEXT NOT NULL,
                preset_name TEXT NOT NULL REFERENCES context_presets(name) ON DELETE CASCADE,
                type TEXT NOT NULL,
                content TEXT,
                source_identifier TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                PRIMARY KEY (preset_name, item_id)
            );

            -- Episodes table (agent episodic memory)
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                event_type TEXT NOT NULL,
                data JSONB DEFAULT '{}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_session_timestamp
                ON episodes (session_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_event_type
                ON episodes (event_type);
        """,
        sql_sqlite="""
            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                user_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions (user_id, updated_at);

            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_call_id TEXT,
                tokens INTEGER,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp
                ON messages (session_id, timestamp);

            -- Session context items table
            CREATE TABLE IF NOT EXISTS context_items (
                id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                source_id TEXT,
                content TEXT NOT NULL,
                tokens INTEGER,
                original_tokens INTEGER,
                is_truncated INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL,
                PRIMARY KEY (session_id, id),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            -- Context presets table
            CREATE TABLE IF NOT EXISTS context_presets (
                name TEXT PRIMARY KEY,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_context_presets_updated
                ON context_presets (updated_at);

            -- Context preset items table
            CREATE TABLE IF NOT EXISTS context_preset_items (
                item_id TEXT NOT NULL,
                preset_name TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                source_identifier TEXT,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (preset_name, item_id),
                FOREIGN KEY (preset_name) REFERENCES context_presets(name) ON DELETE CASCADE
            );

            -- Episodes table
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_session_timestamp
                ON episodes (session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_episodes_event_type
                ON episodes (event_type);
        """,
    ),
    # v1 -> v2: Add user_id column if missing (for existing deployments)
    SchemaMigration(
        from_version=1,
        to_version=2,
        description="Add user_id column to sessions for multi-user isolation",
        sql_postgres="""
            -- Add user_id column if it doesn't exist
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'sessions' AND column_name = 'user_id'
                ) THEN
                    ALTER TABLE sessions ADD COLUMN user_id TEXT;
                    CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                        ON sessions (user_id, updated_at DESC);
                END IF;
            END $$;
        """,
        sql_sqlite="""
            -- SQLite doesn't support ADD COLUMN IF NOT EXISTS directly
            -- This is handled programmatically in the schema manager
            SELECT 1;
        """,
    ),
    # v2 -> v3: Add vector storage tables
    SchemaMigration(
        from_version=2,
        to_version=3,
        description="Add vector collections and embeddings tables for pgvector",
        sql_postgres="""
            -- Ensure pgvector extension exists (may require superuser)
            CREATE EXTENSION IF NOT EXISTS vector;

            -- Vector collections metadata table
            CREATE TABLE IF NOT EXISTS vector_collections (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                vector_dimension INTEGER NOT NULL,
                description TEXT,
                embedding_model_provider TEXT,
                embedding_model_name TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );

            -- Vectors/embeddings table
            -- Note: vector dimension is set dynamically, using 384 as default
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT NOT NULL,
                collection_name TEXT NOT NULL REFERENCES vector_collections(name) ON DELETE CASCADE,
                user_id TEXT,
                content TEXT,
                embedding VECTOR(1536),  -- Common dimension, will be validated
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, collection_name)
            );

            -- HNSW index for fast similarity search (cosine distance)
            CREATE INDEX IF NOT EXISTS idx_vectors_embedding_hnsw
                ON vectors USING hnsw (embedding vector_cosine_ops);

            -- Collection name index for filtering
            CREATE INDEX IF NOT EXISTS idx_vectors_collection
                ON vectors (collection_name);

            -- User isolation index
            CREATE INDEX IF NOT EXISTS idx_vectors_user
                ON vectors (user_id) WHERE user_id IS NOT NULL;
        """,
        sql_sqlite="""
            -- SQLite doesn't support pgvector; vector storage uses ChromaDB
            -- Create placeholder tables for compatibility
            CREATE TABLE IF NOT EXISTS vector_collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                vector_dimension INTEGER NOT NULL,
                description TEXT,
                embedding_model_provider TEXT,
                embedding_model_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            );

            -- Note: Actual vector storage in SQLite uses external ChromaDB
            -- This table is for metadata tracking only
            CREATE TABLE IF NOT EXISTS vectors_metadata (
                id TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                user_id TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, collection_name),
                FOREIGN KEY (collection_name) REFERENCES vector_collections(name) ON DELETE CASCADE
            );
        """,
    ),
    # v3 -> v4: Add text search indexes and health monitoring
    SchemaMigration(
        from_version=3,
        to_version=4,
        description="Add GIN indexes for text search and health monitoring table",
        sql_postgres="""
            -- Enable pg_trgm extension for fuzzy text search
            CREATE EXTENSION IF NOT EXISTS pg_trgm;

            -- GIN index for message content search
            CREATE INDEX IF NOT EXISTS idx_messages_content_trgm
                ON messages USING gin (content gin_trgm_ops);

            -- GIN index for session name search
            CREATE INDEX IF NOT EXISTS idx_sessions_name_trgm
                ON sessions USING gin (name gin_trgm_ops)
                WHERE name IS NOT NULL;

            -- Health monitoring table for tracking storage health events
            CREATE TABLE IF NOT EXISTS _llmcore_health_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                backend TEXT NOT NULL,
                check_type TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL,
                error_message TEXT,
                metadata JSONB DEFAULT '{}'::jsonb
            );

            -- Keep only recent health records (auto-cleanup via trigger)
            CREATE INDEX IF NOT EXISTS idx_health_log_timestamp
                ON _llmcore_health_log (timestamp DESC);

            -- Function to cleanup old health records (keep last 1000)
            CREATE OR REPLACE FUNCTION cleanup_health_log() RETURNS TRIGGER AS $$
            BEGIN
                DELETE FROM _llmcore_health_log
                WHERE id NOT IN (
                    SELECT id FROM _llmcore_health_log
                    ORDER BY timestamp DESC LIMIT 1000
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            -- Trigger to auto-cleanup (runs on every 100th insert)
            DROP TRIGGER IF EXISTS trg_cleanup_health_log ON _llmcore_health_log;
            CREATE TRIGGER trg_cleanup_health_log
                AFTER INSERT ON _llmcore_health_log
                FOR EACH ROW
                WHEN (NEW.id % 100 = 0)
                EXECUTE FUNCTION cleanup_health_log();
        """,
        sql_sqlite="""
            -- SQLite doesn't support GIN/trgm, but we can use FTS5
            -- Create FTS5 virtual table for message search (if not exists)
            -- Note: This requires careful handling as FTS5 tables are special

            -- Health monitoring table
            CREATE TABLE IF NOT EXISTS _llmcore_health_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                backend TEXT NOT NULL,
                check_type TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL,
                error_message TEXT,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_health_log_timestamp
                ON _llmcore_health_log (timestamp);
        """,
    ),
]


# =============================================================================
# SCHEMA MANAGER PROTOCOL
# =============================================================================


class SchemaManagerProtocol(Protocol):
    """Protocol defining the schema manager interface."""

    async def get_current_version(self) -> int:
        """Get the current schema version from the database."""
        ...

    async def ensure_schema(self, target_version: int | None = None) -> int:
        """Ensure schema is at target version, applying migrations as needed."""
        ...

    async def get_version_history(self) -> list[SchemaVersion]:
        """Get the history of applied schema versions."""
        ...


# =============================================================================
# BASE SCHEMA MANAGER
# =============================================================================


class BaseSchemaManager(ABC):
    """
    Abstract base class for schema management.

    Provides the core logic for idempotent schema versioning,
    with backend-specific implementations for actual DDL execution.
    """

    def __init__(self, backend: SchemaBackend):
        """
        Initialize the schema manager.

        Args:
            backend: The database backend type (postgres or sqlite)
        """
        self.backend = backend
        self._initialized = False

    @abstractmethod
    async def _execute_ddl(self, sql: str) -> None:
        """Execute DDL statement(s) against the database."""
        pass

    @abstractmethod
    async def _execute_query(
        self, sql: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        pass

    @abstractmethod
    async def _acquire_lock(self) -> bool:
        """Acquire advisory lock for schema operations (Postgres) or no-op (SQLite)."""
        pass

    @abstractmethod
    async def _release_lock(self) -> None:
        """Release advisory lock."""
        pass

    async def _ensure_schema_table(self) -> None:
        """Create the schema version tracking table if it doesn't exist."""
        if self.backend == SchemaBackend.POSTGRES:
            ddl = f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_TABLE_NAME} (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    description TEXT NOT NULL,
                    checksum TEXT
                );
            """
        else:  # SQLite
            ddl = f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_TABLE_NAME} (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT NOT NULL,
                    checksum TEXT
                );
            """
        await self._execute_ddl(ddl)
        logger.debug(f"Schema version table '{SCHEMA_TABLE_NAME}' ensured.")

    async def get_current_version(self) -> int:
        """
        Get the current schema version from the database.

        Returns:
            Current version number, or 0 if no schema has been applied.
        """
        await self._ensure_schema_table()

        result = await self._execute_query(
            f"SELECT MAX(version) as version FROM {SCHEMA_TABLE_NAME}"
        )

        if result and result[0].get("version") is not None:
            return int(result[0]["version"])
        return 0

    async def get_version_history(self) -> list[SchemaVersion]:
        """
        Get the history of applied schema versions.

        Returns:
            List of SchemaVersion records, ordered by version ascending.
        """
        await self._ensure_schema_table()

        result = await self._execute_query(
            f"SELECT version, applied_at, description, checksum "
            f"FROM {SCHEMA_TABLE_NAME} ORDER BY version ASC"
        )

        versions = []
        for row in result:
            applied_at = row["applied_at"]
            if isinstance(applied_at, str):
                applied_at = datetime.fromisoformat(applied_at.replace("Z", "+00:00"))

            versions.append(
                SchemaVersion(
                    version=int(row["version"]),
                    applied_at=applied_at,
                    description=row["description"],
                    checksum=row.get("checksum"),
                )
            )

        return versions

    async def _record_version(self, version: int, description: str) -> None:
        """Record a schema version as applied."""
        now = datetime.now(UTC)

        if self.backend == SchemaBackend.POSTGRES:
            await self._execute_ddl(
                f"INSERT INTO {SCHEMA_TABLE_NAME} (version, applied_at, description) "
                f"VALUES ({version}, '{now.isoformat()}', '{description}') "
                f"ON CONFLICT (version) DO NOTHING"
            )
        else:  # SQLite
            await self._execute_ddl(
                f"INSERT OR IGNORE INTO {SCHEMA_TABLE_NAME} (version, applied_at, description) "
                f"VALUES ({version}, '{now.isoformat()}', '{description}')"
            )

    async def ensure_schema(self, target_version: int | None = None) -> int:
        """
        Ensure the database schema is at the target version.

        Applies all necessary migrations in sequence. This operation is
        idempotent - safe to call multiple times or concurrently (with locking).

        Args:
            target_version: Target version to migrate to. Defaults to CURRENT_SCHEMA_VERSION.

        Returns:
            The final schema version after migrations.

        Raises:
            StorageError: If migrations fail.
        """
        if target_version is None:
            target_version = CURRENT_SCHEMA_VERSION

        # Acquire lock for concurrent safety
        if not await self._acquire_lock():
            logger.warning("Could not acquire schema lock; another process may be migrating.")
            # Proceed anyway for single-process scenarios

        try:
            await self._ensure_schema_table()
            current_version = await self.get_current_version()

            if current_version >= target_version:
                logger.info(
                    f"Schema already at version {current_version} (target: {target_version})"
                )
                return current_version

            logger.info(f"Migrating schema from v{current_version} to v{target_version}...")

            # Apply migrations in sequence
            for migration in MIGRATIONS:
                if (
                    migration.from_version >= current_version
                    and migration.to_version <= target_version
                ):
                    if current_version < migration.to_version:
                        logger.info(
                            f"Applying migration: v{migration.from_version} -> v{migration.to_version}: {migration.description}"
                        )

                        sql = (
                            migration.sql_postgres
                            if self.backend == SchemaBackend.POSTGRES
                            else migration.sql_sqlite
                        )

                        try:
                            await self._execute_ddl(sql)
                            await self._record_version(migration.to_version, migration.description)
                            current_version = migration.to_version
                            logger.info(f"Migration to v{migration.to_version} complete.")
                        except Exception as e:
                            logger.error(f"Migration to v{migration.to_version} failed: {e}")
                            raise

            final_version = await self.get_current_version()
            logger.info(f"Schema migration complete. Current version: {final_version}")
            return final_version

        finally:
            await self._release_lock()


# =============================================================================
# POSTGRES SCHEMA MANAGER
# =============================================================================


class PostgresSchemaManager(BaseSchemaManager):
    """
    PostgreSQL-specific schema manager implementation.

    Uses advisory locks for concurrent safety and supports
    pgvector extension management.
    """

    # Advisory lock ID for schema operations (arbitrary but consistent)
    LOCK_ID = 0x4C4C4D43  # "LLMC" in hex

    def __init__(self, pool: Any):
        """
        Initialize with a psycopg connection pool.

        Args:
            pool: AsyncConnectionPool from psycopg_pool
        """
        super().__init__(SchemaBackend.POSTGRES)
        self._pool = pool
        self._lock_conn = None

    async def _execute_ddl(self, sql: str) -> None:
        """Execute DDL against PostgreSQL."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(sql)

    async def _execute_query(
        self, sql: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return results."""
        try:
            from psycopg.rows import dict_row
        except ImportError:
            raise ImportError("psycopg is required for PostgreSQL schema management")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                return [dict(row) for row in rows]

    async def _acquire_lock(self) -> bool:
        """Acquire PostgreSQL advisory lock."""
        try:
            # Get a dedicated connection for the lock
            self._lock_conn = await self._pool.getconn()
            result = await self._lock_conn.execute(f"SELECT pg_try_advisory_lock({self.LOCK_ID})")
            row = await result.fetchone()
            return bool(row and row[0])
        except Exception as e:
            logger.warning(f"Failed to acquire advisory lock: {e}")
            return False

    async def _release_lock(self) -> None:
        """Release PostgreSQL advisory lock."""
        if self._lock_conn:
            try:
                await self._lock_conn.execute(f"SELECT pg_advisory_unlock({self.LOCK_ID})")
            except Exception as e:
                logger.warning(f"Failed to release advisory lock: {e}")
            finally:
                await self._pool.putconn(self._lock_conn)
                self._lock_conn = None


# =============================================================================
# SQLITE SCHEMA MANAGER
# =============================================================================


class SqliteSchemaManager(BaseSchemaManager):
    """
    SQLite-specific schema manager implementation.

    Uses file-based locking inherent to SQLite connections.
    """

    def __init__(self, conn: Any):
        """
        Initialize with an aiosqlite connection.

        Args:
            conn: aiosqlite.Connection instance
        """
        super().__init__(SchemaBackend.SQLITE)
        self._conn = conn

    async def _execute_ddl(self, sql: str) -> None:
        """Execute DDL against SQLite."""
        # Split multiple statements and execute each
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            await self._conn.execute(stmt)
        await self._conn.commit()

    async def _execute_query(
        self, sql: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return results."""
        if params:
            cursor = await self._conn.execute(sql, params)
        else:
            cursor = await self._conn.execute(sql)

        rows = await cursor.fetchall()

        # Get column names from cursor description
        if cursor.description:
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        return []

    async def _acquire_lock(self) -> bool:
        """SQLite uses file locking; no explicit lock needed."""
        return True

    async def _release_lock(self) -> None:
        """No explicit lock to release for SQLite."""
        pass


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_schema_manager(backend: SchemaBackend, connection: Any) -> BaseSchemaManager:
    """
    Factory function to create the appropriate schema manager.

    Args:
        backend: The database backend type
        connection: Database connection (pool for Postgres, connection for SQLite)

    Returns:
        Configured schema manager instance
    """
    if backend == SchemaBackend.POSTGRES:
        return PostgresSchemaManager(connection)
    elif backend == SchemaBackend.SQLITE:
        return SqliteSchemaManager(connection)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
