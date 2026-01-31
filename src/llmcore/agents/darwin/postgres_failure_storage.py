# src/llmcore/agents/darwin/postgres_failure_storage.py
"""
PostgreSQL implementation of failure storage backend.

Provides persistent storage of failure logs and patterns using PostgreSQL database.
Suitable for production deployments with multi-user and high-concurrency requirements.
"""

import json
import logging
from datetime import datetime, timedelta
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

from .failure_storage import BaseFailureStorage, FailureLog, FailurePattern

logger = logging.getLogger(__name__)


class PostgresFailureStorage(BaseFailureStorage):
    """
    PostgreSQL-based storage for failure logs and patterns.

    Uses psycopg with connection pooling for async database operations.
    Suitable for production deployments with high concurrency.

    Schema:
        failure_logs: Individual failure records
        failure_patterns: Aggregated patterns of similar failures
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS failure_logs (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        agent_run_id TEXT NOT NULL,
        goal TEXT NOT NULL,
        phase TEXT NOT NULL,
        genotype_id TEXT,
        genotype_summary TEXT,
        failure_type TEXT NOT NULL,
        error_message TEXT NOT NULL,
        error_details JSONB,
        phenotype_id TEXT,
        phenotype_summary TEXT,
        test_results JSONB,
        arbiter_critique TEXT,
        arbiter_score REAL,
        similarity_hash TEXT,
        tags JSONB,
        created_at TIMESTAMP NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_failure_task ON failure_logs(task_id);
    CREATE INDEX IF NOT EXISTS idx_failure_type ON failure_logs(failure_type);
    CREATE INDEX IF NOT EXISTS idx_failure_similarity ON failure_logs(similarity_hash);
    CREATE INDEX IF NOT EXISTS idx_failure_created ON failure_logs(created_at);
    CREATE INDEX IF NOT EXISTS idx_failure_goal_gin ON failure_logs USING gin(to_tsvector('english', goal));

    CREATE TABLE IF NOT EXISTS failure_patterns (
        pattern_id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        failure_type TEXT NOT NULL,
        occurrence_count INTEGER DEFAULT 1,
        first_seen TIMESTAMP NOT NULL,
        last_seen TIMESTAMP NOT NULL,
        common_error_messages JSONB,
        suggested_avoidance TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pattern_type ON failure_patterns(failure_type);
    CREATE INDEX IF NOT EXISTS idx_pattern_count ON failure_patterns(occurrence_count);
    """

    def __init__(self):
        """Initialize PostgreSQL failure storage."""
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
            raise ValueError("PostgreSQL failure storage 'db_url' not specified in config")

        min_pool_size = config.get("min_pool_size", 2)
        max_pool_size = config.get("max_pool_size", 10)
        self._table_prefix = config.get("table_prefix", "")

        try:
            logger.debug(
                f"Initializing PostgreSQL connection pool for failure storage "
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

            logger.info("PostgreSQL failure storage initialized successfully")

        except psycopg.Error as e:
            if self._pool:
                await self._pool.close()
                self._pool = None
            raise RuntimeError(f"Failed to initialize PostgreSQL failure storage: {e}")

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""
        # Apply table prefix if configured
        schema = self.SCHEMA
        if self._table_prefix:
            schema = schema.replace("failure_logs", f"{self._table_prefix}failure_logs")
            schema = schema.replace("failure_patterns", f"{self._table_prefix}failure_patterns")

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(schema)
            await conn.commit()

    def _get_table_name(self, base_name: str) -> str:
        """Get table name with prefix if configured."""
        return f"{self._table_prefix}{base_name}" if self._table_prefix else base_name

    async def log_failure(self, failure: FailureLog) -> FailureLog:
        """
        Persist a failure log to the database.

        Also updates the associated failure pattern.

        Args:
            failure: The failure to log

        Returns:
            The logged failure

        Raises:
            RuntimeError: If not initialized or database operation fails
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("failure_logs")

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Insert failure log
                    await cur.execute(
                        f"""INSERT INTO {table_name}
                           (id, task_id, agent_run_id, goal, phase, genotype_id, genotype_summary,
                            failure_type, error_message, error_details, phenotype_id, phenotype_summary,
                            test_results, arbiter_critique, arbiter_score, similarity_hash, tags, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            failure.id,
                            failure.task_id,
                            failure.agent_run_id,
                            failure.goal,
                            failure.phase,
                            failure.genotype_id,
                            failure.genotype_summary,
                            failure.failure_type,
                            failure.error_message,
                            json.dumps(failure.error_details),
                            failure.phenotype_id,
                            failure.phenotype_summary,
                            json.dumps(failure.test_results) if failure.test_results else None,
                            failure.arbiter_critique,
                            failure.arbiter_score,
                            failure.similarity_hash,
                            json.dumps(failure.tags),
                            failure.created_at,
                        ),
                    )

                    # Update pattern
                    await self._update_pattern(conn, failure)

                await conn.commit()
                return failure

        except psycopg.Error as e:
            raise RuntimeError(f"Failed to log failure: {e}")

    async def _update_pattern(self, conn, failure: FailureLog) -> None:
        """Update or create failure pattern based on new failure."""
        if not failure.similarity_hash:
            return

        table_name = self._get_table_name("failure_patterns")
        now = datetime.utcnow()

        async with conn.cursor() as cur:
            # Check for existing pattern
            await cur.execute(
                f"SELECT pattern_id, occurrence_count FROM {table_name} WHERE pattern_id = %s",
                (failure.similarity_hash,),
            )
            row = await cur.fetchone()

            if row:
                # Update existing pattern
                await cur.execute(
                    f"""UPDATE {table_name} SET
                       occurrence_count = occurrence_count + 1,
                       last_seen = %s
                       WHERE pattern_id = %s""",
                    (now, failure.similarity_hash),
                )
            else:
                # Create new pattern
                await cur.execute(
                    f"""INSERT INTO {table_name}
                       (pattern_id, description, failure_type, occurrence_count,
                        first_seen, last_seen, common_error_messages, suggested_avoidance)
                       VALUES (%s, %s, %s, 1, %s, %s, %s, %s)""",
                    (
                        failure.similarity_hash,
                        f"Failure in: {failure.goal[:100]}",
                        failure.failure_type,
                        now,
                        now,
                        json.dumps([failure.error_message[:200]]),
                        "",  # Will be generated later
                    ),
                )

    async def get_failure(self, failure_id: str) -> Optional[FailureLog]:
        """
        Retrieve a specific failure by ID.

        Args:
            failure_id: The failure ID to retrieve

        Returns:
            The FailureLog if found, None otherwise
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("failure_logs")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT * FROM {table_name} WHERE id = %s", (failure_id,))
                row = await cur.fetchone()

        if row:
            return self._row_to_failure(row)
        return None

    async def get_similar_failures(
        self,
        goal: str,
        failure_types: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[FailureLog]:
        """
        Retrieve similar past failures for a goal.

        Uses similarity hash matching and full-text search on goal.

        Args:
            goal: The current goal
            failure_types: Optional filter by failure types
            limit: Maximum failures to return

        Returns:
            List of similar FailureLog objects
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("failure_logs")
        failures = []

        # Compute similarity hashes
        from .failure_storage import FailureLearningManager

        manager = FailureLearningManager.__new__(FailureLearningManager)

        types_to_check = failure_types or [
            "test_failure",
            "runtime_error",
            "compile_error",
            "validation_failed",
            "timeout",
        ]

        hashes_to_check = []
        for ftype in types_to_check:
            hashes_to_check.append(manager._compute_similarity_hash(goal, ftype))

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                # Query by similarity hash
                await cur.execute(
                    f"""SELECT * FROM {table_name}
                       WHERE similarity_hash = ANY(%s)
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (hashes_to_check, limit),
                )

                rows = await cur.fetchall()
                for row in rows:
                    failures.append(self._row_to_failure(row))

                # Also use full-text search if we don't have enough
                if len(failures) < limit:
                    failure_ids = [f.id for f in failures]
                    id_filter = "AND id != ALL(%s)" if failure_ids else ""

                    await cur.execute(
                        f"""SELECT * FROM {table_name}
                           WHERE to_tsvector('english', goal) @@ plainto_tsquery('english', %s)
                           {id_filter}
                           ORDER BY created_at DESC
                           LIMIT %s""",
                        (goal, *([failure_ids] if failure_ids else []), limit - len(failures)),
                    )

                    additional = await cur.fetchall()
                    for row in additional:
                        failures.append(self._row_to_failure(row))

        return failures[:limit]

    async def get_pattern(self, pattern_id: str) -> Optional[FailurePattern]:
        """
        Retrieve a specific failure pattern by ID.

        Args:
            pattern_id: The pattern ID to retrieve

        Returns:
            The FailurePattern if found, None otherwise
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("failure_patterns")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT * FROM {table_name} WHERE pattern_id = %s", (pattern_id,)
                )
                row = await cur.fetchone()

        if row:
            return self._row_to_pattern(row)
        return None

    async def get_patterns_for_failures(self, failure_ids: List[str]) -> List[FailurePattern]:
        """
        Retrieve patterns associated with given failures.

        Args:
            failure_ids: List of failure IDs

        Returns:
            List of associated FailurePattern objects
        """
        if not self._pool or not failure_ids:
            return []

        failure_table = self._get_table_name("failure_logs")
        pattern_table = self._get_table_name("failure_patterns")

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                # Get unique similarity hashes from failures
                await cur.execute(
                    f"""SELECT DISTINCT similarity_hash FROM {failure_table}
                       WHERE id = ANY(%s)""",
                    (failure_ids,),
                )

                hashes = [
                    row["similarity_hash"] for row in await cur.fetchall() if row["similarity_hash"]
                ]

                if not hashes:
                    return []

                # Get patterns for these hashes
                await cur.execute(
                    f"SELECT * FROM {pattern_table} WHERE pattern_id = ANY(%s)",
                    (hashes,),
                )

                patterns = []
                for row in await cur.fetchall():
                    patterns.append(self._row_to_pattern(row))

        return patterns

    async def update_pattern(self, pattern: FailurePattern) -> None:
        """
        Update an existing failure pattern.

        Args:
            pattern: The pattern to update
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("failure_patterns")

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""UPDATE {table_name} SET
                       description = %s,
                       failure_type = %s,
                       occurrence_count = %s,
                       first_seen = %s,
                       last_seen = %s,
                       common_error_messages = %s,
                       suggested_avoidance = %s
                       WHERE pattern_id = %s""",
                    (
                        pattern.description,
                        pattern.failure_type,
                        pattern.occurrence_count,
                        pattern.first_seen,
                        pattern.last_seen,
                        json.dumps(pattern.common_error_messages),
                        pattern.suggested_avoidance,
                        pattern.pattern_id,
                    ),
                )
            await conn.commit()

    async def get_failure_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get failure statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with statistics
        """
        if not self._pool:
            raise RuntimeError("Storage not initialized")

        table_name = self._get_table_name("failure_logs")
        cutoff = datetime.utcnow() - timedelta(days=days)

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # Total failures
                await cur.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE created_at >= %s",
                    (cutoff,),
                )
                total = (await cur.fetchone())[0]

                # By type
                by_type = {}
                await cur.execute(
                    f"""SELECT failure_type, COUNT(*) FROM {table_name}
                       WHERE created_at >= %s GROUP BY failure_type""",
                    (cutoff,),
                )
                for row in await cur.fetchall():
                    by_type[row[0]] = row[1]

                # Most common errors
                await cur.execute(
                    f"""SELECT error_message, COUNT(*) as cnt FROM {table_name}
                       WHERE created_at >= %s
                       GROUP BY error_message ORDER BY cnt DESC LIMIT 10""",
                    (cutoff,),
                )
                common_errors = [(row[0][:100], row[1]) for row in await cur.fetchall()]

        return {
            "period_days": days,
            "total_failures": total,
            "by_type": by_type,
            "common_errors": common_errors,
        }

    def _row_to_failure(self, row: dict) -> FailureLog:
        """Convert database row to FailureLog."""
        return FailureLog(
            id=row["id"],
            task_id=row["task_id"],
            agent_run_id=row["agent_run_id"],
            goal=row["goal"],
            phase=row["phase"],
            genotype_id=row["genotype_id"],
            genotype_summary=row["genotype_summary"],
            failure_type=row["failure_type"],
            error_message=row["error_message"],
            error_details=self._parse_json_field(row["error_details"], {}),
            phenotype_id=row["phenotype_id"],
            phenotype_summary=row["phenotype_summary"],
            test_results=self._parse_json_field(row["test_results"], None),
            arbiter_critique=row["arbiter_critique"],
            arbiter_score=row["arbiter_score"],
            similarity_hash=row["similarity_hash"],
            tags=self._parse_json_field(row["tags"], []),
            created_at=row["created_at"],
        )

    def _row_to_pattern(self, row: dict) -> FailurePattern:
        """Convert database row to FailurePattern."""
        return FailurePattern(
            pattern_id=row["pattern_id"],
            description=row["description"],
            failure_type=row["failure_type"],
            occurrence_count=row["occurrence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            common_error_messages=self._parse_json_field(row["common_error_messages"], []),
            suggested_avoidance=row["suggested_avoidance"] or "",
        )

    def _parse_json_field(self, value: Any, default: Any) -> Any:
        """
        Parse a JSON field that may already be deserialized by psycopg.

        PostgreSQL JSONB columns are auto-deserialized by psycopg into
        Python dicts/lists, so we need to handle both cases.

        Args:
            value: The field value (str, dict, list, or None)
            default: Default value if field is None/empty

        Returns:
            Parsed JSON value or default
        """
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            # Already deserialized by psycopg
            return value
        if isinstance(value, str):
            # String that needs parsing
            return json.loads(value) if value else default
        return default

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL failure storage closed")
