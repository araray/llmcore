# src/llmcore/agents/darwin/sqlite_failure_storage.py
"""
SQLite implementation of failure storage backend.

Provides persistent storage of failure logs and patterns using SQLite database.
Suitable for development and single-user deployments.
"""

import json
import logging
import os
import pathlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import aiosqlite

    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from .failure_storage import BaseFailureStorage, FailureLog, FailurePattern

logger = logging.getLogger(__name__)


class SqliteFailureStorage(BaseFailureStorage):
    """
    SQLite-based storage for failure logs and patterns.

    Uses aiosqlite for async database operations. Stores all data in a single
    SQLite database file with two tables: failure_logs and failure_patterns.

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
        error_details TEXT,
        phenotype_id TEXT,
        phenotype_summary TEXT,
        test_results TEXT,
        arbiter_critique TEXT,
        arbiter_score REAL,
        similarity_hash TEXT,
        tags TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_failure_task ON failure_logs(task_id);
    CREATE INDEX IF NOT EXISTS idx_failure_type ON failure_logs(failure_type);
    CREATE INDEX IF NOT EXISTS idx_failure_similarity ON failure_logs(similarity_hash);
    CREATE INDEX IF NOT EXISTS idx_failure_created ON failure_logs(created_at);

    CREATE TABLE IF NOT EXISTS failure_patterns (
        pattern_id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        failure_type TEXT NOT NULL,
        occurrence_count INTEGER DEFAULT 1,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        common_error_messages TEXT,
        suggested_avoidance TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pattern_type ON failure_patterns(failure_type);
    CREATE INDEX IF NOT EXISTS idx_pattern_count ON failure_patterns(occurrence_count);
    """

    def __init__(self):
        """Initialize SQLite failure storage."""
        self._db_path: Optional[pathlib.Path] = None
        self._conn: Optional["aiosqlite.Connection"] = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SQLite database and create schema.

        Args:
            config: Configuration dictionary with 'path' key

        Raises:
            ImportError: If aiosqlite is not installed
            ValueError: If path is not provided
            StorageError: If database initialization fails
        """
        if not AIOSQLITE_AVAILABLE:
            raise ImportError(
                "aiosqlite library is not installed. "
                "Install with: pip install aiosqlite"
            )

        db_path_str = config.get("path")
        if not db_path_str:
            raise ValueError("SQLite failure storage 'path' not specified in config")

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

            logger.info(f"SQLite failure storage initialized at: {self._db_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite failure storage: {e}")

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
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            # Insert failure log
            await self._conn.execute(
                """INSERT INTO failure_logs
                   (id, task_id, agent_run_id, goal, phase, genotype_id, genotype_summary,
                    failure_type, error_message, error_details, phenotype_id, phenotype_summary,
                    test_results, arbiter_critique, arbiter_score, similarity_hash, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    failure.created_at.isoformat(),
                ),
            )

            # Update pattern
            await self._update_pattern(failure)

            await self._conn.commit()
            return failure

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to log failure: {e}")

    async def _update_pattern(self, failure: FailureLog) -> None:
        """Update or create failure pattern based on new failure."""
        if not failure.similarity_hash:
            return

        # Check for existing pattern
        cursor = await self._conn.execute(
            "SELECT pattern_id, occurrence_count FROM failure_patterns WHERE pattern_id = ?",
            (failure.similarity_hash,),
        )
        row = await cursor.fetchone()

        now = datetime.utcnow().isoformat()

        if row:
            # Update existing pattern
            await self._conn.execute(
                """UPDATE failure_patterns SET
                   occurrence_count = occurrence_count + 1,
                   last_seen = ?
                   WHERE pattern_id = ?""",
                (now, failure.similarity_hash),
            )
        else:
            # Create new pattern
            await self._conn.execute(
                """INSERT INTO failure_patterns
                   (pattern_id, description, failure_type, occurrence_count,
                    first_seen, last_seen, common_error_messages, suggested_avoidance)
                   VALUES (?, ?, ?, 1, ?, ?, ?, ?)""",
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
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM failure_logs WHERE id = ?", (failure_id,)
        )
        row = await cursor.fetchone()

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

        Uses similarity hash matching and keyword-based search.

        Args:
            goal: The current goal
            failure_types: Optional filter by failure types
            limit: Maximum failures to return

        Returns:
            List of similar FailureLog objects
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

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

        # Query by similarity hash
        placeholders = ",".join("?" * len(hashes_to_check))
        cursor = await self._conn.execute(
            f"""SELECT * FROM failure_logs
               WHERE similarity_hash IN ({placeholders})
               ORDER BY created_at DESC
               LIMIT ?""",
            (*hashes_to_check, limit),
        )

        rows = await cursor.fetchall()
        for row in rows:
            failures.append(self._row_to_failure(row))

        # Also get recent failures with similar keywords if we don't have enough
        if len(failures) < limit:
            keywords = goal.lower().split()[:5]  # First 5 words
            keyword_pattern = "%".join(keywords)

            cursor = await self._conn.execute(
                f"""SELECT * FROM failure_logs
                   WHERE goal LIKE ? AND id NOT IN ({','.join('?' * len(failures)) if failures else 'SELECT NULL WHERE 1=0'})
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{keyword_pattern}%", *[f.id for f in failures], limit - len(failures)),
            )

            additional = await cursor.fetchall()
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
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM failure_patterns WHERE pattern_id = ?", (pattern_id,)
        )
        row = await cursor.fetchone()

        if row:
            return self._row_to_pattern(row)
        return None

    async def get_patterns_for_failures(
        self, failure_ids: List[str]
    ) -> List[FailurePattern]:
        """
        Retrieve patterns associated with given failures.

        Args:
            failure_ids: List of failure IDs

        Returns:
            List of associated FailurePattern objects
        """
        if not self._conn or not failure_ids:
            return []

        # Get unique similarity hashes from failures
        placeholders = ",".join("?" * len(failure_ids))
        cursor = await self._conn.execute(
            f"""SELECT DISTINCT similarity_hash FROM failure_logs
               WHERE id IN ({placeholders})""",
            failure_ids,
        )

        hashes = [row[0] for row in await cursor.fetchall() if row[0]]

        if not hashes:
            return []

        # Get patterns for these hashes
        placeholders = ",".join("?" * len(hashes))
        cursor = await self._conn.execute(
            f"SELECT * FROM failure_patterns WHERE pattern_id IN ({placeholders})",
            hashes,
        )

        patterns = []
        for row in await cursor.fetchall():
            patterns.append(self._row_to_pattern(row))

        return patterns

    async def update_pattern(self, pattern: FailurePattern) -> None:
        """
        Update an existing failure pattern.

        Args:
            pattern: The pattern to update
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        await self._conn.execute(
            """UPDATE failure_patterns SET
               description = ?,
               failure_type = ?,
               occurrence_count = ?,
               first_seen = ?,
               last_seen = ?,
               common_error_messages = ?,
               suggested_avoidance = ?
               WHERE pattern_id = ?""",
            (
                pattern.description,
                pattern.failure_type,
                pattern.occurrence_count,
                pattern.first_seen.isoformat(),
                pattern.last_seen.isoformat(),
                json.dumps(pattern.common_error_messages),
                pattern.suggested_avoidance,
                pattern.pattern_id,
            ),
        )
        await self._conn.commit()

    async def get_failure_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get failure statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with statistics
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Total failures
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM failure_logs WHERE created_at >= ?", (cutoff,)
        )
        total = (await cursor.fetchone())[0]

        # By type
        by_type = {}
        cursor = await self._conn.execute(
            """SELECT failure_type, COUNT(*) FROM failure_logs
               WHERE created_at >= ? GROUP BY failure_type""",
            (cutoff,),
        )
        for row in await cursor.fetchall():
            by_type[row[0]] = row[1]

        # Most common errors
        cursor = await self._conn.execute(
            """SELECT error_message, COUNT(*) as cnt FROM failure_logs
               WHERE created_at >= ?
               GROUP BY error_message ORDER BY cnt DESC LIMIT 10""",
            (cutoff,),
        )
        common_errors = [(row[0][:100], row[1]) for row in await cursor.fetchall()]

        return {
            "period_days": days,
            "total_failures": total,
            "by_type": by_type,
            "common_errors": common_errors,
        }

    def _row_to_failure(self, row) -> FailureLog:
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
            error_details=json.loads(row["error_details"]) if row["error_details"] else {},
            phenotype_id=row["phenotype_id"],
            phenotype_summary=row["phenotype_summary"],
            test_results=json.loads(row["test_results"]) if row["test_results"] else None,
            arbiter_critique=row["arbiter_critique"],
            arbiter_score=row["arbiter_score"],
            similarity_hash=row["similarity_hash"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_pattern(self, row) -> FailurePattern:
        """Convert database row to FailurePattern."""
        return FailurePattern(
            pattern_id=row["pattern_id"],
            description=row["description"],
            failure_type=row["failure_type"],
            occurrence_count=row["occurrence_count"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            common_error_messages=json.loads(row["common_error_messages"])
            if row["common_error_messages"]
            else [],
            suggested_avoidance=row["suggested_avoidance"] or "",
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("SQLite failure storage closed")
