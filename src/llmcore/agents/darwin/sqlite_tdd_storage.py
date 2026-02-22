# src/llmcore/agents/darwin/sqlite_tdd_storage.py
"""
SQLite implementation of TDD storage backend.

Provides persistent storage of test suites, generated tests, and TDD cycle results
using SQLite database. Suitable for development and single-user deployments.

Tables:
    - test_suites: Test suite definitions
    - test_specifications: Individual test specifications
    - generated_tests: Generated executable tests
    - tdd_cycle_results: Results of TDD cycles
    - tdd_sessions: Multi-iteration TDD sessions
"""

import json
import logging
import os
import pathlib
from datetime import datetime, timedelta
from typing import Any

try:
    import aiosqlite

    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from .tdd_support import (
    BaseTDDStorage,
    GeneratedTest,
    TDDCycleResult,
    TDDSession,
    TestExecutionResult,
    TestSpecification,
    TestSuite,
)

logger = logging.getLogger(__name__)


class SqliteTDDStorage(BaseTDDStorage):
    """
    SQLite-based storage for TDD data.

    Uses aiosqlite for async database operations. Stores all data in a single
    SQLite database file with tables for suites, specs, tests, and results.

    Schema:
        test_suites: Test suite metadata
        test_specifications: Individual test specs (linked to suites)
        generated_tests: Executable test code (linked to specs)
        tdd_cycle_results: TDD iteration results
        tdd_sessions: Multi-cycle session tracking
    """

    SCHEMA = """
    -- Test Suites
    CREATE TABLE IF NOT EXISTS test_suites (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        requirements TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'python',
        framework TEXT NOT NULL DEFAULT 'pytest',
        setup_code TEXT,
        teardown_code TEXT,
        fixture_code TEXT,
        import_statements TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_suite_task ON test_suites(task_id);
    CREATE INDEX IF NOT EXISTS idx_suite_created ON test_suites(created_at);

    -- Test Specifications
    CREATE TABLE IF NOT EXISTS test_specifications (
        id TEXT PRIMARY KEY,
        suite_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        test_type TEXT NOT NULL DEFAULT 'unit',
        inputs TEXT,
        expected_output TEXT,
        expected_behavior TEXT,
        expected_exception TEXT,
        priority INTEGER DEFAULT 2,
        tags TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (suite_id) REFERENCES test_suites(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_spec_suite ON test_specifications(suite_id);
    CREATE INDEX IF NOT EXISTS idx_spec_type ON test_specifications(test_type);

    -- Generated Tests
    CREATE TABLE IF NOT EXISTS generated_tests (
        id TEXT PRIMARY KEY,
        spec_id TEXT,
        suite_id TEXT,
        spec_name TEXT NOT NULL,
        test_code TEXT NOT NULL,
        imports TEXT,
        fixtures TEXT,
        validation_status TEXT DEFAULT 'pending',
        validation_errors TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (spec_id) REFERENCES test_specifications(id) ON DELETE SET NULL
    );

    CREATE INDEX IF NOT EXISTS idx_gentest_spec ON generated_tests(spec_id);
    CREATE INDEX IF NOT EXISTS idx_gentest_suite ON generated_tests(suite_id);

    -- TDD Cycle Results
    CREATE TABLE IF NOT EXISTS tdd_cycle_results (
        id TEXT PRIMARY KEY,
        suite_id TEXT,
        session_id TEXT,
        iteration INTEGER NOT NULL,
        tests_generated INTEGER NOT NULL,
        tests_executed INTEGER NOT NULL,
        tests_passed INTEGER NOT NULL,
        tests_failed INTEGER NOT NULL,
        code_generated INTEGER NOT NULL,
        implementation_code TEXT,
        final_code TEXT,
        all_tests_pass INTEGER NOT NULL,
        execution_results TEXT,
        failure_analysis TEXT,
        suggestions TEXT,
        total_time_ms REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (suite_id) REFERENCES test_suites(id) ON DELETE SET NULL
    );

    CREATE INDEX IF NOT EXISTS idx_cycle_suite ON tdd_cycle_results(suite_id);
    CREATE INDEX IF NOT EXISTS idx_cycle_session ON tdd_cycle_results(session_id);
    CREATE INDEX IF NOT EXISTS idx_cycle_created ON tdd_cycle_results(created_at);

    -- TDD Sessions
    CREATE TABLE IF NOT EXISTS tdd_sessions (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        requirements TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'python',
        framework TEXT NOT NULL DEFAULT 'pytest',
        suite_id TEXT,
        cycles TEXT,
        current_iteration INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        best_pass_rate REAL DEFAULT 0.0,
        final_implementation TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY (suite_id) REFERENCES test_suites(id) ON DELETE SET NULL
    );

    CREATE INDEX IF NOT EXISTS idx_session_task ON tdd_sessions(task_id);
    CREATE INDEX IF NOT EXISTS idx_session_status ON tdd_sessions(status);
    """

    def __init__(self):
        """Initialize SQLite TDD storage."""
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
            raise ValueError("SQLite TDD storage 'path' not specified in config")

        self._db_path = pathlib.Path(os.path.expanduser(db_path_str))

        try:
            # Create parent directory if needed
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row

            # Enable WAL mode for better concurrency
            await self._conn.execute("PRAGMA journal_mode=WAL;")
            await self._conn.execute("PRAGMA foreign_keys=ON;")

            # Create schema
            await self._conn.executescript(self.SCHEMA)
            await self._conn.commit()

            logger.info(f"SQLite TDD storage initialized at: {self._db_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite TDD storage: {e}")

    async def save_test_suite(self, suite: TestSuite) -> TestSuite:
        """
        Persist a test suite to the database.

        Also saves the associated test specifications.

        Args:
            suite: The test suite to save

        Returns:
            The saved suite

        Raises:
            RuntimeError: If not initialized or database operation fails
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            # Generate ID if not set
            if not suite.id:
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:18]
                suite.id = f"suite_{timestamp}_{suite.task_id[:8]}"

            now = datetime.utcnow().isoformat()
            suite.updated_at = datetime.utcnow()

            # Insert or replace suite
            await self._conn.execute(
                """INSERT OR REPLACE INTO test_suites
                   (id, task_id, requirements, language, framework, setup_code,
                    teardown_code, fixture_code, import_statements, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    suite.id,
                    suite.task_id,
                    suite.requirements,
                    suite.language,
                    suite.framework,
                    suite.setup_code,
                    suite.teardown_code,
                    suite.fixture_code,
                    json.dumps(suite.import_statements),
                    suite.created_at.isoformat(),
                    suite.updated_at.isoformat(),
                ),
            )

            # Save specifications
            for spec in suite.specifications:
                if not spec.id:
                    spec.id = f"spec_{suite.id}_{len(suite.specifications):03d}"

                await self._conn.execute(
                    """INSERT OR REPLACE INTO test_specifications
                       (id, suite_id, name, description, test_type, inputs,
                        expected_output, expected_behavior, expected_exception,
                        priority, tags, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        spec.id,
                        suite.id,
                        spec.name,
                        spec.description,
                        spec.test_type,
                        json.dumps(spec.inputs),
                        json.dumps(spec.expected_output)
                        if spec.expected_output is not None
                        else None,
                        spec.expected_behavior,
                        spec.expected_exception,
                        spec.priority,
                        json.dumps(spec.tags),
                        spec.created_at.isoformat(),
                    ),
                )

            await self._conn.commit()
            return suite

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save test suite: {e}")

    async def get_test_suite(self, suite_id: str) -> TestSuite | None:
        """
        Retrieve a specific test suite by ID.

        Also retrieves associated test specifications.

        Args:
            suite_id: The suite ID to retrieve

        Returns:
            The TestSuite if found, None otherwise
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute("SELECT * FROM test_suites WHERE id = ?", (suite_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        # Get specifications
        spec_cursor = await self._conn.execute(
            "SELECT * FROM test_specifications WHERE suite_id = ? ORDER BY created_at",
            (suite_id,),
        )
        spec_rows = await spec_cursor.fetchall()

        specifications = [self._row_to_spec(r) for r in spec_rows]

        return self._row_to_suite(row, specifications)

    async def get_test_suites_for_task(self, task_id: str) -> list[TestSuite]:
        """
        Retrieve all test suites for a task.

        Args:
            task_id: The task ID to filter by

        Returns:
            List of TestSuite objects for the task
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM test_suites WHERE task_id = ? ORDER BY created_at DESC",
            (task_id,),
        )
        rows = await cursor.fetchall()

        suites = []
        for row in rows:
            # Get specifications for each suite
            spec_cursor = await self._conn.execute(
                "SELECT * FROM test_specifications WHERE suite_id = ?",
                (row["id"],),
            )
            spec_rows = await spec_cursor.fetchall()
            specifications = [self._row_to_spec(r) for r in spec_rows]
            suites.append(self._row_to_suite(row, specifications))

        return suites

    async def save_generated_test(self, test: GeneratedTest) -> GeneratedTest:
        """
        Persist a generated test to storage.

        Args:
            test: The generated test to save

        Returns:
            The saved test with ID set
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            # Generate ID if not set
            if not test.id:
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:18]
                test.id = f"gen_{timestamp}"

            # Determine suite_id from spec if available
            suite_id = None
            if test.spec_id:
                cursor = await self._conn.execute(
                    "SELECT suite_id FROM test_specifications WHERE id = ?",
                    (test.spec_id,),
                )
                row = await cursor.fetchone()
                if row:
                    suite_id = row["suite_id"]

            await self._conn.execute(
                """INSERT OR REPLACE INTO generated_tests
                   (id, spec_id, suite_id, spec_name, test_code, imports, fixtures,
                    validation_status, validation_errors, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    test.id,
                    test.spec_id,
                    suite_id,
                    test.spec_name,
                    test.test_code,
                    json.dumps(test.imports),
                    test.fixtures,
                    test.validation_status,
                    json.dumps(test.validation_errors),
                    test.created_at.isoformat(),
                ),
            )

            await self._conn.commit()
            return test

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save generated test: {e}")

    async def get_generated_tests(self, suite_id: str) -> list[GeneratedTest]:
        """
        Retrieve all generated tests for a suite.

        Args:
            suite_id: The suite ID to filter by

        Returns:
            List of GeneratedTest objects
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            """SELECT * FROM generated_tests
               WHERE suite_id = ? OR spec_id IN (
                   SELECT id FROM test_specifications WHERE suite_id = ?
               )
               ORDER BY created_at""",
            (suite_id, suite_id),
        )
        rows = await cursor.fetchall()

        return [self._row_to_generated_test(r) for r in rows]

    async def save_cycle_result(self, result: TDDCycleResult) -> TDDCycleResult:
        """
        Persist a TDD cycle result to storage.

        Args:
            result: The cycle result to save

        Returns:
            The saved result with ID set
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            # Generate ID if not set
            if not result.id:
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:18]
                result.id = f"cycle_{timestamp}"

            # Serialize execution results (mode='json' converts datetime to ISO strings)
            exec_results_json = json.dumps(
                [r.model_dump(mode="json") for r in result.execution_results]
            )

            await self._conn.execute(
                """INSERT OR REPLACE INTO tdd_cycle_results
                   (id, suite_id, session_id, iteration, tests_generated, tests_executed,
                    tests_passed, tests_failed, code_generated, implementation_code,
                    final_code, all_tests_pass, execution_results, failure_analysis,
                    suggestions, total_time_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.id,
                    result.suite_id,
                    None,  # session_id - set when saving session
                    result.iteration,
                    result.tests_generated,
                    result.tests_executed,
                    result.tests_passed,
                    result.tests_failed,
                    1 if result.code_generated else 0,
                    result.implementation_code,
                    result.final_code,
                    1 if result.all_tests_pass else 0,
                    exec_results_json,
                    result.failure_analysis,
                    json.dumps(result.suggestions),
                    result.total_time_ms,
                    result.created_at.isoformat(),
                ),
            )

            await self._conn.commit()
            return result

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save cycle result: {e}")

    async def get_cycle_results(
        self,
        suite_id: str,
        limit: int = 10,
    ) -> list[TDDCycleResult]:
        """
        Retrieve cycle results for a suite.

        Args:
            suite_id: The suite ID to filter by
            limit: Maximum number of results to return

        Returns:
            List of TDDCycleResult objects
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            """SELECT * FROM tdd_cycle_results
               WHERE suite_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (suite_id, limit),
        )
        rows = await cursor.fetchall()

        return [self._row_to_cycle_result(r) for r in rows]

    async def save_session(self, session: TDDSession) -> TDDSession:
        """
        Persist a TDD session to storage.

        Args:
            session: The session to save

        Returns:
            The saved session with ID set
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        try:
            # Generate ID if not set
            if not session.id:
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:18]
                session.id = f"sess_{timestamp}"

            # Serialize cycles (mode='json' converts datetime to ISO strings)
            cycles_json = json.dumps([c.model_dump(mode="json") for c in session.cycles])

            await self._conn.execute(
                """INSERT OR REPLACE INTO tdd_sessions
                   (id, task_id, requirements, language, framework, suite_id,
                    cycles, current_iteration, status, best_pass_rate,
                    final_implementation, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.id,
                    session.task_id,
                    session.requirements,
                    session.language,
                    session.framework,
                    session.suite_id,
                    cycles_json,
                    session.current_iteration,
                    session.status,
                    session.best_pass_rate,
                    session.final_implementation,
                    session.started_at.isoformat(),
                    session.completed_at.isoformat() if session.completed_at else None,
                ),
            )

            await self._conn.commit()
            return session

        except Exception as e:
            await self._conn.rollback()
            raise RuntimeError(f"Failed to save session: {e}")

    async def get_session(self, session_id: str) -> TDDSession | None:
        """
        Retrieve a specific session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The TDDSession if found, None otherwise
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute("SELECT * FROM tdd_sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_session(row)

    async def get_sessions_for_task(self, task_id: str) -> list[TDDSession]:
        """
        Retrieve all sessions for a task.

        Args:
            task_id: The task ID to filter by

        Returns:
            List of TDDSession objects
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cursor = await self._conn.execute(
            "SELECT * FROM tdd_sessions WHERE task_id = ? ORDER BY started_at DESC",
            (task_id,),
        )
        rows = await cursor.fetchall()

        return [self._row_to_session(r) for r in rows]

    async def get_stats(self, days: int = 30) -> dict[str, Any]:
        """
        Get TDD statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with statistics
        """
        if not self._conn:
            raise RuntimeError("Storage not initialized")

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Total suites
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM test_suites WHERE created_at >= ?", (cutoff,)
        )
        total_suites = (await cursor.fetchone())[0]

        # Total cycles
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM tdd_cycle_results WHERE created_at >= ?", (cutoff,)
        )
        total_cycles = (await cursor.fetchone())[0]

        # Average pass rate
        cursor = await self._conn.execute(
            """SELECT AVG(CAST(tests_passed AS REAL) / NULLIF(tests_executed, 0) * 100)
               FROM tdd_cycle_results WHERE created_at >= ? AND tests_executed > 0""",
            (cutoff,),
        )
        avg_pass_rate = (await cursor.fetchone())[0] or 0.0

        # Success rate (all tests pass)
        cursor = await self._conn.execute(
            """SELECT COUNT(*) FROM tdd_cycle_results
               WHERE created_at >= ? AND all_tests_pass = 1""",
            (cutoff,),
        )
        successful_cycles = (await cursor.fetchone())[0]

        # By language
        cursor = await self._conn.execute(
            """SELECT language, COUNT(*) FROM test_suites
               WHERE created_at >= ? GROUP BY language""",
            (cutoff,),
        )
        by_language = {row[0]: row[1] for row in await cursor.fetchall()}

        # By framework
        cursor = await self._conn.execute(
            """SELECT framework, COUNT(*) FROM test_suites
               WHERE created_at >= ? GROUP BY framework""",
            (cutoff,),
        )
        by_framework = {row[0]: row[1] for row in await cursor.fetchall()}

        # Average iterations to success
        cursor = await self._conn.execute(
            """SELECT AVG(iteration) FROM tdd_cycle_results
               WHERE created_at >= ? AND all_tests_pass = 1""",
            (cutoff,),
        )
        avg_iterations = (await cursor.fetchone())[0] or 0.0

        return {
            "period_days": days,
            "total_suites": total_suites,
            "total_cycles": total_cycles,
            "average_pass_rate": round(avg_pass_rate, 2),
            "success_rate": round(
                (successful_cycles / total_cycles * 100) if total_cycles > 0 else 0, 2
            ),
            "by_language": by_language,
            "by_framework": by_framework,
            "average_iterations_to_success": round(avg_iterations, 2),
        }

    def _row_to_spec(self, row) -> TestSpecification:
        """Convert database row to TestSpecification."""
        return TestSpecification(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            test_type=row["test_type"],
            inputs=json.loads(row["inputs"]) if row["inputs"] else {},
            expected_output=json.loads(row["expected_output"]) if row["expected_output"] else None,
            expected_behavior=row["expected_behavior"],
            expected_exception=row["expected_exception"],
            priority=row["priority"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_suite(
        self,
        row,
        specifications: list[TestSpecification],
    ) -> TestSuite:
        """Convert database row to TestSuite."""
        return TestSuite(
            id=row["id"],
            task_id=row["task_id"],
            requirements=row["requirements"],
            language=row["language"],
            framework=row["framework"],
            specifications=specifications,
            setup_code=row["setup_code"],
            teardown_code=row["teardown_code"],
            fixture_code=row["fixture_code"],
            import_statements=json.loads(row["import_statements"])
            if row["import_statements"]
            else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_generated_test(self, row) -> GeneratedTest:
        """Convert database row to GeneratedTest."""
        return GeneratedTest(
            id=row["id"],
            spec_id=row["spec_id"],
            spec_name=row["spec_name"],
            test_code=row["test_code"],
            imports=json.loads(row["imports"]) if row["imports"] else [],
            fixtures=row["fixtures"],
            validation_status=row["validation_status"],
            validation_errors=json.loads(row["validation_errors"])
            if row["validation_errors"]
            else [],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_cycle_result(self, row) -> TDDCycleResult:
        """Convert database row to TDDCycleResult."""
        # Parse execution results
        exec_results_data = json.loads(row["execution_results"]) if row["execution_results"] else []
        exec_results = [TestExecutionResult(**r) for r in exec_results_data]

        return TDDCycleResult(
            id=row["id"],
            suite_id=row["suite_id"],
            iteration=row["iteration"],
            tests_generated=row["tests_generated"],
            tests_executed=row["tests_executed"],
            tests_passed=row["tests_passed"],
            tests_failed=row["tests_failed"],
            code_generated=bool(row["code_generated"]),
            implementation_code=row["implementation_code"],
            final_code=row["final_code"],
            all_tests_pass=bool(row["all_tests_pass"]),
            execution_results=exec_results,
            failure_analysis=row["failure_analysis"],
            suggestions=json.loads(row["suggestions"]) if row["suggestions"] else [],
            total_time_ms=row["total_time_ms"] or 0.0,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_session(self, row) -> TDDSession:
        """Convert database row to TDDSession."""
        # Parse cycles
        cycles_data = json.loads(row["cycles"]) if row["cycles"] else []
        cycles = []
        for c_data in cycles_data:
            # Parse nested execution results
            if "execution_results" in c_data:
                c_data["execution_results"] = [
                    TestExecutionResult(**r) for r in c_data["execution_results"]
                ]
            cycles.append(TDDCycleResult(**c_data))

        return TDDSession(
            id=row["id"],
            task_id=row["task_id"],
            requirements=row["requirements"],
            language=row["language"],
            framework=row["framework"],
            suite_id=row["suite_id"],
            cycles=cycles,
            current_iteration=row["current_iteration"],
            status=row["status"],
            best_pass_rate=row["best_pass_rate"] or 0.0,
            final_implementation=row["final_implementation"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("SQLite TDD storage closed")
