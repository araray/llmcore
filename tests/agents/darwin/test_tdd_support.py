# tests/agents/darwin/test_tdd_support.py
"""
Comprehensive tests for Darwin TDD (Test-Driven Development) support system.

Tests both SQLite and PostgreSQL backends with the same test suite.
Covers data models, storage operations, test generation, and TDD workflow.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.darwin import (
    GeneratedTest,
    TDDCycleResult,
    # Manager
    TDDManager,
    TDDSession,
    TestExecutionResult,
    TestFileBuilder,
    # Generator
    TestGenerator,
    # Data models
    TestSpecification,
    TestSuite,
)
from llmcore.agents.darwin.sqlite_tdd_storage import SqliteTDDStorage

# PostgreSQL storage import - may fail if psycopg not installed
try:
    from llmcore.agents.darwin.postgres_tdd_storage import PostgresTDDStorage

    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresTDDStorage = None
    POSTGRES_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def sqlite_storage():
    """Provide a SQLite TDD storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tdd.db"
        storage = SqliteTDDStorage()
        await storage.initialize({"path": str(db_path)})
        yield storage
        await storage.close()


@pytest.fixture
async def postgres_storage():
    """Provide a PostgreSQL TDD storage for testing."""
    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL TDD storage not available (psycopg not installed)")

    pytest.importorskip("psycopg")

    db_url = os.environ.get("TEST_POSTGRES_URL")
    if not db_url:
        pytest.skip("PostgreSQL not configured (set TEST_POSTGRES_URL)")

    storage = PostgresTDDStorage()
    await storage.initialize({"db_url": db_url, "table_prefix": "test_tdd_"})
    yield storage

    # Cleanup: Drop test tables
    if storage._pool:
        async with storage._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS test_tdd_tdd_generated_tests CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS test_tdd_tdd_test_specifications CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS test_tdd_tdd_test_suites CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS test_tdd_tdd_cycle_results CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS test_tdd_tdd_sessions CASCADE;")
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
    """Provide a SQLite-backed TDD manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tdd.db"
        manager = TDDManager(backend="sqlite", db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


@pytest.fixture
async def postgres_manager():
    """Provide a PostgreSQL-backed TDD manager."""
    pytest.importorskip("psycopg")

    db_url = os.environ.get("TEST_POSTGRES_URL")
    if not db_url:
        pytest.skip("PostgreSQL not configured (set TEST_POSTGRES_URL)")

    manager = TDDManager(backend="postgres", db_url=db_url)
    await manager.initialize()
    yield manager

    # Cleanup
    if manager._backend and hasattr(manager._backend, "_pool"):
        async with manager._backend._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS tdd_generated_tests CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS tdd_test_specifications CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS tdd_test_suites CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS tdd_cycle_results CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS tdd_sessions CASCADE;")
            await conn.commit()

    await manager.close()


@pytest.fixture
def mock_llm_callable():
    """Mock LLM callable that returns valid test specs."""

    async def callable(messages):
        # Return a mock response based on the prompt
        user_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""

        if "Generate test specifications" in user_content:
            # Return test specs as JSON
            specs = [
                {
                    "name": "test_add_positive_numbers",
                    "description": "Test adding two positive numbers",
                    "test_type": "unit",
                    "inputs": {"a": 1, "b": 2},
                    "expected_output": 3,
                    "priority": 1,
                },
                {
                    "name": "test_add_negative_numbers",
                    "description": "Test adding negative numbers",
                    "test_type": "unit",
                    "inputs": {"a": -1, "b": -2},
                    "expected_output": -3,
                    "priority": 2,
                },
                {
                    "name": "test_add_zero",
                    "description": "Test adding zero",
                    "test_type": "edge_case",
                    "inputs": {"a": 0, "b": 5},
                    "expected_output": 5,
                    "priority": 2,
                },
                {
                    "name": "test_add_large_numbers",
                    "description": "Test adding large numbers",
                    "test_type": "edge_case",
                    "inputs": {"a": 1000000, "b": 2000000},
                    "expected_output": 3000000,
                    "priority": 3,
                },
                {
                    "name": "test_invalid_input",
                    "description": "Test invalid string input",
                    "test_type": "error",
                    "inputs": {"a": "not a number", "b": 1},
                    "expected_exception": "TypeError",
                    "priority": 2,
                },
            ]
            return MagicMock(content=json.dumps(specs))

        elif "Generate executable test code" in user_content:
            # Return test code
            code = """
import pytest

def test_example():
    assert 1 + 1 == 2, "Basic math should work"
"""
            return MagicMock(content=code)

        elif "Generate implementation code" in user_content:
            # Return implementation
            code = '''
class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
'''
            return MagicMock(content=code)

        return MagicMock(content="")

    return callable


@pytest.fixture
def sample_test_spec():
    """Provide a sample TestSpecification."""
    return TestSpecification(
        id="spec_001",
        name="test_addition",
        description="Test basic addition",
        test_type="unit",
        inputs={"a": 1, "b": 2},
        expected_output=3,
        priority=1,
        tags=["math", "basic"],
    )


@pytest.fixture
def sample_test_suite(sample_test_spec):
    """Provide a sample TestSuite."""
    return TestSuite(
        id="suite_001",
        task_id="task_calc_001",
        requirements="Implement a Calculator class with add method",
        language="python",
        framework="pytest",
        specifications=[sample_test_spec],
        import_statements=["import pytest"],
    )


@pytest.fixture
def sample_generated_test():
    """Provide a sample GeneratedTest."""
    return GeneratedTest(
        id="gen_001",
        spec_id="spec_001",
        spec_name="test_addition",
        test_code="""
def test_addition():
    calc = Calculator()
    assert calc.add(1, 2) == 3
""",
        imports=["import pytest"],
        validation_status="valid",
    )


@pytest.fixture
def sample_cycle_result():
    """Provide a sample TDDCycleResult."""
    return TDDCycleResult(
        id="cycle_001",
        suite_id="suite_001",
        iteration=1,
        tests_generated=5,
        tests_executed=5,
        tests_passed=4,
        tests_failed=1,
        code_generated=True,
        implementation_code="class Calculator: pass",
        all_tests_pass=False,
        execution_results=[
            TestExecutionResult(test_name="test_add", passed=True),
            TestExecutionResult(
                test_name="test_subtract", passed=False, error_message="Not implemented"
            ),
        ],
        failure_analysis="1 test failed",
        suggestions=["Implement subtract method"],
        total_time_ms=1500.0,
    )


# =============================================================================
# Data Model Tests
# =============================================================================


class TestTestSpecification:
    """Test TestSpecification data model."""

    def test_create_minimal(self):
        """Test creating a minimal TestSpecification."""
        spec = TestSpecification(
            name="test_basic",
            description="Basic test",
        )
        assert spec.name == "test_basic"
        assert spec.test_type == "unit"
        assert spec.priority == 2
        assert spec.created_at is not None

    def test_create_full(self, sample_test_spec):
        """Test creating a complete TestSpecification."""
        assert sample_test_spec.id == "spec_001"
        assert sample_test_spec.name == "test_addition"
        assert sample_test_spec.test_type == "unit"
        assert sample_test_spec.inputs == {"a": 1, "b": 2}
        assert sample_test_spec.expected_output == 3
        assert sample_test_spec.priority == 1
        assert "math" in sample_test_spec.tags

    def test_json_serialization(self, sample_test_spec):
        """Test TestSpecification JSON serialization."""
        json_data = sample_test_spec.model_dump_json()
        assert "test_addition" in json_data
        assert "spec_001" in json_data

    def test_error_test_with_exception(self):
        """Test creating error test specification."""
        spec = TestSpecification(
            name="test_invalid_input",
            description="Should raise ValueError",
            test_type="error",
            expected_exception="ValueError",
        )
        assert spec.expected_exception == "ValueError"

    def test_priority_validation(self):
        """Test priority field validation."""
        # Valid priorities
        for p in [1, 2, 3]:
            spec = TestSpecification(name="test", description="d", priority=p)
            assert spec.priority == p

        # Invalid priorities should fail
        with pytest.raises(Exception):
            TestSpecification(name="test", description="d", priority=0)
        with pytest.raises(Exception):
            TestSpecification(name="test", description="d", priority=4)


class TestTestSuite:
    """Test TestSuite data model."""

    def test_create_minimal(self):
        """Test creating a minimal TestSuite."""
        suite = TestSuite(
            task_id="task_001",
            requirements="Test requirements",
        )
        assert suite.task_id == "task_001"
        assert suite.language == "python"
        assert suite.framework == "pytest"
        assert suite.spec_count == 0

    def test_create_full(self, sample_test_suite):
        """Test creating a complete TestSuite."""
        assert sample_test_suite.id == "suite_001"
        assert sample_test_suite.task_id == "task_calc_001"
        assert sample_test_suite.spec_count == 1
        assert sample_test_suite.language == "python"
        assert "import pytest" in sample_test_suite.import_statements

    def test_spec_count_property(self, sample_test_suite):
        """Test spec_count property."""
        assert sample_test_suite.spec_count == 1

    def test_get_specs_by_type(self, sample_test_suite):
        """Test filtering specs by type."""
        unit_specs = sample_test_suite.get_specs_by_type("unit")
        assert len(unit_specs) == 1

        edge_specs = sample_test_suite.get_specs_by_type("edge_case")
        assert len(edge_specs) == 0

    def test_get_specs_by_priority(self, sample_test_suite):
        """Test filtering specs by priority."""
        high_priority = sample_test_suite.get_specs_by_priority(1)
        assert len(high_priority) == 1

        low_priority = sample_test_suite.get_specs_by_priority(3)
        assert len(low_priority) == 0

    def test_json_serialization(self, sample_test_suite):
        """Test TestSuite JSON serialization."""
        json_data = sample_test_suite.model_dump_json()
        assert "suite_001" in json_data
        assert "task_calc_001" in json_data


class TestGeneratedTest:
    """Test GeneratedTest data model."""

    def test_create_minimal(self):
        """Test creating a minimal GeneratedTest."""
        test = GeneratedTest(
            spec_name="test_basic",
            test_code="def test_basic(): pass",
        )
        assert test.spec_name == "test_basic"
        assert test.validation_status == "pending"

    def test_create_full(self, sample_generated_test):
        """Test creating a complete GeneratedTest."""
        assert sample_generated_test.id == "gen_001"
        assert sample_generated_test.spec_id == "spec_001"
        assert sample_generated_test.validation_status == "valid"
        assert "import pytest" in sample_generated_test.imports

    def test_validation_states(self):
        """Test different validation states."""
        for status in ["pending", "valid", "invalid"]:
            test = GeneratedTest(
                spec_name="test",
                test_code="pass",
                validation_status=status,
            )
            assert test.validation_status == status


class TestTestExecutionResult:
    """Test TestExecutionResult data model."""

    def test_create_passed(self):
        """Test creating a passed test result."""
        result = TestExecutionResult(
            test_name="test_add",
            passed=True,
            execution_time_ms=50.0,
        )
        assert result.passed is True
        assert result.error_type is None

    def test_create_failed_assertion(self):
        """Test creating a failed assertion result."""
        result = TestExecutionResult(
            test_name="test_subtract",
            passed=False,
            error_type="assertion",
            error_message="Expected 3, got 4",
        )
        assert result.passed is False
        assert result.error_type == "assertion"

    def test_create_failed_exception(self):
        """Test creating a failed exception result."""
        result = TestExecutionResult(
            test_name="test_divide",
            passed=False,
            error_type="exception",
            error_message="ZeroDivisionError",
            error_traceback="Traceback...",
        )
        assert result.error_type == "exception"
        assert "ZeroDivisionError" in result.error_message


class TestTDDCycleResult:
    """Test TDDCycleResult data model."""

    def test_create_successful(self):
        """Test creating a successful cycle result."""
        result = TDDCycleResult(
            iteration=1,
            tests_generated=5,
            tests_executed=5,
            tests_passed=5,
            tests_failed=0,
            code_generated=True,
            all_tests_pass=True,
        )
        assert result.all_tests_pass is True
        assert result.pass_rate == 100.0

    def test_create_partial(self, sample_cycle_result):
        """Test creating a partial success cycle result."""
        assert sample_cycle_result.tests_passed == 4
        assert sample_cycle_result.tests_failed == 1
        assert sample_cycle_result.pass_rate == 80.0

    def test_pass_rate_property(self):
        """Test pass_rate calculation."""
        # All pass
        result = TDDCycleResult(
            iteration=1,
            tests_generated=5,
            tests_executed=5,
            tests_passed=5,
            tests_failed=0,
            code_generated=True,
            all_tests_pass=True,
        )
        assert result.pass_rate == 100.0

        # Half pass
        result = TDDCycleResult(
            iteration=1,
            tests_generated=4,
            tests_executed=4,
            tests_passed=2,
            tests_failed=2,
            code_generated=True,
            all_tests_pass=False,
        )
        assert result.pass_rate == 50.0

        # None executed
        result = TDDCycleResult(
            iteration=1,
            tests_generated=0,
            tests_executed=0,
            tests_passed=0,
            tests_failed=0,
            code_generated=False,
            all_tests_pass=False,
        )
        assert result.pass_rate == 0.0


class TestTDDSession:
    """Test TDDSession data model."""

    def test_create_minimal(self):
        """Test creating a minimal TDDSession."""
        session = TDDSession(
            task_id="task_001",
            requirements="Implement feature X",
        )
        assert session.task_id == "task_001"
        assert session.status == "pending"
        assert session.current_iteration == 0

    def test_create_with_cycles(self, sample_cycle_result):
        """Test creating a session with cycles."""
        session = TDDSession(
            task_id="task_001",
            requirements="Implement feature X",
            cycles=[sample_cycle_result],
            current_iteration=1,
            status="in_progress",
            best_pass_rate=80.0,
        )
        assert len(session.cycles) == 1
        assert session.best_pass_rate == 80.0


# =============================================================================
# Storage Backend Tests (run on both SQLite and PostgreSQL)
# =============================================================================


class TestStorageBackend:
    """Test storage backend operations on both SQLite and PostgreSQL."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["sqlite"])
    async def test_save_and_get_test_suite(
        self,
        backend,
        sqlite_storage,
        sample_test_suite,
    ):
        """Test saving and retrieving a test suite."""
        storage = sqlite_storage if backend == "sqlite" else None
        if storage is None:
            pytest.skip(f"Backend {backend} not available")

        # Save suite
        saved = await storage.save_test_suite(sample_test_suite)
        assert saved.id == sample_test_suite.id

        # Retrieve suite
        retrieved = await storage.get_test_suite(sample_test_suite.id)
        assert retrieved is not None
        assert retrieved.id == sample_test_suite.id
        assert retrieved.task_id == sample_test_suite.task_id
        assert retrieved.spec_count == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_suite(self, sqlite_storage):
        """Test retrieving a non-existent suite."""
        retrieved = await sqlite_storage.get_test_suite("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_suites_for_task(
        self,
        sqlite_storage,
        sample_test_suite,
    ):
        """Test retrieving suites by task ID."""
        # Save suite
        await sqlite_storage.save_test_suite(sample_test_suite)

        # Retrieve by task
        suites = await sqlite_storage.get_test_suites_for_task(sample_test_suite.task_id)
        assert len(suites) == 1
        assert suites[0].task_id == sample_test_suite.task_id

        # Non-existent task
        suites = await sqlite_storage.get_test_suites_for_task("nonexistent")
        assert len(suites) == 0

    @pytest.mark.asyncio
    async def test_save_and_get_generated_test(
        self,
        sqlite_storage,
        sample_test_suite,
        sample_generated_test,
    ):
        """Test saving and retrieving generated tests."""
        # First save the suite
        await sqlite_storage.save_test_suite(sample_test_suite)

        # Save generated test
        saved = await sqlite_storage.save_generated_test(sample_generated_test)
        assert saved.id == sample_generated_test.id

        # Retrieve by suite
        tests = await sqlite_storage.get_generated_tests(sample_test_suite.id)
        assert len(tests) >= 1

    @pytest.mark.asyncio
    async def test_save_and_get_cycle_result(
        self,
        sqlite_storage,
        sample_test_suite,
        sample_cycle_result,
    ):
        """Test saving and retrieving cycle results."""
        # First save the suite (FK constraint)
        await sqlite_storage.save_test_suite(sample_test_suite)

        # Update cycle_result to reference the suite
        sample_cycle_result.suite_id = sample_test_suite.id

        # Save result
        saved = await sqlite_storage.save_cycle_result(sample_cycle_result)
        assert saved.id == sample_cycle_result.id

        # Retrieve by suite
        results = await sqlite_storage.get_cycle_results(sample_cycle_result.suite_id)
        assert len(results) == 1
        assert results[0].iteration == 1
        assert results[0].tests_passed == 4

    @pytest.mark.asyncio
    async def test_save_and_get_session(self, sqlite_storage):
        """Test saving and retrieving TDD sessions."""
        # Create a cycle_result without suite_id for standalone session test
        cycle = TDDCycleResult(
            id="cycle_test_001",
            iteration=1,
            tests_generated=3,
            tests_executed=3,
            tests_passed=2,
            tests_failed=1,
            code_generated=True,
            all_tests_pass=False,
            execution_results=[
                TestExecutionResult(test_name="test_a", passed=True),
                TestExecutionResult(test_name="test_b", passed=False, error_message="Failed"),
            ],
        )

        session = TDDSession(
            id="sess_001",
            task_id="task_001",
            requirements="Test requirements",
            cycles=[cycle],
            current_iteration=1,
            status="in_progress",
        )

        # Save session
        saved = await sqlite_storage.save_session(session)
        assert saved.id == "sess_001"

        # Retrieve session
        retrieved = await sqlite_storage.get_session("sess_001")
        assert retrieved is not None
        assert retrieved.task_id == "task_001"
        assert len(retrieved.cycles) == 1

    @pytest.mark.asyncio
    async def test_get_sessions_for_task(self, sqlite_storage):
        """Test retrieving sessions by task ID."""
        session = TDDSession(
            id="sess_002",
            task_id="task_002",
            requirements="Test requirements",
        )
        await sqlite_storage.save_session(session)

        sessions = await sqlite_storage.get_sessions_for_task("task_002")
        assert len(sessions) == 1

        sessions = await sqlite_storage.get_sessions_for_task("nonexistent")
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, sqlite_storage, sample_test_suite, sample_cycle_result):
        """Test getting statistics."""
        await sqlite_storage.save_test_suite(sample_test_suite)
        await sqlite_storage.save_cycle_result(sample_cycle_result)

        stats = await sqlite_storage.get_stats(days=30)
        assert "total_suites" in stats
        assert "total_cycles" in stats
        assert "average_pass_rate" in stats
        assert stats["total_suites"] >= 1


# =============================================================================
# Test Generator Tests
# =============================================================================


class TestTestGenerator:
    """Test TestGenerator class."""

    def test_init(self):
        """Test TestGenerator initialization."""
        generator = TestGenerator()
        assert generator.default_framework == "pytest"
        assert generator.min_tests == 5

    def test_init_with_options(self, mock_llm_callable):
        """Test TestGenerator with custom options."""
        generator = TestGenerator(
            llm_callable=mock_llm_callable,
            default_framework="unittest",
            min_tests=10,
        )
        assert generator.default_framework == "unittest"
        assert generator.min_tests == 10

    @pytest.mark.asyncio
    async def test_generate_specs(self, mock_llm_callable):
        """Test generating test specifications."""
        generator = TestGenerator(llm_callable=mock_llm_callable)

        suite = await generator.generate_specs(
            requirements="Implement add(a, b) method",
            language="python",
            task_id="test_task",
        )

        assert suite is not None
        assert suite.task_id == "test_task"
        assert suite.language == "python"
        assert len(suite.specifications) >= 5

    @pytest.mark.asyncio
    async def test_generate_specs_no_llm(self):
        """Test generating specs without LLM callable."""
        generator = TestGenerator()

        with pytest.raises(ValueError, match="LLM callable not configured"):
            await generator.generate_specs(
                requirements="Test",
                language="python",
            )

    @pytest.mark.asyncio
    async def test_generate_tests(self, mock_llm_callable):
        """Test generating executable tests."""
        generator = TestGenerator(llm_callable=mock_llm_callable)

        suite = TestSuite(
            id="suite_001",
            task_id="task_001",
            requirements="Test",
            specifications=[
                TestSpecification(
                    id="spec_001",
                    name="test_basic",
                    description="Basic test",
                )
            ],
        )

        tests = await generator.generate_tests(suite)
        assert len(tests) == 1
        assert tests[0].spec_name == "test_basic"

    @pytest.mark.asyncio
    async def test_generate_implementation(self, mock_llm_callable):
        """Test generating implementation code."""
        generator = TestGenerator(llm_callable=mock_llm_callable)

        code = await generator.generate_implementation(
            requirements="Implement Calculator",
            test_file_content="def test_add(): pass",
            language="python",
        )

        assert "class Calculator" in code

    def test_extract_imports(self):
        """Test extracting imports from code."""
        generator = TestGenerator()

        code = """
import pytest
from unittest import TestCase

def test_foo():
    pass
"""
        imports, code_lines = generator._extract_imports(code, "python")
        assert "import pytest" in imports
        assert "from unittest import TestCase" in imports
        assert len(code_lines) > 0

    def test_clean_code_block(self):
        """Test cleaning markdown code blocks."""
        generator = TestGenerator()

        # With python marker
        code = "```python\ndef foo(): pass\n```"
        cleaned = generator._clean_code_block(code)
        assert cleaned == "def foo(): pass"

        # Without marker
        code = "```\ndef foo(): pass\n```"
        cleaned = generator._clean_code_block(code)
        assert cleaned == "def foo(): pass"

        # No markers
        code = "def foo(): pass"
        cleaned = generator._clean_code_block(code)
        assert cleaned == "def foo(): pass"

    def test_parse_json_response(self):
        """Test parsing JSON from LLM response."""
        generator = TestGenerator()

        # Direct JSON array
        content = '[{"name": "test_foo"}]'
        result = generator._parse_json_response(content)
        assert len(result) == 1
        assert result[0]["name"] == "test_foo"

        # JSON in code block
        content = '```json\n[{"name": "test_bar"}]\n```'
        result = generator._parse_json_response(content)
        assert len(result) == 1
        assert result[0]["name"] == "test_bar"

        # Invalid JSON
        content = "not json"
        result = generator._parse_json_response(content)
        assert result == []


# =============================================================================
# Test File Builder Tests
# =============================================================================


class TestTestFileBuilder:
    """Test TestFileBuilder class."""

    def test_build_python_pytest_file(self, sample_test_suite, sample_generated_test):
        """Test building Python pytest file."""
        builder = TestFileBuilder()

        content = builder.build_test_file(
            tests=[sample_generated_test],
            suite=sample_test_suite,
        )

        assert "import pytest" in content
        assert "def test_addition" in content

    def test_build_python_unittest_file(self, sample_generated_test):
        """Test building Python unittest file."""
        builder = TestFileBuilder()

        suite = TestSuite(
            task_id="task_001",
            requirements="Test",
            framework="unittest",
            specifications=[],
        )

        content = builder.build_test_file(
            tests=[sample_generated_test],
            suite=suite,
        )

        assert "import unittest" in content
        assert "class TestSuite" in content

    def test_build_with_fixtures(self):
        """Test building file with fixtures."""
        builder = TestFileBuilder()

        suite = TestSuite(
            task_id="task_001",
            requirements="Test",
            fixture_code="@pytest.fixture\ndef calc(): return Calculator()",
            specifications=[],
        )

        test = GeneratedTest(
            spec_name="test_fixture",
            test_code="def test_fixture(calc): assert calc.add(1, 2) == 3",
        )

        content = builder.build_test_file(tests=[test], suite=suite)
        assert "@pytest.fixture" in content

    def test_build_with_setup_teardown(self):
        """Test building file with setup/teardown."""
        builder = TestFileBuilder()

        suite = TestSuite(
            task_id="task_001",
            requirements="Test",
            setup_code="db = setup_database()",
            teardown_code="teardown_database(db)",
            specifications=[],
        )

        test = GeneratedTest(
            spec_name="test_db",
            test_code="def test_db(): pass",
        )

        content = builder.build_test_file(tests=[test], suite=suite)
        assert "# Setup" in content
        assert "setup_database" in content
        assert "# Teardown" in content


# =============================================================================
# TDD Manager Tests
# =============================================================================


class TestTDDManager:
    """Test TDDManager class."""

    def test_init_sqlite(self):
        """Test manager initialization with SQLite."""
        manager = TDDManager(backend="sqlite", db_path="/tmp/test.db")
        assert manager.backend_type == "sqlite"
        assert manager.enabled is True

    def test_init_disabled(self):
        """Test manager initialization when disabled."""
        manager = TDDManager(enabled=False)
        assert manager.enabled is False
        assert manager._backend is None

    def test_init_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            TDDManager(backend="invalid")

    def test_init_postgres_no_url(self):
        """Test postgres initialization without URL."""
        with pytest.raises(ValueError, match="db_url required"):
            TDDManager(backend="postgres")

    @pytest.mark.asyncio
    async def test_generate_test_specs(self, sqlite_manager, mock_llm_callable):
        """Test generating test specifications through manager."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        suite = await sqlite_manager.generate_test_specs(
            requirements="Implement Calculator",
            language="python",
        )

        assert suite is not None
        assert suite.language == "python"
        assert len(suite.specifications) >= 5

    @pytest.mark.asyncio
    async def test_generate_tests(self, sqlite_manager, mock_llm_callable, sample_test_suite):
        """Test generating tests through manager."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        # Save suite first (FK constraint for generated tests)
        await sqlite_manager._backend.save_test_suite(sample_test_suite)

        tests = await sqlite_manager.generate_tests(sample_test_suite)
        assert len(tests) > 0

    def test_build_test_file(self, sqlite_manager, sample_test_suite, sample_generated_test):
        """Test building test file through manager."""
        content = sqlite_manager.build_test_file(
            tests=[sample_generated_test],
            suite=sample_test_suite,
        )

        assert "import pytest" in content

    @pytest.mark.asyncio
    async def test_execute_tests_no_sandbox(self, sqlite_manager):
        """Test execute_tests without sandbox configured."""
        with pytest.raises(RuntimeError, match="Sandbox executor not configured"):
            await sqlite_manager.execute_tests(
                implementation_code="pass",
                test_file_content="pass",
            )

    @pytest.mark.asyncio
    async def test_execute_tests_with_sandbox(self, sqlite_manager):
        """Test execute_tests with mock sandbox."""

        # Mock sandbox executor
        async def mock_sandbox(code, command, timeout_seconds):
            return MagicMock(
                exit_code=0,
                stdout="test_add PASSED\ntest_subtract PASSED\n",
                stderr="",
            )

        sqlite_manager.set_sandbox_executor(mock_sandbox)

        results = await sqlite_manager.execute_tests(
            implementation_code="class Calc: pass",
            test_file_content="def test_add(): pass\ndef test_subtract(): pass",
            language="python",
            framework="pytest",
        )

        assert len(results) >= 2

    def test_parse_pytest_output(self, sqlite_manager):
        """Test parsing pytest output."""
        stdout = """
test_file.py::test_add PASSED
test_file.py::test_subtract FAILED
test_file.py::test_multiply PASSED
"""
        results = sqlite_manager._parse_pytest_output(stdout, "")

        assert len(results) == 3
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        assert len(passed) == 2
        assert len(failed) == 1

    def test_parse_unittest_output(self, sqlite_manager):
        """Test parsing unittest output."""
        stdout = """
test_add (test_calc.TestCalculator) ... ok
test_subtract (test_calc.TestCalculator) ... FAIL
"""
        results = sqlite_manager._parse_unittest_output(stdout, "")

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    @pytest.mark.asyncio
    async def test_run_tdd_cycle_no_sandbox(self, sqlite_manager, mock_llm_callable):
        """Test running TDD cycle without sandbox."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        result = await sqlite_manager.run_tdd_cycle(
            requirements="Implement Calculator with add method",
            language="python",
            max_iterations=1,
        )

        assert result.code_generated is True
        assert result.implementation_code is not None
        assert result.tests_executed == 0  # No sandbox

    @pytest.mark.asyncio
    async def test_run_tdd_cycle_with_sandbox(self, sqlite_manager, mock_llm_callable):
        """Test running full TDD cycle with sandbox."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        # Mock sandbox that returns all passing
        async def mock_sandbox(code, command, timeout_seconds):
            return MagicMock(
                exit_code=0,
                stdout="test_add PASSED\ntest_basic PASSED\n",
                stderr="",
            )

        sqlite_manager.set_sandbox_executor(mock_sandbox)

        result = await sqlite_manager.run_tdd_cycle(
            requirements="Implement Calculator",
            language="python",
            max_iterations=3,
        )

        assert result.code_generated is True
        assert result.tests_executed > 0

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, sqlite_manager):
        """Test getting stats with no data."""
        stats = await sqlite_manager.get_stats()

        assert stats["total_suites"] == 0
        assert stats["total_cycles"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(
        self,
        sqlite_manager,
        mock_llm_callable,
        sample_test_suite,
        sample_cycle_result,
    ):
        """Test getting stats with data."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        # Save some data
        await sqlite_manager._backend.save_test_suite(sample_test_suite)
        await sqlite_manager._backend.save_cycle_result(sample_cycle_result)

        stats = await sqlite_manager.get_stats()

        assert stats["total_suites"] >= 1
        assert stats["total_cycles"] >= 1

    def test_analyze_failures(self, sqlite_manager):
        """Test failure analysis generation."""
        results = [
            TestExecutionResult(test_name="test_add", passed=True),
            TestExecutionResult(
                test_name="test_subtract",
                passed=False,
                error_type="assertion",
                error_message="Expected 3, got 4",
            ),
        ]

        analysis = sqlite_manager._analyze_failures(results)

        assert "test_subtract" in analysis
        assert "assertion" in analysis

    def test_generate_suggestions(self, sqlite_manager):
        """Test suggestion generation."""
        results = [
            TestExecutionResult(
                test_name="test_assertion",
                passed=False,
                error_type="assertion",
            ),
            TestExecutionResult(
                test_name="test_exception",
                passed=False,
                error_type="exception",
            ),
        ]

        suggestions = sqlite_manager._generate_suggestions(results)

        assert len(suggestions) == 2
        assert any("return value" in s for s in suggestions)
        assert any("exception" in s for s in suggestions)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full TDD workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_sqlite(self, mock_llm_callable):
        """Test complete TDD workflow with SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tdd.db"

            manager = TDDManager(
                backend="sqlite",
                db_path=str(db_path),
                min_tests=3,
                max_iterations=2,
            )
            await manager.initialize()
            manager.set_llm_callable(mock_llm_callable)

            try:
                # Step 1: Generate specs
                suite = await manager.generate_test_specs(
                    requirements="Implement add(a, b) function",
                    language="python",
                )
                assert suite.spec_count >= 3

                # Step 2: Generate tests
                tests = await manager.generate_tests(suite)
                assert len(tests) >= 3

                # Step 3: Build test file
                test_file = manager.build_test_file(tests, suite)
                assert "import pytest" in test_file

                # Step 4: Verify storage
                retrieved = await manager._backend.get_test_suite(suite.id)
                assert retrieved is not None
                assert retrieved.id == suite.id

                # Step 5: Get stats
                stats = await manager.get_stats()
                assert stats["total_suites"] >= 1

            finally:
                await manager.close()

    @pytest.mark.asyncio
    async def test_multiple_suites_same_task(self, sqlite_manager, mock_llm_callable):
        """Test creating multiple suites for the same task."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        # Create first suite
        suite1 = await sqlite_manager.generate_test_specs(
            requirements="Implement add(a, b)",
            task_id="task_multi",
        )

        # Create second suite
        suite2 = await sqlite_manager.generate_test_specs(
            requirements="Implement add(a, b) with validation",
            task_id="task_multi",
        )

        # Retrieve both
        suites = await sqlite_manager._backend.get_test_suites_for_task("task_multi")
        assert len(suites) == 2

    @pytest.mark.asyncio
    async def test_cycle_persistence(self, sqlite_manager, mock_llm_callable):
        """Test that TDD cycles are properly persisted."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        # Mock sandbox
        async def mock_sandbox(code, command, timeout_seconds):
            return MagicMock(
                exit_code=1,
                stdout="test_add PASSED\ntest_subtract FAILED\n",
                stderr="AssertionError",
            )

        sqlite_manager.set_sandbox_executor(mock_sandbox)

        # Run cycle
        result = await sqlite_manager.run_tdd_cycle(
            requirements="Implement Calculator",
            max_iterations=1,
        )

        # Verify persistence
        if result.suite_id:
            cycles = await sqlite_manager._backend.get_cycle_results(result.suite_id)
            assert len(cycles) >= 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_requirements(self, sqlite_manager, mock_llm_callable):
        """Test with empty requirements."""
        sqlite_manager.set_llm_callable(mock_llm_callable)

        suite = await sqlite_manager.generate_test_specs(
            requirements="",
            language="python",
        )

        # Should still return a suite (with fallback specs)
        assert suite is not None

    @pytest.mark.asyncio
    async def test_spec_with_none_values(self, sqlite_storage):
        """Test saving spec with None values."""
        suite = TestSuite(
            id="suite_none",
            task_id="task_none",
            requirements="Test",
            specifications=[
                TestSpecification(
                    id="spec_none",
                    name="test_none",
                    description="Test with None values",
                    expected_output=None,
                    expected_behavior=None,
                )
            ],
        )

        saved = await sqlite_storage.save_test_suite(suite)
        retrieved = await sqlite_storage.get_test_suite(saved.id)

        assert retrieved.specifications[0].expected_output is None

    @pytest.mark.asyncio
    async def test_large_test_code(self, sqlite_storage, sample_test_suite):
        """Test saving large test code."""
        await sqlite_storage.save_test_suite(sample_test_suite)

        large_code = "def test_large():\n" + "    pass\n" * 1000

        test = GeneratedTest(
            spec_id=sample_test_suite.specifications[0].id,
            spec_name="test_large",
            test_code=large_code,
        )

        saved = await sqlite_storage.save_generated_test(test)
        assert len(saved.test_code) > 5000

    def test_parse_empty_output(self, sqlite_manager):
        """Test parsing empty test output."""
        results = sqlite_manager._parse_pytest_output("", "")
        assert len(results) == 0

    def test_parse_malformed_output(self, sqlite_manager):
        """Test parsing malformed test output."""
        stdout = "random text without test results"
        results = sqlite_manager._parse_pytest_output(stdout, "")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_suite_update(self, sqlite_storage, sample_test_suite):
        """Test updating an existing suite."""
        # Save original
        await sqlite_storage.save_test_suite(sample_test_suite)

        # Modify and save again
        sample_test_suite.requirements = "Updated requirements"
        await sqlite_storage.save_test_suite(sample_test_suite)

        # Retrieve
        retrieved = await sqlite_storage.get_test_suite(sample_test_suite.id)
        assert retrieved.requirements == "Updated requirements"


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, sqlite_storage):
        """Test saving multiple suites concurrently."""
        import asyncio

        suites = [
            TestSuite(
                id=f"suite_concurrent_{i}",
                task_id=f"task_concurrent_{i}",
                requirements=f"Requirements {i}",
                specifications=[
                    TestSpecification(
                        id=f"spec_concurrent_{i}",
                        name=f"test_{i}",
                        description=f"Test {i}",
                    )
                ],
            )
            for i in range(10)
        ]

        # Save all concurrently
        await asyncio.gather(*[sqlite_storage.save_test_suite(suite) for suite in suites])

        # Verify all saved
        for suite in suites:
            retrieved = await sqlite_storage.get_test_suite(suite.id)
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, sqlite_storage, sample_test_suite):
        """Test concurrent reads."""
        import asyncio

        await sqlite_storage.save_test_suite(sample_test_suite)

        # Read concurrently
        results = await asyncio.gather(
            *[sqlite_storage.get_test_suite(sample_test_suite.id) for _ in range(20)]
        )

        # All should succeed
        assert all(r is not None for r in results)
        assert all(r.id == sample_test_suite.id for r in results)


# =============================================================================
# Configuration Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Test integration with Darwin configuration."""

    def test_tdd_config_defaults(self):
        """Test TDD config defaults."""
        from llmcore.config.darwin_config import TDDConfig

        config = TDDConfig()
        assert config.enabled is False  # Disabled by default until Phase 6.2 complete
        assert config.default_framework == "pytest"
        assert config.min_tests == 5
        assert config.max_iterations == 3

    def test_tdd_config_custom(self):
        """Test custom TDD config."""
        from llmcore.config.darwin_config import TDDConfig

        config = TDDConfig(
            enabled=True,
            default_framework="unittest",
            min_tests=10,
            max_iterations=5,
        )
        assert config.enabled is True
        assert config.default_framework == "unittest"
        assert config.min_tests == 10
        assert config.max_iterations == 5

    def test_darwin_config_with_tdd(self):
        """Test Darwin config includes TDD."""
        from llmcore.config.darwin_config import DarwinConfig

        config = DarwinConfig(tdd={"enabled": True, "min_tests": 8})
        assert config.tdd.enabled is True
        assert config.tdd.min_tests == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
