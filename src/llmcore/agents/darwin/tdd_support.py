# src/llmcore/agents/darwin/tdd_support.py
"""
TDD (Test-Driven Development) Support for Darwin Agents.

This module enables test-first development workflow for code generation tasks.
It produces higher quality, more verifiable code by:

1. Generating test specifications based on requirements
2. Creating executable test cases before implementation
3. Generating code that should pass the tests
4. Executing tests in sandbox and iterating on failures

Workflow:
    REQUIREMENTS → TEST_SPECS → TEST_CODE → IMPLEMENTATION → EXECUTE → [ITERATE]

Supported Backends:
    - SQLite: Local file-based storage for development and single-user deployments
    - PostgreSQL: Production-grade storage for multi-user and high-concurrency scenarios

Architecture:
    - BaseTDDStorage: Abstract interface for TDD storage backends
    - TDDManager: High-level API that delegates to the appropriate backend
    - TestGenerator: Generates test specs and executable tests from requirements
    - Backend implementations: SqliteTDDStorage, PostgresTDDStorage

Usage:
    from llmcore.agents.darwin.tdd_support import TDDManager, TestSpecification

    # Initialize with auto-detection from config
    manager = TDDManager(backend="sqlite", db_path="~/.local/share/llmcore/tdd.db")
    await manager.initialize()

    # Generate test specifications
    suite = await manager.generate_test_specs(
        requirements="Implement a Calculator class with add, subtract, multiply, divide methods",
        language="python",
        framework="pytest",
    )

    # Generate executable tests
    tests = await manager.generate_tests(suite)

    # Execute tests against implementation
    results = await manager.execute_tests(
        implementation_code="class Calculator: ...",
        test_suite=suite,
    )

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 6.2
    - Phase 6.1: Failure Learning System (pattern reference)
"""

from __future__ import annotations

import abc
import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class TestSpecification(BaseModel):
    """
    Specification for a single test case.

    Describes what should be tested without containing actual test code.
    Used as input for test code generation.

    Attributes:
        id: Unique identifier for this test spec
        name: Test function name (should start with test_)
        description: Human-readable description of what the test verifies
        test_type: Category of test (unit, integration, edge_case, error)
        inputs: Input values for the test
        expected_output: Expected return value or state
        expected_behavior: Description of expected side effects
        expected_exception: Exception type expected (for error tests)
        priority: 1=must pass, 2=should pass, 3=nice to have
        tags: Categorization tags for filtering
        created_at: When this spec was created
    """

    id: Optional[str] = None
    name: str
    description: str
    test_type: Literal["unit", "integration", "edge_case", "error"] = "unit"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[Any] = None
    expected_behavior: Optional[str] = None
    expected_exception: Optional[str] = None
    priority: int = Field(default=2, ge=1, le=3)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class TestSuite(BaseModel):
    """
    Collection of test specifications for a task.

    Represents a complete test suite with configuration and specifications.

    Attributes:
        id: Unique identifier for this suite
        task_id: ID of the associated task
        requirements: Original requirements text
        language: Programming language (python, javascript, etc.)
        framework: Test framework to use (pytest, unittest, jest, etc.)
        specifications: List of test specifications
        setup_code: Optional setup code to run before tests
        teardown_code: Optional teardown code to run after tests
        fixture_code: Optional fixture definitions
        import_statements: Required imports for test file
        created_at: When this suite was created
        updated_at: Last update timestamp
    """

    id: Optional[str] = None
    task_id: str
    requirements: str
    language: Literal["python", "javascript", "typescript"] = "python"
    framework: Literal["pytest", "unittest", "jest", "mocha"] = "pytest"
    specifications: List[TestSpecification] = Field(default_factory=list)
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    fixture_code: Optional[str] = None
    import_statements: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    @property
    def spec_count(self) -> int:
        """Get the number of test specifications."""
        return len(self.specifications)

    def get_specs_by_type(self, test_type: str) -> List[TestSpecification]:
        """Get specifications filtered by test type."""
        return [s for s in self.specifications if s.test_type == test_type]

    def get_specs_by_priority(self, priority: int) -> List[TestSpecification]:
        """Get specifications filtered by priority."""
        return [s for s in self.specifications if s.priority == priority]


class GeneratedTest(BaseModel):
    """
    A generated executable test.

    Contains the actual test code generated from a specification.

    Attributes:
        id: Unique identifier for this generated test
        spec_id: ID of the source TestSpecification
        spec_name: Name of the test (from spec)
        test_code: Generated test function code
        imports: Required import statements
        fixtures: Fixture code if needed
        validation_status: Whether the code was validated (syntax check)
        validation_errors: Any errors found during validation
        created_at: When this test was generated
    """

    id: Optional[str] = None
    spec_id: Optional[str] = None
    spec_name: str
    test_code: str
    imports: List[str] = Field(default_factory=list)
    fixtures: Optional[str] = None
    validation_status: Literal["pending", "valid", "invalid"] = "pending"
    validation_errors: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class TestExecutionResult(BaseModel):
    """
    Result of running a single test.

    Captures the outcome of executing one test case.

    Attributes:
        test_name: Name of the test that was run
        passed: Whether the test passed
        error_type: Type of error if failed (assertion, exception, etc.)
        error_message: Error message if failed
        error_traceback: Full traceback if available
        execution_time_ms: How long the test took in milliseconds
        stdout: Standard output captured during test
        stderr: Standard error captured during test
        metadata: Additional execution metadata
    """

    test_name: str
    passed: bool
    error_type: Optional[Literal["assertion", "exception", "timeout", "syntax"]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    execution_time_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict()


class TDDCycleResult(BaseModel):
    """
    Result of a complete TDD cycle iteration.

    Captures the outcome of one full generate-test-iterate cycle.

    Attributes:
        id: Unique identifier for this cycle result
        suite_id: ID of the test suite used
        iteration: Which iteration this was (1-based)
        tests_generated: Number of tests that were generated
        tests_executed: Number of tests that were executed
        tests_passed: Number of tests that passed
        tests_failed: Number of tests that failed
        code_generated: Whether implementation code was generated
        implementation_code: The generated implementation
        final_code: The final (possibly refined) implementation
        all_tests_pass: Whether all tests now pass
        execution_results: Individual test execution results
        failure_analysis: Analysis of what failed and why
        suggestions: Suggestions for fixing failures
        total_time_ms: Total time for this cycle
        created_at: When this cycle was executed
    """

    id: Optional[str] = None
    suite_id: Optional[str] = None
    iteration: int
    tests_generated: int
    tests_executed: int
    tests_passed: int
    tests_failed: int
    code_generated: bool
    implementation_code: Optional[str] = None
    final_code: Optional[str] = None
    all_tests_pass: bool
    execution_results: List[TestExecutionResult] = Field(default_factory=list)
    failure_analysis: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    total_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        if self.tests_executed == 0:
            return 0.0
        return (self.tests_passed / self.tests_executed) * 100.0


class TDDSession(BaseModel):
    """
    Persistent session tracking multiple TDD cycles.

    Tracks the full history of a TDD session across multiple iterations.

    Attributes:
        id: Unique session identifier
        task_id: Associated task ID
        requirements: Original requirements
        language: Programming language
        framework: Test framework
        suite_id: ID of the test suite
        cycles: History of TDD cycle results
        current_iteration: Current iteration number
        status: Session status
        best_pass_rate: Best pass rate achieved
        final_implementation: Final implementation if successful
        started_at: When session started
        completed_at: When session completed (if done)
    """

    id: Optional[str] = None
    task_id: str
    requirements: str
    language: str = "python"
    framework: str = "pytest"
    suite_id: Optional[str] = None
    cycles: List[TDDCycleResult] = Field(default_factory=list)
    current_iteration: int = 0
    status: Literal["pending", "in_progress", "success", "failed", "abandoned"] = "pending"
    best_pass_rate: float = 0.0
    final_implementation: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


# Rebuild models for Pydantic v2 with future annotations
TestSpecification.model_rebuild()
TestSuite.model_rebuild()
GeneratedTest.model_rebuild()
TestExecutionResult.model_rebuild()
TDDCycleResult.model_rebuild()
TDDSession.model_rebuild()


# =============================================================================
# BASE STORAGE INTERFACE
# =============================================================================


class BaseTDDStorage(abc.ABC):
    """
    Abstract base class for TDD storage backends.

    Defines the interface that all TDD storage implementations must adhere to.
    Concrete implementations handle the specifics of storing data in SQLite,
    PostgreSQL, or other backends.
    """

    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the storage backend with given configuration.

        Args:
            config: Backend-specific configuration dictionary

        Raises:
            ConfigError: If configuration is invalid
            StorageError: If initialization fails
        """
        pass

    @abc.abstractmethod
    async def save_test_suite(self, suite: TestSuite) -> TestSuite:
        """
        Persist a test suite to storage.

        Args:
            suite: The test suite to save

        Returns:
            The saved suite with ID and computed fields set

        Raises:
            StorageError: If saving fails
        """
        pass

    @abc.abstractmethod
    async def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """
        Retrieve a specific test suite by ID.

        Args:
            suite_id: The suite ID to retrieve

        Returns:
            The TestSuite if found, None otherwise
        """
        pass

    @abc.abstractmethod
    async def get_test_suites_for_task(self, task_id: str) -> List[TestSuite]:
        """
        Retrieve all test suites for a task.

        Args:
            task_id: The task ID to filter by

        Returns:
            List of TestSuite objects for the task
        """
        pass

    @abc.abstractmethod
    async def save_generated_test(self, test: GeneratedTest) -> GeneratedTest:
        """
        Persist a generated test to storage.

        Args:
            test: The generated test to save

        Returns:
            The saved test with ID set
        """
        pass

    @abc.abstractmethod
    async def get_generated_tests(self, suite_id: str) -> List[GeneratedTest]:
        """
        Retrieve all generated tests for a suite.

        Args:
            suite_id: The suite ID to filter by

        Returns:
            List of GeneratedTest objects
        """
        pass

    @abc.abstractmethod
    async def save_cycle_result(self, result: TDDCycleResult) -> TDDCycleResult:
        """
        Persist a TDD cycle result to storage.

        Args:
            result: The cycle result to save

        Returns:
            The saved result with ID set
        """
        pass

    @abc.abstractmethod
    async def get_cycle_results(
        self,
        suite_id: str,
        limit: int = 10,
    ) -> List[TDDCycleResult]:
        """
        Retrieve cycle results for a suite.

        Args:
            suite_id: The suite ID to filter by
            limit: Maximum number of results to return

        Returns:
            List of TDDCycleResult objects
        """
        pass

    @abc.abstractmethod
    async def save_session(self, session: TDDSession) -> TDDSession:
        """
        Persist a TDD session to storage.

        Args:
            session: The session to save

        Returns:
            The saved session with ID set
        """
        pass

    @abc.abstractmethod
    async def get_session(self, session_id: str) -> Optional[TDDSession]:
        """
        Retrieve a specific session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The TDDSession if found, None otherwise
        """
        pass

    @abc.abstractmethod
    async def get_sessions_for_task(self, task_id: str) -> List[TDDSession]:
        """
        Retrieve all sessions for a task.

        Args:
            task_id: The task ID to filter by

        Returns:
            List of TDDSession objects
        """
        pass

    @abc.abstractmethod
    async def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get TDD statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with statistics
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources used by the storage backend."""
        pass


# =============================================================================
# TEST GENERATOR
# =============================================================================


class TestGenerator:
    """
    Generates test specifications and executable tests from requirements.

    This class uses LLM capabilities to:
    1. Analyze requirements and extract testable behaviors
    2. Generate comprehensive test specifications
    3. Create executable test code from specifications

    Usage:
        generator = TestGenerator(llm_callable)

        # Generate test specs from requirements
        suite = await generator.generate_specs(
            requirements="Implement a UserService class with login() and logout() methods",
            language="python",
        )

        # Generate executable tests
        tests = await generator.generate_tests(suite)
    """

    SPEC_GENERATION_PROMPT = """You are an expert test engineer. Generate comprehensive test specifications for the following requirements.

Requirements:
{requirements}

Language: {language}
Test Framework: {framework}
Minimum Tests: {min_tests}

Generate test specifications covering:
1. Happy path / normal operation (at least 2 tests)
2. Edge cases (empty inputs, large inputs, boundary values) (at least 2 tests)
3. Error handling (invalid inputs, exceptions) (at least 1 test)
4. Integration scenarios (if applicable)

For each test, provide a JSON object with:
- name: A descriptive test name starting with test_ (e.g., test_add_positive_numbers)
- description: What the test verifies
- test_type: One of "unit", "integration", "edge_case", "error"
- inputs: The test inputs as a dict (e.g., {{"a": 1, "b": 2}})
- expected_output: What the function should return (can be null for void functions)
- expected_behavior: Any side effects or state changes to verify
- expected_exception: Exception type if testing error handling (e.g., "ValueError")
- priority: 1 (must pass), 2 (should pass), or 3 (nice to have)

Output as a JSON array of test specifications. Output ONLY valid JSON, no explanations or markdown.
"""

    TEST_GENERATION_PROMPT = """Generate executable test code for the following specification.

Specification:
- Name: {name}
- Description: {description}
- Type: {test_type}
- Inputs: {inputs}
- Expected Output: {expected_output}
- Expected Behavior: {expected_behavior}
- Expected Exception: {expected_exception}

Language: {language}
Framework: {framework}

Generate a complete, runnable test function. Include:
1. All necessary imports at the top
2. Any required fixtures or setup
3. The test function with clear assertions
4. Helpful error messages on assertion failure

For pytest:
- Use pytest.raises() for exception testing
- Use descriptive assertion messages
- Use fixtures where appropriate

Output only the Python code, no explanations or markdown backticks.
"""

    IMPLEMENTATION_PROMPT = """Generate implementation code that passes the following tests.

Requirements:
{requirements}

Test File Content:
```{language}
{test_file}
```

Previous Implementation (if any):
{previous_implementation}

Test Failures (if any):
{test_failures}

Generate implementation code that:
1. Satisfies all the requirements
2. Passes all the tests above
3. Follows best practices for {language}
4. Includes proper error handling
5. Has clear docstrings/comments

Output only the implementation code, no explanations or markdown backticks.
"""

    def __init__(
        self,
        llm_callable: Optional[Callable] = None,
        default_framework: str = "pytest",
        min_tests: int = 5,
    ):
        """
        Initialize test generator.

        Args:
            llm_callable: Async callable for LLM generation (takes messages, returns response)
            default_framework: Default test framework to use
            min_tests: Minimum number of tests to generate
        """
        self._llm_callable = llm_callable
        self.default_framework = default_framework
        self.min_tests = min_tests

    def set_llm_callable(self, llm_callable: Callable) -> None:
        """Set the LLM callable for generation."""
        self._llm_callable = llm_callable

    async def generate_specs(
        self,
        requirements: str,
        language: str = "python",
        framework: Optional[str] = None,
        min_tests: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> TestSuite:
        """
        Generate test specifications from requirements.

        Args:
            requirements: What to implement/test
            language: Programming language
            framework: Test framework (defaults to instance default)
            min_tests: Minimum tests to generate (defaults to instance default)
            task_id: Optional task ID for tracking

        Returns:
            TestSuite with generated specifications

        Raises:
            ValueError: If no LLM callable is configured
            RuntimeError: If generation fails
        """
        if not self._llm_callable:
            raise ValueError("LLM callable not configured. Call set_llm_callable() first.")

        framework = framework or self.default_framework
        min_tests = min_tests or self.min_tests
        task_id = task_id or f"task_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:18]}"

        prompt = self.SPEC_GENERATION_PROMPT.format(
            requirements=requirements,
            language=language,
            framework=framework,
            min_tests=min_tests,
        )

        try:
            response = await self._llm_callable(
                [
                    {
                        "role": "system",
                        "content": "You are an expert test engineer. Output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Extract content from response
            content = self._extract_content(response)

            # Parse JSON response
            specs_data = self._parse_json_response(content)

            specifications = []
            for i, spec in enumerate(specs_data):
                spec_id = f"spec_{task_id}_{i:03d}"
                specifications.append(
                    TestSpecification(
                        id=spec_id,
                        name=spec.get("name", f"test_generated_{i}"),
                        description=spec.get("description", "Generated test"),
                        test_type=spec.get("test_type", "unit"),
                        inputs=spec.get("inputs", {}),
                        expected_output=spec.get("expected_output"),
                        expected_behavior=spec.get("expected_behavior"),
                        expected_exception=spec.get("expected_exception"),
                        priority=spec.get("priority", 2),
                        tags=spec.get("tags", []),
                    )
                )

            # Ensure minimum tests
            while len(specifications) < min_tests:
                idx = len(specifications)
                specifications.append(
                    TestSpecification(
                        id=f"spec_{task_id}_{idx:03d}",
                        name=f"test_basic_{idx}",
                        description="Basic functionality test",
                        test_type="unit",
                        inputs={},
                        expected_output=None,
                        priority=2,
                    )
                )

            # Include timestamp for unique suite IDs even with same task_id
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:18]
            suite_id = f"suite_{task_id}_{timestamp}"
            return TestSuite(
                id=suite_id,
                task_id=task_id,
                requirements=requirements,
                language=language,
                framework=framework,
                specifications=specifications,
            )

        except Exception as e:
            logger.error(f"Failed to generate test specs: {e}")
            # Return minimal test suite as fallback
            return TestSuite(
                id=f"suite_{task_id}_fallback",
                task_id=task_id,
                requirements=requirements,
                language=language,
                framework=framework,
                specifications=[
                    TestSpecification(
                        id=f"spec_{task_id}_000",
                        name="test_basic",
                        description="Basic functionality test (fallback)",
                        test_type="unit",
                        inputs={},
                        expected_output=None,
                        priority=1,
                    )
                ],
            )

    async def generate_tests(
        self,
        suite: TestSuite,
    ) -> List[GeneratedTest]:
        """
        Generate executable test code from specifications.

        Args:
            suite: Test suite with specifications

        Returns:
            List of GeneratedTest objects
        """
        if not self._llm_callable:
            raise ValueError("LLM callable not configured. Call set_llm_callable() first.")

        tests = []

        for spec in suite.specifications:
            prompt = self.TEST_GENERATION_PROMPT.format(
                name=spec.name,
                description=spec.description,
                test_type=spec.test_type,
                inputs=json.dumps(spec.inputs),
                expected_output=json.dumps(spec.expected_output)
                if spec.expected_output is not None
                else "None",
                expected_behavior=spec.expected_behavior or "N/A",
                expected_exception=spec.expected_exception or "None",
                language=suite.language,
                framework=suite.framework,
            )

            try:
                response = await self._llm_callable(
                    [
                        {
                            "role": "system",
                            "content": f"You are an expert {suite.language} developer. Output only code.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )

                # Extract code
                code = self._extract_content(response)
                code = self._clean_code_block(code)

                # Extract imports
                imports, code_lines = self._extract_imports(code, suite.language)

                test_id = (
                    f"gen_{spec.id}"
                    if spec.id
                    else f"gen_{spec.name}_{datetime.utcnow().strftime('%H%M%S')}"
                )

                tests.append(
                    GeneratedTest(
                        id=test_id,
                        spec_id=spec.id,
                        spec_name=spec.name,
                        test_code="\n".join(code_lines),
                        imports=imports,
                        validation_status="pending",
                    )
                )

            except Exception as e:
                logger.error(f"Failed to generate test for {spec.name}: {e}")
                tests.append(
                    GeneratedTest(
                        spec_id=spec.id,
                        spec_name=spec.name,
                        test_code=f"# Failed to generate: {e}",
                        imports=[],
                        validation_status="invalid",
                        validation_errors=[str(e)],
                    )
                )

        return tests

    async def generate_implementation(
        self,
        requirements: str,
        test_file_content: str,
        language: str = "python",
        previous_implementation: Optional[str] = None,
        test_failures: Optional[List[TestExecutionResult]] = None,
    ) -> str:
        """
        Generate implementation code that should pass the tests.

        Args:
            requirements: Original requirements
            test_file_content: Complete test file content
            language: Programming language
            previous_implementation: Previous attempt (for iteration)
            test_failures: Previous test failures (for iteration)

        Returns:
            Generated implementation code
        """
        if not self._llm_callable:
            raise ValueError("LLM callable not configured. Call set_llm_callable() first.")

        # Format test failures for prompt
        failure_text = "None"
        if test_failures:
            failure_lines = []
            for f in test_failures:
                if not f.passed:
                    failure_lines.append(f"- {f.test_name}: {f.error_message or 'Failed'}")
            if failure_lines:
                failure_text = "\n".join(failure_lines)

        prompt = self.IMPLEMENTATION_PROMPT.format(
            requirements=requirements,
            language=language,
            test_file=test_file_content,
            previous_implementation=previous_implementation or "None",
            test_failures=failure_text,
        )

        response = await self._llm_callable(
            [
                {
                    "role": "system",
                    "content": f"You are an expert {language} developer. Output only implementation code.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        code = self._extract_content(response)
        code = self._clean_code_block(code)

        return code

    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        if isinstance(response, dict):
            return response.get("content", response.get("text", str(response)))
        return str(response)

    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code block markers from code."""
        code = code.strip()

        # Remove ```python or ```language blocks
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```python)
            lines = lines[1:]
            # Remove last line if it's just ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code

    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSON from LLM response, handling common issues."""
        # Try direct parse first
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "specifications" in result:
                return result["specifications"]
            if isinstance(result, dict):
                return [result]
            return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            result = json.loads(content.strip())
            if isinstance(result, list):
                return result
            return [result]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []

    def _extract_imports(
        self,
        code: str,
        language: str,
    ) -> tuple[List[str], List[str]]:
        """
        Extract import statements from code.

        Returns:
            Tuple of (imports, code_lines)
        """
        imports = []
        code_lines = []

        for line in code.strip().split("\n"):
            stripped = line.strip()
            if language == "python":
                if stripped.startswith("import ") or stripped.startswith("from "):
                    imports.append(stripped)
                else:
                    code_lines.append(line)
            elif language in ("javascript", "typescript"):
                if (
                    stripped.startswith("import ")
                    or (stripped.startswith("const ")
                    and "require" in stripped)
                ):
                    imports.append(stripped)
                else:
                    code_lines.append(line)
            else:
                code_lines.append(line)

        return imports, code_lines


# =============================================================================
# TEST FILE BUILDER
# =============================================================================


class TestFileBuilder:
    """
    Builds complete test files from generated tests.

    Combines imports, fixtures, and test functions into a single executable file.
    """

    def build_test_file(
        self,
        tests: List[GeneratedTest],
        suite: TestSuite,
    ) -> str:
        """
        Build complete test file content.

        Args:
            tests: List of generated tests
            suite: Test suite with configuration

        Returns:
            Complete test file content as string
        """
        if suite.language == "python":
            return self._build_python_test_file(tests, suite)
        elif suite.language == "javascript":
            return self._build_javascript_test_file(tests, suite)
        else:
            raise ValueError(f"Unsupported language: {suite.language}")

    def _build_python_test_file(
        self,
        tests: List[GeneratedTest],
        suite: TestSuite,
    ) -> str:
        """Build Python test file."""
        lines = []

        # Standard imports
        if suite.framework == "pytest":
            lines.append("import pytest")
        else:
            lines.append("import unittest")

        # Collect all unique imports
        all_imports = set(suite.import_statements)
        for test in tests:
            all_imports.update(test.imports)

        # Remove pytest import if already added
        all_imports.discard("import pytest")
        all_imports.discard("import unittest")

        for imp in sorted(all_imports):
            lines.append(imp)

        lines.append("")
        lines.append("")

        # Fixtures
        if suite.fixture_code:
            lines.append("# Fixtures")
            lines.append(suite.fixture_code)
            lines.append("")
            lines.append("")

        # Setup code
        if suite.setup_code:
            lines.append("# Setup")
            lines.append(suite.setup_code)
            lines.append("")
            lines.append("")

        # Test functions
        if suite.framework == "pytest":
            for test in tests:
                if test.fixtures:
                    lines.append(test.fixtures)
                    lines.append("")
                lines.append(test.test_code)
                lines.append("")
                lines.append("")
        else:
            # unittest style
            lines.append("class TestSuite(unittest.TestCase):")
            lines.append('    """Generated test suite."""')
            lines.append("")
            for test in tests:
                # Indent test code
                indented = "\n".join(f"    {line}" for line in test.test_code.split("\n"))
                lines.append(indented)
                lines.append("")

            lines.append("")
            lines.append("if __name__ == '__main__':")
            lines.append("    unittest.main()")

        # Teardown code
        if suite.teardown_code:
            lines.append("")
            lines.append("# Teardown")
            lines.append(suite.teardown_code)

        return "\n".join(lines)

    def _build_javascript_test_file(
        self,
        tests: List[GeneratedTest],
        suite: TestSuite,
    ) -> str:
        """Build JavaScript test file."""
        lines = []

        # Standard imports
        if suite.framework == "jest":
            lines.append("// Jest test file")
        else:
            lines.append("const { expect } = require('chai');")

        # Collect all unique imports
        all_imports = set(suite.import_statements)
        for test in tests:
            all_imports.update(test.imports)

        for imp in sorted(all_imports):
            lines.append(imp)

        lines.append("")

        # Test functions
        lines.append("describe('Generated Tests', () => {")

        if suite.setup_code:
            lines.append("  beforeEach(() => {")
            lines.append(f"    {suite.setup_code}")
            lines.append("  });")
            lines.append("")

        for test in tests:
            lines.append(test.test_code)
            lines.append("")

        if suite.teardown_code:
            lines.append("  afterEach(() => {")
            lines.append(f"    {suite.teardown_code}")
            lines.append("  });")

        lines.append("});")

        return "\n".join(lines)


# =============================================================================
# TDD MANAGER (HIGH-LEVEL API)
# =============================================================================


class TDDManager:
    """
    High-level API for TDD workflow management.

    Provides a unified interface for the TDD workflow:
    1. Generate test specifications
    2. Generate executable tests
    3. Build test files
    4. Execute tests (with sandbox integration)
    5. Iterate on failures

    Supports both SQLite and PostgreSQL backends.

    Usage:
        manager = TDDManager(backend="sqlite", db_path="~/.local/share/llmcore/tdd.db")
        await manager.initialize()

        # Full workflow
        suite = await manager.generate_test_specs(requirements, language="python")
        tests = await manager.generate_tests(suite)
        test_file = manager.build_test_file(tests, suite)

        # Or use the high-level run_tdd_cycle
        result = await manager.run_tdd_cycle(
            requirements="Implement Calculator class",
            max_iterations=3,
        )
    """

    def __init__(
        self,
        backend: Literal["sqlite", "postgres"] = "sqlite",
        db_path: Optional[str] = None,
        db_url: Optional[str] = None,
        enabled: bool = True,
        default_framework: str = "pytest",
        min_tests: int = 5,
        max_iterations: int = 3,
        llm_callable: Optional[Callable] = None,
        sandbox_executor: Optional[Callable] = None,
    ):
        """
        Initialize TDD manager.

        Args:
            backend: Storage backend to use ("sqlite" or "postgres")
            db_path: Path to SQLite database (for sqlite backend)
            db_url: PostgreSQL connection URL (for postgres backend)
            enabled: Whether TDD support is enabled
            default_framework: Default test framework
            min_tests: Minimum tests to generate
            max_iterations: Maximum TDD cycle iterations
            llm_callable: Async callable for LLM generation
            sandbox_executor: Async callable for sandbox execution

        Raises:
            ValueError: If backend is invalid or required config is missing
        """
        self.enabled = enabled
        self.backend_type = backend
        self.default_framework = default_framework
        self.min_tests = min_tests
        self.max_iterations = max_iterations

        self._test_generator = TestGenerator(
            llm_callable=llm_callable,
            default_framework=default_framework,
            min_tests=min_tests,
        )
        self._test_file_builder = TestFileBuilder()
        self._sandbox_executor = sandbox_executor
        self._backend: Optional[BaseTDDStorage] = None
        self._config: Dict[str, Any] = {}

        if not enabled:
            return

        # Import backends only when needed
        if backend == "sqlite":
            from .sqlite_tdd_storage import SqliteTDDStorage

            if not db_path:
                db_path = "~/.local/share/llmcore/tdd.db"
            self._backend = SqliteTDDStorage()
            self._config = {"path": db_path}

        elif backend == "postgres":
            # Validate db_url first before attempting import
            if not db_url:
                raise ValueError("db_url required for postgres backend")

            try:
                from .postgres_tdd_storage import PostgresTDDStorage
            except ImportError as e:
                raise ImportError(
                    "PostgreSQL TDD backend requires psycopg and psycopg_pool. "
                    "Install with: pip install psycopg[binary] psycopg_pool"
                ) from e

            self._backend = PostgresTDDStorage()
            self._config = {"db_url": db_url}

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'sqlite' or 'postgres'.")

    def set_llm_callable(self, llm_callable: Callable) -> None:
        """Set the LLM callable for generation."""
        self._test_generator.set_llm_callable(llm_callable)

    def set_sandbox_executor(self, sandbox_executor: Callable) -> None:
        """Set the sandbox executor for test execution."""
        self._sandbox_executor = sandbox_executor

    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        Must be called before using the manager.

        Raises:
            StorageError: If initialization fails
        """
        if self.enabled and self._backend:
            await self._backend.initialize(self._config)
            logger.info(f"TDDManager initialized with {self.backend_type} backend")

    async def generate_test_specs(
        self,
        requirements: str,
        language: str = "python",
        framework: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> TestSuite:
        """
        Generate test specifications from requirements.

        Args:
            requirements: What to implement/test
            language: Programming language
            framework: Test framework
            task_id: Optional task ID for tracking

        Returns:
            TestSuite with generated specifications
        """
        framework = framework or self.default_framework

        suite = await self._test_generator.generate_specs(
            requirements=requirements,
            language=language,
            framework=framework,
            min_tests=self.min_tests,
            task_id=task_id,
        )

        # Save to storage if enabled
        if self.enabled and self._backend:
            suite = await self._backend.save_test_suite(suite)

        return suite

    async def generate_tests(self, suite: TestSuite) -> List[GeneratedTest]:
        """
        Generate executable test code from specifications.

        Args:
            suite: Test suite with specifications

        Returns:
            List of GeneratedTest objects
        """
        tests = await self._test_generator.generate_tests(suite)

        # Save to storage if enabled
        if self.enabled and self._backend:
            for test in tests:
                await self._backend.save_generated_test(test)

        return tests

    async def generate_implementation(
        self,
        requirements: str,
        test_file_content: str,
        language: str = "python",
        previous_implementation: Optional[str] = None,
        test_failures: Optional[List[TestExecutionResult]] = None,
    ) -> str:
        """
        Generate implementation code that should pass the tests.

        Args:
            requirements: Original requirements
            test_file_content: Complete test file content
            language: Programming language
            previous_implementation: Previous attempt (for iteration)
            test_failures: Previous test failures (for iteration)

        Returns:
            Generated implementation code
        """
        return await self._test_generator.generate_implementation(
            requirements=requirements,
            test_file_content=test_file_content,
            language=language,
            previous_implementation=previous_implementation,
            test_failures=test_failures,
        )

    def build_test_file(
        self,
        tests: List[GeneratedTest],
        suite: TestSuite,
    ) -> str:
        """
        Build complete test file from generated tests.

        Args:
            tests: List of generated tests
            suite: Test suite with configuration

        Returns:
            Complete test file content
        """
        return self._test_file_builder.build_test_file(tests, suite)

    async def execute_tests(
        self,
        implementation_code: str,
        test_file_content: str,
        language: str = "python",
        framework: str = "pytest",
        timeout_seconds: int = 60,
    ) -> List[TestExecutionResult]:
        """
        Execute tests against implementation in sandbox.

        Args:
            implementation_code: The implementation to test
            test_file_content: The test file content
            language: Programming language
            framework: Test framework
            timeout_seconds: Execution timeout

        Returns:
            List of TestExecutionResult objects

        Raises:
            RuntimeError: If sandbox executor is not configured
        """
        if not self._sandbox_executor:
            raise RuntimeError(
                "Sandbox executor not configured. Call set_sandbox_executor() first."
            )

        # Combine implementation and tests
        if language == "python":
            combined = f"{implementation_code}\n\n{test_file_content}"
            if framework == "pytest":
                command = "python -m pytest -v --tb=short"
            else:
                command = "python -m unittest -v"
        else:
            combined = f"{implementation_code}\n\n{test_file_content}"
            command = "npm test"

        # Execute in sandbox
        result = await self._sandbox_executor(
            code=combined,
            command=command,
            timeout_seconds=timeout_seconds,
        )

        # Parse results
        return self._parse_test_output(result, framework)

    def _parse_test_output(
        self,
        execution_result: Any,
        framework: str,
    ) -> List[TestExecutionResult]:
        """Parse test execution output into results."""
        results = []

        # Extract stdout/stderr
        stdout = ""
        stderr = ""
        if hasattr(execution_result, "stdout"):
            stdout = execution_result.stdout
        elif isinstance(execution_result, dict):
            stdout = execution_result.get("stdout", "")

        if hasattr(execution_result, "stderr"):
            stderr = execution_result.stderr
        elif isinstance(execution_result, dict):
            stderr = execution_result.get("stderr", "")

        if framework == "pytest":
            results = self._parse_pytest_output(stdout, stderr)
        elif framework == "unittest":
            results = self._parse_unittest_output(stdout, stderr)
        elif framework in ("jest", "mocha"):
            results = self._parse_jest_output(stdout, stderr)

        # If no results parsed, create a single result based on exit code
        if not results:
            exit_code = 0
            if hasattr(execution_result, "exit_code"):
                exit_code = execution_result.exit_code
            elif isinstance(execution_result, dict):
                exit_code = execution_result.get("exit_code", 0)

            results.append(
                TestExecutionResult(
                    test_name="test_suite",
                    passed=exit_code == 0,
                    error_message=stderr if exit_code != 0 else None,
                    stdout=stdout,
                    stderr=stderr,
                )
            )

        return results

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
    ) -> List[TestExecutionResult]:
        """Parse pytest verbose output into results."""
        results = []

        # Match lines like: test_file.py::test_name PASSED/FAILED
        pattern = r"(?:[\w./]+::)?(\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)"

        for match in re.finditer(pattern, stdout):
            test_name = match.group(1)
            status = match.group(2)

            passed = status == "PASSED"
            error_type = None
            error_message = None

            if status == "FAILED":
                error_type = "assertion"
                # Try to extract error message
                error_pattern = rf"{test_name}.*?(?:AssertionError|assert).*?:\s*(.+?)(?:\n|$)"
                error_match = re.search(error_pattern, stdout, re.DOTALL)
                if error_match:
                    error_message = error_match.group(1).strip()[:200]
            elif status == "ERROR":
                error_type = "exception"
                error_message = "Test raised an exception"

            results.append(
                TestExecutionResult(
                    test_name=test_name,
                    passed=passed,
                    error_type=error_type,
                    error_message=error_message,
                    stdout=stdout[:500] if stdout else "",
                    stderr=stderr[:500] if stderr else "",
                )
            )

        return results

    def _parse_unittest_output(
        self,
        stdout: str,
        stderr: str,
    ) -> List[TestExecutionResult]:
        """Parse unittest output into results."""
        results = []

        # Match lines like: test_name (module.TestClass) ... ok/FAIL/ERROR
        pattern = r"(\w+)\s+\([^)]+\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)"

        for match in re.finditer(pattern, stdout + stderr, re.IGNORECASE):
            test_name = match.group(1)
            status = match.group(2).lower()

            passed = status == "ok"
            error_type = None
            error_message = None

            if status == "fail":
                error_type = "assertion"
            elif status == "error":
                error_type = "exception"

            results.append(
                TestExecutionResult(
                    test_name=test_name,
                    passed=passed,
                    error_type=error_type,
                    error_message=error_message,
                    stdout=stdout[:500] if stdout else "",
                    stderr=stderr[:500] if stderr else "",
                )
            )

        return results

    def _parse_jest_output(
        self,
        stdout: str,
        stderr: str,
    ) -> List[TestExecutionResult]:
        """Parse Jest output into results."""
        results = []

        # Match lines like: ✓ test name (Xms) or ✕ test name
        pass_pattern = r"[✓√]\s+(.+?)(?:\s+\((\d+)\s*ms\))?$"
        fail_pattern = r"[✕×]\s+(.+?)(?:\s+\((\d+)\s*ms\))?$"

        for match in re.finditer(pass_pattern, stdout, re.MULTILINE):
            test_name = match.group(1).strip()
            time_ms = float(match.group(2)) if match.group(2) else 0.0

            results.append(
                TestExecutionResult(
                    test_name=test_name,
                    passed=True,
                    execution_time_ms=time_ms,
                )
            )

        for match in re.finditer(fail_pattern, stdout + stderr, re.MULTILINE):
            test_name = match.group(1).strip()
            time_ms = float(match.group(2)) if match.group(2) else 0.0

            results.append(
                TestExecutionResult(
                    test_name=test_name,
                    passed=False,
                    error_type="assertion",
                    execution_time_ms=time_ms,
                )
            )

        return results

    async def run_tdd_cycle(
        self,
        requirements: str,
        language: str = "python",
        framework: Optional[str] = None,
        max_iterations: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> TDDCycleResult:
        """
        Run a complete TDD cycle.

        1. Generate test specifications
        2. Generate test code
        3. Generate implementation
        4. Execute tests
        5. If failures, iterate

        Args:
            requirements: What to implement
            language: Programming language
            framework: Test framework
            max_iterations: Max iterations before giving up
            task_id: Optional task ID for tracking

        Returns:
            TDDCycleResult with final code and test results
        """
        framework = framework or self.default_framework
        max_iterations = max_iterations or self.max_iterations

        import time

        start_time = time.time()

        # Step 1: Generate test specifications
        logger.info("TDD: Generating test specifications")
        suite = await self.generate_test_specs(
            requirements=requirements,
            language=language,
            framework=framework,
            task_id=task_id,
        )

        # Step 2: Generate test code
        logger.info(f"TDD: Generating {suite.spec_count} tests")
        tests = await self.generate_tests(suite)

        # Build test file
        test_file_content = self.build_test_file(tests, suite)

        implementation = None
        test_results: List[TestExecutionResult] = []

        for iteration in range(1, max_iterations + 1):
            logger.info(f"TDD: Iteration {iteration}/{max_iterations}")

            # Step 3: Generate implementation
            implementation = await self.generate_implementation(
                requirements=requirements,
                test_file_content=test_file_content,
                language=language,
                previous_implementation=implementation,
                test_failures=[r for r in test_results if not r.passed],
            )

            # Step 4: Run tests (if sandbox configured)
            if self._sandbox_executor:
                test_results = await self.execute_tests(
                    implementation_code=implementation,
                    test_file_content=test_file_content,
                    language=language,
                    framework=framework,
                )

                # Check if all tests pass
                passed = sum(1 for r in test_results if r.passed)
                failed = len(test_results) - passed

                logger.info(f"TDD: {passed}/{len(test_results)} tests passed")

                if failed == 0:
                    total_time = (time.time() - start_time) * 1000
                    result = TDDCycleResult(
                        id=f"cycle_{suite.id}_{iteration}",
                        suite_id=suite.id,
                        iteration=iteration,
                        tests_generated=len(tests),
                        tests_executed=len(test_results),
                        tests_passed=passed,
                        tests_failed=0,
                        code_generated=True,
                        implementation_code=implementation,
                        final_code=implementation,
                        all_tests_pass=True,
                        execution_results=test_results,
                        total_time_ms=total_time,
                    )

                    # Save to storage
                    if self.enabled and self._backend:
                        await self._backend.save_cycle_result(result)

                    return result
            else:
                # No sandbox - return with generated code only
                logger.warning(
                    "TDD: No sandbox executor - returning generated code without execution"
                )
                total_time = (time.time() - start_time) * 1000
                return TDDCycleResult(
                    id=f"cycle_{suite.id}_{iteration}",
                    suite_id=suite.id,
                    iteration=iteration,
                    tests_generated=len(tests),
                    tests_executed=0,
                    tests_passed=0,
                    tests_failed=0,
                    code_generated=True,
                    implementation_code=implementation,
                    final_code=implementation,
                    all_tests_pass=False,
                    execution_results=[],
                    failure_analysis="Tests not executed - no sandbox configured",
                    total_time_ms=total_time,
                )

        # Max iterations reached
        total_time = (time.time() - start_time) * 1000
        passed = sum(1 for r in test_results if r.passed)
        failed = len(test_results) - passed

        result = TDDCycleResult(
            id=f"cycle_{suite.id}_{max_iterations}",
            suite_id=suite.id,
            iteration=max_iterations,
            tests_generated=len(tests),
            tests_executed=len(test_results),
            tests_passed=passed,
            tests_failed=failed,
            code_generated=True,
            implementation_code=implementation,
            final_code=implementation,
            all_tests_pass=False,
            execution_results=test_results,
            failure_analysis=self._analyze_failures(test_results),
            suggestions=self._generate_suggestions(test_results),
            total_time_ms=total_time,
        )

        # Save to storage
        if self.enabled and self._backend:
            await self._backend.save_cycle_result(result)

        return result

    def _analyze_failures(self, results: List[TestExecutionResult]) -> str:
        """Analyze test failures for debugging."""
        failures = [r for r in results if not r.passed]

        if not failures:
            return "All tests passed."

        lines = ["Test Failure Analysis:", ""]
        for f in failures:
            lines.append(f"- {f.test_name}:")
            if f.error_type:
                lines.append(f"  Type: {f.error_type}")
            if f.error_message:
                lines.append(f"  Error: {f.error_message[:200]}")
            if f.error_traceback:
                lines.append(f"  Traceback: {f.error_traceback[:300]}")

        return "\n".join(lines)

    def _generate_suggestions(self, results: List[TestExecutionResult]) -> List[str]:
        """Generate suggestions for fixing failures."""
        suggestions = []
        failures = [r for r in results if not r.passed]

        for f in failures:
            if f.error_type == "assertion":
                suggestions.append(f"Fix {f.test_name}: Check return value or expected behavior")
            elif f.error_type == "exception":
                suggestions.append(f"Fix {f.test_name}: Handle or prevent the raised exception")
            elif f.error_type == "syntax":
                suggestions.append(f"Fix {f.test_name}: Correct syntax errors in implementation")
            elif f.error_type == "timeout":
                suggestions.append(
                    f"Fix {f.test_name}: Optimize for performance or check infinite loops"
                )

        return suggestions

    async def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get TDD statistics for analytics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with TDD statistics
        """
        if not self.enabled or not self._backend:
            return {
                "period_days": days,
                "total_suites": 0,
                "total_cycles": 0,
                "average_pass_rate": 0.0,
            }

        return await self._backend.get_stats(days)

    async def close(self) -> None:
        """Clean up resources used by the storage backend."""
        if self._backend:
            await self._backend.close()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data Models
    "TestSpecification",
    "TestSuite",
    "GeneratedTest",
    "TestExecutionResult",
    "TDDCycleResult",
    "TDDSession",
    # Base Interface
    "BaseTDDStorage",
    # Generator
    "TestGenerator",
    "TestFileBuilder",
    # High-level Manager
    "TDDManager",
]
