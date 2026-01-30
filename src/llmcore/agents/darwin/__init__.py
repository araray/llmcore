# src/llmcore/agents/darwin/__init__.py
"""
Darwin Agent Enhancement Module.

This module provides advanced capabilities for the Darwin agent system,
including:

- Failure Learning: SQLite and PostgreSQL-backed failure tracking and pattern detection
- TDD Support: Test-Driven Development workflow integration (Phase 6.2)
- Multi-Attempt Arbiter: N-candidate generation with quality selection (coming in Phase 6.3)

Usage - Failure Learning:
    from llmcore.agents.darwin import FailureLearningManager, FailureLog

    # Initialize failure learning with SQLite
    manager = FailureLearningManager(backend="sqlite")
    await manager.initialize()

    # Log a failure
    failure = FailureLog(
        task_id="task_123",
        agent_run_id="run_456",
        goal="Implement authentication",
        phase="ACT",
        failure_type="test_failure",
        error_message="AssertionError: login() returned None",
    )
    await manager.log_failure(failure)

    # Get failure context before planning
    context = await manager.get_failure_context(goal="Implement authentication")
    print(context.avoidance_instructions)

Usage - TDD Support:
    from llmcore.agents.darwin import TDDManager, TestSpecification

    # Initialize TDD manager with SQLite
    manager = TDDManager(backend="sqlite")
    await manager.initialize()
    manager.set_llm_callable(llm_callable)
    manager.set_sandbox_executor(sandbox_executor)

    # Run TDD cycle
    result = await manager.run_tdd_cycle(
        requirements="Implement Calculator class with add, subtract, multiply, divide",
        max_iterations=3,
    )
    print(f"Tests passed: {result.tests_passed}/{result.tests_executed}")
"""

# Phase 6.1: Failure Learning
from .failure_storage import (
    BaseFailureStorage,
    FailureContext,
    FailureLearningManager,
    FailureLog,
    FailurePattern,
)

# Phase 6.2: TDD Support
from .tdd_support import (
    BaseTDDStorage,
    GeneratedTest,
    TDDCycleResult,
    TDDManager,
    TDDSession,
    TestExecutionResult,
    TestFileBuilder,
    TestGenerator,
    TestSpecification,
    TestSuite,
)

__all__ = [
    # === Phase 6.1: Failure Learning ===
    # Base interface
    "BaseFailureStorage",
    # Data models
    "FailureLog",
    "FailurePattern",
    "FailureContext",
    # High-level manager
    "FailureLearningManager",
    # === Phase 6.2: TDD Support ===
    # Base interface
    "BaseTDDStorage",
    # Data models
    "TestSpecification",
    "TestSuite",
    "GeneratedTest",
    "TestExecutionResult",
    "TDDCycleResult",
    "TDDSession",
    # Components
    "TestGenerator",
    "TestFileBuilder",
    # High-level manager
    "TDDManager",
]

__version__ = "0.2.0"
