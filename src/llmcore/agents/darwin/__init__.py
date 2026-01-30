# src/llmcore/agents/darwin/__init__.py
"""
Darwin Agent Enhancement Module.

This module provides advanced capabilities for the Darwin agent system,
including:

- Failure Learning: SQLite and PostgreSQL-backed failure tracking and pattern detection
- TDD Support: Test-Driven Development workflow integration (coming in Phase 6.2)
- Multi-Attempt Arbiter: N-candidate generation with quality selection (coming in Phase 6.3)

Usage:
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
"""

from .failure_storage import (
    BaseFailureStorage,
    FailureContext,
    FailureLog,
    FailurePattern,
    FailureLearningManager,
)

__all__ = [
    # Base interface
    "BaseFailureStorage",
    # Data models
    "FailureLog",
    "FailurePattern",
    "FailureContext",
    # High-level manager
    "FailureLearningManager",
]

__version__ = "0.1.0"
