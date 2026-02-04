# tests/context/conftest.py
"""
Shared fixtures for Adaptive Context Synthesis tests.

Provides mock sources, token counters, pre-configured synthesizers,
and sample data for all context module tests.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure source is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Token Counter Fixtures
# =============================================================================


@pytest.fixture
def estimate_counter():
    """Create a deterministic EstimateCounter (4 chars/token)."""
    from llmcore.context.synthesis import EstimateCounter

    return EstimateCounter(chars_per_token=4)


# =============================================================================
# Mock Context Source Helpers
# =============================================================================


class MockContextSource:
    """
    Configurable mock context source for testing.

    Attributes:
        source_name: Name returned in the chunk's ``source`` field.
        content: Fixed content to return.
        tokens: Fixed token count to return.
        priority: Priority of the returned chunk.
        relevance: Relevance score for the returned chunk.
        recency: Recency score for the returned chunk.
        fail: If True, get_context raises RuntimeError.
        call_count: Number of times get_context was invoked.
    """

    def __init__(
        self,
        source_name: str = "mock",
        content: str = "Mock context content.",
        tokens: int = 10,
        priority: int = 50,
        relevance: float = 0.8,
        recency: float = 0.7,
        fail: bool = False,
    ) -> None:
        self.source_name = source_name
        self.content = content
        self.tokens = tokens
        self.priority = priority
        self.relevance = relevance
        self.recency = recency
        self.fail = fail
        self.call_count = 0

    async def get_context(
        self,
        task: Optional[Any] = None,
        max_tokens: int = 10_000,
    ):
        """Return a ContextChunk (or raise if configured to fail)."""
        from llmcore.context.synthesis import ContextChunk

        self.call_count += 1
        if self.fail:
            raise RuntimeError(f"Mock source '{self.source_name}' failure")

        return ContextChunk(
            source=self.source_name,
            content=self.content,
            tokens=self.tokens,
            priority=self.priority,
            relevance=self.relevance,
            recency=self.recency,
        )


@pytest.fixture
def mock_source_factory():
    """Factory fixture for creating MockContextSource instances."""
    return MockContextSource


@pytest.fixture
def high_priority_source():
    """Core context source (priority 100)."""
    return MockContextSource(
        source_name="goals",
        content="# Current Goals\n\n## Build the test suite\nStatus: active\nProgress: 50%",
        tokens=20,
        priority=100,
        relevance=1.0,
        recency=1.0,
    )


@pytest.fixture
def medium_priority_source():
    """Recent context source (priority 80)."""
    return MockContextSource(
        source_name="recent",
        content="# Recent History\n\n**USER:** Run the tests\n**ASSISTANT:** All 219 pass.",
        tokens=15,
        priority=80,
        relevance=0.9,
        recency=1.0,
    )


@pytest.fixture
def low_priority_source():
    """Episodic context source (priority 40)."""
    return MockContextSource(
        source_name="episodic",
        content="# Past Experiences\n\n- [80%] Fixed import chain issue.",
        tokens=12,
        priority=40,
        relevance=0.5,
        recency=0.3,
    )


@pytest.fixture
def failing_source():
    """Context source that always raises."""
    return MockContextSource(
        source_name="broken",
        fail=True,
    )


# =============================================================================
# Synthesizer Fixtures
# =============================================================================


@pytest.fixture
def synthesizer(estimate_counter):
    """Pre-configured ContextSynthesizer with deterministic counter."""
    from llmcore.context.synthesis import ContextSynthesizer

    return ContextSynthesizer(
        max_tokens=1000,
        compression_threshold=0.75,
        token_counter=estimate_counter,
    )


@pytest.fixture
def synthesizer_with_sources(
    synthesizer,
    high_priority_source,
    medium_priority_source,
    low_priority_source,
):
    """Synthesizer with three registered sources at different priorities."""
    synthesizer.add_source("goals", high_priority_source, priority=100)
    synthesizer.add_source("recent", medium_priority_source, priority=80)
    synthesizer.add_source("episodic", low_priority_source, priority=40)
    return synthesizer


# =============================================================================
# Mock GoalManager Fixtures
# =============================================================================


@pytest.fixture
def mock_goal():
    """Create a mock Goal object with typical attributes."""
    goal = MagicMock()
    goal.description = "Implement Phase 3 context intelligence"
    goal.status = MagicMock()
    goal.status.value = "active"
    goal.progress = 0.50
    goal.success_criteria = []
    goal.learned_strategies = ["Use protocol-based abstractions"]
    return goal


@pytest.fixture
def mock_goal_manager(mock_goal):
    """Create a mock GoalManager returning sample goals."""
    manager = MagicMock()
    manager.get_active_goals = AsyncMock(return_value=[mock_goal])
    manager.get_all_goals = AsyncMock(return_value=[mock_goal])
    return manager


@pytest.fixture
def mock_goal_manager_empty():
    """Create a mock GoalManager with no active goals."""
    manager = MagicMock()
    manager.get_active_goals = AsyncMock(return_value=[])
    manager.get_all_goals = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_goal_manager_failing():
    """Create a mock GoalManager that raises on all calls."""
    manager = MagicMock()
    manager.get_active_goals = AsyncMock(side_effect=RuntimeError("Storage unavailable"))
    manager.get_all_goals = AsyncMock(side_effect=RuntimeError("Storage unavailable"))
    return manager


# =============================================================================
# Mock Retrieval Function Fixtures
# =============================================================================


@pytest.fixture
def mock_retrieval_fn():
    """Create a mock retrieval function returning sample RAG chunks."""

    async def _retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return [
            {
                "content": "The ContextSynthesizer orchestrates context assembly.",
                "source": "synthesis.py",
                "score": 0.95,
            },
            {
                "content": "Skills are loaded dynamically based on task keywords.",
                "source": "skills.py",
                "score": 0.87,
            },
        ]

    return _retrieve


@pytest.fixture
def mock_retrieval_fn_empty():
    """Retrieval function that returns no results."""

    async def _retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return []

    return _retrieve


@pytest.fixture
def mock_retrieval_fn_failing():
    """Retrieval function that raises."""

    async def _retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        raise ConnectionError("Vector store unavailable")

    return _retrieve


# =============================================================================
# Mock Task Fixtures
# =============================================================================


@pytest.fixture
def sample_task():
    """Simple task object with a description attribute."""
    task = MagicMock()
    task.description = "Write pytest tests for the context module"
    return task


@pytest.fixture
def sample_task_no_description():
    """Task-like object with no description attribute."""
    return object()
