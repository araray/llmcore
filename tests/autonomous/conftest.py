# tests/autonomous/conftest.py
"""
Shared fixtures for autonomous module tests.

Provides common test infrastructure including temporary directories,
mock providers, and pre-configured manager instances.
"""

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add source to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture
def tmp_goals_path(tmp_path):
    """Temporary path for goal storage."""
    return str(tmp_path / "goals.json")


@pytest.fixture
def goal_store(tmp_goals_path):
    """Create a GoalStore with temporary storage."""
    from llmcore.autonomous.goals import GoalStore

    return GoalStore(tmp_goals_path)


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for goal decomposition."""
    provider = MagicMock()
    provider.default_model = "test-model"
    provider.get_name.return_value = "mock"

    # Default: return valid decomposition JSON
    async def mock_chat(*args, **kwargs):
        result = MagicMock()
        result.content = '[{"description": "Sub-goal 1"}, {"description": "Sub-goal 2"}]'
        return result

    provider.chat = AsyncMock(side_effect=mock_chat)
    return provider


@pytest.fixture
def escalation_manager():
    """Create an EscalationManager for testing."""
    from llmcore.autonomous.escalation import (
        EscalationLevel,
        EscalationManager,
    )

    return EscalationManager(
        auto_resolve_below=EscalationLevel.ADVISORY,
        dedup_window_seconds=60,
    )


@pytest.fixture
def heartbeat_manager():
    """Create a HeartbeatManager with short interval for testing."""
    from llmcore.autonomous.heartbeat import HeartbeatManager

    return HeartbeatManager(base_interval=timedelta(milliseconds=50))


@pytest.fixture
def resource_constraints():
    """Create ResourceConstraints for testing."""
    from llmcore.autonomous.resource import ResourceConstraints

    return ResourceConstraints(
        max_cpu_percent=80.0,
        max_memory_percent=80.0,
        max_temperature_c=75.0,
        max_hourly_cost_usd=1.0,
        max_daily_cost_usd=10.0,
        max_hourly_tokens=100_000,
        max_daily_tokens=1_000_000,
        min_disk_free_gb=1.0,
    )


@pytest.fixture
def resource_monitor(resource_constraints):
    """Create a ResourceMonitor for testing."""
    from llmcore.autonomous.resource import ResourceMonitor

    return ResourceMonitor(
        constraints=resource_constraints,
        check_interval=timedelta(seconds=1),
    )
