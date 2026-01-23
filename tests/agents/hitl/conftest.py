# tests/agents/hitl/conftest.py
"""
Pytest configuration and fixtures for HITL tests.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from llmcore.agents.hitl import (
    ActivityInfo,
    ApprovalScope,
    ApprovalStatus,
    AutoApproveCallback,
    HITLConfig,
    HITLDecision,
    HITLManager,
    HITLRequest,
    HITLResponse,
    InMemoryHITLStore,
    RiskAssessment,
    RiskLevel,
    TimeoutPolicy,
)


# =============================================================================
# EVENT LOOP CONFIGURATION
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# COMMON FIXTURES
# =============================================================================


@pytest.fixture
def default_config():
    """Create default HITL config for tests."""
    return HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        default_timeout_seconds=300,
        timeout_policy=TimeoutPolicy.REJECT,
        safe_tools=["final_answer", "respond", "think_aloud"],
        low_risk_tools=["file_read", "search", "web_fetch"],
        high_risk_tools=["bash_exec", "file_write", "file_delete"],
        critical_tools=["sudo_exec", "network_modify"],
    )


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_activity():
    """Create sample activity info."""
    return ActivityInfo(
        activity_type="bash_exec",
        parameters={"command": "ls -la /workspace"},
        reason="List workspace files"
    )


@pytest.fixture
def sample_risk_assessment():
    """Create sample risk assessment."""
    return RiskAssessment(
        overall_level="medium",
        requires_approval=True,
        factors=[],
        dangerous_patterns=[],
        scope_risk=None
    )


@pytest.fixture
def sample_hitl_request(sample_activity, sample_risk_assessment):
    """Create sample HITL request."""
    request = HITLRequest(
        activity=sample_activity,
        risk_assessment=sample_risk_assessment,
        session_id="test-session-123",
        user_id="test-user-456",
        context_summary="Testing HITL workflow"
    )
    request.set_expiration(300)
    return request


@pytest.fixture
def sample_hitl_response(sample_hitl_request):
    """Create sample HITL response."""
    return HITLResponse(
        request_id=sample_hitl_request.request_id,
        approved=True,
        responder_id="test-user-456",
        feedback="Approved for testing"
    )


@pytest.fixture
def sample_decision(sample_hitl_request, sample_hitl_response):
    """Create sample HITL decision."""
    return HITLDecision(
        status=ApprovalStatus.APPROVED,
        reason="Approved by user",
        request=sample_hitl_request,
        response=sample_hitl_response
    )


# =============================================================================
# MANAGER FIXTURES
# =============================================================================


@pytest.fixture
def auto_approve_manager(default_config):
    """Create manager that auto-approves."""
    return HITLManager(
        config=default_config,
        callback=AutoApproveCallback(delay_seconds=0),
        state_store=InMemoryHITLStore(),
        session_id="test-session",
        user_id="test-user"
    )


@pytest.fixture
def auto_reject_manager(default_config):
    """Create manager that auto-rejects."""
    return HITLManager(
        config=default_config,
        callback=AutoApproveCallback(approve=False, delay_seconds=0),
        state_store=InMemoryHITLStore(),
        session_id="test-session",
        user_id="test-user"
    )


@pytest.fixture
def disabled_manager():
    """Create disabled HITL manager."""
    config = HITLConfig(enabled=False)
    return HITLManager(
        config=config,
        state_store=InMemoryHITLStore()
    )
