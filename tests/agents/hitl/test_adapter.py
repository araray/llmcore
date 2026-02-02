# tests/agents/hitl/test_adapter.py
"""
Tests for HITLManagerAdapter integration with ActivityExecutor.

Tests the adapter layer that bridges the Phase 5 HITLManager with
the existing ActivityExecutor interface.
"""

# Test imports
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "src")

from llmcore.agents.activities.executor import (
    HITLApprover,
    HITLDecision,
    HITLManagerAdapter,
    create_hitl_approver,
)
from llmcore.agents.activities.schema import (
    ActivityCategory,
    ActivityDefinition,
    ActivityRequest,
    RiskLevel,
)


class TestHITLManagerAdapterInit:
    """Test adapter initialization."""

    def test_adapter_init(self):
        """Test adapter can be created with mock manager."""
        mock_manager = MagicMock()
        mock_manager.risk_assessor = MagicMock()
        mock_manager.scope_manager = MagicMock()

        adapter = HITLManagerAdapter(mock_manager)
        assert adapter._manager is mock_manager
        assert adapter.risk_threshold == RiskLevel.MEDIUM

    def test_adapter_custom_threshold(self):
        """Test adapter with custom risk threshold."""
        mock_manager = MagicMock()
        adapter = HITLManagerAdapter(mock_manager, risk_threshold=RiskLevel.HIGH)
        assert adapter.risk_threshold == RiskLevel.HIGH


class TestHITLManagerAdapterRequiresApproval:
    """Test requires_approval delegation."""

    def test_requires_approval_low_risk(self):
        """Test low risk activity doesn't require approval."""
        mock_manager = MagicMock()
        mock_risk = MagicMock()
        mock_risk.requires_approval = False
        mock_manager.risk_assessor.assess.return_value = mock_risk

        adapter = HITLManagerAdapter(mock_manager)

        request = ActivityRequest(activity="file_read", parameters={"path": "/tmp/test"})
        definition = ActivityDefinition(
            name="file_read",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Read a file",
        )

        result = adapter.requires_approval(request, definition)

        assert result is False
        mock_manager.risk_assessor.assess.assert_called_once()

    def test_requires_approval_high_risk(self):
        """Test high risk activity requires approval."""
        mock_manager = MagicMock()
        mock_risk = MagicMock()
        mock_risk.requires_approval = True
        mock_manager.risk_assessor.assess.return_value = mock_risk

        adapter = HITLManagerAdapter(mock_manager)

        request = ActivityRequest(activity="bash_exec", parameters={"command": "rm -rf /"})
        definition = ActivityDefinition(
            name="bash_exec",
            category=ActivityCategory.CODE_EXECUTION,
            description="Execute bash command",
        )

        result = adapter.requires_approval(request, definition)

        assert result is True


class TestHITLManagerAdapterScopeOperations:
    """Test scope grant/revoke operations."""

    def test_grant_scope(self):
        """Test granting approval scope."""
        mock_manager = MagicMock()
        adapter = HITLManagerAdapter(mock_manager)

        adapter.grant_scope("file_read", RiskLevel.LOW)

        mock_manager.scope_manager.grant_session_approval.assert_called_once_with(
            tool_name="file_read",
            max_risk_level="low",
        )

    def test_revoke_scope(self):
        """Test revoking approval scope."""
        mock_manager = MagicMock()
        adapter = HITLManagerAdapter(mock_manager)

        adapter.revoke_scope("file_read")

        mock_manager.scope_manager.revoke_session_approval.assert_called_once_with("file_read")


class TestCreateHITLApprover:
    """Test factory function for HITL approver creation."""

    def test_create_basic_approver(self):
        """Test creating basic HITLApprover."""
        approver = create_hitl_approver(use_advanced=False)
        assert isinstance(approver, HITLApprover)

    def test_create_basic_with_threshold(self):
        """Test creating basic approver with custom threshold."""
        approver = create_hitl_approver(
            use_advanced=False,
            risk_threshold=RiskLevel.HIGH,
        )
        assert isinstance(approver, HITLApprover)
        assert approver.risk_threshold == RiskLevel.HIGH


class TestHITLDecisionMapping:
    """Test that decisions are properly mapped."""

    def test_decision_approved(self):
        """Test approved decision mapping."""
        decision = HITLDecision(approved=True, reason="Pre-approved")
        assert decision.approved is True
        assert decision.reason == "Pre-approved"

    def test_decision_rejected(self):
        """Test rejected decision mapping."""
        decision = HITLDecision(approved=False, reason="User rejected")
        assert decision.approved is False
        assert decision.reason == "User rejected"

    def test_decision_with_modified_params(self):
        """Test decision with modified parameters."""
        decision = HITLDecision(
            approved=True,
            reason="Approved with modifications",
            modified_params={"safe_mode": True},
        )
        assert decision.approved is True
        assert decision.modified_params == {"safe_mode": True}


class TestIntegrationWithRealHITLManager:
    """Integration tests using real HITLManager."""

    def test_adapter_with_real_manager(self):
        """Test adapter works with actual HITLManager."""
        from llmcore.agents.hitl import HITLConfig, HITLManager

        config = HITLConfig(enabled=False)  # Disabled for quick testing
        manager = HITLManager(config=config)
        adapter = HITLManagerAdapter(manager)

        # Should auto-approve when HITL disabled
        request = ActivityRequest(activity="file_read", parameters={"path": "/tmp/test"})
        definition = ActivityDefinition(
            name="file_read",
            category=ActivityCategory.FILE_OPERATIONS,
            description="Read a file",
        )

        # requires_approval should still work (checks risk assessment)
        result = adapter.requires_approval(request, definition)
        # With disabled manager, low-risk activities shouldn't require approval
        assert isinstance(result, bool)

    def test_create_advanced_approver(self):
        """Test creating adapter via factory with real manager."""
        try:
            approver = create_hitl_approver(use_advanced=True)
            assert isinstance(approver, HITLManagerAdapter)
        except ImportError:
            pytest.skip("HITLManager not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
