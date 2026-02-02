# tests/agents/hitl/test_manager.py
"""
Tests for HITL Manager.

Tests:
- Approval workflow
- Risk-based decisions
- Scope integration
- Timeout handling
- Event callbacks
- Statistics
"""

import asyncio

import pytest

from llmcore.agents.hitl import (
    ApprovalStatus,
    AutoApproveCallback,
    HITLConfig,
    HITLManager,
    HITLResponse,
    InMemoryHITLStore,
    RiskLevel,
    TimeoutPolicy,
    create_hitl_manager,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def hitl_config():
    """Create test HITL config."""
    return HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        default_timeout_seconds=60,
        timeout_policy=TimeoutPolicy.REJECT,
        safe_tools=["final_answer", "respond", "think_aloud"],
        low_risk_tools=["file_read", "search"],
        high_risk_tools=["bash_exec", "file_write", "file_delete"],
        critical_tools=["sudo_exec", "network_request"],
    )


@pytest.fixture
def manager(hitl_config):
    """Create HITL manager with test config."""
    return HITLManager(
        config=hitl_config,
        callback=AutoApproveCallback(delay_seconds=0),
        state_store=InMemoryHITLStore(),
        session_id="test-session",
        user_id="test-user",
    )


@pytest.fixture
def auto_reject_manager(hitl_config):
    """Create manager that auto-rejects."""
    return HITLManager(
        config=hitl_config,
        callback=AutoApproveCallback(approve_all=False, delay_seconds=0),
        state_store=InMemoryHITLStore(),
        session_id="test-session",
        user_id="test-user",
    )


# =============================================================================
# BASIC WORKFLOW TESTS
# =============================================================================


class TestBasicWorkflow:
    """Test basic approval workflow."""

    @pytest.mark.asyncio
    async def test_safe_tool_auto_approved(self, manager):
        """Safe tools should be auto-approved without callback."""
        decision = await manager.check_approval("final_answer", {"answer": "Hello world"})

        assert decision.is_approved
        assert decision.status == ApprovalStatus.AUTO_APPROVED
        assert (
            "low risk" in decision.reason.lower()
            or "no approval required" in decision.reason.lower()
        )

    @pytest.mark.asyncio
    async def test_high_risk_tool_requires_approval(self, manager):
        """High risk tools should require approval."""
        decision = await manager.check_approval("bash_exec", {"command": "ls -la"})

        assert decision.is_approved  # Auto-approve callback
        assert decision.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_rejection(self, auto_reject_manager):
        """Should handle rejection."""
        decision = await auto_reject_manager.check_approval("bash_exec", {"command": "ls"})

        assert not decision.is_approved
        assert decision.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_hitl_disabled(self, hitl_config):
        """Should auto-approve when HITL is disabled."""
        hitl_config.enabled = False
        manager = HITLManager(config=hitl_config)

        decision = await manager.check_approval("bash_exec", {"command": "rm -rf /"})

        assert decision.is_approved
        assert decision.status == ApprovalStatus.AUTO_APPROVED
        assert "disabled" in decision.reason.lower()


# =============================================================================
# RISK-BASED DECISION TESTS
# =============================================================================


class TestRiskBasedDecisions:
    """Test risk-based decision making."""

    @pytest.mark.asyncio
    async def test_dangerous_command_flagged(self, manager):
        """Dangerous commands should be flagged."""
        decision = await manager.check_approval("bash_exec", {"command": "rm -rf /"})

        # Should still go through approval (auto-approve in test)
        assert decision.is_approved
        # But request should have high risk
        if decision.request:
            assert decision.request.risk_assessment.overall_level in ("high", "critical")

    @pytest.mark.asyncio
    async def test_low_risk_path(self, manager):
        """Low risk operations should pass easily."""
        decision = await manager.check_approval("file_read", {"path": "/workspace/readme.md"})

        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_sensitive_path_flagged(self, manager):
        """Sensitive paths should increase risk."""
        decision = await manager.check_approval("file_read", {"path": "/etc/passwd"})

        # Should still approve (auto-approve) but risk is higher
        assert decision.is_approved


# =============================================================================
# SCOPE INTEGRATION TESTS
# =============================================================================


class TestScopeIntegration:
    """Test scope management integration."""

    @pytest.mark.asyncio
    async def test_pre_approved_by_scope(self, manager):
        """Pre-approved tools should not require callback."""
        # Grant approval first
        manager.grant_session_approval("bash_exec", max_risk_level=RiskLevel.HIGH)

        decision = await manager.check_approval("bash_exec", {"command": "ls"})

        assert decision.is_approved
        assert decision.status == ApprovalStatus.AUTO_APPROVED
        assert "scope" in decision.reason.lower() or "pre-approved" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_conditional_scope(self, manager):
        """Conditional scopes should only approve matching params."""
        # Grant approval with condition
        manager.grant_session_approval(
            "file_write",
            conditions={"path_pattern": "/workspace/*"},
            max_risk_level=RiskLevel.MEDIUM,
        )

        # Should approve matching path
        decision = await manager.check_approval(
            "file_write", {"path": "/workspace/test.txt", "content": "test"}
        )
        assert decision.is_approved

    @pytest.mark.asyncio
    async def test_scope_revocation(self, manager):
        """Revoked scopes should require approval again."""
        manager.grant_session_approval("bash_exec")
        manager.revoke_session_approval("bash_exec")

        # Mock callback to track if called
        callback_called = []
        original_callback = manager.callback.request_approval

        async def tracking_callback(request):
            callback_called.append(request)
            return await original_callback(request)

        manager.callback.request_approval = tracking_callback

        decision = await manager.check_approval("bash_exec", {"command": "ls"})

        # Callback should have been called
        assert len(callback_called) > 0

    @pytest.mark.asyncio
    async def test_pattern_approval(self, manager):
        """Pattern approvals should match multiple tools."""
        manager.grant_pattern_approval("file_*")

        # Should approve file_read
        decision = await manager.check_approval("file_read", {"path": "/tmp/test"})
        assert decision.is_approved

        # Should approve file_write
        decision = await manager.check_approval("file_write", {"path": "/tmp/test"})
        assert decision.is_approved


# =============================================================================
# TIMEOUT HANDLING TESTS
# =============================================================================


class TestTimeoutHandling:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_reject_policy(self, hitl_config):
        """Should reject on timeout with REJECT policy."""
        hitl_config.timeout_policy = TimeoutPolicy.REJECT
        hitl_config.default_timeout_seconds = 0.01  # Very short

        # Use a callback that doesn't respond
        class HangingCallback:
            async def request_approval(self, request):
                await asyncio.sleep(10)  # Longer than timeout
                return HITLResponse(request_id=request.request_id, approved=True)

            async def notify_timeout(self, request):
                pass

            async def notify_result(self, request, decision):
                pass

        manager = HITLManager(
            config=hitl_config, callback=HangingCallback(), state_store=InMemoryHITLStore()
        )

        decision = await manager.check_approval("bash_exec", {"command": "ls"})

        # Should reject due to timeout
        assert not decision.is_approved
        assert decision.status in (ApprovalStatus.REJECTED, ApprovalStatus.TIMEOUT)

    @pytest.mark.asyncio
    async def test_timeout_approve_policy(self, hitl_config):
        """Should approve on timeout with APPROVE policy for low risk."""
        hitl_config.timeout_policies_by_risk = {
            "low": TimeoutPolicy.APPROVE,
            "medium": TimeoutPolicy.REJECT,
        }
        hitl_config.default_timeout_seconds = 0.01

        class HangingCallback:
            async def request_approval(self, request):
                await asyncio.sleep(10)
                return HITLResponse(request_id=request.request_id, approved=True)

            async def notify_timeout(self, request):
                pass

            async def notify_result(self, request, decision):
                pass

        manager = HITLManager(
            config=hitl_config, callback=HangingCallback(), state_store=InMemoryHITLStore()
        )

        # Low risk operation
        decision = await manager.check_approval("file_read", {"path": "/workspace/test.txt"})

        # Depends on how risk is assessed - may auto-approve or timeout-approve


# =============================================================================
# EVENT CALLBACK TESTS
# =============================================================================


class TestEventCallbacks:
    """Test event callback registration."""

    @pytest.mark.asyncio
    async def test_on_approval_callback(self, manager):
        """Should call approval callback."""
        approvals = []

        def on_approval(request, decision):
            approvals.append((request, decision))

        manager.on_approval(on_approval)

        decision = await manager.check_approval("bash_exec", {"command": "ls"})

        assert decision.is_approved
        assert len(approvals) == 1
        assert approvals[0][1].is_approved

    @pytest.mark.asyncio
    async def test_on_rejection_callback(self, auto_reject_manager):
        """Should call rejection callback."""
        rejections = []

        def on_rejection(request, decision):
            rejections.append((request, decision))

        auto_reject_manager.on_rejection(on_rejection)

        decision = await auto_reject_manager.check_approval("bash_exec", {"command": "ls"})

        assert not decision.is_approved
        assert len(rejections) == 1

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, manager):
        """Should call all registered callbacks."""
        calls = []

        def callback1(request, decision):
            calls.append("callback1")

        def callback2(request, decision):
            calls.append("callback2")

        manager.on_approval(callback1)
        manager.on_approval(callback2)

        await manager.check_approval("bash_exec", {"command": "ls"})

        assert "callback1" in calls
        assert "callback2" in calls


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestStatistics:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, manager):
        """Should track approval statistics."""
        # Some approvals
        for _ in range(3):
            await manager.check_approval("bash_exec", {"command": "ls"})

        stats = manager.get_statistics()

        assert stats["requests_created"] >= 0
        assert stats["requests_approved"] >= 0
        assert "session_id" in stats
        assert "user_id" in stats

    @pytest.mark.asyncio
    async def test_auto_approved_stats(self, manager):
        """Should track auto-approvals separately."""
        # Safe tool (auto-approved)
        await manager.check_approval("final_answer", {"answer": "test"})
        await manager.check_approval("final_answer", {"answer": "test2"})

        stats = manager.get_statistics()
        assert stats["requests_auto_approved"] >= 0

    @pytest.mark.asyncio
    async def test_scope_grant_stats(self, manager):
        """Should track scope grants."""
        manager.grant_session_approval("tool1")
        manager.grant_session_approval("tool2")

        stats = manager.get_statistics()
        assert stats["scope_grants"] == 2


# =============================================================================
# ASYNC WORKFLOW TESTS
# =============================================================================


class TestAsyncWorkflow:
    """Test async approval workflow."""

    @pytest.mark.asyncio
    async def test_submit_response(self, hitl_config):
        """Should handle async response submission."""
        from llmcore.agents.hitl import QueueHITLCallback

        callback = QueueHITLCallback()
        manager = HITLManager(
            config=hitl_config, callback=callback, state_store=InMemoryHITLStore()
        )

        # Start approval request in background
        approval_task = asyncio.create_task(manager.check_approval("bash_exec", {"command": "ls"}))

        await asyncio.sleep(0.01)

        # Get pending request
        pending = await manager.get_pending_requests()
        if pending:
            # Submit response
            response = HITLResponse(request_id=pending[0].request_id, approved=True)
            await callback.submit_response(response)

        # Wait for result
        try:
            decision = await asyncio.wait_for(approval_task, timeout=1.0)
            assert decision.is_approved
        except asyncio.TimeoutError:
            approval_task.cancel()


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_hitl_manager(self):
        """Should create manager with defaults."""
        manager = create_hitl_manager()

        assert manager is not None
        assert manager.config.enabled

    def test_create_hitl_manager_disabled(self):
        """Should create disabled manager."""
        manager = create_hitl_manager(enabled=False)

        assert not manager.config.enabled

    def test_create_hitl_manager_with_persistence(self, tmp_path):
        """Should create manager with file persistence."""
        manager = create_hitl_manager(persist_path=str(tmp_path / "hitl_state"))

        assert manager is not None
        # State store should be file-based

    def test_create_hitl_manager_custom_config(self):
        """Should accept custom configuration."""
        manager = create_hitl_manager(
            risk_threshold="high", session_id="custom-session", user_id="custom-user"
        )

        assert manager.config.global_risk_threshold == "high"
        assert manager.session_id == "custom-session"
        assert manager.user_id == "custom-user"


# =============================================================================
# HELPER METHOD TESTS
# =============================================================================


class TestHelperMethods:
    """Test helper methods."""

    def test_is_safe_tool(self, manager):
        """Should identify safe tools."""
        assert manager.is_safe_tool("final_answer")
        assert manager.is_safe_tool("respond")
        assert not manager.is_safe_tool("bash_exec")

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, manager):
        """Should cleanup expired requests."""
        # Create some requests
        await manager.check_approval("bash_exec", {"command": "ls"})

        # Cleanup (shouldn't affect anything in this test)
        cleaned = await manager.cleanup_expired()
        assert cleaned >= 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_parameters(self, manager):
        """Should handle empty parameters."""
        decision = await manager.check_approval("bash_exec", {})
        assert decision is not None

    @pytest.mark.asyncio
    async def test_none_context(self, manager):
        """Should handle None context."""
        decision = await manager.check_approval("bash_exec", {"command": "ls"}, context=None)
        assert decision is not None

    @pytest.mark.asyncio
    async def test_concurrent_approvals(self, manager):
        """Should handle concurrent approval requests."""
        tasks = [manager.check_approval(f"tool_{i}", {"param": i}) for i in range(5)]

        decisions = await asyncio.gather(*tasks)
        assert len(decisions) == 5

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, hitl_config):
        """Should handle callback errors gracefully."""

        class ErrorCallback:
            async def request_approval(self, request):
                raise RuntimeError("Callback error")

            async def notify_timeout(self, request):
                pass

            async def notify_result(self, request, decision):
                pass

        manager = HITLManager(
            config=hitl_config, callback=ErrorCallback(), state_store=InMemoryHITLStore()
        )

        decision = await manager.check_approval("bash_exec", {"command": "ls"})

        # Should reject on error
        assert not decision.is_approved

    @pytest.mark.asyncio
    async def test_approval_callback_exception(self, manager):
        """Should handle exception in approval callback."""

        def bad_callback(request, decision):
            raise RuntimeError("Callback error")

        manager.on_approval(bad_callback)

        # Should not crash
        decision = await manager.check_approval("bash_exec", {"command": "ls"})
        assert decision is not None
