# tests/agents/hitl/test_callbacks.py
"""
Tests for HITL UI Callbacks.

Tests:
- AutoApproveCallback
- QueueHITLCallback
- ConsoleHITLCallback (mocked)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from llmcore.agents.hitl import (
    ActivityInfo,
    ApprovalScope,
    ApprovalStatus,
    AutoApproveCallback,
    ConsoleHITLCallback,
    HITLCallback,
    HITLDecision,
    HITLRequest,
    HITLResponse,
    QueueHITLCallback,
    RiskAssessment,
    RiskLevel,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_request():
    """Create sample HITL request."""
    activity = ActivityInfo(
        activity_type="bash_exec",
        parameters={"command": "ls -la"}
    )
    risk = RiskAssessment(
        overall_level="medium",
        requires_approval=True,
        factors=[]
    )
    request = HITLRequest(
        activity=activity,
        risk_assessment=risk,
        session_id="test-session",
        user_id="test-user",
        context_summary="Test context"
    )
    request.set_expiration(300)
    return request


@pytest.fixture
def sample_decision(sample_request):
    """Create sample decision."""
    return HITLDecision(
        status=ApprovalStatus.APPROVED,
        reason="Test approved",
        request=sample_request
    )


# =============================================================================
# AUTO APPROVE CALLBACK TESTS
# =============================================================================


class TestAutoApproveCallback:
    """Test AutoApproveCallback for testing/automation."""

    @pytest.mark.asyncio
    async def test_auto_approve_default(self, sample_request):
        """Should auto-approve by default."""
        callback = AutoApproveCallback()
        response = await callback.request_approval(sample_request)

        assert response.approved
        assert response.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_auto_reject(self, sample_request):
        """Should auto-reject when configured."""
        # Use approve_all=False instead of approve=False
        callback = AutoApproveCallback(approve_all=False)
        response = await callback.request_approval(sample_request)

        assert not response.approved

    @pytest.mark.asyncio
    async def test_auto_approve_with_delay(self, sample_request):
        """Should respect delay setting."""
        callback = AutoApproveCallback(delay_seconds=0.1)

        start = asyncio.get_event_loop().time()
        response = await callback.request_approval(sample_request)
        elapsed = asyncio.get_event_loop().time() - start

        assert response.approved
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_notify_timeout(self, sample_request):
        """Should handle timeout notification."""
        callback = AutoApproveCallback()
        # Should not raise
        await callback.notify_timeout(sample_request)

    @pytest.mark.asyncio
    async def test_notify_result(self, sample_request, sample_decision):
        """Should handle result notification."""
        callback = AutoApproveCallback()
        # Should not raise
        await callback.notify_result(sample_request, sample_decision)


# =============================================================================
# QUEUE CALLBACK TESTS
# =============================================================================


class TestQueueHITLCallback:
    """Test QueueHITLCallback for async UI integration."""

    @pytest.mark.asyncio
    async def test_request_queued(self, sample_request):
        """Should queue request for external processing."""
        callback = QueueHITLCallback()

        # Start request in background
        request_task = asyncio.create_task(
            callback.request_approval(sample_request)
        )

        # Give it time to queue
        await asyncio.sleep(0.01)

        # Check request is queued
        queued = await callback.get_pending_request()
        assert queued is not None
        assert queued.request_id == sample_request.request_id

        # Submit response
        response = HITLResponse(
            request_id=sample_request.request_id,
            approved=True
        )
        await callback.submit_response(response)

        # Get result
        result = await request_task
        assert result.approved

    @pytest.mark.asyncio
    async def test_multiple_requests(self, sample_request):
        """Should handle multiple queued requests."""
        callback = QueueHITLCallback()

        # Queue multiple requests
        tasks = []
        requests = []
        for i in range(3):
            activity = ActivityInfo(activity_type=f"tool_{i}", parameters={})
            risk = RiskAssessment(overall_level="medium")
            req = HITLRequest(activity=activity, risk_assessment=risk)
            req.set_expiration(300)
            requests.append(req)
            tasks.append(asyncio.create_task(callback.request_approval(req)))

        await asyncio.sleep(0.01)

        # Process all requests
        for i, req in enumerate(requests):
            pending = await callback.get_pending_request()
            assert pending is not None

            response = HITLResponse(
                request_id=pending.request_id,
                approved=True,
                feedback=f"Approved {i}"
            )
            await callback.submit_response(response)

        # All tasks should complete
        results = await asyncio.gather(*tasks)
        assert all(r.approved for r in results)

    @pytest.mark.asyncio
    async def test_reject_via_queue(self, sample_request):
        """Should handle rejection through queue."""
        callback = QueueHITLCallback()

        task = asyncio.create_task(callback.request_approval(sample_request))
        await asyncio.sleep(0.01)

        # Submit rejection
        response = HITLResponse(
            request_id=sample_request.request_id,
            approved=False,
            feedback="Too risky"
        )
        await callback.submit_response(response)

        result = await task
        assert not result.approved
        assert result.feedback == "Too risky"


# =============================================================================
# CONSOLE CALLBACK TESTS (MOCKED)
# =============================================================================


class TestConsoleHITLCallback:
    """Test ConsoleHITLCallback with mocked I/O."""

    @pytest.mark.asyncio
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_approve_via_console(self, mock_print, mock_input, sample_request):
        """Should handle 'y' input as approval."""
        mock_input.return_value = 'y'

        callback = ConsoleHITLCallback()
        response = await callback.request_approval(sample_request)

        assert response.approved

    @pytest.mark.asyncio
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_reject_via_console(self, mock_print, mock_input, sample_request):
        """Should handle 'n' input as rejection."""
        mock_input.return_value = 'n'

        callback = ConsoleHITLCallback()
        response = await callback.request_approval(sample_request)

        assert not response.approved


# =============================================================================
# CALLBACK INTERFACE TESTS
# =============================================================================


class TestCallbackInterface:
    """Test callback interface compliance."""

    def test_abstract_base_class(self):
        """HITLCallback should be abstract."""
        # Cannot instantiate directly
        with pytest.raises(TypeError):
            HITLCallback()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("callback_class,kwargs", [
        (AutoApproveCallback, {}),
        (QueueHITLCallback, {}),
    ])
    async def test_callback_interface(self, callback_class, kwargs, sample_request, sample_decision):
        """All callbacks should implement interface."""
        callback = callback_class(**kwargs)

        # Check interface methods exist
        assert hasattr(callback, 'request_approval')
        assert hasattr(callback, 'notify_timeout')
        assert hasattr(callback, 'notify_result')

        # Test actual calls don't raise
        if callback_class == AutoApproveCallback:
            response = await callback.request_approval(sample_request)
            assert isinstance(response, HITLResponse)

        await callback.notify_timeout(sample_request)
        await callback.notify_result(sample_request, sample_decision)


# =============================================================================
# EDGE CASES
# =============================================================================


class TestCallbackEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_queue_response_for_unknown_request(self):
        """Should handle response for unknown request."""
        callback = QueueHITLCallback()

        response = HITLResponse(
            request_id="unknown-id",
            approved=True
        )

        # Should not raise, but log warning
        await callback.submit_response(response)

    @pytest.mark.asyncio
    async def test_auto_approve_zero_delay(self, sample_request):
        """Should handle zero delay."""
        callback = AutoApproveCallback(delay_seconds=0)
        response = await callback.request_approval(sample_request)
        assert response.approved

    @pytest.mark.asyncio
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_console_keyboard_interrupt(self, mock_print, mock_input, sample_request):
        """Should handle KeyboardInterrupt gracefully."""
        mock_input.side_effect = KeyboardInterrupt()

        callback = ConsoleHITLCallback()
        response = await callback.request_approval(sample_request)

        # Should reject on interrupt
        assert not response.approved

    @pytest.mark.asyncio
    @patch('builtins.input')
    @patch('builtins.print')
    async def test_console_eof_error(self, mock_print, mock_input, sample_request):
        """Should handle EOFError gracefully."""
        mock_input.side_effect = EOFError()

        callback = ConsoleHITLCallback()
        response = await callback.request_approval(sample_request)

        # Should reject on EOF
        assert not response.approved
