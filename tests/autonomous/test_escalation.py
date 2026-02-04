# tests/autonomous/test_escalation.py
"""
Tests for the Escalation Framework.

Covers:
- Escalation data model serialization
- EscalationManager escalate/respond/acknowledge/resolve
- Deduplication within window
- Auto-resolution of low-priority escalations
- Response waiting with timeout
- Built-in notification handlers (file, callback)
- Status reporting
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.autonomous.escalation import (
    Escalation,
    EscalationLevel,
    EscalationReason,
    callback_handler,
    file_handler,
)

# =============================================================================
# Escalation Data Model Tests
# =============================================================================


class TestEscalation:
    """Tests for Escalation data model."""

    def test_create_escalation(self):
        """Test basic escalation creation."""
        esc = Escalation(
            id="esc_test",
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM Request",
            message="Bot wants to chat",
        )
        assert esc.id == "esc_test"
        assert esc.level == EscalationLevel.ACTION
        assert esc.reason == EscalationReason.DM_REQUEST
        assert esc.is_pending() is True

    def test_is_pending_resolved(self):
        """Test that resolved escalations are not pending."""
        esc = Escalation(
            id="esc_test",
            level=EscalationLevel.INFO,
            reason=EscalationReason.GOAL_COMPLETED,
            title="Done",
            message="Goal complete",
            resolved_at=datetime.utcnow(),
        )
        assert esc.is_pending() is False

    def test_is_pending_expired(self):
        """Test that expired escalations are not pending."""
        esc = Escalation(
            id="esc_test",
            level=EscalationLevel.INFO,
            reason=EscalationReason.GOAL_COMPLETED,
            title="Done",
            message="Goal complete",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert esc.is_pending() is False

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = Escalation(
            id="esc_roundtrip",
            level=EscalationLevel.URGENT,
            reason=EscalationReason.REPEATED_FAILURE,
            title="Test Title",
            message="Test Message",
            details={"retry_count": 5},
            goal_id="goal_123",
            task_id="task_456",
        )
        data = original.to_dict()
        restored = Escalation.from_dict(data)

        assert restored.id == original.id
        assert restored.level == original.level
        assert restored.reason == original.reason
        assert restored.title == original.title
        assert restored.message == original.message
        assert restored.details == original.details
        assert restored.goal_id == original.goal_id
        assert restored.task_id == original.task_id

    def test_serialization_with_dates(self):
        """Test serialization preserves datetime fields."""
        now = datetime.utcnow()
        esc = Escalation(
            id="esc_dates",
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Test",
            message="Test",
            acknowledged_at=now,
            resolved_at=now,
            expires_at=now + timedelta(hours=1),
        )
        data = esc.to_dict()
        restored = Escalation.from_dict(data)

        assert restored.acknowledged_at is not None
        assert restored.resolved_at is not None
        assert restored.expires_at is not None


# =============================================================================
# EscalationManager Tests
# =============================================================================


class TestEscalationManager:
    """Tests for EscalationManager operations."""

    @pytest.mark.asyncio
    async def test_escalate_basic(self, escalation_manager):
        """Test basic escalation creation."""
        result = await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM Request",
            message="Bot wants to chat",
        )

        # No wait_for_response, so returns None
        assert result is None
        assert len(escalation_manager.get_all()) == 1
        assert len(escalation_manager.get_pending()) == 1

    @pytest.mark.asyncio
    async def test_auto_resolve_low_priority(self, escalation_manager):
        """Test that low-priority escalations are auto-resolved."""
        # auto_resolve_below = ADVISORY (20), so DEBUG (0) and INFO (10)
        # should be auto-resolved
        await escalation_manager.escalate(
            level=EscalationLevel.DEBUG,
            reason=EscalationReason.GOAL_COMPLETED,
            title="Debug Message",
            message="Just logging",
        )

        # Should NOT be stored (auto-resolved)
        assert len(escalation_manager.get_all()) == 0

    @pytest.mark.asyncio
    async def test_auto_resolve_info(self, escalation_manager):
        """Test INFO level auto-resolution."""
        await escalation_manager.escalate(
            level=EscalationLevel.INFO,
            reason=EscalationReason.MILESTONE_REACHED,
            title="Milestone",
            message="50% progress",
        )

        # INFO (10) < ADVISORY (20), so auto-resolved
        assert len(escalation_manager.get_all()) == 0

    @pytest.mark.asyncio
    async def test_deduplication(self, escalation_manager):
        """Test that duplicate escalations within window are suppressed."""
        # First escalation
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM Request",
            message="Bot wants to chat",
        )

        # Same title+message+reason = duplicate
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM Request",
            message="Bot wants to chat",
        )

        # Only one should be stored
        assert len(escalation_manager.get_all()) == 1

    @pytest.mark.asyncio
    async def test_deduplication_different_messages(self, escalation_manager):
        """Test that different messages are NOT deduplicated."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM from Bot A",
            message="Message 1",
        )

        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM from Bot B",
            message="Message 2",
        )

        # Different messages = two escalations
        assert len(escalation_manager.get_all()) == 2

    @pytest.mark.asyncio
    async def test_respond(self, escalation_manager):
        """Test providing a human response."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM Request",
            message="Bot wants to chat",
        )

        esc = escalation_manager.get_all()[0]
        success = escalation_manager.respond(esc.id, "accept")

        assert success is True
        assert esc.human_response == "accept"
        assert esc.acknowledged_at is not None
        assert esc.resolved_at is not None
        assert esc.is_pending() is False

    @pytest.mark.asyncio
    async def test_respond_nonexistent(self, escalation_manager):
        """Test responding to non-existent escalation."""
        success = escalation_manager.respond("nonexistent", "accept")
        assert success is False

    @pytest.mark.asyncio
    async def test_acknowledge(self, escalation_manager):
        """Test acknowledging an escalation."""
        await escalation_manager.escalate(
            level=EscalationLevel.URGENT,
            reason=EscalationReason.REPEATED_FAILURE,
            title="Failures",
            message="Multiple failures",
        )

        esc = escalation_manager.get_all()[0]
        success = escalation_manager.acknowledge(esc.id)

        assert success is True
        assert esc.acknowledged_at is not None
        assert esc.is_pending() is True  # Still pending (not resolved)

    @pytest.mark.asyncio
    async def test_resolve(self, escalation_manager):
        """Test resolving an escalation without response."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Approval",
            message="Needs approval",
        )

        esc = escalation_manager.get_all()[0]
        success = escalation_manager.resolve(esc.id)

        assert success is True
        assert esc.resolved_at is not None
        assert esc.is_pending() is False

    @pytest.mark.asyncio
    async def test_get_by_level(self, escalation_manager):
        """Test filtering escalations by level."""
        await escalation_manager.escalate(
            level=EscalationLevel.ADVISORY,
            reason=EscalationReason.RATE_LIMITED,
            title="Rate Limited",
            message="Hit rate limit",
        )
        await escalation_manager.escalate(
            level=EscalationLevel.CRITICAL,
            reason=EscalationReason.RESOURCE_EXHAUSTED,
            title="Budget Exceeded",
            message="Daily budget exceeded",
        )

        critical = escalation_manager.get_by_level(EscalationLevel.CRITICAL)
        assert len(critical) == 1

        action_plus = escalation_manager.get_by_level(EscalationLevel.ACTION)
        assert len(action_plus) == 1  # CRITICAL >= ACTION

        advisory_plus = escalation_manager.get_by_level(EscalationLevel.ADVISORY)
        assert len(advisory_plus) == 2  # Both

    @pytest.mark.asyncio
    async def test_clear_resolved(self, escalation_manager):
        """Test clearing old resolved escalations."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Old",
            message="Old escalation",
        )

        esc = escalation_manager.get_all()[0]
        esc.resolved_at = datetime.utcnow() - timedelta(hours=48)

        cleared = escalation_manager.clear_resolved(older_than_hours=24)
        assert cleared == 1
        assert len(escalation_manager.get_all()) == 0

    @pytest.mark.asyncio
    async def test_clear_resolved_keeps_recent(self, escalation_manager):
        """Test that recent resolved escalations are kept."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Recent",
            message="Recent escalation",
        )

        esc = escalation_manager.get_all()[0]
        esc.resolved_at = datetime.utcnow() - timedelta(hours=1)

        cleared = escalation_manager.clear_resolved(older_than_hours=24)
        assert cleared == 0
        assert len(escalation_manager.get_all()) == 1

    @pytest.mark.asyncio
    async def test_wait_for_response(self, escalation_manager):
        """Test waiting for human response with async response."""

        async def respond_later():
            await asyncio.sleep(0.05)
            pending = escalation_manager.get_pending()
            if pending:
                escalation_manager.respond(pending[0].id, "approved")

        # Start responding in background
        asyncio.create_task(respond_later())

        response = await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Approval Needed",
            message="Please approve",
            wait_for_response=True,
            timeout_seconds=5,
        )

        assert response == "approved"

    @pytest.mark.asyncio
    async def test_wait_for_response_timeout(self, escalation_manager):
        """Test that waiting times out correctly."""
        response = await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Will Timeout",
            message="No one will respond",
            wait_for_response=True,
            timeout_seconds=0.1,
        )

        assert response is None

    @pytest.mark.asyncio
    async def test_handler_notification(self, escalation_manager):
        """Test that handlers are called on escalation."""
        received = []

        async def test_handler(esc):
            received.append(esc)
            return True

        escalation_manager.add_handler(test_handler)

        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="Test Handler",
            message="Should trigger handler",
        )

        assert len(received) == 1
        assert received[0].title == "Test Handler"

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self, escalation_manager):
        """Test that handler errors don't prevent other handlers."""
        results = []

        async def failing_handler(esc):
            raise RuntimeError("Handler failed")

        async def working_handler(esc):
            results.append(esc.title)
            return True

        escalation_manager.add_handler(failing_handler)
        escalation_manager.add_handler(working_handler)

        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="Error Isolation",
            message="First handler fails",
        )

        # Working handler should still execute
        assert len(results) == 1
        assert results[0] == "Error Isolation"

    @pytest.mark.asyncio
    async def test_remove_handler(self, escalation_manager):
        """Test handler removal."""
        call_count = 0

        async def test_handler(esc):
            nonlocal call_count
            call_count += 1
            return True

        escalation_manager.add_handler(test_handler)
        escalation_manager.remove_handler(test_handler)

        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="After Removal",
            message="Handler should not be called",
        )

        assert call_count == 0

    @pytest.mark.asyncio
    async def test_get_status(self, escalation_manager):
        """Test status report generation."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="Status Test",
            message="Test",
        )

        status = escalation_manager.get_status()
        assert status["total_escalations"] == 1
        assert status["pending_count"] == 1
        assert "pending_by_level" in status

    @pytest.mark.asyncio
    async def test_expires_in_seconds(self, escalation_manager):
        """Test escalation expiration."""
        await escalation_manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.APPROVAL_NEEDED,
            title="Expiring",
            message="Will expire",
            expires_in_seconds=0.01,
        )

        # Should be pending initially
        assert len(escalation_manager.get_pending()) == 1

        # Wait for expiry
        await asyncio.sleep(0.05)

        # Should no longer be pending (expired)
        assert len(escalation_manager.get_pending()) == 0


# =============================================================================
# Notification Handler Tests
# =============================================================================


class TestNotificationHandlers:
    """Tests for built-in notification handlers."""

    @pytest.mark.asyncio
    async def test_file_handler(self, tmp_path):
        """Test file-based notification handler."""
        filepath = str(tmp_path / "escalations.jsonl")
        handler = file_handler(filepath)

        esc = Escalation(
            id="esc_file",
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="File Test",
            message="Testing file handler",
        )

        result = await handler(esc)
        assert result is True

        # Verify file content
        content = Path(filepath).read_text().strip()
        data = json.loads(content)
        assert data["id"] == "esc_file"
        assert data["title"] == "File Test"

    @pytest.mark.asyncio
    async def test_file_handler_appends(self, tmp_path):
        """Test that file handler appends (JSON-lines)."""
        filepath = str(tmp_path / "escalations.jsonl")
        handler = file_handler(filepath)

        for i in range(3):
            esc = Escalation(
                id=f"esc_{i}",
                level=EscalationLevel.ACTION,
                reason=EscalationReason.DM_REQUEST,
                title=f"Test {i}",
                message=f"Message {i}",
            )
            await handler(esc)

        lines = Path(filepath).read_text().strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_callback_handler(self):
        """Test callback-based notification handler."""
        received = []

        async def my_callback(esc):
            received.append(esc.title)

        handler = callback_handler(my_callback)

        esc = Escalation(
            id="esc_cb",
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="Callback Test",
            message="Testing callback",
        )

        result = await handler(esc)
        assert result is True
        assert received == ["Callback Test"]

    @pytest.mark.asyncio
    async def test_callback_handler_error(self):
        """Test callback handler error returns False."""

        async def failing_callback(esc):
            raise ValueError("Callback error")

        handler = callback_handler(failing_callback)

        esc = Escalation(
            id="esc_fail",
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="Fail Test",
            message="Testing failure",
        )

        result = await handler(esc)
        assert result is False
