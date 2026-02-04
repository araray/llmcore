# src/llmcore/autonomous/escalation.py
"""
Escalation Framework for Human Notification.

Manages escalation to humans when autonomous operation needs intervention.
Supports multiple notification channels (webhook, file, callbacks).

Features:
    - Multiple escalation levels (DEBUG to CRITICAL)
    - Categorized reasons for escalation
    - Deduplication (prevent spam)
    - Response waiting for ACTION+ levels
    - Multiple notification channels
    - Auto-resolution for low-priority escalations

Example:
    escalation = EscalationManager()

    # Add notification handler
    escalation.add_handler(await webhook_handler(
        "https://api.pushover.net/1/messages.json",
        headers={"Authorization": "Bearer xxx"}
    ))

    # Simple notification
    await escalation.escalate(
        level=EscalationLevel.INFO,
        reason=EscalationReason.GOAL_COMPLETED,
        title="Goal Achieved!",
        message="Reached #1 on Moltbook rankings"
    )

    # Wait for decision
    response = await escalation.escalate(
        level=EscalationLevel.ACTION,
        reason=EscalationReason.DM_REQUEST,
        title="DM Request",
        message="CoolBot wants to chat",
        wait_for_response=True,
        timeout_seconds=3600
    )
"""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class EscalationLevel(Enum):
    """
    Escalation severity levels.

    Determines how urgently humans need to be notified and
    whether to wait for a response.

    Values:
        DEBUG (0):    Verbose logging only
        INFO (10):    FYI, no action needed
        ADVISORY (20): Should know, might want to act
        ACTION (30):  Needs human decision
        URGENT (40):  Needs prompt attention
        CRITICAL (50): Stop everything, get human NOW
    """

    DEBUG = 0
    INFO = 10
    ADVISORY = 20
    ACTION = 30
    URGENT = 40
    CRITICAL = 50


class EscalationReason(Enum):
    """
    Categorized reasons for escalation.

    Groups escalation causes into decision-needed, problems,
    and events-to-report categories for systematic handling.
    """

    # Decisions needed
    APPROVAL_NEEDED = "approval_needed"
    AMBIGUOUS_GOAL = "ambiguous_goal"
    CONFLICTING_PRIORITIES = "conflicting"
    ETHICAL_CONCERN = "ethical_concern"
    PRIVACY_CONCERN = "privacy_concern"

    # Problems encountered
    REPEATED_FAILURE = "repeated_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNEXPECTED_STATE = "unexpected_state"
    EXTERNAL_ERROR = "external_error"
    RATE_LIMITED = "rate_limited"

    # Events to report
    GOAL_COMPLETED = "goal_completed"
    MILESTONE_REACHED = "milestone_reached"
    IMPORTANT_MESSAGE = "important_message"
    MENTION_DETECTED = "mention_detected"
    DM_REQUEST = "dm_request"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Escalation:
    """
    An escalation event requiring human attention.

    Tracks the full lifecycle of an escalation from creation through
    acknowledgement to resolution, including optional human response.

    Attributes:
        id: Unique identifier
        level: Severity level
        reason: Categorized reason
        title: Short title for notification
        message: Detailed message
        details: Additional context (dict)
        goal_id: Related goal ID (if any)
        task_id: Related task ID (if any)
        created_at: When escalation was created
        acknowledged_at: When human acknowledged
        resolved_at: When escalation was resolved
        expires_at: When escalation expires (auto-resolve)
        human_response: Response from human
        auto_resolved: Whether auto-resolved (low priority)
    """

    id: str
    level: EscalationLevel
    reason: EscalationReason
    title: str
    message: str

    # Context
    details: Dict[str, Any] = field(default_factory=dict)
    goal_id: Optional[str] = None
    task_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Response
    human_response: Optional[str] = None
    auto_resolved: bool = False

    def is_pending(self) -> bool:
        """
        Check if escalation is still pending resolution.

        An escalation is no longer pending if:
        - It has been resolved
        - It has expired

        Returns:
            True if still awaiting resolution
        """
        if self.resolved_at:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/notification payloads."""
        return {
            "id": self.id,
            "level": self.level.name,
            "reason": self.reason.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "goal_id": self.goal_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": (self.acknowledged_at.isoformat() if self.acknowledged_at else None),
            "resolved_at": (self.resolved_at.isoformat() if self.resolved_at else None),
            "expires_at": (self.expires_at.isoformat() if self.expires_at else None),
            "human_response": self.human_response,
            "auto_resolved": self.auto_resolved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Escalation":
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary with escalation fields

        Returns:
            Escalation instance
        """

        def parse_datetime(value):
            if value is None:
                return None
            if isinstance(value, datetime):
                return value
            return datetime.fromisoformat(value)

        return cls(
            id=data["id"],
            level=EscalationLevel[data["level"]],
            reason=EscalationReason(data["reason"]),
            title=data["title"],
            message=data["message"],
            details=data.get("details", {}),
            goal_id=data.get("goal_id"),
            task_id=data.get("task_id"),
            created_at=(parse_datetime(data.get("created_at")) or datetime.utcnow()),
            acknowledged_at=parse_datetime(data.get("acknowledged_at")),
            resolved_at=parse_datetime(data.get("resolved_at")),
            expires_at=parse_datetime(data.get("expires_at")),
            human_response=data.get("human_response"),
            auto_resolved=data.get("auto_resolved", False),
        )


# Type for notification handlers
NotificationHandler = Callable[[Escalation], Awaitable[bool]]


# =============================================================================
# EscalationManager
# =============================================================================


class EscalationManager:
    """
    Manages escalation to humans during autonomous operation.

    Features:
        - Multiple notification channels
        - Escalation levels with auto-resolution
        - Response tracking and waiting
        - Deduplication to prevent spam

    Example:
        manager = EscalationManager(
            auto_resolve_below=EscalationLevel.ADVISORY,
            dedup_window_seconds=300
        )

        # Add handlers
        manager.add_handler(file_handler("/var/log/escalations.log"))
        manager.add_handler(await webhook_handler("https://..."))

        # Escalate
        response = await manager.escalate(
            level=EscalationLevel.ACTION,
            reason=EscalationReason.DM_REQUEST,
            title="DM Request",
            message="CoolBot wants to chat",
            wait_for_response=True
        )
    """

    def __init__(
        self,
        auto_resolve_below: EscalationLevel = EscalationLevel.ADVISORY,
        dedup_window_seconds: int = 300,
    ):
        """
        Initialize EscalationManager.

        Args:
            auto_resolve_below: Escalations below this level are auto-resolved
            dedup_window_seconds: Window for deduplication (same title+message)
        """
        self.auto_resolve_below = auto_resolve_below
        self.dedup_window = dedup_window_seconds

        self._escalations: Dict[str, Escalation] = {}
        self._handlers: List[NotificationHandler] = []
        self._recent_hashes: Dict[str, datetime] = {}
        self._response_waiters: Dict[str, asyncio.Event] = {}

    def add_handler(self, handler: NotificationHandler) -> None:
        """
        Add a notification handler.

        Handlers are called for all escalations that aren't auto-resolved.

        Args:
            handler: Async function that takes Escalation and returns bool
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: NotificationHandler) -> None:
        """Remove a notification handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def _compute_hash(self, escalation: Escalation) -> str:
        """Compute hash for deduplication."""
        content = f"{escalation.reason.value}:{escalation.title}:{escalation.message}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _is_duplicate(self, escalation: Escalation) -> bool:
        """Check if this is a duplicate within dedup window."""
        hash_key = self._compute_hash(escalation)
        now = datetime.utcnow()

        # Clean old hashes
        self._recent_hashes = {
            k: v
            for k, v in self._recent_hashes.items()
            if (now - v).total_seconds() < self.dedup_window
        }

        if hash_key in self._recent_hashes:
            return True

        self._recent_hashes[hash_key] = now
        return False

    async def escalate(
        self,
        level: EscalationLevel,
        reason: EscalationReason,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        goal_id: Optional[str] = None,
        task_id: Optional[str] = None,
        wait_for_response: bool = False,
        timeout_seconds: float = 3600,
        expires_in_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """
        Create an escalation.

        Args:
            level: Severity level
            reason: Categorized reason
            title: Short title for notifications
            message: Detailed message
            details: Additional context
            goal_id: Related goal ID
            task_id: Related task ID
            wait_for_response: Block until human responds (for ACTION+)
            timeout_seconds: How long to wait for response
            expires_in_seconds: Auto-expire after this time

        Returns:
            Human response if wait_for_response=True, else None

        Example:
            # Simple notification
            await escalation.escalate(
                level=EscalationLevel.INFO,
                reason=EscalationReason.GOAL_COMPLETED,
                title="Goal Achieved!",
                message="Reached #1 on Moltbook rankings"
            )

            # Wait for decision
            response = await escalation.escalate(
                level=EscalationLevel.ACTION,
                reason=EscalationReason.DM_REQUEST,
                title="New DM Request",
                message="Agent 'CoolBot' wants to start a conversation",
                details={"agent": "CoolBot", "preview": "Hi there!"},
                wait_for_response=True,
                timeout_seconds=3600
            )
        """
        escalation = Escalation(
            id=f"esc_{uuid.uuid4().hex[:12]}",
            level=level,
            reason=reason,
            title=title,
            message=message,
            details=details or {},
            goal_id=goal_id,
            task_id=task_id,
        )

        if expires_in_seconds:
            escalation.expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)

        # Check for duplicate
        if self._is_duplicate(escalation):
            logger.debug(f"Duplicate escalation suppressed: {title}")
            return None

        # Auto-resolve low priority
        if level.value < self.auto_resolve_below.value:
            escalation.auto_resolved = True
            escalation.resolved_at = datetime.utcnow()
            logger.info(f"Auto-resolved escalation: {title}")
            return None

        # Store escalation
        self._escalations[escalation.id] = escalation

        # Log with appropriate level
        log_levels = {
            EscalationLevel.DEBUG: logging.DEBUG,
            EscalationLevel.INFO: logging.INFO,
            EscalationLevel.ADVISORY: logging.INFO,
            EscalationLevel.ACTION: logging.WARNING,
            EscalationLevel.URGENT: logging.WARNING,
            EscalationLevel.CRITICAL: logging.ERROR,
        }
        log_level = log_levels.get(level, logging.INFO)
        logger.log(
            log_level,
            f"ESCALATION [{level.name}] {reason.value}: {title} - {message}",
        )

        # Notify all handlers
        for handler in self._handlers:
            try:
                success = await handler(escalation)
                if not success:
                    logger.warning("Escalation handler returned failure")
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")

        # Wait for response if requested (and level warrants it)
        if wait_for_response and level.value >= EscalationLevel.ACTION.value:
            return await self._wait_for_response(escalation.id, timeout_seconds)

        return None

    async def _wait_for_response(
        self,
        escalation_id: str,
        timeout: float,
    ) -> Optional[str]:
        """Wait for human response to an escalation."""
        event = asyncio.Event()
        self._response_waiters[escalation_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            escalation = self._escalations.get(escalation_id)
            return escalation.human_response if escalation else None
        except asyncio.TimeoutError:
            logger.warning(f"Escalation {escalation_id} timed out waiting for response")
            return None
        finally:
            self._response_waiters.pop(escalation_id, None)

    def respond(
        self,
        escalation_id: str,
        response: str,
    ) -> bool:
        """
        Provide a human response to an escalation.

        Args:
            escalation_id: ID of the escalation
            response: Human's response

        Returns:
            True if escalation found and updated
        """
        if escalation_id not in self._escalations:
            logger.warning(f"Escalation not found: {escalation_id}")
            return False

        escalation = self._escalations[escalation_id]
        escalation.human_response = response
        escalation.acknowledged_at = datetime.utcnow()
        escalation.resolved_at = datetime.utcnow()

        # Wake up any waiters
        if escalation_id in self._response_waiters:
            self._response_waiters[escalation_id].set()

        logger.info(f"Escalation {escalation_id} responded: {response}")
        return True

    def acknowledge(self, escalation_id: str) -> bool:
        """
        Acknowledge an escalation (mark as seen, but not resolved).

        Args:
            escalation_id: ID of the escalation

        Returns:
            True if escalation found and updated
        """
        if escalation_id not in self._escalations:
            return False

        self._escalations[escalation_id].acknowledged_at = datetime.utcnow()
        return True

    def resolve(self, escalation_id: str) -> bool:
        """
        Resolve an escalation without a response.

        Args:
            escalation_id: ID of the escalation

        Returns:
            True if escalation found and resolved
        """
        if escalation_id not in self._escalations:
            return False

        escalation = self._escalations[escalation_id]
        escalation.resolved_at = datetime.utcnow()

        # Wake up any waiters with None response
        if escalation_id in self._response_waiters:
            self._response_waiters[escalation_id].set()

        return True

    def get_pending(self) -> List[Escalation]:
        """Get all pending (unresolved) escalations."""
        return [e for e in self._escalations.values() if e.is_pending()]

    def get_escalation(self, escalation_id: str) -> Optional[Escalation]:
        """
        Get an escalation by ID.

        Args:
            escalation_id: ID of the escalation to retrieve.

        Returns:
            The Escalation object, or None if not found.
        """
        return self._escalations.get(escalation_id)

    def get_all(self) -> List[Escalation]:
        """Get all escalations."""
        return list(self._escalations.values())

    def get_by_level(self, level: EscalationLevel) -> List[Escalation]:
        """Get escalations at or above a specific level."""
        return [e for e in self._escalations.values() if e.level.value >= level.value]

    def clear_resolved(self, older_than_hours: int = 24) -> int:
        """
        Clear resolved escalations older than specified hours.

        Args:
            older_than_hours: Clear escalations older than this

        Returns:
            Number of escalations cleared
        """
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        to_remove = [
            eid for eid, e in self._escalations.items() if e.resolved_at and e.resolved_at < cutoff
        ]

        for eid in to_remove:
            del self._escalations[eid]

        return len(to_remove)

    def get_status(self) -> Dict[str, Any]:
        """Get escalation manager status."""
        pending = self.get_pending()
        by_level: Dict[str, int] = {}
        for level in EscalationLevel:
            by_level[level.name] = sum(1 for e in pending if e.level == level)

        return {
            "total_escalations": len(self._escalations),
            "pending_count": len(pending),
            "pending_by_level": by_level,
            "handler_count": len(self._handlers),
        }


# =============================================================================
# Built-in Notification Handlers
# =============================================================================


async def webhook_handler(
    webhook_url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30,
) -> NotificationHandler:
    """
    Create a webhook notification handler.

    Args:
        webhook_url: URL to POST to
        headers: Additional headers (e.g., Authorization)
        timeout: Request timeout

    Returns:
        Handler function

    Example:
        handler = await webhook_handler(
            "https://api.pushover.net/1/messages.json",
            headers={"Authorization": "Bearer xxx"}
        )
        escalation_manager.add_handler(handler)
    """

    async def handler(escalation: Escalation) -> bool:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = escalation.to_dict()
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers or {},
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    success = resp.status < 400
                    if not success:
                        logger.warning(f"Webhook returned {resp.status}: {await resp.text()}")
                    return success
        except ImportError:
            logger.error(
                "aiohttp not installed, webhook handler unavailable. "
                "Install with: pip install aiohttp"
            )
            return False
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False

    return handler


def file_handler(filepath: str) -> NotificationHandler:
    """
    Create a file-based notification handler.

    Appends JSON-lines to the specified file.

    Args:
        filepath: Path to log file

    Returns:
        Handler function

    Example:
        handler = file_handler("/var/log/escalations.jsonl")
        escalation_manager.add_handler(handler)
    """
    import json
    from pathlib import Path

    async def handler(escalation: Escalation) -> bool:
        try:
            path = Path(filepath).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a") as f:
                f.write(json.dumps(escalation.to_dict()) + "\n")

            return True
        except Exception as e:
            logger.error(f"File notification failed: {e}")
            return False

    return handler


def callback_handler(
    callback: Callable[[Escalation], Awaitable[None]],
) -> NotificationHandler:
    """
    Create a callback-based notification handler.

    Args:
        callback: Async function to call with escalation

    Returns:
        Handler function
    """

    async def handler(escalation: Escalation) -> bool:
        try:
            await callback(escalation)
            return True
        except Exception as e:
            logger.error(f"Callback notification failed: {e}")
            return False

    return handler
