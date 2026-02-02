# src/llmcore/agents/hitl/callbacks.py
"""
HITL Callback Interfaces for UI Integration.

Provides abstract and concrete callback implementations for:
- CLI/REPL interfaces
- Web interfaces (async/websocket)
- Custom integrations

The callback interface allows frontends to:
- Display approval requests to users
- Collect user responses
- Handle timeouts gracefully

References:
    - Master Plan: Section 20.3 (HITL Callbacks)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from .models import (
    ApprovalScope,
    ApprovalStatus,
    HITLDecision,
    HITLRequest,
    HITLResponse,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT CALLBACK
# =============================================================================


class HITLCallback(ABC):
    """
    Abstract callback interface for HITL UI integration.

    Implementations should:
    - Display approval requests to users
    - Collect and return user responses
    - Handle timeouts and errors gracefully
    """

    @abstractmethod
    async def request_approval(
        self,
        request: HITLRequest,
    ) -> HITLResponse:
        """
        Request approval from human.

        Args:
            request: Approval request

        Returns:
            HITLResponse with user's decision
        """
        ...

    @abstractmethod
    async def notify_timeout(self, request: HITLRequest) -> None:
        """
        Notify UI of request timeout.

        Args:
            request: Request that timed out
        """
        ...

    @abstractmethod
    async def notify_result(
        self,
        request: HITLRequest,
        decision: HITLDecision,
    ) -> None:
        """
        Notify UI of final decision.

        Args:
            request: Original request
            decision: Final decision
        """
        ...

    async def batch_request_approval(
        self,
        requests: List[HITLRequest],
    ) -> List[HITLResponse]:
        """
        Request approval for multiple similar requests at once.

        Default implementation requests each individually.

        Args:
            requests: List of similar requests

        Returns:
            List of responses
        """
        responses = []
        for request in requests:
            response = await self.request_approval(request)
            responses.append(response)
        return responses


# =============================================================================
# CONSOLE CALLBACK (CLI/REPL)
# =============================================================================


class ConsoleHITLCallback(HITLCallback):
    """
    Console-based HITL callback for CLI/REPL interfaces.

    Displays requests to terminal and reads user input.
    """

    def __init__(
        self,
        input_fn: Optional[Callable[[], str]] = None,
        output_fn: Optional[Callable[[str], None]] = None,
        use_colors: bool = True,
    ):
        """
        Initialize console callback.

        Args:
            input_fn: Custom input function (default: asyncio-wrapped input)
            output_fn: Custom output function (default: print)
            use_colors: Whether to use ANSI colors
        """
        self._input_fn = input_fn
        self._output_fn = output_fn or print
        self._use_colors = use_colors

    def _format_risk(self, level: str) -> str:
        """Format risk level with color."""
        if not self._use_colors:
            return level.upper()

        colors = {
            "none": "\033[92m",  # Green
            "low": "\033[94m",  # Blue
            "medium": "\033[93m",  # Yellow
            "high": "\033[91m",  # Red
            "critical": "\033[95m",  # Magenta
        }
        reset = "\033[0m"
        color = colors.get(level.lower(), "")
        return f"{color}{level.upper()}{reset}"

    def _format_box(self, title: str, content: List[str], width: int = 60) -> str:
        """Format content in a box."""
        lines = []
        lines.append("┌" + "─" * (width - 2) + "┐")
        lines.append(f"│ {title.center(width - 4)} │")
        lines.append("├" + "─" * (width - 2) + "┤")

        for line in content:
            # Truncate if too long
            if len(line) > width - 4:
                line = line[: width - 7] + "..."
            lines.append(f"│ {line.ljust(width - 4)} │")

        lines.append("└" + "─" * (width - 2) + "┘")
        return "\n".join(lines)

    async def _async_input(self, prompt: str) -> str:
        """Get input asynchronously."""
        if self._input_fn:
            return self._input_fn()

        # Run input() in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))

    async def request_approval(
        self,
        request: HITLRequest,
    ) -> HITLResponse:
        """Display approval request and get user response."""
        # Format request display
        risk_display = self._format_risk(request.risk_assessment.overall_level)

        content = [
            f"Activity: {request.activity.activity_type}",
            f"Risk Level: {risk_display}",
            "",
            "Parameters:",
        ]

        # Format parameters
        for key, value in request.activity.parameters.items():
            value_str = str(value)
            if len(value_str) > 40:
                value_str = value_str[:37] + "..."
            content.append(f"  {key}: {value_str}")

        content.append("")
        content.append(f"Reason: {request.risk_assessment.reason}")

        if request.expires_at:
            remaining = request.time_remaining_seconds
            content.append(f"Expires in: {remaining}s")

        # Display the box
        box = self._format_box("⚠️  APPROVAL REQUIRED", content)
        self._output_fn("\n" + box)

        # Show options
        self._output_fn("\nOptions:")
        self._output_fn("  [y]es     - Approve this action")
        self._output_fn("  [n]o      - Reject this action")
        self._output_fn("  [m]odify  - Approve with modifications")
        self._output_fn("  [s]cope   - Approve and grant scope for similar actions")
        self._output_fn("  [d]etails - Show more details")
        self._output_fn("")

        while True:
            try:
                response = await self._async_input("Your choice [y/n/m/s/d]: ")
                response = response.strip().lower()

                if response in ("y", "yes"):
                    return HITLResponse(
                        request_id=request.request_id,
                        approved=True,
                        feedback="Approved by user",
                        responder_id="console_user",
                    )

                elif response in ("n", "no"):
                    # Ask for reason
                    reason = await self._async_input("Reason for rejection (optional): ")
                    return HITLResponse(
                        request_id=request.request_id,
                        approved=False,
                        feedback=reason.strip() or "Rejected by user",
                        responder_id="console_user",
                    )

                elif response in ("m", "modify"):
                    # Get modified parameters
                    self._output_fn("Enter modified parameters as JSON:")
                    params_str = await self._async_input("> ")
                    try:
                        modified = json.loads(params_str)
                        return HITLResponse(
                            request_id=request.request_id,
                            approved=True,
                            modified_parameters=modified,
                            feedback="Approved with modifications",
                            responder_id="console_user",
                        )
                    except json.JSONDecodeError:
                        self._output_fn("Invalid JSON. Please try again.")

                elif response in ("s", "scope"):
                    # Grant scope
                    self._output_fn("Grant scope for:")
                    self._output_fn("  [t]ool   - All uses of this tool")
                    self._output_fn("  [p]attern - Actions matching a pattern")
                    scope_choice = await self._async_input("Scope type [t/p]: ")

                    scope_grant = None
                    if scope_choice.lower() in ("t", "tool"):
                        scope_grant = ApprovalScope.TOOL
                    elif scope_choice.lower() in ("p", "pattern"):
                        scope_grant = ApprovalScope.PATTERN

                    return HITLResponse(
                        request_id=request.request_id,
                        approved=True,
                        feedback="Approved with scope grant",
                        responder_id="console_user",
                        scope_grant=scope_grant,
                    )

                elif response in ("d", "details"):
                    # Show full details
                    self._output_fn("\n--- Full Details ---")
                    self._output_fn(f"Request ID: {request.request_id}")
                    self._output_fn(f"Activity: {request.activity.activity_type}")
                    self._output_fn(
                        f"Parameters:\n{json.dumps(request.activity.parameters, indent=2)}"
                    )
                    self._output_fn("\nRisk Assessment:")
                    self._output_fn(f"  Level: {request.risk_assessment.overall_level}")
                    self._output_fn(
                        f"  Requires Approval: {request.risk_assessment.requires_approval}"
                    )
                    self._output_fn(f"  Reason: {request.risk_assessment.reason}")
                    if request.risk_assessment.factors:
                        self._output_fn("  Factors:")
                        for factor in request.risk_assessment.factors:
                            self._output_fn(
                                f"    - {factor.name}: {factor.level} ({factor.reason})"
                            )
                    if request.risk_assessment.dangerous_patterns:
                        self._output_fn(
                            f"  Dangerous Patterns: {request.risk_assessment.dangerous_patterns}"
                        )
                    self._output_fn(f"\nContext: {request.context_summary}")
                    self._output_fn("---\n")

                else:
                    self._output_fn("Invalid option. Please enter y, n, m, s, or d.")

            except EOFError:
                # Handle Ctrl+D
                return HITLResponse(
                    request_id=request.request_id,
                    approved=False,
                    feedback="Input terminated",
                    responder_id="console_user",
                )
            except KeyboardInterrupt:
                # Handle Ctrl+C
                self._output_fn("\nCancelled.")
                return HITLResponse(
                    request_id=request.request_id,
                    approved=False,
                    feedback="Cancelled by user",
                    responder_id="console_user",
                )

    async def notify_timeout(self, request: HITLRequest) -> None:
        """Notify user of timeout."""
        self._output_fn(
            f"\n⏰ Request {request.request_id[:8]}... timed out "
            f"(activity: {request.activity.activity_type})"
        )

    async def notify_result(
        self,
        request: HITLRequest,
        decision: HITLDecision,
    ) -> None:
        """Notify user of result."""
        status_icons = {
            ApprovalStatus.APPROVED: "✅",
            ApprovalStatus.REJECTED: "❌",
            ApprovalStatus.MODIFIED: "📝",
            ApprovalStatus.TIMEOUT: "⏰",
            ApprovalStatus.AUTO_APPROVED: "🤖",
        }
        icon = status_icons.get(decision.status, "❓")

        self._output_fn(
            f"\n{icon} {decision.status.value.upper()}: "
            f"{request.activity.activity_type} - {decision.reason}"
        )


# =============================================================================
# AUTO-APPROVE CALLBACK (Testing/Automation)
# =============================================================================


class AutoApproveCallback(HITLCallback):
    """
    Callback that auto-approves all requests.

    For testing and automated scenarios only.
    """

    def __init__(
        self,
        approve_all: bool = True,
        delay_seconds: float = 0.0,
        log_requests: bool = True,
    ):
        """
        Initialize auto-approve callback.

        Args:
            approve_all: If True, approve all. If False, reject all.
            delay_seconds: Delay before responding
            log_requests: Whether to log requests
        """
        self.approve_all = approve_all
        self.delay_seconds = delay_seconds
        self.log_requests = log_requests
        self._requests: List[HITLRequest] = []

    async def request_approval(
        self,
        request: HITLRequest,
    ) -> HITLResponse:
        """Auto-approve or reject request."""
        if self.log_requests:
            logger.info(
                f"Auto-{'approving' if self.approve_all else 'rejecting'} "
                f"{request.activity.activity_type}"
            )
            self._requests.append(request)

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        return HITLResponse(
            request_id=request.request_id,
            approved=self.approve_all,
            feedback="Auto-approved" if self.approve_all else "Auto-rejected",
            responder_id="auto_callback",
        )

    async def notify_timeout(self, request: HITLRequest) -> None:
        """Log timeout."""
        if self.log_requests:
            logger.info(f"Request timed out: {request.request_id}")

    async def notify_result(
        self,
        request: HITLRequest,
        decision: HITLDecision,
    ) -> None:
        """Log result."""
        if self.log_requests:
            logger.info(f"Result: {request.activity.activity_type} -> {decision.status.value}")

    def get_requests(self) -> List[HITLRequest]:
        """Get all requests received."""
        return self._requests.copy()


# =============================================================================
# QUEUE-BASED CALLBACK (For async UI)
# =============================================================================


class QueueHITLCallback(HITLCallback):
    """
    Queue-based callback for async UI integration.

    Requests are put in a queue and responses are received from another queue.
    Useful for web interfaces where request/response is async.
    """

    def __init__(self):
        """Initialize with async queues."""
        self._request_queue: asyncio.Queue[HITLRequest] = asyncio.Queue()
        self._response_queue: asyncio.Queue[HITLResponse] = asyncio.Queue()
        self._pending: Dict[str, HITLRequest] = {}
        self._timeout_handlers: List[Callable] = []
        self._result_handlers: List[Callable] = []

    async def request_approval(
        self,
        request: HITLRequest,
    ) -> HITLResponse:
        """Put request in queue and wait for response."""
        self._pending[request.request_id] = request
        await self._request_queue.put(request)

        # Wait for response (with timeout)
        try:
            timeout = request.time_remaining_seconds if request.expires_at else 300
            response = await asyncio.wait_for(
                self._wait_for_response(request.request_id),
                timeout=max(1, timeout),
            )
            return response
        except asyncio.TimeoutError:
            return HITLResponse(
                request_id=request.request_id,
                approved=False,
                feedback="Timed out waiting for response",
                responder_id="queue_callback",
            )
        finally:
            self._pending.pop(request.request_id, None)

    async def _wait_for_response(self, request_id: str) -> HITLResponse:
        """Wait for response for specific request."""
        while True:
            response = await self._response_queue.get()
            if response.request_id == request_id:
                return response
            # Put back responses for other requests
            await self._response_queue.put(response)
            await asyncio.sleep(0.01)

    async def submit_response(self, response: HITLResponse) -> None:
        """Submit response from UI."""
        await self._response_queue.put(response)

    async def get_pending_request(self) -> Optional[HITLRequest]:
        """Get next pending request (for UI to display)."""
        try:
            return self._request_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_pending_requests_blocking(self, timeout: float = 30.0) -> HITLRequest:
        """Wait for a pending request."""
        return await asyncio.wait_for(self._request_queue.get(), timeout=timeout)

    def get_all_pending(self) -> List[HITLRequest]:
        """Get all currently pending requests."""
        return list(self._pending.values())

    async def notify_timeout(self, request: HITLRequest) -> None:
        """Notify handlers of timeout."""
        for handler in self._timeout_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(request)
                else:
                    handler(request)
            except Exception as e:
                logger.warning(f"Timeout handler error: {e}")

    async def notify_result(
        self,
        request: HITLRequest,
        decision: HITLDecision,
    ) -> None:
        """Notify handlers of result."""
        for handler in self._result_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(request, decision)
                else:
                    handler(request, decision)
            except Exception as e:
                logger.warning(f"Result handler error: {e}")

    def on_timeout(self, handler: Callable) -> None:
        """Register timeout handler."""
        self._timeout_handlers.append(handler)

    def on_result(self, handler: Callable) -> None:
        """Register result handler."""
        self._result_handlers.append(handler)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base class
    "HITLCallback",
    # Implementations
    "ConsoleHITLCallback",
    "AutoApproveCallback",
    "QueueHITLCallback",
]
