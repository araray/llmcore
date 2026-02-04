# src/llmcore/agents/hitl/manager.py
"""
Human-In-The-Loop (HITL) Manager.

Central coordinator for the HITL approval workflow:
1. Risk assessment
2. Scope checking
3. Approval request/response
4. Timeout handling
5. Audit logging
6. State persistence

Usage:
    >>> from llmcore.agents.hitl import HITLManager, HITLConfig
    >>> manager = HITLManager(config=HITLConfig(enabled=True))
    >>> decision = await manager.check_approval(activity_request, context)
    >>> if decision.is_approved:
    ...     # Execute activity
    ...     pass

References:
    - Master Plan: Section 20 (Human-In-The-Loop System)
    - Technical Spec: Section 5.5 (HITL Architecture)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections.abc import Callable
from uuid import uuid4

from .callbacks import ConsoleHITLCallback, HITLCallback
from .models import (
    ActivityInfo,
    ApprovalScope,
    ApprovalStatus,
    HITLAuditEvent,
    HITLConfig,
    HITLDecision,
    HITLEventType,
    HITLRequest,
    HITLResponse,
    HITLStorageConfig,
    TimeoutPolicy,
)
from .risk_assessor import RiskAssessor, RiskLevel
from .scope import ApprovalScopeManager
from .state import FileHITLStore, HITLStateStore, InMemoryHITLStore

logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT LOGGER
# =============================================================================


class HITLAuditLogger:
    """
    Audit logger for HITL decisions.

    Logs all approval requests, responses, and decisions for compliance
    and debugging.
    """

    def __init__(
        self,
        log_path: Path | None = None,
        enabled: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            log_path: Path for audit log file (JSON Lines)
            enabled: Whether audit logging is enabled
        """
        self.log_path = log_path
        self.enabled = enabled
        self._events: list[HITLAuditEvent] = []

    def log_event(self, event: HITLAuditEvent) -> None:
        """Log an audit event."""
        if not self.enabled:
            return

        self._events.append(event)

        # Log to file
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write audit log: {e}")

        # Also log to standard logger
        logger.info(
            f"HITL Audit: {event.event_type.value} - "
            f"activity={event.activity_type}, decision={event.decision}"
        )

    def log_request_created(
        self,
        request: HITLRequest,
    ) -> None:
        """Log request creation."""
        self.log_event(
            HITLAuditEvent(
                event_type=HITLEventType.REQUEST_CREATED,
                request_id=request.request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                activity_type=request.activity.activity_type,
                risk_level=request.risk_assessment.overall_level,
                details={
                    "parameters": request.activity.parameters,
                    "risk_factors": [f.name for f in request.risk_assessment.factors],
                },
            )
        )

    def log_decision(
        self,
        request: HITLRequest,
        decision: HITLDecision,
    ) -> None:
        """Log final decision."""
        event_type_map = {
            ApprovalStatus.APPROVED: HITLEventType.REQUEST_APPROVED,
            ApprovalStatus.REJECTED: HITLEventType.REQUEST_REJECTED,
            ApprovalStatus.MODIFIED: HITLEventType.REQUEST_MODIFIED,
            ApprovalStatus.TIMEOUT: HITLEventType.REQUEST_TIMEOUT,
            ApprovalStatus.AUTO_APPROVED: HITLEventType.REQUEST_APPROVED,
            ApprovalStatus.ESCALATED: HITLEventType.REQUEST_ESCALATED,
        }

        self.log_event(
            HITLAuditEvent(
                event_type=event_type_map.get(decision.status, HITLEventType.REQUEST_REJECTED),
                request_id=request.request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                activity_type=request.activity.activity_type,
                risk_level=request.risk_assessment.overall_level,
                decision=decision.status.value,
                details={
                    "reason": decision.reason,
                    "modified_params": decision.modified_parameters,
                    "scope_id": decision.scope_id,
                },
            )
        )

    def log_scope_granted(
        self,
        scope_type: ApprovalScope,
        target: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log scope grant."""
        self.log_event(
            HITLAuditEvent(
                event_type=HITLEventType.SCOPE_GRANTED,
                user_id=user_id,
                session_id=session_id,
                details={
                    "scope_type": scope_type.value,
                    "target": target,
                },
            )
        )

    def get_events(
        self,
        request_id: str | None = None,
        event_type: HITLEventType | None = None,
        limit: int = 100,
    ) -> list[HITLAuditEvent]:
        """Get audit events with optional filtering."""
        events = self._events

        if request_id:
            events = [e for e in events if e.request_id == request_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]


# =============================================================================
# HITL MANAGER
# =============================================================================


class HITLManager:
    """
    Central manager for Human-In-The-Loop approval workflow.

    Coordinates:
    - Risk assessment
    - Scope checking
    - Approval requests
    - Timeout handling
    - State persistence
    - Audit logging

    Example:
        >>> config = HITLConfig(enabled=True, global_risk_threshold="medium")
        >>> manager = HITLManager(config=config)
        >>>
        >>> # Check if activity needs approval
        >>> decision = await manager.check_approval(activity_request)
        >>>
        >>> if decision.is_approved:
        ...     result = await execute_activity(activity_request)
        ... else:
        ...     print(f"Rejected: {decision.reason}")

    Storage Backend Usage:
        >>> # SQLite backend (development/single-user)
        >>> config = HITLConfig(
        ...     storage=HITLStorageConfig(backend="sqlite", sqlite_path="/tmp/hitl.db")
        ... )
        >>> manager = HITLManager(config=config)
        >>> await manager.initialize()
        >>>
        >>> # PostgreSQL backend (production/multi-user)
        >>> config = HITLConfig(
        ...     storage=HITLStorageConfig(
        ...         backend="postgres",
        ...         postgres_url="postgresql://user:pass@localhost/db"
        ...     )
        ... )
        >>> manager = HITLManager(config=config)
        >>> await manager.initialize()
    """

    def __init__(
        self,
        config: HITLConfig | None = None,
        risk_assessor: RiskAssessor | None = None,
        scope_manager: ApprovalScopeManager | None = None,
        callback: HITLCallback | None = None,
        state_store: HITLStateStore | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """
        Initialize HITL manager.

        Args:
            config: HITL configuration (includes storage config)
            risk_assessor: Custom risk assessor
            scope_manager: Custom scope manager
            callback: UI callback for approvals
            state_store: State persistence store (overrides config if provided)
            session_id: Current session ID
            user_id: Current user ID
        """
        self.config = config or HITLConfig()
        self.session_id = session_id or str(uuid4())
        self.user_id = user_id or "default"

        # Initialize components
        self.risk_assessor = risk_assessor or RiskAssessor(config=self.config)
        self.scope_manager = scope_manager or ApprovalScopeManager(
            session_id=self.session_id,
            user_id=self.user_id,
            config=self.config,
        )
        self.callback = callback or ConsoleHITLCallback()

        # State store: use provided or create from config
        self._state_store_provided = state_store is not None
        if state_store is not None:
            self.state_store = state_store
        else:
            self.state_store = self._create_store_from_config()

        # Audit logger
        audit_path = Path(self.config.audit_log_path) if self.config.audit_log_path else None
        self.audit = HITLAuditLogger(
            log_path=audit_path,
            enabled=self.config.audit_logging_enabled,
        )

        # Request batching
        self._pending_batch: list[HITLRequest] = []
        self._batch_task: asyncio.Task | None = None

        # Event callbacks
        self._on_approval: list[Callable] = []
        self._on_rejection: list[Callable] = []

        # Statistics
        self._stats = {
            "requests_created": 0,
            "requests_approved": 0,
            "requests_rejected": 0,
            "requests_auto_approved": 0,
            "requests_timed_out": 0,
            "scope_grants": 0,
        }

        # Initialization flag
        self._initialized = False

    def _create_store_from_config(self) -> HITLStateStore:
        """
        Create storage backend from configuration.

        Returns:
            Appropriate HITLStateStore implementation based on config.

        Raises:
            ValueError: If unknown backend type specified
        """
        storage_config = self.config.storage

        if storage_config.backend == "memory":
            return InMemoryHITLStore()

        elif storage_config.backend == "file":
            path = storage_config.file_path or "~/.local/share/llmcore/hitl_state"
            return FileHITLStore(path)

        elif storage_config.backend == "sqlite":
            from .sqlite_state import SqliteHITLStore

            return SqliteHITLStore()

        elif storage_config.backend == "postgres":
            from .postgres_state import PostgresHITLStore

            return PostgresHITLStore()

        else:
            raise ValueError(f"Unknown storage backend: {storage_config.backend}")

    async def initialize(self) -> None:
        """
        Initialize the manager and storage backend.

        Must be called before using database backends (sqlite/postgres).
        Memory and file backends work without initialization for backwards compatibility.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return

        storage_config = self.config.storage

        # Only initialize database backends that have an initialize method
        if hasattr(self.state_store, "initialize"):
            try:
                if storage_config.backend == "sqlite":
                    await self.state_store.initialize({"path": storage_config.sqlite_path})
                elif storage_config.backend == "postgres":
                    storage_config.validate_backend()
                    await self.state_store.initialize(
                        {
                            "db_url": storage_config.postgres_url,
                            "min_pool_size": storage_config.postgres_min_pool_size,
                            "max_pool_size": storage_config.postgres_max_pool_size,
                            "table_prefix": storage_config.postgres_table_prefix,
                        }
                    )
                # Memory and File stores don't need initialization
                logger.info(f"HITL Manager initialized with {storage_config.backend} backend")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize HITL storage: {e}") from e

        self._initialized = True

    async def close(self) -> None:
        """
        Close the manager and storage backend.

        Should be called when done using the manager to clean up resources.
        """
        if hasattr(self.state_store, "close"):
            await self.state_store.close()
        self._initialized = False
        logger.debug("HITL Manager closed")

    async def check_approval(
        self,
        activity_type: str,
        parameters: dict[str, Any],
        context: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> HITLDecision:
        """
        Check if activity requires approval and handle workflow.

        Args:
            activity_type: Type of activity
            parameters: Activity parameters
            context: Additional context
            reason: Reasoning for activity

        Returns:
            HITLDecision with approval status
        """
        context = context or {}

        # Check if HITL is disabled
        if not self.config.enabled:
            return HITLDecision(
                status=ApprovalStatus.AUTO_APPROVED,
                reason="HITL disabled",
            )

        # 1. Assess risk
        risk = self.risk_assessor.assess(activity_type, parameters, context)

        # 2. Check if approval is required
        if not risk.requires_approval:
            return HITLDecision(
                status=ApprovalStatus.AUTO_APPROVED,
                reason="No approval required (low risk)",
            )

        # 3. Check existing approval scopes
        risk_level = RiskLevel(risk.overall_level)
        scope_approved = self.scope_manager.check_scope(activity_type, parameters, risk_level)

        if scope_approved is True:
            self._stats["requests_auto_approved"] += 1
            return HITLDecision(
                status=ApprovalStatus.AUTO_APPROVED,
                reason="Pre-approved by scope",
                scope_id=f"scope:{activity_type}",
            )
        elif scope_approved is False:
            # Explicitly denied by scope
            return HITLDecision(
                status=ApprovalStatus.REJECTED,
                reason="Explicitly denied by scope",
            )

        # 4. Create approval request
        activity_info = ActivityInfo(
            activity_type=activity_type,
            parameters=parameters,
            reason=reason,
        )

        request = HITLRequest(
            activity=activity_info,
            risk_assessment=risk,
            context_summary=self._build_context_summary(context),
            session_id=self.session_id,
            user_id=self.user_id,
        )

        # Set expiration
        request.set_expiration(self.config.default_timeout_seconds)

        # 5. Save request
        await self.state_store.save_request(request)
        self.audit.log_request_created(request)
        self._stats["requests_created"] += 1

        # 6. Request approval (with batching if enabled)
        if self.config.batch_similar_requests:
            decision = await self._batch_approval_request(request)
        else:
            decision = await self._request_approval(request)

        # 7. Update state and audit
        await self.state_store.update_request_status(
            request.request_id,
            decision.status,
            decision.response,
        )
        self.audit.log_decision(request, decision)

        # 8. Update statistics
        if decision.is_approved:
            self._stats["requests_approved"] += 1
            for handler in self._on_approval:
                try:
                    handler(request, decision)
                except Exception as e:
                    logger.warning(f"Approval handler error: {e}")
        else:
            self._stats["requests_rejected"] += 1
            for handler in self._on_rejection:
                try:
                    handler(request, decision)
                except Exception as e:
                    logger.warning(f"Rejection handler error: {e}")

        # 9. Handle scope grants from response
        if decision.response and decision.response.scope_grant:
            await self._process_scope_grant(
                request,
                decision.response.scope_grant,
                risk_level,
            )

        return decision

    async def _request_approval(self, request: HITLRequest) -> HITLDecision:
        """Request approval from user via callback."""
        try:
            # Request with timeout
            timeout = request.time_remaining_seconds
            response = await asyncio.wait_for(
                self.callback.request_approval(request),
                timeout=max(1, timeout),
            )

            # Process response
            if response.approved:
                if response.modified_parameters:
                    return HITLDecision(
                        status=ApprovalStatus.MODIFIED,
                        reason=response.feedback or "Approved with modifications",
                        modified_parameters=response.modified_parameters,
                        request=request,
                        response=response,
                    )
                else:
                    return HITLDecision(
                        status=ApprovalStatus.APPROVED,
                        reason=response.feedback or "Approved by user",
                        request=request,
                        response=response,
                    )
            else:
                return HITLDecision(
                    status=ApprovalStatus.REJECTED,
                    reason=response.feedback or "Rejected by user",
                    request=request,
                    response=response,
                )

        except TimeoutError:
            self._stats["requests_timed_out"] += 1
            await self.callback.notify_timeout(request)
            return self._handle_timeout(request)

        except Exception as e:
            logger.error(f"Approval request failed: {e}", exc_info=True)
            return HITLDecision(
                status=ApprovalStatus.REJECTED,
                reason=f"Approval error: {e}",
                request=request,
            )

    async def _batch_approval_request(self, request: HITLRequest) -> HITLDecision:
        """Batch similar approval requests."""
        # For now, just use single request
        # TODO: Implement proper batching logic
        return await self._request_approval(request)

    def _handle_timeout(self, request: HITLRequest) -> HITLDecision:
        """Handle approval timeout based on policy."""
        risk_level = request.risk_assessment.overall_level
        policy = self.config.timeout_policies_by_risk.get(risk_level, self.config.timeout_policy)

        if policy == TimeoutPolicy.APPROVE:
            return HITLDecision(
                status=ApprovalStatus.APPROVED,
                reason="Approved on timeout (low risk)",
                request=request,
            )
        elif policy == TimeoutPolicy.ESCALATE:
            return HITLDecision(
                status=ApprovalStatus.ESCALATED,
                reason="Escalated due to timeout",
                request=request,
            )
        elif policy == TimeoutPolicy.RETRY:
            # For retry, we'd need to re-request - for now, reject
            return HITLDecision(
                status=ApprovalStatus.TIMEOUT,
                reason="Timed out waiting for approval",
                request=request,
            )
        else:  # REJECT
            return HITLDecision(
                status=ApprovalStatus.REJECTED,
                reason="Rejected on timeout",
                request=request,
            )

    async def _process_scope_grant(
        self,
        request: HITLRequest,
        scope: ApprovalScope,
        risk_level: RiskLevel,
    ) -> None:
        """Process scope grant from approval response."""
        activity_type = request.activity.activity_type

        if scope == ApprovalScope.TOOL:
            self.scope_manager.grant_session_approval(
                activity_type,
                max_risk_level=risk_level,
                granted_by=self.user_id,
            )
            self.audit.log_scope_granted(scope, activity_type, self.user_id, self.session_id)

        elif scope == ApprovalScope.SESSION:
            self.scope_manager.grant_full_session_approval()
            self.audit.log_scope_granted(scope, "all", self.user_id, self.session_id)

        self._stats["scope_grants"] += 1

    def _build_context_summary(self, context: dict[str, Any]) -> str:
        """Build context summary for approval request."""
        parts = []

        if context.get("goal"):
            parts.append(f"Goal: {context['goal']}")

        if context.get("iteration"):
            parts.append(f"Iteration: {context['iteration']}")

        if context.get("previous_activities"):
            prev = context["previous_activities"][-3:]
            parts.append(f"Recent: {', '.join(str(a) for a in prev)}")

        return " | ".join(parts) if parts else "No context"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def submit_response(self, response: HITLResponse) -> HITLDecision | None:
        """
        Submit response for pending request (for async workflows).

        Args:
            response: Human response

        Returns:
            Decision if request exists, None otherwise
        """
        request = await self.state_store.get_request(response.request_id)
        if not request:
            logger.warning(f"No pending request found: {response.request_id}")
            return None

        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Request already processed: {response.request_id}")
            return None

        # Process response
        if response.approved:
            status = (
                ApprovalStatus.MODIFIED if response.modified_parameters else ApprovalStatus.APPROVED
            )
        else:
            status = ApprovalStatus.REJECTED

        decision = HITLDecision(
            status=status,
            reason=response.feedback or "",
            modified_parameters=response.modified_parameters,
            request=request,
            response=response,
        )

        # Update state
        await self.state_store.update_request_status(response.request_id, status, response)
        await self.state_store.save_response(response)

        # Audit
        self.audit.log_decision(request, decision)

        # Notify callback
        await self.callback.notify_result(request, decision)

        return decision

    async def get_pending_requests(self) -> list[HITLRequest]:
        """Get all pending approval requests."""
        return await self.state_store.get_pending_requests(session_id=self.session_id)

    def grant_session_approval(
        self,
        tool_name: str,
        conditions: dict[str, Any] | None = None,
        max_risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> str:
        """Grant session approval for a tool."""
        scope_id = self.scope_manager.grant_session_approval(
            tool_name,
            conditions=conditions,
            max_risk_level=max_risk_level,
            granted_by=self.user_id,
        )
        self.audit.log_scope_granted(ApprovalScope.TOOL, tool_name, self.user_id, self.session_id)
        self._stats["scope_grants"] += 1
        return scope_id

    def grant_pattern_approval(self, pattern: str) -> str:
        """Grant pattern-based approval."""
        scope_id = self.scope_manager.grant_pattern_approval(pattern)
        self.audit.log_scope_granted(ApprovalScope.PATTERN, pattern, self.user_id, self.session_id)
        return scope_id

    def revoke_session_approval(self, tool_name: str) -> bool:
        """Revoke session approval for a tool."""
        return self.scope_manager.revoke_session_approval(tool_name)

    async def cleanup_expired(self) -> int:
        """Clean up expired requests."""
        return await self.state_store.cleanup_expired()

    def on_approval(self, handler: Callable) -> None:
        """Register approval event handler."""
        self._on_approval.append(handler)

    def on_rejection(self, handler: Callable) -> None:
        """Register rejection event handler."""
        self._on_rejection.append(handler)

    def get_statistics(self) -> dict[str, Any]:
        """Get HITL statistics."""
        return {
            **self._stats,
            "scope_stats": self.scope_manager.get_statistics(),
            "session_id": self.session_id,
            "user_id": self.user_id,
        }

    def is_safe_tool(self, tool_name: str) -> bool:
        """Check if tool is safe (no approval needed)."""
        return self.risk_assessor.is_safe_tool(tool_name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_hitl_manager(
    enabled: bool = True,
    risk_threshold: str = "medium",
    callback: HITLCallback | None = None,
    persist_path: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    storage_backend: str = "memory",
    storage_config: dict[str, Any] | None = None,
) -> HITLManager:
    """
    Create HITL manager with common settings (sync convenience function).

    For database backends (sqlite/postgres), use create_hitl_manager_async() instead
    or call manager.initialize() after creating.

    Args:
        enabled: Whether HITL is enabled
        risk_threshold: Risk threshold for approvals
        callback: UI callback (default: ConsoleHITLCallback)
        persist_path: Path for file-based state persistence (legacy, use storage_config)
        session_id: Session ID
        user_id: User ID
        storage_backend: Storage backend type (memory, file, sqlite, postgres)
        storage_config: Additional storage configuration options

    Returns:
        Configured HITLManager (not initialized for database backends)

    Example:
        >>> # Memory backend (default, no initialization needed)
        >>> manager = create_hitl_manager(enabled=True)
        >>>
        >>> # SQLite backend (requires initialization)
        >>> manager = create_hitl_manager(
        ...     storage_backend="sqlite",
        ...     storage_config={"sqlite_path": "/tmp/hitl.db"}
        ... )
        >>> await manager.initialize()
    """
    # Build storage config
    storage_cfg_dict = {"backend": storage_backend}

    # Handle legacy persist_path for file backend
    if persist_path and storage_backend == "file":
        storage_cfg_dict["file_path"] = persist_path
    elif persist_path and storage_backend == "memory":
        # Legacy behavior: use file backend if persist_path provided
        storage_cfg_dict["backend"] = "file"
        storage_cfg_dict["file_path"] = persist_path

    # Apply any additional storage config
    if storage_config:
        storage_cfg_dict.update(storage_config)

    config = HITLConfig(
        enabled=enabled,
        global_risk_threshold=risk_threshold,
        storage=HITLStorageConfig(**storage_cfg_dict),
    )

    return HITLManager(
        config=config,
        callback=callback,
        session_id=session_id,
        user_id=user_id,
    )


async def create_hitl_manager_async(
    config: HITLConfig | None = None,
    state_store: HITLStateStore | None = None,
    callback: HITLCallback | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> HITLManager:
    """
    Async factory function to create and initialize an HITLManager.

    This is the preferred way to create managers with database backends.

    Args:
        config: HITL configuration (dict will be converted to HITLConfig)
        state_store: Optional pre-configured storage backend (overrides config)
        callback: UI callback for approvals
        session_id: Session ID
        user_id: User ID

    Returns:
        Initialized HITLManager ready for use

    Example:
        >>> # SQLite backend
        >>> manager = await create_hitl_manager_async(
        ...     HITLConfig(storage=HITLStorageConfig(
        ...         backend="sqlite",
        ...         sqlite_path="/tmp/hitl.db"
        ...     ))
        ... )
        >>>
        >>> # PostgreSQL backend
        >>> manager = await create_hitl_manager_async(
        ...     HITLConfig(storage=HITLStorageConfig(
        ...         backend="postgres",
        ...         postgres_url="postgresql://user:pass@localhost/db"
        ...     ))
        ... )
        >>>
        >>> # From dict
        >>> manager = await create_hitl_manager_async(
        ...     HITLConfig(**{
        ...         "storage": {"backend": "sqlite", "sqlite_path": "/tmp/hitl.db"}
        ...     })
        ... )

    Raises:
        RuntimeError: If storage initialization fails
    """
    if config is None:
        config = HITLConfig()

    manager = HITLManager(
        config=config,
        state_store=state_store,
        callback=callback,
        session_id=session_id,
        user_id=user_id,
    )
    await manager.initialize()
    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HITLAuditLogger",
    "HITLManager",
    "create_hitl_manager",
    "create_hitl_manager_async",
]
