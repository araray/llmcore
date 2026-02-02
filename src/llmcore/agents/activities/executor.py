# src/llmcore/agents/activities/executor.py
"""
Activity Executor.

Executes activity requests with validation, HITL approval, and proper error handling.
The executor manages the full lifecycle of activity execution:

    VALIDATION LAYER
    - Activity exists
    - Parameters match schema
    - Security OK
         │
         ▼
    APPROVAL LAYER
    - Check HITL requirements
    - Check approval scopes
         │
         ▼
    EXECUTION LAYER
    - Select appropriate runner
    - Execute in target environment
         │
         ▼
    RESULT LAYER
    - Capture stdout/stderr
    - Parse structured output

References:
    - Master Plan: Section 11 (Execution Engine)
    - Technical Spec: Section 5.4.4 (Executor)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .registry import ActivityRegistry, ExecutionContext, get_default_registry
from .schema import (
    ActivityDefinition,
    ActivityRequest,
    ActivityResult,
    ActivityStatus,
    RiskLevel,
)

if TYPE_CHECKING:
    from ..memory.memory_store import MemoryManager
    from ..sandbox import SandboxProvider

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================


@dataclass
class ValidationResult:
    """Result of activity request validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    activity: Optional[ActivityDefinition] = None


class ActivityValidator:
    """
    Validates activity requests against their definitions.

    Checks:
    - Activity exists in registry
    - Required parameters are provided
    - Parameter types are correct
    - Parameter values are within constraints
    """

    def __init__(self, registry: Optional[ActivityRegistry] = None):
        """
        Initialize the validator.

        Args:
            registry: Activity registry to validate against
        """
        self.registry = registry or get_default_registry()

    def validate(self, request: ActivityRequest) -> ValidationResult:
        """
        Validate an activity request.

        Args:
            request: ActivityRequest to validate

        Returns:
            ValidationResult with status and any errors
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check activity exists
        registered = self.registry.get(request.activity)
        if not registered:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown activity: '{request.activity}'"],
            )

        definition = registered.definition

        # Check if enabled
        if not registered.enabled:
            return ValidationResult(
                valid=False,
                errors=[f"Activity '{request.activity}' is disabled"],
                activity=definition,
            )

        # Check deprecation
        if definition.deprecated:
            warnings.append(
                f"Activity '{request.activity}' is deprecated: {definition.deprecation_message or 'No message'}"
            )

        # Validate parameters
        for param in definition.parameters:
            if param.required and param.name not in request.parameters:
                errors.append(f"Missing required parameter: '{param.name}'")

        # Check for unknown parameters
        known_params = {p.name for p in definition.parameters}
        for param_name in request.parameters:
            if param_name not in known_params:
                warnings.append(f"Unknown parameter: '{param_name}'")

        # Type checking (basic)
        for param in definition.parameters:
            if param.name in request.parameters:
                value = request.parameters[param.name]
                type_error = self._check_type(param.name, value, param.type.value)
                if type_error:
                    errors.append(type_error)

                # Check enum constraints
                if param.enum and value not in param.enum:
                    errors.append(f"Invalid value for '{param.name}': must be one of {param.enum}")

                # Check numeric constraints
                if param.min_value is not None and isinstance(value, (int, float)):
                    if value < param.min_value:
                        errors.append(f"Value for '{param.name}' must be >= {param.min_value}")
                if param.max_value is not None and isinstance(value, (int, float)):
                    if value > param.max_value:
                        errors.append(f"Value for '{param.name}' must be <= {param.max_value}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            activity=definition,
        )

    def _check_type(self, name: str, value: Any, expected_type: str) -> Optional[str]:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": (list, tuple),
            "object": dict,
        }

        expected = type_map.get(expected_type)
        if expected and not isinstance(value, expected):
            return f"Parameter '{name}' must be {expected_type}, got {type(value).__name__}"
        return None


# =============================================================================
# HITL INTEGRATION
# =============================================================================


@dataclass
class HITLDecision:
    """Decision from HITL approval process."""

    approved: bool
    reason: Optional[str] = None
    modified_params: Optional[Dict[str, Any]] = None
    scope_id: Optional[str] = None  # Approval scope if granted


class HITLApprover:
    """
    Human-In-The-Loop approval for risky activities.

    Manages approval based on:
    - Activity risk level
    - Current approval scopes
    - User configuration
    """

    def __init__(
        self,
        risk_threshold: RiskLevel = RiskLevel.MEDIUM,
        auto_approve_callback: Optional[Callable[[ActivityRequest], bool]] = None,
    ):
        """
        Initialize HITL approver.

        Args:
            risk_threshold: Minimum risk level requiring approval
            auto_approve_callback: Optional callback for automatic approval decisions
        """
        self.risk_threshold = risk_threshold
        self.auto_approve_callback = auto_approve_callback
        self._approval_scopes: Dict[str, RiskLevel] = {}

    def requires_approval(self, request: ActivityRequest, definition: ActivityDefinition) -> bool:
        """
        Check if activity requires HITL approval.

        Args:
            request: Activity request
            definition: Activity definition

        Returns:
            True if approval is required
        """
        risk_order = [
            RiskLevel.NONE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        threshold_index = risk_order.index(self.risk_threshold)
        activity_index = risk_order.index(definition.risk_level)

        return activity_index >= threshold_index

    def approve(
        self,
        request: ActivityRequest,
        definition: ActivityDefinition,
        approval_callback: Optional[Callable[[str], bool]] = None,
    ) -> HITLDecision:
        """
        Get approval for an activity.

        Args:
            request: Activity request
            definition: Activity definition
            approval_callback: Callback to prompt user for approval

        Returns:
            HITLDecision with approval status
        """
        if not self.requires_approval(request, definition):
            return HITLDecision(approved=True, reason="No approval required")

        # Check auto-approve callback
        if self.auto_approve_callback and self.auto_approve_callback(request):
            return HITLDecision(approved=True, reason="Auto-approved")

        # Check approval scopes
        scope_key = request.activity
        if scope_key in self._approval_scopes:
            return HITLDecision(
                approved=True,
                reason="Within approval scope",
                scope_id=scope_key,
            )

        # Use approval callback if provided
        if approval_callback:
            prompt = (
                f"Activity '{request.activity}' requires approval.\n"
                f"Risk level: {definition.risk_level.value}\n"
                f"Parameters: {json.dumps(request.parameters, indent=2)}\n"
                f"Reason: {request.reason or 'Not specified'}\n"
                "Approve? (y/n): "
            )
            try:
                approved = approval_callback(prompt)
                return HITLDecision(
                    approved=approved,
                    reason="User decision" if approved else "User rejected",
                )
            except Exception as e:
                return HITLDecision(approved=False, reason=f"Approval error: {e}")

        # Default: require explicit approval
        return HITLDecision(
            approved=False,
            reason="HITL approval required but no callback provided",
        )

    def grant_scope(self, activity: str, max_risk: RiskLevel = RiskLevel.MEDIUM) -> None:
        """Grant approval scope for an activity."""
        self._approval_scopes[activity] = max_risk

    def revoke_scope(self, activity: str) -> None:
        """Revoke approval scope for an activity."""
        self._approval_scopes.pop(activity, None)

    def clear_scopes(self) -> None:
        """Clear all approval scopes."""
        self._approval_scopes.clear()


class HITLManagerAdapter:
    """
    Adapter to use Phase 5 HITLManager with ActivityExecutor.

    Wraps the full-featured HITLManager to provide the same interface
    as the basic HITLApprover, enabling seamless integration with existing
    executor code while leveraging advanced HITL features.

    Features when using HITLManager:
    - Sophisticated risk assessment with dangerous pattern detection
    - Session and persistent approval scopes
    - State persistence across restarts
    - Audit logging
    - Timeout handling with configurable policies
    - Batch approval support

    Example:
        >>> from llmcore.agents.hitl import HITLManager, HITLConfig
        >>> manager = HITLManager(config=HITLConfig(enabled=True))
        >>> adapter = HITLManagerAdapter(manager)
        >>> executor = ActivityExecutor(hitl_approver=adapter)
    """

    def __init__(
        self,
        hitl_manager: Any,  # "HITLManager" - avoid circular import
        risk_threshold: RiskLevel = RiskLevel.MEDIUM,
    ):
        """
        Initialize HITL adapter.

        Args:
            hitl_manager: HITLManager instance from llmcore.agents.hitl
            risk_threshold: Minimum risk level requiring approval
        """
        self._manager = hitl_manager
        self.risk_threshold = risk_threshold
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create one for sync context
            if self._event_loop is None or self._event_loop.is_closed():
                self._event_loop = asyncio.new_event_loop()
            return self._event_loop

    def _run_async(self, coro):
        """Run async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - schedule as task
            return asyncio.ensure_future(coro)
        except RuntimeError:
            # Not in async context - use event loop
            loop = self._get_event_loop()
            return loop.run_until_complete(coro)

    def requires_approval(self, request: ActivityRequest, definition: ActivityDefinition) -> bool:
        """
        Check if activity requires HITL approval.

        Uses HITLManager's risk assessor for sophisticated risk evaluation
        including dangerous pattern detection.

        Args:
            request: Activity request
            definition: Activity definition

        Returns:
            True if approval is required
        """
        # Use manager's risk assessor
        risk = self._manager.risk_assessor.assess(
            request.activity,
            request.parameters,
            {},
        )
        return risk.requires_approval

    def approve(
        self,
        request: ActivityRequest,
        definition: ActivityDefinition,
        approval_callback: Optional[Callable[[str], bool]] = None,
    ) -> HITLDecision:
        """
        Get approval for an activity using HITLManager.

        Args:
            request: Activity request
            definition: Activity definition
            approval_callback: Callback for user approval (passed to manager)

        Returns:
            HITLDecision with approval status
        """
        # Run async check_approval
        try:
            decision = self._run_async(
                self._manager.check_approval(
                    activity_type=request.activity,
                    parameters=request.parameters,
                    context={"definition": definition.name if definition else None},
                    reason=request.reason,
                )
            )

            # Handle future if returned from async context
            if asyncio.isfuture(decision):
                # Need to await - we're in sync context wanting async result
                loop = self._get_event_loop()
                decision = loop.run_until_complete(decision)

            # Map HITLManager decision to executor's HITLDecision
            return HITLDecision(
                approved=decision.is_approved,
                reason=decision.reason,
                modified_params=decision.modified_params if hasattr(decision, 'modified_params') else None,
                scope_id=decision.scope_id,
            )

        except Exception as e:
            logger.error(f"HITL approval error: {e}")
            return HITLDecision(
                approved=False,
                reason=f"HITL error: {e}",
            )

    def grant_scope(self, activity: str, max_risk: RiskLevel = RiskLevel.MEDIUM) -> None:
        """Grant approval scope for an activity."""
        self._manager.scope_manager.grant_session_approval(
            tool_name=activity,
            max_risk_level=max_risk.value,
        )

    def revoke_scope(self, activity: str) -> None:
        """Revoke approval scope for an activity."""
        self._manager.scope_manager.revoke_session_approval(activity)

    def clear_scopes(self) -> None:
        """Clear all approval scopes."""
        # Clear session scopes
        for scope_id in list(self._manager.scope_manager._session_scopes.keys()):
            self._manager.scope_manager.revoke_session_approval(scope_id)


def create_hitl_approver(
    use_advanced: bool = False,
    risk_threshold: RiskLevel = RiskLevel.MEDIUM,
    hitl_config: Optional[Any] = None,  # HITLConfig
    **kwargs,
) -> "HITLApprover | HITLManagerAdapter":
    """
    Factory function to create appropriate HITL approver.

    Args:
        use_advanced: If True, use full HITLManager via adapter
        risk_threshold: Minimum risk level requiring approval
        hitl_config: Configuration for HITLManager (if use_advanced=True)
        **kwargs: Additional arguments passed to constructor

    Returns:
        HITLApprover or HITLManagerAdapter instance
    """
    if use_advanced:
        try:
            from ..hitl import HITLConfig, HITLManager

            config = hitl_config or HITLConfig(
                enabled=True,
                global_risk_threshold=risk_threshold.value,
            )
            manager = HITLManager(config=config)
            return HITLManagerAdapter(manager, risk_threshold=risk_threshold)
        except ImportError as e:
            logger.warning(f"HITLManager not available, falling back to basic approver: {e}")
            return HITLApprover(risk_threshold=risk_threshold, **kwargs)
    else:
        return HITLApprover(risk_threshold=risk_threshold, **kwargs)


# =============================================================================
# ACTIVITY EXECUTOR
# =============================================================================


class ActivityExecutor:
    """
    Execute activity requests.

    Handles the full execution lifecycle:
    1. Validation
    2. HITL approval
    3. Runner selection
    4. Execution
    5. Result capture

    Example:
        >>> executor = ActivityExecutor()
        >>> result = await executor.execute(
        ...     ActivityRequest(
        ...         activity="file_read",
        ...         parameters={"path": "/etc/hosts"},
        ...     )
        ... )
        >>> print(result.output)
    """

    def __init__(
        self,
        registry: Optional[ActivityRegistry] = None,
        validator: Optional[ActivityValidator] = None,
        hitl_approver: Optional[HITLApprover] = None,
        sandbox: Optional["SandboxProvider"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        default_timeout: int = 60,
    ):
        """
        Initialize the executor.

        Args:
            registry: Activity registry
            validator: Request validator
            hitl_approver: HITL approval handler
            sandbox: Sandbox provider for isolated execution
            memory_manager: Memory manager for memory activities (G3 Phase 3)
            default_timeout: Default execution timeout in seconds
        """
        self.registry = registry or get_default_registry()
        self.validator = validator or ActivityValidator(self.registry)
        self.hitl_approver = hitl_approver or HITLApprover()
        self.sandbox = sandbox
        self.memory_manager = memory_manager
        self.default_timeout = default_timeout

        # Execution handlers by activity name
        self._handlers: Dict[str, Callable] = {}
        self._register_builtin_handlers()

    def _register_builtin_handlers(self) -> None:
        """Register handlers for built-in activities."""
        self._handlers.update(
            {
                "file_read": self._handle_file_read,
                "file_write": self._handle_file_write,
                "file_search": self._handle_file_search,
                "file_delete": self._handle_file_delete,
                "python_exec": self._handle_python_exec,
                "bash_exec": self._handle_bash_exec,
                "json_query": self._handle_json_query,
                "memory_store": self._handle_memory_store,
                "memory_search": self._handle_memory_search,
                "final_answer": self._handle_final_answer,
                "ask_human": self._handle_ask_human,
                "think_aloud": self._handle_think_aloud,
            }
        )

    async def execute(
        self,
        request: ActivityRequest,
        context: Optional[ExecutionContext] = None,
        approval_callback: Optional[Callable[[str], bool]] = None,
    ) -> ActivityResult:
        """
        Execute an activity request.

        Args:
            request: Activity request to execute
            context: Execution context
            approval_callback: HITL approval callback

        Returns:
            ActivityResult with execution status and output
        """
        start_time = time.time()
        context = context or ExecutionContext()

        # 1. Validation
        validation = self.validator.validate(request)
        if not validation.valid:
            return ActivityResult(
                activity=request.activity,
                status=ActivityStatus.FAILED,
                error="; ".join(validation.errors),
                duration_ms=int((time.time() - start_time) * 1000),
                request_id=request.request_id,
            )

        definition = validation.activity
        assert definition is not None

        # Log warnings
        for warning in validation.warnings:
            logger.warning(f"Activity warning: {warning}")

        # 2. HITL Approval
        if self.hitl_approver.requires_approval(request, definition):
            decision = self.hitl_approver.approve(request, definition, approval_callback)
            if not decision.approved:
                return ActivityResult(
                    activity=request.activity,
                    status=ActivityStatus.REJECTED,
                    error=f"HITL rejected: {decision.reason}",
                    duration_ms=int((time.time() - start_time) * 1000),
                    request_id=request.request_id,
                )

            # Apply modified params if provided
            if decision.modified_params:
                request.parameters.update(decision.modified_params)

        # 3. Execute
        try:
            timeout = request.timeout_seconds or definition.timeout_seconds or self.default_timeout

            # Check if we need sandbox
            if definition.requires_sandbox and not self.sandbox:
                logger.warning(
                    f"Activity '{request.activity}' requires sandbox but none available, "
                    "executing locally"
                )

            # Get handler
            handler = self._handlers.get(request.activity)
            if handler:
                output = await asyncio.wait_for(
                    self._run_handler(handler, request, context),
                    timeout=timeout,
                )
            else:
                # No handler - try generic execution
                output = f"Activity '{request.activity}' executed (no handler)"

            return ActivityResult(
                activity=request.activity,
                status=ActivityStatus.SUCCESS,
                output=output,
                duration_ms=int((time.time() - start_time) * 1000),
                request_id=request.request_id,
                target=request.target,
            )

        except asyncio.TimeoutError:
            return ActivityResult(
                activity=request.activity,
                status=ActivityStatus.TIMEOUT,
                error=f"Execution timed out after {timeout}s",
                duration_ms=int((time.time() - start_time) * 1000),
                request_id=request.request_id,
            )
        except Exception as e:
            logger.error(f"Activity execution failed: {e}", exc_info=True)
            return ActivityResult(
                activity=request.activity,
                status=ActivityStatus.FAILED,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
                request_id=request.request_id,
            )

    async def _run_handler(
        self,
        handler: Callable,
        request: ActivityRequest,
        context: ExecutionContext,
    ) -> str:
        """Run activity handler (sync or async)."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(request, context)
        else:
            # Run sync handler in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, handler, request, context)

    # =========================================================================
    # BUILT-IN HANDLERS
    # =========================================================================

    def _handle_file_read(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle file_read activity."""
        path = request.parameters["path"]
        encoding = request.parameters.get("encoding", "utf-8")
        max_bytes = request.parameters.get("max_bytes")

        try:
            with open(path, "r", encoding=encoding) as f:
                if max_bytes:
                    content = f.read(max_bytes)
                else:
                    content = f.read()
            return content
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    def _handle_file_write(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle file_write activity."""
        path = request.parameters["path"]
        content = request.parameters["content"]
        encoding = request.parameters.get("encoding", "utf-8")
        append = request.parameters.get("append", False)

        mode = "a" if append else "w"

        try:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")

    def _handle_file_search(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle file_search activity."""
        path = request.parameters["path"]
        pattern = request.parameters.get("pattern", "*")
        recursive = request.parameters.get("recursive", True)
        max_results = request.parameters.get("max_results", 100)

        try:
            base_path = Path(path)
            if not base_path.exists():
                return f"Path does not exist: {path}"

            if recursive:
                matches = list(base_path.rglob(pattern))[:max_results]
            else:
                matches = list(base_path.glob(pattern))[:max_results]

            if not matches:
                return f"No files found matching '{pattern}' in {path}"

            return "\n".join(str(m) for m in matches)
        except Exception as e:
            raise RuntimeError(f"File search failed: {e}")

    def _handle_file_delete(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle file_delete activity."""
        path = request.parameters["path"]
        recursive = request.parameters.get("recursive", False)

        try:
            p = Path(path)
            if not p.exists():
                return f"Path does not exist: {path}"

            if p.is_dir():
                if recursive:
                    import shutil

                    shutil.rmtree(path)
                else:
                    p.rmdir()
            else:
                p.unlink()

            return f"Successfully deleted: {path}"
        except Exception as e:
            raise RuntimeError(f"Failed to delete: {e}")

    def _handle_python_exec(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle python_exec activity."""
        code = request.parameters["code"]
        timeout = request.parameters.get("timeout", 30)

        # Create temp file for code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=context.working_dir,
                env={**os.environ, **context.environment},
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Python execution timed out after {timeout}s")
        finally:
            os.unlink(temp_path)

    def _handle_bash_exec(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle bash_exec activity."""
        command = request.parameters["command"]
        timeout = request.parameters.get("timeout", 30)
        working_dir = request.parameters.get("working_dir", context.working_dir)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env={**os.environ, **context.environment},
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Bash execution timed out after {timeout}s")

    def _handle_json_query(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle json_query activity."""
        data_str = request.parameters["data"]
        query = request.parameters["query"]

        try:
            data = json.loads(data_str)

            # Simple path query (e.g., "results.0.name")
            parts = query.split(".")
            result = data
            for part in parts:
                if isinstance(result, list):
                    result = result[int(part)]
                elif isinstance(result, dict):
                    result = result[part]

            return json.dumps(result, indent=2)
        except Exception as e:
            raise RuntimeError(f"JSON query failed: {e}")

    async def _handle_memory_store(
        self, request: ActivityRequest, context: ExecutionContext
    ) -> str:
        """
        Handle memory_store activity.

        Stores a key-value pair in working memory (in-context) or long-term memory
        (persistent) depending on the memory_type parameter.

        Parameters from request:
            key: Unique identifier for the memory
            value: Content to store
            memory_type: "working" (default) or "longterm"
            metadata: Optional metadata dict

        Returns:
            Status message indicating where data was stored.
        """
        key = request.parameters["key"]
        value = request.parameters["value"]
        memory_type = request.parameters.get("memory_type", "working")
        metadata = request.parameters.get("metadata", {})

        # Add standard metadata
        from datetime import datetime, timezone

        metadata.update({
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "activity_id": request.request_id,
            "source": "activity_memory_store",
        })

        if memory_type == "working":
            # Store in execution context's working memory (always available)
            context.working_memory[key] = {
                "value": value,
                "metadata": metadata,
            }
            return f"Stored in working memory: key='{key}', size={len(str(value))} chars"

        elif memory_type == "longterm":
            # Store in long-term memory via memory manager
            if self.memory_manager is None:
                # Fallback: store in working memory if no manager available
                logger.warning(
                    "Long-term memory not available (no memory_manager). "
                    "Storing in working memory instead."
                )
                context.working_memory[key] = {
                    "value": value,
                    "metadata": {**metadata, "fallback": True},
                }
                return (
                    f"[FALLBACK] Long-term memory not available. "
                    f"Stored in working memory: key='{key}'"
                )

            try:
                # Import memory types for proper storage
                from ..memory.memory_store import MemoryImportance, MemoryType

                # Store as semantic memory
                item_id = await self.memory_manager.remember(
                    content=f"{key}: {value}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=MemoryImportance.MEDIUM,
                    metadata={**metadata, "memory_key": key},
                )
                return f"Stored in long-term memory: key='{key}', id='{item_id}'"
            except Exception as e:
                logger.error(f"Failed to store in long-term memory: {e}", exc_info=True)
                # Fallback to working memory
                context.working_memory[key] = {
                    "value": value,
                    "metadata": {**metadata, "error": str(e)},
                }
                return f"[ERROR] Failed to store in long-term memory: {e}. Stored in working memory instead."

        else:
            return f"[ERROR] Unknown memory_type: '{memory_type}'. Valid values: 'working', 'longterm'"

    async def _handle_memory_search(
        self, request: ActivityRequest, context: ExecutionContext
    ) -> str:
        """
        Handle memory_search activity.

        Searches working memory and/or long-term memory for relevant items.

        Parameters from request:
            query: Search query
            max_results: Maximum results to return (default: 5)
            search_working: Search working memory (default: True)
            search_longterm: Search long-term memory (default: True)

        Returns:
            Formatted search results.
        """
        query = request.parameters["query"]
        max_results = request.parameters.get("max_results", 5)
        search_working = request.parameters.get("search_working", True)
        search_longterm = request.parameters.get("search_longterm", True)

        results = []
        query_lower = query.lower()

        # Search working memory (in-context)
        if search_working and context.working_memory:
            for key, data in context.working_memory.items():
                value = data.get("value", "")
                # Simple relevance: check if query terms appear in key or value
                content_str = f"{key} {value}".lower()
                if query_lower in content_str or any(
                    term in content_str for term in query_lower.split()
                ):
                    results.append({
                        "source": "working",
                        "key": key,
                        "value": value,
                        "metadata": data.get("metadata", {}),
                    })

        # Search long-term memory
        if search_longterm and self.memory_manager is not None:
            try:
                recall_result = await self.memory_manager.recall(
                    query=query,
                    max_results=max_results,
                    include_working=False,  # We handle working memory separately
                )
                for item in recall_result.memories:
                    results.append({
                        "source": "longterm",
                        "content": item.content,
                        "relevance": item.relevance,
                        "memory_type": item.memory_type.value,
                        "metadata": item.metadata,
                    })
            except Exception as e:
                logger.error(f"Long-term memory search failed: {e}", exc_info=True)
                results.append({
                    "source": "error",
                    "message": f"Long-term memory search failed: {e}",
                })

        # Format results
        if not results:
            return f"No memories found for query: '{query}'"

        # Limit results
        results = results[:max_results]

        # Build output
        output_lines = [f"Found {len(results)} memory item(s) for '{query}':"]
        for i, item in enumerate(results, 1):
            if item.get("source") == "working":
                output_lines.append(
                    f"  {i}. [Working] {item['key']}: {str(item['value'])[:100]}..."
                    if len(str(item['value'])) > 100
                    else f"  {i}. [Working] {item['key']}: {item['value']}"
                )
            elif item.get("source") == "longterm":
                score = item.get("relevance", "N/A")
                output_lines.append(
                    f"  {i}. [Long-term] {item['content'][:100]}... (relevance: {score})"
                    if len(item.get("content", "")) > 100
                    else f"  {i}. [Long-term] {item.get('content', '')} (relevance: {score})"
                )
            elif item.get("source") == "error":
                output_lines.append(f"  {i}. [Error] {item['message']}")

        return "\n".join(output_lines)

    def _handle_final_answer(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle final_answer activity."""
        answer = request.parameters["answer"]
        return f"[FINAL ANSWER]\n{answer}"

    def _handle_ask_human(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle ask_human activity."""
        question = request.parameters["question"]
        context_str = request.parameters.get("context", "")
        return f"[HUMAN INPUT REQUIRED]\nQuestion: {question}\nContext: {context_str}"

    def _handle_think_aloud(self, request: ActivityRequest, context: ExecutionContext) -> str:
        """Handle think_aloud activity."""
        thought = request.parameters["thought"]
        return f"[THOUGHT]\n{thought}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ActivityExecutor",
    "ActivityValidator",
    "HITLApprover",
    "HITLDecision",
    "ValidationResult",
]
