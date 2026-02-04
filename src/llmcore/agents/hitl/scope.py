# src/llmcore/agents/hitl/scope.py
"""
Approval Scope Management for HITL System.

Manages approval scopes that allow activities to be auto-approved:
- Session scopes: Valid for current session only
- Persistent scopes: Saved across sessions
- Tool scopes: Approve all uses of a specific tool
- Pattern scopes: Approve activities matching patterns

References:
    - Master Plan: Section 22 (Approval Scope Management)
"""

from __future__ import annotations

import fnmatch
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import (
    ApprovalScope,
    HITLConfig,
    PersistentScope,
    SessionScope,
    ToolScope,
)
from .risk_assessor import RiskLevel

logger = logging.getLogger(__name__)


# =============================================================================
# SCOPE CONDITION MATCHER
# =============================================================================


class ScopeConditionMatcher:
    """
    Matches activity parameters against scope conditions.

    Supports:
    - Exact match: {"path": "/workspace/test.txt"}
    - Pattern match: {"path_pattern": "/workspace/*"}
    - Range match: {"timeout_max": 60}
    """

    def matches(
        self,
        parameters: dict[str, Any],
        conditions: dict[str, Any],
    ) -> bool:
        """
        Check if parameters match all conditions.

        Args:
            parameters: Activity parameters
            conditions: Scope conditions

        Returns:
            True if all conditions match
        """
        if not conditions:
            return True

        for key, expected_value in conditions.items():
            # Pattern matching
            if key.endswith("_pattern"):
                actual_key = key[:-8]  # Remove "_pattern" suffix
                actual_value = parameters.get(actual_key, "")
                if not self._match_pattern(str(actual_value), str(expected_value)):
                    return False

            # Range matching (max)
            elif key.endswith("_max"):
                actual_key = key[:-4]  # Remove "_max" suffix
                actual_value = parameters.get(actual_key)
                if actual_value is not None and actual_value > expected_value:
                    return False

            # Range matching (min)
            elif key.endswith("_min"):
                actual_key = key[:-4]  # Remove "_min" suffix
                actual_value = parameters.get(actual_key)
                if actual_value is not None and actual_value < expected_value:
                    return False

            # Exact match
            else:
                if key not in parameters:
                    return False
                if parameters[key] != expected_value:
                    return False

        return True

    def _match_pattern(self, value: str, pattern: str) -> bool:
        """Match value against pattern (glob or regex)."""
        # Try glob first (simpler, more common)
        if fnmatch.fnmatch(value, pattern):
            return True

        # Try regex if glob doesn't match
        try:
            if re.match(pattern, value):
                return True
        except re.error:
            pass

        return False


# =============================================================================
# APPROVAL SCOPE MANAGER
# =============================================================================


class ApprovalScopeManager:
    """
    Manage approval scopes for HITL decisions.

    Scopes allow certain activities to be auto-approved:
    - SINGLE: One-time approval (not stored)
    - TOOL: All uses of a specific tool
    - PATTERN: Actions matching specific patterns
    - CATEGORY: All tools in a category
    - SESSION: All actions for the session

    Usage:
        >>> manager = ApprovalScopeManager()
        >>> manager.grant_session_approval("file_read", conditions={"path_pattern": "/workspace/*"})
        >>> is_approved = manager.check_scope("file_read", {"path": "/workspace/test.txt"})
        >>> print(is_approved)  # True
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        config: HITLConfig | None = None,
    ):
        """
        Initialize scope manager.

        Args:
            session_id: Current session ID
            user_id: Current user ID
            config: HITL configuration
        """
        self.session_id = session_id or "default"
        self.user_id = user_id or "default"
        self.config = config or HITLConfig()

        # Session scope (valid for this session only)
        self._session_scope = SessionScope(session_id=self.session_id)

        # Persistent scope (loaded from storage)
        self._persistent_scope = PersistentScope(user_id=self.user_id)

        # Condition matcher
        self._matcher = ScopeConditionMatcher()

        # Scope event callbacks
        self._on_scope_granted: list[callable] = []
        self._on_scope_revoked: list[callable] = []

    def check_scope(
        self,
        activity_type: str,
        parameters: dict[str, Any],
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> bool | None:
        """
        Check if activity is pre-approved within scope.

        Args:
            activity_type: Activity type
            parameters: Activity parameters
            risk_level: Current risk level

        Returns:
            True if approved by scope
            False if explicitly denied by scope
            None if no scope applies (needs explicit approval)
        """
        # 1. Check full session approval
        if self._session_scope.session_approval:
            logger.debug(f"Activity '{activity_type}' auto-approved by session scope")
            return True

        # 2. Check tool-specific session scopes
        for tool_scope in self._session_scope.approved_tools:
            if tool_scope.tool_name == activity_type:
                # Check risk level
                if risk_level > RiskLevel(tool_scope.max_risk_level):
                    continue

                # Check conditions
                if tool_scope.conditions:
                    if self._matcher.matches(parameters, tool_scope.conditions):
                        logger.debug(
                            f"Activity '{activity_type}' auto-approved by tool scope with conditions"
                        )
                        return tool_scope.approved
                else:
                    logger.debug(f"Activity '{activity_type}' auto-approved by tool scope")
                    return tool_scope.approved

        # 3. Check pattern scopes
        for pattern in self._session_scope.approved_patterns:
            if self._matches_approval_pattern(activity_type, parameters, pattern):
                logger.debug(f"Activity '{activity_type}' auto-approved by pattern: {pattern}")
                return True

        # 4. Check persistent scopes
        for tool_scope in self._persistent_scope.approved_tools:
            if tool_scope.tool_name == activity_type:
                if risk_level > RiskLevel(tool_scope.max_risk_level):
                    continue

                if tool_scope.conditions:
                    if self._matcher.matches(parameters, tool_scope.conditions):
                        logger.debug(
                            f"Activity '{activity_type}' auto-approved by persistent scope"
                        )
                        return tool_scope.approved
                else:
                    logger.debug(f"Activity '{activity_type}' auto-approved by persistent scope")
                    return tool_scope.approved

        # No scope applies
        return None

    def grant_session_approval(
        self,
        tool_name: str,
        conditions: dict[str, Any] | None = None,
        max_risk_level: RiskLevel = RiskLevel.MEDIUM,
        granted_by: str = "",
    ) -> str:
        """
        Grant approval scope for tool in current session.

        Args:
            tool_name: Tool to approve
            conditions: Conditions for approval
            max_risk_level: Maximum risk level allowed
            granted_by: Who granted the scope

        Returns:
            Scope ID
        """
        # Check if scope already exists
        for i, scope in enumerate(self._session_scope.approved_tools):
            if scope.tool_name == tool_name:
                # Update existing scope
                self._session_scope.approved_tools[i] = ToolScope(
                    tool_name=tool_name,
                    approved=True,
                    conditions=conditions,
                    granted_by=granted_by,
                    max_risk_level=max_risk_level.value,
                )
                logger.info(f"Updated session scope for '{tool_name}'")
                return f"session:{self.session_id}:tool:{tool_name}"

        # Add new scope
        tool_scope = ToolScope(
            tool_name=tool_name,
            approved=True,
            conditions=conditions,
            granted_by=granted_by,
            max_risk_level=max_risk_level.value,
        )
        self._session_scope.approved_tools.append(tool_scope)

        # Fire callbacks
        for callback in self._on_scope_granted:
            try:
                callback(ApprovalScope.TOOL, tool_name, conditions)
            except Exception as e:
                logger.warning(f"Scope callback error: {e}")

        logger.info(f"Granted session scope for '{tool_name}' with conditions: {conditions}")
        return f"session:{self.session_id}:tool:{tool_name}"

    def grant_pattern_approval(
        self,
        pattern: str,
    ) -> str:
        """
        Grant approval for activities matching pattern.

        Patterns:
        - "read any file in /workspace"
        - "execute shell commands"

        Args:
            pattern: Approval pattern string

        Returns:
            Scope ID
        """
        if pattern not in self._session_scope.approved_patterns:
            self._session_scope.approved_patterns.append(pattern)
            logger.info(f"Granted pattern scope: '{pattern}'")

        for callback in self._on_scope_granted:
            try:
                callback(ApprovalScope.PATTERN, pattern, None)
            except Exception as e:
                logger.warning(f"Scope callback error: {e}")

        return f"session:{self.session_id}:pattern:{pattern}"

    def grant_full_session_approval(
        self,
        expires_in_seconds: int | None = None,
    ) -> str:
        """
        Grant full session approval for all activities.

        Args:
            expires_in_seconds: Optional expiration time

        Returns:
            Scope ID
        """
        self._session_scope.session_approval = True

        if expires_in_seconds:
            self._session_scope.expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)

        for callback in self._on_scope_granted:
            try:
                callback(ApprovalScope.SESSION, "all", None)
            except Exception as e:
                logger.warning(f"Scope callback error: {e}")

        logger.info("Granted full session approval")
        return f"session:{self.session_id}:full"

    def grant_persistent_approval(
        self,
        tool_name: str,
        conditions: dict[str, Any] | None = None,
        max_risk_level: RiskLevel = RiskLevel.LOW,
        granted_by: str = "",
    ) -> str:
        """
        Grant persistent approval for tool.

        Args:
            tool_name: Tool to approve
            conditions: Conditions for approval
            max_risk_level: Maximum risk level
            granted_by: Who granted

        Returns:
            Scope ID
        """
        # Check if exists
        for i, scope in enumerate(self._persistent_scope.approved_tools):
            if scope.tool_name == tool_name:
                self._persistent_scope.approved_tools[i] = ToolScope(
                    tool_name=tool_name,
                    approved=True,
                    conditions=conditions,
                    granted_by=granted_by,
                    max_risk_level=max_risk_level.value,
                )
                self._persistent_scope.updated_at = datetime.now()
                logger.info(f"Updated persistent scope for '{tool_name}'")
                return f"persistent:{self.user_id}:tool:{tool_name}"

        tool_scope = ToolScope(
            tool_name=tool_name,
            approved=True,
            conditions=conditions,
            granted_by=granted_by,
            max_risk_level=max_risk_level.value,
        )
        self._persistent_scope.approved_tools.append(tool_scope)
        self._persistent_scope.updated_at = datetime.now()

        logger.info(f"Granted persistent scope for '{tool_name}'")
        return f"persistent:{self.user_id}:tool:{tool_name}"

    def revoke_session_approval(self, tool_name: str) -> bool:
        """Revoke session approval for tool."""
        original_len = len(self._session_scope.approved_tools)
        self._session_scope.approved_tools = [
            t for t in self._session_scope.approved_tools if t.tool_name != tool_name
        ]

        revoked = len(self._session_scope.approved_tools) < original_len
        if revoked:
            for callback in self._on_scope_revoked:
                try:
                    callback(ApprovalScope.TOOL, tool_name)
                except Exception as e:
                    logger.warning(f"Scope callback error: {e}")
            logger.info(f"Revoked session scope for '{tool_name}'")

        return revoked

    def revoke_pattern_approval(self, pattern: str) -> bool:
        """Revoke pattern approval."""
        if pattern in self._session_scope.approved_patterns:
            self._session_scope.approved_patterns.remove(pattern)
            for callback in self._on_scope_revoked:
                try:
                    callback(ApprovalScope.PATTERN, pattern)
                except Exception as e:
                    logger.warning(f"Scope callback error: {e}")
            logger.info(f"Revoked pattern scope: '{pattern}'")
            return True
        return False

    def revoke_full_session_approval(self) -> None:
        """Revoke full session approval."""
        self._session_scope.session_approval = False
        self._session_scope.expires_at = None
        for callback in self._on_scope_revoked:
            try:
                callback(ApprovalScope.SESSION, "all")
            except Exception as e:
                logger.warning(f"Scope callback error: {e}")
        logger.info("Revoked full session approval")

    def revoke_persistent_approval(self, tool_name: str) -> bool:
        """Revoke persistent approval for tool."""
        original_len = len(self._persistent_scope.approved_tools)
        self._persistent_scope.approved_tools = [
            t for t in self._persistent_scope.approved_tools if t.tool_name != tool_name
        ]
        self._persistent_scope.updated_at = datetime.now()

        revoked = len(self._persistent_scope.approved_tools) < original_len
        if revoked:
            logger.info(f"Revoked persistent scope for '{tool_name}'")
        return revoked

    def clear_session_scopes(self) -> int:
        """Clear all session scopes."""
        count = len(self._session_scope.approved_tools) + len(self._session_scope.approved_patterns)
        self._session_scope = SessionScope(session_id=self.session_id)
        logger.info(f"Cleared {count} session scopes")
        return count

    def clear_persistent_scopes(self) -> int:
        """Clear all persistent scopes."""
        count = len(self._persistent_scope.approved_tools)
        self._persistent_scope = PersistentScope(user_id=self.user_id)
        logger.info(f"Cleared {count} persistent scopes")
        return count

    def _matches_approval_pattern(
        self,
        activity_type: str,
        parameters: dict[str, Any],
        pattern: str,
    ) -> bool:
        """Check if activity matches an approval pattern string."""
        # First, try direct glob matching on activity_type
        # This handles patterns like "file_*", "bash_*", etc.
        if fnmatch.fnmatch(activity_type, pattern):
            return True

        pattern_lower = pattern.lower()

        # Common patterns
        if "any file" in pattern_lower or "all files" in pattern_lower:
            if activity_type in ("file_read", "file_search", "list_directory"):
                # Check path condition in pattern
                if "in" in pattern_lower:
                    # Extract path pattern
                    path_match = re.search(r"in\s+([/\w*]+)", pattern_lower)
                    if path_match:
                        path_pattern = path_match.group(1)
                        actual_path = parameters.get("path", "")
                        if fnmatch.fnmatch(actual_path, path_pattern):
                            return True
                else:
                    return True

        if "shell command" in pattern_lower or "bash" in pattern_lower:
            if activity_type in ("bash_exec", "python_exec"):
                return True

        if "execute" in pattern_lower:
            if "exec" in activity_type:
                return True

        return False

    def get_session_scope(self) -> SessionScope:
        """Get current session scope."""
        return self._session_scope

    def get_persistent_scope(self) -> PersistentScope:
        """Get current persistent scope."""
        return self._persistent_scope

    def load_persistent_scope(self, scope: PersistentScope) -> None:
        """Load persistent scope from storage."""
        self._persistent_scope = scope
        logger.info(f"Loaded persistent scope with {len(scope.approved_tools)} tool approvals")

    def on_scope_granted(self, callback: callable) -> None:
        """Register callback for scope grants."""
        self._on_scope_granted.append(callback)

    def on_scope_revoked(self, callback: callable) -> None:
        """Register callback for scope revocations."""
        self._on_scope_revoked.append(callback)

    def get_statistics(self) -> dict[str, Any]:
        """Get scope statistics."""
        return {
            "session": {
                "tool_scopes": len(self._session_scope.approved_tools),
                "pattern_scopes": len(self._session_scope.approved_patterns),
                "full_session": self._session_scope.session_approval,
                "expires_at": self._session_scope.expires_at.isoformat()
                if self._session_scope.expires_at
                else None,
            },
            "persistent": {
                "tool_scopes": len(self._persistent_scope.approved_tools),
            },
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ApprovalScopeManager",
    "ScopeConditionMatcher",
]
