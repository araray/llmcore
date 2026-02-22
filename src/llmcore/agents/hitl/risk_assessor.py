# src/llmcore/agents/hitl/risk_assessor.py
"""
Risk Assessment Engine for HITL System.

Evaluates the risk level of agent activities based on:
- Activity type and inherent risk
- Resource scope (system files vs workspace)
- Reversibility (read-only vs destructive)
- Dangerous patterns detection
- User-defined tool classifications

Research Foundation:
- CRITIC: Self-Correction with Tool-Interactive Critiquing (Gou et al., 2024)

References:
    - Master Plan: Section 21 (Risk Assessment Engine)
    - Technical Spec: Section 5.5.2 (Risk Levels)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any

from .models import (
    HITLConfig,
    RiskAssessment,
    RiskFactor,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RISK LEVELS
# =============================================================================


class RiskLevel(str, Enum):
    """Risk levels for HITL decisions."""

    NONE = "none"  # Safe, no approval needed (think_aloud, final_answer)
    LOW = "low"  # Inform user, proceed (read_file, list_directory)
    MEDIUM = "medium"  # Request approval, default reject on timeout
    HIGH = "high"  # Require explicit approval (execute_shell)
    CRITICAL = "critical"  # Block without admin override (execute_sudo)

    @property
    def order(self) -> int:
        """Get numeric order for comparison."""
        return ["none", "low", "medium", "high", "critical"].index(self.value)

    def __lt__(self, other: RiskLevel) -> bool:
        return self.order < other.order

    def __le__(self, other: RiskLevel) -> bool:
        return self.order <= other.order

    def __gt__(self, other: RiskLevel) -> bool:
        return self.order > other.order

    def __ge__(self, other: RiskLevel) -> bool:
        return self.order >= other.order


# =============================================================================
# DANGEROUS PATTERNS
# =============================================================================


@dataclass
class DangerousPattern:
    """Pattern that indicates dangerous activity."""

    pattern: str
    description: str
    risk_level: RiskLevel = RiskLevel.HIGH
    parameter_name: str = "command"  # Which parameter to check
    compiled: Pattern | None = field(default=None, repr=False)

    def __post_init__(self):
        """Compile regex pattern."""
        try:
            self.compiled = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Invalid dangerous pattern '{self.pattern}': {e}")
            self.compiled = None

    def matches(self, value: str) -> bool:
        """Check if pattern matches value."""
        if not self.compiled:
            return False
        return bool(self.compiled.search(value))


# Default dangerous patterns
DEFAULT_DANGEROUS_PATTERNS = [
    # Shell dangers
    DangerousPattern(
        pattern=r"rm\s+-rf\s+/",
        description="Recursive delete from root",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"rm\s+-rf\s+\*",
        description="Recursive delete all",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r">\s*/dev/sd[a-z]",
        description="Direct disk write",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"dd\s+if=.*of=/dev/",
        description="Direct disk write with dd",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"mkfs\.",
        description="Filesystem format",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"chmod\s+777",
        description="Overly permissive chmod",
        risk_level=RiskLevel.HIGH,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"chown\s+.*:.*\s+/",
        description="Chown system files",
        risk_level=RiskLevel.HIGH,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"curl.*\|\s*(ba)?sh",
        description="Remote code execution via pipe",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    DangerousPattern(
        pattern=r"wget.*\|\s*(ba)?sh",
        description="Remote code execution via pipe",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="command",
    ),
    # SQL dangers
    DangerousPattern(
        pattern=r"DROP\s+DATABASE",
        description="Database deletion",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="code",
    ),
    DangerousPattern(
        pattern=r"DROP\s+TABLE",
        description="Table deletion",
        risk_level=RiskLevel.HIGH,
        parameter_name="code",
    ),
    DangerousPattern(
        pattern=r"TRUNCATE\s+TABLE",
        description="Table truncation",
        risk_level=RiskLevel.HIGH,
        parameter_name="code",
    ),
    DangerousPattern(
        pattern=r"DELETE\s+FROM\s+\w+\s*($|;)",
        description="Delete without WHERE clause",
        risk_level=RiskLevel.HIGH,
        parameter_name="code",
    ),
    # Path dangers
    DangerousPattern(
        pattern=r"^/etc/",
        description="System configuration access",
        risk_level=RiskLevel.HIGH,
        parameter_name="path",
    ),
    DangerousPattern(
        pattern=r"^/root/",
        description="Root home access",
        risk_level=RiskLevel.HIGH,
        parameter_name="path",
    ),
    DangerousPattern(
        pattern=r"^/var/log/",
        description="System log access",
        risk_level=RiskLevel.MEDIUM,
        parameter_name="path",
    ),
    DangerousPattern(
        pattern=r"\.ssh/",
        description="SSH key access",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="path",
    ),
    DangerousPattern(
        pattern=r"\.aws/",
        description="AWS credentials access",
        risk_level=RiskLevel.CRITICAL,
        parameter_name="path",
    ),
    DangerousPattern(
        pattern=r"\.env",
        description="Environment file access",
        risk_level=RiskLevel.HIGH,
        parameter_name="path",
    ),
    DangerousPattern(
        pattern=r"password|secret|token|key|credential",
        description="Sensitive data pattern",
        risk_level=RiskLevel.MEDIUM,
        parameter_name="path",
    ),
]


# =============================================================================
# RESOURCE SCOPE PATTERNS
# =============================================================================


@dataclass
class ResourceScope:
    """Defines resource scope risk levels."""

    # Safe paths (workspace, temp)
    safe_patterns: list[str] = field(
        default_factory=lambda: [
            r"^/workspace/",
            r"^/tmp/",
            r"^\./",
            r"^\.\.?/",  # Relative paths
        ]
    )

    # Sensitive paths (system)
    sensitive_patterns: list[str] = field(
        default_factory=lambda: [
            r"^/etc/",
            r"^/root/",
            r"^/home/(?!claude)",  # Other users' homes
            r"^/var/",
            r"^/usr/",
            r"^/opt/",
        ]
    )

    def get_scope_risk(self, path: str) -> RiskLevel:
        """Get risk level based on path scope."""
        # Check safe patterns
        for pattern in self.safe_patterns:
            if re.match(pattern, path, re.IGNORECASE):
                return RiskLevel.LOW

        # Check sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.match(pattern, path, re.IGNORECASE):
                return RiskLevel.HIGH

        # Default
        return RiskLevel.MEDIUM


# =============================================================================
# RISK ASSESSOR
# =============================================================================


class RiskAssessor:
    """
    Assess risk level of agent activities.

    Evaluates risk based on:
    - Activity type (shell commands = HIGH, read file = LOW)
    - Resource scope (system files = HIGH, workspace = LOW)
    - Reversibility (delete = HIGH, read = NONE)
    - User-defined tool classifications
    - Dangerous pattern detection

    Usage:
        >>> assessor = RiskAssessor()
        >>> risk = assessor.assess(
        ...     activity_type="bash_exec",
        ...     parameters={"command": "ls -la"}
        ... )
        >>> print(risk.overall_level, risk.requires_approval)
    """

    def __init__(
        self,
        config: HITLConfig | None = None,
        custom_patterns: list[DangerousPattern] | None = None,
        resource_scope: ResourceScope | None = None,
    ):
        """
        Initialize risk assessor.

        Args:
            config: HITL configuration
            custom_patterns: Additional dangerous patterns
            resource_scope: Resource scope configuration
        """
        self.config = config or HITLConfig()
        self.resource_scope = resource_scope or ResourceScope()

        # Build tool risk map from config
        self._tool_risks: dict[str, RiskLevel] = {}
        self._build_tool_risk_map()

        # Dangerous patterns
        self._patterns = list(DEFAULT_DANGEROUS_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # Custom risk assessors (activity_type -> assessor function)
        self._custom_assessors: dict[str, Callable[..., RiskLevel]] = {}

    def _build_tool_risk_map(self) -> None:
        """Build tool -> risk level mapping from config."""
        for tool in self.config.safe_tools:
            self._tool_risks[tool] = RiskLevel.NONE
        for tool in self.config.low_risk_tools:
            self._tool_risks[tool] = RiskLevel.LOW
        for tool in self.config.high_risk_tools:
            self._tool_risks[tool] = RiskLevel.HIGH
        for tool in self.config.critical_tools:
            self._tool_risks[tool] = RiskLevel.CRITICAL

    def assess(
        self,
        activity_type: str,
        parameters: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> RiskAssessment:
        """
        Assess risk for an activity.

        Args:
            activity_type: Type of activity
            parameters: Activity parameters
            context: Additional context

        Returns:
            RiskAssessment with overall level and factors
        """
        factors: list[RiskFactor] = []
        dangerous_patterns: list[str] = []
        context = context or {}

        # 1. Check tool-based risk
        tool_risk = self._assess_tool_risk(activity_type)
        factors.append(
            RiskFactor(
                name="tool_type",
                level=tool_risk.value,
                reason=f"Tool '{activity_type}' has inherent risk level",
                weight=1.5,
            )
        )

        # 2. Check resource scope
        scope_risk = self._assess_scope_risk(activity_type, parameters)
        if scope_risk > RiskLevel.NONE:
            factors.append(
                RiskFactor(
                    name="resource_scope",
                    level=scope_risk.value,
                    reason="Based on resource paths accessed",
                    weight=1.0,
                )
            )

        # 3. Check reversibility
        reversibility_risk = self._assess_reversibility(activity_type, parameters)
        if reversibility_risk > RiskLevel.NONE:
            factors.append(
                RiskFactor(
                    name="reversibility",
                    level=reversibility_risk.value,
                    reason="Based on destructive potential",
                    weight=1.2,
                )
            )

        # 4. Check dangerous patterns
        pattern_risk, matched_patterns = self._check_dangerous_patterns(activity_type, parameters)
        if pattern_risk > RiskLevel.NONE:
            factors.append(
                RiskFactor(
                    name="dangerous_patterns",
                    level=pattern_risk.value,
                    reason=f"Matched patterns: {', '.join(matched_patterns)}",
                    weight=2.0,
                )
            )
            dangerous_patterns.extend(matched_patterns)

        # 5. Check custom assessors
        if activity_type in self._custom_assessors:
            custom_risk = self._custom_assessors[activity_type](parameters, context)
            factors.append(
                RiskFactor(
                    name="custom_assessment",
                    level=custom_risk.value,
                    reason="Custom risk assessment",
                    weight=1.0,
                )
            )

        # Calculate overall risk level
        overall_level = self._calculate_overall_risk(factors)

        # Determine if approval is required
        threshold = RiskLevel(self.config.global_risk_threshold)
        requires_approval = overall_level >= threshold

        # Build reason
        high_factors = [f for f in factors if RiskLevel(f.level) >= RiskLevel.MEDIUM]
        if dangerous_patterns:
            reason = f"Dangerous patterns detected: {', '.join(dangerous_patterns)}"
        elif high_factors:
            reason = f"High risk factors: {', '.join(f.name for f in high_factors)}"
        else:
            reason = f"Overall risk level: {overall_level.value}"

        return RiskAssessment(
            overall_level=overall_level.value,
            factors=factors,
            requires_approval=requires_approval,
            reason=reason,
            dangerous_patterns=dangerous_patterns,
        )

    def _assess_tool_risk(self, activity_type: str) -> RiskLevel:
        """Get tool-based risk level."""
        if activity_type in self._tool_risks:
            return self._tool_risks[activity_type]

        # Default risk by common patterns
        if "exec" in activity_type or "shell" in activity_type:
            return RiskLevel.HIGH
        elif "delete" in activity_type or "remove" in activity_type:
            return RiskLevel.HIGH
        elif "write" in activity_type or "modify" in activity_type:
            return RiskLevel.MEDIUM
        elif "read" in activity_type or "search" in activity_type:
            return RiskLevel.LOW
        else:
            return RiskLevel.MEDIUM

    def _assess_scope_risk(self, activity_type: str, parameters: dict[str, Any]) -> RiskLevel:
        """Assess risk based on resource scope."""
        max_risk = RiskLevel.NONE

        # Check path-based parameters
        for param_name in ["path", "file", "directory", "target", "source", "destination"]:
            if param_name in parameters:
                path = str(parameters[param_name])
                risk = self.resource_scope.get_scope_risk(path)
                max_risk = max(max_risk, risk)

        return max_risk

    def _assess_reversibility(self, activity_type: str, parameters: dict[str, Any]) -> RiskLevel:
        """Assess risk based on reversibility."""
        # Destructive activities
        destructive = {
            "file_delete": RiskLevel.HIGH,
            "bash_exec": RiskLevel.MEDIUM,  # Depends on command
            "python_exec": RiskLevel.MEDIUM,  # Depends on code
        }

        if activity_type in destructive:
            return destructive[activity_type]

        # Check for destructive patterns in parameters
        for value in parameters.values():
            if isinstance(value, str):
                if any(
                    kw in value.lower()
                    for kw in ["delete", "remove", "drop", "truncate", "destroy"]
                ):
                    return RiskLevel.HIGH

        return RiskLevel.NONE

    def _check_dangerous_patterns(
        self, activity_type: str, parameters: dict[str, Any]
    ) -> tuple[RiskLevel, list[str]]:
        """Check for dangerous patterns in parameters."""
        max_risk = RiskLevel.NONE
        matched: list[str] = []

        for pattern in self._patterns:
            param_name = pattern.parameter_name
            # Check the specific parameter or all string parameters
            values_to_check: list[str] = []

            if param_name in parameters:
                values_to_check.append(str(parameters[param_name]))
            else:
                # Check all string parameters
                for value in parameters.values():
                    if isinstance(value, str):
                        values_to_check.append(value)

            for value in values_to_check:
                if pattern.matches(value):
                    matched.append(pattern.description)
                    max_risk = max(max_risk, pattern.risk_level)
                    logger.warning(f"Dangerous pattern detected: {pattern.description}")

        return max_risk, matched

    def _calculate_overall_risk(self, factors: list[RiskFactor]) -> RiskLevel:
        """Calculate overall risk from factors."""
        if not factors:
            return RiskLevel.NONE

        # Weighted approach
        risk_scores = {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }

        total_weight = sum(f.weight for f in factors)
        weighted_sum = sum(risk_scores.get(f.level, 0) * f.weight for f in factors)
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Also check for any critical factors
        if any(f.level == "critical" for f in factors):
            return RiskLevel.CRITICAL

        # Map back to risk level
        if avg_score >= 3.5:
            return RiskLevel.CRITICAL
        elif avg_score >= 2.5:
            return RiskLevel.HIGH
        elif avg_score >= 1.5:
            return RiskLevel.MEDIUM
        elif avg_score >= 0.5:
            return RiskLevel.LOW
        else:
            return RiskLevel.NONE

    def get_tool_risk_level(self, tool_name: str) -> RiskLevel:
        """Get configured risk level for a tool."""
        return self._tool_risks.get(tool_name, RiskLevel.MEDIUM)

    def set_tool_risk_level(self, tool_name: str, risk_level: RiskLevel) -> None:
        """Set risk level for a tool."""
        self._tool_risks[tool_name] = risk_level
        logger.info(f"Set tool '{tool_name}' risk level to {risk_level.value}")

    def add_dangerous_pattern(self, pattern: DangerousPattern) -> None:
        """Add a custom dangerous pattern."""
        self._patterns.append(pattern)
        logger.info(f"Added dangerous pattern: {pattern.description}")

    def register_custom_assessor(
        self,
        activity_type: str,
        assessor: Callable[[dict[str, Any], dict[str, Any]], RiskLevel],
    ) -> None:
        """
        Register a custom risk assessor for an activity type.

        Args:
            activity_type: Activity type to assess
            assessor: Function(parameters, context) -> RiskLevel
        """
        self._custom_assessors[activity_type] = assessor
        logger.info(f"Registered custom assessor for '{activity_type}'")

    def is_safe_tool(self, tool_name: str) -> bool:
        """Check if tool is classified as safe (no approval needed)."""
        return (
            tool_name in self.config.safe_tools or self._tool_risks.get(tool_name) == RiskLevel.NONE
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_risk_assessor(
    config: HITLConfig | None = None,
    additional_safe_tools: list[str] | None = None,
    additional_dangerous_patterns: list[DangerousPattern] | None = None,
) -> RiskAssessor:
    """Create a risk assessor with custom settings."""
    cfg = config or HITLConfig()

    if additional_safe_tools:
        cfg.safe_tools.extend(additional_safe_tools)

    return RiskAssessor(
        config=cfg,
        custom_patterns=additional_dangerous_patterns,
    )


def quick_assess(
    activity_type: str,
    parameters: dict[str, Any],
) -> RiskAssessment:
    """Quickly assess risk without creating an instance."""
    assessor = RiskAssessor()
    return assessor.assess(activity_type, parameters)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "RiskAssessor",
    # Support classes
    "RiskLevel",
    "DangerousPattern",
    "ResourceScope",
    # Defaults
    "DEFAULT_DANGEROUS_PATTERNS",
    # Functions
    "create_risk_assessor",
    "quick_assess",
]
