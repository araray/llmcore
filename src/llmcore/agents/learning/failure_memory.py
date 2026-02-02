# src/llmcore/agents/learning/failure_memory.py
"""
Failure Memory for Agent Learning.

Tracks failures across sessions to enable learning from past mistakes.
Provides pattern recognition for recurring failures and suggestions
for avoiding them.

Key Features:
    - Record failures with context
    - Pattern detection for recurring issues
    - Suggestion generation for similar situations
    - Session and persistent storage options

Usage:
    from llmcore.agents.learning import FailureMemory

    memory = FailureMemory()

    # Record a failure
    memory.record_failure(
        failure_type=FailureType.TOOL_ERROR,
        description="Model does not support tools",
        context={"model": "gemma:7b", "tool": "file_read"}
    )

    # Check for similar failures
    similar = memory.find_similar("use tools with gemma")
    if similar:
        print(f"Warning: {similar[0].suggestion}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class FailureType(str, Enum):
    """Types of failures that can occur."""

    TOOL_ERROR = "tool_error"  # Tool execution failed
    MODEL_ERROR = "model_error"  # Model-related error
    CAPABILITY_ERROR = "capability"  # Model doesn't support feature
    TIMEOUT = "timeout"  # Operation timed out
    VALIDATION_ERROR = "validation"  # Validation failed
    EXECUTION_ERROR = "execution"  # General execution error
    LOOP_DETECTED = "loop"  # Stuck in loop
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
    CONTEXT_ERROR = "context"  # Context/memory error
    PARSE_ERROR = "parse"  # Parsing error
    NETWORK_ERROR = "network"  # Network/API error
    PERMISSION_ERROR = "permission"  # Permission denied
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    """Severity of failures."""

    LOW = "low"  # Minor issue, can continue
    MEDIUM = "medium"  # Significant but recoverable
    HIGH = "high"  # Serious, needs intervention
    CRITICAL = "critical"  # Blocking, cannot proceed


@dataclass
class FailureRecord:
    """Record of a single failure."""

    id: str
    failure_type: FailureType
    description: str
    context: Dict[str, Any]
    severity: FailureSeverity = FailureSeverity.MEDIUM
    suggestion: Optional[str] = None

    # Tracking
    session_id: Optional[str] = None
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)

    # Pattern detection
    fingerprint: str = ""  # Hash for deduplication
    occurrence_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Learning
    resolution: Optional[str] = None
    resolved: bool = False
    resolution_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "failure_type": self.failure_type.value,
            "description": self.description,
            "context": self.context,
            "severity": self.severity.value,
            "suggestion": self.suggestion,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            failure_type=FailureType(data["failure_type"]),
            description=data["description"],
            context=data.get("context", {}),
            severity=FailureSeverity(data.get("severity", "medium")),
            suggestion=data.get("suggestion"),
            session_id=data.get("session_id"),
            iteration=data.get("iteration", 0),
            timestamp=data.get("timestamp", time.time()),
            fingerprint=data.get("fingerprint", ""),
            occurrence_count=data.get("occurrence_count", 1),
            first_seen=data.get("first_seen", time.time()),
            last_seen=data.get("last_seen", time.time()),
            resolution=data.get("resolution"),
            resolved=data.get("resolved", False),
            resolution_timestamp=data.get("resolution_timestamp"),
        )


@dataclass
class FailurePattern:
    """Detected pattern of recurring failures."""

    pattern_type: FailureType
    description: str
    occurrences: int
    affected_models: List[str]
    affected_tools: List[str]
    common_context: Dict[str, Any]
    suggestion: str
    severity: FailureSeverity
    confidence: float  # 0-1, how confident we are in this pattern

    def format(self) -> str:
        """Format pattern for display."""
        return f"""
Pattern: {self.pattern_type.value}
Description: {self.description}
Occurrences: {self.occurrences}
Affected models: {", ".join(self.affected_models) or "any"}
Affected tools: {", ".join(self.affected_tools) or "any"}
Severity: {self.severity.value}
Suggestion: {self.suggestion}
Confidence: {self.confidence:.0%}
"""


# =============================================================================
# Failure Suggestions
# =============================================================================


# Pre-defined suggestions for common failure patterns
FAILURE_SUGGESTIONS: Dict[FailureType, Dict[str, str]] = {
    FailureType.CAPABILITY_ERROR: {
        "default": "Check model capabilities before attempting operation",
        "tools": "Use activity system instead of native tools for this model",
        "vision": "Use a model with vision capabilities",
        "long_context": "Use a model with larger context window",
    },
    FailureType.TOOL_ERROR: {
        "default": "Verify tool parameters and try again",
        "not_found": "Tool not registered, check activity registry",
        "permission": "Check permissions for the operation",
        "network": "Check network connectivity and retry",
    },
    FailureType.TIMEOUT: {
        "default": "Increase timeout or simplify operation",
        "model": "Model response slow, consider using a faster model",
        "tool": "Tool execution slow, check for blocking operations",
    },
    FailureType.LOOP_DETECTED: {
        "default": "Break the loop by trying a different approach",
        "same_error": "Address the underlying error before retrying",
        "no_progress": "Reassess the goal or request clarification",
    },
    FailureType.MAX_ITERATIONS: {
        "default": "Goal may be too complex, consider decomposition",
        "planning": "Over-planning detected, use direct approach",
        "validation": "Relax validation criteria or get human input",
    },
    FailureType.PARSE_ERROR: {
        "default": "Check response format and parsing logic",
        "xml": "Model not following XML format, try clearer instructions",
        "json": "Model not producing valid JSON, add examples",
    },
}


def get_suggestion(
    failure_type: FailureType,
    context: Dict[str, Any],
) -> str:
    """Get suggestion for a failure type and context."""
    type_suggestions = FAILURE_SUGGESTIONS.get(failure_type, {})

    # Try to match context keys
    for key, suggestion in type_suggestions.items():
        if key in str(context).lower():
            return suggestion

    return type_suggestions.get("default", "Review the failure and adjust approach")


# =============================================================================
# Failure Memory Implementation
# =============================================================================


class FailureMemory:
    """
    Memory for tracking and learning from failures.

    Provides:
    - Recording of failures with context
    - Pattern detection for recurring issues
    - Similarity search for related failures
    - Suggestions based on past experience
    - Persistence (optional)

    Args:
        persist_path: Path for persistence (None = in-memory only)
        max_records: Maximum failure records to keep
        dedup_window_seconds: Window for deduplicating same failures
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        max_records: int = 1000,
        dedup_window_seconds: float = 60.0,
        session_id: Optional[str] = None,
    ):
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_records = max_records
        self.dedup_window_seconds = dedup_window_seconds
        self.session_id = session_id

        self._records: Dict[str, FailureRecord] = {}
        self._fingerprint_index: Dict[str, str] = {}  # fingerprint -> record_id
        self._type_index: Dict[FailureType, List[str]] = {}  # type -> record_ids

        # Load from persistence
        if self.persist_path and self.persist_path.exists():
            self._load()

    def record_failure(
        self,
        failure_type: FailureType,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        severity: Optional[FailureSeverity] = None,
        suggestion: Optional[str] = None,
        iteration: int = 0,
    ) -> FailureRecord:
        """
        Record a failure.

        Args:
            failure_type: Type of failure
            description: Human-readable description
            context: Additional context (model, tool, params, etc.)
            severity: Failure severity (auto-detected if None)
            suggestion: Custom suggestion (auto-generated if None)
            iteration: Current iteration number

        Returns:
            The created or updated FailureRecord
        """
        context = context or {}

        # Generate fingerprint for deduplication
        fingerprint = self._generate_fingerprint(failure_type, description, context)

        # Check for duplicate within window
        if fingerprint in self._fingerprint_index:
            existing_id = self._fingerprint_index[fingerprint]
            existing = self._records.get(existing_id)
            if existing and (time.time() - existing.last_seen) < self.dedup_window_seconds:
                # Update existing record
                existing.occurrence_count += 1
                existing.last_seen = time.time()
                existing.iteration = iteration
                logger.debug(f"Updated existing failure record: {existing.id}")
                return existing

        # Auto-detect severity if not provided
        if severity is None:
            severity = self._infer_severity(failure_type, context)

        # Auto-generate suggestion if not provided
        if suggestion is None:
            suggestion = get_suggestion(failure_type, context)

        # Create new record
        record_id = f"fail_{int(time.time() * 1000)}_{len(self._records)}"
        record = FailureRecord(
            id=record_id,
            failure_type=failure_type,
            description=description,
            context=context,
            severity=severity,
            suggestion=suggestion,
            session_id=self.session_id,
            iteration=iteration,
            fingerprint=fingerprint,
        )

        # Store
        self._records[record_id] = record
        self._fingerprint_index[fingerprint] = record_id

        # Update type index
        if failure_type not in self._type_index:
            self._type_index[failure_type] = []
        self._type_index[failure_type].append(record_id)

        # Prune if needed
        self._prune()

        # Persist
        if self.persist_path:
            self._save()

        logger.debug(f"Recorded failure: {failure_type.value} - {description[:50]}")
        return record

    def find_similar(
        self,
        query: str,
        failure_type: Optional[FailureType] = None,
        limit: int = 5,
    ) -> List[FailureRecord]:
        """
        Find failures similar to a query.

        Uses simple text matching. For production, consider
        using embeddings for semantic similarity.

        Args:
            query: Query string to match against
            failure_type: Filter to specific type
            limit: Maximum results

        Returns:
            List of similar FailureRecords
        """
        candidates = list(self._records.values())

        # Filter by type
        if failure_type:
            candidates = [r for r in candidates if r.failure_type == failure_type]

        # Score by text similarity
        query_words = set(query.lower().split())

        def score(record: FailureRecord) -> float:
            record_text = f"{record.description} {json.dumps(record.context)}".lower()
            record_words = set(record_text.split())

            if not query_words or not record_words:
                return 0.0

            overlap = len(query_words & record_words)
            return overlap / len(query_words)

        # Sort by score
        scored = [(score(r), r) for r in candidates]
        scored = [(s, r) for s, r in scored if s > 0.1]  # Min threshold
        scored.sort(key=lambda x: (-x[0], -x[1].occurrence_count))

        return [r for _, r in scored[:limit]]

    def get_patterns(self, min_occurrences: int = 3) -> List[FailurePattern]:
        """
        Detect patterns in failure records.

        Args:
            min_occurrences: Minimum occurrences to form a pattern

        Returns:
            List of detected FailurePatterns
        """
        patterns = []

        # Group by type
        for failure_type, record_ids in self._type_index.items():
            records = [self._records[rid] for rid in record_ids if rid in self._records]

            if len(records) < min_occurrences:
                continue

            # Find common context
            common_context = self._find_common_context(records)

            # Extract affected models/tools
            affected_models = list(
                set(r.context.get("model", "") for r in records if r.context.get("model"))
            )
            affected_tools = list(
                set(
                    r.context.get("tool", r.context.get("activity", ""))
                    for r in records
                    if r.context.get("tool") or r.context.get("activity")
                )
            )

            # Calculate confidence
            total = len(self._records)
            confidence = len(records) / total if total > 0 else 0

            # Determine severity (worst case)
            severity = max(
                (r.severity for r in records),
                key=lambda s: {
                    FailureSeverity.LOW: 0,
                    FailureSeverity.MEDIUM: 1,
                    FailureSeverity.HIGH: 2,
                    FailureSeverity.CRITICAL: 3,
                }.get(s, 0),
            )

            # Generate pattern suggestion
            suggestion = self._generate_pattern_suggestion(
                failure_type, common_context, affected_models, affected_tools
            )

            pattern = FailurePattern(
                pattern_type=failure_type,
                description=f"Recurring {failure_type.value} failures",
                occurrences=len(records),
                affected_models=affected_models,
                affected_tools=affected_tools,
                common_context=common_context,
                suggestion=suggestion,
                severity=severity,
                confidence=min(confidence * 3, 1.0),  # Scale up confidence
            )
            patterns.append(pattern)

        # Sort by occurrences
        patterns.sort(key=lambda p: -p.occurrences)

        return patterns

    def mark_resolved(
        self,
        record_id: str,
        resolution: str,
    ) -> bool:
        """
        Mark a failure as resolved.

        Args:
            record_id: ID of the failure record
            resolution: Description of how it was resolved

        Returns:
            True if record found and updated
        """
        record = self._records.get(record_id)
        if not record:
            return False

        record.resolved = True
        record.resolution = resolution
        record.resolution_timestamp = time.time()

        if self.persist_path:
            self._save()

        logger.info(f"Failure {record_id} marked as resolved: {resolution[:50]}")
        return True

    def get_unresolved(
        self,
        failure_type: Optional[FailureType] = None,
        severity: Optional[FailureSeverity] = None,
    ) -> List[FailureRecord]:
        """
        Get unresolved failures.

        Args:
            failure_type: Filter by type
            severity: Filter by minimum severity

        Returns:
            List of unresolved FailureRecords
        """
        records = [r for r in self._records.values() if not r.resolved]

        if failure_type:
            records = [r for r in records if r.failure_type == failure_type]

        if severity:
            severity_order = {
                FailureSeverity.LOW: 0,
                FailureSeverity.MEDIUM: 1,
                FailureSeverity.HIGH: 2,
                FailureSeverity.CRITICAL: 3,
            }
            min_level = severity_order.get(severity, 0)
            records = [r for r in records if severity_order.get(r.severity, 0) >= min_level]

        return sorted(records, key=lambda r: -r.occurrence_count)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about failure memory."""
        if not self._records:
            return {
                "total_failures": 0,
                "unique_failures": 0,
                "by_type": {},
                "by_severity": {},
                "resolved_rate": 0.0,
            }

        by_type = {}
        by_severity = {}
        resolved_count = 0
        total_occurrences = 0

        for record in self._records.values():
            by_type[record.failure_type.value] = by_type.get(record.failure_type.value, 0) + 1
            by_severity[record.severity.value] = by_severity.get(record.severity.value, 0) + 1
            total_occurrences += record.occurrence_count
            if record.resolved:
                resolved_count += 1

        return {
            "total_failures": len(self._records),
            "total_occurrences": total_occurrences,
            "by_type": by_type,
            "by_severity": by_severity,
            "resolved_rate": resolved_count / len(self._records) if self._records else 0.0,
            "patterns_detected": len(self.get_patterns()),
        }

    def clear(self, keep_resolved: bool = True) -> int:
        """
        Clear failure memory.

        Args:
            keep_resolved: Whether to keep resolved failures

        Returns:
            Number of records cleared
        """
        if keep_resolved:
            to_remove = [rid for rid, r in self._records.items() if not r.resolved]
        else:
            to_remove = list(self._records.keys())

        for rid in to_remove:
            record = self._records.pop(rid, None)
            if record:
                self._fingerprint_index.pop(record.fingerprint, None)

        # Rebuild type index
        self._type_index = {}
        for rid, record in self._records.items():
            if record.failure_type not in self._type_index:
                self._type_index[record.failure_type] = []
            self._type_index[record.failure_type].append(rid)

        if self.persist_path:
            self._save()

        return len(to_remove)

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _generate_fingerprint(
        self,
        failure_type: FailureType,
        description: str,
        context: Dict[str, Any],
    ) -> str:
        """Generate fingerprint for deduplication."""
        # Include key context fields
        key_fields = ["model", "tool", "activity", "error_code"]
        context_str = "|".join(f"{k}={context.get(k, '')}" for k in key_fields if context.get(k))

        content = f"{failure_type.value}|{description[:100]}|{context_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _infer_severity(
        self,
        failure_type: FailureType,
        context: Dict[str, Any],
    ) -> FailureSeverity:
        """Infer severity from failure type and context."""
        # Critical failures
        if failure_type in [FailureType.LOOP_DETECTED, FailureType.PERMISSION_ERROR]:
            return FailureSeverity.CRITICAL

        # High severity
        if failure_type in [FailureType.CAPABILITY_ERROR, FailureType.MAX_ITERATIONS]:
            return FailureSeverity.HIGH

        # Check context for indicators
        error_msg = str(context.get("error", "")).lower()
        if any(w in error_msg for w in ["critical", "fatal", "crash"]):
            return FailureSeverity.CRITICAL
        if any(w in error_msg for w in ["permission", "denied", "unauthorized"]):
            return FailureSeverity.HIGH

        return FailureSeverity.MEDIUM

    def _find_common_context(
        self,
        records: List[FailureRecord],
    ) -> Dict[str, Any]:
        """Find common context across records."""
        if not records:
            return {}

        # Count value occurrences for each key
        key_values: Dict[str, Dict[str, int]] = {}

        for record in records:
            for key, value in record.context.items():
                if key not in key_values:
                    key_values[key] = {}
                str_value = str(value)
                key_values[key][str_value] = key_values[key].get(str_value, 0) + 1

        # Keep values that appear in >50% of records
        threshold = len(records) * 0.5
        common = {}

        for key, values in key_values.items():
            for value, count in values.items():
                if count >= threshold:
                    common[key] = value
                    break

        return common

    def _generate_pattern_suggestion(
        self,
        failure_type: FailureType,
        common_context: Dict[str, Any],
        affected_models: List[str],
        affected_tools: List[str],
    ) -> str:
        """Generate suggestion for a pattern."""
        base = get_suggestion(failure_type, common_context)

        # Add specific context
        parts = [base]

        if affected_models:
            parts.append(f"Consider using different models than: {', '.join(affected_models[:3])}")

        if affected_tools:
            parts.append(f"Review usage of: {', '.join(affected_tools[:3])}")

        return ". ".join(parts)

    def _prune(self) -> None:
        """Prune old records if over limit."""
        if len(self._records) <= self.max_records:
            return

        # Sort by last_seen and keep most recent
        sorted_records = sorted(
            self._records.items(),
            key=lambda x: x[1].last_seen,
            reverse=True,
        )

        # Remove oldest
        to_remove = sorted_records[self.max_records :]
        for rid, record in to_remove:
            self._records.pop(rid, None)
            self._fingerprint_index.pop(record.fingerprint, None)

        # Rebuild type index
        self._type_index = {}
        for rid, record in self._records.items():
            if record.failure_type not in self._type_index:
                self._type_index[record.failure_type] = []
            self._type_index[record.failure_type].append(rid)

    def _save(self) -> None:
        """Save to persistence."""
        if not self.persist_path:
            return

        try:
            data = {
                "version": "1.0",
                "records": [r.to_dict() for r in self._records.values()],
            }

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.warning(f"Failed to save failure memory: {e}")

    def _load(self) -> None:
        """Load from persistence."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            data = json.loads(self.persist_path.read_text())

            for record_data in data.get("records", []):
                record = FailureRecord.from_dict(record_data)
                self._records[record.id] = record
                self._fingerprint_index[record.fingerprint] = record.id

                if record.failure_type not in self._type_index:
                    self._type_index[record.failure_type] = []
                self._type_index[record.failure_type].append(record.id)

            logger.info(f"Loaded {len(self._records)} failure records")

        except Exception as e:
            logger.warning(f"Failed to load failure memory: {e}")


__all__ = [
    # Enums
    "FailureType",
    "FailureSeverity",
    # Data models
    "FailureRecord",
    "FailurePattern",
    # Memory
    "FailureMemory",
    # Utilities
    "get_suggestion",
]
