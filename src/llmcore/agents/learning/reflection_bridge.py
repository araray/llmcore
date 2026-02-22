# src/llmcore/agents/learning/reflection_bridge.py
"""
Reflection-to-Action Bridge.

The "missing piece" that connects REFLECT phase insights to actual
behavioral changes in subsequent iterations.

Problem:
    Current state: Model accurately recognizes problems in REFLECT phase
    but exhibits ZERO behavioral adaptation in subsequent iterations.

Solution:
    This bridge:
    1. Captures insights from REFLECT phase
    2. Converts them to actionable constraints/guidance
    3. Injects them into future prompts
    4. Tracks whether behavior actually changed

Usage:
    from llmcore.agents.learning import ReflectionBridge

    bridge = ReflectionBridge()

    # After REFLECT phase
    bridge.add_insight(insight)

    # Before THINK phase
    guidance = bridge.get_guidance()
    # Inject guidance into prompt
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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


class InsightType(str, Enum):
    """Types of insights from reflection."""

    PLANNING = "planning"  # Over-planning, under-planning
    TOOL_USE = "tool_use"  # Wrong tool, tool errors
    APPROACH = "approach"  # Wrong approach to problem
    CONTEXT = "context"  # Missing context, wrong assumptions
    CAPABILITY = "capability"  # Model limitations
    LOOP = "loop"  # Stuck in loop, repeating errors
    PROGRESS = "progress"  # Lack of progress
    UNKNOWN = "unknown"


class InsightPriority(str, Enum):
    """Priority of insights."""

    CRITICAL = "critical"  # Must address immediately
    HIGH = "high"  # Should address soon
    MEDIUM = "medium"  # Consider addressing
    LOW = "low"  # Nice to address


@dataclass
class ReflectionInsight:
    """A single insight from reflection."""

    content: str
    insight_type: InsightType = InsightType.UNKNOWN
    priority: InsightPriority = InsightPriority.MEDIUM
    source_iteration: int = 0
    actionable: bool = True
    action_guidance: str | None = None
    avoid_patterns: list[str] = field(default_factory=list)
    prefer_patterns: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    applied_count: int = 0
    effective: bool | None = None  # True if behavior changed after applying

    def to_guidance(self) -> str:
        """Convert insight to actionable guidance."""
        parts = [f"[{self.insight_type.value.upper()}] {self.content}"]

        if self.action_guidance:
            parts.append(f"Action: {self.action_guidance}")

        if self.avoid_patterns:
            parts.append(f"AVOID: {', '.join(self.avoid_patterns)}")

        if self.prefer_patterns:
            parts.append(f"PREFER: {', '.join(self.prefer_patterns)}")

        return " | ".join(parts)


@dataclass
class GuidanceSet:
    """Collection of guidance for a phase."""

    constraints: list[str] = field(default_factory=list)  # Hard constraints
    recommendations: list[str] = field(default_factory=list)  # Soft recommendations
    avoid: set[str] = field(default_factory=set)  # Patterns to avoid
    prefer: set[str] = field(default_factory=set)  # Patterns to prefer
    context: list[str] = field(default_factory=list)  # Additional context

    def format(self, max_items: int = 5) -> str:
        """Format guidance for injection into prompts."""
        lines = []

        if self.constraints:
            lines.append("=== CONSTRAINTS (Must Follow) ===")
            for c in self.constraints[:max_items]:
                lines.append(f"❌ {c}")

        if self.recommendations:
            lines.append("=== RECOMMENDATIONS ===")
            for r in self.recommendations[:max_items]:
                lines.append(f"💡 {r}")

        if self.avoid:
            lines.append("=== AVOID ===")
            lines.append(", ".join(list(self.avoid)[:max_items]))

        if self.prefer:
            lines.append("=== PREFER ===")
            lines.append(", ".join(list(self.prefer)[:max_items]))

        if self.context:
            lines.append("=== CONTEXT FROM PREVIOUS ATTEMPTS ===")
            for c in self.context[:max_items]:
                lines.append(f"📝 {c}")

        return "\n".join(lines) if lines else ""

    def is_empty(self) -> bool:
        """Check if guidance set is empty."""
        return not any(
            [
                self.constraints,
                self.recommendations,
                self.avoid,
                self.prefer,
                self.context,
            ]
        )


# =============================================================================
# Insight Extraction
# =============================================================================


# Patterns for extracting insights from REFLECT phase output
INSIGHT_PATTERNS = {
    InsightType.PLANNING: [
        r"(over[- ]?plan|excessive planning|too (much|detailed) plan)",
        r"(under[- ]?plan|insufficient planning|no plan)",
        r"(plan.*complex|unnecessary.*step|too many step)",
    ],
    InsightType.TOOL_USE: [
        r"(wrong tool|incorrect tool|tool.*error)",
        r"(tool.*not support|unsupported.*tool)",
        r"(missing.*tool|need.*different.*tool)",
    ],
    InsightType.APPROACH: [
        r"(wrong approach|different approach|better way)",
        r"(should have|could have|need to)",
        r"(approach.*not.*work|strategy.*fail)",
    ],
    InsightType.CONTEXT: [
        r"(missing.*context|need.*information|lack.*detail)",
        r"(wrong.*assumption|incorrect.*assumption)",
        r"(misunderstood|didn't understand)",
    ],
    InsightType.CAPABILITY: [
        r"(model.*limit|cannot|unable to)",
        r"(not.*capable|beyond.*abilit)",
        r"(limitation|constraint)",
    ],
    InsightType.LOOP: [
        r"(stuck.*loop|repeating|same.*error)",
        r"(infinite|circular|going.*circles)",
        r"(trap|falling.*into)",
    ],
    InsightType.PROGRESS: [
        r"(no.*progress|lack.*progress|stall)",
        r"(not.*advancing|not.*moving)",
        r"(same.*place|haven't.*achiev)",
    ],
}


class InsightExtractor:
    """Extract structured insights from reflection text."""

    def __init__(self):
        import re

        self._patterns = {
            insight_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for insight_type, patterns in INSIGHT_PATTERNS.items()
        }

    def extract(
        self,
        reflection_text: str,
        iteration: int = 0,
    ) -> list[ReflectionInsight]:
        """Extract insights from reflection text."""
        insights = []

        # Split into sentences for granular analysis
        sentences = self._split_sentences(reflection_text)

        for sentence in sentences:
            insight_type = self._classify_sentence(sentence)
            if insight_type != InsightType.UNKNOWN:
                insight = self._create_insight(sentence, insight_type, iteration)
                insights.append(insight)

        # Deduplicate and prioritize
        return self._deduplicate(insights)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _classify_sentence(self, sentence: str) -> InsightType:
        """Classify sentence by insight type."""
        for insight_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(sentence):
                    return insight_type
        return InsightType.UNKNOWN

    def _create_insight(
        self,
        sentence: str,
        insight_type: InsightType,
        iteration: int,
    ) -> ReflectionInsight:
        """Create insight from sentence."""
        # Determine priority based on type and keywords
        priority = InsightPriority.MEDIUM
        if insight_type in [InsightType.LOOP, InsightType.CAPABILITY]:
            priority = InsightPriority.CRITICAL
        elif insight_type in [InsightType.TOOL_USE, InsightType.APPROACH]:
            priority = InsightPriority.HIGH

        # Extract action guidance
        action_guidance = self._extract_action_guidance(sentence)

        # Extract avoid/prefer patterns
        avoid_patterns = self._extract_avoid_patterns(sentence)
        prefer_patterns = self._extract_prefer_patterns(sentence)

        return ReflectionInsight(
            content=sentence,
            insight_type=insight_type,
            priority=priority,
            source_iteration=iteration,
            actionable=bool(action_guidance or avoid_patterns or prefer_patterns),
            action_guidance=action_guidance,
            avoid_patterns=avoid_patterns,
            prefer_patterns=prefer_patterns,
        )

    def _extract_action_guidance(self, sentence: str) -> str | None:
        """Extract actionable guidance from sentence."""
        import re

        # Look for "should", "need to", "must" patterns
        patterns = [
            r"(should|need to|must|have to)\s+(.+?)(?:[,.]|$)",
            r"(instead.*?)\s+(.*?)(?:[,.]|$)",
            r"(try|use|consider)\s+(.*?)(?:[,.]|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        return None

    def _extract_avoid_patterns(self, sentence: str) -> list[str]:
        """Extract patterns to avoid."""
        import re

        patterns = []

        # "Don't X", "Avoid X", "Stop X"
        matches = re.findall(
            r"(?:don'?t|avoid|stop|cease)\s+(\w+(?:\s+\w+){0,3})",
            sentence,
            re.IGNORECASE,
        )
        patterns.extend(matches)

        # "X is wrong/bad/incorrect"
        matches = re.findall(
            r"(\w+(?:\s+\w+){0,2})\s+(?:is|was|are|were)\s+(?:wrong|bad|incorrect)",
            sentence,
            re.IGNORECASE,
        )
        patterns.extend(matches)

        return patterns

    def _extract_prefer_patterns(self, sentence: str) -> list[str]:
        """Extract patterns to prefer."""
        import re

        patterns = []

        # "Use X", "Try X", "Consider X"
        matches = re.findall(
            r"(?:use|try|consider|prefer)\s+(\w+(?:\s+\w+){0,3})",
            sentence,
            re.IGNORECASE,
        )
        patterns.extend(matches)

        # "X is better/correct/right"
        matches = re.findall(
            r"(\w+(?:\s+\w+){0,2})\s+(?:is|was)\s+(?:better|correct|right)",
            sentence,
            re.IGNORECASE,
        )
        patterns.extend(matches)

        return patterns

    def _deduplicate(self, insights: list[ReflectionInsight]) -> list[ReflectionInsight]:
        """Deduplicate and merge similar insights."""
        if not insights:
            return []

        # Group by type
        by_type: dict[InsightType, list[ReflectionInsight]] = {}
        for insight in insights:
            by_type.setdefault(insight.insight_type, []).append(insight)

        # Take highest priority from each type
        result = []
        priority_order = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3,
        }

        for insight_type, type_insights in by_type.items():
            sorted_insights = sorted(
                type_insights,
                key=lambda i: priority_order.get(i.priority, 99),
            )
            # Take top 2 from each type
            result.extend(sorted_insights[:2])

        return result


# =============================================================================
# Reflection Bridge
# =============================================================================


class ReflectionBridge:
    """
    Bridge between REFLECT phase insights and future behavior.

    This is the core mechanism that enables learning within a session.
    It captures insights, converts them to actionable guidance, and
    tracks whether the guidance was effective.

    Usage:
        bridge = ReflectionBridge()

        # After each REFLECT phase
        bridge.add_reflection(reflection_output, iteration)

        # Before THINK/PLAN phases
        guidance = bridge.get_guidance_for_phase("think")
        prompt = f"{base_prompt}\n\n{guidance.format()}"

        # After execution, mark effectiveness
        bridge.mark_effective(insight_id, effective=True)
    """

    def __init__(
        self,
        max_insights: int = 20,
        insight_ttl_iterations: int = 10,
        auto_extract: bool = True,
    ):
        self.max_insights = max_insights
        self.insight_ttl_iterations = insight_ttl_iterations
        self.auto_extract = auto_extract

        self._insights: list[ReflectionInsight] = []
        self._extractor = InsightExtractor()
        self._current_iteration = 0

    def add_reflection(
        self,
        reflection_text: str,
        iteration: int,
    ) -> list[ReflectionInsight]:
        """
        Process a REFLECT phase output and extract insights.

        Args:
            reflection_text: Raw output from REFLECT phase
            iteration: Current iteration number

        Returns:
            List of extracted insights
        """
        self._current_iteration = iteration

        if self.auto_extract:
            insights = self._extractor.extract(reflection_text, iteration)
        else:
            # Create single insight from full text
            insights = [
                ReflectionInsight(
                    content=reflection_text,
                    source_iteration=iteration,
                )
            ]

        # Add to collection
        for insight in insights:
            self._add_insight(insight)

        logger.debug(f"Extracted {len(insights)} insights from reflection")
        return insights

    def add_insight(self, insight: ReflectionInsight) -> None:
        """Add a pre-constructed insight."""
        self._add_insight(insight)

    def _add_insight(self, insight: ReflectionInsight) -> None:
        """Internal method to add insight with housekeeping."""
        # Check for duplicates
        for existing in self._insights:
            if self._similar(existing, insight):
                # Merge avoid/prefer patterns
                existing.avoid_patterns.extend(insight.avoid_patterns)
                existing.prefer_patterns.extend(insight.prefer_patterns)
                return

        self._insights.append(insight)

        # Prune old insights
        self._prune_insights()

    def _similar(self, a: ReflectionInsight, b: ReflectionInsight) -> bool:
        """Check if two insights are similar."""
        if a.insight_type != b.insight_type:
            return False

        # Simple word overlap check
        words_a = set(a.content.lower().split())
        words_b = set(b.content.lower().split())

        if not words_a or not words_b:
            return False

        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
        return overlap > 0.6

    def _prune_insights(self) -> None:
        """Remove old or ineffective insights."""
        # Remove expired insights
        self._insights = [
            i
            for i in self._insights
            if (self._current_iteration - i.source_iteration) < self.insight_ttl_iterations
        ]

        # Remove ineffective insights
        self._insights = [
            i for i in self._insights if i.effective is not False or i.applied_count < 3
        ]

        # Limit total count
        if len(self._insights) > self.max_insights:
            # Keep highest priority
            priority_order = {
                InsightPriority.CRITICAL: 0,
                InsightPriority.HIGH: 1,
                InsightPriority.MEDIUM: 2,
                InsightPriority.LOW: 3,
            }
            self._insights = sorted(
                self._insights,
                key=lambda i: priority_order.get(i.priority, 99),
            )[: self.max_insights]

    def get_guidance(
        self,
        insight_types: list[InsightType] | None = None,
        max_items: int = 10,
    ) -> GuidanceSet:
        """
        Get guidance based on accumulated insights.

        Args:
            insight_types: Filter to specific types (None = all)
            max_items: Maximum items per category

        Returns:
            GuidanceSet ready for prompt injection
        """
        guidance = GuidanceSet()

        insights = self._insights
        if insight_types:
            insights = [i for i in insights if i.insight_type in insight_types]

        # Sort by priority
        priority_order = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3,
        }
        insights = sorted(
            insights,
            key=lambda i: priority_order.get(i.priority, 99),
        )

        for insight in insights[:max_items]:
            # Mark as applied
            insight.applied_count += 1

            # Critical insights become constraints
            if insight.priority == InsightPriority.CRITICAL:
                guidance.constraints.append(insight.content)
            else:
                guidance.recommendations.append(insight.content)

            # Collect avoid/prefer patterns
            guidance.avoid.update(insight.avoid_patterns)
            guidance.prefer.update(insight.prefer_patterns)

            # Add action guidance as context
            if insight.action_guidance:
                guidance.context.append(insight.action_guidance)

        return guidance

    def get_guidance_for_phase(self, phase: str) -> GuidanceSet:
        """
        Get guidance specific to a cognitive phase.

        Args:
            phase: Phase name (think, plan, act, etc.)

        Returns:
            GuidanceSet tailored for the phase
        """
        phase_insights = {
            "perceive": [InsightType.CONTEXT],
            "plan": [InsightType.PLANNING, InsightType.APPROACH],
            "think": [InsightType.APPROACH, InsightType.TOOL_USE],
            "validate": [InsightType.CAPABILITY],
            "act": [InsightType.TOOL_USE, InsightType.CAPABILITY],
            "observe": [InsightType.CONTEXT, InsightType.PROGRESS],
            "reflect": [InsightType.LOOP, InsightType.PROGRESS],
            "update": [InsightType.PROGRESS],
        }

        relevant_types = phase_insights.get(phase.lower())
        return self.get_guidance(insight_types=relevant_types)

    def mark_effective(
        self,
        insight: ReflectionInsight,
        effective: bool,
    ) -> None:
        """
        Mark whether an insight was effective.

        Args:
            insight: The insight to mark
            effective: Whether it led to behavior change
        """
        insight.effective = effective
        if not effective:
            logger.debug(f"Insight marked as ineffective: {insight.content[:50]}")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about accumulated insights."""
        if not self._insights:
            return {
                "total_insights": 0,
                "by_type": {},
                "by_priority": {},
                "effective_rate": 0.0,
            }

        by_type = {}
        by_priority = {}
        effective_count = 0
        evaluated_count = 0

        for insight in self._insights:
            by_type[insight.insight_type.value] = by_type.get(insight.insight_type.value, 0) + 1
            by_priority[insight.priority.value] = by_priority.get(insight.priority.value, 0) + 1

            if insight.effective is not None:
                evaluated_count += 1
                if insight.effective:
                    effective_count += 1

        return {
            "total_insights": len(self._insights),
            "by_type": by_type,
            "by_priority": by_priority,
            "effective_rate": effective_count / evaluated_count if evaluated_count > 0 else 0.0,
            "current_iteration": self._current_iteration,
        }

    def clear(self) -> None:
        """Clear all insights."""
        self._insights = []
        self._current_iteration = 0

    @property
    def insights(self) -> list[ReflectionInsight]:
        """Get all current insights."""
        return self._insights.copy()


__all__ = [
    # Enums
    "InsightType",
    "InsightPriority",
    # Data models
    "ReflectionInsight",
    "GuidanceSet",
    # Extractor
    "InsightExtractor",
    # Bridge
    "ReflectionBridge",
]
