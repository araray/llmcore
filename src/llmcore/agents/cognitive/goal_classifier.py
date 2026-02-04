# src/llmcore/agents/cognitive/goal_classifier.py
"""
Goal Complexity Classifier for LLMCore Agent System.

This module classifies user goals by complexity to enable fast-path routing:
- TRIVIAL: Direct response, no tools needed (greetings, thanks)
- SIMPLE: Single tool call (read file, search)
- MODERATE: Multiple coordinated steps
- COMPLEX: Full cognitive cycle required
- AMBIGUOUS: Needs clarification from user

Usage:
    from llmcore.agents.cognitive.goal_classifier import GoalClassifier, GoalComplexity

    classifier = GoalClassifier()
    result = classifier.classify("hello")

    if result.complexity == GoalComplexity.TRIVIAL:
        # Fast path - direct LLM response
        pass

Integration Point:
    Call at start of CognitiveCycle.run() to route appropriately.

Author: llmcore team
Date: 2026-01-21
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

# Use Pydantic if available, otherwise use dataclasses
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


if TYPE_CHECKING:
    from llmcore.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================


class GoalComplexity(str, Enum):
    """Classification of goal complexity levels."""

    TRIVIAL = "trivial"  # Direct response, no tools/actions needed
    SIMPLE = "simple"  # Single tool call or straightforward action
    MODERATE = "moderate"  # Multiple coordinated steps
    COMPLEX = "complex"  # Full cognitive cycle with planning
    AMBIGUOUS = "ambiguous"  # Needs clarification before proceeding


class GoalIntent(str, Enum):
    """The primary intent type of the goal."""

    GREETING = "greeting"  # Social niceties (hi, hello, thanks)
    FAREWELL = "farewell"  # Goodbye messages
    QUESTION = "question"  # Factual questions
    TASK = "task"  # Action-oriented requests
    CREATIVE = "creative"  # Creative writing, generation
    ANALYSIS = "analysis"  # Research, analysis tasks
    CLARIFICATION = "clarification"  # User asking for clarification
    META = "meta"  # Questions about the agent itself
    UNKNOWN = "unknown"


class ExecutionStrategy(str, Enum):
    """Recommended execution strategy."""

    DIRECT = "direct"  # Single LLM call, no tools
    REACT = "react"  # ReAct reasoning loop
    REFLEXION = "reflexion"  # ReAct with reflection/retry
    HIERARCHICAL = "hierarchical"  # Decompose into subgoals


class ModelTier(str, Enum):
    """Recommended model capability tier."""

    FAST = "fast"  # GPT-4o-mini, Claude Haiku
    BALANCED = "balanced"  # GPT-4o, Claude Sonnet
    CAPABLE = "capable"  # GPT-4-turbo, Claude Opus


if PYDANTIC_AVAILABLE:

    class GoalClassification(BaseModel):
        """Complete goal classification result."""

        complexity: GoalComplexity
        intent: GoalIntent = GoalIntent.UNKNOWN
        confidence: float = Field(default=0.5, ge=0.0, le=1.0)

        # Execution recommendations
        suggested_strategy: ExecutionStrategy = ExecutionStrategy.REACT
        suggested_model_tier: ModelTier = ModelTier.BALANCED
        max_iterations: int = Field(default=10, ge=1, le=100)
        requires_tools: bool = True

        # Clarification
        clarification_needed: bool = False
        clarification_questions: list[str] = Field(default_factory=list)

        # Debug info
        matched_pattern: str | None = None
        classification_method: str = "heuristic"
else:

    @dataclass
    class GoalClassification:
        """Complete goal classification result (dataclass version)."""

        complexity: GoalComplexity
        intent: GoalIntent = GoalIntent.UNKNOWN
        confidence: float = 0.5

        suggested_strategy: ExecutionStrategy = ExecutionStrategy.REACT
        suggested_model_tier: ModelTier = ModelTier.BALANCED
        max_iterations: int = 10
        requires_tools: bool = True

        clarification_needed: bool = False
        clarification_questions: list[str] = field(default_factory=list)

        matched_pattern: str | None = None
        classification_method: str = "heuristic"


# =============================================================================
# Pattern Definitions
# =============================================================================

# Trivial patterns - greetings, thanks, simple social
TRIVIAL_PATTERNS: list[tuple[str, GoalIntent]] = [
    # Greetings
    (r"^(hi|hello|hey|howdy|greetings)[\s!.,?]*$", GoalIntent.GREETING),
    (r"^good\s*(morning|afternoon|evening|day)[\s!.,?]*$", GoalIntent.GREETING),
    (r"^(what'?s?\s*up|sup|yo)[\s!.,?]*$", GoalIntent.GREETING),
    # Farewells
    (r"^(bye|goodbye|later|see\s*ya|ciao|farewell)[\s!.,?]*$", GoalIntent.FAREWELL),
    (r"^(thanks?(\s*you)?|thank\s*you(\s*so\s*much)?|thx|ty)[\s!.,?]*$", GoalIntent.FAREWELL),
    # Meta/simple
    (r"^(ok|okay|sure|got\s*it|alright|understood)[\s!.,?]*$", GoalIntent.META),
    (r"^(yes|no|yeah|nope|yep|nah)[\s!.,?]*$", GoalIntent.META),
]

# Simple patterns - single action requests
SIMPLE_PATTERNS: list[tuple[str, GoalIntent]] = [
    # File operations
    (r"^(read|show|display|cat|view|open)\s+(the\s+)?(file|document)\s+\S+", GoalIntent.TASK),
    (r"^(list|ls|show)\s+(the\s+)?(files?|directory|folder|contents)", GoalIntent.TASK),
    (r"^what('?s|\s+is)\s+(in|inside)\s+(the\s+)?(file|folder|directory)", GoalIntent.QUESTION),
    # Simple questions
    (r"^what\s+time\s+is\s+it", GoalIntent.QUESTION),
    (r"^what('?s|\s+is)\s+the\s+(date|day)", GoalIntent.QUESTION),
    (r"^what('?s|\s+is)\s+\d+\s*[+\-*/]\s*\d+", GoalIntent.QUESTION),  # Simple math
    # Simple search
    (r"^(find|search\s+for|look\s+for|locate)\s+\S+", GoalIntent.TASK),
    # Definition questions
    (r"^what\s+is\s+(a|an|the)\s+\w+\?*$", GoalIntent.QUESTION),
    (r"^(define|meaning\s+of)\s+\w+", GoalIntent.QUESTION),
]

# Moderate patterns - multi-step but bounded
MODERATE_PATTERNS: list[tuple[str, GoalIntent]] = [
    # File operations with processing
    (r"(read|load).+(and|then).+(process|analyze|parse|extract)", GoalIntent.TASK),
    (r"(find|search).+(and|then).+(show|display|list)", GoalIntent.TASK),
    # Comparison/analysis
    (r"compare\s+.+\s+(to|with|and)\s+", GoalIntent.ANALYSIS),
    (r"(summarize|summary\s+of)\s+", GoalIntent.ANALYSIS),
    # Debugging
    (r"(fix|debug|find\s+the\s+bug|what'?s?\s+wrong)", GoalIntent.TASK),
]

# Complex patterns - require planning and multiple phases
COMPLEX_PATTERNS: list[tuple[str, GoalIntent]] = [
    # Multi-step analysis
    (
        r"(analyze|research|investigate).+(create|write|generate).+(report|document|summary)",
        GoalIntent.ANALYSIS,
    ),
    # Building/creating
    (
        r"(build|create|develop|implement|make)\s+(a|an|the)\s+\w+.+(application|app|system|tool|script|program)",
        GoalIntent.TASK,
    ),
    # Comprehensive tasks
    (
        r"(comprehensive|thorough|complete|full)\s+(analysis|review|audit|assessment)",
        GoalIntent.ANALYSIS,
    ),
    # Multi-phase explicit
    (r"(first|step\s*1).+(then|next|step\s*2)", GoalIntent.TASK),
]

# Ambiguous patterns - need clarification
AMBIGUOUS_PATTERNS: list[tuple[str, list[str]]] = [
    # Vague references
    (
        r"^(do|fix|check|handle)\s+(it|that|this|the\s+thing)[\s!.,?]*$",
        [
            "What specifically would you like me to do?",
            "Could you provide more details about what you're referring to?",
        ],
    ),
    # Missing context
    (
        r"^(continue|go\s+on|keep\s+going)[\s!.,?]*$",
        ["Continue with what task?", "What would you like me to continue doing?"],
    ),
    # Underspecified
    (
        r"^help[\s!.,?]*$",
        ["What kind of help do you need?", "Is there a specific task I can assist you with?"],
    ),
]


# =============================================================================
# Goal Classifier Implementation
# =============================================================================


class GoalClassifier:
    """
    Classify goal complexity using heuristics and optional LLM fallback.

    The classifier uses a two-stage approach:
    1. Fast heuristic matching using regex patterns
    2. LLM-based classification for uncertain cases (optional)

    Args:
        use_llm_fallback: Whether to use LLM for uncertain classifications
        llm_provider: LLM provider for fallback classification
        confidence_threshold: Minimum confidence for heuristic result (default: 0.85)
    """

    def __init__(
        self,
        use_llm_fallback: bool = False,
        llm_provider: BaseLLMProvider | None = None,
        confidence_threshold: float = 0.85,
    ):
        self.use_llm_fallback = use_llm_fallback
        self.llm_provider = llm_provider
        self.confidence_threshold = confidence_threshold

        # Compile patterns for performance
        self._trivial_patterns = [
            (re.compile(p, re.IGNORECASE), intent) for p, intent in TRIVIAL_PATTERNS
        ]
        self._simple_patterns = [
            (re.compile(p, re.IGNORECASE), intent) for p, intent in SIMPLE_PATTERNS
        ]
        self._moderate_patterns = [
            (re.compile(p, re.IGNORECASE), intent) for p, intent in MODERATE_PATTERNS
        ]
        self._complex_patterns = [
            (re.compile(p, re.IGNORECASE), intent) for p, intent in COMPLEX_PATTERNS
        ]
        self._ambiguous_patterns = [
            (re.compile(p, re.IGNORECASE), questions) for p, questions in AMBIGUOUS_PATTERNS
        ]

    def classify(self, goal: str) -> GoalClassification:
        """
        Classify goal complexity (sync version).

        Args:
            goal: The user's goal/request string

        Returns:
            GoalClassification with complexity, intent, and recommendations
        """
        # Normalize input
        goal_clean = goal.strip()

        if not goal_clean:
            return GoalClassification(
                complexity=GoalComplexity.AMBIGUOUS,
                intent=GoalIntent.UNKNOWN,
                confidence=1.0,
                clarification_needed=True,
                clarification_questions=[
                    "I didn't receive any input. What would you like me to help you with?"
                ],
                classification_method="empty_input",
            )

        # Try heuristic classification
        result = self._classify_heuristic(goal_clean)

        # If confidence is high enough, return heuristic result
        if result.confidence >= self.confidence_threshold:
            logger.debug(
                f"Goal classified by heuristic: {result.complexity.value} (conf={result.confidence:.2f})"
            )
            return result

        # If LLM fallback is enabled and available, use it
        if self.use_llm_fallback and self.llm_provider:
            logger.debug(f"Heuristic confidence low ({result.confidence:.2f}), using LLM fallback")
            # Note: In production, this would be async
            # For now, return heuristic with lower confidence
            result.classification_method = "heuristic_low_confidence"
            return result

        # Return heuristic result even if low confidence
        logger.debug(
            f"Goal classified (low confidence): {result.complexity.value} (conf={result.confidence:.2f})"
        )
        return result

    async def classify_async(self, goal: str) -> GoalClassification:
        """
        Classify goal complexity (async version with LLM fallback).

        Args:
            goal: The user's goal/request string

        Returns:
            GoalClassification with complexity, intent, and recommendations
        """
        # Get heuristic result first
        result = self.classify(goal)

        # If confidence is sufficient or no LLM available, return
        if result.confidence >= self.confidence_threshold or not self.llm_provider:
            return result

        # Use LLM for classification
        try:
            llm_result = await self._classify_with_llm(goal)
            llm_result.classification_method = "llm"
            return llm_result
        except Exception as e:
            logger.warning(f"LLM classification failed, using heuristic: {e}")
            result.classification_method = "heuristic_llm_failed"
            return result

    def _classify_heuristic(self, goal: str) -> GoalClassification:
        """
        Classify using heuristic pattern matching.

        The order of checking is important:
        1. Ambiguous (needs clarification)
        2. Trivial (fast path)
        3. Simple (single action)
        4. Complex (full cycle)
        5. Moderate (default for medium-length goals)
        """

        # Check for ambiguous patterns first
        for pattern, questions in self._ambiguous_patterns:
            if pattern.match(goal):
                return GoalClassification(
                    complexity=GoalComplexity.AMBIGUOUS,
                    intent=GoalIntent.UNKNOWN,
                    confidence=0.9,
                    clarification_needed=True,
                    clarification_questions=questions,
                    matched_pattern=pattern.pattern,
                    suggested_strategy=ExecutionStrategy.DIRECT,
                    requires_tools=False,
                    max_iterations=1,
                )

        # Check trivial patterns
        for pattern, intent in self._trivial_patterns:
            if pattern.match(goal):
                return GoalClassification(
                    complexity=GoalComplexity.TRIVIAL,
                    intent=intent,
                    confidence=0.95,
                    suggested_strategy=ExecutionStrategy.DIRECT,
                    suggested_model_tier=ModelTier.FAST,
                    max_iterations=1,
                    requires_tools=False,
                    matched_pattern=pattern.pattern,
                )

        # Check simple patterns
        for pattern, intent in self._simple_patterns:
            if pattern.search(goal):
                return GoalClassification(
                    complexity=GoalComplexity.SIMPLE,
                    intent=intent,
                    confidence=0.85,
                    suggested_strategy=ExecutionStrategy.REACT,
                    suggested_model_tier=ModelTier.BALANCED,
                    max_iterations=5,
                    requires_tools=True,
                    matched_pattern=pattern.pattern,
                )

        # Check complex patterns
        for pattern, intent in self._complex_patterns:
            if pattern.search(goal):
                return GoalClassification(
                    complexity=GoalComplexity.COMPLEX,
                    intent=intent,
                    confidence=0.80,
                    suggested_strategy=ExecutionStrategy.HIERARCHICAL,
                    suggested_model_tier=ModelTier.CAPABLE,
                    max_iterations=25,
                    requires_tools=True,
                    matched_pattern=pattern.pattern,
                )

        # Check moderate patterns
        for pattern, intent in self._moderate_patterns:
            if pattern.search(goal):
                return GoalClassification(
                    complexity=GoalComplexity.MODERATE,
                    intent=intent,
                    confidence=0.75,
                    suggested_strategy=ExecutionStrategy.REACT,
                    suggested_model_tier=ModelTier.BALANCED,
                    max_iterations=15,
                    requires_tools=True,
                    matched_pattern=pattern.pattern,
                )

        # Use length-based heuristics as fallback
        word_count = len(goal.split())

        if word_count <= 3:
            # Very short - likely simple or trivial
            return GoalClassification(
                complexity=GoalComplexity.SIMPLE,
                intent=GoalIntent.UNKNOWN,
                confidence=0.6,
                suggested_strategy=ExecutionStrategy.REACT,
                suggested_model_tier=ModelTier.BALANCED,
                max_iterations=5,
                requires_tools=True,
                classification_method="length_heuristic",
            )

        if word_count > 50:
            # Very long - likely complex
            return GoalClassification(
                complexity=GoalComplexity.COMPLEX,
                intent=GoalIntent.UNKNOWN,
                confidence=0.7,
                suggested_strategy=ExecutionStrategy.HIERARCHICAL,
                suggested_model_tier=ModelTier.CAPABLE,
                max_iterations=25,
                requires_tools=True,
                classification_method="length_heuristic",
            )

        # Default to moderate for medium-length goals
        return GoalClassification(
            complexity=GoalComplexity.MODERATE,
            intent=GoalIntent.UNKNOWN,
            confidence=0.5,
            suggested_strategy=ExecutionStrategy.REACT,
            suggested_model_tier=ModelTier.BALANCED,
            max_iterations=15,
            requires_tools=True,
            classification_method="default",
        )

    async def _classify_with_llm(self, goal: str) -> GoalClassification:
        """
        Use LLM for classification (for uncertain cases).

        This is called when heuristic confidence is below threshold.
        """
        if not self.llm_provider:
            raise ValueError("LLM provider not configured")

        prompt = f"""Classify the following user goal by complexity.

User Goal: "{goal}"

Respond with exactly one line in this format:
COMPLEXITY: [TRIVIAL|SIMPLE|MODERATE|COMPLEX|AMBIGUOUS]
INTENT: [GREETING|FAREWELL|QUESTION|TASK|CREATIVE|ANALYSIS|META|UNKNOWN]
CONFIDENCE: [0.0-1.0]
REQUIRES_TOOLS: [true|false]
MAX_ITERATIONS: [number 1-25]

Classification guidelines:
- TRIVIAL: Greetings, thanks, simple acknowledgments (no tools needed)
- SIMPLE: Single file read, simple search, basic question (1-3 iterations)
- MODERATE: Multi-step task, comparison, debugging (5-15 iterations)
- COMPLEX: Research, analysis with report, building something (15-25 iterations)
- AMBIGUOUS: Unclear what user wants, needs clarification

Respond only with the classification, no explanation."""

        # Call LLM
        response = await self.llm_provider.complete(
            prompt=prompt,
            max_tokens=100,
            temperature=0.0,
        )

        # Parse response
        return self._parse_llm_response(response.text)

    def _parse_llm_response(self, response: str) -> GoalClassification:
        """Parse LLM classification response."""

        # Default values
        complexity = GoalComplexity.MODERATE
        intent = GoalIntent.UNKNOWN
        confidence = 0.7
        requires_tools = True
        max_iterations = 10

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip().upper()

            if line.startswith("COMPLEXITY:"):
                value = line.split(":", 1)[1].strip()
                try:
                    complexity = GoalComplexity(value.lower())
                except ValueError:
                    pass

            elif line.startswith("INTENT:"):
                value = line.split(":", 1)[1].strip()
                try:
                    intent = GoalIntent(value.lower())
                except ValueError:
                    pass

            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith("REQUIRES_TOOLS:"):
                requires_tools = "TRUE" in line

            elif line.startswith("MAX_ITERATIONS:"):
                try:
                    max_iterations = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        # Determine strategy based on complexity
        strategy_map = {
            GoalComplexity.TRIVIAL: ExecutionStrategy.DIRECT,
            GoalComplexity.SIMPLE: ExecutionStrategy.REACT,
            GoalComplexity.MODERATE: ExecutionStrategy.REACT,
            GoalComplexity.COMPLEX: ExecutionStrategy.HIERARCHICAL,
            GoalComplexity.AMBIGUOUS: ExecutionStrategy.DIRECT,
        }

        tier_map = {
            GoalComplexity.TRIVIAL: ModelTier.FAST,
            GoalComplexity.SIMPLE: ModelTier.BALANCED,
            GoalComplexity.MODERATE: ModelTier.BALANCED,
            GoalComplexity.COMPLEX: ModelTier.CAPABLE,
            GoalComplexity.AMBIGUOUS: ModelTier.FAST,
        }

        return GoalClassification(
            complexity=complexity,
            intent=intent,
            confidence=confidence,
            suggested_strategy=strategy_map.get(complexity, ExecutionStrategy.REACT),
            suggested_model_tier=tier_map.get(complexity, ModelTier.BALANCED),
            max_iterations=max_iterations,
            requires_tools=requires_tools,
            classification_method="llm",
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def classify_goal(goal: str) -> GoalClassification:
    """
    Convenience function for quick goal classification.

    Uses default classifier with heuristics only.
    """
    classifier = GoalClassifier()
    return classifier.classify(goal)


def is_trivial_goal(goal: str) -> bool:
    """Check if a goal is trivial (can be handled with direct response)."""
    result = classify_goal(goal)
    return result.complexity == GoalComplexity.TRIVIAL


def needs_clarification(goal: str) -> tuple[bool, list[str]]:
    """Check if a goal needs clarification before proceeding."""
    result = classify_goal(goal)
    return result.clarification_needed, result.clarification_questions


# =============================================================================
# Tests (run with: python -m pytest goal_classifier.py -v)
# =============================================================================

if __name__ == "__main__":
    # Self-test
    print("Running Goal Classifier self-tests...\n")

    classifier = GoalClassifier()

    test_cases = [
        # (goal, expected_complexity, expected_trivial)
        ("hello", GoalComplexity.TRIVIAL, True),
        ("hi there!", GoalComplexity.TRIVIAL, True),
        ("good morning", GoalComplexity.TRIVIAL, True),
        ("thanks", GoalComplexity.TRIVIAL, True),
        ("bye", GoalComplexity.TRIVIAL, True),
        ("read file config.yaml", GoalComplexity.SIMPLE, False),
        ("list the files in /var/log", GoalComplexity.SIMPLE, False),
        ("what is 2+2", GoalComplexity.SIMPLE, False),
        ("find errors in the code", GoalComplexity.SIMPLE, False),
        ("analyze the logs and find the bug", GoalComplexity.MODERATE, False),
        ("compare file1 with file2", GoalComplexity.MODERATE, False),
        (
            "analyze the sales data and create a comprehensive report with visualizations",
            GoalComplexity.COMPLEX,
            False,
        ),
        ("build a web scraper application", GoalComplexity.COMPLEX, False),
        ("do it", GoalComplexity.AMBIGUOUS, False),
        ("fix that", GoalComplexity.AMBIGUOUS, False),
    ]

    passed = 0
    failed = 0

    for goal, expected_complexity, expected_trivial in test_cases:
        result = classifier.classify(goal)

        complexity_ok = result.complexity == expected_complexity
        trivial_ok = (result.complexity == GoalComplexity.TRIVIAL) == expected_trivial

        if complexity_ok and trivial_ok:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1

        print(
            f"{status} '{goal[:40]:<40}' -> {result.complexity.value:<10} (expected: {expected_complexity.value})"
        )
        if not complexity_ok:
            print(f"   Matched pattern: {result.matched_pattern}")
            print(f"   Confidence: {result.confidence:.2f}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed! ✅")
    else:
        print("Some tests failed ❌")
        exit(1)
