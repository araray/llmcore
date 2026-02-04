# src/llmcore/agents/routing/model_router.py
"""
Model Router for Intelligent Model Selection.

Routes tasks to optimal models based on complexity, capabilities, and cost.
Implements tier-based routing with fallback chains for resilience.

Key Features:
    - Three-tier model classification (FAST, BALANCED, CAPABLE)
    - Capability-aware routing (tools, vision, context size)
    - Cost optimization (35-85% savings on simple tasks)
    - Fallback chains for availability issues
    - Rate limit awareness

Research:
    - Not all tasks need GPT-4
    - Simple tasks: use GPT-4-mini/Haiku (0.05x cost)
    - Complex reasoning: use Opus/GPT-4 (1.0x cost)

Usage:
    from llmcore.agents.routing import ModelRouter, ModelTier
    from llmcore.agents.cognitive import GoalClassifier

    router = ModelRouter()
    classifier = GoalClassifier()

    classification = await classifier.classify(goal)
    selection = router.select_model(
        classification=classification,
        required_capabilities=["tools"]
    )

    print(f"Selected: {selection.model} ({selection.tier.value})")

"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from collections.abc import Callable

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


if TYPE_CHECKING:
    from llmcore.agents.cognitive.goal_classifier import GoalClassification, GoalComplexity

logger = logging.getLogger(__name__)


# =============================================================================
# Model Tiers
# =============================================================================


class ModelTier(str, Enum):
    """Model performance/cost tiers."""

    FAST = "fast"  # Cheapest, fastest, least capable
    BALANCED = "balanced"  # Good balance of cost and capability
    CAPABLE = "capable"  # Most capable, highest cost


class ModelCapability(str, Enum):
    """Model capabilities for routing decisions."""

    NATIVE_TOOLS = "native_tools"  # Function calling support
    VISION = "vision"  # Image input support
    LONG_CONTEXT = "long_context"  # >100K context window
    CODE_EXECUTION = "code_execution"  # Code interpreter
    STRUCTURED_OUTPUT = "structured_output"  # JSON mode
    STREAMING = "streaming"  # Streaming responses
    REASONING = "reasoning"  # Extended thinking


@dataclass
class ModelInfo:
    """Information about a specific model."""

    id: str
    tier: ModelTier
    provider: str
    capabilities: set[ModelCapability]
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_output_tokens: int = 4096
    available: bool = True
    rate_limit_rpm: int = 60

    @property
    def cost_multiplier(self) -> float:
        """Relative cost compared to capable tier baseline."""
        tier_multipliers = {
            ModelTier.FAST: 0.05,
            ModelTier.BALANCED: 0.3,
            ModelTier.CAPABLE: 1.0,
        }
        return tier_multipliers.get(self.tier, 1.0)


@dataclass
class TierConfig:
    """Configuration for a model tier."""

    models: list[str]  # Model IDs in preference order
    cost_multiplier: float
    description: str


# =============================================================================
# Default Model Registry
# =============================================================================


# Default model configurations
DEFAULT_MODELS: dict[str, ModelInfo] = {
    # OpenAI - Fast Tier
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        tier=ModelTier.FAST,
        provider="openai",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.STREAMING,
        },
        context_window=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        max_output_tokens=16384,
    ),
    "gpt-3.5-turbo": ModelInfo(
        id="gpt-3.5-turbo",
        tier=ModelTier.FAST,
        provider="openai",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.STREAMING,
        },
        context_window=16384,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        max_output_tokens=4096,
    ),
    # OpenAI - Balanced Tier
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        tier=ModelTier.BALANCED,
        provider="openai",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.STREAMING,
        },
        context_window=128000,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        max_output_tokens=16384,
    ),
    # OpenAI - Capable Tier
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        tier=ModelTier.CAPABLE,
        provider="openai",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.STREAMING,
        },
        context_window=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        max_output_tokens=4096,
    ),
    "o1-preview": ModelInfo(
        id="o1-preview",
        tier=ModelTier.CAPABLE,
        provider="openai",
        capabilities={
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
        },
        context_window=128000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.06,
        max_output_tokens=32768,
    ),
    # Anthropic - Fast Tier
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        tier=ModelTier.FAST,
        provider="anthropic",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
        },
        context_window=200000,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        max_output_tokens=4096,
    ),
    # Anthropic - Balanced Tier
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        tier=ModelTier.BALANCED,
        provider="anthropic",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
            ModelCapability.REASONING,
        },
        context_window=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_output_tokens=8192,
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        tier=ModelTier.BALANCED,
        provider="anthropic",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
            ModelCapability.CODE_EXECUTION,
        },
        context_window=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_output_tokens=8192,
    ),
    # Anthropic - Capable Tier
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        tier=ModelTier.CAPABLE,
        provider="anthropic",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
            ModelCapability.REASONING,
            ModelCapability.CODE_EXECUTION,
        },
        context_window=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        max_output_tokens=32768,
    ),
    # Google - Balanced Tier
    "gemini-1.5-pro": ModelInfo(
        id="gemini-1.5-pro",
        tier=ModelTier.BALANCED,
        provider="google",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        },
        context_window=2000000,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        max_output_tokens=8192,
    ),
    "gemini-1.5-flash": ModelInfo(
        id="gemini-1.5-flash",
        tier=ModelTier.FAST,
        provider="google",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        },
        context_window=1000000,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
        max_output_tokens=8192,
    ),
    # DeepSeek - Fast/Balanced
    "deepseek-chat": ModelInfo(
        id="deepseek-chat",
        tier=ModelTier.FAST,
        provider="deepseek",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.STREAMING,
        },
        context_window=64000,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        max_output_tokens=8192,
    ),
    "deepseek-reasoner": ModelInfo(
        id="deepseek-reasoner",
        tier=ModelTier.BALANCED,
        provider="deepseek",
        capabilities={
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        context_window=64000,
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        max_output_tokens=8192,
    ),
    # Local models via Ollama (no cost, variable capability)
    "llama3.3:70b": ModelInfo(
        id="llama3.3:70b",
        tier=ModelTier.BALANCED,
        provider="ollama",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.STREAMING,
        },
        context_window=128000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_output_tokens=8192,
    ),
    "qwen2.5:32b": ModelInfo(
        id="qwen2.5:32b",
        tier=ModelTier.FAST,
        provider="ollama",
        capabilities={
            ModelCapability.NATIVE_TOOLS,
            ModelCapability.STREAMING,
        },
        context_window=32000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_output_tokens=8192,
    ),
}


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ModelSelection:
    """Result of model selection."""

    model: str
    tier: ModelTier
    provider: str
    reason: str
    alternatives: list[str] = field(default_factory=list)
    estimated_cost_multiplier: float = 1.0
    capabilities_matched: set[ModelCapability] = field(default_factory=set)
    capabilities_missing: set[ModelCapability] = field(default_factory=set)


@dataclass
class RoutingContext:
    """Context for routing decisions."""

    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    required_capabilities: set[ModelCapability] = field(default_factory=set)
    preferred_providers: list[str] = field(default_factory=list)
    excluded_models: set[str] = field(default_factory=set)
    max_cost_per_request: float | None = None
    prefer_local: bool = False


# =============================================================================
# Model Router Implementation
# =============================================================================


class ModelRouter:
    """
    Routes tasks to optimal models based on complexity and requirements.

    Implements:
    - Tier-based routing (FAST → BALANCED → CAPABLE)
    - Capability-aware model selection
    - Cost optimization
    - Fallback chains for availability

    Args:
        models: Custom model registry (uses defaults if None)
        default_provider: Preferred provider for tie-breaking
        cost_optimization: Whether to prefer cheaper models
    """

    def __init__(
        self,
        models: dict[str, ModelInfo] | None = None,
        default_provider: str = "anthropic",
        cost_optimization: bool = True,
    ):
        self._models = models or dict(DEFAULT_MODELS)
        self.default_provider = default_provider
        self.cost_optimization = cost_optimization

        # Build indices
        self._by_tier: dict[ModelTier, list[str]] = {}
        self._by_provider: dict[str, list[str]] = {}
        self._by_capability: dict[ModelCapability, list[str]] = {}
        self._rebuild_indices()

        # Track rate limits and availability
        self._rate_limit_tracker: dict[str, tuple[int, float]] = {}  # model -> (count, reset_time)
        self._availability: dict[str, bool] = {}

    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices."""
        self._by_tier = {tier: [] for tier in ModelTier}
        self._by_provider = {}
        self._by_capability = {cap: [] for cap in ModelCapability}

        for model_id, info in self._models.items():
            self._by_tier[info.tier].append(model_id)

            if info.provider not in self._by_provider:
                self._by_provider[info.provider] = []
            self._by_provider[info.provider].append(model_id)

            for cap in info.capabilities:
                self._by_capability[cap].append(model_id)

    def select_model(
        self,
        classification: GoalClassification | None = None,
        context: RoutingContext | None = None,
        complexity: GoalComplexity | None = None,
    ) -> ModelSelection:
        """
        Select optimal model for a task.

        Args:
            classification: Goal classification from GoalClassifier
            context: Additional routing context
            complexity: Direct complexity override

        Returns:
            ModelSelection with chosen model and alternatives
        """
        context = context or RoutingContext()

        # Determine minimum tier from complexity
        if complexity:
            min_tier = self._complexity_to_tier(complexity)
        elif classification:
            min_tier = self._complexity_to_tier(classification.complexity)
        else:
            min_tier = ModelTier.BALANCED  # Default to balanced

        # Get candidate models
        candidates = self._get_candidates(min_tier, context)

        if not candidates:
            # No suitable models - fall back to any available
            candidates = list(self._models.keys())
            logger.warning("No suitable models found, using fallback selection")

        # Filter by capabilities
        if context.required_capabilities:
            candidates = self._filter_by_capabilities(candidates, context.required_capabilities)

        # Filter by context size
        total_tokens = context.estimated_input_tokens + context.estimated_output_tokens
        if total_tokens > 0:
            candidates = [m for m in candidates if self._models[m].context_window >= total_tokens]

        # Filter by cost
        if context.max_cost_per_request is not None:
            candidates = self._filter_by_cost(candidates, context, context.max_cost_per_request)

        # Filter by availability
        candidates = [m for m in candidates if self._is_available(m)]

        # Filter by excluded models
        candidates = [m for m in candidates if m not in context.excluded_models]

        if not candidates:
            # Emergency fallback
            candidates = [list(self._models.keys())[0]]
            logger.warning(f"Using emergency fallback model: {candidates[0]}")

        # Rank and select
        selected = self._rank_and_select(candidates, context)

        # Get model info
        model_info = self._models[selected]

        # Determine capabilities match
        caps_matched = model_info.capabilities & context.required_capabilities
        caps_missing = context.required_capabilities - model_info.capabilities

        return ModelSelection(
            model=selected,
            tier=model_info.tier,
            provider=model_info.provider,
            reason=self._build_reason(model_info, min_tier, context),
            alternatives=candidates[:5] if len(candidates) > 1 else [],
            estimated_cost_multiplier=model_info.cost_multiplier,
            capabilities_matched=caps_matched,
            capabilities_missing=caps_missing,
        )

    def _complexity_to_tier(self, complexity: GoalComplexity) -> ModelTier:
        """Map goal complexity to minimum model tier."""
        # Import here to avoid circular imports
        try:
            from llmcore.agents.cognitive.goal_classifier import GoalComplexity

            mapping = {
                GoalComplexity.TRIVIAL: ModelTier.FAST,
                GoalComplexity.SIMPLE: ModelTier.FAST,
                GoalComplexity.MODERATE: ModelTier.BALANCED,
                GoalComplexity.COMPLEX: ModelTier.CAPABLE,
                GoalComplexity.AMBIGUOUS: ModelTier.BALANCED,
            }
            return mapping.get(complexity, ModelTier.BALANCED)
        except ImportError:
            return ModelTier.BALANCED

    def _get_candidates(
        self,
        min_tier: ModelTier,
        context: RoutingContext,
    ) -> list[str]:
        """Get candidate models for minimum tier."""
        tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.CAPABLE]
        min_index = tier_order.index(min_tier)

        candidates = []
        for tier in tier_order[min_index:]:
            candidates.extend(self._by_tier[tier])

        # Prefer local if requested
        if context.prefer_local:
            local = [m for m in candidates if self._models[m].provider == "ollama"]
            if local:
                candidates = local + [m for m in candidates if m not in local]

        # Prefer specific providers
        if context.preferred_providers:
            preferred = [
                m for m in candidates if self._models[m].provider in context.preferred_providers
            ]
            if preferred:
                candidates = preferred + [m for m in candidates if m not in preferred]

        return candidates

    def _filter_by_capabilities(
        self,
        candidates: list[str],
        required: set[ModelCapability],
    ) -> list[str]:
        """Filter models by required capabilities."""
        result = []
        for model_id in candidates:
            model_caps = self._models[model_id].capabilities
            if required <= model_caps:  # All required caps present
                result.append(model_id)
        return result if result else candidates  # Fall back to original if none match

    def _filter_by_cost(
        self,
        candidates: list[str],
        context: RoutingContext,
        max_cost: float,
    ) -> list[str]:
        """Filter models by estimated cost."""
        result = []
        for model_id in candidates:
            info = self._models[model_id]
            estimated_cost = (context.estimated_input_tokens / 1000) * info.cost_per_1k_input + (
                context.estimated_output_tokens / 1000
            ) * info.cost_per_1k_output
            if estimated_cost <= max_cost:
                result.append(model_id)
        return result if result else candidates

    def _is_available(self, model_id: str) -> bool:
        """Check if model is available."""
        # Check explicit availability flag
        if model_id in self._availability:
            return self._availability[model_id]

        # Check rate limits
        if model_id in self._rate_limit_tracker:
            count, reset_time = self._rate_limit_tracker[model_id]
            info = self._models.get(model_id)
            if info and count >= info.rate_limit_rpm and time.time() < reset_time:
                return False

        return True

    def _rank_and_select(
        self,
        candidates: list[str],
        context: RoutingContext,
    ) -> str:
        """Rank candidates and select best."""
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate
        scores: list[tuple[float, str]] = []

        for model_id in candidates:
            info = self._models[model_id]
            score = 0.0

            # Cost factor (lower is better)
            if self.cost_optimization:
                score -= info.cost_multiplier * 10

            # Provider preference
            if info.provider == self.default_provider:
                score += 5
            if context.preferred_providers and info.provider in context.preferred_providers:
                score += 3

            # Context window headroom
            total_tokens = context.estimated_input_tokens + context.estimated_output_tokens
            if total_tokens > 0 and info.context_window > 0:
                headroom_ratio = info.context_window / total_tokens
                if headroom_ratio > 2:
                    score += 2  # Good headroom

            # Capability match bonus
            if context.required_capabilities:
                match_ratio = len(info.capabilities & context.required_capabilities) / len(
                    context.required_capabilities
                )
                score += match_ratio * 5

            # Local model bonus if preferred
            if context.prefer_local and info.provider == "ollama":
                score += 10

            scores.append((score, model_id))

        # Sort by score (descending) and return best
        scores.sort(key=lambda x: -x[0])
        return scores[0][1]

    def _build_reason(
        self,
        model_info: ModelInfo,
        min_tier: ModelTier,
        context: RoutingContext,
    ) -> str:
        """Build human-readable reason for selection."""
        reasons = []

        tier_names = {
            ModelTier.FAST: "fast/cheap",
            ModelTier.BALANCED: "balanced",
            ModelTier.CAPABLE: "high-capability",
        }

        reasons.append(f"Tier: {tier_names.get(model_info.tier, 'unknown')}")

        if context.required_capabilities:
            matched = len(model_info.capabilities & context.required_capabilities)
            total = len(context.required_capabilities)
            reasons.append(f"Capabilities: {matched}/{total} matched")

        if self.cost_optimization:
            reasons.append(f"Cost: {model_info.cost_multiplier:.0%} of baseline")

        return "; ".join(reasons)

    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------

    def register_model(self, model_id: str, info: ModelInfo) -> None:
        """Register a new model."""
        self._models[model_id] = info
        self._rebuild_indices()
        logger.info(f"Registered model: {model_id} ({info.tier.value})")

    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        if model_id in self._models:
            del self._models[model_id]
            self._rebuild_indices()
            return True
        return False

    def set_availability(self, model_id: str, available: bool) -> None:
        """Set model availability."""
        self._availability[model_id] = available
        logger.debug(f"Model {model_id} availability: {available}")

    def record_rate_limit(self, model_id: str) -> None:
        """Record a rate limit hit."""
        if model_id not in self._rate_limit_tracker:
            self._rate_limit_tracker[model_id] = (0, time.time() + 60)

        count, reset_time = self._rate_limit_tracker[model_id]
        if time.time() >= reset_time:
            # Reset counter
            self._rate_limit_tracker[model_id] = (1, time.time() + 60)
        else:
            self._rate_limit_tracker[model_id] = (count + 1, reset_time)

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a model."""
        return self._models.get(model_id)

    def list_models(
        self,
        tier: ModelTier | None = None,
        provider: str | None = None,
        capability: ModelCapability | None = None,
    ) -> list[str]:
        """List models with optional filters."""
        models = list(self._models.keys())

        if tier:
            models = [m for m in models if self._models[m].tier == tier]

        if provider:
            models = [m for m in models if self._models[m].provider == provider]

        if capability:
            models = [m for m in models if capability in self._models[m].capabilities]

        return models

    def get_statistics(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "total_models": len(self._models),
            "by_tier": {tier.value: len(models) for tier, models in self._by_tier.items()},
            "by_provider": {
                provider: len(models) for provider, models in self._by_provider.items()
            },
            "rate_limited": len(
                [m for m, (count, reset) in self._rate_limit_tracker.items() if time.time() < reset]
            ),
            "unavailable": len([m for m, avail in self._availability.items() if not avail]),
        }


# =============================================================================
# Fallback Chain
# =============================================================================


class ModelFallbackChain:
    """
    Executes requests with automatic fallback to alternative models.

    Usage:
        chain = ModelFallbackChain(router, llm_providers)
        result = await chain.execute_with_fallback(request)
    """

    def __init__(
        self,
        router: ModelRouter,
        max_retries: int = 2,
        retry_delay_base: float = 1.0,
    ):
        self.router = router
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base

        self._stats = {
            "total_requests": 0,
            "successful_primary": 0,
            "successful_fallback": 0,
            "all_failed": 0,
        }

    async def execute_with_fallback(
        self,
        execute_fn: Callable[[str], Any],
        selection: ModelSelection,
    ) -> tuple[Any, str]:
        """
        Execute with fallback to alternatives on failure.

        Args:
            execute_fn: Async function that takes model_id and executes request
            selection: Model selection with alternatives

        Returns:
            Tuple of (result, model_id used)
        """
        self._stats["total_requests"] += 1

        # Try primary model
        models_to_try = [selection.model] + selection.alternatives

        for i, model_id in enumerate(models_to_try):
            for attempt in range(self.max_retries):
                try:
                    result = await execute_fn(model_id)

                    if i == 0:
                        self._stats["successful_primary"] += 1
                    else:
                        self._stats["successful_fallback"] += 1

                    return result, model_id

                except Exception as e:
                    error_type = type(e).__name__
                    logger.warning(f"Model {model_id} attempt {attempt + 1} failed: {error_type}")

                    # Check if rate limited
                    if "rate" in str(e).lower() or "429" in str(e):
                        self.router.record_rate_limit(model_id)
                        break  # Try next model

                    # Check if model unavailable
                    if "unavailable" in str(e).lower() or "503" in str(e):
                        self.router.set_availability(model_id, False)
                        break  # Try next model

                    # Retry with delay
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay_base * (2**attempt)
                        await asyncio.sleep(delay)

        self._stats["all_failed"] += 1
        raise RuntimeError(f"All models failed: {models_to_try}")

    def get_statistics(self) -> dict[str, Any]:
        """Get fallback chain statistics."""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "primary_success_rate": (
                self._stats["successful_primary"] / total if total > 0 else 0.0
            ),
            "fallback_rate": (self._stats["successful_fallback"] / total if total > 0 else 0.0),
            "failure_rate": (self._stats["all_failed"] / total if total > 0 else 0.0),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def get_default_router() -> ModelRouter:
    """Get a default model router instance."""
    return ModelRouter()


def select_model_for_complexity(
    complexity: GoalComplexity,
    required_capabilities: list[str] | None = None,
    preferred_provider: str | None = None,
) -> ModelSelection:
    """
    Convenience function for quick model selection.

    Args:
        complexity: Task complexity level
        required_capabilities: List of required capability names
        preferred_provider: Preferred provider name

    Returns:
        ModelSelection
    """
    router = get_default_router()

    context = RoutingContext()
    if required_capabilities:
        context.required_capabilities = {
            ModelCapability(c)
            for c in required_capabilities
            if c in [cap.value for cap in ModelCapability]
        }
    if preferred_provider:
        context.preferred_providers = [preferred_provider]

    return router.select_model(complexity=complexity, context=context)


__all__ = [
    # Enums
    "ModelTier",
    "ModelCapability",
    # Data models
    "ModelInfo",
    "TierConfig",
    "ModelSelection",
    "RoutingContext",
    # Main classes
    "ModelRouter",
    "ModelFallbackChain",
    # Registry
    "DEFAULT_MODELS",
    # Convenience
    "get_default_router",
    "select_model_for_complexity",
]
