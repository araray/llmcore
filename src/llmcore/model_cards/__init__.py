# src/llmcore/model_cards/__init__.py
# llmcore/model_cards/__init__.py
"""
Model Card Library for LLMCore.

This package provides comprehensive model metadata management for LLM applications,
enabling model discovery, validation, cost estimation, and capability querying.

Key Components:
    - ModelCard: Complete model metadata including context, capabilities, pricing
    - ModelCardRegistry: Singleton for loading and querying model cards
    - Provider-specific extensions: OllamaExtension, OpenAIExtension, etc.

Quick Start:
    >>> from llmcore.model_cards import get_model_card_registry
    >>>
    >>> # Get the registry singleton
    >>> registry = get_model_card_registry()
    >>>
    >>> # Look up a model card
    >>> card = registry.get("openai", "gpt-4o")
    >>> if card:
    ...     print(f"Context length: {card.get_context_length()}")
    ...     print(f"Supports vision: {card.capabilities.vision}")
    ...     print(f"Input price: ${card.pricing.per_million_tokens.input}/1M")
    >>>
    >>> # List models by provider
    >>> cards = registry.list_cards(provider="anthropic")
    >>> for summary in cards:
    ...     print(f"{summary.model_id}: {summary.context_length} tokens")
    >>>
    >>> # Get pricing for cost estimation
    >>> pricing = registry.get_pricing("openai", "gpt-4o")
    >>> if pricing:
    ...     cost = (1000 / 1_000_000) * pricing["input"]  # Cost for 1K tokens

User Cards:
    Users can add custom model cards to ~/.config/llmcore/model_cards/
    These override built-in cards with the same model_id.

Version: 1.0.0
Phase: Foundation
"""

# =============================================================================
# Schema Exports - Core Data Models
# =============================================================================

# =============================================================================
# Registry Exports - Card Management
# =============================================================================
from .registry import (
    ModelCardRegistry,
    clear_model_card_cache,
    get_model_card,
    get_model_card_registry,
)
from .schema import (
    # Provider extensions
    AnthropicExtension,
    # Enums
    ArchitectureType,
    # Core structures
    ContextTier,
    DeepSeekExtension,
    EmbeddingConfig,
    GoogleExtension,
    MistralExtension,
    ModelArchitecture,
    ModelCapabilities,
    # Main models
    ModelCard,
    ModelCardSummary,
    ModelContext,
    ModelLifecycle,
    ModelPricing,
    ModelStatus,
    ModelType,
    OllamaExtension,
    OpenAIExtension,
    Provider,
    QwenExtension,
    RateLimits,
    TokenPricing,
    XAIExtension,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "ArchitectureType",
    "ModelStatus",
    "ModelType",
    "Provider",
    # Core structures
    "ContextTier",
    "EmbeddingConfig",
    "ModelArchitecture",
    "ModelCapabilities",
    "ModelContext",
    "ModelLifecycle",
    "ModelPricing",
    "RateLimits",
    "TokenPricing",
    # Provider extensions
    "AnthropicExtension",
    "DeepSeekExtension",
    "GoogleExtension",
    "MistralExtension",
    "OllamaExtension",
    "OpenAIExtension",
    "QwenExtension",
    "XAIExtension",
    # Main models
    "ModelCard",
    "ModelCardSummary",
    # Registry
    "ModelCardRegistry",
    "get_model_card_registry",
    "get_model_card",
    "clear_model_card_cache",
]

__version__ = "1.0.0"
