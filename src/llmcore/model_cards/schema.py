# src/llmcore/model_cards/schema.py
# llmcore/model_cards/schema.py
"""
Pydantic models for the Model Card Library.

This module defines the complete schema for model cards, supporting:
- Chat models (LLMs)
- Embedding models
- Multimodal models
- Provider-specific extensions

The schema is designed to be:
1. Comprehensive - covers all common model attributes
2. Extensible - provider-specific fields via extension models
3. Type-safe - full Pydantic validation
4. JSON-serializable - for file storage

Version: 1.0.0
Phase: Foundation
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ModelType(str, Enum):
    """Type of model."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE_GENERATION = "image-generation"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class ArchitectureType(str, Enum):
    """Model architecture type."""

    TRANSFORMER = "transformer"
    MOE = "moe"  # Mixture of Experts
    SSM = "ssm"  # State Space Model (e.g., Mamba)
    HYBRID = "hybrid"


class ModelStatus(str, Enum):
    """Model lifecycle status."""

    ACTIVE = "active"
    PREVIEW = "preview"
    BETA = "beta"
    DEPRECATED = "deprecated"
    LEGACY = "legacy"
    RETIRED = "retired"


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    KIMI = "kimi"
    XAI = "xai"
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    REPLICATE = "replicate"
    PERPLEXITY = "perplexity"
    AI21 = "ai21"
    GROQ = "groq"
    LOCAL = "local"


# =============================================================================
# COMMON STRUCTURES
# =============================================================================


class ModelArchitecture(BaseModel):
    """Model architecture details."""

    family: str | None = Field(
        None, description="Model family (llama, gpt, claude, gemini, etc.)"
    )
    parameter_count: str | None = Field(
        None, description="Total parameters (e.g., '70B', '8B', '405B')"
    )
    active_parameters: str | None = Field(
        None, description="Active parameters for MoE models (e.g., '37B' for DeepSeek)"
    )
    architecture_type: ArchitectureType | None = Field(
        None, description="Architecture type (transformer, moe, ssm, hybrid)"
    )

    model_config = {"use_enum_values": True}


class ModelContext(BaseModel):
    """Context window configuration."""

    max_input_tokens: int = Field(..., description="Maximum input context length in tokens")
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens (if different from input)"
    )
    default_output_tokens: int | None = Field(
        None, description="Default output length if not specified by user"
    )


class ModelCapabilities(BaseModel):
    """Model capability flags indicating supported features."""

    streaming: bool = Field(True, description="Supports streaming responses")
    function_calling: bool = Field(False, description="Supports function/tool calling")
    tool_use: bool = Field(False, description="Supports tool use (modern term)")
    json_mode: bool = Field(False, description="Supports JSON output mode")
    structured_output: bool = Field(False, description="Supports structured output schemas")
    vision: bool = Field(False, description="Supports image input")
    audio_input: bool = Field(False, description="Supports audio input")
    audio_output: bool = Field(False, description="Supports audio output (TTS)")
    video_input: bool = Field(False, description="Supports video input")
    image_generation: bool = Field(False, description="Can generate images")
    code_execution: bool = Field(False, description="Can execute code in sandbox")
    web_search: bool = Field(False, description="Has web search capability")
    reasoning: bool = Field(False, description="Extended thinking / chain-of-thought")
    file_processing: bool = Field(False, description="Can process uploaded files")


class TokenPricing(BaseModel):
    """Token pricing per million tokens."""

    input: float = Field(..., description="Input token price per 1M tokens")
    output: float = Field(..., description="Output token price per 1M tokens")
    cached_input: float | None = Field(
        None, description="Cached input price per 1M tokens (prompt caching)"
    )
    reasoning_output: float | None = Field(
        None, description="Reasoning token price per 1M (for o1/thinking models)"
    )


class ContextTier(BaseModel):
    """Pricing tier based on context window usage."""

    threshold_tokens: int = Field(..., description="Token threshold for this tier")
    input_price: float = Field(..., description="Input price per 1M at this tier")
    output_price: float = Field(..., description="Output price per 1M at this tier")


class ModelPricing(BaseModel):
    """Complete pricing information for a model."""

    currency: str = Field("USD", description="Currency code (ISO 4217)")
    per_million_tokens: TokenPricing
    batch_discount_percent: float | None = Field(
        None, description="Discount percentage for batch API (e.g., 50 for 50% off)"
    )
    context_tiers: list[ContextTier] | None = Field(
        None, description="Tiered pricing based on context usage"
    )

    def get_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """
        Calculate cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (subset of input)

        Returns:
            Total cost in the model's currency
        """
        # Calculate base input cost
        non_cached_input = max(0, input_tokens - cached_tokens)
        input_cost = (non_cached_input / 1_000_000) * self.per_million_tokens.input

        # Add cached token cost if applicable
        cached_cost = 0.0
        if cached_tokens > 0 and self.per_million_tokens.cached_input is not None:
            cached_cost = (cached_tokens / 1_000_000) * self.per_million_tokens.cached_input
        elif cached_tokens > 0:
            # If no cached price, charge full input price
            cached_cost = (cached_tokens / 1_000_000) * self.per_million_tokens.input

        # Calculate output cost
        output_cost = (output_tokens / 1_000_000) * self.per_million_tokens.output

        return input_cost + cached_cost + output_cost


class ModelLifecycle(BaseModel):
    """Model lifecycle and availability information."""

    release_date: date | None = Field(None, description="Initial release date")
    knowledge_cutoff: str | None = Field(
        None, description="Training data cutoff (e.g., '2024-04', '2023-10')"
    )
    deprecation_date: date | None = Field(
        None, description="Date when model was/will be deprecated"
    )
    shutdown_date: date | None = Field(None, description="Date when model will be/was shut down")
    successor_model: str | None = Field(None, description="Recommended replacement model ID")
    status: ModelStatus = Field(ModelStatus.ACTIVE, description="Current lifecycle status")

    model_config = {"use_enum_values": True}


class RateLimits(BaseModel):
    """Rate limit configuration for a pricing tier."""

    requests_per_minute: int | None = Field(None, description="RPM limit")
    tokens_per_minute: int | None = Field(None, description="TPM limit")
    input_tokens_per_minute: int | None = Field(
        None, description="Input TPM limit (if separate)"
    )
    output_tokens_per_minute: int | None = Field(
        None, description="Output TPM limit (if separate)"
    )


class EmbeddingConfig(BaseModel):
    """Configuration specific to embedding models."""

    dimensions_default: int = Field(..., description="Default embedding dimensions")
    dimensions_configurable: list[int] | None = Field(
        None, description="Available dimension options (for configurable models)"
    )
    supports_matryoshka: bool = Field(
        False, description="Supports Matryoshka representation learning"
    )
    similarity_metrics: list[str] = Field(
        default_factory=lambda: ["cosine"],
        description="Supported similarity metrics",
    )
    normalization: str | None = Field(None, description="Output normalization (L2, none)")
    task_types: list[str] | None = Field(
        None,
        description="Supported task types (retrieval_query, retrieval_document, classification, etc.)",
    )
    output_types: list[str] | None = Field(
        None, description="Output types (float, int8, binary, ubinary)"
    )
    truncation_strategy: str | None = Field(
        None, description="How input is truncated if too long"
    )
    prefixes: dict[str, str] | None = Field(
        None, description="Required prefixes for different task types"
    )
    batch_limits: dict[str, int] | None = Field(None, description="Batch processing limits")
    languages_supported: int | list[str] | None = Field(
        None, description="Number of languages or list of language codes"
    )
    multimodal: bool = Field(False, description="Supports multimodal input (text + image)")


# =============================================================================
# PROVIDER-SPECIFIC EXTENSIONS
# =============================================================================


class OllamaExtension(BaseModel):
    """Ollama-specific model fields."""

    format: Literal["gguf", "safetensors"] | None = Field(None, description="Model file format")
    quantization_level: str | None = Field(
        None, description="Quantization level (Q4_0, Q4_K_M, Q5_K_M, Q8_0, FP16, etc.)"
    )
    file_size_bytes: int | None = Field(None, description="Model file size in bytes")
    digest: str | None = Field(None, description="Model digest/hash for verification")
    template: str | None = Field(None, description="Chat template (Go template syntax)")
    system_prompt: str | None = Field(None, description="Default system prompt")
    modelfile_parameters: dict[str, Any] | None = Field(
        None, description="Modelfile parameters (num_ctx, temperature, stop, etc.)"
    )
    gguf_metadata: dict[str, Any] | None = Field(
        None, description="Metadata extracted from GGUF file"
    )
    parent_model: str | None = Field(None, description="Base model this was derived from")
    modified_at: datetime | None = Field(
        None, description="When the model was last modified locally"
    )


class OpenAIExtension(BaseModel):
    """OpenAI-specific model fields."""

    owned_by: str | None = Field(None, description="Organization that owns the model")
    supports_reasoning: bool = Field(False, description="o1/o3 series model with reasoning")
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        None, description="Reasoning effort level for o1/o3 models"
    )
    supports_predicted_outputs: bool = Field(
        False, description="Supports predicted outputs feature"
    )
    fine_tuning_available: bool = Field(False, description="Available for fine-tuning")
    moderation_model: bool = Field(False, description="This is a moderation model")
    tier_requirements: dict[str, Any] | None = Field(
        None, description="Minimum tier requirements for access"
    )


class AnthropicExtension(BaseModel):
    """Anthropic-specific model fields."""

    extended_thinking: dict[str, Any] | None = Field(
        None,
        description="Extended thinking configuration (supported, budget_tokens_range)",
    )
    computer_use: bool = Field(False, description="Supports computer use capability")
    prompt_caching: dict[str, float] | None = Field(
        None,
        description="Prompt caching multipliers (cache_write_5m, cache_write_1h, cache_read)",
    )
    beta_features: list[str] | None = Field(None, description="Available beta features")


class GoogleExtension(BaseModel):
    """Google/Gemini-specific model fields."""

    supported_inputs: list[str] | None = Field(
        None, description="Supported input types (text, image, video, audio, pdf)"
    )
    supported_outputs: list[str] | None = Field(None, description="Supported output types")
    grounding: dict[str, Any] | None = Field(
        None, description="Grounding capabilities (google_search, maps)"
    )
    thinking: dict[str, Any] | None = Field(
        None, description="Thinking mode configuration (supported, budget_range)"
    )
    live_api: bool = Field(False, description="Supports Live API for real-time")
    url_context: bool = Field(False, description="Supports URL context input")
    safety_settings: list[dict[str, str]] | None = Field(
        None, description="Default safety settings"
    )
    versions: dict[str, str] | None = Field(
        None, description="Version aliases (stable, preview, experimental)"
    )


class DeepSeekExtension(BaseModel):
    """DeepSeek-specific model fields."""

    thinking_mode: dict[str, Any] | None = Field(
        None,
        description="Deep thinking mode configuration (supported, param, default)",
    )
    cache_hit_discount: float | None = Field(
        None, description="Automatic cache hit discount rate (e.g., 0.90 for 90% off)"
    )
    fill_in_middle: bool = Field(False, description="Supports fill-in-middle for code completion")
    moe_architecture: dict[str, str] | None = Field(
        None, description="MoE details (total_parameters, active_parameters)"
    )


class QwenExtension(BaseModel):
    """Qwen/Alibaba-specific model fields."""

    deployment_regions: list[str] | None = Field(
        None, description="Available deployment regions"
    )
    thinking_mode: dict[str, Any] | None = Field(None, description="Thinking mode configuration")
    context_tiers: list[dict[str, Any]] | None = Field(
        None, description="Context-based pricing tiers"
    )
    cache_types: dict[str, float] | None = Field(
        None, description="Cache discount rates (implicit, explicit)"
    )
    specialized_variant: str | None = Field(
        None, description="Specialized variant (base, coder, vl, omni, math)"
    )


class MistralExtension(BaseModel):
    """Mistral-specific model fields."""

    open_weights: bool = Field(False, description="Open weights model")
    license_type: str | None = Field(
        None, description="License type (apache-2.0, mrl, commercial)"
    )
    fill_in_middle: bool = Field(False, description="Codestral FIM support for code completion")
    guardrails: dict[str, bool] | None = Field(None, description="Available guardrail settings")
    fine_tuning: dict[str, Any] | None = Field(None, description="Fine-tuning capabilities")


class XAIExtension(BaseModel):
    """xAI/Grok-specific model fields."""

    live_search: dict[str, Any] | None = Field(
        None, description="Live search configuration (enabled, cost_per_source)"
    )
    x_integration: dict[str, Any] | None = Field(
        None, description="X (Twitter) integration settings"
    )
    server_tools: list[str] | None = Field(
        None,
        description="Available server-side tools (web_search, X_search, code_execution, image_generation)",
    )


# =============================================================================
# MAIN MODEL CARD
# =============================================================================


class ModelCard(BaseModel):
    """
    Complete model card schema.

    This is the primary data structure for model metadata in llmcore.
    It supports chat models, embedding models, and multimodal models
    with provider-specific extensions.

    Attributes:
        model_id: Unique model identifier (e.g., "gpt-4o", "claude-sonnet-4")
        display_name: Human-readable name for UI display
        provider: Provider that hosts this model
        model_type: Type of model (chat, embedding, etc.)
        architecture: Architecture details (optional)
        context: Context window configuration
        capabilities: Supported features
        pricing: Token pricing information (optional, None for local models)
        rate_limits: Rate limits by tier (optional)
        lifecycle: Release and deprecation info
        license: License identifier (e.g., "MIT", "apache-2.0")
        open_weights: Whether model weights are publicly available
        aliases: Alternative model identifiers
        description: Human-readable description
        tags: Categorization tags
        embedding_config: Config for embedding models
        provider_*: Provider-specific extensions
        source: Where this card came from (builtin, user, api)
        last_updated: When the card was last updated
    """

    # -------------------------------------------------------------------------
    # Core Identity
    # -------------------------------------------------------------------------
    model_id: str = Field(..., description="Unique model identifier")
    display_name: str | None = Field(None, description="Human-readable name")
    provider: Provider | str = Field(..., description="Provider that hosts this model")
    model_type: ModelType | str = Field(..., description="Type of model")

    # -------------------------------------------------------------------------
    # Technical Specifications
    # -------------------------------------------------------------------------
    architecture: ModelArchitecture | None = Field(
        None, description="Model architecture details"
    )
    context: ModelContext = Field(..., description="Context window configuration")
    capabilities: ModelCapabilities = Field(
        default_factory=ModelCapabilities, description="Supported features"
    )

    # -------------------------------------------------------------------------
    # Commercial Information
    # -------------------------------------------------------------------------
    pricing: ModelPricing | None = Field(
        None, description="Token pricing (None for local/free models)"
    )
    rate_limits: dict[str, RateLimits] | None = Field(
        None, description="Rate limits by tier (tier_1, tier_2, etc.)"
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    lifecycle: ModelLifecycle = Field(
        default_factory=ModelLifecycle, description="Release and deprecation info"
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    license: str | None = Field(None, description="License identifier")
    open_weights: bool = Field(False, description="Weights publicly available")
    aliases: list[str] = Field(default_factory=list, description="Alternative model IDs")
    description: str | None = Field(None, description="Model description")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")

    # -------------------------------------------------------------------------
    # Type-Specific Configuration
    # -------------------------------------------------------------------------
    embedding_config: EmbeddingConfig | None = Field(
        None, description="Configuration for embedding models"
    )

    # -------------------------------------------------------------------------
    # Provider-Specific Extensions
    # -------------------------------------------------------------------------
    provider_ollama: OllamaExtension | None = None
    provider_openai: OpenAIExtension | None = None
    provider_anthropic: AnthropicExtension | None = None
    provider_google: GoogleExtension | None = None
    provider_deepseek: DeepSeekExtension | None = None
    provider_qwen: QwenExtension | None = None
    provider_mistral: MistralExtension | None = None
    provider_xai: XAIExtension | None = None
    provider_extension: dict[str, Any] | None = Field(
        None, description="Generic extension for other providers"
    )

    # -------------------------------------------------------------------------
    # Source Tracking
    # -------------------------------------------------------------------------
    source: Literal["builtin", "user", "api"] = Field(
        "builtin", description="Where this card came from"
    )
    last_updated: datetime | None = Field(None, description="When card was last updated")

    model_config = {"use_enum_values": True}

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_context_length(self) -> int:
        """Get maximum input context length."""
        return self.context.max_input_tokens

    def get_max_output(self) -> int | None:
        """Get maximum output tokens."""
        return self.context.max_output_tokens

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        """
        Estimate cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens

        Returns:
            Cost in model's currency, or None if no pricing data
        """
        if self.pricing:
            return self.pricing.get_cost(input_tokens, output_tokens, cached_tokens)
        return None

    def supports_capability(self, capability: str) -> bool:
        """
        Check if model supports a specific capability.

        Args:
            capability: Capability name (e.g., "vision", "tool_use", "reasoning")

        Returns:
            True if capability is supported
        """
        return getattr(self.capabilities, capability, False)

    def get_provider_extension(self) -> BaseModel | None:
        """Get the provider-specific extension if present."""
        provider_str = self.provider if isinstance(self.provider, str) else self.provider.value

        ext_map = {
            "ollama": self.provider_ollama,
            "openai": self.provider_openai,
            "anthropic": self.provider_anthropic,
            "google": self.provider_google,
            "deepseek": self.provider_deepseek,
            "qwen": self.provider_qwen,
            "mistral": self.provider_mistral,
            "xai": self.provider_xai,
        }
        return ext_map.get(provider_str) or self.provider_extension

    def is_local(self) -> bool:
        """Check if this is a locally-hosted model (no API cost)."""
        provider_str = self.provider if isinstance(self.provider, str) else self.provider.value
        return provider_str in ("ollama", "local") or self.pricing is None

    def is_deprecated(self) -> bool:
        """Check if model is deprecated or legacy."""
        status = self.lifecycle.status
        if isinstance(status, str):
            return status in ("deprecated", "legacy", "retired")
        return status in (ModelStatus.DEPRECATED, ModelStatus.LEGACY, ModelStatus.RETIRED)


class ModelCardSummary(BaseModel):
    """
    Lightweight summary for listing model cards.

    Used when listing models to avoid loading full card data.
    """

    model_id: str
    display_name: str | None = None
    provider: str
    model_type: str
    context_length: int
    status: str
    source: str
    has_pricing: bool = False
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_card(cls, card: ModelCard) -> ModelCardSummary:
        """Create summary from a full ModelCard."""
        provider_str = card.provider if isinstance(card.provider, str) else card.provider.value
        model_type_str = (
            card.model_type if isinstance(card.model_type, str) else card.model_type.value
        )
        status_str = (
            card.lifecycle.status
            if isinstance(card.lifecycle.status, str)
            else card.lifecycle.status.value
        )

        return cls(
            model_id=card.model_id,
            display_name=card.display_name,
            provider=provider_str,
            model_type=model_type_str,
            context_length=card.context.max_input_tokens,
            status=status_str,
            source=card.source,
            has_pricing=card.pricing is not None,
            tags=card.tags,
        )
