# src/llmcore/models.py
"""
Core data models for the LLMCore library.

This module defines the Pydantic models used to represent fundamental
data structures such as messages, roles, chat sessions, context documents,
context items, and context presets. It also includes models for the
unified tool-calling interface and dynamic provider introspection.

UPDATED: Migrated to Pydantic V2 patterns (ConfigDict, field_serializer)
to eliminate deprecation warnings.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class Role(str, Enum):
    """
    Enumeration of possible roles in a conversation.
    These roles define the origin or type of a message.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # Added for tool results

    @classmethod
    def _missing_(cls, value: object):  # type: ignore[misc]
        """
        Handles case-insensitive matching and common aliases for roles.
        For example, "Agent" or "AGENT" will be mapped to Role.ASSISTANT.
        """
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value == "agent":
                return cls.ASSISTANT
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


class Message(BaseModel):
    """
    Represents a single message within a chat session.

    Attributes:
        id: A unique identifier for the message.
        session_id: The identifier of the chat session this message belongs to.
        role: The role of the entity that produced the message.
        content: The textual content of the message.
        timestamp: The date and time when the message was created or recorded.
        tool_call_id: For messages with role 'tool', the ID of the tool call this is a response to.
        tokens: An optional count of tokens for the message content.
        metadata: An optional dictionary for storing additional, unstructured information.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the message."
    )
    session_id: Optional[str] = Field(
        default=None, description="Identifier of the chat session this message belongs to."
    )
    role: Role = Field(
        description="The role of the message sender (system, user, assistant, or tool)."
    )
    content: str = Field(description="The textual content of the message.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when the message was created (UTC).",
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="For role 'tool', the ID of the corresponding tool call."
    )
    tokens: Optional[int] = Field(
        default=None, description="Optional token count for the message content."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional dictionary for additional message metadata."
    )

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format with Z suffix."""
        return dt.isoformat().replace("+00:00", "Z")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            try:
                if v.endswith("Z"):
                    v_parsed = datetime.fromisoformat(v[:-1] + "+00:00")
                else:
                    v_parsed = datetime.fromisoformat(v)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        v_parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid datetime format: {v}")

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None:
            return datetime.now(timezone.utc)
        return v


class ContextItemType(str, Enum):
    """
    Enumeration of types for items that can be part of the LLM context pool or a saved preset.
    """

    HISTORY_MESSAGE = "history_message"
    USER_TEXT = "user_text"
    USER_FILE = "user_file"
    RAG_SNIPPET = "rag_snippet"
    PRESET_TEXT_CONTENT = "preset_text_content"
    PRESET_FILE_REFERENCE = "preset_file_reference"
    PRESET_RAG_CONTENT = "preset_rag_content"

    @classmethod
    def _missing_(cls, value: object):  # type: ignore[misc]
        if isinstance(value, str):
            lower_value = value.lower()
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


class ContextItem(BaseModel):
    """
    Represents an individual item that can be part of the LLM's context pool,
    typically managed within a ChatSession's `context_items` list (workspace).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ContextItemType
    source_id: Optional[str] = None
    content: str
    tokens: Optional[int] = None
    original_tokens: Optional[int] = None
    is_truncated: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format with Z suffix."""
        return dt.isoformat().replace("+00:00", "Z")


class EpisodeType(str, Enum):
    """Enumeration of possible event types in an agent's episodic memory."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    USER_INTERACTION = "user_interaction"
    AGENT_REFLECTION = "agent_reflection"

    @classmethod
    def _missing_(cls, value: object):  # type: ignore[misc]
        if isinstance(value, str):
            lower_value = value.lower()
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


class Episode(BaseModel):
    """Represents a single event in an agent's experience log (Episodic Memory)."""

    episode_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the episode."
    )
    session_id: str = Field(description="The session this episode belongs to.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the event (UTC).",
    )
    event_type: EpisodeType = Field(description="The type of event that occurred.")
    data: Dict[str, Any] = Field(
        description="A JSON blob containing the structured data of the event."
    )

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format with Z suffix."""
        return dt.isoformat().replace("+00:00", "Z")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            try:
                if v.endswith("Z"):
                    v_parsed = datetime.fromisoformat(v[:-1] + "+00:00")
                else:
                    v_parsed = datetime.fromisoformat(v)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        v_parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid datetime format: {v}")

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None:
            return datetime.now(timezone.utc)
        return v


class ChatSession(BaseModel):
    """
    Represents a single conversation or chat session.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    context_items: List[ContextItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format with Z suffix."""
        return dt.isoformat().replace("+00:00", "Z")

    def add_message(
        self,
        message_content: str,
        role: Role,
        session_id_override: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a new message to the session.

        Args:
            message_content: The content of the message.
            role: The role of the message sender.
            session_id_override: Optional session ID override.
            metadata: Optional metadata dictionary for tracking additional info
                      (e.g., provider/model for assistant, context info for user).

        Returns:
            The newly created Message object.
        """
        new_message = Message(
            content=message_content,
            role=role,
            session_id=session_id_override or self.id,
            metadata=metadata or {},
        )
        self.messages.append(new_message)
        self.updated_at = datetime.now(timezone.utc)
        return new_message

    def add_context_item(self, item: ContextItem) -> None:
        self.context_items = [ci for ci in self.context_items if ci.id != item.id]
        self.context_items.append(item)
        self.context_items.sort(key=lambda x: x.timestamp)
        self.updated_at = datetime.now(timezone.utc)

    def remove_context_item(self, item_id: str) -> bool:
        initial_len = len(self.context_items)
        self.context_items = [ci for ci in self.context_items if ci.id != item_id]
        if len(self.context_items) < initial_len:
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def get_context_item(self, item_id: str) -> Optional[ContextItem]:
        for item in self.context_items:
            if item.id == item_id:
                return item
        return None


class ContextDocument(BaseModel):
    """
    Represents a document used for context, typically in Retrieval Augmented Generation (RAG).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

    model_config = ConfigDict(validate_assignment=True)


class ContextPreparationDetails(BaseModel):
    """
    Detailed information about how context was prepared for an LLM interaction.

    This model provides comprehensive metadata for UI rendering, debugging,
    and analytics purposes. It captures both the context preparation phase
    and post-interaction token usage.

    Attributes:
        prepared_messages: The final list of messages sent to the LLM.
        final_token_count: The token count of the prepared context (prompt tokens).
        max_tokens_for_model: The maximum context window for the model used.
        rag_documents_used: Optional list of RAG documents included in context.
        rendered_rag_template_content: The formatted RAG prompt if RAG was used.
        truncation_actions_taken: Details of any truncation operations performed.
        provider: The LLM provider name used (e.g., 'openai', 'anthropic').
        model: The specific model used (e.g., 'gpt-4', 'claude-3-5-sonnet-20241022').
        prompt_tokens: Number of tokens in the prompt sent to the LLM.
        completion_tokens: Number of tokens in the LLM's response.
        total_tokens: Total tokens used (prompt + completion).
        rag_used: Whether RAG was enabled for this interaction.
        rag_documents_retrieved: Number of RAG documents retrieved (if applicable).
        context_truncation_applied: Whether context truncation was necessary.
        max_context_length: Maximum context window size for the model.
        reserved_response_tokens: Tokens reserved for the LLM response.
        available_context_tokens: Tokens available for context after reserving response space.
    """

    # === Core Context Preparation Fields (existing) ===
    prepared_messages: List[Message] = Field(
        default_factory=list, description="The final list of messages prepared for the LLM"
    )
    final_token_count: int = Field(default=0, description="The token count of the prepared context")
    max_tokens_for_model: int = Field(
        default=0, description="The maximum context window for the model used"
    )
    rag_documents_used: Optional[List[ContextDocument]] = Field(
        default=None, description="List of RAG documents included in context"
    )
    rendered_rag_template_content: Optional[str] = Field(
        default=None, description="The formatted RAG prompt with injected context"
    )
    truncation_actions_taken: Dict[str, Any] = Field(
        default_factory=dict, description="Details of any truncation operations performed"
    )

    # === Provider and Model Information (NEW) ===
    provider: str = Field(
        default="", description="LLM provider name (e.g., 'openai', 'anthropic', 'ollama')"
    )
    model: str = Field(
        default="", description="Specific model used (e.g., 'gpt-4', 'claude-3-5-sonnet-20241022')"
    )

    # === Token Usage (NEW) ===
    prompt_tokens: int = Field(
        default=0, description="Number of tokens in the prompt sent to the LLM"
    )
    completion_tokens: int = Field(default=0, description="Number of tokens in the LLM's response")
    total_tokens: int = Field(default=0, description="Total tokens used (prompt + completion)")

    # === RAG Information (NEW) ===
    rag_used: bool = Field(
        default=False, description="Whether RAG was enabled for this interaction"
    )
    rag_documents_retrieved: Optional[int] = Field(
        default=None, description="Number of RAG documents retrieved (if applicable)"
    )

    # === Context Management (NEW) ===
    context_truncation_applied: bool = Field(
        default=False, description="Whether context truncation was necessary"
    )
    max_context_length: int = Field(
        default=0, description="Maximum context window size for the model"
    )
    reserved_response_tokens: int = Field(
        default=0, description="Tokens reserved for the LLM response"
    )
    available_context_tokens: int = Field(
        default=0, description="Tokens available for context after reserving response space"
    )

    model_config = ConfigDict(validate_assignment=True)


class ContextPresetItem(BaseModel):
    """
    Represents an item within a saved ContextPreset.
    """

    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ContextItemType
    content: Optional[str] = None
    source_identifier: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )


class ContextPreset(BaseModel):
    """
    Represents a named, saved collection of context items (a "Context Preset").
    """

    name: str
    description: Optional[str] = None
    items: List[ContextPresetItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format with Z suffix."""
        return dt.isoformat().replace("+00:00", "Z")

    @model_validator(mode="before")
    @classmethod
    def ensure_name_is_valid_identifier(cls, data: Any) -> Any:
        if isinstance(data, dict):
            name = data.get("name")
            if name and not isinstance(name, str):
                raise ValueError("Preset name must be a string.")
            if name and (
                not name.strip()
                or any(c in name for c in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"])
            ):
                raise ValueError(f"Preset name '{name}' contains invalid characters or is empty.")
        return data


# --- Models for spec5.md ---


class ModelDetails(BaseModel):
    """
    Represents detailed information about a specific LLM model, discovered dynamically.

    This model aggregates information from multiple sources:
    1. Model card registry (authoritative metadata)
    2. Provider APIs (for Ollama local, OpenAI list, etc.)
    3. Configuration (user-specified models)

    Attributes:
        id: The unique identifier for the model (e.g., "gpt-4o").
        provider_name: The name of the provider this model belongs to.
        display_name: Human-friendly name for the model.
        context_length: The maximum context window size in tokens.
        max_output_tokens: Maximum number of output tokens the model can generate.
        supports_streaming: Flag indicating if the model supports streaming responses.
        supports_tools: Flag indicating if the model supports tool/function calling.
        supports_vision: Flag indicating if the model supports vision/image inputs.
        supports_reasoning: Flag indicating if the model supports extended reasoning.
        family: Model family (e.g., "GPT-4", "Claude", "Llama").
        parameter_count: Model parameter count as string (e.g., "70B", "8x7B").
        quantization_level: Quantization level for local models (e.g., "Q4_K_M").
        file_size_bytes: File size on disk for local models (Ollama).
        model_type: Type of model ("chat", "embedding", "completion").
        metadata: A dictionary for any other provider-specific metadata.
    """

    id: str = Field(description="The unique identifier for the model.")
    provider_name: str = Field(description="The name of the provider this model belongs to.")
    display_name: Optional[str] = Field(
        default=None, description="Human-friendly name for the model."
    )
    context_length: int = Field(
        default=4096, description="The maximum context window size in tokens."
    )
    max_output_tokens: Optional[int] = Field(
        default=None, description="Maximum number of output tokens the model can generate."
    )
    supports_streaming: bool = Field(
        default=True, description="Indicates if the model supports streaming responses."
    )
    supports_tools: bool = Field(
        default=False, description="Indicates if the model supports tool/function calling."
    )
    supports_vision: bool = Field(
        default=False, description="Indicates if the model supports vision/image inputs."
    )
    supports_reasoning: bool = Field(
        default=False, description="Indicates if the model supports extended reasoning."
    )
    family: Optional[str] = Field(
        default=None, description="Model family (e.g., 'GPT-4', 'Claude', 'Llama')."
    )
    parameter_count: Optional[str] = Field(
        default=None, description="Model parameter count as string (e.g., '70B', '8x7B')."
    )
    quantization_level: Optional[str] = Field(
        default=None, description="Quantization level for local models (e.g., 'Q4_K_M')."
    )
    file_size_bytes: Optional[int] = Field(
        default=None, description="File size on disk for local models (Ollama)."
    )
    model_type: Optional[str] = Field(
        default="chat", description="Type of model ('chat', 'embedding', 'completion')."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific metadata."
    )


class ModelValidationResult(BaseModel):
    """
    Result of validating a model for a provider.

    Used by validate_model_for_provider() to return comprehensive validation
    information including suggestions for similar models when validation fails.

    Attributes:
        is_valid: Whether the model is available for the provider.
        canonical_name: The correct/canonical model name (may differ in case).
        suggestions: List of similar model names if not found.
        error_message: Human-readable error or note message.
        model_details: Full model details if validation succeeded.
    """

    is_valid: bool = Field(description="Whether the model is available for the provider.")
    canonical_name: Optional[str] = Field(
        default=None, description="The correct/canonical model name (may differ in case)."
    )
    suggestions: List[str] = Field(
        default_factory=list, description="List of similar model names if not found."
    )
    error_message: Optional[str] = Field(
        default=None, description="Human-readable error or note message."
    )
    model_details: Optional[ModelDetails] = Field(
        default=None, description="Full model details if validation succeeded."
    )


class PullProgress(BaseModel):
    """
    Progress update during model pull/download operation.

    Used by pull_model() to report download progress via callback.

    Attributes:
        status: Current status ("pulling manifest", "downloading", "verifying", "success").
        digest: Layer/file digest being processed.
        total_bytes: Total bytes to download (may be None if unknown).
        completed_bytes: Bytes downloaded so far.
        percent_complete: Percentage complete (0-100).
        layer: Current layer identifier (Ollama-specific).
    """

    status: str = Field(description="Current status of the pull operation.")
    digest: Optional[str] = Field(default=None, description="Layer/file digest being processed.")
    total_bytes: Optional[int] = Field(
        default=None, description="Total bytes to download (may be None if unknown)."
    )
    completed_bytes: Optional[int] = Field(default=None, description="Bytes downloaded so far.")
    percent_complete: Optional[float] = Field(
        default=None, description="Percentage complete (0-100)."
    )
    layer: Optional[str] = Field(
        default=None, description="Current layer identifier (Ollama-specific)."
    )


class PullResult(BaseModel):
    """
    Result of a model pull/download operation.

    Attributes:
        success: Whether the pull completed successfully.
        model_name: The name of the model that was pulled.
        error_message: Error message if pull failed.
        duration_seconds: Time taken to pull the model.
    """

    success: bool = Field(description="Whether the pull completed successfully.")
    model_name: str = Field(description="The name of the model that was pulled.")
    error_message: Optional[str] = Field(default=None, description="Error message if pull failed.")
    duration_seconds: float = Field(
        default=0.0, description="Time taken to pull the model in seconds."
    )


class SessionTokenStats(BaseModel):
    """
    Cumulative token statistics for a session.

    This model aggregates token usage data from all interactions within a
    session, providing summary statistics useful for monitoring, cost estimation,
    and usage analytics.

    Attributes:
        session_id: The session this statistics object belongs to.
        total_prompt_tokens: Cumulative count of all input/prompt tokens.
        total_completion_tokens: Cumulative count of all output/completion tokens.
        total_tokens: Sum of prompt and completion tokens.
        total_cached_tokens: Cumulative count of tokens served from provider cache.
        interaction_count: Number of chat interactions in the session.
        avg_prompt_tokens: Average prompt tokens per interaction.
        avg_completion_tokens: Average completion tokens per interaction.
        max_prompt_tokens: Maximum prompt tokens in a single interaction.
        max_completion_tokens: Maximum completion tokens in a single interaction.
        first_interaction_at: Timestamp of the first interaction.
        last_interaction_at: Timestamp of the most recent interaction.
        by_model: Token breakdown by model used (for multi-model sessions).
    """

    session_id: str = Field(description="Session identifier.")
    total_prompt_tokens: int = Field(default=0, description="Total input tokens used.")
    total_completion_tokens: int = Field(default=0, description="Total output tokens used.")
    total_tokens: int = Field(default=0, description="Total tokens (prompt + completion).")
    total_cached_tokens: int = Field(default=0, description="Total tokens served from cache.")
    interaction_count: int = Field(default=0, description="Number of chat interactions.")
    avg_prompt_tokens: float = Field(
        default=0.0, description="Average prompt tokens per interaction."
    )
    avg_completion_tokens: float = Field(
        default=0.0, description="Average completion tokens per interaction."
    )
    max_prompt_tokens: int = Field(
        default=0, description="Maximum prompt tokens in any interaction."
    )
    max_completion_tokens: int = Field(
        default=0, description="Maximum completion tokens in any interaction."
    )
    first_interaction_at: Optional[datetime] = Field(
        default=None, description="Timestamp of first interaction (UTC)."
    )
    last_interaction_at: Optional[datetime] = Field(
        default=None, description="Timestamp of most recent interaction (UTC)."
    )
    by_model: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Token usage breakdown by model. Format: "
            "{'model_name': {'prompt': int, 'completion': int, 'count': int}}"
        ),
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_serializer("first_interaction_at", "last_interaction_at")
    def serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format with Z suffix."""
        if dt is None:
            return None
        return dt.isoformat().replace("+00:00", "Z")


class CostEstimate(BaseModel):
    """
    Cost estimation result for token usage.

    This model provides a detailed breakdown of estimated costs based on
    token counts and model card pricing data.

    Attributes:
        input_cost: Cost for input/prompt tokens.
        output_cost: Cost for output/completion tokens.
        cached_discount: Cost savings from cached tokens (if applicable).
        reasoning_cost: Cost for reasoning tokens (for thinking models).
        total_cost: Total estimated cost.
        currency: Currency code (ISO 4217, default "USD").
        pricing_source: Source of pricing data ("model_card", "estimated", "unavailable").
        prompt_tokens: Number of prompt tokens used in estimate.
        completion_tokens: Number of completion tokens used in estimate.
        cached_tokens: Number of cached tokens (subset of prompt).
        reasoning_tokens: Number of reasoning tokens (for thinking models).
        input_price_per_million: Input price per 1M tokens used.
        output_price_per_million: Output price per 1M tokens used.
        cached_price_per_million: Cached input price per 1M tokens used.
        model_id: Model identifier used for pricing.
        provider: Provider name used for pricing.
    """

    input_cost: float = Field(default=0.0, description="Cost for input tokens.")
    output_cost: float = Field(default=0.0, description="Cost for output tokens.")
    cached_discount: float = Field(default=0.0, description="Cost savings from cached tokens.")
    reasoning_cost: float = Field(default=0.0, description="Cost for reasoning tokens.")
    total_cost: float = Field(default=0.0, description="Total estimated cost.")
    currency: str = Field(default="USD", description="Currency code (ISO 4217).")
    pricing_source: str = Field(
        default="model_card",
        description="Source of pricing data: 'model_card', 'estimated', 'unavailable'.",
    )

    # Token counts for reference
    prompt_tokens: int = Field(default=0, description="Prompt tokens in this estimate.")
    completion_tokens: int = Field(default=0, description="Completion tokens in this estimate.")
    cached_tokens: int = Field(default=0, description="Cached tokens (subset of prompt).")
    reasoning_tokens: int = Field(default=0, description="Reasoning tokens used.")

    # Pricing rates used
    input_price_per_million: Optional[float] = Field(
        default=None, description="Input price per 1M tokens."
    )
    output_price_per_million: Optional[float] = Field(
        default=None, description="Output price per 1M tokens."
    )
    cached_price_per_million: Optional[float] = Field(
        default=None, description="Cached input price per 1M tokens."
    )

    # Model identification
    model_id: Optional[str] = Field(default=None, description="Model identifier used for pricing.")
    provider: Optional[str] = Field(default=None, description="Provider name.")

    model_config = ConfigDict(validate_assignment=True)

    def format_cost(self, precision: int = 4) -> str:
        """
        Format the total cost as a currency string.

        Args:
            precision: Decimal places to display.

        Returns:
            Formatted cost string (e.g., "$0.0123").
        """
        if self.pricing_source == "unavailable":
            return "N/A (local model)"
        symbol = "$" if self.currency == "USD" else self.currency
        return f"{symbol}{self.total_cost:.{precision}f}"


class Tool(BaseModel):
    """
    Represents a function that can be called by an LLM, defined in a provider-agnostic way.

    Attributes:
        name: The name of the tool/function.
        description: A description of what the tool does, used by the LLM to decide when to call it.
        parameters: A dictionary representing the JSON Schema for the tool's input parameters.
    """

    name: str = Field(description="The name of the tool/function.")
    description: str = Field(description="A description of what the tool does.")
    parameters: Dict[str, Any] = Field(
        description="A JSON Schema object defining the tool's input parameters."
    )


class ToolCall(BaseModel):
    """
    Represents a request from the LLM to execute a specific tool with provided arguments.

    Attributes:
        id: A unique identifier for this specific tool call, used to match it with a ToolResult.
        name: The name of the tool to be executed.
        arguments: A dictionary of arguments for the tool, as generated by the LLM.
    """

    id: str = Field(description="Unique identifier for this specific tool call.")
    name: str = Field(description="The name of the tool to be executed.")
    arguments: Dict[str, Any] = Field(
        description="A dictionary of arguments for the tool, generated by the LLM."
    )


class ToolResult(BaseModel):
    """
    Represents the output from the execution of a tool, to be sent back to the LLM.

    Attributes:
        tool_call_id: The ID of the ToolCall this result corresponds to.
        content: The string representation of the tool's output.
        is_error: Whether the tool execution resulted in an error.
    """

    tool_call_id: str = Field(description="The ID of the ToolCall this result corresponds to.")
    content: str = Field(description="The string representation of the tool's output.")
    is_error: bool = Field(
        default=False, description="Whether the tool execution resulted in an error."
    )


# --- Models for Phase 4: Agentic Loop ---


class AgentState(BaseModel):
    """
    Represents the agent's "Working Memory" or "scratchpad" for a specific task.
    This model holds the transient, short-term cognitive state required for the
    agent's reasoning loop.

    Attributes:
        goal: The high-level objective the agent is trying to achieve.
        plan: A list of strings representing the decomposed steps the agent intends to take.
        current_plan_step_index: Index of the current step being executed in the plan.
        plan_steps_status: List tracking the status of each plan step ('pending', 'completed', 'failed').
        history_of_thoughts: A log of the agent's internal reasoning steps ("Thoughts").
        observations: A dictionary mapping tool calls or actions to their observed results.
        scratchpad: A free-form text field for intermediate reasoning or notes.
    """

    goal: str = Field(description="The high-level objective for the agent.")
    plan: List[str] = Field(default_factory=list, description="The decomposed plan of sub-tasks.")
    current_plan_step_index: int = Field(
        default=0, description="Index of the current plan step being executed."
    )
    plan_steps_status: List[str] = Field(
        default_factory=list,
        description="Status of each plan step ('pending', 'completed', 'failed').",
    )
    history_of_thoughts: List[str] = Field(
        default_factory=list, description="A chronological log of the agent's internal 'Thoughts'."
    )
    observations: Dict[str, Any] = Field(
        default_factory=dict, description="A mapping of actions to their observed results."
    )
    scratchpad: str = Field(
        default="", description="A transient workspace for intermediate reasoning."
    )

    model_config = ConfigDict(validate_assignment=True)


class AgentTask(BaseModel):
    """
    Represents an asynchronous agentic task being managed by the TaskMaster service.
    This model tracks the lifecycle and state of a long-running agent operation.

    UPDATED: Added HITL workflow support with pending_action_data and approval_prompt fields.

    Attributes:
        task_id: A unique identifier for the agent task.
        status: The current status of the task (e.g., PENDING, RUNNING, SUCCESS, FAILURE, PENDING_APPROVAL).
        goal: The original goal provided by the user.
        agent_state: The current working memory (AgentState) of the agent performing the task.
        pending_action_data: JSON representation of the ToolCall awaiting human approval (HITL workflow).
        approval_prompt: The question/prompt for the human operator (HITL workflow).
        created_at: The timestamp when the task was created.
        updated_at: The timestamp when the task was last updated.
    """

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the agent task.",
    )
    status: str = Field(default="PENDING", description="The current status of the task.")
    goal: str = Field(description="The original high-level goal for the task.")
    agent_state: AgentState = Field(description="The agent's current working memory state.")
    pending_action_data: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON representation of the ToolCall awaiting approval (HITL)."
    )
    approval_prompt: Optional[str] = Field(
        default=None, description="The question for the human operator (HITL workflow)."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of task creation (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last task update (UTC).",
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format with Z suffix."""
        return dt.isoformat().replace("+00:00", "Z")
