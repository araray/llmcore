# src/llmcore/models.py
"""
Core data models for the LLMCore library.

This module defines the Pydantic models used to represent fundamental
data structures such as messages, roles, chat sessions, and context documents.
These models ensure data consistency, validation, and ease of serialization/deserialization
throughout the library.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Role(str, Enum):
    """
    Enumeration of possible roles in a conversation.
    These roles define the origin or type of a message.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def _missing_(cls, value: object): # type: ignore[misc] # Pydantic uses this signature
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
        return None # Let Pydantic handle the error for truly invalid values


class Message(BaseModel):
    """
    Represents a single message within a chat session.

    Attributes:
        id: A unique identifier for the message.
        session_id: The identifier of the chat session this message belongs to.
        role: The role of the entity that produced the message.
        content: The textual content of the message.
        timestamp: The date and time when the message was created or recorded.
        tokens: An optional count of tokens for the message content.
        metadata: An optional dictionary for storing additional, unstructured information.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the message.")
    session_id: str = Field(description="Identifier of the chat session this message belongs to.")
    role: Role = Field(description="The role of the message sender (system, user, or assistant).")
    content: str = Field(description="The textual content of the message.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the message was created (UTC).")
    tokens: Optional[int] = Field(default=None, description="Optional token count for the message content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional message metadata.")

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            try:
                if v.endswith('Z'):
                    v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
                else:
                    v_parsed = datetime.fromisoformat(v)
            except ValueError:
                 # Attempt to parse common formats if fromisoformat fails
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        v_parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                else: # If all formats fail
                    raise ValueError(f"Invalid datetime format: {v}")

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None: # Should not happen with default_factory, but defensive
            return datetime.now(timezone.utc)
        return v

class ContextItemType(str, Enum):
    """
    Enumeration of types for items that can be part of the LLM context pool.
    """
    HISTORY_MESSAGE = "history_message" # Not typically used for user-added items, but for internal representation
    USER_TEXT = "user_text"
    USER_FILE = "user_file"
    RAG_SNIPPET = "rag_snippet" # For RAG documents pinned by the user to the context pool

    @classmethod
    def _missing_(cls, value: object): # type: ignore[misc]
        """Handle case-insensitive matching for ContextItemType."""
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

    Attributes:
        id: Unique identifier for this context item.
        type: The type of context item (e.g., user_text, user_file, rag_snippet).
        source_id: Optional ID linking back to the original source (e.g., file path for user_file,
                   or original RAG document ID for rag_snippet).
        content: The textual content of the item.
        tokens: Estimated or actual token count for the content (potentially after per-item truncation).
        original_tokens: Optional original token count for the content before any per-item truncation by ContextManager.
        is_truncated: Flag indicating if the content was truncated by the ContextManager due to per-item limits.
        metadata: Additional metadata (e.g., filename for USER_FILE, source for RAG_SNIPPET, ignore_char_limit preference).
        timestamp: Timestamp of creation or relevance for ordering.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the context item.")
    type: ContextItemType = Field(description="Type of the context item.")
    source_id: Optional[str] = Field(default=None, description="Identifier of the original source (e.g., file path, original RAG doc ID).")
    content: str = Field(description="Textual content of the context item.")
    tokens: Optional[int] = Field(default=None, description="Token count for the content, possibly after per-item truncation.")
    original_tokens: Optional[int] = Field(default=None, description="Original token count before any per-item truncation by ContextManager.")
    is_truncated: bool = Field(default=False, description="Flag indicating if the content was truncated by ContextManager due to per-item limits.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., filename, RAG source info, ignore_char_limit).")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of item creation/relevance.")

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            try:
                if v.endswith('Z'):
                    v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
                else:
                    v_parsed = datetime.fromisoformat(v)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        v_parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                else: # If all formats fail
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

    @field_validator('type', mode='before')
    @classmethod
    def validate_type_enum(cls, value: Any) -> ContextItemType:
        """Ensure 'type' is correctly parsed into ContextItemType enum."""
        if isinstance(value, ContextItemType):
            return value
        if isinstance(value, str):
            try:
                return ContextItemType(value.lower())
            except ValueError:
                raise ValueError(f"Invalid ContextItemType: '{value}'. Must be one of {[e.value for e in ContextItemType]}.")
        raise TypeError(f"Invalid type for ContextItemType: {type(value)}. Must be str or ContextItemType enum.")


class ChatSession(BaseModel):
    """
    Represents a single conversation or chat session.

    Attributes:
        id: Unique identifier for the chat session.
        name: Optional human-readable name for the session.
        messages: List of `Message` objects (chat history).
        context_items: List of user-added `ContextItem` objects (e.g., text snippets, files, pinned RAG snippets)
                       that form the session's "workspace" or context pool.
        created_at: Timestamp of session creation.
        updated_at: Timestamp of last session modification.
        metadata: Additional session metadata.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the chat session.")
    name: Optional[str] = Field(default=None, description="Optional human-readable name for the session.")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation, ordered chronologically.")
    context_items: List[ContextItem] = Field(default_factory=list, description="List of user-added items to the context pool for this session.")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the session was created (UTC).")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the session was last updated (UTC).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional session metadata.")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def ensure_utc_timestamps(cls, v: Any) -> datetime:
        """Ensure created_at and updated_at timestamps are timezone-aware and in UTC."""
        # Same implementation as Message.ensure_utc_timestamp
        if isinstance(v, str):
            try:
                if v.endswith('Z'): v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
                else: v_parsed = datetime.fromisoformat(v)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try: v_parsed = datetime.strptime(v, fmt); break
                    except ValueError: continue
                else: raise ValueError(f"Invalid datetime format: {v}")
            if v_parsed.tzinfo is None: return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None: return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None: return datetime.now(timezone.utc)
        return v

    def add_message(self, message_content: str, role: Role, session_id_override: Optional[str] = None) -> Message:
        """
        Creates a new message, adds it to the session, and updates the `updated_at` timestamp.
        (Implementation unchanged)
        """
        new_message = Message(
            content=message_content, role=role, session_id=session_id_override or self.id
        )
        self.messages.append(new_message)
        self.updated_at = datetime.now(timezone.utc)
        return new_message

    def add_context_item(self, item: ContextItem) -> None:
        """Adds a ContextItem to the session's context pool and updates timestamp. (Implementation unchanged)"""
        self.context_items = [ci for ci in self.context_items if ci.id != item.id]
        self.context_items.append(item)
        self.context_items.sort(key=lambda x: x.timestamp)
        self.updated_at = datetime.now(timezone.utc)

    def remove_context_item(self, item_id: str) -> bool:
        """Removes a ContextItem from the pool by its ID. (Implementation unchanged)"""
        initial_len = len(self.context_items)
        self.context_items = [ci for ci in self.context_items if ci.id != item_id]
        if len(self.context_items) < initial_len:
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def get_context_item(self, item_id: str) -> Optional[ContextItem]:
        """Gets a specific context item by its ID. (Implementation unchanged)"""
        for item in self.context_items:
            if item.id == item_id:
                return item
        return None


class ContextDocument(BaseModel):
    """
    Represents a document used for context, typically in Retrieval Augmented Generation (RAG).
    (Implementation unchanged)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the context document.")
    content: str = Field(description="The textual content of the document.")
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding of the document content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional document metadata (e.g., source, title).")
    score: Optional[float] = Field(default=None, description="Optional relevance score from a similarity search.")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


# New Model for ContextManager's prepare_context output
class ContextPreparationDetails(BaseModel):
    """
    Structured output from ContextManager.prepare_context, providing detailed
    information about the context that would be sent to the LLM and how it was formed.
    """
    prepared_messages: List[Message] = Field(description="The final list of messages prepared for the LLM.")
    final_token_count: int = Field(description="The total token count of the prepared_messages.")
    max_tokens_for_model: int = Field(description="The maximum context token limit for the target model.")

    rag_documents_used: Optional[List[ContextDocument]] = Field(default=None, description="List of RAG documents included in the context, if RAG was enabled and successful.")
    rendered_rag_template_content: Optional[str] = Field(default=None, description="The content of the user's query after being rendered with the RAG prompt template (if RAG was active).")

    truncation_actions_taken: Dict[str, Any] = Field(default_factory=dict, description="Details about any truncation performed on context components. E.g., {'history_chat_removed_count': 2, 'user_items_active_removed_ids': ['id1']}")

    # Optional: Could add more detailed token counts per component before/after truncation if needed
    # component_tokens_before_truncation: Dict[str, int] = Field(default_factory=dict)
    # component_tokens_after_truncation: Dict[str, int] = Field(default_factory=dict)

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        # Allow arbitrary types for truncation_actions_taken if complex objects are stored there
        # arbitrary_types_allowed = True
