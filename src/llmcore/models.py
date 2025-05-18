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

from pydantic import BaseModel, Field, field_validator


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
    HISTORY_MESSAGE = "history_message" # Represents a message from the chat history (less used directly as ContextItem)
    USER_TEXT = "user_text"             # User-added arbitrary text snippet
    USER_FILE = "user_file"             # Content from a user-added file
    RAG_SNIPPET = "rag_snippet"         # A snippet retrieved from RAG and pinned by the user


class ContextItem(BaseModel):
    """
    Represents an individual item that can be part of the LLM's context pool.
    This can include user-provided text, file contents, or RAG snippets pinned by the user.

    Attributes:
        id: Unique identifier for this context item.
        type: The type of context item (e.g., user_text, user_file, rag_snippet).
        source_id: Optional ID linking back to the original source (e.g., file path for user_file,
                   or original RAG document ID for rag_snippet).
        content: The textual content of the item.
        tokens: Estimated or actual token count for the content (potentially after truncation).
        original_tokens: Optional original token count for the content before any truncation by ContextManager.
        metadata: Additional metadata (e.g., filename for USER_FILE, source for RAG_SNIPPET).
        timestamp: Timestamp of creation or relevance for ordering.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the context item.")
    type: ContextItemType = Field(description="Type of the context item.")
    source_id: Optional[str] = Field(default=None, description="Identifier of the original source (e.g., file path, original RAG doc ID).")
    content: str = Field(description="Textual content of the context item.")
    tokens: Optional[int] = Field(default=None, description="Token count for the content, possibly after truncation.")
    original_tokens: Optional[int] = Field(default=None, description="Original token count before any truncation by ContextManager.") # New field
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., filename, RAG source info).")
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

class ChatSession(BaseModel):
    """
    Represents a single conversation or chat session.

    Attributes:
        id: Unique identifier for the chat session.
        name: Optional human-readable name for the session.
        messages: List of `Message` objects (chat history).
        context_items: List of user-added `ContextItem` objects (e.g., text snippets, files, pinned RAG snippets).
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
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def ensure_utc_timestamps(cls, v: Any) -> datetime:
        """Ensure created_at and updated_at timestamps are timezone-aware and in UTC."""
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

    def add_message(self, message_content: str, role: Role, session_id_override: Optional[str] = None) -> Message:
        """
        Creates a new message, adds it to the session, and updates the `updated_at` timestamp.

        Args:
            message_content: The content of the message.
            role: The role of the message sender.
            session_id_override: Optionally override the session_id for the message.
                                 Defaults to this session's ID.

        Returns:
            The created Message object.
        """
        new_message = Message(
            content=message_content,
            role=role,
            session_id=session_id_override or self.id
        )
        self.messages.append(new_message)
        self.updated_at = datetime.now(timezone.utc)
        return new_message

    def add_context_item(self, item: ContextItem) -> None:
        """Adds a ContextItem to the session's context pool and updates timestamp."""
        # Ensure item ID is unique within this session's context_items
        # Remove existing item with the same ID before adding the new one (effectively an upsert)
        self.context_items = [ci for ci in self.context_items if ci.id != item.id]
        self.context_items.append(item)
        self.context_items.sort(key=lambda x: x.timestamp) # Keep them sorted if desired
        self.updated_at = datetime.now(timezone.utc)

    def remove_context_item(self, item_id: str) -> bool:
        """Removes a ContextItem from the pool by its ID."""
        initial_len = len(self.context_items)
        self.context_items = [ci for ci in self.context_items if ci.id != item_id]
        if len(self.context_items) < initial_len:
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def get_context_item(self, item_id: str) -> Optional[ContextItem]:
        """Gets a specific context item by its ID."""
        for item in self.context_items:
            if item.id == item_id:
                return item
        return None


class ContextDocument(BaseModel):
    """
    Represents a document used for context, typically in Retrieval Augmented Generation (RAG).

    This model holds the content of a document that might be retrieved from a vector store
    and used to augment the context provided to an LLM.

    Attributes:
        id: A unique identifier for the document.
        content: The textual content of the document.
        embedding: An optional list of floats representing the vector embedding.
        metadata: An optional dictionary for storing additional information.
        score: An optional float indicating the relevance score from similarity search.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the context document.")
    content: str = Field(description="The textual content of the document.")
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding of the document content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional document metadata (e.g., source, title).")
    score: Optional[float] = Field(default=None, description="Optional relevance score from a similarity search.")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
