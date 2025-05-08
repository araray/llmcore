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
from typing import Any, Dict, List, Optional

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
                    This is crucial for linking messages to their respective sessions,
                    especially when messages might be stored or processed independently.
        role: The role of the entity that produced the message (e.g., system, user, assistant).
        content: The textual content of the message.
        timestamp: The date and time when the message was created or recorded.
                   Defaults to the current UTC time if not provided.
        tokens: An optional count of tokens for the message content, useful for context management.
                This is often calculated by a specific LLM provider's tokenizer.
        metadata: An optional dictionary for storing additional, unstructured information
                  related to the message (e.g., source, processing details, client info).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the message.")
    session_id: str = Field(description="Identifier of the chat session this message belongs to.")
    role: Role = Field(description="The role of the message sender (system, user, or assistant).")
    content: str = Field(description="The textual content of the message.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the message was created (UTC).")
    tokens: Optional[int] = Field(default=None, description="Optional token count for the message content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional message metadata.") # Changed from Optional[Dict] to Dict

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True # Ensures enum values are used in serialization
        validate_assignment = True # Re-validate on assignment
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z') # Ensure datetime is ISO Z-formatted
        }

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            # Handle 'Z' for UTC and ensure timezone info
            if v.endswith('Z'):
                v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
            else:
                v_parsed = datetime.fromisoformat(v)

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc) # Assume UTC if naive
            return v.astimezone(timezone.utc) # Convert to UTC
        # If default_factory is used, it will already be timezone-aware UTC
        if v is None: # Should be handled by default_factory, but as a safeguard
            return datetime.now(timezone.utc)
        return v


class ChatSession(BaseModel):
    """
    Represents a single conversation or chat session.

    A chat session typically consists of a sequence of messages exchanged between
    a user, an assistant, and potentially system directives.

    Attributes:
        id: A unique identifier for the chat session.
        name: An optional human-readable name for the session (e.g., "Project X Planning").
              If not provided, a default name might be generated.
        messages: A list of `Message` objects that constitute the conversation history,
                  ordered chronologically.
        created_at: The date and time when the session was initiated.
                    Defaults to the current UTC time.
        updated_at: The date and time when the session was last modified (e.g., a new message added).
                    Defaults to the current UTC time and should be updated on modifications.
        metadata: An optional dictionary for storing additional, unstructured information
                  about the session (e.g., user ID, context tags, LLM model used).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the chat session.")
    name: Optional[str] = Field(default=None, description="Optional human-readable name for the session.")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation, ordered chronologically.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the session was created (UTC).")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the session was last updated (UTC).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional session metadata.") # Changed from Optional[Dict] to Dict

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z') # Ensure datetime is ISO Z-formatted
        }

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def ensure_utc_timestamps(cls, v: Any) -> datetime:
        """Ensure created_at and updated_at timestamps are timezone-aware and in UTC."""
        if isinstance(v, str):
            if v.endswith('Z'):
                v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
            else:
                v_parsed = datetime.fromisoformat(v)

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None: # Should be handled by default_factory
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
            session_id=session_id_override or self.id # Use current session ID by default
        )
        self.messages.append(new_message)
        self.updated_at = datetime.now(timezone.utc)
        return new_message


class ContextDocument(BaseModel):
    """
    Represents a document used for context, typically in Retrieval Augmented Generation (RAG).

    This model holds the content of a document that might be retrieved from a vector store
    and used to augment the context provided to an LLM.

    Attributes:
        id: A unique identifier for the document. This could be its ID from a vector store
            or any other unique reference.
        content: The textual content of the document.
        embedding: An optional list of floats representing the vector embedding of the document's content.
                   This is often used for similarity searches but might not always be needed
                   after retrieval if the content itself is the primary focus.
        metadata: An optional dictionary for storing additional information about the document,
                  such as its source, title, author, creation date, or relevance score.
        score: An optional float indicating the relevance score of the document,
               typically assigned by a similarity search algorithm during retrieval.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the context document.")
    content: str = Field(description="The textual content of the document.")
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding of the document content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional document metadata (e.g., source, title).") # Changed from Optional[Dict] to Dict
    score: Optional[float] = Field(default=None, description="Optional relevance score from a similarity search.")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
