# src/llmcore/storage/base_session.py
"""
Abstract Base Class for Session Storage backends.

This module defines the interface that all session storage implementations
must adhere to within the LLMCore library.
"""

import abc
from typing import List, Optional, Dict, Any

# Import ChatSession for type hinting
# Use a forward reference ('ChatSession') if needed to avoid circular imports,
# although direct import should be fine here.
from ..models import ChatSession


class BaseSessionStorage(abc.ABC):
    """
    Abstract Base Class for chat session storage.

    Defines the standard methods required for managing the persistence
    of ChatSession objects. Concrete implementations will handle the specifics
    of storing data (e.g., in JSON files, SQLite, PostgreSQL).
    """

    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the storage backend with given configuration.

        This method should be called asynchronously when the storage is first
        instantiated or used. It sets up necessary resources like database
        connections or directories.

        Args:
            config: Backend-specific configuration dictionary derived from
                    the main LLMCore configuration (e.g., path, db_url).
        """
        pass

    @abc.abstractmethod
    async def save_session(self, session: ChatSession) -> None:
        """
        Save or update a chat session in the storage.

        If the session already exists (based on its ID), it should be updated.
        If it's a new session, it should be created.

        Args:
            session: The ChatSession object to save.
        """
        pass

    @abc.abstractmethod
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a specific chat session by its unique ID.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found, otherwise None.
        """
        pass

    @abc.abstractmethod
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available persistent chat sessions, returning metadata only.

        This method should return a list of dictionaries, each containing
        basic information about a session (e.g., id, name, created_at,
        updated_at, message_count) but *not* the full message history,
        for performance reasons.

        Returns:
            A list of dictionaries, each representing session metadata.
            Example: [{'id': 'abc', 'name': 'Session 1', 'updated_at': '...', 'message_count': 5}]
        """
        pass

    @abc.abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific chat session from storage by its ID.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session was found and deleted successfully, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Clean up resources used by the storage backend.

        This method should be called asynchronously when the application
        is shutting down to close database connections, file handles, etc.
        """
        pass
