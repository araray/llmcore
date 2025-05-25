# src/llmcore/storage/base_session.py
"""
Abstract Base Class for Session Storage backends.

This module defines the interface that all session storage implementations
must adhere to within the LLMCore library. It now also includes
methods for managing Context Presets.
"""

import abc
from typing import Any, Dict, List, Optional

# Import ChatSession and ContextPreset for type hinting
from ..models import ChatSession, ContextPreset


class BaseSessionStorage(abc.ABC):
    """
    Abstract Base Class for chat session and context preset storage.

    Defines the standard methods required for managing the persistence
    of ChatSession objects and ContextPreset objects. Concrete implementations
    will handle the specifics of storing data (e.g., in JSON files, SQLite, PostgreSQL).
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

    # --- New methods for Context Preset Management ---

    @abc.abstractmethod
    async def save_context_preset(self, preset: ContextPreset) -> None:
        """
        Save or update a context preset in the storage.

        If a preset with the same name already exists, it should be updated.
        If it's a new preset, it should be created.

        Args:
            preset: The ContextPreset object to save.
        """
        pass

    @abc.abstractmethod
    async def get_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """
        Retrieve a specific context preset by its unique name.

        Args:
            preset_name: The name of the context preset to retrieve.

        Returns:
            The ContextPreset object if found, otherwise None.
        """
        pass

    @abc.abstractmethod
    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """
        List available context presets, returning metadata only.

        This method should return a list of dictionaries, each containing
        basic information about a preset (e.g., name, description, item_count,
        created_at, updated_at).

        Returns:
            A list of dictionaries, each representing preset metadata.
            Example: [{'name': 'my_debug_setup', 'description': 'Preset for debugging', 'item_count': 3, 'updated_at': '...'}]
        """
        pass

    @abc.abstractmethod
    async def delete_context_preset(self, preset_name: str) -> bool:
        """
        Delete a specific context preset from storage by its name.

        Args:
            preset_name: The name of the context preset to delete.

        Returns:
            True if the preset was found and deleted successfully, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        """
        Rename an existing context preset.

        Args:
            old_name: The current name of the preset.
            new_name: The new name for the preset.

        Returns:
            True if the preset was found and renamed successfully, False otherwise
            (e.g., if old_name doesn't exist or new_name already exists).

        Raises:
            ValueError: If new_name is invalid (e.g., contains forbidden characters).
            StorageError: For other storage-related issues during rename.
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
