# src/llmcore/storage/json_session.py
"""
JSON file-based storage for ChatSession objects.

This module implements the BaseSessionStorage interface, storing each
chat session as a separate JSON file in a specified directory.
It uses aiofiles for asynchronous file operations.
"""

import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import aiofiles
import aiofiles.os as aios

from ..exceptions import ConfigError, SessionStorageError
from ..models import (ChatSession,  # Added ContextItem for validation
                      ContextItem)
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)


class JsonSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession objects in JSON files.

    Each session is stored as a separate JSON file in the configured
    storage directory. File operations are performed asynchronously.
    Includes storage and retrieval of `context_items` within the session JSON.
    """
    _storage_dir: pathlib.Path
    _file_extension: str

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the JSON session storage.

        Creates the storage directory if it doesn't exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'path': The directory path for storing session files.
                    'file_extension' (optional): Extension for session files (default: '.json').

        Raises:
            ConfigError: If the 'path' is not provided in the config.
            SessionStorageError: If the storage directory cannot be created.
        """
        storage_path_str = config.get("path")
        if not storage_path_str:
            raise ConfigError("JSON session storage 'path' not specified in configuration.")

        self._storage_dir = pathlib.Path(os.path.expanduser(storage_path_str))
        self._file_extension = config.get("file_extension", ".json")
        if not self._file_extension.startswith('.'):
            self._file_extension = f".{self._file_extension}"

        try:
            await aios.makedirs(self._storage_dir, exist_ok=True)
            logger.info(f"JSON session storage initialized at: {self._storage_dir.resolve()}")
        except OSError as e:
            logger.error(f"Failed to create JSON storage directory at {self._storage_dir}: {e}")
            raise SessionStorageError(f"Could not create storage directory: {e}")

    def _get_session_path(self, session_id: str) -> pathlib.Path:
        """
        Construct the file path for a given session ID.

        Args:
            session_id: The unique identifier of the session.

        Returns:
            The pathlib.Path object for the session's JSON file.
        """
        return self._storage_dir / f"{session_id}{self._file_extension}"

    async def save_session(self, session: ChatSession) -> None:
        """
        Save or update a chat session to a JSON file asynchronously.

        The session object, including its `messages` and `context_items`,
        is serialized to JSON and written to a file named after its ID.

        Args:
            session: The ChatSession object to save.

        Raises:
            SessionStorageError: If an error occurs during file writing or JSON serialization.
        """
        session_file_path = self._get_session_path(session.id)
        try:
            # Pydantic's model_dump_json handles datetime and enum serialization correctly.
            # This will now include 'context_items' as part of the ChatSession model.
            session_json = session.model_dump_json(indent=2)
            async with aiofiles.open(session_file_path, mode="w", encoding="utf-8") as f:
                await f.write(session_json)
            logger.debug(f"Session '{session.id}' with {len(session.messages)} messages and {len(session.context_items)} context items saved to {session_file_path}")
        except TypeError as e:
            logger.error(f"Error serializing session '{session.id}' to JSON: {e}")
            raise SessionStorageError(f"Failed to serialize session data for '{session.id}': {e}")
        except IOError as e:
            logger.error(f"Error writing session '{session.id}' to file {session_file_path}: {e}")
            raise SessionStorageError(f"Failed to write session file for '{session.id}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving session '{session.id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error saving session '{session.id}': {e}")

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a specific chat session by its ID asynchronously.

        Reads the corresponding JSON file and deserializes it into a
        ChatSession object, including `messages` and `context_items`.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found, otherwise None.

        Raises:
            SessionStorageError: If an error occurs during file reading or JSON/Pydantic deserialization.
        """
        session_file_path = self._get_session_path(session_id)
        try:
            if not await aios.path.exists(session_file_path):
                logger.debug(f"Session file not found for ID '{session_id}' at {session_file_path}")
                return None

            async with aiofiles.open(session_file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()

            session_data = json.loads(content)
            # Validate the entire session data, including context_items, using Pydantic
            # Pydantic will automatically parse nested models like Message and ContextItem
            chat_session = ChatSession.model_validate(session_data)

            logger.debug(f"Session '{session_id}' loaded from {session_file_path} with {len(chat_session.messages)} messages and {len(chat_session.context_items)} context items.")
            return chat_session
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for session '{session_id}' from {session_file_path}: {e}")
            raise SessionStorageError(f"Corrupted session file for '{session_id}': {e}")
        except IOError as e:
            logger.error(f"Error reading session file '{session_file_path}' for session '{session_id}': {e}")
            raise SessionStorageError(f"Failed to read session file for '{session_id}': {e}")
        except Exception as e: # Catches Pydantic validation errors (e.g., pydantic.ValidationError)
            logger.error(f"An unexpected error or validation error occurred while loading session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error or invalid data loading session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available persistent chat sessions, returning metadata only.

        Scans the storage directory for session files and extracts metadata
        (id, name, created_at, updated_at, message_count, context_item_count) from each.

        Returns:
            A list of dictionaries, each representing session metadata.

        Raises:
            SessionStorageError: If the storage directory cannot be accessed.
        """
        session_metadata_list: List[Dict[str, Any]] = []
        try:
            if not await aios.path.isdir(self._storage_dir):
                logger.warning(f"Storage directory {self._storage_dir} does not exist or is not a directory.")
                return []

            for filename in await aios.listdir(self._storage_dir):
                if filename.endswith(self._file_extension):
                    session_file_path = self._storage_dir / filename
                    try:
                        async with aiofiles.open(session_file_path, mode="r", encoding="utf-8") as f:
                            # Read only enough to get metadata. A full load is needed for context_items_count
                            # For simplicity, we load the whole file, but this could be optimized
                            # if session files become extremely large and metadata is at the top.
                            content = await f.read()
                            data = json.loads(content)

                        session_id_from_file = data.get("id")
                        if not session_id_from_file:
                            logger.warning(f"Session file {filename} is missing an 'id' field. Skipping.")
                            continue

                        metadata = {
                            "id": session_id_from_file,
                            "name": data.get("name"),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "message_count": len(data.get("messages", [])),
                            "context_item_count": len(data.get("context_items", [])), # Add count of context items
                            "metadata": data.get("metadata", {}),
                        }
                        session_metadata_list.append(metadata)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from session file {filename}. Skipping.")
                    except Exception as e:
                        logger.warning(f"Error processing session file {filename}: {e}. Skipping.")

            session_metadata_list.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
            logger.debug(f"Found {len(session_metadata_list)} sessions in {self._storage_dir}")

        except OSError as e:
            logger.error(f"Error listing session files in {self._storage_dir}: {e}")
            raise SessionStorageError(f"Could not list sessions: {e}")

        return session_metadata_list

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific chat session file asynchronously.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session file was found and deleted, False otherwise.

        Raises:
            SessionStorageError: If an error occurs during file deletion.
        """
        session_file_path = self._get_session_path(session_id)
        try:
            if await aios.path.exists(session_file_path):
                await aios.remove(session_file_path)
                logger.info(f"Session '{session_id}' deleted from {session_file_path}")
                return True
            else:
                logger.warning(f"Attempted to delete non-existent session '{session_id}' at {session_file_path}")
                return False
        except OSError as e:
            logger.error(f"Error deleting session file {session_file_path} for session '{session_id}': {e}")
            raise SessionStorageError(f"Failed to delete session file for '{session_id}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error deleting session '{session_id}': {e}")

    async def close(self) -> None:
        """
        Clean up resources. For JSON storage, no explicit closing action is typically needed.
        """
        logger.debug("JSONSessionStorage closed (no specific action needed).")
        pass
