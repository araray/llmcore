# src/llmcore/storage/json_session.py
"""
JSON file-based storage for ChatSession objects and ContextPreset objects.

This module implements the BaseSessionStorage interface, storing each
chat session as a separate JSON file in a specified directory, and each
context preset in a 'context_presets' subdirectory.
It uses aiofiles for asynchronous file operations.
"""

import json
import logging
import os
import pathlib
import re # For validating preset names as filenames
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiofiles
import aiofiles.os as aios

from ..exceptions import ConfigError, SessionStorageError, StorageError
from ..models import (ChatSession, ContextItem, ContextItemType, Message, Role, ContextPreset, ContextPresetItem, Episode, EpisodeType)
from .base_session import BaseSessionStorage

logger = logging.getLogger(__name__)


class JsonSessionStorage(BaseSessionStorage):
    """
    Manages persistence of ChatSession and ContextPreset objects in JSON files.

    Each session is stored as a separate JSON file in the configured
    storage directory. Each context preset is stored in a 'context_presets'
    subdirectory within that main storage directory.
    File operations are performed asynchronously.
    """
    _storage_dir: pathlib.Path
    _presets_dir: pathlib.Path # Directory for context presets
    _episodes_dir: pathlib.Path # Directory for episodes
    _file_extension: str
    _presets_dir_name: str = "context_presets" # Standardized subdirectory name
    _episodes_dir_name: str = "episodes" # Standardized subdirectory name for episodes

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the JSON session and preset storage.

        Creates the main storage directory and subdirectories for context presets
        and episodes if they don't exist.

        Args:
            config: Configuration dictionary. Expected keys:
                    'path': The directory path for storing session files.
                    'file_extension' (optional): Extension for session/preset files (default: '.json').

        Raises:
            ConfigError: If the 'path' is not provided in the config.
            SessionStorageError: If storage directories cannot be created.
        """
        storage_path_str = config.get("path")
        if not storage_path_str:
            raise ConfigError("JSON session storage 'path' not specified in configuration.")

        self._storage_dir = pathlib.Path(os.path.expanduser(storage_path_str))
        self._presets_dir = self._storage_dir / self._presets_dir_name
        self._episodes_dir = self._storage_dir / self._episodes_dir_name
        self._file_extension = config.get("file_extension", ".json")
        if not self._file_extension.startswith('.'):
            self._file_extension = f".{self._file_extension}"

        try:
            await aios.makedirs(self._storage_dir, exist_ok=True)
            await aios.makedirs(self._presets_dir, exist_ok=True) # Create presets subdirectory
            await aios.makedirs(self._episodes_dir, exist_ok=True) # Create episodes subdirectory
            logger.info(f"JSON session storage initialized at: {self._storage_dir.resolve()}")
            logger.info(f"JSON context preset storage initialized at: {self._presets_dir.resolve()}")
            logger.info(f"JSON episode storage initialized at: {self._episodes_dir.resolve()}")
        except OSError as e:
            logger.error(f"Failed to create JSON storage directories (main: {self._storage_dir}, presets: {self._presets_dir}, episodes: {self._episodes_dir}): {e}")
            raise SessionStorageError(f"Could not create storage directories: {e}")

    def _get_session_path(self, session_id: str) -> pathlib.Path:
        """Constructs the file path for a given session ID."""
        return self._storage_dir / f"{session_id}{self._file_extension}"

    def _get_preset_path(self, preset_name: str) -> pathlib.Path:
        """Constructs the file path for a given context preset name."""
        # Basic sanitization for filename from preset_name, though Pydantic model should validate 'name'
        sane_filename = re.sub(r'[^\w\-. ]', '_', preset_name)
        return self._presets_dir / f"{sane_filename}{self._file_extension}"

    def _get_episode_path(self, session_id: str) -> pathlib.Path:
        """Constructs the file path for episodes of a given session ID using JSON Lines format."""
        return self._episodes_dir / f"{session_id}_episodes.jsonl"

    async def save_session(self, session: ChatSession) -> None:
        """
        Save or update a chat session to a JSON file asynchronously.

        Args:
            session: The ChatSession object to save.

        Raises:
            SessionStorageError: If serialization or file I/O fails.
        """
        session_file_path = self._get_session_path(session.id)
        try:
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
        Retrieve a specific chat session by its ID asynchronously from its JSON file.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found, otherwise None.

        Raises:
            SessionStorageError: If the file is corrupted or another I/O or validation error occurs.
        """
        session_file_path = self._get_session_path(session_id)
        try:
            if not await aios.path.exists(session_file_path):
                logger.debug(f"Session file not found for ID '{session_id}' at {session_file_path}")
                return None
            async with aiofiles.open(session_file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
            session_data = json.loads(content)
            chat_session = ChatSession.model_validate(session_data)
            logger.debug(f"Session '{session_id}' loaded from {session_file_path} with {len(chat_session.messages)} messages and {len(chat_session.context_items)} context items.")
            return chat_session
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for session '{session_id}' from {session_file_path}: {e}")
            raise SessionStorageError(f"Corrupted session file for '{session_id}': {e}")
        except IOError as e:
            logger.error(f"Error reading session file '{session_file_path}' for session '{session_id}': {e}")
            raise SessionStorageError(f"Failed to read session file for '{session_id}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error or validation error occurred while loading session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Unexpected error or invalid data loading session '{session_id}': {e}")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List available persistent chat sessions, returning metadata only from each JSON file.

        Returns:
            A list of dictionaries, each representing session metadata.

        Raises:
            SessionStorageError: If there's an error listing files in the storage directory.
        """
        session_metadata_list: List[Dict[str, Any]] = []
        try:
            if not await aios.path.isdir(self._storage_dir):
                logger.warning(f"Storage directory {self._storage_dir} does not exist or is not a directory.")
                return []
            for filename in await aios.listdir(self._storage_dir):
                if filename.endswith(self._file_extension) and not (self._storage_dir / filename).is_dir(): # Ensure it's a file, not preset dir
                    session_file_path = self._storage_dir / filename
                    try:
                        async with aiofiles.open(session_file_path, mode="r", encoding="utf-8") as f:
                            content = await f.read()
                            data = json.loads(content)
                        session_id_from_file = data.get("id")
                        if not session_id_from_file:
                            logger.warning(f"Session file {filename} is missing an 'id' field. Skipping.")
                            continue
                        metadata = {
                            "id": session_id_from_file, "name": data.get("name"),
                            "created_at": data.get("created_at"), "updated_at": data.get("updated_at"),
                            "message_count": len(data.get("messages", [])),
                            "context_item_count": len(data.get("context_items", [])),
                            "metadata": data.get("metadata", {}),
                        }
                        session_metadata_list.append(metadata)
                    except json.JSONDecodeError: logger.warning(f"Could not decode JSON from session file {filename}. Skipping.")
                    except Exception as e: logger.warning(f"Error processing session file {filename}: {e}. Skipping.")
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
            SessionStorageError: If there's an error during the file deletion.
        """
        session_file_path = self._get_session_path(session_id)
        try:
            if await aios.path.exists(session_file_path):
                await aios.remove(session_file_path)
                logger.info(f"Session '{session_id}' deleted from {session_file_path}")

                # Also delete associated episodes file if it exists
                episode_file_path = self._get_episode_path(session_id)
                if await aios.path.exists(episode_file_path):
                    await aios.remove(episode_file_path)
                    logger.debug(f"Episodes file for session '{session_id}' deleted from {episode_file_path}")

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

    async def update_session_name(self, session_id: str, new_name: str) -> bool:
        """
        Updates the human-readable name of an existing session by loading it,
        modifying the name, and saving it back to its JSON file.

        Args:
            session_id: The ID of the session to update.
            new_name: The new name for the session.

        Returns:
            True if the session was found and updated successfully, False otherwise.

        Raises:
            SessionStorageError: If loading or saving the session fails.
        """
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot update name: session '{session_id}' not found.")
            return False

        session.name = new_name
        session.updated_at = datetime.now(timezone.utc)
        try:
            await self.save_session(session)
            logger.info(f"Session '{session_id}' name updated to '{new_name}'.")
            return True
        except SessionStorageError as e:
            logger.error(f"Failed to save session '{session_id}' after updating name: {e}")
            # Re-raise the original storage error
            raise

    # --- New methods for Context Preset Management ---

    async def save_context_preset(self, preset: ContextPreset) -> None:
        """
        Save or update a context preset to a JSON file in the presets subdirectory.
        The file is named after the preset's `name`.

        Args:
            preset: The ContextPreset object to save.

        Raises:
            StorageError: If an error occurs during file writing or JSON serialization.
            ValueError: If preset name is invalid for a filename (should be caught by Pydantic model).
        """
        if not preset.name: # Should be caught by Pydantic validation if name is required
            raise ValueError("Preset name cannot be empty.")
        # Pydantic model_validator for ContextPreset.name should have already validated it.

        preset_file_path = self._get_preset_path(preset.name)
        try:
            preset_json = preset.model_dump_json(indent=2)
            async with aiofiles.open(preset_file_path, mode="w", encoding="utf-8") as f:
                await f.write(preset_json)
            logger.info(f"Context preset '{preset.name}' saved to {preset_file_path}")
        except TypeError as e:
            logger.error(f"Error serializing context preset '{preset.name}' to JSON: {e}")
            raise StorageError(f"Failed to serialize preset data for '{preset.name}': {e}")
        except IOError as e:
            logger.error(f"Error writing context preset '{preset.name}' to file {preset_file_path}: {e}")
            raise StorageError(f"Failed to write preset file for '{preset.name}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving preset '{preset.name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error saving preset '{preset.name}': {e}")

    async def get_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        """
        Retrieve a specific context preset by its name.

        Args:
            preset_name: The name of the context preset to retrieve.

        Returns:
            The ContextPreset object if found, otherwise None.

        Raises:
            StorageError: If an error occurs during file reading or JSON/Pydantic deserialization.
        """
        preset_file_path = self._get_preset_path(preset_name)
        try:
            if not await aios.path.exists(preset_file_path):
                logger.debug(f"Context preset file not found for name '{preset_name}' at {preset_file_path}")
                return None

            async with aiofiles.open(preset_file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()

            preset_data = json.loads(content)
            context_preset = ContextPreset.model_validate(preset_data)
            logger.debug(f"Context preset '{preset_name}' loaded from {preset_file_path} with {len(context_preset.items)} items.")
            return context_preset
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for context preset '{preset_name}' from {preset_file_path}: {e}")
            raise StorageError(f"Corrupted context preset file for '{preset_name}': {e}")
        except IOError as e:
            logger.error(f"Error reading context preset file '{preset_file_path}' for preset '{preset_name}': {e}")
            raise StorageError(f"Failed to read preset file for '{preset_name}': {e}")
        except Exception as e: # Catches Pydantic validation errors
            logger.error(f"An unexpected error or validation error occurred while loading preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error or invalid data loading preset '{preset_name}': {e}")

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        """
        List available context presets, returning metadata only.
        Scans the presets subdirectory for JSON files.

        Returns:
            A list of dictionaries, each representing preset metadata.
        """
        preset_metadata_list: List[Dict[str, Any]] = []
        try:
            if not await aios.path.isdir(self._presets_dir):
                logger.warning(f"Context presets directory {self._presets_dir} does not exist.")
                return []

            for filename in await aios.listdir(self._presets_dir):
                if filename.endswith(self._file_extension):
                    preset_file_path = self._presets_dir / filename
                    try:
                        async with aiofiles.open(preset_file_path, mode="r", encoding="utf-8") as f:
                            content = await f.read()
                            data = json.loads(content)

                        preset_name_from_file = data.get("name")
                        if not preset_name_from_file:
                             logger.warning(f"Preset file {filename} is missing a 'name' field. Skipping.")
                             continue

                        metadata = {
                            "name": preset_name_from_file,
                            "description": data.get("description"),
                            "item_count": len(data.get("items", [])),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "metadata": data.get("metadata", {}), # Preset-level metadata
                        }
                        preset_metadata_list.append(metadata)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from preset file {filename}. Skipping.")
                    except Exception as e:
                        logger.warning(f"Error processing preset file {filename}: {e}. Skipping.")

            preset_metadata_list.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
            logger.debug(f"Found {len(preset_metadata_list)} context presets in {self._presets_dir}")

        except OSError as e:
            logger.error(f"Error listing context preset files in {self._presets_dir}: {e}")
            raise StorageError(f"Could not list context presets: {e}")

        return preset_metadata_list

    async def delete_context_preset(self, preset_name: str) -> bool:
        """
        Delete a specific context preset file.

        Args:
            preset_name: The name of the context preset to delete.

        Returns:
            True if the preset file was found and deleted, False otherwise.
        """
        preset_file_path = self._get_preset_path(preset_name)
        try:
            if await aios.path.exists(preset_file_path):
                await aios.remove(preset_file_path)
                logger.info(f"Context preset '{preset_name}' deleted from {preset_file_path}")
                return True
            else:
                logger.warning(f"Attempted to delete non-existent context preset '{preset_name}' at {preset_file_path}")
                return False
        except OSError as e:
            logger.error(f"Error deleting context preset file {preset_file_path} for preset '{preset_name}': {e}")
            raise StorageError(f"Failed to delete preset file for '{preset_name}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while deleting preset '{preset_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error deleting preset '{preset_name}': {e}")

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        """
        Rename an existing context preset.
        This involves loading the preset, updating its 'name' attribute,
        saving it under the new name, and then deleting the old file.

        Args:
            old_name: The current name of the preset.
            new_name: The new name for the preset.

        Returns:
            True if successful, False otherwise.

        Raises:
            ValueError: If new_name is invalid (delegated to Pydantic model validation).
            StorageError: For other storage-related issues.
        """
        if old_name == new_name:
            logger.info(f"Attempted to rename preset '{old_name}' to the same name. No action taken.")
            return True # Or False, depending on desired idempotency semantics. True seems reasonable.

        old_path = self._get_preset_path(old_name)
        new_path = self._get_preset_path(new_name)

        # Validate new_name using a temporary ContextPreset object (Pydantic validator will run)
        try:
            ContextPreset(name=new_name, items=[]) # Minimal valid preset for name validation
        except ValueError as ve: # Pydantic ValidationError is a subclass of ValueError
            logger.error(f"Invalid new preset name '{new_name}': {ve}")
            raise # Re-raise the ValueError from Pydantic

        if not await aios.path.exists(old_path):
            logger.warning(f"Cannot rename: preset '{old_name}' not found at {old_path}.")
            return False
        if await aios.path.exists(new_path):
            logger.warning(f"Cannot rename: preset with new name '{new_name}' already exists at {new_path}.")
            return False

        try:
            preset_to_rename = await self.get_context_preset(old_name)
            if not preset_to_rename: # Should be caught by exists check, but defensive
                logger.error(f"Preset '{old_name}' disappeared before rename operation could complete.")
                return False

            # Update the preset object's attributes
            preset_to_rename.name = new_name
            preset_to_rename.updated_at = datetime.now(timezone.utc)

            # Save under new name
            await self.save_context_preset(preset_to_rename) # This uses preset_to_rename.name (which is new_name)

            # If save was successful, delete the old file
            await aios.remove(old_path)
            logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}'.")
            return True

        except StorageError as se: # Catch errors from get_context_preset or save_context_preset
            logger.error(f"Storage error during rename of preset '{old_name}' to '{new_name}': {se}", exc_info=True)
            # Attempt to clean up if new file was created but old one not deleted (or vice-versa)
            # This is complex to make fully atomic with just file operations.
            # For now, if save_context_preset failed, new_path might not exist.
            # If aios.remove(old_path) failed after new save, we might have both.
            if await aios.path.exists(new_path) and not await aios.path.exists(old_path):
                 logger.warning(f"Rename partially failed: new preset '{new_name}' saved, but old '{old_name}' might still exist if deletion failed after save.")
            raise # Re-raise the original storage error
        except OSError as e:
            logger.error(f"OS error during rename of preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            raise StorageError(f"OS error during preset rename: {e}")
        except Exception as e:
            logger.error(f"Unexpected error renaming preset '{old_name}' to '{new_name}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error during preset rename: {e}")

    # --- New methods for Episodic Memory Management ---

    async def add_episode(self, episode: Episode) -> None:
        """
        Adds a new episode to the episodic memory log for a session.
        Episodes are stored in JSON Lines format for efficient append operations.

        Args:
            episode: The Episode object to add.

        Raises:
            StorageError: If an error occurs during file writing or JSON serialization.
        """
        episode_file_path = self._get_episode_path(episode.session_id)
        try:
            episode_json = episode.model_dump_json()
            async with aiofiles.open(episode_file_path, mode="a", encoding="utf-8") as f:
                await f.write(episode_json + "\n")
            logger.debug(f"Episode '{episode.episode_id}' for session '{episode.session_id}' appended to {episode_file_path}")
        except TypeError as e:
            logger.error(f"Error serializing episode '{episode.episode_id}' to JSON: {e}")
            raise StorageError(f"Failed to serialize episode data for '{episode.episode_id}': {e}")
        except IOError as e:
            logger.error(f"Error writing episode '{episode.episode_id}' to file {episode_file_path}: {e}")
            raise StorageError(f"Failed to write episode file for '{episode.episode_id}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving episode '{episode.episode_id}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error saving episode '{episode.episode_id}': {e}")

    async def get_episodes(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Episode]:
        """
        Retrieves a list of episodes for a given session, ordered by timestamp.
        Reads from JSON Lines format and applies pagination.

        Args:
            session_id: The ID of the session to retrieve episodes for.
            limit: The maximum number of episodes to return.
            offset: The number of episodes to skip (for pagination).

        Returns:
            A list of Episode objects.

        Raises:
            StorageError: If an error occurs during file reading or JSON deserialization.
        """
        episode_file_path = self._get_episode_path(session_id)
        episodes: List[Episode] = []

        try:
            if not await aios.path.exists(episode_file_path):
                logger.debug(f"Episode file not found for session '{session_id}' at {episode_file_path}")
                return []

            async with aiofiles.open(episode_file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()

            lines = content.strip().split("\n")
            if not lines or (len(lines) == 1 and not lines[0]):
                logger.debug(f"No episodes found for session '{session_id}'")
                return []

            # Parse each JSON line and create Episode objects
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                    episode = Episode.model_validate(episode_data)
                    episodes.append(episode)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in episode file for session '{session_id}': {e}")
                except Exception as e:
                    logger.warning(f"Error parsing episode on line {line_num} for session '{session_id}': {e}")

            # Sort by timestamp (most recent first for consistency with other list methods)
            episodes.sort(key=lambda ep: ep.timestamp, reverse=True)

            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_episodes = episodes[start_idx:end_idx]

            logger.debug(f"Retrieved {len(paginated_episodes)} episodes for session '{session_id}' (offset={offset}, limit={limit})")
            return paginated_episodes

        except IOError as e:
            logger.error(f"Error reading episode file '{episode_file_path}' for session '{session_id}': {e}")
            raise StorageError(f"Failed to read episode file for session '{session_id}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading episodes for session '{session_id}': {e}", exc_info=True)
            raise StorageError(f"Unexpected error loading episodes for session '{session_id}': {e}")

    async def close(self) -> None:
        """
        Clean up resources. For JSON storage, no explicit closing action is typically needed.
        """
        logger.debug("JSONSessionStorage (including presets and episodes) closed (no specific action needed).")
        pass
