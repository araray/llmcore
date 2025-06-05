# src/llmcore/sessions/manager.py
"""
Session Management for LLMCore.

This module defines the SessionManager class, responsible for handling
the lifecycle and state of ChatSession objects, including loading from
and saving to a configured storage backend provided by StorageManager.
"""

import logging
from typing import Optional

from ..exceptions import (LLMCoreError, SessionNotFoundError,
                          SessionStorageError)
from ..models import ChatSession, Message, Role
from ..storage.base_session import \
    BaseSessionStorage  # Keep BaseSessionStorage for type hint

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages ChatSession objects, interacting with a storage backend.

    Coordinates loading, creating, and saving conversation sessions using
    the storage instance provided during initialization.
    """

    def __init__(self, storage: BaseSessionStorage):
        """
        Initializes the SessionManager.

        Args:
            storage: An initialized instance of a BaseSessionStorage backend
                     (obtained from StorageManager).
        """
        if not storage:
            # This check ensures storage is provided, preventing None errors later.
            # LLMCore.create should guarantee storage is initialized before SessionManager.
            logger.error("SessionManager initialized without a valid storage backend.")
            raise LLMCoreError("SessionManager requires a valid storage backend instance.")
        self._storage = storage # Store the provided storage instance
        logger.debug("SessionManager initialized with storage backend: %s", type(storage).__name__)

    async def load_or_create_session(
        self,
        session_id: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> ChatSession:
        """
        Loads an existing session or creates a new one.

        If session_id is provided:
          - Tries to load the session from storage.
          - If not found, creates a *new persistent session* with that ID.
        If session_id is None:
          - Creates a new, temporary (in-memory) ChatSession object with a new UUID.
        If system_message is provided *and* a *new* session is created (either
        temporary or persistent):
          - The system message is added as the first message.

        Args:
            session_id: The ID of the session to load or create. If None,
                        a temporary session with a new UUID is created.
            system_message: An optional system message to add if a new session
                            is created.

        Returns:
            The loaded or newly created ChatSession object.

        Raises:
            SessionStorageError: If there's an error interacting with storage during loading.
        """
        if session_id:
            logger.debug(f"Attempting to load or create session with ID: {session_id}")
            try:
                # Use the injected storage backend instance
                session = await self._storage.get_session(session_id)
                if session:
                    logger.info(f"Loaded existing persistent session: {session_id}")
                    # Note: We currently don't modify an existing session's system message here.
                    # If a system message is provided for an existing session, it might be ignored
                    # or handled by the ContextManager depending on strategy.
                    if system_message:
                         logger.warning(f"System message provided for existing session '{session_id}'. "
                                        "It might be ignored or handled by ContextManager depending on context strategy. "
                                        "The loaded session's existing system message (if any) will be used unless explicitly overridden by LLMCore.chat().")
                    return session
                else:
                    # Session ID was provided, but not found in storage -> Create new persistent session
                    logger.info(f"Session ID '{session_id}' not found in persistent storage. Creating new persistent session.")
                    new_session = ChatSession(id=session_id) # Create with the specified ID
                    if system_message:
                        logger.debug(f"Adding system message to new persistent session '{session_id}'.")
                        new_session.add_message(message_content=system_message, role=Role.SYSTEM)
                    # Note: The new session is not saved here; it's saved by LLMCore.chat() if save_session=True.
                    return new_session

            except SessionStorageError as e:
                # Log and re-raise storage errors encountered during the load attempt
                logger.error(f"Storage error while trying to load session '{session_id}': {e}")
                raise
            except Exception as e:
                # Catch other potential errors during loading/creation
                logger.error(f"Unexpected error loading or creating session '{session_id}': {e}", exc_info=True)
                raise SessionStorageError(f"Failed to load or create session '{session_id}': {e}")

        # If session_id is None, create a new temporary (in-memory) session object
        logger.info("Creating new temporary session object (session_id was None).")
        # The ID will be generated by ChatSession's default_factory
        new_temporary_session = ChatSession() # ID generated automatically
        if system_message:
            logger.debug("Adding system message to new temporary session object.")
            new_temporary_session.add_message(message_content=system_message, role=Role.SYSTEM)
        return new_temporary_session

    async def get_session_if_exists(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieves a session from storage only if it exists.
        Does not create a new session if the ID is not found.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found in persistent storage, otherwise None.

        Raises:
            SessionStorageError: If there's an error interacting with storage.
            ValueError: If session_id is None or empty.
        """
        if not session_id:
            logger.error("get_session_if_exists called with no session_id.")
            # Raise ValueError as this is an invalid argument for this specific method.
            raise ValueError("session_id cannot be None or empty for get_session_if_exists.")

        logger.debug(f"Attempting to get session if exists (persistent only): {session_id}")
        try:
            session = await self._storage.get_session(session_id) # This method in BaseSessionStorage should return Optional[ChatSession]
            if session:
                logger.info(f"Found existing persistent session: {session_id}")
            else:
                logger.info(f"Persistent session '{session_id}' not found by storage backend.")
            return session
        except SessionStorageError as e: # Catch specific storage errors
            logger.error(f"Storage error while trying to get session '{session_id}': {e}")
            raise # Re-raise to be handled by caller (e.g., LLMCore API)
        except Exception as e: # Catch other unexpected errors from storage
            logger.error(f"Unexpected error getting session '{session_id}': {e}", exc_info=True)
            # Wrap in SessionStorageError if it's not already one
            raise SessionStorageError(f"Failed to get session '{session_id}': {e}")


    async def save_session(self, session: ChatSession) -> None:
        """
        Saves a session to the configured storage backend.
        This method assumes that if a session object is passed here, it's intended
        for persistent storage. Transient session logic (not saving) should be handled
        by the caller (e.g., LLMCore.chat based on save_session flag).

        Args:
            session: The ChatSession object to save.

        Raises:
            SessionStorageError: If there's an error saving the session.
            ValueError: If the session or its ID is invalid.
        """
        if not session or not session.id:
            logger.error("Attempted to save an invalid or unidentifiable session.")
            raise ValueError("Cannot save a session without a valid ID.")

        logger.debug(f"Attempting to save session to persistent storage: {session.id}")
        try:
            await self._storage.save_session(session)
            logger.info(f"Session '{session.id}' saved successfully to persistent storage.")
        except SessionStorageError as e:
            logger.error(f"Storage error while saving session '{session.id}': {e}")
            raise # Re-raise specific storage errors
        except Exception as e:
            logger.error(f"Unexpected error saving session '{session.id}': {e}", exc_info=True)
            # Wrap unexpected errors in SessionStorageError
            raise SessionStorageError(f"Failed to save session '{session.id}': {e}")
