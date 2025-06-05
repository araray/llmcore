# src/llmcore/api.py
"""
Core API Facade for the LLMCore library.
"""

import asyncio
import importlib.resources
import json
import logging
import pathlib
import uuid # Ensure uuid is imported
from datetime import datetime, timezone # Ensure timezone is imported
from typing import (Any, AsyncGenerator, Dict, List, Optional, Tuple, Type,
                    Union)

import aiofiles # Keep for file operations in context item management

from .context.manager import ContextManager
from .embedding.manager import EmbeddingManager
from .exceptions import (ConfigError, ContextLengthError, EmbeddingError,
                         LLMCoreError, ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError)
from .models import (ChatSession, ContextDocument, ContextItem, # Ensure all models are imported
                     ContextItemType, Message, Role, ContextPreparationDetails,
                     ContextPreset, ContextPresetItem)
from .providers.base import BaseProvider
from .providers.manager import ProviderManager
from .sessions.manager import SessionManager
from .storage.manager import StorageManager

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]
try:
    import tomllib # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib # Fallback for Python < 3.11
    except ImportError:
        tomllib = None # type: ignore [assignment]

# For type checking Ollama stream responses if the library is used
try:
    from ollama import ChatResponse as OllamaChatResponse
except ImportError:
    OllamaChatResponse = None # type: ignore [assignment]


logger = logging.getLogger(__name__)


class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, Retrieval Augmented
    Generation (RAG), context pool management (user-added context items),
    context preview, and management of saved context presets.
    It also allows dynamic control over logging levels and raw payload logging.

    It is initialized asynchronously using the `LLMCore.create()` classmethod.
    This class orchestrates various managers (provider, storage, session,
    context, embedding) to deliver its functionalities.
    """

    config: ConfyConfig # The loaded confy config object
    _storage_manager: StorageManager
    _provider_manager: ProviderManager
    _session_manager: SessionManager
    _context_manager: ContextManager
    _embedding_manager: EmbeddingManager
    _transient_last_interaction_info_cache: Dict[str, ContextPreparationDetails]
    _transient_sessions_cache: Dict[str, ChatSession]

    # Instance attributes for dynamic settings, initialized from config
    _log_raw_payloads_enabled: bool
    _llmcore_log_level_str: str


    def __init__(self):
        """
        Private constructor. Use the `LLMCore.create()` classmethod for
        asynchronous initialization.
        Initializes internal caches for transient session data.
        """
        self._transient_last_interaction_info_cache = {}
        self._transient_sessions_cache = {}
        # Initialization of managers and dynamic settings is handled by the `create` method.
        pass

    @classmethod
    async def create(
        cls,
        config_overrides: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[str] = None,
        env_prefix: Optional[str] = "LLMCORE"
    ) -> "LLMCore":
        """
        Asynchronously creates and initializes an LLMCore instance.

        This method handles the loading of configurations (using `confy`),
        initialization of all internal managers (ProviderManager,
        StorageManager, SessionManager, EmbeddingManager, ContextManager),
        and sets up initial logging states based on the configuration.

        Args:
            config_overrides: Dictionary of configuration values to override settings
                              from files or environment variables. Highest precedence.
                              Keys use dot-notation (e.g., "providers.openai.default_model").
            config_file_path: Path to a custom TOML or JSON configuration file.
            env_prefix: The prefix for environment variables that LLMCore will
                        consider for configuration overrides (default: "LLMCORE").
                        Set to `""` to consider all non-system env vars, or `None`
                        to disable environment variable loading.

        Returns:
            A fully initialized LLMCore instance.

        Raises:
            ConfigError: If essential configuration is missing, invalid, or if
                         dependencies like `tomli(b)` are not found.
            LLMCoreError: For general failures during manager initializations if not
                          more specifically caught (e.g., ProviderError, StorageError).
            ProviderError: If a configured provider fails to initialize.
            StorageError: If a configured storage backend fails to initialize.
            EmbeddingError: If the default embedding model fails to initialize.
        """
        instance = cls()
        logger.info("Initializing LLMCore asynchronously...")

        # 1. Initialize self.config using confy.Config(...)
        try:
            from confy.loader import Config as ActualConfyConfig # Ensure it's the actual class
            if not tomllib:
                raise ImportError("tomli (for Python < 3.11) or tomllib is required for loading default_config.toml.")

            default_config_dict = {}
            try:
                # Use importlib.resources for robust path handling within package
                if hasattr(importlib.resources, 'files'): # Python 3.9+
                    default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
                    with default_config_path_obj.open('rb') as f: # tomllib.load expects bytes
                        default_config_dict = tomllib.load(f)
                else: # Fallback for Python 3.8 (importlib.resources.read_text)
                    default_config_content = importlib.resources.read_text('llmcore.config', 'default_config.toml', encoding='utf-8') # type: ignore
                    default_config_dict = tomllib.loads(default_config_content) # type: ignore
            except Exception as e:
                 raise ConfigError(f"Failed to load default configuration from llmcore.config: {e}")

            instance.config = ActualConfyConfig(
                defaults=default_config_dict,
                file_path=config_file_path,
                prefix=env_prefix,
                overrides_dict=config_overrides,
                mandatory=[] # No mandatory keys at this level, handled by components
            )
            logger.info("confy configuration loaded successfully.")
        except ImportError as e:
             # This typically means confy itself or a core dep is missing
             raise ConfigError(f"Configuration dependency (confy or toml parser) missing: {e}")
        except ConfigError: # Re-raise ConfigErrors from confy or default loading
            raise
        except Exception as e: # Catch any other unexpected error during config init
            raise ConfigError(f"LLMCore configuration initialization failed: {e}")

        # --- Initialize dynamic settings from config ---
        instance._log_raw_payloads_enabled = instance.config.get('llmcore.log_raw_payloads', False)
        instance._llmcore_log_level_str = instance.config.get('llmcore.log_level', 'INFO').upper()
        # --- End Dynamic Settings Initialization ---


        # --- Logging Enhancements: Initialize from instance attributes ---
        logger.info(f"Raw payload logging initially set to: {instance._log_raw_payloads_enabled}")

        llmcore_log_level_int = logging.getLevelName(instance._llmcore_log_level_str)
        if isinstance(llmcore_log_level_int, int):
            logging.getLogger("llmcore").setLevel(llmcore_log_level_int)
            logger.info(f"LLMCore base logger level set to: {instance._llmcore_log_level_str}")
        else:
            logging.getLogger("llmcore").setLevel(logging.INFO) # Default if invalid
            logger.warning(f"Invalid llmcore.log_level '{instance._llmcore_log_level_str}' in config, defaulting LLMCore logger to INFO.")
        # --- End Logging Enhancements Initialization ---

        # 2. Initialize Managers, passing self.config
        # ProviderManager will now implicitly use instance.config['llmcore.log_raw_payloads']
        # when initializing individual providers.
        try:
            # ProviderManager now gets the initial log_raw_payloads state directly
            instance._provider_manager = ProviderManager(instance.config)
            logger.info("ProviderManager initialized.")
        except (ConfigError, ProviderError) as e:
            logger.error(f"Failed to initialize ProviderManager: {e}", exc_info=True)
            raise
        except Exception as e:
            raise LLMCoreError(f"ProviderManager initialization failed unexpectedly: {e}")

        try:
            instance._storage_manager = StorageManager(instance.config)
            await instance._storage_manager.initialize_storages() # This is async
            logger.info("StorageManager initialized.")
        except (ConfigError, StorageError) as e:
            logger.error(f"Failed to initialize StorageManager: {e}", exc_info=True)
            raise
        except Exception as e:
            raise LLMCoreError(f"StorageManager initialization failed unexpectedly: {e}")

        try:
            session_storage = instance._storage_manager.get_session_storage()
            instance._session_manager = SessionManager(session_storage)
            logger.info("SessionManager initialized.")
        except StorageError as e:
            raise LLMCoreError(f"SessionManager initialization failed due to storage issue: {e}")
        except Exception as e:
            raise LLMCoreError(f"SessionManager initialization failed unexpectedly: {e}")

        try:
            instance._embedding_manager = EmbeddingManager(instance.config)
            default_embedding_model_id = instance.config.get('llmcore.default_embedding_model')
            if default_embedding_model_id:
                logger.info(f"Pre-initializing default embedding model: {default_embedding_model_id}")
                await instance._embedding_manager.get_model(default_embedding_model_id)
                logger.info(f"Default embedding model '{default_embedding_model_id}' initialized successfully.")
            else:
                logger.info("No default_embedding_model configured in 'llmcore' section. Skipping pre-initialization.")
            logger.info("EmbeddingManager setup complete.")
        except (ConfigError, EmbeddingError) as e:
             logger.error(f"Failed to initialize EmbeddingManager or default model: {e}", exc_info=True)
             raise
        except Exception as e:
            raise LLMCoreError(f"EmbeddingManager initialization or default model setup failed unexpectedly: {e}")

        try:
            instance._context_manager = ContextManager(
                config=instance.config, # Pass the main confy object
                provider_manager=instance._provider_manager,
                storage_manager=instance._storage_manager,
                embedding_manager=instance._embedding_manager
            )
            logger.info("ContextManager initialized.")
        except Exception as e:
            raise LLMCoreError(f"ContextManager initialization failed unexpectedly: {e}")

        logger.info("LLMCore asynchronous initialization complete.")
        return instance

    # --- Logging Control Methods ---
    def set_raw_payload_logging(self, enable: bool):
        """
        Dynamically enables or disables raw payload logging for all providers.
        This updates an internal LLMCore state and propagates the change to
        all currently initialized provider instances. Raw payloads are logged
        at DEBUG level by providers.

        Args:
            enable: True to enable raw payload logging, False to disable.
        """
        self._log_raw_payloads_enabled = enable
        logger.info(f"LLMCore raw payload logging has been {'ENABLED' if enable else 'DISABLED'}.")

        # Propagate to already initialized providers via ProviderManager
        if hasattr(self, '_provider_manager') and self._provider_manager:
            self._provider_manager.update_log_raw_payloads_setting(enable)
        else:
            logger.warning("ProviderManager not available to propagate raw payload logging setting.")

        # This log was redundant, previous one is sufficient
        # logger.info(f"LLMCore raw payload logging has been {'ENABLED' if enable else 'DISABLED'}.")
        if enable and logging.getLogger("llmcore").getEffectiveLevel() > logging.DEBUG:
            logger.warning("Raw payload logging enabled, but LLMCore log level is not DEBUG. "
                           "Raw payloads may not appear unless LLMCore log level is also set to DEBUG.")

    def set_log_level(self, level_name: str):
        """
        Dynamically sets the log level for the 'llmcore' logger and its children.
        Also updates an internal state attribute reflecting this level.

        Args:
            level_name: The desired log level string (e.g., "DEBUG", "INFO", "ERROR").
                        Case-insensitive.
        """
        level_name_upper = level_name.upper()
        log_level_int = logging.getLevelName(level_name_upper)

        if not isinstance(log_level_int, int):
            logger.error(f"Invalid log level name: '{level_name}'. No change made to log levels.")
            return

        # Set for the main 'llmcore' logger. Children will inherit.
        logging.getLogger("llmcore").setLevel(log_level_int)
        self._llmcore_log_level_str = level_name_upper # Update instance state

        logger.info(f"LLMCore log level set to: {level_name_upper}.")
        # Add a warning if raw payload logging is enabled but level is not DEBUG
        if self._log_raw_payloads_enabled and log_level_int > logging.DEBUG:
            logger.warning("Raw payload logging is currently enabled, but LLMCore log level is not DEBUG. "
                           "Raw payloads may not appear in logs.")

    # --- Core Chat Method ---
    async def chat(
        self,
        message: str,
        *, # Force subsequent arguments to be keyword-only
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None, # This is the provider for THIS call
        model_name: Optional[str] = None,    # This is the model for THIS call
        stream: bool = False,
        save_session: bool = True,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False,            # RAG setting for THIS call
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None,
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the LLM, managing history, user-added context, RAG, and explicitly staged items.
        The operational settings used for this call (provider, model, RAG config, system message, prompt values)
        are persisted into the ChatSession's metadata if `save_session` is True and `session_id` is provided.

        (Args docstring largely unchanged, see previous versions for full details)
        """
        # Determine the actual provider and model to use for this interaction
        actual_provider_name_for_call = provider_name or self._provider_manager.get_default_provider().get_name()
        active_provider = self._provider_manager.get_provider(actual_provider_name_for_call)
        actual_model_name_for_call = model_name or active_provider.default_model
        if not actual_model_name_for_call:
             raise ConfigError(f"Target model undetermined for provider '{active_provider.get_name()}'.")

        logger.debug(
            f"LLMCore.chat: session='{session_id}', save_session={save_session}, "
            f"provider_for_call='{actual_provider_name_for_call}', model_for_call='{actual_model_name_for_call}', "
            f"stream={stream}, RAG_for_call={enable_rag}, "
            f"RAG_filter_keys_for_call={list(rag_metadata_filter.keys()) if rag_metadata_filter else 'None'}, "
            f"prompt_values_count_for_call={len(prompt_template_values) if prompt_template_values else 0}, "
            f"active_user_items_count={len(active_context_item_ids) if active_context_item_ids else 0}, "
            f"explicitly_staged_items_count={len(explicitly_staged_items) if explicitly_staged_items else 0}"
        )

        try:
            # --- Session Handling ---
            chat_session: ChatSession
            is_new_persistent_session = False
            if session_id:
                if not save_session: # Transient session logic
                    if session_id in self._transient_sessions_cache:
                        chat_session = self._transient_sessions_cache[session_id]
                        logger.debug(f"Using transient session '{session_id}' from cache.")
                        # Update system message if provided for existing transient session
                        # (and different from existing or not present)
                        current_sys_msg_obj = next((m for m in chat_session.messages if m.role == Role.SYSTEM), None)
                        if system_message is not None and (not current_sys_msg_obj or current_sys_msg_obj.content != system_message):
                            chat_session.messages = [m for m in chat_session.messages if m.role != Role.SYSTEM] # Remove old
                            chat_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=session_id))
                            logger.debug(f"Updated/Added system message to cached transient session '{session_id}'.")
                        elif system_message is None and current_sys_msg_obj: # If system_message=None for call, remove from transient if present
                            chat_session.messages = [m for m in chat_session.messages if m.role != Role.SYSTEM]
                            logger.debug(f"Removed system message from cached transient session '{session_id}' as per call.")

                    else: # New transient session
                        chat_session = ChatSession(id=session_id)
                        if system_message is not None: # Only add if explicitly provided
                            chat_session.add_message(message_content=system_message, role=Role.SYSTEM)
                        self._transient_sessions_cache[session_id] = chat_session
                        logger.debug(f"Created new transient session '{session_id}' and cached it.")
                else: # Persistent session
                    existing_session = await self._session_manager.get_session_if_exists(session_id) # New method in SessionManager
                    if existing_session:
                        chat_session = existing_session
                        # Update system message if provided and different for existing persistent session
                        current_sys_msg_obj = next((m for m in chat_session.messages if m.role == Role.SYSTEM), None)
                        if system_message is not None and (not current_sys_msg_obj or current_sys_msg_obj.content != system_message):
                            chat_session.messages = [m for m in chat_session.messages if m.role != Role.SYSTEM] # Remove old
                            chat_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=session_id))
                            logger.debug(f"Updated/Added system message to existing persistent session '{session_id}' for this interaction.")
                        elif system_message is None and current_sys_msg_obj: # Remove if system_message=None for call
                            chat_session.messages = [m for m in chat_session.messages if m.role != Role.SYSTEM]
                            logger.debug(f"Removed system message from persistent session '{session_id}' for this interaction as per call.")
                    else:
                        is_new_persistent_session = True
                        chat_session = ChatSession(id=session_id)
                        if system_message is not None:
                            chat_session.add_message(message_content=system_message, role=Role.SYSTEM)
                        logger.info(f"Creating new persistent session with ID: {session_id}")
            else: # Stateless call (no session_id provided)
                chat_session = ChatSession(id=f"temp_stateless_{uuid.uuid4().hex[:8]}") # Temporary ID
                if system_message is not None:
                    chat_session.add_message(message_content=system_message, role=Role.SYSTEM)

            user_msg_obj = chat_session.add_message(message_content=message, role=Role.USER)
            logger.debug(f"User message '{user_msg_obj.id}' added to session '{chat_session.id}' (is_transient={not save_session and bool(session_id)}, is_new_persistent={is_new_persistent_session})")

            # --- Prepare context ---
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=chat_session,
                provider_name=actual_provider_name_for_call,
                model_name=actual_model_name_for_call,
                active_context_item_ids=active_context_item_ids,
                explicitly_staged_items=explicitly_staged_items,
                rag_enabled=enable_rag,
                rag_k=rag_retrieval_k,
                rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter,
                prompt_template_values=prompt_template_values
            )
            context_payload = context_details.prepared_messages
            logger.info(f"Prepared context with {len(context_payload)} messages ({context_details.final_token_count} tokens) for model '{actual_model_name_for_call}'.")

            if session_id:
                self._transient_last_interaction_info_cache[session_id] = context_details

            # --- Call Provider ---
            response_data_or_generator = await active_provider.chat_completion(
                context=context_payload, model=actual_model_name_for_call, stream=stream, **provider_kwargs
            )

            # --- Handle Response (Stream or Full) & Persist Settings to Metadata ---
            if stream:
                logger.debug(f"Processing stream response from provider '{active_provider.get_name()}'")
                provider_stream: AsyncGenerator[Any, None] = response_data_or_generator # type: ignore
                return self._stream_response_wrapper(
                    provider_stream, active_provider, chat_session,
                    (save_session and session_id is not None),
                    # Pass settings used for this call to be saved in metadata
                    operational_settings_for_metadata={
                        "current_provider_name": actual_provider_name_for_call,
                        "current_model_name": actual_model_name_for_call,
                        "system_message": system_message, # The system message active for *this* call
                        "rag_enabled": enable_rag,
                        "rag_collection_name": rag_collection_name,
                        "rag_k_value": rag_retrieval_k,
                        "rag_filter": rag_metadata_filter,
                        "prompt_template_values": prompt_template_values
                    }
                )
            else: # Non-streaming
                if not isinstance(response_data_or_generator, dict):
                     logger.error(f"Expected dict response for non-streaming chat, got {type(response_data_or_generator).__name__}")
                     raise ProviderError(active_provider.get_name(), "Invalid response format (expected dict).")

                response_data = response_data_or_generator
                full_response_content = self._extract_full_content(response_data, active_provider)

                if full_response_content is None:
                     logger.warning(f"Content extraction failed for non-streaming response from {active_provider.get_name()}. Response data: {str(response_data)[:200]}")
                     full_response_content = f"[LLMCore Error: Failed to extract content from provider response]"
                else:
                     logger.debug(f"Received full response content (length: {len(full_response_content)}).")

                assistant_msg = chat_session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                logger.debug(f"Assistant message '{assistant_msg.id}' added to session '{chat_session.id}'.")

                if save_session and session_id:
                    # Persist operational settings into metadata
                    if chat_session.metadata is None: chat_session.metadata = {}
                    chat_session.metadata["current_provider_name"] = actual_provider_name_for_call
                    chat_session.metadata["current_model_name"] = actual_model_name_for_call
                    # Only store if not None, to avoid overwriting with None if not explicitly passed
                    if system_message is not None: chat_session.metadata["system_message"] = system_message
                    chat_session.metadata["rag_enabled"] = enable_rag
                    if rag_collection_name is not None: chat_session.metadata["rag_collection_name"] = rag_collection_name
                    if rag_retrieval_k is not None: chat_session.metadata["rag_k_value"] = rag_retrieval_k
                    if rag_metadata_filter is not None: chat_session.metadata["rag_filter"] = rag_metadata_filter
                    if prompt_template_values is not None: chat_session.metadata["prompt_template_values"] = prompt_template_values
                    chat_session.updated_at = datetime.now(timezone.utc) # Ensure updated_at is set before save
                    await self._session_manager.save_session(chat_session)
                    logger.info(f"Persistent session '{session_id}' saved with operational metadata.")

                return full_response_content

        except (SessionNotFoundError, StorageError, ProviderError, ContextLengthError,
                ConfigError, EmbeddingError, VectorStorageError) as e:
             logger.error(f"Chat failed: {type(e).__name__} - {e}")
             if session_id and not save_session and session_id in self._transient_sessions_cache:
                 del self._transient_sessions_cache[session_id]
                 logger.debug(f"Cleared failed transient session '{session_id}' from cache.")
             raise
        except Exception as e:
             logger.error(f"Unexpected error during chat execution: {e}", exc_info=True)
             if session_id and not save_session and session_id in self._transient_sessions_cache:
                 del self._transient_sessions_cache[session_id]
             raise LLMCoreError(f"Chat execution failed due to an unexpected error: {e}")


    async def preview_context_for_chat(
        self,
        current_user_query: str,
        *, # Force keyword-only arguments
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        active_context_item_ids: Optional[List[str]] = None,
        explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        rag_metadata_filter: Optional[Dict[str, Any]] = None,
        prompt_template_values: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Previews the context that would be prepared for an LLM chat call
        without making the actual API call to the LLM provider.

        (Docstring largely unchanged, see previous versions for full details)
        """
        actual_provider_name_for_preview = provider_name or self._provider_manager.get_default_provider().get_name()
        active_provider = self._provider_manager.get_provider(actual_provider_name_for_preview)
        actual_model_name_for_preview = model_name or active_provider.default_model
        if not actual_model_name_for_preview:
             raise ConfigError(f"Target model undetermined for provider '{active_provider.get_name()}' for preview.")

        logger.info(f"Previewing context for query: '{current_user_query[:50]}...' "
                    f"(Provider: {actual_provider_name_for_preview}, Model: {actual_model_name_for_preview}, Session: {session_id})")

        preview_session: ChatSession
        if session_id:
            loaded_session = await self._session_manager.get_session_if_exists(session_id)
            if loaded_session:
                preview_session = loaded_session
                # Handle system_message override for preview
                current_sys_msg_obj = next((m for m in preview_session.messages if m.role == Role.SYSTEM), None)
                if system_message is not None: # If system_message is explicitly provided for preview
                    if not current_sys_msg_obj or current_sys_msg_obj.content != system_message:
                        preview_session.messages = [m for m in preview_session.messages if m.role != Role.SYSTEM]
                        preview_session.messages.insert(0, Message(role=Role.SYSTEM, content=system_message, session_id=preview_session.id))
                        logger.debug("Preview: Using provided system_message, overriding/adding to session's system message.")
                # If system_message is None for preview, retain session's existing system message (if any)
            else: # Session ID provided but not found, simulate new session for preview
                preview_session = ChatSession(id=session_id)
                if system_message is not None:
                    preview_session.add_message(message_content=system_message, role=Role.SYSTEM)
                logger.debug(f"Preview: Session '{session_id}' not found, simulating new session for preview context.")
        else: # No session_id, create a fully temporary session for preview
            preview_session = ChatSession(id=f"preview_stateless_{uuid.uuid4().hex[:8]}")
            if system_message is not None:
                preview_session.add_message(message_content=system_message, role=Role.SYSTEM)

        preview_session.add_message(message_content=current_user_query, role=Role.USER)

        try:
            context_details: ContextPreparationDetails = await self._context_manager.prepare_context(
                session=preview_session,
                provider_name=actual_provider_name_for_preview,
                model_name=actual_model_name_for_preview,
                active_context_item_ids=active_context_item_ids,
                explicitly_staged_items=explicitly_staged_items,
                rag_enabled=enable_rag,
                rag_k=rag_retrieval_k,
                rag_collection=rag_collection_name,
                rag_metadata_filter=rag_metadata_filter,
                prompt_template_values=prompt_template_values
            )
            return context_details.model_dump(mode="json")

        except Exception as e:
            logger.error(f"Error during context preview preparation: {e}", exc_info=True)
            if isinstance(e, (ConfigError, ContextLengthError, EmbeddingError, ProviderError, VectorStorageError)):
                raise
            raise LLMCoreError(f"Context preview failed due to an internal error: {e}")


    async def _stream_response_wrapper(
        self, provider_stream: AsyncGenerator[Any, None], provider: BaseProvider,
        session: ChatSession, do_save_session: bool,
        operational_settings_for_metadata: Optional[Dict[str, Any]] = None # New arg
    ) -> AsyncGenerator[str, None]:
        """
        Wraps provider's stream, yields text chunks, and handles session saving
        including persisting operational settings to metadata.
        """
        full_response_content = ""; error_occurred = False; provider_name = provider.get_name()
        try:
            async for chunk in provider_stream:
                chunk_dict: Optional[Dict[str, Any]] = None
                if isinstance(chunk, dict): chunk_dict = chunk
                elif OllamaChatResponse and isinstance(chunk, OllamaChatResponse): # type: ignore
                    try: chunk_dict = chunk.model_dump()
                    except Exception as dump_err: logger.warning(f"Could not dump Ollama stream object: {dump_err}. Chunk: {chunk}"); continue
                else: logger.warning(f"Received non-dict/non-OllamaResponse chunk: {type(chunk)} - {chunk}"); continue
                if not chunk_dict: continue

                text_delta = self._extract_delta_content(chunk_dict, provider)
                error_message = chunk_dict.get('error')
                finish_reason = None # Extracted same way as before
                if 'choices' in chunk_dict and chunk_dict['choices'] and isinstance(chunk_dict['choices'][0], dict):
                    finish_reason = chunk_dict['choices'][0].get('finish_reason')
                elif 'finish_reason' in chunk_dict: # For Gemini-like direct finish_reason
                    finish_reason = chunk_dict.get('finish_reason')
                elif chunk_dict.get("type") == "message_delta" and chunk_dict.get("delta"): # Anthropic
                    finish_reason = chunk_dict["delta"].get("stop_reason")


                if text_delta: full_response_content += text_delta; yield text_delta
                if error_message: logger.error(f"Error during stream: {error_message}"); raise ProviderError(provider_name, error_message)
                # Check for various stop reasons (Anthropic, Gemini, OpenAI use different ones)
                if finish_reason and finish_reason not in ["stop", "length", None, "STOP_SEQUENCE", "MAX_TOKENS", "TOOL_USE", "stop_token", "max_tokens", "NOT_SET", "FINISH_REASON_UNSPECIFIED"]:
                    logger.warning(f"Stream stopped due to reason: {finish_reason}")
        except Exception as e:
            error_occurred = True
            logger.error(f"Error processing stream from {provider_name}: {e}", exc_info=True)
            if isinstance(e, ProviderError): raise
            raise ProviderError(provider_name, f"Stream processing error: {e}")
        finally:
            logger.debug(f"Stream from {provider_name} finished.")
            if full_response_content or not error_occurred:
                assistant_msg = session.add_message(message_content=full_response_content, role=Role.ASSISTANT)
                logger.debug(f"Assistant message '{assistant_msg.id}' (len: {len(full_response_content)}) added to session '{session.id}' (in-memory) after stream.")
            else:
                 logger.debug(f"No assistant message added to session '{session.id}' due to stream error or empty response.")

            if do_save_session: # This implies session.id is not None
                try:
                    # Persist operational settings into metadata before saving
                    if operational_settings_for_metadata:
                        if session.metadata is None: session.metadata = {}
                        for key, value in operational_settings_for_metadata.items():
                            if value is not None: # Only store non-None values
                                session.metadata[key] = value
                            elif key in session.metadata: # Remove key if value is None and key exists
                                del session.metadata[key]
                        logger.debug(f"Updated session metadata for '{session.id}' with operational settings: {list(operational_settings_for_metadata.keys())}")

                    session.updated_at = datetime.now(timezone.utc) # Ensure updated_at is set
                    await self._session_manager.save_session(session)
                    logger.info(f"Persistent session '{session.id}' saved with operational metadata after stream.")
                except Exception as save_e:
                     logger.error(f"Failed to save session {session.id} after stream: {save_e}", exc_info=True)

    def _extract_delta_content(self, chunk: Dict[str, Any], provider: BaseProvider) -> str:
        """Extracts text delta from stream chunk. (Implementation unchanged)"""
        provider_name = provider.get_name(); text_delta = ""
        try:
            if provider_name == "openai" or provider_name == "gemini": # Gemini stream chunks also have choices[0].delta.content-like structure after SDK processing
                choices = chunk.get('choices', [])
                if choices and isinstance(choices[0], dict) and choices[0].get('delta'): text_delta = choices[0]['delta'].get('content', '') or ""
            elif provider_name == "anthropic":
                type_val = chunk.get("type")
                # Anthropic SDK v0.8+ new streaming events
                if type_val == "content_block_delta" and chunk.get('delta', {}).get('type') == "text_delta":
                    text_delta = chunk.get('delta', {}).get('text', "") or ""
                # Older Anthropic SDK event types (might be needed for compatibility or if custom handling before SDK)
                # elif type_val == "content_block_delta" and chunk.get('delta', {}).get('type') == 'text_delta':
                #     text_delta = chunk['delta'].get('text', "")
            elif provider_name == "ollama":
                message_chunk = chunk.get('message', {});
                if message_chunk and isinstance(message_chunk, dict): text_delta = message_chunk.get('content', '') or ""
                # Older ollama lib versions might just have 'response'
                elif 'response' in chunk and isinstance(chunk['response'], str): text_delta = chunk.get('response', '') or ""
        except Exception as e: logger.warning(f"Error extracting delta from {provider_name} chunk: {e}. Chunk: {str(chunk)[:200]}"); text_delta = ""
        return text_delta or "" # Ensure it's always a string

    def _extract_full_content(self, response_data: Dict[str, Any], provider: BaseProvider) -> Optional[str]:
        """Extracts full response content. (Implementation unchanged)"""
        provider_name = provider.get_name(); full_response_content: Optional[str] = None
        try:
            if provider_name == "openai":
                choices = response_data.get('choices', [])
                if choices and isinstance(choices[0], dict) and choices[0].get('message'): full_response_content = choices[0]['message'].get('content')
            elif provider_name == "gemini": # Gemini non-streaming response structure
                candidates = response_data.get('candidates', [])
                if candidates and isinstance(candidates[0], dict) and candidates[0].get('content'):
                    content_parts = candidates[0]['content'].get('parts', [])
                    if content_parts and isinstance(content_parts[0], dict): full_response_content = content_parts[0].get('text')
            elif provider_name == "anthropic":
                content_blocks = response_data.get('content', []) # List of content blocks
                if content_blocks and isinstance(content_blocks, list) and content_blocks[0].get("type") == "text":
                    full_response_content = content_blocks[0].get("text")
            elif provider_name == "ollama":
                message_part = response_data.get('message', {}) # ollama library returns dict with 'message' key
                if message_part and isinstance(message_part, dict):
                    full_response_content = message_part.get('content')
                # Older ollama might have 'response' directly
                elif 'response' in response_data and isinstance(response_data['response'], str):
                    full_response_content = response_data.get('response')

            if full_response_content is None and response_data: # Check if extraction failed but data was present
                 logger.warning(f"Could not extract content from {provider_name} response: {str(response_data)[:200]}")
                 return None # Explicitly return None
            return str(full_response_content) if full_response_content is not None else None
        except Exception as e:
            logger.error(f"Error extracting full content from {provider_name}: {e}. Response: {str(response_data)[:200]}", exc_info=True)
            return None


    # --- Session Management Methods ---
    # (get_session, list_sessions, delete_session - implementations largely unchanged)
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieves a specific chat session object (including messages and context items).
        Checks transient cache first, then persistent storage.
        If not found in persistent storage via SessionManager, this method will now
        ensure it returns None, rather than SessionManager potentially creating a new one.
        """
        logger.debug(f"LLMCore.get_session for ID: {session_id}")
        if session_id in self._transient_sessions_cache:
            logger.debug(f"Returning session '{session_id}' from transient cache.")
            return self._transient_sessions_cache[session_id]
        try:
            # Use a more direct "get if exists" from SessionManager
            return await self._session_manager.get_session_if_exists(session_id)
        except SessionStorageError as e: # Changed from StorageError to SessionStorageError
            logger.error(f"SessionStorageError getting session '{session_id}': {e}")
            raise # Re-raise to be handled by caller

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """(Implementation unchanged)"""
        logger.debug("LLMCore.list_sessions called (persistent sessions only).")
        try:
            session_storage = self._storage_manager.get_session_storage()
            return await session_storage.list_sessions()
        except StorageError as e: logger.error(f"Storage error listing sessions: {e}"); raise

    async def delete_session(self, session_id: str) -> bool:
        """(Implementation unchanged)"""
        logger.debug(f"LLMCore.delete_session for ID: {session_id}")
        was_in_transient = self._transient_sessions_cache.pop(session_id, None) is not None
        if was_in_transient: logger.debug(f"Removed session '{session_id}' from transient cache during delete.")
        self._transient_last_interaction_info_cache.pop(session_id, None)
        try:
            session_storage = self._storage_manager.get_session_storage()
            deleted_persistent = await session_storage.delete_session(session_id)
            if deleted_persistent: logger.info(f"Session '{session_id}' deleted from persistent storage.")
            elif not was_in_transient: logger.warning(f"Session '{session_id}' not found for deletion in persistent storage.")
            return deleted_persistent or was_in_transient
        except StorageError as e: logger.error(f"Storage error deleting session '{session_id}': {e}"); raise
        return was_in_transient


    # --- RAG / Vector Store Management Methods ---
    # (add_document_to_vector_store, add_documents_to_vector_store, search_vector_store,
    #  delete_documents_from_vector_store, list_rag_collections - implementations largely unchanged)
    async def add_document_to_vector_store(self, content: str, *, metadata: Optional[Dict]=None, doc_id: Optional[str]=None, collection_name: Optional[str]=None) -> str:
        logger.debug(f"Adding document to vector store (Collection: {collection_name or 'default'})...")
        try:
            embedding = await self._embedding_manager.generate_embedding(content)
            doc_metadata = metadata if metadata is not None else {}
            doc = ContextDocument(id=doc_id if doc_id else str(uuid.uuid4()), content=content, embedding=embedding, metadata=doc_metadata)
            vector_storage = self._storage_manager.get_vector_storage()
            added_ids = await vector_storage.add_documents([doc], collection_name=collection_name)
            if not added_ids: raise VectorStorageError("Failed to add document, no ID returned.")
            logger.info(f"Document '{added_ids[0]}' added to vector store collection '{collection_name or vector_storage._default_collection_name}'.") # type: ignore
            return added_ids[0]
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to add document: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding document: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def add_documents_to_vector_store(self, documents: List[Dict[str, Any]], *, collection_name: Optional[str]=None) -> List[str]:
        if not documents: return []
        logger.debug(f"Adding batch of {len(documents)} documents to vector store (Collection: {collection_name or 'default'})...")
        try:
            contents = [doc_data["content"] for doc_data in documents if isinstance(doc_data.get("content"), str)]
            if len(contents) != len(documents): raise ValueError("All documents must have string 'content'.")
            embeddings = await self._embedding_manager.generate_embeddings(contents)
            if len(embeddings) != len(documents): raise EmbeddingError("Mismatch between texts and generated embeddings count.")
            vector_storage = self._storage_manager.get_vector_storage()
            resolved_collection_name = collection_name or vector_storage._default_collection_name # type: ignore
            docs_to_add = [ContextDocument(id=d.get("id", str(uuid.uuid4())), content=d["content"], embedding=emb, metadata=d.get("metadata",{})) for d, emb in zip(documents, embeddings)]
            added_ids = await vector_storage.add_documents(docs_to_add, collection_name=collection_name)
            logger.info(f"Batch of {len(added_ids)} docs added/updated in collection '{resolved_collection_name}'.")
            return added_ids
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError, ValueError) as e: logger.error(f"Failed to add documents batch: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error adding documents batch: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def search_vector_store(self, query: str, *, k: int, collection_name: Optional[str]=None, filter_metadata: Optional[Dict]=None) -> List[ContextDocument]:
        if k <= 0: raise ValueError("'k' must be positive.")
        logger.debug(f"Searching vector store (k={k}, Collection: {collection_name or 'default'}) for query: '{query[:50]}...'")
        try:
            query_embedding = await self._embedding_manager.generate_embedding(query)
            vector_storage = self._storage_manager.get_vector_storage()
            results = await vector_storage.similarity_search(query_embedding=query_embedding, k=k, collection_name=collection_name, filter_metadata=filter_metadata)
            logger.info(f"Vector store search returned {len(results)} documents.")
            return results
        except (EmbeddingError, VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to search vector store: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error searching vector store: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def delete_documents_from_vector_store(self, document_ids: List[str], *, collection_name: Optional[str]=None) -> bool:
        if not document_ids: logger.warning("delete_documents_from_vector_store called with empty ID list."); return True
        logger.debug(f"Deleting {len(document_ids)} documents from vector store (Collection: {collection_name or 'default'})...")
        try:
            vector_storage = self._storage_manager.get_vector_storage()
            success = await vector_storage.delete_documents(document_ids=document_ids, collection_name=collection_name)
            logger.info(f"Deletion attempt for {len(document_ids)} documents completed (Success: {success}).")
            return success
        except (VectorStorageError, ConfigError, StorageError) as e: logger.error(f"Failed to delete documents: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error deleting documents: {e}", exc_info=True); raise VectorStorageError(f"Unexpected error: {e}")

    async def list_rag_collections(self) -> List[str]:
        logger.debug("LLMCore.list_rag_collections called.")
        try: return await self._storage_manager.list_vector_collection_names()
        except StorageError as e: logger.error(f"Storage error listing RAG collections: {e}"); raise
        except Exception as e: logger.error(f"Unexpected error listing RAG collections: {e}", exc_info=True); raise LLMCoreError(f"Failed to list RAG collections: {e}")


    # --- User Context Item Management Methods (Session Pool / Workspace) ---
    # (add_text_context_item, add_file_context_item, update_context_item, remove_context_item,
    #  get_session_context_items, get_context_item - implementations largely unchanged)
    async def add_text_context_item(self, session_id: str, content: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        if not session_id: raise ValueError("session_id is required to add a context item.")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)
        item_id_actual = item_id or str(uuid.uuid4()); item_metadata = metadata or {}
        if ignore_char_limit: item_metadata['ignore_char_limit'] = True
        item = ContextItem(id=item_id_actual, type=ContextItemType.USER_TEXT, source_id=item_id_actual, content=content, metadata=item_metadata)
        try: provider = self._provider_manager.get_default_provider(); item.tokens = await provider.count_tokens(content, model=provider.default_model); item.original_tokens = item.tokens
        except Exception as e: logger.warning(f"Could not count tokens for user text item '{item.id}': {e}. Tokens set to None."); item.tokens = None; item.original_tokens = None
        session.add_context_item(item)
        if not is_transient: await self._session_manager.save_session(session)
        logger.info(f"Added text context item '{item.id}' to session '{session_id}' (is_transient={is_transient}, ignore_limit={ignore_char_limit}).")
        return item

    async def add_file_context_item(self, session_id: str, file_path: str, item_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, ignore_char_limit: bool = False) -> ContextItem:
        if not session_id: raise ValueError("session_id is required.")
        path_obj = pathlib.Path(file_path).expanduser().resolve()
        if not path_obj.is_file(): raise FileNotFoundError(f"File not found: {path_obj}")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)
        try:
            async with aiofiles.open(path_obj, "r", encoding="utf-8", errors="ignore") as f: content = await f.read()
        except Exception as e:
            raise LLMCoreError(f"Failed to read file content from {path_obj}: {e}")
        file_metadata = metadata or {}; file_metadata.setdefault("filename", path_obj.name); file_metadata.setdefault("original_path", str(path_obj))
        if ignore_char_limit: file_metadata['ignore_char_limit'] = True
        item_id_actual = item_id or str(uuid.uuid4())
        item = ContextItem(id=item_id_actual, type=ContextItemType.USER_FILE, source_id=str(path_obj), content=content, metadata=file_metadata)
        try: provider = self._provider_manager.get_default_provider(); item.tokens = await provider.count_tokens(content, model=provider.default_model); item.original_tokens = item.tokens
        except Exception as e: logger.warning(f"Could not count tokens for file item '{item.id}': {e}. Tokens set to None."); item.tokens = None; item.original_tokens = None
        session.add_context_item(item)
        if not is_transient: await self._session_manager.save_session(session)
        logger.info(f"Added file context item '{item.id}' (from {path_obj}) to session '{session_id}' (is_transient={is_transient}, ignore_limit={ignore_char_limit}).")
        return item

    async def update_context_item(self, session_id: str, item_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        if not session_id: raise ValueError("session_id is required.")
        is_transient = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient else await self._session_manager.load_or_create_session(session_id)
        item_to_update = session.get_context_item(item_id)
        if not item_to_update: raise LLMCoreError(f"Context item '{item_id}' not found in session '{session_id}'.")
        updated = False
        if content is not None:
            item_to_update.content = content; item_to_update.is_truncated = False
            try: provider = self._provider_manager.get_default_provider(); item_to_update.tokens = await provider.count_tokens(content, model=provider.default_model); item_to_update.original_tokens = item_to_update.tokens
            except Exception: item_to_update.tokens = None; item_to_update.original_tokens = None
            updated = True
        if metadata is not None:
            existing_ignore_limit = item_to_update.metadata.get('ignore_char_limit')
            item_to_update.metadata.update(metadata)
            if 'ignore_char_limit' not in metadata and existing_ignore_limit is not None: item_to_update.metadata['ignore_char_limit'] = existing_ignore_limit
            updated = True
        if updated:
            item_to_update.timestamp = datetime.now(timezone.utc)
            if not is_transient: await self._session_manager.save_session(session)
        logger.info(f"Updated context item '{item_id}' in session '{session_id}' (is_transient={is_transient}).")
        return item_to_update

    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        if not session_id: raise ValueError("session_id is required.")
        is_transient = session_id in self._transient_sessions_cache
        session_obj_to_modify: Optional[ChatSession] = self._transient_sessions_cache.get(session_id)
        if not session_obj_to_modify: session_obj_to_modify = await self._session_manager.load_or_create_session(session_id, system_message=None)
        if not session_obj_to_modify: logger.warning(f"Session '{session_id}' not found for removing context item."); return False
        removed = session_obj_to_modify.remove_context_item(item_id)
        if removed:
            if not is_transient: await self._session_manager.save_session(session_obj_to_modify)
            logger.info(f"Removed context item '{item_id}' from session '{session_id}' (is_transient={is_transient}).")
        else: logger.warning(f"Context item '{item_id}' not found in session '{session_id}' for removal.")
        return removed

    async def get_session_context_items(self, session_id: str) -> List[ContextItem]:
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        if not session: raise SessionNotFoundError(session_id=session_id, message="Session not found when trying to list its context items.")
        return session.context_items

    async def get_context_item(self, session_id: str, item_id: str) -> Optional[ContextItem]:
        if not session_id: raise ValueError("session_id is required.")
        session = await self.get_session(session_id)
        if not session: return None
        return session.get_context_item(item_id)

    async def get_last_interaction_context_info(self, session_id: str) -> Optional[ContextPreparationDetails]:
        if not session_id: logger.warning("get_last_interaction_context_info called without session_id."); return None
        cached_info = self._transient_last_interaction_info_cache.get(session_id)
        if cached_info: logger.debug(f"Retrieved last interaction context details from transient cache for session '{session_id}'.")
        else: logger.debug(f"No interaction context details found in transient cache for session '{session_id}'.")
        return cached_info

    async def get_last_used_rag_documents(self, session_id: str) -> Optional[List[ContextDocument]]:
        context_details = await self.get_last_interaction_context_info(session_id)
        if context_details: return context_details.rag_documents_used
        return None

    async def pin_rag_document_as_context_item(self, session_id: str, original_rag_doc_id: str, custom_item_id: Optional[str] = None, custom_metadata: Optional[Dict[str, Any]] = None) -> ContextItem:
        if not session_id: raise ValueError("session_id is required for pinning RAG document.")
        if not original_rag_doc_id: raise ValueError("original_rag_doc_id is required for pinning.")
        is_transient_session = session_id in self._transient_sessions_cache
        session = self._transient_sessions_cache[session_id] if is_transient_session else await self._session_manager.load_or_create_session(session_id)
        context_details = await self.get_last_interaction_context_info(session_id)
        last_rag_docs = context_details.rag_documents_used if context_details else None
        if not last_rag_docs: raise LLMCoreError(f"No RAG documents from the last turn found in cache for session '{session_id}'. Cannot pin.")
        doc_to_pin: Optional[ContextDocument] = next((doc for doc in last_rag_docs if doc.id == original_rag_doc_id), None)
        if not doc_to_pin: raise LLMCoreError(f"RAG document with ID '{original_rag_doc_id}' not found among last used RAG documents for session '{session_id}'.")
        new_item_id = custom_item_id or str(uuid.uuid4())
        new_item_metadata = {"pinned_from_rag": True, "original_rag_doc_id": doc_to_pin.id, "original_rag_doc_metadata": doc_to_pin.metadata, "pinned_timestamp": datetime.now(timezone.utc).isoformat()} # type: ignore
        if custom_metadata: new_item_metadata.update(custom_metadata)
        pinned_item = ContextItem(id=new_item_id, type=ContextItemType.RAG_SNIPPET, source_id=doc_to_pin.id, content=doc_to_pin.content, metadata=new_item_metadata, timestamp=datetime.now(timezone.utc))
        try: provider = self._provider_manager.get_default_provider(); pinned_item.tokens = await provider.count_tokens(pinned_item.content, model=provider.default_model); pinned_item.original_tokens = pinned_item.tokens
        except Exception as e: logger.warning(f"Could not count tokens for pinned RAG snippet '{pinned_item.id}': {e}. Tokens set to None."); pinned_item.tokens = None; pinned_item.original_tokens = None
        session.add_context_item(pinned_item)
        if not is_transient_session: await self._session_manager.save_session(session)
        logger.info(f"Pinned RAG document '{original_rag_doc_id}' as new context item '{pinned_item.id}' in session '{session_id}' (is_transient={is_transient_session}).")
        return pinned_item

    # --- Provider Info Methods ---
    # (get_available_providers, get_models_for_provider - implementations largely unchanged)
    def get_available_providers(self) -> List[str]:
        logger.debug("LLMCore.get_available_providers called.")
        return self._provider_manager.get_available_providers()

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        logger.debug(f"LLMCore.get_models_for_provider for: {provider_name}")
        try: provider = self._provider_manager.get_provider(provider_name); return provider.get_available_models()
        except (ConfigError, ProviderError) as e: logger.error(f"Error getting models for provider '{provider_name}': {e}"); raise
        except Exception as e: logger.error(f"Unexpected error getting models for provider '{provider_name}': {e}", exc_info=True); raise ProviderError(provider_name, f"Failed to retrieve models due to an unexpected error: {e}")

    # --- Context Preset Management API Methods ---
    # (save_context_preset, load_context_preset, list_context_presets, delete_context_preset,
    #  update_context_preset_description, rename_context_preset - implementations largely unchanged)
    async def save_context_preset(self, preset_name: str, items: List[ContextPresetItem], description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ContextPreset:
        logger.info(f"Saving context preset '{preset_name}' with {len(items)} items.")
        try:
            preset = ContextPreset(name=preset_name, items=items, description=description, metadata=metadata or {}, updated_at=datetime.now(timezone.utc))
            session_storage = self._storage_manager.get_session_storage()
            await session_storage.save_context_preset(preset)
            logger.info(f"Context preset '{preset.name}' saved successfully.")
            return preset
        except ValueError as ve: logger.error(f"Invalid data for context preset '{preset_name}': {ve}"); raise
        except StorageError as se: logger.error(f"Failed to save context preset '{preset_name}' to storage: {se}", exc_info=True); raise
        except Exception as e: logger.error(f"Unexpected error saving context preset '{preset_name}': {e}", exc_info=True); raise StorageError(f"Unexpected error occurred while saving preset '{preset_name}': {e}")

    async def load_context_preset(self, preset_name: str) -> Optional[ContextPreset]:
        logger.info(f"Loading context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            preset = await session_storage.get_context_preset(preset_name)
            if preset: logger.info(f"Context preset '{preset_name}' loaded successfully with {len(preset.items)} items.")
            else: logger.info(f"Context preset '{preset_name}' not found in storage.")
            return preset
        except StorageError as se: logger.error(f"Failed to load context preset '{preset_name}' from storage: {se}", exc_info=True); raise
        except Exception as e: logger.error(f"Unexpected error loading context preset '{preset_name}': {e}", exc_info=True); raise StorageError(f"Unexpected error occurred while loading preset '{preset_name}': {e}")

    async def list_context_presets(self) -> List[Dict[str, Any]]:
        logger.info("Listing all available context presets.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            presets_meta = await session_storage.list_context_presets()
            logger.info(f"Found {len(presets_meta)} context presets.")
            return presets_meta
        except StorageError as se: logger.error(f"Failed to list context presets from storage: {se}", exc_info=True); raise
        except Exception as e: logger.error(f"Unexpected error listing context presets: {e}", exc_info=True); raise StorageError(f"Unexpected error occurred while listing presets: {e}")

    async def delete_context_preset(self, preset_name: str) -> bool:
        logger.info(f"Deleting context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            deleted = await session_storage.delete_context_preset(preset_name)
            if deleted: logger.info(f"Context preset '{preset_name}' deleted successfully from storage.")
            else: logger.warning(f"Context preset '{preset_name}' not found in storage for deletion.")
            return deleted
        except StorageError as se: logger.error(f"Failed to delete context preset '{preset_name}' from storage: {se}", exc_info=True); raise
        except Exception as e: logger.error(f"Unexpected error deleting context preset '{preset_name}': {e}", exc_info=True); raise StorageError(f"Unexpected error occurred while deleting preset '{preset_name}': {e}")

    async def update_context_preset_description(self, preset_name: str, new_description: str) -> bool:
        logger.info(f"Updating description for context preset '{preset_name}'.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            preset = await session_storage.get_context_preset(preset_name)
            if not preset: logger.warning(f"Context preset '{preset_name}' not found for updating description."); return False
            preset.description = new_description; preset.updated_at = datetime.now(timezone.utc)
            await session_storage.save_context_preset(preset)
            logger.info(f"Description for context preset '{preset_name}' updated successfully.")
            return True
        except StorageError as se: logger.error(f"Failed to update description for context preset '{preset_name}': {se}", exc_info=True); raise
        except Exception as e: logger.error(f"Unexpected error updating description for preset '{preset_name}': {e}", exc_info=True); raise StorageError(f"Unexpected error updating preset description for '{preset_name}': {e}")

    async def rename_context_preset(self, old_name: str, new_name: str) -> bool:
        logger.info(f"Attempting to rename context preset from '{old_name}' to '{new_name}'.")
        if not new_name or not new_name.strip(): raise ValueError("New preset name cannot be empty or just whitespace.")
        try:
            session_storage = self._storage_manager.get_session_storage()
            renamed_successfully = await session_storage.rename_context_preset(old_name, new_name)
            if renamed_successfully: logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}'.")
            else: logger.warning(f"Failed to rename context preset '{old_name}' to '{new_name}' (check storage logs for details like 'not found' or 'new name exists').")
            return renamed_successfully
        except ValueError: raise
        except StorageError as se: logger.error(f"Storage error occurred while renaming context preset '{old_name}' to '{new_name}': {se}", exc_info=True); raise
        except Exception as e: logger.error(f"Unexpected error renaming context preset '{old_name}' to '{new_name}': {e}", exc_info=True); raise StorageError(f"Unexpected error occurred during preset rename: {e}")


    # --- Utility / Cleanup ---
    # (close, __aenter__, __aexit__ - implementations unchanged)
    async def close(self):
        logger.info("LLMCore.close() called. Cleaning up resources...")
        close_tasks = [self._provider_manager.close_providers(), self._storage_manager.close_storages(), self._embedding_manager.close()]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
             if isinstance(result, Exception): logger.error(f"Error during LLMCore resource cleanup: {result}", exc_info=result)
        self._transient_last_interaction_info_cache.clear(); self._transient_sessions_cache.clear()
        logger.info("LLMCore resources cleanup complete and transient caches cleared.")

    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc_val, exc_tb): await self.close()
