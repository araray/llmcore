# src/llmcore/embedding/manager.py
"""
Embedding Model Manager for LLMCore.

Handles the dynamic loading and management of text embedding model instances
based on configuration. Allows for multiple embedding models to be used
concurrently based on identifiers.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore [no-redef]


from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel
from .google import GoogleAIEmbedding
from .ollama import OllamaEmbedding
from .openai import OpenAIEmbedding
from .sentence_transformer import SentenceTransformerEmbedding

logger = logging.getLogger(__name__)

# Mapping from config provider type string to BaseEmbeddingModel class
EMBEDDING_PROVIDER_CLASS_MAP: Dict[str, Type[BaseEmbeddingModel]] = {
    "sentence-transformers": SentenceTransformerEmbedding,
    "openai": OpenAIEmbedding,
    "google": GoogleAIEmbedding, # Maps to GoogleAIEmbedding
    "ollama": OllamaEmbedding,
}


class EmbeddingManager:
    """
    Manages the initialization and access to text embedding models.

    Reads configuration, determines the appropriate embedding model type based
    on an identifier, instantiates it, handles its asynchronous initialization,
    and caches instances.
    """
    _global_config: ConfyConfig
    _initialized_models: Dict[str, BaseEmbeddingModel] # Cache for model instances
    _model_init_locks: Dict[str, asyncio.Lock] # Locks per model identifier

    def __init__(self, global_config: ConfyConfig):
        """
        Initializes the EmbeddingManager.

        Args:
            global_config: The main LLMCore configuration object (ConfyConfig instance).
        """
        self._global_config = global_config
        self._initialized_models = {}
        self._model_init_locks = {} # Initialize locks dictionary
        logger.info("EmbeddingManager initialized. Models will be loaded on demand via get_model().")

    async def get_model(self, model_identifier: str) -> BaseEmbeddingModel:
        """
        Retrieves or creates, initializes, and caches an embedding model instance
        based on the provided identifier.

        The identifier can be a model name (implying sentence-transformers) or
        in the format "provider_type:model_name" (e.g., "openai:text-embedding-3-small").

        Args:
            model_identifier: The identifier for the desired embedding model.

        Returns:
            An initialized BaseEmbeddingModel instance.

        Raises:
            ConfigError: If configuration for the model/provider is invalid or missing.
            EmbeddingError: If the model fails to initialize.
        """
        if not model_identifier:
            raise ValueError("model_identifier cannot be empty.")

        # Get or create a lock for this specific model_identifier
        if model_identifier not in self._model_init_locks:
            self._model_init_locks[model_identifier] = asyncio.Lock()

        lock = self._model_init_locks[model_identifier]

        async with lock: # Ensure only one coroutine initializes a specific model_identifier
            if model_identifier in self._initialized_models:
                logger.debug(f"Returning cached embedding model for identifier: '{model_identifier}'")
                return self._initialized_models[model_identifier]

            logger.info(f"Attempting to initialize embedding model for identifier: '{model_identifier}'")

            provider_type_for_map: str
            actual_model_name_for_class: str

            if ":" in model_identifier:
                parts = model_identifier.split(":", 1)
                provider_type_for_map = parts[0].lower()
                actual_model_name_for_class = parts[1]
                if provider_type_for_map not in EMBEDDING_PROVIDER_CLASS_MAP:
                    raise ConfigError(
                        f"Unknown embedding provider type '{provider_type_for_map}' in identifier '{model_identifier}'. "
                        f"Known types: {list(EMBEDDING_PROVIDER_CLASS_MAP.keys())}"
                    )
            else:
                # Default to sentence-transformers if no prefix
                provider_type_for_map = "sentence-transformers"
                actual_model_name_for_class = model_identifier

            EmbeddingCls = EMBEDDING_PROVIDER_CLASS_MAP[provider_type_for_map]

            # Get the base configuration block for this provider type from global config
            # e.g., if provider_type_for_map is "openai", get "[embedding.openai]"
            # if provider_type_for_map is "sentence-transformers", get "[embedding.sentence_transformer]"
            config_section_key_for_provider = provider_type_for_map # Direct match now

            # For sentence-transformers, the config section might be named 'sentence_transformer' (singular)
            if provider_type_for_map == "sentence-transformers":
                config_section_key_for_provider = "sentence_transformer"

            provider_base_config = self._global_config.get(f"embedding.{config_section_key_for_provider}", {})
            if not isinstance(provider_base_config, dict): # Should be a dict from confy
                logger.warning(f"Configuration for embedding provider type '{config_section_key_for_provider}' is not a dictionary. Using empty config.")
                provider_base_config = {}

            # Create a specific configuration dictionary for this model instance
            instance_config = provider_base_config.copy()

            # The individual embedding classes expect the model name under a specific key
            if EmbeddingCls == SentenceTransformerEmbedding:
                instance_config["model_name_or_path"] = actual_model_name_for_class
            elif EmbeddingCls in [OpenAIEmbedding, GoogleAIEmbedding, OllamaEmbedding]:
                # These classes expect 'default_model' in their config dict to pick the model
                instance_config["default_model"] = actual_model_name_for_class
            else:
                # Fallback or for new classes, could set a generic key
                instance_config["model_name"] = actual_model_name_for_class


            logger.debug(f"Instantiating {EmbeddingCls.__name__} with model '{actual_model_name_for_class}' "
                         f"using derived config: {instance_config}")

            try:
                model_instance = EmbeddingCls(instance_config)
                await model_instance.initialize()
                self._initialized_models[model_identifier] = model_instance
                logger.info(f"Successfully initialized and cached embedding model for identifier: '{model_identifier}' ({EmbeddingCls.__name__})")
                return model_instance
            except ImportError as e_imp:
                logger.error(f"Failed to initialize model '{model_identifier}': Missing library for {provider_type_for_map}. Error: {e_imp}")
                raise EmbeddingError(model_name=model_identifier, message=f"Missing dependency for {provider_type_for_map}: {e_imp}") from e_imp
            except Exception as e_init:
                logger.error(f"Failed to initialize model '{model_identifier}' ({EmbeddingCls.__name__}): {e_init}", exc_info=True)
                raise EmbeddingError(model_name=model_identifier, message=f"Initialization failed: {e_init}") from e_init

    async def generate_embedding(self, text: str, model_identifier: Optional[str] = None) -> List[float]:
        """
        Generates an embedding for a single text using the specified or default model.

        Args:
            text: The text to embed.
            model_identifier: Optional. The identifier of the model to use
                              (e.g., "openai:text-embedding-3-small" or "all-MiniLM-L6-v2").
                              If None, uses the `llmcore.default_embedding_model` from config.

        Returns:
            The generated embedding as a list of floats.

        Raises:
            EmbeddingError: If the model is not available or embedding fails.
            ConfigError: If the default model identifier is not configured and model_identifier is None.
        """
        effective_identifier = model_identifier
        if not effective_identifier:
            effective_identifier = self._global_config.get('llmcore.default_embedding_model')
            if not effective_identifier:
                raise ConfigError("No model_identifier provided and 'llmcore.default_embedding_model' is not set.")

        logger.debug(f"generate_embedding called for text (len: {len(text)}) using model_identifier: '{effective_identifier}'")
        model_instance = await self.get_model(effective_identifier)
        return await model_instance.generate_embedding(text)

    async def generate_embeddings(self, texts: List[str], model_identifier: Optional[str] = None) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts using the specified or default model.

        Args:
            texts: The list of texts to embed.
            model_identifier: Optional. The identifier of the model to use.
                              If None, uses the `llmcore.default_embedding_model`.

        Returns:
            A list of generated embeddings.

        Raises:
            EmbeddingError: If the model is not available or embedding fails.
            ConfigError: If the default model identifier is not configured and model_identifier is None.
        """
        effective_identifier = model_identifier
        if not effective_identifier:
            effective_identifier = self._global_config.get('llmcore.default_embedding_model')
            if not effective_identifier:
                raise ConfigError("No model_identifier provided and 'llmcore.default_embedding_model' is not set.")

        logger.debug(f"generate_embeddings called for {len(texts)} texts using model_identifier: '{effective_identifier}'")
        model_instance = await self.get_model(effective_identifier)
        return await model_instance.generate_embeddings(texts)

    async def close(self) -> None:
        """Cleans up resources used by all cached embedding models."""
        logger.info(f"Closing EmbeddingManager and {len(self._initialized_models)} cached embedding model(s)...")
        close_tasks = []
        for model_id, model_instance in self._initialized_models.items():
            if hasattr(model_instance, 'close') and asyncio.iscoroutinefunction(model_instance.close):
                logger.debug(f"Queueing close for model: {model_id}")
                close_tasks.append(model_instance.close()) # type: ignore
            else:
                logger.debug(f"Model {model_id} (type: {type(model_instance).__name__}) does not have an async close method.")

        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                model_id_closed = list(self._initialized_models.keys())[i] # Assuming order is maintained
                if isinstance(result, Exception):
                    logger.error(f"Error closing embedding model '{model_id_closed}': {result}", exc_info=result)

        self._initialized_models.clear()
        self._model_init_locks.clear()
        logger.info("EmbeddingManager closed and cache cleared.")
