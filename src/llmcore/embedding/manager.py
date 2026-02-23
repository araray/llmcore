# src/llmcore/embedding/manager.py
"""
Embedding Model Manager for LLMCore.

Handles the dynamic loading and management of text embedding model instances
based on configuration. Allows for multiple embedding models to be used
concurrently based on identifiers.

UPDATED: Added storage_manager parameter to __init__ for API compatibility.
UPDATED: Added async initialize() method for future async initialization needs.
"""

import asyncio
import logging
from typing import Any

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = dict[str, Any]  # type: ignore [no-redef]


from ..exceptions import ConfigError, EmbeddingError
from ..storage.manager import StorageManager
from .base import BaseEmbeddingModel
from .google import GoogleAIEmbedding
from .ollama import OllamaEmbedding
from .openai import OpenAIEmbedding
from .sentence_transformer import SentenceTransformerEmbedding

logger = logging.getLogger(__name__)

# Mapping from config provider type string to BaseEmbeddingModel class
EMBEDDING_PROVIDER_CLASS_MAP: dict[str, type[BaseEmbeddingModel]] = {
    "sentence-transformers": SentenceTransformerEmbedding,
    "openai": OpenAIEmbedding,
    "google": GoogleAIEmbedding,
    "ollama": OllamaEmbedding,
}

# Lazy-loaded providers that require optional dependencies.
# These are resolved in _resolve_provider_class() to avoid ImportError
# at module load time when the SDK is not installed.
_LAZY_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    # key: (module_path_relative_to_embedding_pkg, class_name)
    "cohere": (".cohere", "CohereEmbedding"),
    "voyageai": (".voyageai", "VoyageAIEmbedding"),
    "voyage": (".voyageai", "VoyageAIEmbedding"),  # alias
}


class EmbeddingManager:
    """
    Manages the initialization and access to text embedding models.

    Reads configuration, determines the appropriate embedding model type based
    on an identifier, instantiates it, handles its asynchronous initialization,
    and caches instances.

    UPDATED: Now accepts an optional storage_manager parameter for API
    compatibility with LLMCore.create().
    """

    _global_config: ConfyConfig
    _storage_manager: StorageManager | None
    _initialized_models: dict[str, BaseEmbeddingModel]
    _model_init_locks: dict[str, asyncio.Lock]
    _initialized: bool

    def __init__(self, global_config: ConfyConfig, storage_manager: StorageManager | None = None):
        """
        Initializes the EmbeddingManager.

        Args:
            global_config: The main LLMCore configuration object (ConfyConfig instance).
            storage_manager: Optional StorageManager instance. Currently stored for
                           potential future use (e.g., caching embeddings to vector store).
                           This parameter is accepted for API compatibility with LLMCore.
        """
        self._global_config = global_config
        self._storage_manager = storage_manager
        self._initialized_models = {}
        self._model_init_locks = {}
        self._initialized = False
        logger.info(
            "EmbeddingManager initialized. Models will be loaded on demand via get_model()."
        )

    async def initialize(self) -> None:
        """
        Asynchronous initialization hook for the EmbeddingManager.

        Currently, embedding models are loaded on-demand via get_model().
        This method is provided for:
        1. API compatibility with LLMCore's async initialization pattern
        2. Future support for pre-loading default models
        3. Post-construction async setup tasks

        This method is idempotent - calling it multiple times is safe.
        """
        if self._initialized:
            logger.debug("EmbeddingManager already initialized, skipping async initialize()")
            return

        # Future: Could pre-load the default embedding model here
        # default_model = self._global_config.get('llmcore.default_embedding_model')
        # if default_model:
        #     await self.get_model(default_model)

        self._initialized = True
        logger.debug("EmbeddingManager async initialize() complete")

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

        async with lock:
            if model_identifier in self._initialized_models:
                logger.debug(
                    f"Returning cached embedding model for identifier: '{model_identifier}'"
                )
                return self._initialized_models[model_identifier]

            logger.info(
                f"Attempting to initialize embedding model for identifier: '{model_identifier}'"
            )

            provider_type_for_map: str
            actual_model_name_for_class: str

            if ":" in model_identifier:
                parts = model_identifier.split(":", 1)
                provider_type_for_map = parts[0].lower()
                actual_model_name_for_class = parts[1]
                if provider_type_for_map not in EMBEDDING_PROVIDER_CLASS_MAP:
                    # Try lazy-loaded providers
                    resolved = self._resolve_lazy_provider(provider_type_for_map)
                    if resolved is None:
                        raise ConfigError(
                            f"Unknown embedding provider type '{provider_type_for_map}' in identifier '{model_identifier}'. "
                            f"Known types: {list(EMBEDDING_PROVIDER_CLASS_MAP.keys()) + list(_LAZY_PROVIDER_MAP.keys())}"
                        )
            else:
                # Default to sentence-transformers if no prefix
                provider_type_for_map = "sentence-transformers"
                actual_model_name_for_class = model_identifier

            EmbeddingCls = EMBEDDING_PROVIDER_CLASS_MAP.get(provider_type_for_map)
            if EmbeddingCls is None:
                EmbeddingCls = self._resolve_lazy_provider(provider_type_for_map)
            if EmbeddingCls is None:
                raise ConfigError(
                    f"Could not resolve embedding provider '{provider_type_for_map}'."
                )

            # Get the base configuration block for this provider type from global config
            config_section_key_for_provider = provider_type_for_map

            # For sentence-transformers, the config section might be named 'sentence_transformer' (singular)
            if provider_type_for_map == "sentence-transformers":
                config_section_key_for_provider = "sentence_transformer"

            provider_base_config = self._global_config.get(
                f"embedding.{config_section_key_for_provider}", {}
            )
            if not isinstance(provider_base_config, dict):
                logger.warning(
                    f"Configuration for embedding provider type '{config_section_key_for_provider}' is not a dictionary. Using empty config."
                )
                provider_base_config = {}

            # Create a specific configuration dictionary for this model instance
            instance_config = provider_base_config.copy()

            # The individual embedding classes expect the model name under a specific key
            if EmbeddingCls == SentenceTransformerEmbedding:
                instance_config["model_name_or_path"] = actual_model_name_for_class
            else:
                # OpenAI, Google, Ollama, Cohere, VoyageAI all use 'default_model'
                instance_config["default_model"] = actual_model_name_for_class

            logger.debug(
                f"Instantiating {EmbeddingCls.__name__} with model '{actual_model_name_for_class}' "
                f"using derived config: {instance_config}"
            )

            try:
                model_instance = EmbeddingCls(instance_config)
                await model_instance.initialize()
                self._initialized_models[model_identifier] = model_instance
                logger.info(
                    f"Successfully initialized and cached embedding model for identifier: '{model_identifier}' ({EmbeddingCls.__name__})"
                )
                return model_instance
            except ImportError as e_imp:
                logger.error(
                    f"Failed to initialize model '{model_identifier}': Missing library for {provider_type_for_map}. Error: {e_imp}"
                )
                raise EmbeddingError(
                    model_name=model_identifier,
                    message=f"Missing dependency for {provider_type_for_map}: {e_imp}",
                ) from e_imp
            except Exception as e_init:
                logger.error(
                    f"Failed to initialize model '{model_identifier}' ({EmbeddingCls.__name__}): {e_init}",
                    exc_info=True,
                )
                raise EmbeddingError(
                    model_name=model_identifier, message=f"Initialization failed: {e_init}"
                ) from e_init

    @staticmethod
    def _resolve_lazy_provider(provider_type: str) -> type[BaseEmbeddingModel] | None:
        """Resolve a lazily-loaded embedding provider class.

        Looks up ``provider_type`` in :data:`_LAZY_PROVIDER_MAP`, performs
        a dynamic import, registers the class in
        :data:`EMBEDDING_PROVIDER_CLASS_MAP` for future fast-path access,
        and returns the class.

        Returns:
            The embedding model class, or *None* if *provider_type* is
            unknown or the required SDK is not installed.
        """
        spec = _LAZY_PROVIDER_MAP.get(provider_type)
        if spec is None:
            return None

        module_path, class_name = spec
        try:
            import importlib

            mod = importlib.import_module(module_path, package="llmcore.embedding")
            cls: type[BaseEmbeddingModel] = getattr(mod, class_name)
            # Cache for subsequent calls
            EMBEDDING_PROVIDER_CLASS_MAP[provider_type] = cls
            logger.debug("Lazy-loaded embedding provider '%s' → %s.", provider_type, class_name)
            return cls
        except ImportError as exc:
            logger.warning(
                "Embedding provider '%s' requires library '%s' which is not installed: %s",
                provider_type,
                module_path,
                exc,
            )
            return None
        except Exception as exc:
            logger.error("Failed to lazy-load embedding provider '%s': %s", provider_type, exc)
            return None

    async def generate_embedding(
        self, text: str, model_identifier: str | None = None
    ) -> list[float]:
        """
        Generates an embedding for a single text using the specified or default model.

        Args:
            text: The text to embed.
            model_identifier: Optional. The identifier of the model to use.
                              If None, uses the `llmcore.default_embedding_model` from config.

        Returns:
            The generated embedding as a list of floats.

        Raises:
            ConfigError: If no model identifier is provided and no default is configured.
            EmbeddingError: If embedding generation fails.
        """
        if not model_identifier:
            model_identifier = self._global_config.get("llmcore.default_embedding_model")
            if not model_identifier:
                raise ConfigError(
                    "No model_identifier provided and 'llmcore.default_embedding_model' is not set in config."
                )

        model = await self.get_model(model_identifier)
        return await model.generate_embedding(text)

    async def generate_embeddings(
        self, texts: list[str], model_identifier: str | None = None
    ) -> list[list[float]]:
        """
        Generates embeddings for multiple texts using the specified or default model.

        Args:
            texts: List of texts to embed.
            model_identifier: Optional. The identifier of the model to use.
                              If None, uses the `llmcore.default_embedding_model` from config.

        Returns:
            List of embeddings, each as a list of floats.

        Raises:
            ConfigError: If no model identifier is provided and no default is configured.
            EmbeddingError: If embedding generation fails.
        """
        if not model_identifier:
            model_identifier = self._global_config.get("llmcore.default_embedding_model")
            if not model_identifier:
                raise ConfigError(
                    "No model_identifier provided and 'llmcore.default_embedding_model' is not set in config."
                )

        model = await self.get_model(model_identifier)
        return await model.generate_embeddings(texts)

    def get_cached_models(self) -> list[str]:
        """Returns a list of currently cached model identifiers."""
        return list(self._initialized_models.keys())

    async def close(self) -> None:
        """
        Closes all cached embedding model instances and releases resources.
        """
        logger.info("Closing EmbeddingManager and all cached models...")
        for model_id, model_instance in self._initialized_models.items():
            try:
                if hasattr(model_instance, "close"):
                    await model_instance.close()
                logger.debug(f"Closed embedding model: {model_id}")
            except Exception as e:
                logger.error(f"Error closing embedding model '{model_id}': {e}", exc_info=True)
        self._initialized_models.clear()
        self._initialized = False
        logger.info("EmbeddingManager closed.")
