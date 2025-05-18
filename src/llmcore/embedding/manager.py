# src/llmcore/embedding/manager.py
"""
Embedding Model Manager for LLMCore.

Handles the dynamic loading and management of text embedding model instances
based on configuration.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore

from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel
from .google import GoogleAIEmbedding  # Added for Phase 3
from .ollama import OllamaEmbedding  # Added for Phase 3
from .openai import OpenAIEmbedding  # Added for Phase 3
# Import concrete implementations
from .sentence_transformer import SentenceTransformerEmbedding

logger = logging.getLogger(__name__)

# --- Mapping from config identifier prefix to class ---
# Use prefixes in the 'llmcore.default_embedding_model' string
# Default (no prefix or 'sentence-transformers') maps to SentenceTransformer
EMBEDDING_MAP: Dict[str, Type[BaseEmbeddingModel]] = {
    "sentence-transformers": SentenceTransformerEmbedding, # Explicit prefix
    "default": SentenceTransformerEmbedding, # Default if no prefix
    "openai": OpenAIEmbedding, # Added for Phase 3
    "google": GoogleAIEmbedding, # Added for Phase 3
    "ollama": OllamaEmbedding, # Added for Phase 3
}
# --- End Mapping ---


class EmbeddingManager:
    """
    Manages the initialization and access to text embedding models.

    Reads configuration, determines the appropriate embedding model type,
    instantiates it, and handles its asynchronous initialization.
    """
    _embedding_model: Optional[BaseEmbeddingModel] = None
    _config: ConfyConfig
    _is_initialized: bool = False
    _initialization_lock = asyncio.Lock() # Ensure initialize runs only once

    def __init__(self, config: ConfyConfig):
        """
        Initializes the EmbeddingManager.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
        """
        self._config = config
        logger.info("EmbeddingManager initialized.")
        # Defer actual model loading to initialize_embedding_model()

    async def initialize_embedding_model(self) -> None:
        """
        Loads and initializes the configured embedding model asynchronously.

        This method should be called after the manager is created and before
        calling get_embedding_model. It's idempotent.

        Raises:
            ConfigError: If the embedding model configuration is invalid or missing.
            EmbeddingError: If the embedding model fails to initialize.
        """
        async with self._initialization_lock:
            if self._is_initialized:
                logger.debug("Embedding model already initialized.")
                return

            logger.info("Initializing embedding model...")
            model_identifier = self._config.get('llmcore.default_embedding_model')
            if not model_identifier:
                logger.warning("No default embedding model specified ('llmcore.default_embedding_model'). "
                               "RAG functionality will be unavailable.")
                self._is_initialized = True # Mark as initialized (no model)
                return

            # Parse identifier (e.g., "openai:text-embedding-3-small", "ollama:nomic-embed-text", "all-MiniLM-L6-v2")
            model_type_key = "default" # Assume sentence-transformer by default
            model_name_or_path = model_identifier

            if ":" in model_identifier:
                parts = model_identifier.split(":", 1)
                prefix = parts[0].lower()
                if prefix in EMBEDDING_MAP:
                    model_type_key = prefix
                    model_name_or_path = parts[1] # The part after the prefix is the model name/path
                else:
                    logger.warning(f"Unknown embedding model prefix '{prefix}' in '{model_identifier}'. "
                                   f"Assuming it's a Sentence Transformer model name/path.")
                    # Keep model_type_key as "default" and use full identifier as path
                    model_name_or_path = model_identifier # Use the full string
            else:
                # No prefix, assume it's a sentence-transformer model name/path
                model_type_key = "default"
                model_name_or_path = model_identifier

            embedding_cls = EMBEDDING_MAP.get(model_type_key)
            if not embedding_cls:
                raise ConfigError(f"Internal error: No embedding class mapped for type key '{model_type_key}'.")

            # Prepare config for the specific embedding model class
            # Get the relevant section from the main config, e.g., [embedding.openai]
            # Use the model_type_key (e.g., 'openai', 'google', 'ollama', 'sentence_transformer')
            # to find the correct config section. Use 'sentence_transformer' for the 'default' key.
            config_section_key = model_type_key if model_type_key != "default" else "sentence_transformer"
            model_specific_config = self._config.get(f"embedding.{config_section_key}", {})

            # Add the parsed model name/path to the specific config dict
            # This ensures the embedding class constructor receives the correct model identifier.
            if embedding_cls == SentenceTransformerEmbedding:
                model_specific_config["model_name_or_path"] = model_name_or_path
            elif embedding_cls in [OpenAIEmbedding, GoogleAIEmbedding, OllamaEmbedding]:
                 # For API-based or Ollama models, the part after the prefix is the model name
                 model_specific_config["default_model"] = model_name_or_path
                 # We might also need to pass the API key or host if not handled internally by the class
                 # The embedding classes are designed to look for keys within their passed config dict.
                 # Example: OpenAIEmbedding needs 'api_key', 'base_url', 'timeout', 'default_model'
                 # Example: GoogleAIEmbedding needs 'api_key', 'default_model'
                 # Example: OllamaEmbedding needs 'host', 'timeout', 'default_model'
                 # The manager copies the relevant section, e.g., [embedding.openai]
                 # into model_specific_config.

            try:
                logger.info(f"Loading embedding model type '{config_section_key}' "
                            f"with model identifier '{model_name_or_path}' "
                            f"using config: {model_specific_config}")

                # Instantiate the class with its specific config section
                self._embedding_model = embedding_cls(model_specific_config)

                # Call the model's own async initialize method
                await self._embedding_model.initialize()
                logger.info(f"Embedding model '{model_identifier}' initialized successfully.")
                self._is_initialized = True
            except ImportError as e:
                logger.error(f"Failed to initialize embedding model '{model_identifier}': Missing required library. "
                             f"Install dependencies for '{config_section_key}' (e.g., 'pip install llmcore[{config_section_key}]'). Error: {e}")
                self._embedding_model = None
                self._is_initialized = True # Mark as initialized (failed)
                raise EmbeddingError(model_name=model_identifier, message=f"Initialization failed due to missing dependency: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model '{model_identifier}': {e}", exc_info=True)
                self._embedding_model = None
                self._is_initialized = True # Mark as initialized (failed)
                raise EmbeddingError(model_name=model_identifier, message=f"Initialization failed: {e}")


    def get_embedding_model(self) -> BaseEmbeddingModel:
        """
        Returns the initialized embedding model instance.

        Raises:
            EmbeddingError: If the embedding model is not configured, not initialized,
                            or failed to initialize. Call `initialize_embedding_model` first.
        """
        if not self._is_initialized:
             raise EmbeddingError(message="EmbeddingManager is not initialized. Call and await initialize_embedding_model() first.")
        if self._embedding_model is None:
            model_identifier = self._config.get('llmcore.default_embedding_model', 'None')
            if model_identifier == 'None':
                 raise EmbeddingError(message="No embedding model configured ('llmcore.default_embedding_model'). RAG unavailable.")
            else:
                 raise EmbeddingError(
                     model_name=model_identifier,
                     message="Embedding model failed to initialize (check logs for details). RAG unavailable."
                 )
        return self._embedding_model

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text using the configured model.

        Ensures the model is initialized before generating.

        Args:
            text: The text to embed.

        Returns:
            The generated embedding as a list of floats.

        Raises:
            EmbeddingError: If the model is not available or embedding fails.
        """
        model = self.get_embedding_model() # Will raise if not initialized/failed
        return await model.generate_embedding(text)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts using the configured model.

        Ensures the model is initialized before generating.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of generated embeddings.

        Raises:
            EmbeddingError: If the model is not available or embedding fails.
        """
        model = self.get_embedding_model() # Will raise if not initialized/failed
        return await model.generate_embeddings(texts)

    async def close(self) -> None:
        """Cleans up resources used by the embedding model."""
        logger.info("Closing embedding model resources...")
        if self._embedding_model and hasattr(self._embedding_model, 'close') and asyncio.iscoroutinefunction(self._embedding_model.close):
            try:
                await self._embedding_model.close() # type: ignore
                logger.info("Embedding model resources closed.")
            except Exception as e:
                logger.error(f"Error closing embedding model: {e}", exc_info=True)
        self._embedding_model = None
        self._is_initialized = False # Reset state
        logger.info("EmbeddingManager closed.")
