# src/llmcore/embedding/manager.py
"""
Embedding Model Manager for LLMCore.

Handles the dynamic loading and management of text embedding model instances
based on configuration.
"""

import asyncio
import logging
from typing import Dict, Any, Type, Optional, List

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore

from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

# Import concrete implementations (add more as they are created)
from .sentence_transformer import SentenceTransformerEmbedding
# from .openai import OpenAIEmbedding # Example for future
# from .google import GoogleAIEmbedding # Example for future

logger = logging.getLogger(__name__)

# --- Mapping from config identifier prefix to class ---
# Use prefixes in the 'llmcore.default_embedding_model' string
# Default (no prefix) maps to SentenceTransformer
EMBEDDING_MAP: Dict[str, Type[BaseEmbeddingModel]] = {
    "sentence-transformers": SentenceTransformerEmbedding, # Explicit prefix
    "default": SentenceTransformerEmbedding, # Default if no prefix
    # "openai": OpenAIEmbedding, # Add when implemented
    # "google": GoogleAIEmbedding, # Add when implemented
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
                # Set initialized flag even if no model is loaded, to prevent re-attempts
                self._is_initialized = True
                return # No model to load

            # Parse identifier (e.g., "openai:text-embedding-3-small", "all-MiniLM-L6-v2")
            model_type_key = "default" # Assume sentence-transformer by default
            model_name_or_path = model_identifier

            if ":" in model_identifier:
                parts = model_identifier.split(":", 1)
                prefix = parts[0].lower()
                if prefix in EMBEDDING_MAP:
                    model_type_key = prefix
                    model_name_or_path = parts[1]
                else:
                    logger.warning(f"Unknown embedding model prefix '{prefix}' in '{model_identifier}'. "
                                   f"Assuming it's a Sentence Transformer model name/path.")
                    # Keep model_type_key as "default" and use full identifier as path

            embedding_cls = EMBEDDING_MAP.get(model_type_key)
            if not embedding_cls:
                # This should not happen if EMBEDDING_MAP is correct, but check defensively
                raise ConfigError(f"Internal error: No embedding class mapped for type key '{model_type_key}'.")

            # Prepare config for the specific embedding model class
            # For SentenceTransformer, we need 'model_name_or_path' and optional 'device'
            # For others (like OpenAI), we might need API keys from their specific sections
            model_config: Dict[str, Any] = {}
            if embedding_cls == SentenceTransformerEmbedding:
                model_config["model_name_or_path"] = model_name_or_path
                # Get device preference from sentence_transformer specific config if available
                st_config = self._config.get("embedding.sentence_transformer", {})
                model_config["device"] = st_config.get("device")
            # Add elif blocks here for other providers (OpenAI, Google)
            # elif embedding_cls == OpenAIEmbedding:
            #     oa_config = self._config.get("embedding.openai", {})
            #     model_config["api_key"] = oa_config.get("api_key") # Or inherit from main provider?
            #     model_config["model_name"] = model_name_or_path or oa_config.get("default_model")
            # elif embedding_cls == GoogleAIEmbedding:
            #     gg_config = self._config.get("embedding.google", {})
            #     model_config["api_key"] = gg_config.get("api_key")
            #     model_config["model_name"] = model_name_or_path or gg_config.get("default_model")

            try:
                logger.info(f"Loading embedding model type '{model_type_key}' with config: {model_config}")
                self._embedding_model = embedding_cls(model_config)
                # Call the model's own async initialize method
                await self._embedding_model.initialize()
                logger.info(f"Embedding model '{model_identifier}' initialized successfully.")
                self._is_initialized = True
            except ImportError as e:
                logger.error(f"Failed to initialize embedding model '{model_identifier}': Missing required library. "
                             f"Install dependencies for '{model_type_key}' (e.g., 'pip install llmcore[{model_type_key}]'). Error: {e}")
                self._embedding_model = None
                self._is_initialized = True # Mark as initialized (failed) to prevent retries
                raise EmbeddingError(model_name=model_identifier, message=f"Initialization failed due to missing dependency: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model '{model_identifier}': {e}", exc_info=True)
                self._embedding_model = None
                self._is_initialized = True # Mark as initialized (failed)
                # Re-raise as EmbeddingError
                raise EmbeddingError(model_name=model_identifier, message=f"Initialization failed: {e}")


    def get_embedding_model(self) -> BaseEmbeddingModel:
        """
        Returns the initialized embedding model instance.

        Raises:
            EmbeddingError: If the embedding model is not configured, not initialized,
                            or failed to initialize. Call `initialize_embedding_model` first.
        """
        if not self._is_initialized:
             # This indicates initialize_embedding_model was never awaited
             raise EmbeddingError(message="EmbeddingManager is not initialized. Call and await initialize_embedding_model() first.")
        if self._embedding_model is None:
            # This indicates initialization was attempted but failed, or no model was configured
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
