# src/llmcore/embedding/sentence_transformer.py
"""
Sentence Transformer embedding model implementation for LLMCore.

Uses the sentence-transformers library to generate embeddings locally.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

# Import sentence-transformers library
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    SentenceTransformer = None # type: ignore

from ..exceptions import EmbeddingError, ConfigError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using local Sentence Transformer models.

    Requires the `sentence-transformers` library to be installed.
    Models are loaded based on configuration.
    """
    _model: Optional[SentenceTransformer] = None
    _model_name_or_path: Optional[str] = None
    _device: Optional[str] = None # e.g., 'cpu', 'cuda', 'mps'

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the SentenceTransformerEmbedding model loader.

        Args:
            config: Configuration dictionary. Expected keys:
                    'model_name_or_path': The name (HuggingFace Hub) or local path
                                          of the Sentence Transformer model.
                    'device' (optional): The device to run the model on ('cpu', 'cuda', 'mps').
                                         Defaults to None (library default).
        """
        if not sentence_transformers_available:
            raise ImportError(
                "Sentence Transformers library not found. "
                "Please install `sentence-transformers` (e.g., `pip install llmcore[sentence_transformers]`)."
            )

        # Store config values, model loading happens in initialize
        self._model_name_or_path = config.get("model_name_or_path")
        self._device = config.get("device")

        if not self._model_name_or_path:
            raise ConfigError("SentenceTransformerEmbedding requires 'model_name_or_path' in its configuration.")

        logger.info(f"SentenceTransformerEmbedding configured with model '{self._model_name_or_path}' "
                    f"(Device: {self._device or 'default'}). Model will be loaded on initialize.")

    async def initialize(self) -> None:
        """
        Loads the Sentence Transformer model into memory.

        This is done asynchronously using `asyncio.to_thread` as model loading
        can be blocking.
        """
        if self._model:
            logger.debug(f"Sentence Transformer model '{self._model_name_or_path}' already initialized.")
            return

        if not self._model_name_or_path:
             # Should be caught in __init__, but double-check
             raise ConfigError("Cannot initialize SentenceTransformerEmbedding without 'model_name_or_path'.")

        logger.info(f"Initializing Sentence Transformer model: {self._model_name_or_path}...")
        try:
            # Use asyncio.to_thread to run the blocking model loading code
            self._model = await asyncio.to_thread(
                self._load_model_sync, self._model_name_or_path, self._device
            )
            logger.info(f"Sentence Transformer model '{self._model_name_or_path}' loaded successfully "
                        f"onto device '{self._model.device}'.") # type: ignore
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{self._model_name_or_path}': {e}", exc_info=True)
            self._model = None # Ensure model is None on failure
            raise EmbeddingError(
                model_name=self._model_name_or_path,
                message=f"Failed to load model: {e}"
            )

    def _load_model_sync(self, model_name_or_path: str, device: Optional[str]) -> SentenceTransformer:
        """Synchronous helper function to load the model."""
        # This function will run in a separate thread via asyncio.to_thread
        return SentenceTransformer(model_name_or_path, device=device)

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous helper function for encoding."""
        if not self._model:
            # This should ideally not happen if initialize was called successfully
            raise EmbeddingError(model_name=self._model_name_or_path, message="Model is not loaded.")
        try:
            # The encode method handles batching internally and is often CPU/GPU bound
            embeddings = self._model.encode(texts, convert_to_numpy=False) # Get list of lists
            # Ensure the output is List[List[float]]
            return [[float(val) for val in emb] for emb in embeddings] # type: ignore
        except Exception as e:
            logger.error(f"Error during Sentence Transformer encoding: {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name_or_path, message=f"Encoding failed: {e}")


    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text string asynchronously.

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails or the model is not loaded.
        """
        if not self._model:
            raise EmbeddingError(model_name=self._model_name_or_path, message="Model not initialized. Call initialize() first.")
        if not text:
            # Return a zero vector or handle as appropriate for empty input
            # Getting dimension might require model loading, do it lazily or store it
            # For now, raise error or return empty list? Let's raise.
            # Alternatively, return empty list or zeros of correct dimension if known.
             logger.warning("generate_embedding called with empty text.")
             # Get dimension if possible (might be slow first time)
             try:
                 dim = self._model.get_sentence_embedding_dimension()
                 return [0.0] * dim if dim else []
             except Exception:
                 return [] # Fallback

        logger.debug(f"Generating embedding for single text (length: {len(text)})...")
        try:
            # Run the synchronous encode method in a thread
            embeddings = await asyncio.to_thread(self._encode_sync, [text])
            return embeddings[0]
        except Exception as e:
             # Catch potential errors from _encode_sync or asyncio.to_thread
             logger.error(f"Error generating single embedding: {e}", exc_info=True)
             # Ensure it's wrapped in EmbeddingError
             if isinstance(e, EmbeddingError): raise
             raise EmbeddingError(model_name=self._model_name_or_path, message=f"Embedding generation failed: {e}")


    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text strings asynchronously.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the embedding generation fails or the model is not loaded.
        """
        if not self._model:
            raise EmbeddingError(model_name=self._model_name_or_path, message="Model not initialized. Call initialize() first.")
        if not texts:
            return [] # Return empty list if input is empty

        logger.debug(f"Generating embeddings for batch of {len(texts)} texts...")
        try:
            # Run the synchronous encode method (which handles batching) in a thread
            embeddings = await asyncio.to_thread(self._encode_sync, texts)
            return embeddings
        except Exception as e:
             # Catch potential errors from _encode_sync or asyncio.to_thread
             logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
             # Ensure it's wrapped in EmbeddingError
             if isinstance(e, EmbeddingError): raise
             raise EmbeddingError(model_name=self._model_name_or_path, message=f"Batch embedding generation failed: {e}")

    # Optional: Implement close if there are specific resources to release
    # async def close(self) -> None:
    #     """Clean up resources (e.g., release GPU memory if applicable)."""
    #     logger.debug(f"Closing SentenceTransformerEmbedding (model: {self._model_name_or_path}).")
    #     # Models loaded via sentence-transformers don't typically require explicit closing,
    #     # but specific backends (like ONNX Runtime sessions) might.
    #     # For standard PyTorch models, Python's garbage collection usually handles it.
    #     self._model = None # Allow garbage collection
