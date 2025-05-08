# src/llmcore/embedding/google.py
"""
Google AI (Gemini) Embedding model implementation for LLMCore.

Uses the google-generativeai Python SDK to generate embeddings via their API.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

# Import Google Generative AI library
try:
    import google.generativeai as genai
    # Import specific exceptions if available
    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        google_exceptions = None # type: ignore
    google_ai_available = True
except ImportError:
    google_ai_available = False
    genai = None # type: ignore
    google_exceptions = None # type: ignore


from ..exceptions import EmbeddingError, ConfigError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_GOOGLE_EMBEDDING_MODEL = "models/embedding-001" # A common model, check for latest

class GoogleAIEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using the Google AI (Gemini) API.

    Requires the `google-generativeai` library to be installed.
    Configuration for API key and model is read from `[embedding.google]`.
    """
    _model_instance: Optional[Any] = None # Will be an instance of genai.GenerativeModel
    _model_name: str
    _api_key: Optional[str] = None
    # Google AI SDK typically doesn't use base_url or timeout in the same way as OpenAI's client
    # It's configured globally or through client options if available.

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GoogleAIEmbedding model.

        Args:
            config: Configuration dictionary. Expected keys from `[embedding.google]`:
                    'api_key' (optional): Google AI API key for embeddings.
                                          Defaults to env var GOOGLE_API_KEY.
                    'default_model' (optional): Google AI embedding model name
                                                (e.g., "models/embedding-001").
        """
        if not google_ai_available:
            raise ImportError(
                "Google Generative AI library not found. "
                "Please install `google-generativeai` (e.g., `pip install llmcore[gemini]`)."
            )

        embedding_specific_config = config

        self._api_key = embedding_specific_config.get('api_key')
        # If not in specific config, genai.configure() will look for GOOGLE_API_KEY env var.

        self._model_name = embedding_specific_config.get("default_model", DEFAULT_GOOGLE_EMBEDDING_MODEL)

        logger.info(f"GoogleAIEmbedding configured with model '{self._model_name}'. "
                    f"API key source: {'config (if provided)' if self._api_key else 'environment/SDK default'}.")
        # Client/model initialization is deferred to `initialize`

    async def initialize(self) -> None:
        """
        Configures the genai library and prepares the model instance.
        """
        if self._model_instance:
            logger.debug("GoogleAIEmbedding model instance already initialized.")
            return

        logger.debug(f"Initializing Google AI embedding model: {self._model_name}...")
        try:
            # Configure the API key globally for the genai library.
            # This is the standard way for this SDK.
            # If self._api_key is None, it expects GOOGLE_API_KEY env var.
            genai.configure(api_key=self._api_key)

            # For embeddings, we use genai.embed_content, which doesn't require
            # a GenerativeModel instance in the same way chat does.
            # However, to check model validity or get info, a model instance might be useful.
            # For now, we'll directly use genai.embed_content.
            # If we needed a model instance: self._model_instance = genai.GenerativeModel(self._model_name)
            # Let's test if the configured model is valid by trying to get model info.
            # This is an optional step but good for early failure detection.
            try:
                await asyncio.to_thread(genai.get_model, self._model_name) # Use to_thread for sync call
                logger.info(f"Google AI embedding model '{self._model_name}' validated.")
            except Exception as model_val_err:
                 # Log as warning, as some embedding-only models might not be listable via get_model
                 logger.warning(f"Could not explicitly validate Google AI model '{self._model_name}' via get_model: {model_val_err}. "
                                f"Proceeding, but embedding calls might fail if model is invalid.")


            logger.info(f"GoogleAIEmbedding for model '{self._model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GoogleAIEmbedding for model '{self._model_name}': {e}", exc_info=True)
            self._model_instance = None # Ensure it's None on failure
            raise ConfigError(f"Google AI (Gemini) embedding initialization failed: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text string using Google AI API.

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        if not genai: # Should be caught by __init__
            raise EmbeddingError(model_name=self._model_name, message="Google Generative AI library not available.")

        if not text:
            logger.warning("generate_embedding called with empty text for Google AI.")
            # Gemini API errors on empty content for embeddings.
            # Return a zero vector of a typical dimension or raise error.
            # For now, let's raise an error as the API would.
            raise EmbeddingError(model_name=self._model_name, message="Input text cannot be empty for Google AI embeddings.")

        logger.debug(f"Generating Google AI embedding for single text (length: {len(text)}, model: {self._model_name})...")
        try:
            # Use asyncio.to_thread for the synchronous genai.embed_content call
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT" # Or "SEMANTIC_SIMILARITY", "RETRIEVAL_QUERY"
                                               # "RETRIEVAL_DOCUMENT" is common for storing docs.
            )
            embedding_data = result.get('embedding')
            if embedding_data:
                logger.debug(f"Successfully generated Google AI embedding, dimension: {len(embedding_data)}.")
                return embedding_data
            else:
                logger.error(f"Google AI embedding API returned no embedding data for model '{self._model_name}'. Response: {result}")
                raise EmbeddingError(model_name=self._model_name, message="API returned no embedding data.")
        except (google_exceptions.GoogleAPIError, AttributeError, TypeError, ValueError) if google_exceptions else Exception as e: # Catch specific Google errors if possible
            logger.error(f"Google AI API error during embedding generation (model: {self._model_name}): {e}", exc_info=True)
            # Try to get more specific error details if it's a GoogleAPIError
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            message = str(e)
            if hasattr(e, 'message'): message = e.message # type: ignore

            raise EmbeddingError(model_name=self._model_name, message=f"Google AI API Error (Code: {status_code}): {message}")
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error generating Google AI embedding (model: {self._model_name}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected error: {e}")


    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text strings using Google AI API.
        The `genai.embed_content` function can take a list of strings for batching.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        if not genai:
            raise EmbeddingError(model_name=self._model_name, message="Google Generative AI library not available.")
        if not texts:
            return []

        # Check for empty strings in the batch, as API might error
        processed_texts = []
        for i, text_content in enumerate(texts):
            if not text_content:
                logger.warning(f"Empty string found at index {i} in batch texts for Google AI. This may cause an API error.")
                # Option 1: Raise error
                # raise EmbeddingError(model_name=self._model_name, message=f"Empty string at index {i} not allowed.")
                # Option 2: Replace with a placeholder (e.g., a space) if API allows
                # processed_texts.append(" ")
                # Option 3: Skip (will cause length mismatch if not handled)
                # For now, let's pass them as is and let the API handle it, or rely on previous error for single empty string.
                # If the API strictly disallows empty strings in batches, this will fail.
                # The `embed_content` API expects `ContentLikable` which can be List[str].
            processed_texts.append(text_content)


        logger.debug(f"Generating Google AI embeddings for batch of {len(processed_texts)} texts (model: {self._model_name})...")
        try:
            # Use asyncio.to_thread for the synchronous genai.embed_content call
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._model_name,
                content=processed_texts, # Pass the list of strings
                task_type="RETRIEVAL_DOCUMENT" # Assuming these are documents to be stored
            )
            # The result for batch input is a dict with a single 'embedding' key,
            # which contains a list of embeddings.
            embeddings_data = result.get('embedding')
            if embeddings_data and isinstance(embeddings_data, list) and len(embeddings_data) == len(texts):
                logger.debug(f"Successfully generated batch of {len(embeddings_data)} Google AI embeddings.")
                return embeddings_data # type: ignore # Expect List[List[float]]
            else:
                logger.error(f"Google AI embedding API returned mismatched data count or no data for model '{self._model_name}'. "
                             f"Expected {len(texts)}, got {len(embeddings_data) if embeddings_data else 0}. Response: {result}")
                raise EmbeddingError(model_name=self._model_name, message="API returned mismatched or no embedding data for batch.")

        except (google_exceptions.GoogleAPIError, AttributeError, TypeError, ValueError) if google_exceptions else Exception as e:
            logger.error(f"Google AI API error during batch embedding generation (model: {self._model_name}): {e}", exc_info=True)
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            message = str(e)
            if hasattr(e, 'message'): message = e.message # type: ignore
            raise EmbeddingError(model_name=self._model_name, message=f"Google AI API Error (Code: {status_code}): {message}")
        except Exception as e:
            logger.error(f"Unexpected error generating batch Google AI embeddings (model: {self._model_name}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected batch error: {e}")

    async def close(self) -> None:
        """
        Clean up resources for GoogleAIEmbedding.
        The `google-generativeai` library typically does not require explicit client closing
        as connections are managed by underlying HTTP libraries or gRPC.
        """
        logger.debug(f"GoogleAIEmbedding for model '{self._model_name}' closed (no specific client cleanup typically needed).")
        self._model_instance = None # Clear any cached model instance
        # No explicit client.close() for genai library in this context.
