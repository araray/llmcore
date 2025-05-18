# src/llmcore/embedding/google.py
"""
Google AI (Gemini) Embedding model implementation for LLMCore.

Uses the google-genai Python SDK (v0.8.0+) to generate embeddings via their API.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

# --- Use the new google-genai library ---
try:
    import google.genai as genai
    from google.genai import errors as genai_errors
    from google.genai import \
        types as genai_types  # Keep types import for other potential uses
    google_genai_available = True
except ImportError:
    google_genai_available = False
    genai = None # type: ignore
    genai_types = None # type: ignore
    genai_errors = None # type: ignore
# --- End new library import ---


from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Default model from the new SDK docs
DEFAULT_GOOGLE_EMBEDDING_MODEL = "text-embedding-004"
# Define known valid task types as strings based on SDK documentation/usage
VALID_TASK_TYPES = [
    "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY",
    "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION"
    # Add others as documented by Google
]

class GoogleAIEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using the Google AI (Gemini) API via google-genai.

    Requires the `google-genai` library to be installed.
    Configuration for API key and model is read from `[embedding.google]`.
    """
    _client: Optional[genai.Client] = None # Use the new client type
    _model_name: str
    _api_key: Optional[str] = None
    # --- Updated Type Hint for _task_type ---
    _task_type: str = "RETRIEVAL_DOCUMENT" # Default task type, now using str
    # --- End Update ---

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GoogleAIEmbedding model using google-genai.

        Args:
            config: Configuration dictionary. Expected keys from `[embedding.google]`:
                    'api_key' (optional): Google AI API key for embeddings.
                                          Defaults to env var GOOGLE_API_KEY.
                    'default_model' (optional): Google AI embedding model name
                                                (e.g., "text-embedding-004").
                    'task_type' (optional): Embedding task type string ('RETRIEVAL_DOCUMENT',
                                            'RETRIEVAL_QUERY', 'SEMANTIC_SIMILARITY', etc.).
                                            Defaults to 'RETRIEVAL_DOCUMENT'.
        """
        if not google_genai_available:
            raise ImportError(
                "Google Gen AI library (`google-genai`) not found. "
                "Please install `google-genai` (e.g., `pip install llmcore[gemini]`)."
            )

        embedding_specific_config = config

        self._api_key = embedding_specific_config.get('api_key')
        # Ensure model name doesn't have the 'models/' prefix from config initially
        raw_model_name = embedding_specific_config.get("default_model", DEFAULT_GOOGLE_EMBEDDING_MODEL)
        self._model_name = raw_model_name.replace("models/", "") # Store without prefix

        # Parse task type string
        task_type_str = embedding_specific_config.get("task_type", "RETRIEVAL_DOCUMENT").upper()
        # --- Removed Validation Check ---
        # Validate against known valid strings if desired, but SDK likely handles it
        if task_type_str not in VALID_TASK_TYPES:
             logger.warning(f"Provided task_type '{task_type_str}' is not in the known valid list {VALID_TASK_TYPES}. "
                            f"Using it anyway, but it might cause API errors if invalid.")
        self._task_type = task_type_str
        # --- End Removal ---

        logger.info(f"GoogleAIEmbedding configured with model '{self._model_name}' and task_type '{self._task_type}'. " # Log without prefix
                    f"API key source: {'config (if provided)' if self._api_key else 'environment/SDK default'}.")
        # Client initialization deferred to `initialize`

    async def initialize(self) -> None:
        """
        Initializes the google-genai client.
        """
        if self._client:
            logger.debug("GoogleAIEmbedding client already initialized.")
            return

        logger.debug(f"Initializing Google Gen AI client for embeddings (model: {self._model_name})...")
        try:
            client_options = {}
            if self._api_key:
                client_options['api_key'] = self._api_key

            self._client = genai.Client(**client_options)
            # Optional: Test connection or check model validity if possible
            # models = self._client.models.list() # Example check
            logger.info(f"GoogleAIEmbedding client initialized successfully for model '{self._model_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client for embeddings: {e}", exc_info=True)
            self._client = None # Ensure it's None on failure
            raise ConfigError(f"Google Gen AI client initialization for embeddings failed: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text string using Google AI API (google-genai).

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        if not self._client or not genai_errors:
            raise EmbeddingError(model_name=self._model_name, message="Google Gen AI client/errors not available.")
        if not text:
            logger.warning("generate_embedding called with empty text for Google AI.")
            raise EmbeddingError(model_name=self._model_name, message="Input text cannot be empty for Google AI embeddings.")

        # --- Add 'models/' prefix before API call ---
        model_with_prefix = f"models/{self._model_name}"
        # --- End Add Prefix ---
        logger.debug(f"Generating Google AI embedding for single text (length: {len(text)}, model: {model_with_prefix}, task: {self._task_type})...")
        try:
            # Use the async client (client.aio)
            result = await self._client.aio.models.embed_content(
                model=model_with_prefix, # Use model name with prefix
                content=text,
                task_type=self._task_type # Pass task type string directly
                # output_dimensionality can be added if needed
            )
            embedding_data = result.get('embedding')
            if embedding_data:
                logger.debug(f"Successfully generated Google AI embedding, dimension: {len(embedding_data)}.")
                return embedding_data
            else:
                logger.error(f"Google AI embedding API returned no embedding data for model '{model_with_prefix}'. Response: {result}")
                raise EmbeddingError(model_name=self._model_name, message="API returned no embedding data.")
        except genai_errors.GoogleAPIError as e:
            logger.error(f"Google AI API error during embedding generation (model: {model_with_prefix}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Google AI API Error: {e}")
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error generating Google AI embedding (model: {model_with_prefix}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected error: {e}")


    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text strings using Google AI API (google-genai).

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        if not self._client or not genai_errors:
            raise EmbeddingError(model_name=self._model_name, message="Google Gen AI client/errors not available.")
        if not texts:
            return []

        # Check for empty strings in the batch
        processed_texts = []
        indices_with_content = []
        for i, text_content in enumerate(texts):
            if not text_content:
                logger.warning(f"Empty string found at index {i} in batch texts for Google AI. Skipping.")
            else:
                processed_texts.append(text_content)
                indices_with_content.append(i)

        if not processed_texts:
            logger.warning("generate_embeddings called with batch containing only empty strings.")
            return [] # Return empty list if all inputs were empty

        # --- Add 'models/' prefix before API call ---
        model_with_prefix = f"models/{self._model_name}"
        # --- End Add Prefix ---
        logger.debug(f"Generating Google AI embeddings for batch of {len(processed_texts)} non-empty texts (model: {model_with_prefix}, task: {self._task_type})...")
        try:
            # Use the async client (client.aio)
            result = await self._client.aio.models.embed_content(
                model=model_with_prefix, # Use model name with prefix
                content=processed_texts, # Pass the list of non-empty strings
                task_type=self._task_type # Pass task type string directly
            )

            embeddings_data = result.get('embedding')
            if embeddings_data and isinstance(embeddings_data, list) and len(embeddings_data) == len(processed_texts):
                logger.debug(f"Successfully generated batch of {len(embeddings_data)} Google AI embeddings.")
                # Need to reconstruct the full list including placeholders for skipped empty strings
                final_embeddings: List[Optional[List[float]]] = [None] * len(texts)
                for i, original_index in enumerate(indices_with_content):
                    final_embeddings[original_index] = embeddings_data[i]

                # Fill None placeholders with zero vectors (or handle differently if needed)
                output_embeddings: List[List[float]] = []
                # Get dimension from the first valid embedding
                dimension = len(embeddings_data[0]) if embeddings_data else 0
                for emb in final_embeddings:
                    if emb is not None:
                        output_embeddings.append(emb)
                    else:
                        output_embeddings.append([0.0] * dimension) # Use zero vector for empty inputs

                return output_embeddings
            else:
                logger.error(f"Google AI embedding API returned mismatched data count or no data for model '{model_with_prefix}'. "
                             f"Expected {len(processed_texts)}, got {len(embeddings_data) if embeddings_data else 0}. Response: {result}")
                raise EmbeddingError(model_name=self._model_name, message="API returned mismatched or no embedding data for batch.")

        except genai_errors.GoogleAPIError as e:
            logger.error(f"Google AI API error during batch embedding generation (model: {model_with_prefix}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Google AI API Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating batch Google AI embeddings (model: {model_with_prefix}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected batch error: {e}")

    async def close(self) -> None:
        """
        Clean up resources for GoogleAIEmbedding.
        The google-genai client typically does not require explicit closing.
        """
        logger.debug(f"GoogleAIEmbedding for model '{self._model_name}' closed (no specific client cleanup typically needed).")
        self._client = None # Dereference the client
