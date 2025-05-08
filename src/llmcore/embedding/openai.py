# src/llmcore/embedding/openai.py
"""
OpenAI Embedding model implementation for LLMCore.

Uses the OpenAI Python SDK to generate embeddings via their API.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

# Import OpenAI library
try:
    import openai
    from openai import AsyncOpenAI, OpenAIError
    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None # type: ignore
    OpenAIError = Exception # type: ignore


from ..exceptions import EmbeddingError, ConfigError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

class OpenAIEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using the OpenAI API.

    Requires the `openai` library to be installed.
    Configuration for API key and model is read from `[embedding.openai]`
    or `[providers.openai]` as a fallback for the key.
    """
    _client: Optional[AsyncOpenAI] = None
    _model_name: str
    _api_key: Optional[str] = None
    _base_url: Optional[str] = None
    _timeout: float = 60.0 # Default timeout

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OpenAIEmbedding model.

        Args:
            config: Configuration dictionary. Expected keys from `[embedding.openai]`:
                    'api_key' (optional): OpenAI API key for embeddings.
                                          If not found, tries `providers.openai.api_key`.
                                          Defaults to env var OPENAI_API_KEY.
                    'base_url' (optional): Custom OpenAI API endpoint URL.
                    'default_model' (optional): OpenAI embedding model name.
                    'timeout' (optional): Request timeout in seconds.
        """
        if not openai_available:
            raise ImportError(
                "OpenAI library not found. "
                "Please install `openai` (e.g., `pip install llmcore[openai]`)."
            )

        # Config specific to embeddings, e.g., [embedding.openai]
        embedding_specific_config = config # The manager passes the specific [embedding.openai] section

        # API key: Try embedding-specific, then general OpenAI provider, then env var
        self._api_key = embedding_specific_config.get('api_key')
        if not self._api_key:
            # Fallback: Try to get API key from the main OpenAI provider config
            # This requires access to the full config, which is not directly passed here.
            # The EmbeddingManager should ideally resolve this or LLMCore.create should.
            # For now, assume if not in embedding_specific_config, it relies on env var.
            # A better approach would be for EmbeddingManager to pass a more complete config
            # or for LLMCore to inject the resolved API key if shared.
            logger.debug("API key not in [embedding.openai], will try main OpenAI provider or env var.")
            # This will be picked up by AsyncOpenAI if None.
            # self._api_key = main_config.get("providers.openai.api_key") or os.environ.get("OPENAI_API_KEY")
            pass # AsyncOpenAI will use OPENAI_API_KEY env var if self._api_key is None

        self._model_name = embedding_specific_config.get("default_model", DEFAULT_OPENAI_EMBEDDING_MODEL)
        self._base_url = embedding_specific_config.get("base_url") # Allow None for default
        self._timeout = float(embedding_specific_config.get("timeout", 60.0))

        logger.info(f"OpenAIEmbedding configured with model '{self._model_name}'. "
                    f"API key source: {'config' if self._api_key else 'environment/SDK default'}. "
                    f"Base URL: {self._base_url or 'default'}.")
        # Client initialization is deferred to `initialize`

    async def initialize(self) -> None:
        """
        Initializes the AsyncOpenAI client.
        """
        if self._client:
            logger.debug("OpenAIEmbedding client already initialized.")
            return

        logger.debug("Initializing AsyncOpenAI client for embeddings...")
        try:
            self._client = AsyncOpenAI(
                api_key=self._api_key, # Can be None, SDK will look for env var
                base_url=self._base_url,
                timeout=self._timeout
            )
            # Optionally, make a test call if there's a cheap way to verify credentials,
            # but usually, the first actual API call will reveal issues.
            logger.info("AsyncOpenAI client for embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client for embeddings: {e}", exc_info=True)
            self._client = None # Ensure client is None on failure
            raise ConfigError(f"OpenAI client initialization for embeddings failed: {e}")


    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text string using OpenAI API.

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails or the client is not initialized.
        """
        if not self._client:
            raise EmbeddingError(model_name=self._model_name, message="OpenAI client not initialized. Call initialize() first.")
        if not text:
            logger.warning("generate_embedding called with empty text for OpenAI.")
            # OpenAI API might error on empty string or return a specific embedding.
            # text_to_embed = " " # Replace empty string with a space, or handle as API expects
            # For consistency, let's try to get a dimension and return zeros, or let API handle it.
            # However, getting dimension requires a call. Best to let the API handle empty string.
            text_to_embed = text # Pass empty string as is.
        else:
            text_to_embed = text.replace("\n", " ") # OpenAI recommends replacing newlines

        logger.debug(f"Generating OpenAI embedding for single text (length: {len(text_to_embed)}, model: {self._model_name})...")
        try:
            response = await self._client.embeddings.create(
                model=self._model_name,
                input=[text_to_embed] # API expects a list of strings
            )
            if response.data and len(response.data) > 0:
                embedding_data = response.data[0].embedding
                logger.debug(f"Successfully generated OpenAI embedding, dimension: {len(embedding_data)}.")
                return embedding_data
            else:
                logger.error(f"OpenAI embedding API returned no data for model '{self._model_name}'. Response: {response}")
                raise EmbeddingError(model_name=self._model_name, message="API returned no embedding data.")
        except OpenAIError as e:
            logger.error(f"OpenAI API error during embedding generation (model: {self._model_name}): {e.status_code} - {e.message}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"OpenAI API Error ({e.status_code}): {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to OpenAI embeddings API timed out (model: {self._model_name}).")
            raise EmbeddingError(model_name=self._model_name, message="Request timed out.")
        except Exception as e:
            logger.error(f"Unexpected error generating OpenAI embedding (model: {self._model_name}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected error: {e}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text strings using OpenAI API.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the embedding generation fails or the client is not initialized.
        """
        if not self._client:
            raise EmbeddingError(model_name=self._model_name, message="OpenAI client not initialized. Call initialize() first.")
        if not texts:
            return []

        # OpenAI recommends replacing newlines for their embedding models
        processed_texts = [text.replace("\n", " ") if text else "" for text in texts]

        logger.debug(f"Generating OpenAI embeddings for batch of {len(processed_texts)} texts (model: {self._model_name})...")
        try:
            response = await self._client.embeddings.create(
                model=self._model_name,
                input=processed_texts
            )
            if response.data and len(response.data) == len(texts):
                embeddings_data = [item.embedding for item in response.data]
                logger.debug(f"Successfully generated batch of {len(embeddings_data)} OpenAI embeddings.")
                return embeddings_data
            else:
                logger.error(f"OpenAI embedding API returned mismatched data count or no data for model '{self._model_name}'. "
                             f"Expected {len(texts)}, got {len(response.data) if response.data else 0}. Response: {response}")
                raise EmbeddingError(model_name=self._model_name, message="API returned mismatched or no embedding data for batch.")
        except OpenAIError as e:
            logger.error(f"OpenAI API error during batch embedding generation (model: {self._model_name}): {e.status_code} - {e.message}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"OpenAI API Error ({e.status_code}): {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to OpenAI embeddings API timed out during batch (model: {self._model_name}).")
            raise EmbeddingError(model_name=self._model_name, message="Batch request timed out.")
        except Exception as e:
            logger.error(f"Unexpected error generating batch OpenAI embeddings (model: {self._model_name}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected batch error: {e}")

    async def close(self) -> None:
        """
        Closes the OpenAI client if it was initialized.
        The OpenAI SDK v1.x+ typically manages its own HTTP client lifecycle,
        but providing an explicit close is good practice if the underlying httpx client
        needs to be released.
        """
        if self._client:
            try:
                # For openai>=1.0.0, AsyncOpenAI uses an httpx.AsyncClient internally.
                # It's good practice to close it, though the SDK might also do it on garbage collection.
                await self._client.close()
                logger.info("OpenAIEmbedding client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OpenAIEmbedding client: {e}", exc_info=True)
            finally:
                self._client = None
