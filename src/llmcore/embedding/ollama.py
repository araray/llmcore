# src/llmcore/embedding/ollama.py
"""
Ollama Embedding model implementation for LLMCore.

Uses the official ollama library to generate embeddings via a local Ollama instance.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ResponseError
    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None # type: ignore
    ResponseError = Exception # type: ignore


from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Default Ollama embedding model if not specified (e.g., mxbai-embed-large is popular)
# User needs to ensure the model is pulled in Ollama.
DEFAULT_OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"

class OllamaEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using a local Ollama instance via the official library.

    Requires the `ollama` library to be installed.
    Configuration for the host and model is read from `[embedding.ollama]`.
    """
    _client: Optional[AsyncClient] = None
    _model_name: str
    _host: Optional[str] = None
    _timeout: Optional[float] = None

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OllamaEmbedding model.

        Args:
            config: Configuration dictionary. Expected keys from `[embedding.ollama]`:
                    'host' (optional): Hostname/IP of the Ollama server (default: library default).
                    'default_model' (optional): Ollama embedding model name (e.g., "mxbai-embed-large").
                    'timeout' (optional): Request timeout in seconds.
        """
        if not ollama_available:
            raise ImportError(
                "Ollama library not found. "
                "Please install `ollama` (e.g., `pip install llmcore[ollama]`)."
            )

        embedding_specific_config = config

        self._host = embedding_specific_config.get("host") # Can be None
        self._model_name = embedding_specific_config.get("default_model", DEFAULT_OLLAMA_EMBEDDING_MODEL)
        timeout_val = embedding_specific_config.get("timeout")
        self._timeout = float(timeout_val) if timeout_val is not None else None


        logger.info(f"OllamaEmbedding configured with model '{self._model_name}'. "
                    f"Host: {self._host or 'default'}.")
        # Client initialization is deferred to `initialize`

    async def initialize(self) -> None:
        """
        Initializes the AsyncOllama client.
        """
        if self._client:
            logger.debug("OllamaEmbedding client already initialized.")
            return

        logger.debug("Initializing AsyncOllama client for embeddings...")
        try:
            client_args = {}
            if self._host:
                client_args['host'] = self._host
            if self._timeout:
                client_args['timeout'] = self._timeout

            self._client = AsyncClient(**client_args)
            # Optionally test connection or model availability
            # await self._client.list() # Example check, might be slow
            logger.info("AsyncOllama client for embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOllama client for embeddings: {e}", exc_info=True)
            self._client = None # Ensure client is None on failure
            raise ConfigError(f"Ollama client initialization for embeddings failed: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text string using Ollama API.

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails or the client is not initialized.
        """
        if not self._client:
            raise EmbeddingError(model_name=self._model_name, message="Ollama client not initialized. Call initialize() first.")
        if not text:
            logger.warning("generate_embedding called with empty text for Ollama.")
            # Ollama API might error on empty string. Let's raise an error proactively.
            raise EmbeddingError(model_name=self._model_name, message="Input text cannot be empty for Ollama embeddings.")

        logger.debug(f"Generating Ollama embedding for single text (length: {len(text)}, model: {self._model_name})...")
        try:
            # The ollama library's embeddings method handles single prompts
            response = await self._client.embeddings(
                model=self._model_name,
                prompt=text
                # options=... # Add options if needed (e.g., temperature for some models?)
            )
            embedding_data = response.get('embedding')
            if embedding_data and isinstance(embedding_data, list):
                logger.debug(f"Successfully generated Ollama embedding, dimension: {len(embedding_data)}.")
                return embedding_data # type: ignore # Expect List[float]
            else:
                logger.error(f"Ollama embedding API returned no embedding data for model '{self._model_name}'. Response: {response}")
                raise EmbeddingError(model_name=self._model_name, message="API returned no embedding data.")
        except ResponseError as e:
            error_detail = e.error if hasattr(e, 'error') else str(e)
            logger.error(f"Ollama API error during embedding generation (model: {self._model_name}): {e.status_code} - {error_detail}", exc_info=True)
            if "model not found" in str(error_detail).lower():
                 raise EmbeddingError(model_name=self._model_name, message=f"Model '{self._model_name}' not found locally. Pull it with 'ollama pull {self._model_name}'.")
            raise EmbeddingError(model_name=self._model_name, message=f"Ollama API Error ({e.status_code}): {error_detail}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Ollama embeddings API timed out (model: {self._model_name}).")
            raise EmbeddingError(model_name=self._model_name, message="Request timed out.")
        except Exception as e:
            logger.error(f"Unexpected error generating Ollama embedding (model: {self._model_name}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected error: {e}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a batch of text strings using Ollama API.

        Note: The current `ollama` library's `embeddings` method only takes a single prompt.
        We will call it sequentially for each text in the batch. This might be inefficient.
        Future versions of the library might support batching directly.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the embedding generation fails for any text or the client is not initialized.
        """
        if not self._client:
            raise EmbeddingError(model_name=self._model_name, message="Ollama client not initialized. Call initialize() first.")
        if not texts:
            return []

        logger.debug(f"Generating Ollama embeddings sequentially for batch of {len(texts)} texts (model: {self._model_name})...")
        embeddings_list: List[List[float]] = []
        try:
            # Call generate_embedding for each text sequentially
            # Use asyncio.gather for potential concurrency, though underlying calls might still be sequential
            # depending on the client and server capabilities.
            tasks = [self.generate_embedding(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results, raising error if any task failed
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating embedding for text at index {i} in batch: {result}")
                    # Re-raise the first encountered error
                    if isinstance(result, EmbeddingError): raise result
                    raise EmbeddingError(model_name=self._model_name, message=f"Batch embedding failed at index {i}: {result}")
                elif isinstance(result, list):
                    embeddings_list.append(result)
                else:
                    # Should not happen if generate_embedding returns correctly
                    logger.error(f"Unexpected result type for embedding at index {i}: {type(result)}")
                    raise EmbeddingError(model_name=self._model_name, message=f"Unexpected result type in batch at index {i}")

            if len(embeddings_list) != len(texts):
                 # This case might occur if an error wasn't caught properly by gather
                 logger.error(f"Mismatch in expected ({len(texts)}) and generated ({len(embeddings_list)}) embeddings count.")
                 raise EmbeddingError(model_name=self._model_name, message="Failed to generate embeddings for all texts in the batch.")

            logger.debug(f"Successfully generated batch of {len(embeddings_list)} Ollama embeddings sequentially.")
            return embeddings_list

        except EmbeddingError: # Re-raise specific EmbeddingErrors from gather
            raise
        except Exception as e: # Catch any other unexpected errors during gather or processing
            logger.error(f"Unexpected error generating batch Ollama embeddings (model: {self._model_name}): {e}", exc_info=True)
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected batch error: {e}")


    async def close(self) -> None:
        """
        Closes the Ollama client.
        The underlying httpx client used by the ollama library should be closed.
        """
        if self._client:
            logger.debug("Closing OllamaEmbedding client...")
            # The ollama library's AsyncClient might have an aclose method or manage httpx client internally.
            # Checking if 'aclose' exists is safer.
            if hasattr(self._client, 'aclose') and asyncio.iscoroutinefunction(self._client.aclose):
                try:
                    await self._client.aclose()
                    logger.info("OllamaEmbedding client closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing OllamaEmbedding client: {e}", exc_info=True)
            else:
                 logger.debug("Ollama AsyncClient does not have an explicit 'aclose' method. Closure handled by library/GC.")
            self._client = None
