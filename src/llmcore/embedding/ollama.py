# src/llmcore/embedding/ollama.py
"""
Ollama Embedding model implementation for LLMCore.

Uses the official ollama library to generate embeddings via a local Ollama instance.

Tested against ollama-python SDK v0.6.1.

Supports:
- Modern ``/api/embed`` endpoint (replaces deprecated ``/api/embeddings``)
- Native batch embedding (single HTTP request for multiple texts)
- Output dimensionality control via ``dimensions`` parameter
- Configurable truncation behaviour
"""

from __future__ import annotations

import logging
from typing import Any

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ResponseError

    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None  # type: ignore
    ResponseError = Exception  # type: ignore


from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Default Ollama embedding model if not specified.
# User needs to ensure the model is pulled in Ollama.
DEFAULT_OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"


class OllamaEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using a local Ollama instance via the official library.

    Requires the ``ollama`` library to be installed.
    Configuration for the host and model is read from ``[embedding.ollama]``.

    Uses the modern ``/api/embed`` endpoint which supports:
    - Batch input (``input: Union[str, Sequence[str]]``)
    - Dimensionality control (``dimensions: Optional[int]``)
    - Truncation control (``truncate: Optional[bool]``)

    The deprecated ``/api/embeddings`` endpoint (single-prompt, no batching)
    is **not** used.
    """

    _client: AsyncClient | None = None
    _model_name: str
    _host: str | None = None
    _timeout: float | None = None
    _dimensions: int | None = None
    _truncate: bool | None = None

    def __init__(self, config: dict[str, Any]):
        """
        Initializes the OllamaEmbedding model.

        Args:
            config: Configuration dictionary.  Expected keys from ``[embedding.ollama]``:
                    'host' (optional): Hostname/IP of the Ollama server
                        (default: library default = http://127.0.0.1:11434).
                    'default_model' (optional): Ollama embedding model name
                        (e.g., "mxbai-embed-large", "nomic-embed-text").
                    'timeout' (optional): Request timeout in seconds.
                    'dimensions' (optional): Output embedding dimensionality.
                        Truncates the embedding vector to this size (Matryoshka
                        embeddings).  Only effective with models that support it.
                    'truncate' (optional): Whether to truncate input text to the
                        model's maximum token length.
        """
        if not ollama_available:
            raise ImportError(
                "Ollama library not found. "
                "Please install `ollama` (e.g., `pip install llmcore[ollama]`)."
            )

        self._host = config.get("host")
        self._model_name = config.get("default_model", DEFAULT_OLLAMA_EMBEDDING_MODEL)
        timeout_val = config.get("timeout")
        self._timeout = float(timeout_val) if timeout_val is not None else None

        dimensions_val = config.get("dimensions")
        self._dimensions = int(dimensions_val) if dimensions_val is not None else None

        truncate_val = config.get("truncate")
        self._truncate = bool(truncate_val) if truncate_val is not None else None

        logger.info(
            f"OllamaEmbedding configured with model '{self._model_name}'. "
            f"Host: {self._host or 'default'}. "
            f"Dimensions: {self._dimensions or 'model default'}."
        )
        # Client initialization is deferred to ``initialize()``

    async def initialize(self) -> None:
        """
        Initializes the AsyncOllama client.
        """
        if self._client:
            logger.debug("OllamaEmbedding client already initialized.")
            return

        logger.debug("Initializing AsyncOllama client for embeddings...")
        try:
            client_args: dict[str, Any] = {}
            if self._host:
                client_args["host"] = self._host
            if self._timeout is not None:
                client_args["timeout"] = self._timeout

            self._client = AsyncClient(**client_args)
            logger.info("AsyncOllama client for embeddings initialized successfully.")
        except Exception as e:
            logger.error(
                f"Failed to initialize AsyncOllama client for embeddings: {e}", exc_info=True
            )
            self._client = None
            raise ConfigError(f"Ollama client initialization for embeddings failed: {e}")

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for a single text string using the modern
        ``/api/embed`` endpoint.

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails or the client
                is not initialized.
        """
        if not self._client:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Ollama client not initialized. Call initialize() first.",
            )
        if not text:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Input text cannot be empty for Ollama embeddings.",
            )

        logger.debug(
            f"Generating Ollama embedding for single text "
            f"(length: {len(text)}, model: {self._model_name})..."
        )
        try:
            # Use the modern embed() endpoint with single string input.
            # It returns EmbedResponse with .embeddings: Sequence[Sequence[float]]
            embed_kwargs: dict[str, Any] = {
                "model": self._model_name,
                "input": text,
            }
            if self._truncate is not None:
                embed_kwargs["truncate"] = self._truncate
            if self._dimensions is not None:
                embed_kwargs["dimensions"] = self._dimensions

            response = await self._client.embed(**embed_kwargs)

            # EmbedResponse.embeddings is Sequence[Sequence[float]] — one
            # embedding per input.  For a single string input we want [0].
            embeddings = response.embeddings
            if embeddings and len(embeddings) > 0:
                embedding_vector = list(embeddings[0])
                logger.debug(
                    f"Successfully generated Ollama embedding, dimension: {len(embedding_vector)}."
                )
                return embedding_vector
            else:
                logger.error(
                    f"Ollama embed API returned no embeddings for model "
                    f"'{self._model_name}'. Response: {response}"
                )
                raise EmbeddingError(
                    model_name=self._model_name,
                    message="API returned no embedding data.",
                )
        except ResponseError as e:
            error_detail = e.error if hasattr(e, "error") else str(e)
            logger.error(
                f"Ollama API error during embedding generation "
                f"(model: {self._model_name}): {e.status_code} - {error_detail}",
                exc_info=True,
            )
            if "model not found" in str(error_detail).lower():
                raise EmbeddingError(
                    model_name=self._model_name,
                    message=(
                        f"Model '{self._model_name}' not found locally. "
                        f"Pull it with 'ollama pull {self._model_name}'."
                    ),
                )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Ollama API Error ({e.status_code}): {error_detail}",
            )
        except TimeoutError:
            logger.error(f"Request to Ollama embed API timed out (model: {self._model_name}).")
            raise EmbeddingError(model_name=self._model_name, message="Request timed out.")
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating Ollama embedding (model: {self._model_name}): {e}",
                exc_info=True,
            )
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected error: {e}")

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate vector embeddings for a batch of text strings using the
        modern ``/api/embed`` endpoint's native batch support.

        A single HTTP request is made with all texts, which is significantly
        more efficient than calling ``generate_embedding()`` N times.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the embedding generation fails or the client
                is not initialized.
        """
        if not self._client:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Ollama client not initialized. Call initialize() first.",
            )
        if not texts:
            return []

        # Filter out empty strings — they would cause API errors
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            raise EmbeddingError(
                model_name=self._model_name,
                message="All input texts are empty.",
            )

        logger.debug(
            f"Generating Ollama embeddings for batch of {len(non_empty_texts)} texts "
            f"(model: {self._model_name})..."
        )
        try:
            embed_kwargs: dict[str, Any] = {
                "model": self._model_name,
                "input": non_empty_texts,
            }
            if self._truncate is not None:
                embed_kwargs["truncate"] = self._truncate
            if self._dimensions is not None:
                embed_kwargs["dimensions"] = self._dimensions

            response = await self._client.embed(**embed_kwargs)

            embeddings = response.embeddings
            if not embeddings or len(embeddings) != len(non_empty_texts):
                raise EmbeddingError(
                    model_name=self._model_name,
                    message=(
                        f"Expected {len(non_empty_texts)} embeddings, "
                        f"got {len(embeddings) if embeddings else 0}."
                    ),
                )

            # If we filtered out empty strings, reconstruct the full result
            # list with empty vectors at the original positions.
            if len(non_empty_texts) == len(texts):
                result = [list(emb) for emb in embeddings]
            else:
                # Some texts were empty — fill with zero-length vectors
                dim = len(embeddings[0]) if embeddings[0] else 0
                result: list[list[float]] = [[] for _ in texts]
                for idx, emb_idx in enumerate(non_empty_indices):
                    result[emb_idx] = list(embeddings[idx])
                # Fill empties with zero vectors of matching dimension
                for i in range(len(texts)):
                    if i not in non_empty_indices:
                        result[i] = [0.0] * dim

            logger.debug(
                f"Successfully generated batch of {len(non_empty_texts)} Ollama embeddings."
            )
            return result

        except ResponseError as e:
            error_detail = e.error if hasattr(e, "error") else str(e)
            logger.error(
                f"Ollama API error during batch embedding "
                f"(model: {self._model_name}): {e.status_code} - {error_detail}",
                exc_info=True,
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Ollama API Error ({e.status_code}): {error_detail}",
            )
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating batch Ollama embeddings "
                f"(model: {self._model_name}): {e}",
                exc_info=True,
            )
            raise EmbeddingError(
                model_name=self._model_name, message=f"Unexpected batch error: {e}"
            )

    async def close(self) -> None:
        """
        Closes the Ollama client.

        The ollama SDK's ``AsyncClient`` exposes an async ``close()``
        method (inherited from ``BaseClient``) which internally calls
        ``self._client.aclose()`` on the httpx ``AsyncClient``.
        """
        if self._client:
            logger.debug("Closing OllamaEmbedding client...")
            try:
                await self._client.close()
                logger.info("OllamaEmbedding client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OllamaEmbedding client: {e}", exc_info=True)
            finally:
                self._client = None
