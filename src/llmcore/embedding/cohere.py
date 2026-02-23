# src/llmcore/embedding/cohere.py
"""
Cohere Embedding model implementation for LLMCore.

Uses the Cohere Python SDK to generate embeddings via their API.
Supports Cohere's embedding models including embed-english-v3.0 and
embed-multilingual-v3.0 with configurable input types for optimal
retrieval vs. classification performance.

Cohere embeddings support an ``input_type`` parameter:
- ``"search_document"`` — for indexing documents into a search system
- ``"search_query"`` — for search queries
- ``"classification"`` — for text classification
- ``"clustering"`` — for clustering tasks

References:
    - https://docs.cohere.com/reference/embed
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §6.2 (Embedding System)
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import cohere
    from cohere import AsyncClientV2 as AsyncCohereClient

    cohere_available = True
except ImportError:
    cohere_available = False
    AsyncCohereClient = None  # type: ignore[assignment,misc]

from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_COHERE_EMBEDDING_MODEL = "embed-english-v3.0"
DEFAULT_INPUT_TYPE = "search_document"


class CohereEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using the Cohere API.

    Requires the ``cohere`` library (``pip install cohere``).

    Configuration keys (from ``[embedding.cohere]``):

    - ``api_key``: Cohere API key (falls back to ``CO_API_KEY`` env var).
    - ``default_model``: Model name (default: ``embed-english-v3.0``).
    - ``input_type``: One of ``search_document``, ``search_query``,
      ``classification``, ``clustering`` (default: ``search_document``).
    - ``timeout``: Request timeout in seconds (default: 60).
    - ``truncate``: Truncation strategy — ``"NONE"``, ``"START"``, or
      ``"END"`` (default: ``"END"``).
    """

    _client: AsyncCohereClient | None = None
    _model_name: str
    _api_key: str | None
    _input_type: str
    _truncate: str
    _timeout: float

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the Cohere embedding model.

        Args:
            config: Configuration dictionary from ``[embedding.cohere]``.
        """
        if not cohere_available:
            raise ImportError("Cohere library not found. Install with: pip install cohere")

        self._api_key = config.get("api_key")
        self._model_name = config.get("default_model", DEFAULT_COHERE_EMBEDDING_MODEL)
        self._input_type = config.get("input_type", DEFAULT_INPUT_TYPE)
        self._truncate = config.get("truncate", "END")
        self._timeout = float(config.get("timeout", 60.0))

        logger.info(
            "CohereEmbedding configured: model=%s, input_type=%s, truncate=%s.",
            self._model_name,
            self._input_type,
            self._truncate,
        )

    async def initialize(self) -> None:
        """Initialize the async Cohere client."""
        if self._client is not None:
            logger.debug("CohereEmbedding client already initialized.")
            return

        logger.debug("Initializing Cohere async client for embeddings...")
        try:
            self._client = AsyncCohereClient(
                api_key=self._api_key,  # None → SDK reads CO_API_KEY env var
                timeout=self._timeout,
            )
            logger.info("Cohere async client initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Cohere client: %s", e, exc_info=True)
            self._client = None
            raise ConfigError(f"Cohere client initialization failed: {e}") from e

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a single embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingError: On API failure or uninitialized client.
        """
        if self._client is None:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Cohere client not initialized. Call initialize() first.",
            )

        text_to_embed = text.replace("\n", " ") if text else ""

        logger.debug(
            "Generating Cohere embedding (len=%d, model=%s, input_type=%s)...",
            len(text_to_embed),
            self._model_name,
            self._input_type,
        )
        try:
            response = await self._client.embed(
                texts=[text_to_embed],
                model=self._model_name,
                input_type=self._input_type,
                truncate=self._truncate,
            )
            embeddings = response.embeddings
            if embeddings and len(embeddings) > 0:
                # Cohere v2 SDK returns embeddings as list of lists
                embedding = (
                    embeddings[0] if isinstance(embeddings[0], list) else list(embeddings[0])
                )
                logger.debug("Cohere embedding generated, dim=%d.", len(embedding))
                return embedding
            else:
                raise EmbeddingError(
                    model_name=self._model_name,
                    message="Cohere API returned no embedding data.",
                )
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "Cohere embedding error (model=%s): %s", self._model_name, e, exc_info=True
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Cohere API error: {e}",
            ) from e

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Cohere's API natively supports batching (up to 96 texts per call).

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: On API failure.
        """
        if self._client is None:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Cohere client not initialized. Call initialize() first.",
            )
        if not texts:
            return []

        processed = [t.replace("\n", " ") if t else "" for t in texts]

        logger.debug(
            "Generating Cohere batch embeddings (%d texts, model=%s)...",
            len(processed),
            self._model_name,
        )

        # Cohere allows up to 96 texts per request; chunk if needed
        max_batch = 96
        all_embeddings: list[list[float]] = []

        for i in range(0, len(processed), max_batch):
            batch = processed[i : i + max_batch]
            try:
                response = await self._client.embed(
                    texts=batch,
                    model=self._model_name,
                    input_type=self._input_type,
                    truncate=self._truncate,
                )
                embeddings = response.embeddings
                if embeddings and len(embeddings) == len(batch):
                    for emb in embeddings:
                        all_embeddings.append(emb if isinstance(emb, list) else list(emb))
                else:
                    raise EmbeddingError(
                        model_name=self._model_name,
                        message=(
                            f"Cohere batch returned {len(embeddings) if embeddings else 0} "
                            f"embeddings, expected {len(batch)}."
                        ),
                    )
            except EmbeddingError:
                raise
            except Exception as e:
                logger.error(
                    "Cohere batch embedding error (model=%s): %s",
                    self._model_name,
                    e,
                    exc_info=True,
                )
                raise EmbeddingError(
                    model_name=self._model_name,
                    message=f"Cohere batch API error: {e}",
                ) from e

        logger.debug("Cohere batch complete: %d embeddings.", len(all_embeddings))
        return all_embeddings

    async def close(self) -> None:
        """Close the Cohere client."""
        if self._client is not None:
            try:
                await self._client.close()
                logger.info("CohereEmbedding client closed.")
            except Exception as e:
                logger.error("Error closing Cohere client: %s", e, exc_info=True)
            finally:
                self._client = None
