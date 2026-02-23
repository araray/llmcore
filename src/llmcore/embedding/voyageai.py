# src/llmcore/embedding/voyageai.py
"""
VoyageAI Embedding model implementation for LLMCore.

Uses the VoyageAI Python SDK to generate embeddings via their API.
VoyageAI specialises in retrieval-optimised embeddings with state-of-the-art
performance on MTEB benchmarks.

Supported models:
- ``voyage-3-large`` (1024 dims, high performance)
- ``voyage-3`` (1024 dims, balanced)
- ``voyage-3-lite`` (512 dims, fast)
- ``voyage-code-3`` (1024 dims, code-optimised)

VoyageAI supports an ``input_type`` parameter:
- ``"document"`` — for indexing content
- ``"query"`` — for search queries

References:
    - https://docs.voyageai.com/docs/embeddings
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §6.2 (Embedding System)
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import voyageai
    from voyageai import AsyncClient as AsyncVoyageClient

    voyageai_available = True
except ImportError:
    voyageai_available = False
    AsyncVoyageClient = None  # type: ignore[assignment,misc]

from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_VOYAGEAI_EMBEDDING_MODEL = "voyage-3"
DEFAULT_INPUT_TYPE = "document"


class VoyageAIEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using the VoyageAI API.

    Requires the ``voyageai`` library (``pip install voyageai``).

    Configuration keys (from ``[embedding.voyageai]``):

    - ``api_key``: VoyageAI API key (falls back to ``VOYAGE_API_KEY`` env var).
    - ``default_model``: Model name (default: ``voyage-3``).
    - ``input_type``: ``"document"`` or ``"query"`` (default: ``"document"``).
    - ``timeout``: Request timeout in seconds (default: 60).
    - ``truncation``: Whether to auto-truncate (default: ``True``).
    """

    _client: AsyncVoyageClient | None = None
    _model_name: str
    _api_key: str | None
    _input_type: str | None
    _truncation: bool
    _timeout: float

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the VoyageAI embedding model.

        Args:
            config: Configuration dictionary from ``[embedding.voyageai]``.
        """
        if not voyageai_available:
            raise ImportError("VoyageAI library not found. Install with: pip install voyageai")

        self._api_key = config.get("api_key")
        self._model_name = config.get("default_model", DEFAULT_VOYAGEAI_EMBEDDING_MODEL)
        self._input_type = config.get("input_type", DEFAULT_INPUT_TYPE)
        self._truncation = config.get("truncation", True)
        self._timeout = float(config.get("timeout", 60.0))

        logger.info(
            "VoyageAIEmbedding configured: model=%s, input_type=%s, truncation=%s.",
            self._model_name,
            self._input_type,
            self._truncation,
        )

    async def initialize(self) -> None:
        """Initialize the async VoyageAI client."""
        if self._client is not None:
            logger.debug("VoyageAIEmbedding client already initialized.")
            return

        logger.debug("Initializing VoyageAI async client for embeddings...")
        try:
            self._client = AsyncVoyageClient(
                api_key=self._api_key,  # None → SDK reads VOYAGE_API_KEY env var
                timeout=self._timeout,
            )
            logger.info("VoyageAI async client initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize VoyageAI client: %s", e, exc_info=True)
            self._client = None
            raise ConfigError(f"VoyageAI client initialization failed: {e}") from e

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
                message="VoyageAI client not initialized. Call initialize() first.",
            )

        text_to_embed = text.replace("\n", " ") if text else ""

        logger.debug(
            "Generating VoyageAI embedding (len=%d, model=%s)...",
            len(text_to_embed),
            self._model_name,
        )
        try:
            response = await self._client.embed(
                texts=[text_to_embed],
                model=self._model_name,
                input_type=self._input_type,
                truncation=self._truncation,
            )
            if response.embeddings and len(response.embeddings) > 0:
                embedding = response.embeddings[0]
                logger.debug("VoyageAI embedding generated, dim=%d.", len(embedding))
                return list(embedding)
            else:
                raise EmbeddingError(
                    model_name=self._model_name,
                    message="VoyageAI API returned no embedding data.",
                )
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "VoyageAI embedding error (model=%s): %s", self._model_name, e, exc_info=True
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"VoyageAI API error: {e}",
            ) from e

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        VoyageAI supports batching up to 128 texts per request.

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
                message="VoyageAI client not initialized. Call initialize() first.",
            )
        if not texts:
            return []

        processed = [t.replace("\n", " ") if t else "" for t in texts]

        logger.debug(
            "Generating VoyageAI batch embeddings (%d texts, model=%s)...",
            len(processed),
            self._model_name,
        )

        # VoyageAI allows up to 128 texts per request
        max_batch = 128
        all_embeddings: list[list[float]] = []

        for i in range(0, len(processed), max_batch):
            batch = processed[i : i + max_batch]
            try:
                response = await self._client.embed(
                    texts=batch,
                    model=self._model_name,
                    input_type=self._input_type,
                    truncation=self._truncation,
                )
                if response.embeddings and len(response.embeddings) == len(batch):
                    for emb in response.embeddings:
                        all_embeddings.append(list(emb))
                else:
                    raise EmbeddingError(
                        model_name=self._model_name,
                        message=(
                            f"VoyageAI batch returned {len(response.embeddings) if response.embeddings else 0} "
                            f"embeddings, expected {len(batch)}."
                        ),
                    )
            except EmbeddingError:
                raise
            except Exception as e:
                logger.error(
                    "VoyageAI batch embedding error (model=%s): %s",
                    self._model_name,
                    e,
                    exc_info=True,
                )
                raise EmbeddingError(
                    model_name=self._model_name,
                    message=f"VoyageAI batch API error: {e}",
                ) from e

        logger.debug("VoyageAI batch complete: %d embeddings.", len(all_embeddings))
        return all_embeddings

    async def close(self) -> None:
        """Close the VoyageAI client."""
        if self._client is not None:
            try:
                await self._client.close()
                logger.info("VoyageAIEmbedding client closed.")
            except Exception as e:
                logger.error("Error closing VoyageAI client: %s", e, exc_info=True)
            finally:
                self._client = None
