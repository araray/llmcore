# src/llmcore/embedding/google.py
"""
Google AI (Gemini) Embedding model implementation for LLMCore.

Uses the google-genai Python SDK (v1.72.0+) to generate embeddings via their API.

BREAKING CHANGE vs. prior code (pre-v1.72.0 SDK):
  - embed_content() now takes ``contents=`` (not ``content=``)
  - task_type is now inside ``config=EmbedContentConfig(task_type=...)``
  - Response is ``EmbedContentResponse`` (Pydantic model), not a dict.
    Access embeddings via ``result.embeddings[i].values``.
"""

from __future__ import annotations

import logging
from typing import Any

# --- google-genai library imports ---
try:
    import google.genai as genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    google_genai_available = True
except ImportError:
    google_genai_available = False
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    genai_errors = None  # type: ignore

from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Default model — gemini-embedding-001 went GA Feb 2026, replaces
# the shutdown text-embedding-004 (shutdown Jan 14, 2026).
DEFAULT_GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"

# Known valid task types as documented by Google
VALID_TASK_TYPES = [
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
]


class GoogleAIEmbedding(BaseEmbeddingModel):
    """
    Generates text embeddings using the Google AI (Gemini) API via google-genai.

    Requires the ``google-genai`` library to be installed.
    Configuration for API key and model is read from ``[embedding.google]``.

    Updated for google-genai SDK v1.72.0:
    - Uses ``contents=`` parameter (not ``content=``)
    - Wraps ``task_type`` in ``EmbedContentConfig`` (not a top-level kwarg)
    - Parses ``EmbedContentResponse.embeddings[i].values`` (not dict access)
    - Supports ``output_dimensionality`` via config
    """

    _client: Any = None
    _model_name: str
    _api_key: str | None = None
    _task_type: str = "RETRIEVAL_DOCUMENT"
    _output_dimensionality: int | None = None

    def __init__(self, config: dict[str, Any]):
        """
        Initializes the GoogleAIEmbedding model using google-genai.

        Args:
            config: Configuration dictionary. Expected keys from ``[embedding.google]``:
                'api_key' (optional): Google AI API key.
                'default_model' (optional): Embedding model name
                    (e.g., ``"text-embedding-004"``).
                'task_type' (optional): Embedding task type string
                    (``'RETRIEVAL_DOCUMENT'``, ``'RETRIEVAL_QUERY'``, etc.).
                    Defaults to ``'RETRIEVAL_DOCUMENT'``.
                'output_dimensionality' (optional): Reduce output embedding
                    dimension. Supported by newer models (not embedding-001).
        """
        if not google_genai_available:
            raise ImportError(
                "Google Gen AI library (`google-genai`) not found. "
                "Please install `google-genai` (e.g., `pip install llmcore[gemini]`)."
            )

        embedding_specific_config = config

        self._api_key = embedding_specific_config.get("api_key")
        raw_model_name = embedding_specific_config.get(
            "default_model", DEFAULT_GOOGLE_EMBEDDING_MODEL
        )
        # Store without 'models/' prefix — we add it at call time
        self._model_name = raw_model_name.replace("models/", "")

        # Parse task type
        task_type_str = embedding_specific_config.get(
            "task_type", "RETRIEVAL_DOCUMENT"
        ).upper()
        if task_type_str not in VALID_TASK_TYPES:
            logger.warning(
                f"Provided task_type '{task_type_str}' is not in the known "
                f"valid list {VALID_TASK_TYPES}. Using it anyway, but it "
                f"might cause API errors if invalid."
            )
        self._task_type = task_type_str

        # Optional: output dimensionality
        output_dim = embedding_specific_config.get("output_dimensionality")
        if output_dim is not None:
            self._output_dimensionality = int(output_dim)

        logger.info(
            f"GoogleAIEmbedding configured with model '{self._model_name}' "
            f"and task_type '{self._task_type}'. API key source: "
            f"{'config' if self._api_key else 'environment/SDK default'}."
        )

    def _build_embed_config(self) -> Any:
        """Build an EmbedContentConfig with task_type and optional dimensionality.

        Returns:
            A ``genai_types.EmbedContentConfig`` instance.
        """
        config_kwargs: dict[str, Any] = {"task_type": self._task_type}
        if self._output_dimensionality is not None:
            config_kwargs["output_dimensionality"] = self._output_dimensionality
        return genai_types.EmbedContentConfig(**config_kwargs)

    async def initialize(self) -> None:
        """Initializes the google-genai client."""
        if self._client:
            logger.debug("GoogleAIEmbedding client already initialized.")
            return

        logger.debug(
            f"Initializing Google Gen AI client for embeddings "
            f"(model: {self._model_name})..."
        )
        try:
            client_options: dict[str, Any] = {}
            if self._api_key:
                client_options["api_key"] = self._api_key
            self._client = genai.Client(**client_options)
            logger.info(
                f"GoogleAIEmbedding client initialized successfully "
                f"for model '{self._model_name}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Google Gen AI client for embeddings: {e}",
                exc_info=True,
            )
            self._client = None
            raise ConfigError(
                f"Google Gen AI client initialization for embeddings failed: {e}"
            )

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text string.

        Uses the google-genai v1.72.0+ API:
        - ``contents=`` parameter (not ``content=``)
        - ``config=EmbedContentConfig(task_type=...)``
        - Response: ``EmbedContentResponse.embeddings[0].values``

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        if not self._client or not genai_errors:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Google Gen AI client/errors not available.",
            )
        if not text:
            logger.warning(
                "generate_embedding called with empty text for Google AI."
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message="Input text cannot be empty for Google AI embeddings.",
            )

        model_with_prefix = f"models/{self._model_name}"
        embed_config = self._build_embed_config()

        logger.debug(
            f"Generating Google AI embedding for single text "
            f"(length: {len(text)}, model: {model_with_prefix}, "
            f"task: {self._task_type})..."
        )
        try:
            result = await self._client.aio.models.embed_content(
                model=model_with_prefix,
                contents=text,
                config=embed_config,
            )

            # SDK v1.72.0: result is EmbedContentResponse (Pydantic model).
            # Access: result.embeddings[0].values
            if result.embeddings and len(result.embeddings) > 0:
                embedding_values = result.embeddings[0].values
                if embedding_values:
                    logger.debug(
                        f"Successfully generated Google AI embedding, "
                        f"dimension: {len(embedding_values)}."
                    )
                    return embedding_values

            logger.error(
                f"Google AI embedding API returned no embedding data for "
                f"model '{model_with_prefix}'. Response: {result}"
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message="API returned no embedding data.",
            )
        except genai_errors.GoogleAPIError as e:
            logger.error(
                f"Google AI API error during embedding generation "
                f"(model: {model_with_prefix}): {e}",
                exc_info=True,
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Google AI API Error: {e}",
            )
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating Google AI embedding "
                f"(model: {model_with_prefix}): {e}",
                exc_info=True,
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Unexpected error: {e}",
            )

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of text strings.

        Uses the google-genai v1.72.0+ batch API.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of float-lists, one per input text (zero-vectors for empties).

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        if not self._client or not genai_errors:
            raise EmbeddingError(
                model_name=self._model_name,
                message="Google Gen AI client/errors not available.",
            )
        if not texts:
            return []

        # Filter empty strings, track original indices
        processed_texts: list[str] = []
        indices_with_content: list[int] = []
        for i, text_content in enumerate(texts):
            if not text_content:
                logger.warning(
                    f"Empty string found at index {i} in batch texts for "
                    f"Google AI. Skipping."
                )
            else:
                processed_texts.append(text_content)
                indices_with_content.append(i)

        if not processed_texts:
            logger.warning(
                "generate_embeddings called with batch containing only "
                "empty strings."
            )
            return []

        model_with_prefix = f"models/{self._model_name}"
        embed_config = self._build_embed_config()

        logger.debug(
            f"Generating Google AI embeddings for batch of "
            f"{len(processed_texts)} non-empty texts "
            f"(model: {model_with_prefix}, task: {self._task_type})..."
        )
        try:
            result = await self._client.aio.models.embed_content(
                model=model_with_prefix,
                contents=processed_texts,
                config=embed_config,
            )

            # SDK v1.72.0: result.embeddings is list[ContentEmbedding]
            embeddings_data = result.embeddings
            if (
                embeddings_data
                and len(embeddings_data) == len(processed_texts)
            ):
                # Extract .values from each ContentEmbedding
                raw_embeddings = [emb.values for emb in embeddings_data]

                logger.debug(
                    f"Successfully generated batch of "
                    f"{len(raw_embeddings)} Google AI embeddings."
                )

                # Reconstruct full list with zero-vectors for skipped empties
                dimension = (
                    len(raw_embeddings[0]) if raw_embeddings else 0
                )
                final_embeddings: list[list[float] | None] = [None] * len(
                    texts
                )
                for i, original_index in enumerate(indices_with_content):
                    final_embeddings[original_index] = raw_embeddings[i]

                output_embeddings: list[list[float]] = []
                for emb in final_embeddings:
                    if emb is not None:
                        output_embeddings.append(emb)
                    else:
                        output_embeddings.append([0.0] * dimension)

                return output_embeddings
            else:
                actual_count = (
                    len(embeddings_data) if embeddings_data else 0
                )
                logger.error(
                    f"Google AI embedding API returned mismatched data count "
                    f"for model '{model_with_prefix}'. Expected "
                    f"{len(processed_texts)}, got {actual_count}."
                )
                raise EmbeddingError(
                    model_name=self._model_name,
                    message="API returned mismatched or no embedding data "
                    "for batch.",
                )

        except genai_errors.GoogleAPIError as e:
            logger.error(
                f"Google AI API error during batch embedding generation "
                f"(model: {model_with_prefix}): {e}",
                exc_info=True,
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Google AI API Error: {e}",
            )
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating batch Google AI embeddings "
                f"(model: {model_with_prefix}): {e}",
                exc_info=True,
            )
            raise EmbeddingError(
                model_name=self._model_name,
                message=f"Unexpected batch error: {e}",
            )

    async def close(self) -> None:
        """Clean up resources for GoogleAIEmbedding.

        The google-genai client typically does not require explicit closing.
        """
        logger.debug(
            f"GoogleAIEmbedding for model '{self._model_name}' closed "
            f"(no specific client cleanup typically needed)."
        )
        self._client = None
