# src/llmcore/embedding/openai.py
"""
OpenAI Embedding model implementation for LLMCore.

Supports dimensions parameter for Matryoshka embeddings and
encoding_format for base64 transfer efficiency.

Tested against openai Python SDK v2.31.0.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import openai
    from openai import AsyncOpenAI
    from openai._exceptions import (
        APIConnectionError as OpenAIAPIConnectionError,
    )
    from openai._exceptions import (
        APIError as OpenAIAPIError,
    )
    from openai._exceptions import (
        APIStatusError as OpenAIAPIStatusError,
    )
    from openai._exceptions import (
        APITimeoutError as OpenAIAPITimeoutError,
    )
    from openai._exceptions import (
        OpenAIError,
    )

    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    OpenAIAPIError = Exception  # type: ignore
    OpenAIAPIStatusError = Exception  # type: ignore
    OpenAIAPIConnectionError = Exception  # type: ignore
    OpenAIAPITimeoutError = Exception  # type: ignore

from ..exceptions import ConfigError, EmbeddingError
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedding(BaseEmbeddingModel):
    """Generates text embeddings using the OpenAI API.

    Supports dimensions parameter for text-embedding-3-* models.
    """

    _client: AsyncOpenAI | None = None
    _model_name: str
    _api_key: str | None = None
    _base_url: str | None = None
    _timeout: float = 60.0
    _dimensions: int | None = None
    _encoding_format: str | None = None

    def __init__(self, config: dict[str, Any]):
        if not openai_available:
            raise ImportError("OpenAI library not found.")

        self._api_key = config.get("api_key")
        if not self._api_key:
            logger.debug("API key not in embedding config, using env var.")

        self._model_name = config.get("default_model", DEFAULT_OPENAI_EMBEDDING_MODEL)
        self._base_url = config.get("base_url")
        self._timeout = float(config.get("timeout", 60.0))
        self._dimensions = config.get("dimensions")
        self._encoding_format = config.get("encoding_format")

        logger.info(
            "OpenAIEmbedding: model='%s', dimensions=%s",
            self._model_name,
            self._dimensions or "default",
        )

    async def initialize(self) -> None:
        if self._client:
            return
        try:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
            logger.info("OpenAI embedding client initialized.")
        except Exception as e:
            self._client = None
            raise ConfigError(f"OpenAI embedding init failed: {e}")

    def _build_embed_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        if self._encoding_format is not None:
            kwargs["encoding_format"] = self._encoding_format
        return kwargs

    async def generate_embedding(self, text: str) -> list[float]:
        if not self._client:
            raise EmbeddingError(model_name=self._model_name, message="Client not initialized.")
        text_to_embed = text.replace("\n", " ") if text else text
        try:
            response = await self._client.embeddings.create(
                model=self._model_name,
                input=[text_to_embed],
                **self._build_embed_kwargs(),
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            raise EmbeddingError(model_name=self._model_name, message="No data returned.")
        except OpenAIAPIStatusError as e:
            raise EmbeddingError(
                model_name=self._model_name, message=f"API Error ({e.status_code}): {e}"
            )
        except OpenAIAPITimeoutError:
            raise EmbeddingError(model_name=self._model_name, message="Request timed out.")
        except OpenAIAPIConnectionError as e:
            raise EmbeddingError(model_name=self._model_name, message=f"Connection error: {e}")
        except OpenAIAPIError as e:
            raise EmbeddingError(model_name=self._model_name, message=f"API error: {e}")
        except OpenAIError as e:
            raise EmbeddingError(model_name=self._model_name, message=f"Error: {e}")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(model_name=self._model_name, message=f"Unexpected: {e}")

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            raise EmbeddingError(model_name=self._model_name, message="Client not initialized.")
        if not texts:
            return []
        processed = [t.replace("\n", " ") if t else "" for t in texts]
        try:
            response = await self._client.embeddings.create(
                model=self._model_name,
                input=processed,
                **self._build_embed_kwargs(),
            )
            if response.data and len(response.data) == len(texts):
                return [item.embedding for item in response.data]
            raise EmbeddingError(model_name=self._model_name, message="Mismatched batch data.")
        except OpenAIAPIStatusError as e:
            raise EmbeddingError(
                model_name=self._model_name, message=f"API Error ({e.status_code}): {e}"
            )
        except OpenAIAPITimeoutError:
            raise EmbeddingError(model_name=self._model_name, message="Batch timed out.")
        except OpenAIAPIConnectionError as e:
            raise EmbeddingError(model_name=self._model_name, message=f"Connection error: {e}")
        except OpenAIAPIError as e:
            raise EmbeddingError(model_name=self._model_name, message=f"API error: {e}")
        except OpenAIError as e:
            raise EmbeddingError(model_name=self._model_name, message=f"Error: {e}")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                model_name=self._model_name, message=f"Unexpected batch error: {e}"
            )

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.close()
                logger.info("OpenAIEmbedding closed.")
            except Exception as e:
                logger.error(f"Error closing embedding client: {e}", exc_info=True)
            finally:
                self._client = None
