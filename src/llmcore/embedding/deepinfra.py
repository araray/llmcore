# src/llmcore/embedding/deepinfra.py
"""
DeepInfra embedding model implementation for LLMCore.

DeepInfra exposes an OpenAI-compatible embeddings endpoint
(``POST https://api.deepinfra.com/v1/openai/embeddings``), so this backend is a
thin subclass of :class:`~llmcore.embedding.openai.OpenAIEmbedding` that:

* defaults ``base_url`` to the DeepInfra OpenAI-compatible endpoint,
* resolves the API key from ``DEEPINFRA_TOKEN`` / ``DEEPINFRA_API_KEY`` (or the
  configured ``api_key`` / ``api_key_env_var``),
* defaults the model to ``Qwen/Qwen3-Embedding-8B``, and
* never sends a ``dimensions`` parameter — DeepInfra supports only
  ``encoding_format="float"``.

Reference: https://docs.deepinfra.com/apis/embeddings
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

#: OpenAI-compatible embeddings base URL.
DEFAULT_DEEPINFRA_EMBEDDING_BASE_URL = "https://api.deepinfra.com/v1/openai"

#: Default embedding model (DeepInfra embeddings doc).
DEFAULT_DEEPINFRA_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

#: Environment variables searched (in order) for the API token.
_DEEPINFRA_API_KEY_ENV_VARS = ("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY")


class DeepInfraEmbedding(OpenAIEmbedding):
    """Generates text embeddings using the DeepInfra OpenAI-compatible API.

    All request/response handling is inherited from
    :class:`OpenAIEmbedding`; only the credential resolution and DeepInfra
    defaults differ.  The ``dimensions`` parameter is intentionally dropped
    because DeepInfra does not support it.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialise the DeepInfra embedding backend.

        Args:
            config: Embedding configuration.  Recognised keys include
                ``api_key``, ``api_key_env_var``, ``base_url``,
                ``default_model``, ``timeout`` and ``encoding_format``.
        """
        cfg = dict(config)
        cfg.setdefault("base_url", DEFAULT_DEEPINFRA_EMBEDDING_BASE_URL)
        cfg.setdefault("default_model", DEFAULT_DEEPINFRA_EMBEDDING_MODEL)

        if not cfg.get("api_key"):
            env_var = cfg.get("api_key_env_var")
            resolved: str | None = os.environ.get(env_var) if env_var else None
            if not resolved:
                for candidate in _DEEPINFRA_API_KEY_ENV_VARS:
                    resolved = os.environ.get(candidate)
                    if resolved:
                        break
            if resolved:
                cfg["api_key"] = resolved

        # DeepInfra does not accept a ``dimensions`` parameter.
        cfg.pop("dimensions", None)

        super().__init__(cfg)
        logger.info(
            "DeepInfraEmbedding: model='%s' base_url='%s'",
            self._model_name,
            self._base_url,
        )
