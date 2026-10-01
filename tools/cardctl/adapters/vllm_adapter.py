# tools/cardctl/adapters/vllm_adapter.py
"""vLLM adapter — discovers whatever a given deployment actually loaded.

Unlike every other adapter here there is no vendor catalog, because vLLM is
self-hosted: "the model list" is whatever the server in front of you is serving.
So this adapter requires a ``base_url`` and reports that server's
``/v1/models``, which makes the resulting cards specific to one deployment
rather than universal. Pass ``--base-url`` when generating.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import NormalizedModel
from .openai_compat import OpenAICompatAdapter

logger = logging.getLogger(__name__)


class VLLMAdapter(OpenAICompatAdapter):
    """Cards for the models a specific vLLM server is serving."""

    provider_name = "vllm"
    api_key_env_var = "VLLM_API_KEY"
    # Self-hosted: there is no default, and guessing localhost would silently
    # produce cards for whatever happens to be running on this machine.
    base_url = ""
    requires_api_key = False

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch from the configured server.

        Raises:
            RuntimeError: If no ``base_url`` was supplied, since there is no
                sensible default for a self-hosted server.
        """
        if not self.base_url:
            raise RuntimeError(
                "The vLLM adapter needs the address of your server: there is no "
                "vendor catalog, because vLLM serves whatever you loaded. Pass "
                "--base-url http://host:8000/v1 (the resulting cards describe that "
                "deployment, not vLLM in general)."
            )
        return await super().fetch_models()

    def _enrich_model(self, model: NormalizedModel, raw: dict[str, Any]) -> NormalizedModel:
        """Record what the server reported, including its context window."""
        model.open_weights = True
        model.tags = [*model.tags, "self-hosted"]
        # vLLM reports the real serving context, which is the deployment's
        # --max-model-len rather than the model's theoretical maximum. That is
        # the more useful number for a caller, so it is used as-is.
        ctx = raw.get("max_model_len")
        if isinstance(ctx, int) and ctx > 0:
            model.context_length = ctx
        model.supports_streaming = True
        model.supports_tools = True
        return model
