# src/llmcore/providers/gpuai_provider.py
"""LLMCore provider for the gpu.ai serverless inference API.

gpu.ai sells three things: text inference, media generation (image and
video), and GPU rental. This provider covers the first. The media surface
belongs to :mod:`llmcore.media` and the rental surface to
:mod:`llmcore.runtimes`, which has no gpu.ai backend yet -- the same gap
DeepInfra has, since that subsystem has only ever had one backend.

The text surface is OpenAI-compatible, verified against the live endpoint,
so almost everything is inherited from :class:`OpenAIProvider`. What is
actually gpu.ai-specific:

* credentials (``GPUAI_API_KEY``) and the base URL;
* an unusually rich ``/v1/models``, which states ``context_length``,
  ``pricing``, ``supported_parameters`` and ``aliases`` inline -- more than
  any other provider integrated here, and enough to generate near-complete
  model cards;
* filtering that catalogue by ``modality``, because two thirds of it is
  image and video models this provider cannot serve;
* ``/v1/embeddings``, which the parent does not implement.

Two behaviours worth knowing before relying on cost numbers, both found by
calling the API rather than reading about it:

**Reasoning tokens are billed as output, with no breakdown.** Responses
carry no ``completion_tokens_details``, so the reasoning share cannot be
separated from visible output -- ``gpuai/gpt-oss-120b`` spent 45 completion
tokens to answer "OK". Costs here therefore assume reasoning is billed at
the output rate, which the catalogue's single output price supports. It is
an assumption, recorded rather than hidden.

**``max_tokens`` includes reasoning.** A budget smaller than the reasoning
overhead returns *empty content*, not shorter content. A cost-saving layer
that trims ``max_tokens`` will silently produce nothing on these models, so
:data:`MIN_REASONING_MAX_TOKENS` documents a floor worth respecting.

Pricing from the catalogue is the vendor's published rate card and should be
treated as a reference rather than an invoice; see ``docs/model_cards.md``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..exceptions import ConfigError, ProviderError
from ..models import ModelDetails
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

#: The OpenAI-compatible root. ``/v1`` is included because the catalogue,
#: chat and embedding endpoints all sit directly beneath it.
DEFAULT_GPUAI_BASE_URL = "https://api.gpu.ai/v1"

#: Cheapest chat model at the time of integration (15 cents/Mtok in).
DEFAULT_GPUAI_MODEL = "gpuai/qwen3.8-flash"

#: Used only when neither the catalogue nor a card states a window.
DEFAULT_GPUAI_CONTEXT_LENGTH = 32_768

#: Env vars consulted, in order.
_GPUAI_API_KEY_ENV_VARS = ("GPUAI_API_KEY", "GPU_AI_API_KEY")

#: ``modality`` values this provider can serve. The catalogue also lists
#: ``image`` and ``video``, which are the media subsystem's business.
CHAT_MODALITIES = frozenset({"chat"})
EMBEDDING_MODALITIES = frozenset({"embedding"})

#: A floor for ``max_tokens`` on reasoning models. Below roughly this, the
#: reasoning pass consumes the whole budget and the response arrives with
#: empty content -- observed at ``max_tokens=8``, which returned ``""``
#: alongside ``completion_tokens: 8``.
MIN_REASONING_MAX_TOKENS = 256


class GpuAiProvider(OpenAIProvider):
    """gpu.ai serverless text inference (chat + embeddings)."""

    #: Image and video generation live on gpu.ai but are not served here;
    #: they need media adapters and the non-token pricing the schema only
    #: just gained.
    _MEDIA_CAPABILITIES: frozenset[Any] = frozenset()

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialise the gpu.ai provider.

        Args:
            config: Recognised keys include ``api_key``, ``api_key_env_var``,
                ``base_url``, ``default_model``, ``timeout``, ``max_retries``
                and ``default_context_length``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If no API key can be resolved.
        """
        cfg = dict(config)
        cfg.setdefault("base_url", DEFAULT_GPUAI_BASE_URL)
        cfg.setdefault("default_model", DEFAULT_GPUAI_MODEL)

        if not cfg.get("api_key"):
            resolved: str | None = None
            env_var = cfg.get("api_key_env_var")
            if env_var:
                resolved = os.environ.get(env_var)
            if not resolved:
                for candidate in _GPUAI_API_KEY_ENV_VARS:
                    if resolved := os.environ.get(candidate):
                        break
            if resolved:
                cfg["api_key"] = resolved

        super().__init__(cfg, log_raw_payloads=log_raw_payloads)

        try:
            self._default_context_length = int(
                config.get("default_context_length", DEFAULT_GPUAI_CONTEXT_LENGTH)
            )
        except (TypeError, ValueError):
            self._default_context_length = DEFAULT_GPUAI_CONTEXT_LENGTH

        #: model id (and alias) -> context window, from the catalogue.
        self._catalogue_context: dict[str, int] = {}

        logger.debug(
            "GpuAiProvider initialised: base_url=%s default_model=%s",
            self.base_url,
            self.default_model,
        )

    def get_name(self) -> str:
        """Return the provider instance name (``"gpuai"`` by default)."""
        return self._provider_instance_name or "gpuai"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def get_models_details(self) -> list[ModelDetails]:
        """Read ``/v1/models`` and return only the text models.

        The catalogue is richer than most: each entry states
        ``context_length``, ``pricing``, ``supported_parameters`` and
        ``aliases``. It also mixes modalities, and two thirds of it is image
        and video, so it is filtered -- listing a video model as a chat
        model would make it selectable by a router that cannot call it.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        try:
            if self._transport == "httpx":
                # Preferred: raw JSON keeps gpu.ai's extra fields, which a
                # typed SDK model can drop. `context_length` and `aliases`
                # are exactly the fields worth keeping.
                response = await self._get_http().get("/models")
                if response.status_code >= 400:
                    self._raise_direct_status(
                        response.status_code, response.text, "models"
                    )
                entries: list[Any] = response.json().get("data") or []
            else:
                listing = await self._client.models.list()
                entries = list(getattr(listing, "data", None) or [])
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - network
            raise ProviderError(
                self.get_name(), f"model discovery failed: {exc}"
            ) from exc

        models: list[ModelDetails] = []
        for entry in entries:
            item = entry if isinstance(entry, dict) else _as_dict(entry)
            model_id = item.get("id")
            if not model_id:
                continue

            modality = str(item.get("modality") or "").lower()
            if modality not in CHAT_MODALITIES | EMBEDDING_MODALITIES:
                continue

            window = item.get("context_length")
            window = int(window) if isinstance(window, int) and window > 0 else None
            if window:
                self._catalogue_context[model_id] = window
                for alias in item.get("aliases") or []:
                    self._catalogue_context[str(alias)] = window

            params = set(item.get("supported_parameters") or [])
            models.append(
                ModelDetails(
                    id=str(model_id),
                    provider_name=self.get_name(),
                    display_name=item.get("name") or str(model_id),
                    context_length=window or self._default_context_length,
                    # The catalogue enumerates *accepted parameters*, so a
                    # missing `stream` means streaming is not accepted --
                    # a real negative, unlike the capability stubs on
                    # generated cards. Tools and vision are not described
                    # by that list at all, so they are left to the card
                    # rather than guessed from it.
                    supports_streaming="stream" in params,
                    model_type="embedding" if modality in EMBEDDING_MODALITIES else "chat",
                    metadata={
                        "source": "gpuai_catalogue",
                        "modality": modality,
                        "tier": item.get("tier"),
                        "status": item.get("status"),
                        "author": item.get("author"),
                        "aliases": list(item.get("aliases") or []),
                        "supported_parameters": sorted(params),
                        "pricing_raw": item.get("pricing"),
                    },
                )
            )

        logger.debug("gpu.ai discovery: %d text models", len(models))
        return models

    def get_max_context_length(self, model: str | None = None) -> int:
        """Context window for *model*, preferring what the catalogue said.

        Order: the live catalogue (if discovery has run), then the parent's
        card-registry lookup, then the configured default. The catalogue is
        preferred because it is the vendor describing its own endpoint.
        """
        target = model or self.default_model
        if cached := self._catalogue_context.get(str(target)):
            return cached
        try:
            inherited = super().get_max_context_length(target)
        except Exception:  # pragma: no cover - defensive
            inherited = 0
        return inherited or self._default_context_length

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def create_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed *texts* via gpu.ai's OpenAI-compatible endpoint.

        The parent does not implement this, and gpu.ai's ``/v1/embeddings``
        is OpenAI-shaped (verified: 4096 dimensions for
        ``gpuai/qwen3-embedding-8b``).

        Deliberately uses the direct HTTP path regardless of the configured
        transport, because the OpenAI SDK injects ``encoding_format`` and
        gpu.ai rejects it outright::

            Parameter "encoding_format" is not supported in v1.1

        Not a value it dislikes -- the parameter itself. The SDK gives no
        clean way to suppress it, so the payload is built here instead,
        which is the only way to send exactly the fields gpu.ai accepts.
        """
        if not texts:
            return []
        target = model or "gpuai/qwen3-embedding-8b"
        payload: dict[str, Any] = {"model": target, "input": texts}
        # Only pass through what the caller asked for; nothing is added.
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        try:
            response = await self._get_http().post("/embeddings", json=payload)
            if response.status_code >= 400:
                self._raise_direct_status(
                    response.status_code, response.text, "embeddings"
                )
            data = response.json().get("data") or []
            return [list(item["embedding"]) for item in data]
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                self.get_name(), f"embedding request failed: {exc}"
            ) from exc


def _as_dict(entry: Any) -> dict[str, Any]:
    """Best-effort dict view of an SDK model object.

    The OpenAI SDK may hand back a typed object that drops gpu.ai's extra
    fields from its attributes while keeping them in ``model_extra``; both
    are merged so ``context_length`` and ``aliases`` survive.
    """
    for attr in ("model_dump", "dict"):
        dumper = getattr(entry, attr, None)
        if callable(dumper):
            try:
                data = dumper()
            except Exception:  # pragma: no cover - defensive
                continue
            if isinstance(data, dict):
                extra = data.get("model_extra")
                return {**data, **(extra if isinstance(extra, dict) else {})}
    extra = getattr(entry, "model_extra", None)
    base = {k: getattr(entry, k) for k in ("id", "object", "owned_by")
            if hasattr(entry, k)}
    return {**base, **(extra if isinstance(extra, dict) else {})}


__all__ = [
    "CHAT_MODALITIES",
    "DEFAULT_GPUAI_BASE_URL",
    "DEFAULT_GPUAI_MODEL",
    "EMBEDDING_MODALITIES",
    "GpuAiProvider",
    "MIN_REASONING_MAX_TOKENS",
]
