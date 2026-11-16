# src/llmcore/providers/mistral_provider.py
"""
Native Mistral AI API provider implementation for the LLMCore library.

Provides a dedicated provider for the Mistral AI platform, supporting:
- Chat completions (with Mistral-specific parameters: random_seed,
  safe_prompt, reasoning_effort, guardrails, prediction, parallel_tool_calls)
- Fill-in-the-Middle (FIM) completions for code (Codestral)
- Text-to-Speech (TTS) via /v1/audio/speech with voice cloning support
- Speech-to-Text (STT) via /v1/audio/transcriptions with diarization
- OCR via /v1/ocr with structured document annotation
- Embeddings via /v1/embeddings (with output_dimension, output_dtype)
- Classification and Moderation endpoints
- Dynamic model discovery via /v1/models

Uses ``httpx.AsyncClient`` directly; the ``mistralai`` SDK (v3.7) is an
optional second transport (``backend = "sdk"``) for chat, models and embeddings.

Reasoning (Mistral Small 4, Medium 3.5, Large 4, the hosted GLM models):
``effort=`` maps to ``reasoning_effort`` within each model's accepted values.
A reasoning answer's ``content`` is a list of thinking and text chunks; this
provider returns the text as ``content``, the thinking as
``reasoning_content``, and the raw chunks as ``mistral_content``, which
:meth:`assistant_replay_metadata` hands back so the next turn replays them
(Mistral: "do not strip ThinkChunk from assistant messages").

``responses_api = "always"`` routes chat through Mistral's stateless
``/v1/responses`` instead of ``/v1/chat/completions``.

Audited against the live API and mistralai 3.7 on 2026-11-16.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import secrets
from collections.abc import AsyncGenerator
from typing import Any

try:
    import httpx

    httpx_available = True
except ImportError:
    httpx_available = False
    httpx = None  # type: ignore

try:
    import tiktoken

    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None  # type: ignore

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from ..tokens import EstimateCounter as _EstimateCounter
from . import openai_responses as rsp
from .base import BaseProvider, ContextPayload
from ._media_bridge import LegacyMediaBridge
from .openai_provider import nearest_effort

try:
    from mistralai.client import Mistral as _MistralSDK

    mistralai_available = True
except ImportError:
    mistralai_available = False
    _MistralSDK = None  # type: ignore

logger = logging.getLogger(__name__)

# Default base URL for the Mistral AI API.
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

#: Calls the ``mistralai`` v3 SDK covers in a shape llmcore can consume
#: directly. Mistral's surface is wide — OCR, audio, classification,
#: moderation, FIM — and the SDK models those with its own typed resources
#: rather than OpenAI-shaped dicts, so converting all of them would mean
#: maintaining a second translation layer. The honest middle ground is to use
#: the SDK where it is a drop-in and keep the rest on direct REST, logging that
#: choice so ``backend = "sdk"`` never silently does nothing.
_SDK_CAPABILITIES: frozenset[str] = frozenset(
    {"chat", "chat_stream", "models", "embeddings"}
)

# Default model if none is configured. mistral-large-latest is not offered to
# every plan (it is absent from /v1/models for some keys); Small 4 is the
# general-purpose default Mistral documents, with reasoning and vision.
DEFAULT_MODEL = "mistral-small-latest"

# Fallback context lengths keyed by model-id prefix (longest prefix wins).
# These are used when the model card registry has no entry and the /v1/models
# endpoint was not queried yet.
DEFAULT_MISTRAL_TOKEN_LIMITS: dict[str, int] = {
    # From the live /v1/models listing (2026-11-16).
    "codestral-2508": 256000,
    "codestral-embed": 8192,
    "codestral-embed-2505": 8192,
    "codestral-latest": 256000,
    "labs-leanstral-1-5": 262144,
    "labs-leanstral-1-5-1": 262144,
    "magistral-medium-latest": 262144,
    "magistral-small-latest": 262144,
    "ministral-14b-2512": 262144,
    "ministral-14b-latest": 262144,
    "ministral-3b-2512": 131072,
    "ministral-3b-latest": 131072,
    "ministral-8b-2512": 262144,
    "ministral-8b-latest": 262144,
    "mistral-code-fim-latest": 256000,
    "mistral-code-latest": 256000,
    "mistral-embed": 8192,
    "mistral-embed-2312": 8192,
    "mistral-medium": 262144,
    "mistral-medium-2604": 262144,
    "mistral-medium-3": 262144,
    "mistral-medium-3-5": 262144,
    "mistral-medium-3.5": 262144,
    "mistral-medium-latest": 262144,
    "mistral-moderation-2603": 131071,
    "mistral-ocr-2512": 16384,
    "mistral-ocr-3": 16384,
    "mistral-ocr-3-0": 16384,
    "mistral-ocr-4": 16384,
    "mistral-ocr-4-1": 16384,
    "mistral-ocr-latest": 16384,
    "mistral-small-2603": 262144,
    "mistral-small-latest": 262144,
    "mistral-vibe-cli-fast": 262144,
    "mistral-vibe-cli-latest": 262144,
    "mistral-vibe-cli-with-tools": 262144,
    "voxtral-mini-2602": 16384,
    "voxtral-mini-latest": 16384,
    "voxtral-mini-realtime-2602": 32768,
    "voxtral-mini-realtime-latest": 32768,
    "voxtral-mini-transcribe-3": 65536,
    "voxtral-mini-transcribe-3-0": 65536,
    "voxtral-mini-transcribe-realtime-2602": 32768,
    "voxtral-mini-transcribe-realtime-3": 65536,
    "voxtral-mini-transcribe-realtime-3-0": 65536,
    "voxtral-mini-tts-2603": 4096,
    "voxtral-mini-tts-3": 4096,
    "voxtral-mini-tts-3-0": 4096,
    "voxtral-mini-tts-latest": 4096,
    "voxtral-small-2507": 32768,
    "voxtral-small-latest": 32768,
    "voxtral-small-transcribe-3": 65536,
    "voxtral-small-transcribe-3-0": 65536,
}

# Prefix-based heuristic for future model snapshots not yet in the map.
_MISTRAL_PREFIX_CONTEXT_HEURISTICS: list[tuple[str, int]] = [
    ("mistral-large", 262144),
    ("mistral-medium", 262144),
    ("mistral-small", 262144),
    ("mistral-moderation", 8192),
    ("mistral-ocr", 0),
    ("mistral-embed", 8192),
    ("ministral-3", 262144),
    ("ministral", 131072),
    ("magistral-medium", 262144),
    ("magistral-small", 262144),
    ("magistral", 262144),
    ("codestral-embed", 32768),
    ("codestral", 262144),
    ("devstral", 262144),
    ("pixtral", 131072),
    ("voxtral", 32768),
    ("open-mistral", 131072),
    ("labs-mistral", 131072),
]

#: ``reasoning_effort`` values each reasoning family accepts, longest prefix
#: first. From docs.mistral.ai/capabilities/reasoning (2026-11-16): Small 4 and
#: Medium 3.5 take only "none"/"high"; Large 4 refuses "none"; the hosted GLM
#: models never turn thinking off. The magistral-* ids are aliases of Small
#: and Medium now.
_REASONING_EFFORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mistral-large-4", ("low", "medium", "high")),
    ("zai-glm", ("low", "high", "max")),
    ("mistral-small", ("none", "high")),
    ("mistral-medium", ("none", "high")),
    ("magistral", ("none", "high")),
    ("mistral-vibe-cli", ("none", "high")),
    ("labs-leanstral", ("none", "high")),
)

#: "min_tokens support is not enabled for this model", "top_k sampling is
#: not enabled ...", "reasoning_effort is not enabled ...": a parameter this
#: model does not take (code 3051, all three measured live 2026-11-16).
_RX_NOT_ENABLED = re.compile(
    r"\b([a-z_]+)(?: support| sampling)? is not enabled for this model", re.I
)


def _reasoning_efforts(model: str) -> tuple[str, ...] | None:
    """Accepted ``reasoning_effort`` values for *model*, or None if unknown."""
    for prefix, efforts in _REASONING_EFFORTS:
        if model.startswith(prefix):
            return efforts
    return None


def _is_mistral_reasoning_model(model: str) -> bool:
    """Whether *model* is known to accept ``reasoning_effort``."""
    return _reasoning_efforts(model) is not None


def _chunk_texts(chunks: Any) -> tuple[str, str]:
    """(answer text, thinking text) from a content list of Mistral chunks."""
    text: list[str] = []
    thinking: list[str] = []
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        kind = chunk.get("type")
        if kind == "text":
            text.append(chunk.get("text") or "")
        elif kind == "thinking":
            for inner in chunk.get("thinking") or []:
                if isinstance(inner, dict) and inner.get("type", "text") == "text":
                    thinking.append(inner.get("text") or "")
    return "".join(text), "".join(thinking)


def _normalize_message_content(message: dict[str, Any]) -> None:
    """Flatten a chunked ``content`` in place, keeping the chunks for replay."""
    content = message.get("content")
    if not isinstance(content, list):
        return
    text, thinking = _chunk_texts(content)
    if any(isinstance(c, dict) and c.get("type") == "thinking" for c in content):
        message["mistral_content"] = content
    message["content"] = text
    if thinking:
        message["reasoning_content"] = thinking


class MistralProvider(LegacyMediaBridge, BaseProvider):
    """LLMCore provider for the native Mistral AI API.

    Uses ``httpx.AsyncClient`` directly.  No ``mistralai`` SDK dependency.

    Multimodal via Message.metadata conventions (same as OpenAI provider):
    - metadata["content_parts"]: raw content part list (overrides content)
    - metadata["inline_images"]: list of image dicts → image_url parts
    - metadata["inline_audio"]: list of audio dicts → input_audio parts
    - metadata["tool_calls"]: assistant tool calls for multi-turn
    - metadata["name"]: participant name
    """

    _client: Any  # httpx.AsyncClient
    _encoding: Any | None = None
    _discovered_context_lengths: dict[str, int]

    #: Parameters a model answered "not enabled for this model" to, per model.
    #: Learned once per process; the request is retried without them.
    _REJECTED_PARAMS: dict[str, set[str]] = {}

    #: Served through llm.media by the legacy-method bridge (_media_bridge).
    _MEDIA_CAPABILITIES: frozenset[str] = frozenset({"tts", "asr", "ocr"})

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads)
        if not httpx_available:
            raise ImportError("httpx library not installed. Install with: pip install httpx")

        # API key resolution
        api_key = config.get("api_key")
        if not api_key:
            env_var = config.get("api_key_env_var", "MISTRAL_API_KEY")
            api_key = os.environ.get(env_var)
        if not api_key:
            api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ConfigError(
                "Mistral API key not found. Set MISTRAL_API_KEY or configure api_key."
            )

        self.api_key = api_key
        self.base_url = config.get("base_url", DEFAULT_MISTRAL_BASE_URL).rstrip("/")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        #: Speech model used when a call names none. The API now requires a
        #: model (422 "No model provided for speech" without one).
        #: voxtral-mini-tts-3 takes preset voices and ref_audio; -latest (2603)
        #: takes ref_audio only (measured 2026-11-16).
        self.tts_model: str = str(config.get("tts_model") or "voxtral-mini-tts-3")
        self.timeout = float(config.get("timeout", 120.0))
        self.max_retries = int(config.get("max_retries", 2))
        self._discovered_context_lengths = {}
        #: model id (and every alias) -> its /v1/models capabilities
        self._discovered_capabilities: dict[str, dict[str, Any]] = {}
        responses_api = str(config.get("responses_api", "never")).lower()
        if responses_api not in ("never", "always"):
            raise ConfigError(
                f"Mistral responses_api must be 'never' or 'always', not {responses_api!r}."
            )
        self.responses_api = responses_api

        # --- Transport: direct REST first, official SDK as the fallback ------
        self._backend = self._resolve_backend(config.get("backend"))
        self._sdk: Any = None
        if self._backend == "sdk":
            # The SDK appends its own version prefix, so handing it llmcore's
            # base_url verbatim produces /v1/v1/... and a "no Route match" 404.
            # Strip the suffix, and only override the server when the base URL
            # has actually been customised.
            sdk_server = self.base_url
            if sdk_server.endswith("/v1"):
                sdk_server = sdk_server[: -len("/v1")]
            sdk_kwargs: dict[str, Any] = {"api_key": api_key}
            if sdk_server and sdk_server != DEFAULT_MISTRAL_BASE_URL.removesuffix("/v1"):
                sdk_kwargs["server_url"] = sdk_server
            self._sdk = _MistralSDK(**sdk_kwargs)

        # Initialise httpx client
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
        )
        logger.debug(
            "MistralProvider initialized: base_url=%s, default_model=%s",
            self.base_url,
            self.default_model,
        )

        # Load tokenizer for token counting
        self._load_tokenizer(self.default_model)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport, preferring direct REST.

        Mistral publishes ``mistralai`` v3, so llmcore's house rule applies:
        call the API directly by default and fall back to the vendor SDK when
        asked. Direct stays the default because this provider's wide surface
        (OCR, audio, classification, moderation, FIM) is OpenAI-shaped over
        REST, and llmcore's extractors already speak that shape.
        """
        available = {"httpx": httpx_available, "sdk": mistralai_available}
        req = (requested or "auto").lower()
        if req not in ("auto", "httpx", "sdk"):
            logger.warning("Unknown Mistral backend '%s'; using auto-detection.", req)
            req = "auto"
        if req != "auto" and available.get(req):
            return req
        if req != "auto":
            logger.warning(
                "Requested Mistral backend '%s' is unavailable; falling back. "
                "Install it with: pip install llmcore[mistral]",
                req,
            )
        for backend in ("httpx", "sdk"):
            if available[backend]:
                return backend
        raise ConfigError("No usable Mistral transport is installed.")

    def _use_sdk_for(self, capability: str) -> bool:
        """Whether *capability* should go through the SDK.

        Only the calls in :data:`_SDK_CAPABILITIES` do. Anything else stays on
        direct REST even when ``backend = "sdk"``, and logs that once, because a
        backend that silently does nothing for half its methods is worse than
        one that is explicit about its coverage.
        """
        if self._backend != "sdk":
            return False
        if capability in _SDK_CAPABILITIES:
            return True
        logger.debug(
            "Mistral '%s' has no drop-in SDK equivalent; using direct REST for it "
            "even though backend='sdk'.",
            capability,
        )
        return False

    @staticmethod
    def _sdk_dump(obj: Any) -> dict[str, Any]:
        """Normalize an SDK model into the dict shape the extractors expect."""
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            return dict(dump(mode="json", exclude_none=True))
        return dict(obj) if isinstance(obj, dict) else {}

    def _load_tokenizer(self, model_name: str) -> None:
        """Load tiktoken encoding for Mistral models.

        Mistral uses Tekken / SentencePiece tokenizers natively, but tiktoken's
        ``cl100k_base`` provides a reasonable approximation for token counting.
        """
        if not tiktoken_available:
            self._encoding = None
            return
        try:
            # Mistral models are not in tiktoken's model map, so use fallback.
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("Failed to load tiktoken encoding: %s", e)
            self._encoding = None

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return self._provider_instance_name or "mistral"

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover models via GET /v1/models."""
        try:
            if self._use_sdk_for("models"):
                # Feed the SAME `data` the REST path produces, so the parsing
                # below is shared rather than duplicated per transport.
                listing = await self._sdk.models.list_async()
                data = self._sdk_dump(listing)
            else:
                resp = await self._client.get("/models")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.get_name(), f"Models list error ({e.response.status_code}): {e}"
            )
        except Exception as e:
            raise ProviderError(self.get_name(), f"Models list error: {e}")

        models_list = data.get("data", [])
        provider_name = self.get_name()
        result: list[ModelDetails] = []

        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        for m in models_list:
            mid = m.get("id", "")
            caps = m.get("capabilities", {})
            ctx_len = m.get("max_context_length", 32768)

            # Cache discovered context length and capabilities, aliases too.
            for name in [mid, *(m.get("aliases") or [])]:
                self._discovered_context_lengths.setdefault(name, ctx_len)
                self._discovered_capabilities.setdefault(name, caps)

            supports_tools = caps.get("function_calling", False)
            supports_vision = caps.get("vision", False)
            supports_audio = caps.get("audio", False) or caps.get("audio_transcription", False)

            # Enrich from card registry
            if registry:
                card = registry.get(provider_name, mid)
                if card and card.capabilities:
                    supports_tools = supports_tools or (
                        card.capabilities.tool_use or card.capabilities.function_calling
                    )
                    supports_vision = supports_vision or card.capabilities.vision

            result.append(
                ModelDetails(
                    id=mid,
                    context_length=ctx_len,
                    supports_streaming=caps.get("completion_chat", True),
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    provider_name=provider_name,
                    supports_reasoning=bool(caps.get("reasoning")),
                    metadata={
                        "owned_by": m.get("owned_by", "mistralai"),
                        "created": m.get("created"),
                        "type": m.get("type", "base"),
                        "capabilities": caps,
                        "supports_audio": supports_audio,
                        "aliases": m.get("aliases", []),
                        "description": m.get("description"),
                        "default_model_temperature": m.get("default_model_temperature"),
                        "deprecation": m.get("deprecation"),
                    },
                )
            )
        return result

    def _supports_reasoning(self, model: str) -> bool:
        caps = self._discovered_capabilities.get(model)
        if caps is not None:
            return bool(caps.get("reasoning"))
        return _is_mistral_reasoning_model(model)

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        model_name = model or self.default_model
        is_reasoning = self._supports_reasoning(model_name)

        params: dict[str, Any] = {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.5},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_tokens": {"type": "integer", "minimum": 1},
            "stop": {"type": "array", "items": {"type": "string"}},
            "random_seed": {"type": "integer"},
            "n": {"type": "integer", "minimum": 1},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "response_format": {"type": "object"},
            "safe_prompt": {"type": "boolean"},
            "parallel_tool_calls": {"type": "boolean"},
            "prediction": {"type": "object"},
            "metadata": {"type": "object"},
            # Added to the API in 2026 (mistralai 3.7). Not every model takes
            # every one: a "not enabled for this model" 400 drops the
            # parameter, retries, and is remembered for that model.
            "top_k": {"type": "integer", "minimum": 1},
            "min_tokens": {"type": "integer", "minimum": 0},
            "repetition_penalty": {"type": "number"},
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer", "minimum": 0},
            "prompt_logprobs": {"type": "boolean"},
            "top_prompt_logprobs": {"type": "integer", "minimum": 0},
            "prompt_cache_key": {"type": "string"},
            "service_tier": {"type": "string", "enum": ["auto", "standard_only"]},
            # llmcore's cross-provider reasoning knob; mapped per model below.
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            },
            # Sent verbatim. Accepted values depend on the model.
            "reasoning_effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            },
        }
        if is_reasoning:
            params["prompt_mode"] = {
                "type": "string",
                "enum": ["reasoning", "default"],
            }
        params["guardrails"] = {"type": "array", "items": {"type": "object"}}
        return params

    def get_max_context_length(self, model: str | None = None) -> int:
        model_name = model or self.default_model

        # 1. Check discovered context lengths (from /v1/models)
        if model_name in self._discovered_context_lengths:
            return self._discovered_context_lengths[model_name]

        # 2. Check model card registry
        try:
            registry = get_model_card_registry()
            card = registry.get(self.get_name(), model_name)
            if card and card.context and card.context.max_input_tokens:
                return card.context.max_input_tokens
        except Exception:
            pass

        # 3. Check static table
        limit = DEFAULT_MISTRAL_TOKEN_LIMITS.get(model_name)
        if limit is not None:
            return limit

        # 4. Prefix heuristic
        for prefix, ctx in _MISTRAL_PREFIX_CONTEXT_HEURISTICS:
            if model_name.startswith(prefix):
                return ctx

        # 5. Default fallback
        return 32768

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_message_payload(self, msg: Message, model_name: str) -> dict[str, Any]:
        """Convert an llmcore Message to a Mistral API message dict.

        Mistral's chat API follows the OpenAI message format with roles:
        system, user, assistant, tool.
        """
        role_str = msg.role if isinstance(msg.role, str) else msg.role.value
        metadata = msg.metadata or {}
        content: Any = msg.content

        # Handle multimodal content parts (vision, audio)
        if "content_parts" in metadata:
            content = metadata["content_parts"]
        elif "inline_images" in metadata or "inline_audio" in metadata:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"type": "text", "text": msg.content})
            for img in metadata.get("inline_images", []):
                if isinstance(img, str):
                    parts.append({"type": "image_url", "image_url": {"url": img}})
                elif isinstance(img, dict):
                    img_p: dict[str, Any] = {"url": img.get("url", "")}
                    if "detail" in img:
                        img_p["detail"] = img["detail"]
                    parts.append({"type": "image_url", "image_url": img_p})
            for aud in metadata.get("inline_audio", []):
                if isinstance(aud, dict):
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": aud.get("data", ""),
                                "format": aud.get("format", "wav"),
                            },
                        }
                    )
            content = parts

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        # Tool result messages need tool_call_id
        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
            # Mistral expects name for tool messages too
            if "name" in metadata:
                msg_dict["name"] = metadata["name"]

        # Assistant tool calls for multi-turn. First-class Message.tool_calls
        # (R-2) takes precedence over the legacy metadata["tool_calls"] channel
        # so a native tool-role result has its preceding assistant tool_calls.
        if role_str == "assistant":
            # A reasoning turn goes back exactly as it came: thinking chunk
            # included ("do not strip ThinkChunk", docs.mistral.ai).
            if metadata.get("mistral_content"):
                msg_dict["content"] = list(metadata["mistral_content"])
            tool_calls = getattr(msg, "tool_calls", None) or metadata.get("tool_calls")
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
                if not msg_dict["content"]:
                    msg_dict["content"] = ""

        name = metadata.get("name")
        if name and role_str != "tool":
            msg_dict["name"] = name

        # Prefix for Mistral's prefix-mode
        if metadata.get("prefix"):
            msg_dict["prefix"] = True

        return msg_dict

    # ------------------------------------------------------------------
    # Chat Completion
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        model_name = model or self.default_model
        supported = self.get_supported_parameters(model_name)
        for key in kwargs:
            if key not in supported:
                raise ValueError(f"Unsupported parameter '{key}' for Mistral provider.")

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")
        if not context:
            raise ProviderError(self.get_name(), "No valid messages.")

        params = dict(kwargs)
        self._apply_effort(params, model_name)

        if self.responses_api == "always":
            return await self._responses_completion(
                context, model_name, stream=stream, tools=tools, tool_choice=tool_choice, params=params
            )

        messages_payload = [self._build_message_payload(msg, model_name) for msg in context]

        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages_payload,
            "stream": stream,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        if tool_choice:
            body["tool_choice"] = tool_choice

        # Mistral-specific params (random_seed, safe_prompt, ...) go in as-is,
        # minus whatever this model has already refused.
        body.update(params)
        for name in self._REJECTED_PARAMS.get(model_name, ()):
            body.pop(name, None)

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM REQUEST (%s): %s",
                self.get_name(),
                json.dumps(body, indent=2, default=str),
            )

        while True:
            try:
                return await self._chat_once(body, model_name, stream=stream)
            except ProviderError as e:
                if not self._learn_rejection(e, body, model_name):
                    raise

    async def _chat_once(
        self, body: dict[str, Any], model_name: str, *, stream: bool
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """One request on the configured transport; a stream is primed, so an
        HTTP error surfaces here rather than on the caller's first iteration."""
        if self._use_sdk_for("chat_stream" if stream else "chat"):
            result = await self._chat_via_sdk(body, model_name, stream=stream)
        else:
            try:
                if stream:
                    result = await self._stream_completion(body, model_name)
                else:
                    resp = await self._client.post("/chat/completions", json=body)
                    self._raise_for_status(resp, model_name)
                    result = resp.json()
                    if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RAW LLM RESPONSE (%s): %s",
                            self.get_name(),
                            json.dumps(result, indent=2, default=str),
                        )
            except ProviderError:
                raise
            except httpx.TimeoutException as e:
                raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
            except httpx.HTTPError as e:
                raise ProviderError(
                    self.get_name(), f"Connection error: {e}", model_name=model_name
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", e, exc_info=True)
                raise ProviderError(self.get_name(), f"Unexpected error: {e}") from e
        if stream:
            return await self._primed(self._normalize_stream(result))
        for choice in result.get("choices") or []:
            if isinstance(choice.get("message"), dict):
                _normalize_message_content(choice["message"])
        return result

    def _apply_effort(self, params: dict[str, Any], model_name: str) -> None:
        """Map llmcore's ``effort`` onto this model's ``reasoning_effort``.

        Each family takes its own values (Small 4: none/high; Large 4:
        low/medium/high; GLM: low/high/max), so the nearest accepted value is
        sent. A model without reasoning ignores ``effort``; an explicit
        ``reasoning_effort`` is sent as given.
        """
        effort = params.pop("effort", None)
        if effort is None or "reasoning_effort" in params:
            return
        if not self._supports_reasoning(model_name):
            logger.debug("Mistral model %s has no reasoning; effort=%r ignored.", model_name, effort)
            return
        requested = str(effort).strip().lower()
        accepted = _reasoning_efforts(model_name) or ("none", "high")
        value = requested if requested in accepted else nearest_effort(requested, accepted)
        if value:
            params["reasoning_effort"] = value

    def _learn_rejection(self, error: ProviderError, body: dict[str, Any], model_name: str) -> bool:
        """Drop a parameter the model said it does not take; True to retry."""
        match = _RX_NOT_ENABLED.search(str(error))
        if not match:
            return False
        name = match.group(1).lower()
        if name in ("model", "messages") or name not in body:
            return False
        body.pop(name)
        self._REJECTED_PARAMS.setdefault(model_name, set()).add(name)
        logger.warning(
            "Mistral model %s does not take '%s'; retrying without it (remembered).",
            model_name,
            name,
        )
        return True

    @staticmethod
    async def _primed(
        stream: AsyncGenerator[dict[str, Any], None],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Pull the first chunk now, so connection and HTTP errors raise from
        ``chat_completion`` (where they can be retried) instead of mid-iteration."""
        try:
            first: dict[str, Any] | None = await stream.__anext__()
        except StopAsyncIteration:
            first = None

        async def _gen() -> AsyncGenerator[dict[str, Any], None]:
            if first is not None:
                yield first
            async for chunk in stream:
                yield chunk

        return _gen()

    @staticmethod
    async def _normalize_stream(
        stream: AsyncGenerator[dict[str, Any], None],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Flatten chunked deltas and attach the turn's chunks at the end.

        A reasoning stream's ``delta.content`` is a list (thinking, then the
        transition with the first text) and later a plain string. Each delta
        becomes text ``content`` plus ``reasoning_content``; the chunk with the
        finish reason carries ``mistral_content``, the whole turn to replay.
        """
        thinking: list[str] = []
        text: list[str] = []
        signature: str | None = None
        saw_thinking = False
        async for chunk in stream:
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta")
                if not isinstance(delta, dict) or choice.get("index", 0) != 0:
                    continue
                content = delta.get("content")
                if isinstance(content, list):
                    piece, thought = _chunk_texts(content)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "thinking":
                            saw_thinking = True
                            signature = part.get("signature") or signature
                    delta["content"] = piece
                    if thought:
                        delta["reasoning_content"] = thought
                    thinking.append(thought)
                    text.append(piece)
                elif isinstance(content, str):
                    text.append(content)
                if choice.get("finish_reason") and saw_thinking:
                    block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "".join(thinking)}],
                    }
                    if signature:
                        block["signature"] = signature
                    delta["mistral_content"] = [block, {"type": "text", "text": "".join(text)}]
            yield chunk

    async def _chat_via_sdk(
        self, body: dict[str, Any], model_name: str, *, stream: bool
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Run a chat completion through ``mistralai``.

        Returns the same shapes the direct path returns — a response dict, or an
        async generator of chunk dicts — because every ``extract_*`` method
        downstream parses those dicts rather than SDK objects.
        """
        payload = {k: v for k, v in body.items() if k not in ("model", "messages", "stream")}
        try:
            if stream:
                return await self._stream_via_sdk(body, model_name, payload)
            result = await self._sdk.chat.complete_async(
                model=body.get("model") or model_name,
                messages=body.get("messages"),
                **payload,
            )
        except Exception as e:
            raise self._sdk_error(e, model_name) from e
        return self._sdk_dump(result)

    async def _stream_via_sdk(
        self, body: dict[str, Any], model_name: str, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Return an async generator of chunk dicts from the SDK.

        Deliberately an ``async def`` that *returns* a generator, the shape
        :meth:`_stream_completion` has, so both transports are awaited the same
        way by :meth:`_chat_once`.
        """

        async def _generator() -> AsyncGenerator[dict[str, Any], None]:
            try:
                events = await self._sdk.chat.stream_async(
                    model=body.get("model") or model_name,
                    messages=body.get("messages"),
                    **payload,
                )
                async for event in events:
                    # The SDK wraps each chunk in an envelope carrying `.data`;
                    # unwrapping it keeps the chunk shape identical to the REST
                    # path's, so extract_delta_content needs no transport branch.
                    chunk = getattr(event, "data", event)
                    yield self._sdk_dump(chunk)
            except ProviderError:
                raise
            except Exception as e:
                raise self._sdk_error(e, model_name) from e

        return _generator()

    def _sdk_error(self, error: Exception, model_name: str) -> ProviderError:
        """An SDK exception mapped exactly like a REST error.

        ``mistralai`` errors keep the HTTP response; running it through
        :meth:`_raise_for_status` gives the same status codes, retryability,
        zero-quota and context-overflow handling as the direct transport.
        """
        response = getattr(error, "raw_response", None)
        if response is not None and getattr(response, "status_code", 0) >= 400:
            try:
                self._raise_for_status(response, model_name)
            except ProviderError as mapped:
                return mapped
        return ProviderError(self.get_name(), f"Mistral SDK error: {error}", model_name=model_name)

    async def _stream_completion(
        self, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Internal streaming helper for chat completions."""

        async def _generator() -> AsyncGenerator[dict[str, Any], None]:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=body,
            ) as resp:
                await self._raise_for_status_stream(resp, model_name)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse SSE chunk: %s", data_str[:200])
                        continue
                    if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RAW STREAM CHUNK (%s): %s",
                            self.get_name(),
                            json.dumps(chunk, default=str),
                        )
                    yield chunk

        return _generator()

    # ------------------------------------------------------------------
    # Responses API (stateless /v1/responses)
    # ------------------------------------------------------------------

    async def _responses_completion(
        self,
        context: list[Message],
        model_name: str,
        *,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: Any,
        params: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat through ``/v1/responses``, normalised to the Chat Completions shape.

        Mistral's Responses API is stateless (``store`` must be false and there
        is no ``previous_response_id``), so the whole history goes in ``input``
        each time, reasoning items included via ``metadata["reasoning_items"]``.
        Chat-only parameters (``safe_prompt``, ``random_seed``, ``prediction``,
        ``guardrails``) are not part of it.
        """
        for name in ("safe_prompt", "random_seed", "prediction", "guardrails", "prompt_mode", "n"):
            if name in params:
                raise ValueError(f"'{name}' is not available on Mistral's Responses API.")
        body = rsp.build_responses_request(
            model_name, context, stream=stream, tools=tools, tool_choice=tool_choice, **params
        )
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug("RAW RESPONSES REQUEST (%s): %s", self.get_name(), json.dumps(body, default=str))
        if stream:
            return await self._primed(self._responses_stream(body, model_name))
        try:
            resp = await self._client.post("/responses", json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}", model_name=model_name) from e
        self._raise_for_status(resp, model_name)
        data = resp.json()
        if data.get("error"):
            raise self._responses_failed(rsp.ResponsesFailed(data["error"]), model_name)
        return rsp.normalize_response(data)

    async def _responses_stream(
        self, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalizer = rsp.ResponsesStreamNormalizer()
        try:
            async with self._client.stream("POST", "/responses", json=body) as resp:
                await self._raise_for_status_stream(resp, model_name)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # `event:` lines repeat the payload's own `type`
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        logger.warning("Skipping unparseable Responses event from Mistral.")
                        continue
                    try:
                        chunks = normalizer.feed(event)
                    except rsp.ResponsesFailed as exc:
                        raise self._responses_failed(exc, model_name) from None
                    for chunk in chunks:
                        yield chunk
        except ProviderError:
            raise
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}", model_name=model_name) from e

    def _responses_failed(self, exc: Any, model_name: str) -> ProviderError:
        error = getattr(exc, "error", None) or {}
        message = str(error.get("message") or exc)
        if self._is_context_overflow(message):
            return ContextLengthError(
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=message,
            )
        return ProviderError(self.get_name(), f"Response failed: {message}", model_name=model_name)

    # ------------------------------------------------------------------
    # Fill-in-the-Middle (FIM)
    # ------------------------------------------------------------------

    async def fim_completion(
        self,
        prompt: str,
        suffix: str = "",
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        random_seed: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Fill-in-the-Middle completion for code (Codestral).

        Args:
            prompt: Code before the cursor.
            suffix: Code after the cursor.
            model: FIM model (default: ``codestral-latest``).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            random_seed: Deterministic sampling seed.
            stream: Whether to stream the response.
            **kwargs: Additional API parameters.

        Returns:
            API response dict or async generator of stream chunks.
        """
        model_name = model or "codestral-latest"

        body: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "suffix": suffix,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop is not None:
            body["stop"] = stop
        if random_seed is not None:
            body["random_seed"] = random_seed
        body.update(kwargs)

        try:
            if stream:
                return await self._stream_fim(body, model_name)
            else:
                resp = await self._client.post("/fim/completions", json=body)
                self._raise_for_status(resp, model_name)
                return resp.json()
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"FIM error: {e}")

    async def _stream_fim(
        self, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming helper for FIM completions."""

        async def _generator() -> AsyncGenerator[dict[str, Any], None]:
            async with self._client.stream("POST", "/fim/completions", json=body) as resp:
                await self._raise_for_status_stream(resp, model_name)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        pass

        return _generator()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        output_dimension: int | None = None,
        output_dtype: str | None = None,
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create embeddings via POST /v1/embeddings.

        Args:
            input_texts: Single string or list of strings to embed.
            model: Embedding model (default: ``mistral-embed``).
            output_dimension: Desired output dimensionality (if model supports).
            output_dtype: Output dtype (float, int8, uint8, binary, ubinary).
            encoding_format: Response encoding (float, base64).
            **kwargs: Extra API parameters.

        Returns:
            Raw API response dict with ``data``, ``model``, ``usage``.
        """
        embed_model = model or "mistral-embed"
        body: dict[str, Any] = {
            "model": embed_model,
            "input": input_texts if isinstance(input_texts, list) else [input_texts],
            "encoding_format": encoding_format,
        }
        if output_dimension is not None:
            body["output_dimension"] = output_dimension
        if output_dtype is not None:
            body["output_dtype"] = output_dtype
        body.update(kwargs)

        try:
            if self._use_sdk_for("embeddings"):
                options = {
                    k: body[k]
                    for k in ("output_dimension", "output_dtype", "encoding_format", "metadata")
                    if k in body
                }
                result = await self._sdk.embeddings.create_async(
                    model=body.get("model"), inputs=body.get("input"), **options
                )
                return self._sdk_dump(result)
            resp = await self._client.post("/embeddings", json=body)
            self._raise_for_status(resp, embed_model)
            return resp.json()
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"Embeddings error: {e}")

    # ------------------------------------------------------------------
    # Response extraction helpers
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            content = choices[0].get("message", {}).get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""
        return _chunk_texts(content)[0] if isinstance(content, list) else content or ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            content = choices[0].get("delta", {}).get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""
        return _chunk_texts(content)[0] if isinstance(content, list) else content or ""

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """The model's thinking (reasoning models with ``reasoning_effort`` on)."""
        try:
            message = (response.get("choices") or [{}])[0].get("message") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        if isinstance(message.get("content"), list):
            return _chunk_texts(message["content"])[1] or None
        return message.get("reasoning_content") or message.get("reasoning") or None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """Streaming counterpart of :meth:`extract_reasoning_content`."""
        try:
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        if isinstance(delta.get("content"), list):
            return _chunk_texts(delta["content"])[1] or None
        return delta.get("reasoning_content") or delta.get("reasoning") or None

    def assistant_replay_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """What the next assistant turn needs to carry: the thinking chunks
        (chat) or the reasoning items (``responses_api = "always"``). For a
        stream, pass the last chunk."""
        try:
            choice = (response.get("choices") or [{}])[0]
            source = choice.get("message") or choice.get("delta") or {}
        except (AttributeError, IndexError, TypeError):
            return {}
        if source.get("mistral_content"):
            return {"mistral_content": list(source["mistral_content"])}
        if source.get("reasoning_items"):
            return {"reasoning_items": list(source["reasoning_items"])}
        return {}

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a Mistral response, normalized to ToolCall objects."""
        out: list[ToolCall] = []
        try:
            choices = response.get("choices", [])
            if not choices:
                return out
            raw_calls = choices[0].get("message", {}).get("tool_calls")
            if not raw_calls:
                return out
            for tc in raw_calls:
                tc_type = tc.get("type", "function")
                if tc_type == "function":
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {"_raw": args_str}
                    out.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=args_dict,
                        )
                    )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to extract tool calls: %s", e)
        return out

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract extended usage information from a Mistral response."""
        usage = response.get("usage", {})
        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"]:
            result["prompt_tokens_details"] = usage["prompt_tokens_details"]
        # Audio seconds (for transcription billing)
        if usage.get("prompt_audio_seconds"):
            result["prompt_audio_seconds"] = usage["prompt_audio_seconds"]
        return result

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        if not text:
            return 0
        if self._encoding:
            return len(self._encoding.encode(text))
        return _EstimateCounter().count(text)

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        total = 0
        for msg in messages:
            # Per-message overhead: ~4 tokens for role/formatting
            total += 4
            if msg.content:
                total += await self.count_tokens(msg.content, model)
            # Account for tool_calls in metadata
            if msg.metadata and "tool_calls" in msg.metadata:
                total += await self.count_tokens(
                    json.dumps(msg.metadata["tool_calls"], default=str), model
                )
        total += 2  # reply priming
        return total

    # ------------------------------------------------------------------
    # Multimodal: Text-to-Speech (TTS)
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate speech via POST /v1/audio/speech.

        Mistral TTS needs a voice: a ``voice`` (``voice_id``: a preset or saved
        voice, listed by ``GET /v1/audio/voices``) or ``ref_audio`` (base64
        audio for zero-shot cloning). Without either it answers 400.
        ``voxtral-mini-tts-latest`` (2603, "speech v1") takes ``ref_audio``
        only and rejects preset voices; ``voxtral-mini-tts-3`` takes both
        (measured 2026-11-16). Newer
        options pass through as keyword arguments: ``expressiveness``,
        ``phoneme_hints``, ``conversation_history``, ``prompt_cache_key``.

        Args:
            text: Text to convert to speech.
            voice: Voice id (Mistral ``voice_id``).
            model: TTS model (e.g. ``voxtral-mini-tts-latest``).
            response_format: Output format (mp3, wav, pcm, flac, opus).
            speed: Not directly supported by Mistral; ignored.
            instructions: Not supported by Mistral; ignored.
            **kwargs: ``ref_audio`` for cloning, plus any other body field.

        Returns:
            ``SpeechResult`` with raw audio bytes.
        """
        from ..models_multimodal import SpeechResult

        tts_model = model or kwargs.pop("tts_model", None) or self.tts_model

        body: dict[str, Any] = {
            "input": text,
            "response_format": response_format,
        }
        if tts_model:
            body["model"] = tts_model
        if voice:
            body["voice_id"] = voice

        # Support ref_audio for zero-shot voice cloning
        ref_audio = kwargs.pop("ref_audio", None)
        if ref_audio:
            body["ref_audio"] = ref_audio
        if not voice and not ref_audio:
            raise ProviderError(
                self.get_name(),
                "Mistral TTS needs voice= (a voice id from GET /v1/audio/voices) or "
                "ref_audio= (base64 audio to clone).",
            )
        # expressiveness, phoneme_hints, conversation_history, metadata, ...
        body.update(kwargs)

        logger.debug(
            "Mistral TTS: model=%s, voice=%s, format=%s, len=%d",
            tts_model,
            voice,
            response_format,
            len(text),
        )

        try:
            resp = await self._client.post("/audio/speech", json=body)
            self._raise_for_status(resp, tts_model or "tts")
            response_data = resp.json()

            audio_b64 = response_data.get("audio_data", "")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

            return SpeechResult(
                audio_data=audio_bytes,
                format=response_format,
                model=tts_model or "mistral-tts",
                voice=voice or "ref_audio",  # a cloned voice has no id
                metadata={},
            )
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"TTS error: {e}")

    # ------------------------------------------------------------------
    # Multimodal: Speech-to-Text (STT)
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_data: bytes | str,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Transcribe audio via POST /v1/audio/transcriptions.

        Mistral STT adds:
        - ``diarize``: Speaker diarization (bool)
        - ``context_bias``: List of bias words for improved accuracy
        - ``file_url``: URL to audio file (alternative to file upload)
        - ``file_id``: ID of pre-uploaded file

        Args:
            audio_data: Raw audio bytes or file path string.
            model: STT model (default: ``voxtral-mini-latest``).
            language: Input language hint (ISO-639-1, e.g. ``"en"``).
            prompt: Not used by Mistral; ignored.
            response_format: Output format (json).
            temperature: Sampling temperature.
            timestamp_granularities: ``["word"]`` and/or ``["segment"]``.
            **kwargs: Extra params (``diarize``, ``context_bias``, ``file_url``).

        Returns:
            ``TranscriptionResult`` with transcribed text and segments.
        """
        from ..models_multimodal import TranscriptionResult, TranscriptionSegment

        stt_model = model or "voxtral-mini-latest"
        diarize = kwargs.pop("diarize", False)
        context_bias = kwargs.pop("context_bias", None)
        file_url = kwargs.pop("file_url", None)
        file_id = kwargs.pop("file_id", None)

        # Build multipart form data
        # Mistral's transcription API uses multipart/form-data
        form_data: dict[str, Any] = {"model": stt_model}
        files_param = None

        if file_url:
            form_data["file_url"] = file_url
        elif file_id:
            form_data["file_id"] = file_id
        elif isinstance(audio_data, str):
            # File path. Read it whole: the handle used to stay open whenever
            # the request failed.
            import mimetypes
            import pathlib

            file_path = pathlib.Path(audio_data)
            mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            files_param = {"file": (file_path.name, file_path.read_bytes(), mime)}
        elif isinstance(audio_data, bytes):
            files_param = {"file": ("audio.wav", io.BytesIO(audio_data), "audio/wav")}

        if language:
            form_data["language"] = language
        if temperature is not None:
            form_data["temperature"] = str(temperature)
        if diarize:
            form_data["diarize"] = "true"
        # Array fields go as repeated form fields (what Mistral's own SDK sends),
        # not as one JSON-encoded string.
        if context_bias:
            form_data["context_bias"] = (
                [context_bias] if isinstance(context_bias, str) else list(context_bias)
            )
        if timestamp_granularities:
            form_data["timestamp_granularities"] = list(timestamp_granularities)
        # Anything else (speaker_references, ...) used to be dropped silently.
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, bool):
                form_data[key] = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                form_data[key] = [v if isinstance(v, str) else json.dumps(v) for v in value]
            elif isinstance(value, dict):
                form_data[key] = json.dumps(value)
            else:
                form_data[key] = str(value)

        logger.debug(
            "Mistral STT: model=%s, language=%s, diarize=%s",
            stt_model,
            language,
            diarize,
        )

        try:
            # The client's default "Content-Type: application/json" survives
            # httpx's header merge and would label this multipart body as JSON
            # (with no boundary), so set the form content type explicitly; httpx
            # takes the multipart boundary from this header.
            if files_param:
                content_type = f"multipart/form-data; boundary={secrets.token_hex(16)}"
            else:
                content_type = "application/x-www-form-urlencoded"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": content_type,
            }
            resp = await self._client.post(
                "/audio/transcriptions",
                data=form_data,
                files=files_param,
                headers=headers,
            )
            self._raise_for_status(resp, stt_model)
            result = resp.json()

            text = result.get("text", "")
            detected_lang = result.get("language")
            # Segments (present only when timestamp_granularities was sent) carry
            # text, start, end, an optional score and, when diarized, speaker_id.
            # The response schema has no separate word list (mistralai SDK
            # TranscriptionResponse: text, language, segments, usage).
            segments: list[TranscriptionSegment] = []
            for seg in result.get("segments") or []:
                speaker = seg.get("speaker_id")
                segments.append(
                    TranscriptionSegment(
                        text=seg.get("text", "") or "",
                        start=float(seg.get("start") or 0.0),
                        end=float(seg.get("end") or 0.0),
                        speaker=str(speaker) if speaker is not None else None,
                    )
                )

            return TranscriptionResult(
                text=text,
                language=detected_lang or language,
                duration_seconds=None,  # Not in Mistral response
                segments=segments,
                model=stt_model,
                metadata={
                    "usage": result.get("usage", {}),
                },
            )
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"STT error: {e}")

    # ------------------------------------------------------------------
    # OCR (Document Intelligence)
    # ------------------------------------------------------------------

    async def ocr(
        self,
        document: str | bytes | dict[str, Any],
        *,
        model: str | None = None,
        pages: list[int] | None = None,
        include_image_base64: bool | None = None,
        image_limit: int | None = None,
        image_min_size: int | None = None,
        table_format: str | None = None,
        document_annotation_format: dict[str, Any] | None = None,
        document_annotation_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Process a document with Mistral OCR via POST /v1/ocr.

        Args:
            document: Can be:
                - A URL string (treated as document_url)
                - Raw bytes of a file (base64-encoded and sent as file_data)
                - A dict with ``type`` key (``document_url``, ``image_url``,
                  ``file``) passed directly.
            model: OCR model (default: ``mistral-ocr-latest``).
            pages: List of page indices to process (0-based).
            include_image_base64: Include image data in response.
            image_limit: Max images to extract.
            image_min_size: Minimum image dimensions to extract.
            table_format: Table extraction format (``markdown`` or ``html``).
            document_annotation_format: JSON schema for structured extraction.
            document_annotation_prompt: Prompt to guide annotation.
            **kwargs: Extra API parameters.

        Returns:
            ``OCRResult`` object with pages, text, and metadata.
        """
        from ..models_multimodal import OCRResult

        ocr_model = model or "mistral-ocr-latest"

        # Build document payload
        if isinstance(document, str):
            # URL
            doc_payload: dict[str, Any] = {
                "type": "document_url",
                "document_url": document,
            }
        elif isinstance(document, bytes):
            # Raw file bytes → base64
            b64 = base64.b64encode(document).decode("ascii")
            doc_payload = {
                "type": "file",
                "file_data": b64,
            }
        elif isinstance(document, dict):
            doc_payload = document
        else:
            raise ProviderError(
                self.get_name(),
                f"Unsupported document type: {type(document).__name__}",
            )

        body: dict[str, Any] = {
            "model": ocr_model,
            "document": doc_payload,
        }
        if pages is not None:
            body["pages"] = pages
        if include_image_base64 is not None:
            body["include_image_base64"] = include_image_base64
        if image_limit is not None:
            body["image_limit"] = image_limit
        if image_min_size is not None:
            body["image_min_size"] = image_min_size
        if table_format is not None:
            body["table_format"] = table_format
        if document_annotation_format is not None:
            body["document_annotation_format"] = document_annotation_format
        if document_annotation_prompt is not None:
            body["document_annotation_prompt"] = document_annotation_prompt
        body.update(kwargs)

        logger.debug("Mistral OCR: model=%s, doc_type=%s", ocr_model, doc_payload.get("type"))

        try:
            resp = await self._client.post("/ocr", json=body)
            self._raise_for_status(resp, ocr_model)
            result = resp.json()

            pages_out: list[dict[str, Any]] = result.get("pages", [])
            usage = result.get("usage_info", {})

            return OCRResult(
                pages=pages_out,
                model=result.get("model", ocr_model),
                document_annotation=result.get("document_annotation"),
                pages_processed=usage.get("pages_processed", len(pages_out)),
                doc_size_bytes=usage.get("doc_size_bytes"),
                metadata={
                    "usage_info": usage,
                },
            )
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"OCR error: {e}")

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def classify(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Classify text via POST /v1/classifications.

        Args:
            input_texts: Text or list of texts to classify.
            model: Classification model.
            **kwargs: Extra API parameters.

        Returns:
            Raw API response dict.
        """
        body: dict[str, Any] = {
            "input": input_texts if isinstance(input_texts, list) else [input_texts],
        }
        if model:
            body["model"] = model
        body.update(kwargs)

        try:
            resp = await self._client.post("/classifications", json=body)
            self._raise_for_status(resp, model or "classifier")
            return resp.json()
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"Classification error: {e}")

    # ------------------------------------------------------------------
    # Moderation
    # ------------------------------------------------------------------

    async def moderate(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Moderate text via POST /v1/moderations.

        Args:
            input_texts: Text or list of texts to moderate.
            model: Moderation model (e.g. ``mistral-moderation-latest``).
            **kwargs: Extra API parameters.

        Returns:
            Raw API response dict with moderation results.
        """
        body: dict[str, Any] = {
            "input": input_texts if isinstance(input_texts, list) else [input_texts],
        }
        if model:
            body["model"] = model
        body.update(kwargs)

        try:
            resp = await self._client.post("/moderations", json=body)
            self._raise_for_status(resp, model or "moderation")
            return resp.json()
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.get_name(), f"Moderation error: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("Mistral httpx client closed.")

    # ------------------------------------------------------------------
    # Error handling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_context_overflow(message: str) -> bool:
        text = message.lower()
        return any(
            phrase in text
            for phrase in (
                "context_length",
                "context length",
                "too long",
                "token limit",
                "too large for model",
                "exceeds the maximum",
                "maximum context",
            )
        )

    def _raise_for_status(self, resp: Any, model_name: str) -> None:
        """Raise a ProviderError with useful context on HTTP errors."""
        if resp.status_code < 400:
            return
        try:
            error_body = resp.json()
            nested = error_body.get("error") if isinstance(error_body.get("error"), dict) else {}
            error_msg = (
                error_body.get("message")
                or nested.get("message")
                or error_body.get("detail")
                or resp.text
            )
        except Exception:
            error_msg = resp.text
        error_msg = str(error_msg)
        status = resp.status_code
        name = self.get_name()

        if status == 401:
            raise ProviderError(
                name,
                f"Authentication error: Invalid or missing API key. {error_msg}",
                status_code=status,
                retryable=False,
            )
        if status == 429:
            # A plan without access to a model gets a 429 with a per-minute
            # limit of 0, not a 403. Retrying cannot help, so say what it is.
            if resp.headers.get("x-ratelimit-limit-req-minute") == "0":
                raise ProviderError(
                    name,
                    f"Your Mistral plan has no quota for '{model_name}' (its request "
                    f"limit is 0 per minute). Use another model or upgrade the plan.",
                    model_name=model_name,
                    status_code=status,
                    retryable=False,
                )
            raise ProviderError(
                name, f"Rate limit exceeded. {error_msg}", status_code=status, retryable=True
            )
        if status in (400, 422) and self._is_context_overflow(error_msg):
            raise ContextLengthError(
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=error_msg,
            )
        if status in (400, 404) and any(
            phrase in error_msg.lower()
            for phrase in ("model not", "does not exist", "invalid model", "not found")
        ):
            raise ProviderError(
                name,
                f"Model '{model_name}' not found. Default is "
                f"'{self.default_model}'. Error: {error_msg}",
                model_name=model_name,
                status_code=status,
                retryable=False,
            )
        raise ProviderError(
            name,
            f"API Error ({status}): {error_msg}",
            model_name=model_name,
            status_code=status,
        )

    async def _raise_for_status_stream(self, resp: Any, model_name: str) -> None:
        """Raise on HTTP errors before a streamed body is consumed, with the
        server's message (it used to be dropped, leaving only the status)."""
        if resp.status_code >= 400:
            await resp.aread()
            self._raise_for_status(resp, model_name)
