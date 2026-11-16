# src/llmcore/providers/gemini_provider.py
"""
LLMCore provider for interacting with the Google Gemini API using google-genai SDK.

Supports:
- Chat completion (text, streaming)
- Tool/function calling with normalized OpenAI-compatible response format
- Multimodal content (vision: images via inline_data or file_data)
- Thinking config (Gemini 2.5+ models)
- Structured output (response_schema / response_json_schema)
- Speech config (response_modalities with AUDIO)
- Token counting (text and message-level)
- Dynamic model discovery
- Vertex AI mode

Tested against google-genai SDK v1.72.0.
"""

import base64
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from ..tokens import EstimateCounter as _EstimateCounter
from .base import BaseProvider, ContextPayload

google_genai_available = False
genai: Any | None = None
types: Any | None = None
APIError: type[Exception] = Exception
PermissionDenied: type[Exception] = Exception
InvalidArgument: type[Exception] = Exception
_google_genai_import_attempted = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ..media.models import (
        MediaCapability,
        MediaExecution,
        MediaJob,
        MediaRef,
        MediaResult,
    )

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover
    httpx_available = False
    httpx = None  # type: ignore

logger = logging.getLogger(__name__)

# Updated for current-generation Gemini models (April 2026).
# Used as last-resort fallback when neither the model card registry
# nor dynamic API discovery have context length data.
#: Input token limits per model, from the Developer API's own models.list
#: (v1beta, 2026-11-16). Used when neither a card nor live discovery knows a
#: model.
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash-preview-tts": 8192,
    "gemini-2.5-pro-preview-tts": 8192,
    "gemini-flash-latest": 1048576,
    "gemini-flash-lite-latest": 1048576,
    "gemini-pro-latest": 1048576,
    "gemini-2.5-flash-lite": 1048576,
    "gemini-2.5-flash-image": 32768,
    "gemini-3-flash-preview": 1048576,
    "gemini-3.1-pro-preview": 1048576,
    "gemini-3.1-pro-preview-customtools": 1048576,
    "gemini-3.1-flash-lite-preview": 1048576,
    "gemini-3.1-flash-lite": 1048576,
    "gemini-3-pro-image-preview": 131072,
    "gemini-3-pro-image": 131072,
    "gemini-3.1-flash-image-preview": 65536,
    "gemini-3.1-flash-image": 65536,
    "gemini-3.1-flash-lite-image": 65536,
    "gemini-nano-banana-2.1": 65536,
    "gemini-3.5-flash": 1048576,
    "gemini-3.5-flash-lite": 1048576,
    "gemini-omni-flash-preview": 131072,
    "gemini-omni-1.1-flash": 131072,
    "gemini-omni-1.2-flash": 131072,
    "gemini-3.5-transcribe": 98304,
    "gemini-3.6-flash": 1048576,
    "gemini-3.7-flash": 1048576,
    "gemini-3.8-flash": 1048576,
    "gemini-4-argon": 1048576,
    "gemini-4-carbon": 1048576,
    "gemini-4-hydrogen": 1048576,
    "gemini-3.1-flash-tts-preview": 8192,
    "gemini-3.8-flash-tts": 8192,
    "gemini-3.8-flash-lite-tts": 8192,
    "gemini-robotics-er-2-preview": 131072,
    "gemini-2.5-computer-use-preview-10-2025": 131072,
}
#: gemini-3.1-flash-lite-preview (the old default) is shut down and only
#: aliased; gemini-3.5-flash-lite is the current GA lite model.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model",
}

# Cache for dynamically discovered context lengths, shared across instances.
_discovered_context_lengths: dict[str, int] = {}


def _ensure_google_genai_imported() -> bool:
    """Import google-genai only when the Gemini provider is instantiated."""
    global _google_genai_import_attempted
    global APIError, InvalidArgument, PermissionDenied, genai, google_genai_available, types

    if google_genai_available and genai is not None:
        return True
    if _google_genai_import_attempted and not google_genai_available:
        return False

    _google_genai_import_attempted = True
    try:
        from google import genai as _genai
        from google.genai import types as _types
        from google.genai.errors import APIError as _APIError
    except ImportError:
        google_genai_available = False
        genai = None
        types = None
        APIError = Exception
        PermissionDenied = Exception
        InvalidArgument = Exception
        return False

    # google-api-core is optional: google-genai raises its own APIError
    # hierarchy and never these, so they are kept only for callers that still
    # catch them. Requiring the package made the whole provider unavailable
    # without it.
    try:
        from google.api_core.exceptions import InvalidArgument as _InvalidArgument
        from google.api_core.exceptions import PermissionDenied as _PermissionDenied
    except ImportError:
        _InvalidArgument = _PermissionDenied = _APIError  # type: ignore[assignment,misc]

    genai = _genai
    types = _types
    APIError = _APIError
    PermissionDenied = _PermissionDenied
    InvalidArgument = _InvalidArgument
    google_genai_available = True
    return True


# ---------------------------------------------------------------------------
# Parts: the unit Gemini replays, with thought signatures
# ---------------------------------------------------------------------------

#: The documented placeholder for function calls that did not come from this
#: model (foreign or hand-built history), as base64. Gemini 3+ rejects a
#: replayed function call without a signature ("Function call is missing a
#: thought_signature"), measured live 2026-11-16.
_SKIP_SIGNATURE = base64.b64encode(b"skip_thought_signature_validator").decode()


def _part_dict(part: Any) -> dict[str, Any]:
    """One response part as a plain dict, signatures as base64 text.

    SDK parts are dumped in JSON mode (``thought_signature`` is bytes on the
    object and base64 on the wire); direct-transport parts are already wire
    dicts. Both shapes are accepted back by both transports.
    """
    if isinstance(part, _WireValue):
        return dict(part._data) if isinstance(part._data, dict) else {}
    if isinstance(part, dict):
        return dict(part)
    dump = getattr(part, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    # Any other part-like object: read the fields this provider uses.
    out: dict[str, Any] = {}
    for name in ("text", "thought", "thought_signature", "function_call", "inline_data"):
        value = getattr(part, name, None)
        if value is None:
            continue
        if name == "thought_signature" and isinstance(value, (bytes, bytearray)):
            value = base64.b64encode(bytes(value)).decode()
        elif name == "function_call":
            value = {k: getattr(value, k, None) for k in ("name", "args", "id")}
        out[name] = value
    return out


def _part_signature(part: dict[str, Any]) -> str | None:
    return part.get("thought_signature") or part.get("thoughtSignature")


def _part_function_call(part: dict[str, Any]) -> dict[str, Any] | None:
    call = part.get("function_call") or part.get("functionCall")
    return call if isinstance(call, dict) else None


def _response_parts(response: Any) -> list[dict[str, Any]]:
    """The first candidate's parts as dicts."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return [_part_dict(p) for p in (getattr(content, "parts", None) or [])]


def _tool_call_from_part(part: dict[str, Any], index: int) -> dict[str, Any] | None:
    """A function-call part as an OpenAI-shaped tool call.

    The signature travels in ``extra_content.google.thought_signature``, which
    is where Google's own OpenAI-compatible endpoint puts it, so a caller that
    replays only ``tool_calls`` still sends it back.
    """
    call = _part_function_call(part)
    if call is None:
        return None
    tool_call: dict[str, Any] = {
        "id": call.get("id") or f"call_{index}_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": call.get("name"), "arguments": json.dumps(call.get("args") or {})},
    }
    signature = _part_signature(part)
    if signature:
        tool_call["extra_content"] = {"google": {"thought_signature": signature}}
    return tool_call


def _merge_parts(parts: list[dict[str, Any]], new: dict[str, Any]) -> None:
    """Append a streamed part, joining plain text fragments into one part."""
    plain = (
        "text" in new
        and not new.get("thought")
        and not _part_signature(new)
        and set(new) <= {"text"}
    )
    if plain and parts and set(parts[-1]) == {"text"}:
        parts[-1] = {"text": parts[-1]["text"] + (new.get("text") or "")}
        return
    if set(new) <= {"text"} and not new.get("text"):
        return  # empty keep-alive fragment
    parts.append(new)


#: Developer-API root for the direct transport. Vertex mode keeps using the SDK
#: (see ``GeminiProvider._resolve_backend``), because Vertex authenticates with
#: Google ADC rather than an API key and the SDK owns that exchange.
_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: Wire field -> SDK attribute. The REST API is camelCase while google-genai
#: exposes snake_case, so the shim below translates rather than making every
#: reader know both spellings.
_WIRE_ALIASES: dict[str, str] = {
    "finish_reason": "finishReason",
    "usage_metadata": "usageMetadata",
    "function_call": "functionCall",
    "prompt_token_count": "promptTokenCount",
    "candidates_token_count": "candidatesTokenCount",
    "total_token_count": "totalTokenCount",
    "thoughts_token_count": "thoughtsTokenCount",
    "cached_content_token_count": "cachedContentTokenCount",
}


#: Wire fields the SDK exposes as enums, so readers do ``field.name``. The REST
#: API returns them as plain strings, which would raise ``AttributeError`` on
#: ``.name`` — found exactly that way, on the first live call.
_WIRE_ENUM_FIELDS: frozenset[str] = frozenset({"finish_reason", "finishReason", "blockReason"})


class _WireEnum:
    """A wire string presented like the SDK's enum members."""

    __slots__ = ("name", "value")

    def __init__(self, value: str) -> None:
        self.name = value
        self.value = value

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    def __bool__(self) -> bool:
        return bool(self.name)


class _WireValue:
    """Attribute access over decoded REST JSON, shaped like the SDK's objects.

    The Gemini chat path reads a typed ``GenerateContentResponse`` —
    ``response.text``, ``response.candidates[0].content.parts``, ``p.thought``,
    ``finish_reason.name`` — rather than a dict. Rewriting that normalization
    for the direct transport would mean maintaining two copies of the trickiest
    logic in this provider.

    So the direct transport wraps its JSON in this instead, and the existing
    normalization runs unchanged over both. It translates camelCase wire names
    to the SDK's snake_case, and exposes the handful of derived members the
    readers use (``text``, ``function_calls``, enum-like ``.name``).
    """

    __slots__ = ("_data",)

    def __init__(self, data: Any) -> None:
        self._data = data

    # --- generic access ------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if not isinstance(self._data, dict):
            return None
        for key in (name, _WIRE_ALIASES.get(name)):
            if key and key in self._data:
                value = self._data[key]
                if name in _WIRE_ENUM_FIELDS and isinstance(value, str):
                    return _WireEnum(value)
                return _wrap_wire(value)
        return None

    def __bool__(self) -> bool:
        return bool(self._data)

    # --- derived members the readers rely on ---------------------------

    @property
    def name(self) -> Any:
        """Enum-like access: ``finish_reason.name`` on a wire string."""
        if isinstance(self._data, str):
            return self._data
        if isinstance(self._data, dict) and "name" in self._data:
            return self._data["name"]
        return None

    @property
    def text(self) -> Any:
        """Text for this node.

        ``text`` means two different things in this object graph: on a response
        it is the joined non-thought parts (the SDK's ``.text``), while on a
        *part* it is that part's own string. A property that only did the join
        silently returned ``""`` for every part, which made streaming yield
        empty deltas — found on the first live stream. So the level is decided
        by the data: a node carrying ``candidates`` joins, anything else returns
        its own field.
        """
        if not isinstance(self._data, dict):
            return ""
        if "candidates" not in self._data:
            return self._data.get("text")
        candidates = self._data.get("candidates") or []
        chunks: list[str] = []
        for candidate in candidates:
            content = (candidate or {}).get("content") or {}
            for part in content.get("parts") or []:
                if isinstance(part, dict) and part.get("text") and not part.get("thought"):
                    chunks.append(str(part["text"]))
        return "".join(chunks)

    @property
    def function_calls(self) -> list[Any]:
        """Flatten ``functionCall`` parts, matching the SDK's ``.function_calls``."""
        if not isinstance(self._data, dict):
            return []
        calls: list[Any] = []
        for candidate in self._data.get("candidates") or []:
            content = (candidate or {}).get("content") or {}
            for part in content.get("parts") or []:
                call = isinstance(part, dict) and part.get("functionCall")
                if call:
                    calls.append(_WireFunctionCall(call))
        return calls


class _WireFunctionCall:
    """One ``functionCall`` part, exposing ``name``/``args``/``id``."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def name(self) -> str:
        return str(self._data.get("name") or "")

    @property
    def args(self) -> dict[str, Any]:
        return dict(self._data.get("args") or {})

    @property
    def id(self) -> str | None:
        return self._data.get("id")


def _wrap_wire(value: Any) -> Any:
    """Wrap dicts and lists so nested access keeps working."""
    if isinstance(value, dict):
        return _WireValue(value)
    if isinstance(value, list):
        return [_wrap_wire(v) for v in value]
    return value


class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API using google-genai.

    Handles List[Message] context type and standardized tool-calling.
    Supports multimodal content via Message.metadata conventions:
      - metadata["parts"]: list of Part dicts (inline_data, file_data, etc.)
      - metadata["inline_images"]: list of {"mime_type": str, "data": base64_str}
      - metadata["file_uris"]: list of {"file_uri": str, "mime_type": str}

    Tool call responses are normalized to OpenAI-compatible format:
      choices[0].message.tool_calls = [{"id": ..., "type": "function",
        "function": {"name": ..., "arguments": ...}}]
    """

    _client: Any | None = None
    _api_key_env_var: str | None = None
    _safety_settings: list[dict[str, Any]] | None = None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the GeminiProvider using the google-genai SDK.

        Args:
            config: Configuration dictionary from ``[providers.gemini]`` containing:
                'api_key' (optional): Google AI API key.
                'api_key_env_var' (optional): Env var name for the API key.
                'default_model' (optional): Default model to use.
                'safety_settings' (optional): Dict for configuring safety settings.
                'vertex_ai' (optional): If True, use Vertex AI backend.
                'project' (optional): GCP project for Vertex AI.
                'location' (optional): GCP location for Vertex AI.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not _ensure_google_genai_imported():
            raise ImportError(
                "Google Gen AI library (`google-genai`) not installed. "
                "Install with 'pip install llmcore[gemini]'."
            )

        self._api_key_env_var = config.get("api_key_env_var")
        api_key = config.get("api_key")
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")

        self.api_key = api_key
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        self.fallback_context_length = int(
            config.get("fallback_context_length", 1048576)
        )
        self._safety_settings = self._parse_safety_settings(
            config.get("safety_settings")
        )

        # Vertex AI configuration
        self._vertex_ai = config.get("vertex_ai", False)
        # Optional per-capability media model overrides:
        #   default_image_model / default_video_model /
        #   default_tts_model / default_asr_model / default_embedding_model
        self._media_model_config: dict[str, Any] = {
            "image": config.get("default_image_model"),
            "video": config.get("default_video_model"),
            "tts": config.get("default_tts_model"),
            "asr": config.get("default_asr_model"),
            "music": config.get("default_music_model"),
            "decision": config.get("default_decision_model"),
            "embed": config.get("default_embedding_model"),
        }
        self._project = config.get("project")
        self._location = config.get("location")

        if not self.api_key and not self._vertex_ai:
            raise ConfigError(
                "Google API key not found. Set GOOGLE_API_KEY environment "
                "variable or configure api_key in config, or enable vertex_ai mode."
            )

        try:
            client_kwargs: dict[str, Any] = {}
            if self._vertex_ai:
                client_kwargs["vertexai"] = True
                if self._project:
                    client_kwargs["project"] = self._project
                if self._location:
                    client_kwargs["location"] = self._location
            else:
                client_kwargs["api_key"] = self.api_key
                # google-genai 3.x defaults the Developer API to v1, which lacks
                # TTS (speechConfig: "Unknown name"), explicit caching, the Live
                # API and the -latest/-preview aliases. v1beta has all of them.
                client_kwargs["http_options"] = types.HttpOptions(
                    api_version=str(config.get("api_version") or "v1beta")
                )

            self._client = genai.Client(**client_kwargs)
            self._backend = self._resolve_backend(config.get("backend"))
            self._http: Any = None
            logger.debug("Google Gen AI client initialized successfully.")
        except Exception as e:
            raise ConfigError(f"Google Gen AI configuration failed: {e}") from e

    def _parse_safety_settings(
        self, settings_config: dict[str, str] | None
    ) -> list[dict[str, Any]] | None:
        """Parses safety settings from config into the format expected by the SDK."""
        if not settings_config:
            return None
        parsed_settings = []
        for key, value in settings_config.items():
            try:
                category = types.HarmCategory[key.upper()]
                threshold = types.HarmBlockThreshold[value.upper()]
                parsed_settings.append(
                    {"category": category, "threshold": threshold}
                )
            except (KeyError, AttributeError):
                logger.warning(f"Invalid safety setting: {key}={value}. Skipping.")
        return parsed_settings if parsed_settings else None

    def get_name(self) -> str:
        """Returns the provider instance name (e.g. 'gemini', 'google')."""
        return self._provider_instance_name or "gemini"

    def supports_native_search(self, model: str | None = None) -> bool:
        """Gemini exposes Google Search grounding as a native search surface."""
        return True

    def _apply_native_search(
        self, generation_config_kwargs: dict[str, Any]
    ) -> None:
        """Append Google Search grounding to the request's tool list.

        Additive: preserves any function-calling tools already present. Failures
        to construct the grounding tool degrade to a logged no-op rather than
        breaking the request.
        """
        try:
            search_tool = types.Tool(google_search=types.GoogleSearch())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not build Google Search grounding tool: %s", exc)
            return
        existing = generation_config_kwargs.get("tools")
        if existing:
            generation_config_kwargs["tools"] = [*existing, search_tool]
        else:
            generation_config_kwargs["tools"] = [search_tool]

    async def get_models_details(self) -> list[ModelDetails]:
        """Dynamically discovers available models from the Google AI API.

        Uses the native async client (``client.aio.models.list()``) instead
        of wrapping the sync method in ``asyncio.to_thread()``.

        Enriches discovered models with authoritative data from the Model Card
        Registry when available (capabilities, pricing, lifecycle status).

        Updates the shared ``_discovered_context_lengths`` cache for use by
        ``get_max_context_length()``.
        """
        global _discovered_context_lengths
        details_list = []

        # Load model card registry once for enrichment
        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        try:
            models_pager = await self._client.aio.models.list()
            for model in models_pager:
                model_id = (model.name or "").replace("models/", "")
                if not model_id:
                    continue

                # SDK v1.72.0: `supported_generation_methods` replaced by
                # `supported_actions`. Check both for backward compat.
                supported_actions = (
                    getattr(model, "supported_actions", None) or []
                )
                supported_methods = (
                    getattr(model, "supported_generation_methods", None) or []
                )

                all_supported = supported_actions + supported_methods
                if all_supported:
                    generation_indicators = {
                        "generateContent",
                        "generate_content",
                        "countTokens",
                        "embedContent",
                    }
                    if not any(
                        ind in generation_indicators for ind in all_supported
                    ):
                        continue

                # Get base values from API
                input_limit = getattr(model, "input_token_limit", None)
                output_limit = getattr(model, "output_token_limit", None)
                context_len = input_limit or self.fallback_context_length
                supports_thinking = getattr(model, "thinking", False) or False

                _discovered_context_lengths[model_id] = context_len

                # Defaults (conservative)
                supports_tools = True
                supports_vision = True
                supports_streaming = True
                supports_reasoning = supports_thinking
                display_name = getattr(model, "display_name", None)
                model_type = "chat"

                # Enrich from model card if available
                card = None
                if registry is not None:
                    try:
                        card = registry.get("google", model_id)
                    except Exception:
                        pass

                metadata: dict[str, Any] = {
                    "display_name": display_name,
                    "version": getattr(model, "version", None),
                    "supported_actions": supported_actions,
                    "thinking": supports_thinking,
                    "max_temperature": getattr(
                        model, "max_temperature", None
                    ),
                }

                if card is not None:
                    # Override with authoritative card data
                    context_len = card.get_context_length()
                    output_limit = card.get_max_output()
                    supports_tools = (
                        card.capabilities.function_calling
                        or card.capabilities.tool_use
                    )
                    supports_vision = card.capabilities.vision
                    supports_reasoning = card.capabilities.reasoning
                    display_name = card.display_name or display_name
                    model_type_val = card.model_type
                    if isinstance(model_type_val, str):
                        model_type = model_type_val
                    else:
                        model_type = model_type_val.value

                    metadata["from_model_card"] = True
                    metadata["lifecycle_status"] = card.lifecycle.status

                details = ModelDetails(
                    id=model_id,
                    display_name=display_name,
                    context_length=context_len,
                    max_output_tokens=output_limit,
                    supports_streaming=supports_streaming,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    supports_reasoning=supports_reasoning,
                    model_type=model_type,
                    provider_name=self.get_name(),
                    metadata=metadata,
                )
                details_list.append(details)
            logger.info(
                f"Discovered {len(details_list)} supported models from Google AI."
            )
        except Exception as e:
            logger.error(
                f"Failed to list models from Google AI: {e}", exc_info=True
            )
            raise ProviderError(self.get_name(), f"Failed to list models: {e}") from e
        return details_list

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Returns a schema of supported GenerateContentConfig parameters.

        Updated for google-genai SDK v1.72.0 — covers all parameters that
        ``GenerateContentConfig`` accepts.
        """
        return {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "top_p": {"type": "number"},
            "top_k": {"type": "number"},
            "candidate_count": {"type": "integer"},
            "max_output_tokens": {"type": "integer"},
            "stop_sequences": {"type": "array", "items": {"type": "string"}},
            "response_mime_type": {"type": "string"},
            "response_schema": {"type": "object"},
            "response_json_schema": {"type": "object"},
            "seed": {"type": "integer"},
            "presence_penalty": {"type": "number"},
            "frequency_penalty": {"type": "number"},
            "response_logprobs": {"type": "boolean"},
            "logprobs": {"type": "integer"},
            "response_modalities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "media_resolution": {"type": "string"},
            "audio_timestamp": {"type": "boolean"},
            "speech_config": {"type": "object"},
            "thinking_config": {"type": "object"},
            "image_config": {"type": "object"},
            "cached_content": {"type": "string"},
            "routing_config": {"type": "object"},
            "model_selection_config": {"type": "object"},
            "service_tier": {"type": "string"},
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                "description": "Mapped to thinking_config (thinking_level on Gemini 3+).",
            },
            "thinking_display": {
                "type": "string",
                "enum": ["summarized", "omitted"],
                "description": "summarized = include_thoughts (thought summaries).",
            },
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Returns the maximum context length (tokens) for the given Gemini model.

        Resolution order (first match wins):
        1. Model Card Registry (authoritative, file-based metadata)
        2. Dynamically discovered cache (from ``get_models_details()`` API call)
        3. Static ``DEFAULT_GEMINI_TOKEN_LIMITS`` table (hardcoded fallback)
        4. Configured ``fallback_context_length`` (last resort)
        """
        model_name = model or self.default_model

        # 1. Model Card Registry (most authoritative)
        try:
            registry = get_model_card_registry()
            card = registry.get("google", model_name)
            if card is not None:
                limit = card.get_context_length()
                logger.debug(
                    "Resolved context length for 'google/%s' from model "
                    "card: %d",
                    model_name,
                    limit,
                )
                return limit
        except Exception as e:
            logger.debug(f"Model card registry lookup failed: {e}")

        # 2. Dynamic discovery cache (populated by get_models_details())
        limit = _discovered_context_lengths.get(model_name)
        if limit is not None:
            return limit

        # 3. Static fallback table
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is not None:
            return limit

        # 4. Configured fallback
        logger.warning(
            f"Unknown context length for Gemini model '{model_name}'. "
            f"Using fallback: {self.fallback_context_length}."
        )
        return self.fallback_context_length

    # ------------------------------------------------------------------
    # Multimodal Content Building
    # ------------------------------------------------------------------

    def _build_multimodal_parts(self, msg: Message) -> list[dict[str, Any]]:
        """Build a list of Gemini Part dicts from a Message.

        Supports multimodal content via ``Message.metadata`` conventions:

        1. ``metadata["parts"]`` — raw Part dicts passed through directly.
        2. ``metadata["inline_images"]`` — convenience list of image dicts:
           ``[{"mime_type": "image/png", "data": "<base64>"}]``
        3. ``metadata["file_uris"]`` — convenience list of file URI dicts:
           ``[{"file_uri": "gs://...", "mime_type": "image/jpeg"}]``

        Text content from ``msg.content`` is always included first if non-empty.

        Args:
            msg: An LLMCore Message instance.

        Returns:
            A list of Part dicts for the Gemini API contents format.
        """
        parts: list[dict[str, Any]] = []
        metadata = msg.metadata or {}

        if msg.content:
            parts.append({"text": msg.content})

        if "parts" in metadata:
            for part_dict in metadata["parts"]:
                parts.append(part_dict)

        if "inline_images" in metadata:
            for img in metadata["inline_images"]:
                parts.append({
                    "inline_data": {
                        "mime_type": img.get("mime_type", "image/png"),
                        "data": img["data"],
                    }
                })

        if "file_uris" in metadata:
            for file_ref in metadata["file_uris"]:
                parts.append({
                    "file_data": {
                        "file_uri": file_ref["file_uri"],
                        "mime_type": file_ref.get(
                            "mime_type", "application/octet-stream"
                        ),
                    }
                })

        if not parts:
            parts.append({"text": ""})

        return parts

    def _convert_llmcore_msgs_to_genai_contents(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Converts LLMCore messages to Gemini ``contents`` format.

        Handles:
        - System messages -> extracted as system_instruction text
        - User/Assistant messages -> user/model roles with multimodal parts
        - Tool messages -> functionResponse parts (sent as ``user`` role)
        - Consecutive same-role messages -> merged (Gemini API requirement)

        Args:
            messages: List of LLMCore Message instances.

        Returns:
            Tuple of (genai_contents list, system_instruction_text or None).
        """
        genai_history: list[dict[str, Any]] = []
        system_instruction_text: str | None = None

        processed_messages = list(messages)

        # Extract leading system message(s)
        while (
            processed_messages
            and processed_messages[0].role == LLMCoreRole.SYSTEM
        ):
            sys_msg = processed_messages.pop(0)
            if system_instruction_text:
                system_instruction_text += f"\n{sys_msg.content}"
            else:
                system_instruction_text = sys_msg.content

        # Tool results name their function; the call id alone is not a name.
        call_names: dict[str, str] = {}
        for msg in processed_messages:
            for tc in getattr(msg, "tool_calls", None) or (msg.metadata or {}).get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    func = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    call_names[str(tc["id"])] = str(func.get("name") or "")

        last_role = None
        for msg in processed_messages:
            # Handle tool result messages -> functionResponse parts
            if msg.role == LLMCoreRole.TOOL:
                tool_name = (msg.metadata or {}).get("tool_name") or call_names.get(
                    msg.tool_call_id or "", msg.tool_call_id or "unknown"
                )
                function_response: dict[str, Any] = {
                    "name": tool_name,
                    "response": {"result": msg.content},
                }
                if msg.tool_call_id:
                    # Gemini 3.8 pairs results to calls by id.
                    function_response["id"] = msg.tool_call_id
                function_response_part = {"function_response": function_response}
                if last_role == "user" and genai_history:
                    genai_history[-1]["parts"].append(function_response_part)
                else:
                    genai_history.append({
                        "role": "user",
                        "parts": [function_response_part],
                    })
                    last_role = "user"
                continue

            genai_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not genai_role:
                continue

            if genai_role == "model":
                parts = self._model_turn_parts(msg)
            else:
                parts = self._build_multimodal_parts(msg)

            if genai_role == last_role and genai_history:
                genai_history[-1]["parts"].extend(parts)
                continue

            genai_history.append({"role": genai_role, "parts": parts})
            last_role = genai_role

        return genai_history, system_instruction_text

    def _model_turn_parts(self, msg: Message) -> list[dict[str, Any]]:
        """Parts for an assistant (``model``) turn, ready to replay.

        Gemini 3+ validates thought signatures on replayed function calls. A
        call without one is a 400. So, in order of preference:

        1. ``metadata["gemini_parts"]`` (from :meth:`assistant_replay_metadata`)
           is the turn exactly as the model produced it, and is sent verbatim.
        2. Otherwise the turn is rebuilt from ``tool_calls``, taking each call's
           signature from ``extra_content.google.thought_signature``.
        3. If no call carries one (history from another model, or built by
           hand), the first call gets Google's documented placeholder.
        """
        metadata = msg.metadata or {}
        replay = metadata.get("gemini_parts")
        if replay:
            return [dict(p) for p in replay]
        tool_calls = getattr(msg, "tool_calls", None) or metadata.get("tool_calls")
        if not tool_calls:
            return self._build_multimodal_parts(msg)
        parts: list[dict[str, Any]] = []
        if msg.content:
            parts.append({"text": msg.content})
        calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            args = func.get("arguments", func.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except ValueError:
                    args = {"_raw": args}
            call: dict[str, Any] = {"name": func.get("name"), "args": args or {}}
            if tc.get("id"):
                call["id"] = tc["id"]
            part: dict[str, Any] = {"function_call": call}
            signature = ((tc.get("extra_content") or {}).get("google") or {}).get("thought_signature")
            if signature:
                part["thought_signature"] = signature
            calls.append(part)
        if calls and not any(_part_signature(c) for c in calls):
            calls[0]["thought_signature"] = _SKIP_SIGNATURE
        return parts + calls

    # ------------------------------------------------------------------
    # Tool Call Normalization
    # ------------------------------------------------------------------

    def _resolve_backend(self, requested: str | None) -> str:
        """Resolve the transport. ``"sdk"`` by default; ``"httpx"`` is opt-in.

        Vertex mode is forced onto the SDK: Vertex authenticates with Google
        Application Default Credentials rather than an API key, and reproducing
        that token exchange in llmcore would be reimplementing the part of
        ``google-genai`` that earns its keep.
        """
        req = (requested or "sdk").lower()
        if req not in ("sdk", "httpx"):
            logger.warning("Unknown Gemini backend '%s'; using the SDK.", req)
            return "sdk"
        if req == "httpx" and self._vertex_ai:
            logger.warning(
                "The Gemini direct transport targets the Developer API, which "
                "authenticates with an API key; Vertex mode uses Google ADC. "
                "Falling back to the SDK for this instance."
            )
            return "sdk"
        if req == "httpx" and not httpx_available:
            logger.warning("httpx is not installed; using the Gemini SDK.")
            return "sdk"
        return req

    def _direct_base_url(self) -> str:
        """REST root for the direct transport."""
        return _GEMINI_DEFAULT_BASE_URL

    def _get_http(self) -> Any:
        """Return the lazily-built direct HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._direct_base_url(),
                headers={
                    "x-goog-api-key": self.api_key or "",
                    "content-type": "application/json",
                },
                timeout=getattr(self, "timeout", 120.0),
            )
        return self._http

    @staticmethod
    def _config_to_wire(config: Any) -> dict[str, Any]:
        """Render a ``GenerateContentConfig`` as the REST request body fields.

        The SDK accepts a typed config object; the REST API wants
        ``generationConfig`` plus a few siblings. Only the keys the SDK actually
        set are emitted, so a request carries exactly what the caller asked for.
        """
        if config is None:
            return {}
        dump = getattr(config, "model_dump", None)
        raw = (
            dump(mode="json", exclude_none=True, by_alias=True)
            if callable(dump)
            else dict(config or {})
        )

        body: dict[str, Any] = {}
        # These are top-level on the wire rather than inside generationConfig.
        for key in ("tools", "toolConfig", "safetySettings", "systemInstruction"):
            snake = "".join("_" + c.lower() if c.isupper() else c for c in key)
            if key in raw:
                body[key] = raw.pop(key)
            elif snake in raw:
                body[key] = raw.pop(snake)
        if raw:
            body["generationConfig"] = raw
        return body

    def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
        """Map a direct failure onto the same exceptions the SDK path raises.

        Raises:
            ProviderError: Always.
        """
        if status in (401, 403):
            raise ProviderError(
                self.get_name(),
                f"Gemini authentication failed. Check GOOGLE_API_KEY / GEMINI_API_KEY. "
                f"Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"Gemini model '{model_name}' not found. Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 429:
            raise ProviderError(
                self.get_name(),
                f"Gemini rate limit reached. Error: {body}",
                model_name=model_name,
                status_code=status,
                retryable=True,
            )
        raise ProviderError(
            self.get_name(),
            f"API Error ({status}): {body}",
            model_name=model_name,
            status_code=status,
            retryable=status >= 500,
        )

    async def _direct_generate_content(
        self, model_name: str, contents: Any, config: Any
    ) -> Any:
        """POST ``:generateContent`` and wrap the reply for the SDK-shaped readers."""
        body = {"contents": self._contents_to_wire(contents), **self._config_to_wire(config)}
        try:
            resp = await self._get_http().post(
                f"/models/{model_name}:generateContent", json=body
            )
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}") from e
        if resp.status_code >= 400:
            self._raise_direct_status(resp.status_code, resp.text, model_name)
        return _WireValue(resp.json())

    async def _direct_generate_content_stream(
        self, model_name: str, contents: Any, config: Any
    ) -> AsyncGenerator[Any, None]:
        """Yield ``:streamGenerateContent`` events, wrapped like SDK chunks."""
        body = {"contents": self._contents_to_wire(contents), **self._config_to_wire(config)}
        client = self._get_http()
        async with client.stream(
            "POST",
            f"/models/{model_name}:streamGenerateContent",
            params={"alt": "sse"},
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_direct_status(resp.status_code, resp.text, model_name)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    yield _WireValue(json.loads(payload))
                except ValueError:
                    logger.warning("Skipping unparseable Gemini SSE chunk.")

    @staticmethod
    def _contents_to_wire(contents: Any) -> Any:
        """Render SDK ``Content`` objects as REST ``contents`` entries."""
        if contents is None:
            return []
        items = contents if isinstance(contents, list) else [contents]
        wire: list[Any] = []
        for item in items:
            dump = getattr(item, "model_dump", None)
            if callable(dump):
                # JSON mode: thought signatures are bytes on SDK objects.
                wire.append(dump(mode="json", exclude_none=True, by_alias=True))
            elif isinstance(item, dict):
                wire.append(item)
            else:
                wire.append({"role": "user", "parts": [{"text": str(item)}]})
        return wire

    def _normalize_tool_calls_from_response(
        self, response: Any
    ) -> list[dict[str, Any]] | None:
        """Extract and normalize function calls from a GenerateContentResponse.

        Converts Gemini FunctionCall objects into OpenAI-compatible format::

            [{"id": "...", "type": "function",
              "function": {"name": "...", "arguments": "..."}}]

        Args:
            response: A ``GenerateContentResponse`` from the SDK.

        Returns:
            List of normalized tool call dicts, or None if none found.
        """
        func_calls = getattr(response, "function_calls", None)
        if not func_calls:
            return None

        normalized = []
        for fc in func_calls:
            call_id = getattr(fc, "id", None) or str(uuid.uuid4())
            normalized.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": fc.name,
                    "arguments": json.dumps(fc.args or {}),
                },
            })
        return normalized

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
        native_search: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Sends a chat completion request to the Google Gemini API.

        Supports all ``GenerateContentConfig`` parameters via ``**kwargs``.
        Tool calls are normalized to OpenAI-compatible format in the response.
        Multimodal content is supported via ``Message.metadata`` conventions.

        Args:
            context: List of LLMCore Message objects.
            model: Model identifier (e.g., ``"gemini-2.5-flash"``).
            stream: If True, returns an async generator of streaming chunks.
            tools: Optional list of Tool definitions for function calling.
            tool_choice: Tool choice mode (``"auto"``, ``"any"``, ``"none"``).
            native_search: If True, attach Google Search grounding as a native
                tool so the model can ground its answer on live web results
                (plan §4/F9 dependency). Additive: it is appended to any
                function tools and defaults to ``False``.
            **kwargs: Additional ``GenerateContentConfig`` parameters
                (temperature, thinking_config, response_schema, etc.).

        Returns:
            Dict with OpenAI-normalized response, or async generator for streaming.

        Raises:
            ProviderError: On API or configuration errors.
            ContextLengthError: When input exceeds model context window.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Gemini client not initialized.")

        model_name = model or self.default_model

        if not (
            isinstance(context, list)
            and all(isinstance(msg, Message) for msg in context)
        ):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        genai_contents, system_instruction_text = (
            self._convert_llmcore_msgs_to_genai_contents(context)
        )
        if not genai_contents:
            raise ProviderError(self.get_name(), "No valid messages to send.")

        # Build GenerateContentConfig kwargs
        generation_config_kwargs: dict[str, Any] = {}

        effort = kwargs.pop("effort", None)
        display = kwargs.pop("thinking_display", None)
        if effort or display:
            thinking = dict(kwargs.pop("thinking_config", None) or {})
            if effort:
                thinking.update(self._effort_to_thinking(str(effort).strip().lower(), model_name))
            if display:
                thinking["include_thoughts"] = display == "summarized"
            kwargs["thinking_config"] = thinking

        # Portable names every other provider accepts. Passed through
        # untranslated they reached GenerateContentConfig, which rejects
        # unknown fields: chat(max_tokens=...) raised a pydantic error.
        for portable, native in (
            ("max_tokens", "max_output_tokens"),
            ("max_completion_tokens", "max_output_tokens"),
            ("stop", "stop_sequences"),
            ("n", "candidate_count"),
        ):
            if portable in kwargs:
                value = kwargs.pop(portable)
                if native == "stop_sequences" and isinstance(value, str):
                    value = [value]
                kwargs.setdefault(native, value)

        supported = self.get_supported_parameters()
        for key, value in kwargs.items():
            if key not in supported:
                logger.warning(
                    f"Parameter '{key}' not in declared supported parameters "
                    f"for Gemini. Passing through anyway."
                )
            generation_config_kwargs[key] = value

        if system_instruction_text:
            generation_config_kwargs["system_instruction"] = system_instruction_text

        if self._safety_settings:
            generation_config_kwargs["safety_settings"] = self._safety_settings

        # Tool definitions. (This used FunctionDeclaration.from_dict, which
        # google-genai has never had: every call with tools raised
        # AttributeError.) parameters_json_schema takes a JSON Schema as-is.
        function_declarations = (
            [
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters_json_schema=tool.parameters or {"type": "object", "properties": {}},
                )
                for tool in tools
            ]
            if tools
            else None
        )
        if function_declarations:
            generation_config_kwargs["tools"] = [
                types.Tool(function_declarations=function_declarations)
            ]

        # Tool choice -> ToolConfig mapping
        if tool_choice and function_declarations:
            mode_map = {
                "auto": "AUTO",
                "any": "ANY",
                "required": "ANY",
                "none": "NONE",
            }
            mode_str = mode_map.get(tool_choice)
            if mode_str:
                generation_config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=mode_str
                    )
                )

        if native_search:
            self._apply_native_search(generation_config_kwargs)

        config = (
            types.GenerateContentConfig(**generation_config_kwargs)
            if generation_config_kwargs
            else None
        )

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "contents": genai_contents,
                "stream": stream,
                "config": str(generation_config_kwargs),
            }
            logger.debug(
                f"RAW LLM REQUEST ({self.get_name()}): "
                f"{json.dumps(log_data, indent=2, default=str)}"
            )

        try:
            if stream:
                return await self._handle_streaming(
                    model_name, genai_contents, config
                )
            else:
                return await self._handle_non_streaming(
                    model_name, genai_contents, config
                )
        except (APIError, InvalidArgument) as e:
            status = getattr(e, "code", None)
            status = status if isinstance(status, int) else None
            logger.debug("Google AI API error (%s): %s", status, e)
            if self._is_context_overflow(str(e)):
                raise ContextLengthError(
                    model_name=model_name, message=str(e)
                ) from e
            raise ProviderError(
                self.get_name(),
                f"Google AI API Error: {e}",
                model_name=model_name,
                status_code=status,
                retryable=status in (429, 500, 503, 504) if status else None,
                original_exception=e,
            ) from e
        except PermissionDenied as e:
            logger.error(
                f"Permission denied during Gemini chat: {e}", exc_info=True
            )
            raise ProviderError(self.get_name(), f"Permission denied: {e}") from e
        except (ContextLengthError, ProviderError):
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during Gemini chat: {e}", exc_info=True
            )
            raise ProviderError(
                self.get_name(), f"An unexpected error occurred: {e}"
            ) from e

    async def _handle_non_streaming(
        self,
        model_name: str,
        genai_contents: list[dict[str, Any]],
        config: Any | None,
    ) -> dict[str, Any]:
        """Process a non-streaming generate_content call and normalize the response.

        Args:
            model_name: The model identifier string.
            genai_contents: The converted contents list.
            config: The GenerateContentConfig or None.

        Returns:
            OpenAI-normalized response dict.
        """
        if self._backend == "httpx":
            response = await self._direct_generate_content(model_name, genai_contents, config)
        else:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=genai_contents,
                config=config,
            )

        parts = _response_parts(response)
        text_content = "".join(
            p.get("text") or "" for p in parts if "text" in p and not p.get("thought")
        )
        thinking_content = "".join(
            p.get("text") or "" for p in parts if p.get("thought")
        ) or None
        tool_calls = [
            tc for i, p in enumerate(parts) if (tc := _tool_call_from_part(p, i)) is not None
        ] or None

        # Finish reason
        finish_reason = None
        if response.candidates:
            fr = response.candidates[0].finish_reason
            finish_reason = fr.name if fr else None

        # Build message dict
        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": text_content if not tool_calls else (text_content or None),
        }
        if tool_calls:
            message_dict["tool_calls"] = tool_calls
        if thinking_content:
            message_dict["thinking"] = thinking_content
        if tool_calls or any(_part_signature(p) for p in parts):
            # The turn exactly as produced, signatures included: what a tool
            # loop must send back (see assistant_replay_metadata).
            message_dict["gemini_parts"] = parts

        usage_dict = self._normalize_usage_metadata(response.usage_metadata)

        response_dict: dict[str, Any] = {
            "id": getattr(response, "response_id", None),
            "model": model_name,
            "model_version": getattr(response, "model_version", None),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_dict,
        }

        if self.log_raw_payloads_enabled:
            logger.debug(
                f"RAW LLM RESPONSE ({self.get_name()}): "
                f"{json.dumps(response_dict, indent=2, default=str)}"
            )
        return response_dict

    async def _handle_streaming(
        self,
        model_name: str,
        genai_contents: list[dict[str, Any]],
        config: Any | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process a streaming generate_content_stream call.

        Yields OpenAI-normalized streaming chunks. Handles thought parts
        by placing them in ``delta.thinking`` separate from ``delta.content``.

        Args:
            model_name: The model identifier string.
            genai_contents: The converted contents list.
            config: The GenerateContentConfig or None.

        Yields:
            Dicts with ``choices[0].delta.content`` and optionally
            ``choices[0].delta.thinking``.
        """
        # Both transports assign `response_stream`, so the normalization below
        # is shared rather than written once per transport. The direct generator
        # yields _WireValue chunks, which read exactly like the SDK's.
        if self._backend == "httpx":
            response_stream = self._direct_generate_content_stream(
                model_name, genai_contents, config
            )
        else:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=model_name,
                contents=genai_contents,
                config=config,
            )

        async def stream_wrapper():
            parts: list[dict[str, Any]] = []
            tool_index = 0
            finish_reason = None
            usage: dict[str, Any] = {}
            async for chunk in response_stream:
                if self.log_raw_payloads_enabled:
                    logger.debug(
                        f"RAW LLM STREAM CHUNK ({self.get_name()}): {chunk}"
                    )

                text_delta = ""
                thought_delta = ""
                calls: list[dict[str, Any]] = []
                for part in _response_parts(chunk):
                    _merge_parts(parts, part)
                    if part.get("thought"):
                        thought_delta += part.get("text") or ""
                    elif _part_function_call(part) is not None:
                        # Gemini streams each function call whole, so it is
                        # emitted as one complete tool-call delta.
                        call = _tool_call_from_part(part, tool_index)
                        if call is not None:
                            calls.append({"index": tool_index, **call})
                            tool_index += 1
                    elif part.get("text") is not None:
                        text_delta += part.get("text") or ""

                delta: dict[str, Any] = {"content": text_delta}
                if thought_delta:
                    delta["thinking"] = thought_delta
                if calls:
                    delta["tool_calls"] = calls
                candidates = getattr(chunk, "candidates", None) or []
                reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                if reason:
                    finish_reason = getattr(reason, "name", str(reason))
                if getattr(chunk, "usage_metadata", None):
                    usage = self._normalize_usage_metadata(chunk.usage_metadata)
                yield {"choices": [{"delta": delta}]}

            # Final chunk: finish reason, usage (both were dropped before) and,
            # when the turn has calls or signatures, the parts to replay.
            final_delta: dict[str, Any] = {"content": ""}
            if tool_index or any(_part_signature(p) for p in parts):
                final_delta["gemini_parts"] = parts
            if tool_index and finish_reason in (None, "STOP"):
                finish_reason = "tool_calls"
            yield {"choices": [{"delta": final_delta, "finish_reason": finish_reason}], "usage": usage}

        return stream_wrapper()

    @staticmethod
    def _normalize_usage_metadata(um: Any) -> dict[str, Any]:
        """Gemini usage with the OpenAI keys llmcore reads.

        Thinking tokens are billed as output, so ``completion_tokens`` counts
        them. It used to be ``candidates_token_count`` alone, which undercounted
        every thinking model's cost.
        """
        if not um:
            return {}
        prompt = getattr(um, "prompt_token_count", None) or 0
        candidates = getattr(um, "candidates_token_count", None) or 0
        thoughts = getattr(um, "thoughts_token_count", None) or 0
        usage: dict[str, Any] = {
            "prompt_token_count": prompt,
            "candidates_token_count": candidates,
            "total_token_count": getattr(um, "total_token_count", None),
            "prompt_tokens": prompt,
            "completion_tokens": candidates + thoughts,
            "total_tokens": getattr(um, "total_token_count", None) or prompt + candidates + thoughts,
        }
        if thoughts:
            usage["thoughts_token_count"] = thoughts
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}
        cached = getattr(um, "cached_content_token_count", None)
        if cached:
            usage["cached_content_token_count"] = cached
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return usage

    @staticmethod
    def _effort_to_thinking(effort: str, model: str) -> dict[str, Any]:
        """llmcore's effort as Gemini's thinking settings for ``model``.

        Gemini 3+ takes ``thinking_level`` (minimal/low/medium/high, xhigh on
        gemini-4-hydrogen) and cannot switch thinking off; Gemini 4 and 3.8
        reject ``minimal``. Gemini 2.5 takes a token budget instead.
        """
        if model.startswith("gemini-2"):
            budgets = {"none": 0, "minimal": 512, "low": 1024, "medium": 8192,
                       "high": 24576, "xhigh": 32768, "max": 32768}
            return {"thinking_budget": budgets.get(effort, 8192)}
        no_minimal = model.startswith(("gemini-4", "gemini-3.8"))
        levels = {
            "none": "low" if no_minimal else "minimal",
            "minimal": "low" if no_minimal else "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh" if "hydrogen" in model else "high",
            "max": "xhigh" if "hydrogen" in model else "high",
        }
        return {"thinking_level": levels.get(effort, "medium")}

    @staticmethod
    def _is_context_overflow(text: str) -> bool:
        """Gemini's prompt-too-long error, not just any message about tokens.

        "The input token count (N) exceeds the maximum number of tokens allowed
        (M)." The old test (``"token" in message``) also caught
        max_output_tokens and continuation-token errors.
        """
        lowered = text.lower()
        return "exceeds the maximum number of tokens" in lowered or (
            "input token count" in lowered and "exceed" in lowered
        ) or "context length" in lowered

    def assistant_replay_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """The parts to send back on the next assistant turn of a tool loop.

        Gemini 3+ requires the thought signature that came with a function call
        to be sent back with it. A replayed call without one is a 400. For a
        stream, pass the last chunk.
        """
        try:
            choice = (response.get("choices") or [{}])[0]
        except (AttributeError, IndexError, TypeError):
            return {}
        source = choice.get("message") or choice.get("delta") or {}
        parts = source.get("gemini_parts")
        return {"gemini_parts": list(parts)} if parts else {}

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """Thought summaries (with ``thinking_display="summarized"``)."""
        try:
            return ((response.get("choices") or [{}])[0].get("message") or {}).get("thinking")
        except (AttributeError, IndexError, TypeError):
            return None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        try:
            return ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("thinking")
        except (AttributeError, IndexError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Tool Call Extraction (Public API)
    # ------------------------------------------------------------------

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a normalized Gemini response dict.

        Extracts from the OpenAI-normalized response format produced by
        ``chat_completion()``.

        Args:
            response: The normalized response dict from ``chat_completion()``.

        Returns:
            List of ``ToolCall`` objects. Empty list if no tool calls.
        """
        tool_calls_out: list[ToolCall] = []
        try:
            choices = response.get("choices", [])
            if not choices:
                return tool_calls_out
            message = choices[0].get("message", {})
            raw_calls = message.get("tool_calls")
            if not raw_calls:
                return tool_calls_out
            for tc in raw_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    arguments = (
                        json.loads(args_str)
                        if isinstance(args_str, str)
                        else args_str
                    )
                except json.JSONDecodeError:
                    arguments = {"raw": args_str}
                tool_calls_out.append(
                    ToolCall(
                        id=tc.get("id", str(uuid.uuid4())),
                        name=func.get("name", "unknown"),
                        arguments=arguments,
                    )
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(
                f"Failed to extract tool calls from Gemini response: {e}"
            )
        return tool_calls_out

    # ------------------------------------------------------------------
    # Token Counting
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Counts tokens for a text string using the Gemini API."""
        if not self._client:
            logger.warning(
                "Gemini client not available. Approximating token count."
            )
            return _EstimateCounter().count(text)
        if not text:
            return 0

        target_model = model or self.default_model
        try:
            response = await self._client.aio.models.count_tokens(
                contents=[text], model=target_model
            )
            return response.total_tokens
        except Exception as e:
            logger.error(
                f"Failed to count tokens with Gemini API for model "
                f"'{target_model}': {e}",
                exc_info=True,
            )
            return _EstimateCounter().count(text)

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Counts tokens for a list of messages using the Gemini API."""
        if not self._client:
            logger.warning(
                "Gemini client not available. Approximating message token count."
            )
            counter = _EstimateCounter()
            return sum(counter.count(msg.content) for msg in messages) + len(messages)
        if not messages:
            return 0

        target_model = model or self.default_model
        genai_contents, system_text = (
            self._convert_llmcore_msgs_to_genai_contents(messages)
        )
        if not genai_contents:
            if system_text:
                return await self.count_tokens(
                    system_text, model=target_model
                )
            return 0
        try:
            response = await self._client.aio.models.count_tokens(
                contents=genai_contents, model=target_model
            )
            return response.total_tokens
        except Exception as e:
            logger.error(
                f"Failed to count message tokens with Gemini API for model "
                f"'{target_model}': {e}",
                exc_info=True,
            )
            counter = _EstimateCounter()
            total = sum(
                counter.count(part.get("text", ""))
                for content_dict in genai_contents
                for part in content_dict.get("parts", [])
                if "text" in part
            )
            return total + len(genai_contents)

    # ------------------------------------------------------------------
    # Response Content Extraction
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Extract text content from Gemini non-streaming response.

        Handles GenerateContentResponse objects, OpenAI-normalized dicts,
        and native Gemini dict format.

        Args:
            response: The raw response (dict or GenerateContentResponse).

        Returns:
            The extracted text content.
        """
        try:
            if hasattr(response, "text"):
                return response.text or ""

            if isinstance(response, dict) and "choices" in response:
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content") or ""

            if isinstance(response, dict):
                candidates = response.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text") or ""
                if "text" in response:
                    return response.get("text") or ""

            return ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(
                f"Failed to extract content from Gemini response: {e}"
            )
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Extract text delta from Gemini streaming chunk.

        Handles GenerateContentResponse objects, OpenAI-normalized dicts,
        and native Gemini dict format.

        Args:
            chunk: A single streaming chunk (dict or GenerateContentResponse).

        Returns:
            The extracted text delta.
        """
        try:
            if hasattr(chunk, "text"):
                return chunk.text or ""

            if isinstance(chunk, dict) and "choices" in chunk:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    return delta.get("content") or ""

            if isinstance(chunk, dict):
                candidates = chunk.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text") or ""
                if "text" in chunk:
                    return chunk.get("text") or ""

            return ""
        except (KeyError, IndexError, TypeError):
            return ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------


    # ==================================================================
    # Media subsystem adapter (llmcore.media protocols)
    # ==================================================================
    #
    # Phase M4 of the media subsystem design spec. Gemini has the largest media
    # surface llmcore curates, and Veo makes it the FIRST true async-job
    # provider — so this is where the MediaJob lifecycle is validated against a
    # real vendor rather than the in-repo fake.
    #
    # Veo returns a google-genai ``GenerateVideosOperation``; llmcore keeps the
    # operation object in provider_metadata and refreshes it through
    # ``client.aio.operations.get()``. MediaJobManager owns the backoff,
    # timeout and cancellation policy — this adapter only reports state.

    #: Capabilities available on BOTH the Gemini Developer API and Vertex AI.
    _MEDIA_CAPABILITIES: frozenset[str] = frozenset({"image_generate", "tts"})

    #: Capabilities that exist ONLY on Vertex AI. Verified live: the Developer
    #: API rejects ``models.generate_images`` / ``edit_image`` / ``upscale_image``
    #: with "This method is only supported in Gemini Enterprise Agent Platform
    #: mode". Declaring them unconditionally would make the router call an
    #: endpoint that always errors for Developer-API users.
    _VERTEX_ONLY_MEDIA_CAPABILITIES: frozenset[str] = frozenset(
        # Veo is Vertex-only too: the Developer API shut its Veo previews down
        # on 2026-10-22 (models 404), and google-genai 3.x refuses
        # generate_videos outside Vertex before sending anything.
        {"image_edit", "image_upscale", "video_generate"}
    )

    #: Developer-API-only capabilities: ``gemini-3.5-transcribe`` is listed and
    #: verified there (generateContent, 2026-11-16); its Vertex availability
    #: is unverified, so it is not declared in Vertex mode.
    _DEV_ONLY_MEDIA_CAPABILITIES: frozenset[str] = frozenset(
        # Live API (bidiGenerateContent) sessions, verified on the Developer
        # API: gemini-3.5-transcribe-live (asr_stream) and gemini-3.8-live
        # (voice_agent).
        # music: Lyria (generateContent, MP3 out), verified 2026-11-16.
        {"asr", "asr_stream", "voice_agent", "music"}
    )

    #: Default models per media capability, overridable per call and via config.
    _DEFAULT_IMAGE_MODEL = "imagen-4.0-generate-001"        # Vertex only
    _DEFAULT_DEV_IMAGE_MODEL = "gemini-nano-banana-2.1"     # Developer API (GA)
    _DEFAULT_VIDEO_MODEL = "veo-3.1-generate-001"           # Vertex only
    _DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
    _DEFAULT_ASR_MODEL = "gemini-3.5-transcribe"
    _DEFAULT_MUSIC_MODEL = "lyria-3.5"
    _DEFAULT_EMBED_MODEL = "gemini-embedding-2"

    def media_capabilities(self) -> "frozenset[MediaCapability]":
        """Capabilities Gemini can serve in the CURRENT auth mode.

        Image editing and upscaling go through Imagen's dedicated endpoints,
        which exist only on Vertex AI, so they are declared only when
        ``vertex_ai = true``.
        """
        from ..media.models import MediaCapability

        names = set(self._MEDIA_CAPABILITIES)
        if self._vertex_ai:
            names |= self._VERTEX_ONLY_MEDIA_CAPABILITIES
        else:
            names |= self._DEV_ONLY_MEDIA_CAPABILITIES
        return frozenset(MediaCapability(c) for c in names)

    def media_execution(
        self, capability: "MediaCapability", model: str | None = None
    ) -> "MediaExecution":
        """Return how *capability* completes.

        Veo is a long-running operation; everything else answers synchronously.
        """
        from ..media.models import MediaCapability, MediaExecution

        if capability in (MediaCapability.ASR_STREAM, MediaCapability.VOICE_AGENT):
            return MediaExecution.STREAM
        if capability is MediaCapability.VIDEO_GENERATE:
            return MediaExecution.ASYNC_JOB
        return MediaExecution.REQUEST_RESPONSE

    # --- helpers ---

    def _media_model(self, capability: str, model: str | None) -> str:
        """Resolve the model for a media capability, honouring config."""
        if model:
            return model
        configured = self._media_model_config.get(capability)
        if configured:
            return str(configured)
        return {
            "image": self._DEFAULT_IMAGE_MODEL,
            "video": self._DEFAULT_VIDEO_MODEL,
            "tts": self._DEFAULT_TTS_MODEL,
            "asr": self._DEFAULT_ASR_MODEL,
            "music": self._DEFAULT_MUSIC_MODEL,
            "embed": self._DEFAULT_EMBED_MODEL,
        }[capability]

    @staticmethod
    async def _to_genai_image(ref: "MediaRef") -> Any:
        """Convert a :class:`MediaRef` into a ``types.Image``."""
        from google.genai import types as genai_types

        if ref.is_remote:
            from ..media.artifacts import default_fetcher

            data = await default_fetcher()(ref.url or "")
        else:
            data = ref.read_bytes()
        return genai_types.Image(image_bytes=data, mime_type=ref.mime_type or "image/png")

    def _images_to_artifacts(self, generated: Any) -> list[Any]:
        """Map ``GeneratedImage`` objects onto media artifacts."""
        import hashlib

        from ..media.models import MediaArtifact, MediaKind, MediaProvenance

        artifacts: list[MediaArtifact] = []
        for item in generated or []:
            image = getattr(item, "image", None)
            if image is None:
                continue
            data = getattr(image, "image_bytes", None)
            artifacts.append(
                MediaArtifact(
                    kind=MediaKind.IMAGE,
                    data=data,
                    uri=getattr(image, "gcs_uri", None),
                    mime_type=getattr(image, "mime_type", None) or "image/png",
                    checksum_sha256=hashlib.sha256(data).hexdigest() if data else None,
                    provenance=MediaProvenance(
                        watermarked=True, generator="google", provider_declared=True
                    ),
                    provider_metadata={
                        "enhanced_prompt": getattr(item, "enhanced_prompt", None),
                        "rai_filtered_reason": getattr(item, "rai_filtered_reason", None),
                    },
                )
            )
        return artifacts

    # --- image ---

    async def generate_image_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        reference_images: "Sequence[MediaRef] | None" = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Generate images with Imagen.

        Unlike OpenAI, Imagen supports ``negative_prompt`` natively, so it is
        forwarded rather than dropped. ``size`` maps to ``image_size``; an
        ``aspect_ratio`` kwarg is also accepted and passed through.
        """
        from google.genai import types as genai_types

        from ..media.models import MediaCapability, MediaResult, MediaUsage

        if reference_images:
            return await self.edit_image_media(
                prompt, image=reference_images[0], model=model, n=n, size=size, **kwargs
            )

        if not self._vertex_ai:
            # Developer API: Imagen's dedicated endpoint is unavailable, but the
            # image-capable Gemini models generate through generate_content with
            # an IMAGE response modality. Same capability, different transport.
            return await self._generate_image_via_generate_content(
                prompt, model=model, n=n, size=size, **kwargs
            )

        img_model = self._media_model("image", model)
        config_kwargs: dict[str, Any] = {"number_of_images": n}
        if size is not None:
            config_kwargs["image_size"] = size
        if negative_prompt is not None:
            config_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            logger.debug("Imagen does not expose a seed parameter; ignoring it.")
        config_kwargs.update(kwargs)

        try:
            response = await self._client.aio.models.generate_images(
                model=img_model,
                prompt=prompt,
                config=genai_types.GenerateImagesConfig(**config_kwargs),
            )
        except Exception as e:
            self._raise_media_error(e, img_model, "image generation")

        artifacts = self._images_to_artifacts(getattr(response, "generated_images", None))
        return MediaResult(
            capability=MediaCapability.IMAGE_GENERATE,
            provider=self.get_name(),
            model=img_model,
            artifacts=tuple(artifacts),
            usage=MediaUsage(
                provider=self.get_name(),
                model=img_model,
                basis="per_image",
                images=len(artifacts),
            ),
            raw={"generated": len(artifacts)},
        )

    async def _generate_image_via_generate_content(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Generate images on the Gemini Developer API.

        Imagen's ``generate_images`` endpoint is Vertex-only, so on the
        Developer API image generation runs through ``generate_content`` with an
        ``IMAGE`` response modality. ``n`` is not supported on this path — the
        model returns what it returns — so it is logged rather than faked.
        """
        import hashlib

        from google.genai import types as genai_types

        from ..media.models import (
            MediaArtifact,
            MediaCapability,
            MediaKind,
            MediaProvenance,
            MediaResult,
            MediaUsage,
        )

        img_model = model or self._media_model_config.get("image") or self._DEFAULT_DEV_IMAGE_MODEL
        if n != 1:
            logger.debug(
                "The Gemini Developer API image path has no image-count parameter; "
                "requested n=%d is advisory.",
                n,
            )
        # The shape and resolution belong in ``image_config``: GenerateContentConfig
        # rejects a top-level ``aspect_ratio``. ``size`` is Gemini's "1K"/"2K"/"4K";
        # a pixel size such as "1024x1024" means nothing here and is dropped.
        image_config = {k: kwargs.pop(k) for k in ("aspect_ratio", "image_size") if kwargs.get(k) is not None}
        for k in ("aspect_ratio", "image_size"):
            kwargs.pop(k, None)
        if size is not None and "image_size" not in image_config:
            if str(size).upper() in ("1K", "2K", "4K"):
                image_config["image_size"] = str(size).upper()
            else:
                logger.debug("Gemini image size is 1K, 2K or 4K; ignoring size=%r.", size)
        if image_config:
            kwargs["image_config"] = genai_types.ImageConfig(**image_config)
        try:
            response = await self._client.aio.models.generate_content(
                model=img_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"], **kwargs
                ),
            )
        except Exception as e:
            self._raise_media_error(e, img_model, "image generation")

        artifacts: list[MediaArtifact] = []
        for candidate in getattr(response, "candidates", None) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline is not None else None
                if not data:
                    continue
                artifacts.append(
                    MediaArtifact(
                        kind=MediaKind.IMAGE,
                        data=data,
                        mime_type=getattr(inline, "mime_type", None) or "image/png",
                        checksum_sha256=hashlib.sha256(data).hexdigest(),
                        provenance=MediaProvenance(
                            watermarked=True, generator="google", provider_declared=True
                        ),
                    )
                )

        return MediaResult(
            capability=MediaCapability.IMAGE_GENERATE,
            provider=self.get_name(),
            model=img_model,
            artifacts=tuple(artifacts),
            usage=MediaUsage(
                provider=self.get_name(),
                model=img_model,
                basis="per_image",
                images=len(artifacts),
            ),
            raw={"transport": "generate_content"},
        )

    async def edit_image_media(
        self,
        prompt: str,
        *,
        image: "MediaRef",
        mask: "MediaRef | None" = None,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Edit an image with Imagen's edit surface."""
        from google.genai import types as genai_types

        from ..media.models import MediaCapability, MediaResult, MediaUsage

        img_model = self._media_model("image", model)
        reference: list[Any] = [
            genai_types.RawReferenceImage(
                reference_id=1, reference_image=await self._to_genai_image(image)
            )
        ]
        if mask is not None:
            reference.append(
                genai_types.MaskReferenceImage(
                    reference_id=2, reference_image=await self._to_genai_image(mask)
                )
            )
        config_kwargs: dict[str, Any] = {"number_of_images": n, **kwargs}

        try:
            response = await self._client.aio.models.edit_image(
                model=img_model,
                prompt=prompt,
                reference_images=reference,
                config=genai_types.EditImageConfig(**config_kwargs),
            )
        except Exception as e:
            self._raise_media_error(e, img_model, "image editing")

        artifacts = self._images_to_artifacts(getattr(response, "generated_images", None))
        return MediaResult(
            capability=MediaCapability.IMAGE_EDIT,
            provider=self.get_name(),
            model=img_model,
            artifacts=tuple(artifacts),
            usage=MediaUsage(
                provider=self.get_name(),
                model=img_model,
                basis="per_image",
                images=len(artifacts),
            ),
        )

    async def upscale_image_media(
        self,
        *,
        image: "MediaRef",
        model: str | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Upscale an image. Imagen accepts discrete factors such as ``x2``/``x4``."""
        from google.genai import types as genai_types

        from ..media.models import MediaCapability, MediaResult, MediaUsage

        img_model = self._media_model("image", model)
        factor = f"x{int(scale)}" if scale else "x2"
        try:
            response = await self._client.aio.models.upscale_image(
                model=img_model,
                image=await self._to_genai_image(image),
                upscale_factor=factor,
                config=genai_types.UpscaleImageConfig(**kwargs) if kwargs else None,
            )
        except Exception as e:
            self._raise_media_error(e, img_model, "image upscaling")

        artifacts = self._images_to_artifacts(getattr(response, "generated_images", None))
        return MediaResult(
            capability=MediaCapability.IMAGE_UPSCALE,
            provider=self.get_name(),
            model=img_model,
            artifacts=tuple(artifacts),
            usage=MediaUsage(
                provider=self.get_name(), model=img_model, basis="per_image", images=len(artifacts)
            ),
        )

    # --- audio ---

    async def synthesize_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Synthesize speech with Gemini's native TTS.

        Gemini TTS runs through ``generate_content`` with an audio response
        modality rather than a dedicated endpoint, and returns raw PCM
        (24 kHz, 16-bit mono) — there is no container-format or speed
        parameter, so those are ignored with a debug log.
        """
        import hashlib

        from google.genai import types as genai_types

        from ..media.models import (
            MediaArtifact,
            MediaCapability,
            MediaKind,
            MediaResult,
            MediaUsage,
        )

        for unsupported, value in (("audio_format", audio_format), ("speed", speed)):
            if value is not None:
                logger.debug("Gemini TTS ignores '%s' (it returns raw PCM).", unsupported)

        tts_model = self._media_model("tts", model)
        speech_config = genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=voice or "Kore"
                )
            )
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=tts_model,
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"], speech_config=speech_config, **kwargs
                ),
            )
        except Exception as e:
            self._raise_media_error(e, tts_model, "speech synthesis")

        data = b""
        mime = "audio/L16;rate=24000"
        for candidate in getattr(response, "candidates", None) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    data = inline.data
                    mime = getattr(inline, "mime_type", None) or mime
                    break

        artifact = MediaArtifact(
            kind=MediaKind.AUDIO,
            data=data,
            mime_type=mime,
            sample_rate_hz=sample_rate_hz or 24000,
            checksum_sha256=hashlib.sha256(data).hexdigest() if data else None,
        )
        return MediaResult(
            capability=MediaCapability.TTS,
            provider=self.get_name(),
            model=tts_model,
            artifacts=(artifact,),
            usage=MediaUsage(
                provider=self.get_name(),
                model=tts_model,
                basis="per_character",
                characters=len(text),
            ),
        )

    # --- transcription ---

    async def transcribe_media(
        self,
        *,
        audio: "MediaRef",
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Transcribe audio with ``gemini-3.5-transcribe`` (``generateContent``).

        The model answers with an ``audio_transcription`` part (``text``
        only; ``response.text`` is empty), and ignores prompting for
        timestamps, so ``timestamps`` / ``diarize`` are not offered. A
        ``language`` hint is sent as a short text part.
        """
        from google.genai import types as genai_types

        from ..media.models import MediaCapability, MediaResult, MediaUsage
        from ..models_multimodal import TranscriptionResult

        for unsupported, value in (("timestamps", timestamps), ("diarize", diarize)):
            if value:
                logger.debug("Gemini transcription ignores '%s' (text only).", unsupported)
        asr_model = self._media_model("asr", model)
        if audio.is_remote:
            from ..media.artifacts import default_fetcher

            data = await default_fetcher()(audio.url or "")
        else:
            data = audio.read_bytes()
        parts: list[Any] = [
            genai_types.Part.from_bytes(data=data, mime_type=audio.mime_type or "audio/mpeg")
        ]
        if language:
            parts.append(genai_types.Part.from_text(text=f"The audio is in {language}."))
        try:
            response = await self._client.aio.models.generate_content(
                model=asr_model,
                contents=parts,
                config=genai_types.GenerateContentConfig(**kwargs) if kwargs else None,
            )
        except Exception as e:
            self._raise_media_error(e, asr_model, "transcription")

        pieces: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                spoken = getattr(part, "audio_transcription", None)
                if spoken is not None and getattr(spoken, "text", None):
                    pieces.append(spoken.text)
                elif getattr(part, "text", None):
                    pieces.append(part.text)
        usage = getattr(response, "usage_metadata", None)
        transcript = TranscriptionResult(
            text=" ".join(p.strip() for p in pieces if p.strip()),
            language=language,
            model=str(getattr(response, "model_version", None) or asr_model),
            metadata={"response_id": getattr(response, "response_id", None)},
        )
        return MediaResult(
            capability=MediaCapability.ASR,
            provider=self.get_name(),
            model=transcript.model,
            artifacts=(transcript.to_artifact(),),
            usage=MediaUsage(
                provider=self.get_name(),
                model=transcript.model,
                basis="per_token",
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            ),
        )

    # --- decisions (gemini-decision-1) ---

    async def create_decision(
        self,
        input: Any,  # noqa: A002 - the API's field name
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Any:
        """Calibrated typed judgments via ``models/gemini-decision-1:createDecision``.

        The same primitives as TypeSafe's Jev: ``noul`` (yes/no probability),
        ``choice`` (one option, per-option probabilities) and ``score``
        (rubric level). Questions are the TypeSafe builders
        (:class:`~llmcore.providers.typesafe_provider.Noul`, ``Choice``,
        ``Score``) or equivalent dicts; the result is the same
        :class:`~llmcore.providers.typesafe_provider.SystemOneResult`, so code
        written against one decision provider works with the other. Gemini
        has no ``box`` question. Output tokens are not billed (0).

        Raises:
            ValueError: For invalid or ``box`` questions (before any request).
            ProviderError: On API errors.
        """
        from .typesafe_provider import SystemOneResult, normalize_questions

        wire = normalize_questions(questions)
        for name, question in wire.items():
            if question.get("type") == "box":
                raise ValueError(f"Question {name!r}: Gemini decisions have no 'box' type.")
            if question.get("type") == "choice":
                question["criteria"] = {k: ("" if v is None else v) for k, v in question["criteria"].items()}
        used = model or self._media_model_config.get("decision") or "gemini-decision-1"
        try:
            response = await self._client.aio.decisions.create(model=used, input=input, questions=wire)
        except Exception as e:
            self._raise_media_error(e, used, "decision")
        data = response.model_dump(mode="json", exclude_none=True) if hasattr(response, "model_dump") else dict(response)
        answers = {
            k: v for k, v in (data.get("answers") or {}).items()
            if isinstance(v, dict) and v.get("type") in ("noul", "choice", "score")
        }
        usage = data.get("usage") or {}
        return SystemOneResult.model_validate({
            "model": used,
            "answers": answers,
            "usage": {"input_tokens": usage.get("total_input_tokens"),
                      "output_tokens": usage.get("total_output_tokens")},
            "raw": data,
        })

    #: Parity with TypeSafeProvider / CommandCodeProvider.system_one.
    async def system_one(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Any:
        """Alias of :meth:`create_decision` (``state`` is the shared input)."""
        return await self.create_decision(state, questions, model=model)

    # --- music (Lyria) ---

    async def generate_music_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Generate music with Lyria (``generateContent``).

        Lyria answers with MP3 audio inline plus a text part carrying the
        lyrics (``<instrumental>`` for instrumentals), kept in
        ``provider_metadata["lyrics"]``. It takes no duration parameter, so
        ``duration_seconds`` is added to the prompt as a hint.
        """
        import hashlib

        from google.genai import types as genai_types

        from ..media.models import MediaArtifact, MediaCapability, MediaKind, MediaResult, MediaUsage

        music_model = self._media_model("music", model)
        text = prompt if duration_seconds is None else f"{prompt} (about {int(duration_seconds)} seconds)"
        try:
            response = await self._client.aio.models.generate_content(
                model=music_model,
                contents=text,
                config=genai_types.GenerateContentConfig(**kwargs) if kwargs else None,
            )
        except Exception as e:
            self._raise_media_error(e, music_model, "music generation")

        artifacts: list[Any] = []
        lyrics: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    artifacts.append(MediaArtifact(
                        kind=MediaKind.AUDIO,
                        data=inline.data,
                        mime_type=getattr(inline, "mime_type", None) or "audio/mpeg",
                        checksum_sha256=hashlib.sha256(inline.data).hexdigest(),
                    ))
                elif getattr(part, "text", None):
                    lyrics.append(part.text)
        if lyrics:
            artifacts = [
                MediaArtifact(**{**{f: getattr(a, f) for f in a.__slots__},
                                 "provider_metadata": {"lyrics": "\n".join(lyrics)}})
                for a in artifacts
            ]
        usage = getattr(response, "usage_metadata", None)
        return MediaResult(
            capability=MediaCapability.MUSIC,
            provider=self.get_name(),
            model=music_model,
            artifacts=tuple(artifacts),
            usage=MediaUsage(
                provider=self.get_name(), model=music_model, basis="per_token",
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            ),
        )

    # --- Live API sessions ---

    @asynccontextmanager
    async def open_transcription_session(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        sample_rate_hz: int | None = None,
        **kwargs: Any,
    ) -> "AsyncIterator[GeminiTranscriptionStream]":
        """Realtime transcription over the Live API (``asr_stream``).

        ``gemini-3.5-transcribe-live`` streams ``interim`` hypotheses and a
        ``final`` per utterance (live 2026-11-16). Send 16-bit mono PCM at
        ``sample_rate_hz`` (default 16000); ``finalize()`` ends the audio
        stream. ``language`` is accepted for parity and passed as a hint.
        """
        from google.genai import types as genai_types

        used = model or "gemini-3.5-transcribe-live"
        config = genai_types.LiveConnectConfig(input_audio_transcription={}, **kwargs)
        if language:
            logger.debug("Gemini Live transcription detects the language itself; hint %s noted.", language)
        try:
            async with self._client.aio.live.connect(model=used, config=config) as session:
                yield GeminiTranscriptionStream(session, rate=int(sample_rate_hz or 16000))
        except ProviderError:
            raise
        except Exception as e:
            self._raise_media_error(e, used, "live transcription")

    @asynccontextmanager
    async def open_voice_agent(
        self,
        *,
        model: str | None = None,
        prompt: str | None = None,
        functions: list[Any] | None = None,
        greeting: str | None = None,
        voice: str | None = None,
        transcribe_user: bool = True,
        settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "AsyncIterator[GeminiVoiceAgentSession]":
        """A Live API speech-to-speech agent (``voice_agent``).

        Same session surface as the Deepgram and OpenAI agents. Microphone
        audio is 16-bit mono PCM at 16 kHz; the agent answers in 24 kHz PCM.
        Live sessions fix the system instruction at connect time, so
        ``update_prompt`` raises. Default model ``gemini-3.8-live``.
        """
        from google.genai import types as genai_types

        used = model or "gemini-3.8-live"
        options: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": {},
        }
        if transcribe_user:
            options["input_audio_transcription"] = {}
        if prompt:
            options["system_instruction"] = prompt
        if functions:
            options["tools"] = [genai_types.Tool(function_declarations=[
                genai_types.FunctionDeclaration(**_live_function(f)) for f in functions
            ])]
        if voice:
            options["speech_config"] = genai_types.SpeechConfig(voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)))
        options.update(kwargs)
        options.update(settings or {})
        try:
            async with self._client.aio.live.connect(
                model=used, config=genai_types.LiveConnectConfig(**options)
            ) as session:
                agent = GeminiVoiceAgentSession(session)
                if greeting:
                    await agent.inject_agent_message(greeting)
                yield agent
        except ProviderError:
            raise
        except Exception as e:
            self._raise_media_error(e, used, "live voice agent")

    # --- video (async job) ---

    async def generate_video_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        first_frame: "MediaRef | None" = None,
        last_frame: "MediaRef | None" = None,
        reference_images: "Sequence[MediaRef] | None" = None,
        duration_seconds: float | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        fps: float | None = None,
        with_audio: bool | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Submit a Veo video generation and return a job handle.

        ``first_frame`` conditions the opening image; ``last_frame`` requests a
        *generative transition* to that image — distinct from frame
        interpolation, which fills between existing frames.
        """
        from google.genai import types as genai_types

        from ..media.models import MediaCapability, MediaJob

        video_model = self._media_model("video", model)
        config_kwargs: dict[str, Any] = {}
        if duration_seconds is not None:
            config_kwargs["duration_seconds"] = int(duration_seconds)
        if aspect_ratio is not None:
            config_kwargs["aspect_ratio"] = aspect_ratio
        if fps is not None:
            config_kwargs["fps"] = int(fps)
        if with_audio is not None:
            config_kwargs["generate_audio"] = with_audio
        if last_frame is not None:
            config_kwargs["last_frame"] = await self._to_genai_image(last_frame)
        if reference_images:
            config_kwargs["reference_images"] = [
                await self._to_genai_image(ref) for ref in reference_images
            ]
        if resolution is not None:
            config_kwargs.setdefault("resolution", resolution)
        if seed is not None:
            logger.debug("Veo does not expose a seed parameter; ignoring it.")
        config_kwargs.update(kwargs)

        # ``prompt=``/``image=`` are deprecated in google-genai (removal no
        # earlier than 2026-07-31); ``source=`` is the supported shape.
        source_kwargs: dict[str, Any] = {"prompt": prompt}
        if first_frame is not None:
            source_kwargs["image"] = await self._to_genai_image(first_frame)
        call_kwargs: dict[str, Any] = {
            "model": video_model,
            "source": genai_types.GenerateVideosSource(**source_kwargs),
        }
        if config_kwargs:
            call_kwargs["config"] = genai_types.GenerateVideosConfig(**config_kwargs)

        try:
            operation = await self._client.aio.models.generate_videos(**call_kwargs)
        except Exception as e:
            self._raise_media_error(e, video_model, "video generation")

        job = MediaJob(
            capability=MediaCapability.VIDEO_GENERATE,
            provider=self.get_name(),
            model=video_model,
            provider_job_id=getattr(operation, "name", None),
        )
        return self._apply_video_operation(job, operation)

    def _apply_video_operation(self, job: "MediaJob", operation: Any) -> "MediaJob":
        """Fold a Veo operation's state into *job*.

        The live operation object is kept in ``provider_metadata`` because the
        SDK's ``operations.get()`` takes the object, not just its name.
        """
        import hashlib

        from ..media.models import (
            MediaArtifact,
            MediaJobStatus,
            MediaKind,
            MediaProvenance,
            MediaUsage,
        )

        job.provider_metadata["operation"] = operation
        job.provider_job_id = getattr(operation, "name", None) or job.provider_job_id
        job.touch()

        error = getattr(operation, "error", None)
        if error:
            job.status = MediaJobStatus.FAILED
            job.error = str(getattr(error, "message", None) or error)
            return job

        if not getattr(operation, "done", False):
            job.status = MediaJobStatus.RUNNING
            return job

        response = getattr(operation, "response", None) or getattr(operation, "result", None)
        videos = getattr(response, "generated_videos", None) or []
        artifacts: list[MediaArtifact] = []
        for item in videos:
            video = getattr(item, "video", None)
            if video is None:
                continue
            data = getattr(video, "video_bytes", None)
            artifacts.append(
                MediaArtifact(
                    kind=MediaKind.VIDEO,
                    data=data,
                    uri=getattr(video, "uri", None),
                    mime_type=getattr(video, "mime_type", None) or "video/mp4",
                    checksum_sha256=hashlib.sha256(data).hexdigest() if data else None,
                    provenance=MediaProvenance(
                        watermarked=True, generator="google", provider_declared=True
                    ),
                )
            )

        job.artifacts = artifacts
        job.progress = 1.0
        job.status = MediaJobStatus.SUCCEEDED
        job.usage = MediaUsage(
            provider=self.get_name(),
            model=job.model,
            basis="per_video",
            seconds=artifacts[0].duration_seconds if artifacts else None,
        )
        return job

    async def poll_media_job(self, job: "MediaJob") -> "MediaJob":
        """Refresh a Veo job against the long-running-operations API."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        operation = job.provider_metadata.get("operation")
        if operation is None:
            job.status = MediaJobStatus.FAILED
            job.error = "Lost the Veo operation handle; the job cannot be polled."
            return job
        try:
            refreshed = await self._client.aio.operations.get(operation)
        except Exception as e:
            self._raise_media_error(e, job.model, "video job polling")
        return self._apply_video_operation(job, refreshed)

    async def cancel_media_job(self, job: "MediaJob") -> "MediaJob":
        """Veo exposes no cancellation, so report that honestly.

        Returning a falsely-cancelled job would let a caller believe billing
        stopped when it has not.
        """
        raise ProviderError(
            self.get_name(),
            "Veo video operations cannot be cancelled once submitted; the job "
            "will run to completion and be billed.",
            model_name=job.model,
        )

    # --- embeddings ---

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create text embeddings via ``models.embed_content``.

        Args:
            input_texts: A string or list of strings to embed.
            model: Embedding model; defaults to ``gemini-embedding-2``.
            dimensions: Output dimensionality, where the model supports it.
            task_type: Google's retrieval task hint (e.g.
                ``RETRIEVAL_DOCUMENT``), which materially changes the vectors.
            **kwargs: Extra config fields.

        Returns:
            An OpenAI-shaped dict (``data``/``model``/``usage``) so callers can
            treat embeddings uniformly across providers.
        """
        from google.genai import types as genai_types

        embed_model = self._media_model("embed", model)
        config_kwargs: dict[str, Any] = dict(kwargs)
        if dimensions is not None:
            config_kwargs["output_dimensionality"] = dimensions
        if task_type is not None:
            config_kwargs["task_type"] = task_type

        texts = [input_texts] if isinstance(input_texts, str) else list(input_texts)
        # One Content per text. gemini-embedding-2 is multimodal and reads a
        # plain list of strings as the parts of ONE input, returning a single
        # aggregated vector for N texts (measured live 2026-11-16).
        contents = [genai_types.Content(parts=[genai_types.Part(text=t)]) for t in texts]
        try:
            response = await self._client.aio.models.embed_content(
                model=embed_model,
                contents=contents,
                config=genai_types.EmbedContentConfig(**config_kwargs)
                if config_kwargs
                else None,
            )
        except Exception as e:
            self._raise_media_error(e, embed_model, "embeddings")

        return {
            "object": "list",
            "model": embed_model,
            "data": [
                {"object": "embedding", "index": i, "embedding": list(item.values or [])}
                for i, item in enumerate(getattr(response, "embeddings", None) or [])
            ],
            "usage": {},
        }

    def _raise_media_error(self, error: Exception, model: str, operation: str) -> None:
        """Map a google-genai media failure onto a ProviderError.

        Raises:
            ProviderError: Always.
        """
        if isinstance(error, ProviderError):
            raise error
        logger.error("Gemini %s failed on %s: %s", operation, model, error, exc_info=True)
        raise ProviderError(
            self.get_name(), f"{operation.capitalize()} failed: {error}", model_name=model
        )

    async def close(self) -> None:
        """Close the google-genai client.

        The google-genai Client supports ``aclose()`` for proper cleanup.
        """
        if self._client:
            try:
                aio = getattr(self._client, "aio", None)
                if aio and hasattr(aio, "aclose"):
                    await aio.aclose()
            except Exception as e:
                logger.debug(f"Error during Gemini client cleanup: {e}")
        logger.debug("GeminiProvider closed.")
        self._client = None


def _live_function(function: Any) -> dict[str, Any]:
    """FunctionDeclaration kwargs from an llmcore Tool or a plain dict."""
    if isinstance(function, dict):
        data = dict(function.get("function") or function)
    else:
        params = getattr(function, "parameters", None)
        data = {
            "name": getattr(function, "name", None),
            "description": getattr(function, "description", None),
            "parameters": params.model_dump(exclude_none=True) if hasattr(params, "model_dump") else params,
        }
    return {
        "name": data.get("name"),
        "description": data.get("description") or "",
        "parameters_json_schema": data.get("parameters") or {"type": "object", "properties": {}},
    }


async def _live_messages(session: Any) -> Any:
    """Every server message of a Live session.

    ``AsyncSession.receive()`` stops after each model turn; a session that
    yields nothing on a fresh call has closed.
    """
    while True:
        got_any = False
        async for message in session.receive():
            got_any = True
            yield message
        if not got_any:
            return


class GeminiTranscriptionStream:
    """Live API transcription session (Deepgram/ElevenLabs/OpenAI surface)."""

    def __init__(self, session: Any, *, rate: int = 16000) -> None:
        self._session = session
        self._mime = f"audio/pcm;rate={rate}"

    async def send_audio(self, chunk: bytes) -> None:
        """16-bit mono PCM at the session's rate."""
        from google.genai import types as genai_types

        await self._session.send_realtime_input(audio=genai_types.Blob(data=bytes(chunk), mime_type=self._mime))

    async def finalize(self) -> None:
        """End the audio stream so the last utterance is finalized."""
        await self._session.send_realtime_input(audio_stream_end=True)

    commit = finalize

    async def keepalive(self) -> None:
        """No-op: Live sessions need no keepalive."""

    async def close(self) -> None:
        """Closed by the context manager that opened the session."""

    async def __aiter__(self) -> "AsyncIterator[Any]":
        from ..models_multimodal import StreamEventType, TranscriptionStreamEvent

        async for message in _live_messages(self._session):
            content = getattr(message, "server_content", None)
            if content is None:
                continue
            raw = message.model_dump(exclude_none=True, mode="json") if hasattr(message, "model_dump") else {}
            interim = getattr(content, "interim_input_transcription", None)
            if interim is not None and getattr(interim, "text", None):
                yield TranscriptionStreamEvent(type=StreamEventType.INTERIM, text=interim.text,
                                               is_final=False, provider="gemini", raw=raw)
            final = getattr(content, "input_transcription", None)
            if final is not None and getattr(final, "text", None):
                yield TranscriptionStreamEvent(type=StreamEventType.FINAL, text=final.text.strip(),
                                               is_final=True, speech_final=True, provider="gemini", raw=raw)


class GeminiVoiceAgentSession:
    """Live API speech-to-speech agent with the shared voice-agent surface."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._call_names: dict[str, str] = {}
        self._said: list[str] = []
        self._heard: list[str] = []
        self._speaking = False

    async def send_audio(self, chunk: bytes) -> None:
        """Microphone audio: 16-bit mono PCM at 16 kHz."""
        from google.genai import types as genai_types

        await self._session.send_realtime_input(
            audio=genai_types.Blob(data=bytes(chunk), mime_type="audio/pcm;rate=16000"))

    async def inject_user_message(self, content: str) -> None:
        """Add a typed user turn; the agent replies."""
        from google.genai import types as genai_types

        await self._session.send_client_content(
            turns=genai_types.Content(role="user", parts=[genai_types.Part(text=content)]), turn_complete=True)

    async def inject_agent_message(self, message: str) -> None:
        """Have the agent say *message* now."""
        await self.inject_user_message(f"Say exactly this, and nothing else: {message}")

    async def update_prompt(self, prompt: str) -> None:
        """Not available: Live fixes the system instruction at connect time."""
        raise ProviderError(
            "gemini", "Gemini Live sessions cannot change their system instruction; open a new session.",
            retryable=False,
        )

    async def respond_to_function_call(self, function_call_id: str, output: Any, name: str | None = None) -> None:
        """Return a tool result; the agent continues its turn."""
        from google.genai import types as genai_types

        response = output if isinstance(output, dict) else {"result": output}
        await self._session.send_tool_response(function_responses=[genai_types.FunctionResponse(
            id=function_call_id, name=name or self._call_names.get(function_call_id, ""), response=response)])

    async def keepalive(self) -> None:
        """No-op: Live sessions need no keepalive."""

    async def close(self) -> None:
        """Closed by the context manager that opened the session."""

    async def __aiter__(self) -> "AsyncIterator[Any]":
        from ..models_multimodal import VoiceAgentEvent, VoiceAgentEventType, VoiceAgentFunctionCall

        def event(type_: Any, **fields: Any) -> Any:
            return VoiceAgentEvent(type=type_, provider="gemini", **fields)

        async for message in _live_messages(self._session):
            if getattr(message, "setup_complete", None) is not None:
                yield event(VoiceAgentEventType.WELCOME)
            tool_call = getattr(message, "tool_call", None)
            for call in getattr(tool_call, "function_calls", None) or []:
                self._call_names[str(call.id)] = str(call.name)
                yield event(VoiceAgentEventType.FUNCTION_CALL_REQUEST, function_call=VoiceAgentFunctionCall(
                    id=str(call.id or ""), name=str(call.name or ""), arguments=dict(call.args or {})))
            content = getattr(message, "server_content", None)
            if content is None:
                continue
            heard = getattr(content, "input_transcription", None)
            if heard is not None and getattr(heard, "text", None):
                self._heard.append(heard.text)
            if getattr(content, "interrupted", None):
                yield event(VoiceAgentEventType.USER_STARTED_SPEAKING)
            for part in getattr(getattr(content, "model_turn", None), "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    if self._heard:
                        yield event(VoiceAgentEventType.CONVERSATION_TEXT, role="user", content="".join(self._heard).strip())
                        self._heard = []
                    if not self._speaking:
                        self._speaking = True
                        yield event(VoiceAgentEventType.AGENT_STARTED_SPEAKING)
                    yield event(VoiceAgentEventType.AUDIO, audio=inline.data)
            said = getattr(content, "output_transcription", None)
            if said is not None and getattr(said, "text", None):
                self._said.append(said.text)
            if getattr(content, "turn_complete", None):
                if self._speaking:
                    self._speaking = False
                    yield event(VoiceAgentEventType.AGENT_AUDIO_DONE)
                if self._said:
                    yield event(VoiceAgentEventType.CONVERSATION_TEXT, role="assistant", content="".join(self._said).strip())
                    self._said = []
