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

import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
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
DEFAULT_GEMINI_TOKEN_LIMITS = {
    # Gemini 3.x family (preview)
    "gemini-3.1-pro-preview": 1048576,
    "gemini-3-flash-preview": 1048576,
    "gemini-3.1-flash-lite-preview": 1048576,
    "gemini-3.1-flash-image-preview": 128000,
    "gemini-3.1-flash-live-preview": 128000,
    # Gemini 2.5 family (stable, deprecates Oct 2026)
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite": 1048576,
    "gemini-2.5-flash-image": 65536,
    # Gemini 2.0 family (deprecated, shutdown June 2026)
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-lite": 1048576,
}
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

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
        from google.api_core.exceptions import InvalidArgument as _InvalidArgument
        from google.api_core.exceptions import PermissionDenied as _PermissionDenied
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

    genai = _genai
    types = _types
    APIError = _APIError
    PermissionDenied = _PermissionDenied
    InvalidArgument = _InvalidArgument
    google_genai_available = True
    return True


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
        #   default_tts_model / default_embedding_model
        self._media_model_config: dict[str, Any] = {
            "image": config.get("default_image_model"),
            "video": config.get("default_video_model"),
            "tts": config.get("default_tts_model"),
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

        last_role = None
        for msg in processed_messages:
            # Handle tool result messages -> functionResponse parts
            if msg.role == LLMCoreRole.TOOL:
                tool_name = (msg.metadata or {}).get(
                    "tool_name", msg.tool_call_id or "unknown"
                )
                function_response_part = {
                    "function_response": {
                        "name": tool_name,
                        "response": {"result": msg.content},
                    }
                }
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

            parts = self._build_multimodal_parts(msg)

            if genai_role == last_role and genai_history:
                genai_history[-1]["parts"].extend(parts)
                continue

            genai_history.append({"role": genai_role, "parts": parts})
            last_role = genai_role

        return genai_history, system_instruction_text

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
        raw = dump(exclude_none=True, by_alias=True) if callable(dump) else dict(config or {})

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
                wire.append(dump(exclude_none=True, by_alias=True))
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

        # Tool definitions
        function_declarations = (
            [
                types.FunctionDeclaration.from_dict(tool.model_dump())
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
            logger.error(f"Google AI API error: {e}", exc_info=True)
            err_str = str(e).lower()
            if "context length" in err_str or "token" in err_str:
                raise ContextLengthError(
                    model_name=model_name, message=str(e)
                ) from e
            raise ProviderError(self.get_name(), f"Google AI API Error: {e}") from e
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

        # .text property excludes thought parts
        text_content = response.text or ""

        # Extract thinking content if present
        thinking_content = None
        if response.candidates and response.candidates[0].content:
            thought_parts = [
                p.text
                for p in (response.candidates[0].content.parts or [])
                if getattr(p, "thought", False) and p.text
            ]
            if thought_parts:
                thinking_content = "".join(thought_parts)

        # Extract and normalize tool calls
        tool_calls = self._normalize_tool_calls_from_response(response)

        # Finish reason
        finish_reason = None
        if response.candidates:
            fr = response.candidates[0].finish_reason
            finish_reason = fr.name if fr else None

        # Build message dict
        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": text_content if not tool_calls else None,
        }
        if tool_calls:
            message_dict["tool_calls"] = tool_calls
        if thinking_content:
            message_dict["thinking"] = thinking_content

        # Build usage dict with OpenAI-compatible aliases
        usage_dict: dict[str, Any] = {}
        if response.usage_metadata:
            um = response.usage_metadata
            usage_dict = {
                "prompt_token_count": um.prompt_token_count,
                "candidates_token_count": um.candidates_token_count,
                "total_token_count": um.total_token_count,
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count,
                "total_tokens": um.total_token_count,
            }
            if um.thoughts_token_count:
                usage_dict["thoughts_token_count"] = um.thoughts_token_count
            if um.cached_content_token_count:
                usage_dict["cached_content_token_count"] = (
                    um.cached_content_token_count
                )

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
            async for chunk in response_stream:
                if self.log_raw_payloads_enabled:
                    logger.debug(
                        f"RAW LLM STREAM CHUNK ({self.get_name()}): {chunk}"
                    )

                text_delta = ""
                thought_delta = ""
                if chunk.candidates and chunk.candidates[0].content:
                    for part in chunk.candidates[0].content.parts or []:
                        if getattr(part, "thought", False):
                            thought_delta += part.text or ""
                        elif part.text is not None:
                            text_delta += part.text

                delta: dict[str, Any] = {"content": text_delta}
                if thought_delta:
                    delta["thinking"] = thought_delta

                yield {"choices": [{"delta": delta}]}

        return stream_wrapper()

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
    # Phase M4 of docs/MEDIA_SUBSYSTEM_SPEC.md. Gemini has the largest media
    # surface llmcore curates, and Veo makes it the FIRST true async-job
    # provider — so this is where the MediaJob lifecycle is validated against a
    # real vendor rather than the in-repo fake.
    #
    # Veo returns a google-genai ``GenerateVideosOperation``; llmcore keeps the
    # operation object in provider_metadata and refreshes it through
    # ``client.aio.operations.get()``. MediaJobManager owns the backoff,
    # timeout and cancellation policy — this adapter only reports state.

    #: Capabilities available on BOTH the Gemini Developer API and Vertex AI.
    _MEDIA_CAPABILITIES: frozenset[str] = frozenset(
        {"image_generate", "tts", "video_generate"}
    )

    #: Capabilities that exist ONLY on Vertex AI. Verified live: the Developer
    #: API rejects ``models.generate_images`` / ``edit_image`` / ``upscale_image``
    #: with "This method is only supported in Gemini Enterprise Agent Platform
    #: mode". Declaring them unconditionally would make the router call an
    #: endpoint that always errors for Developer-API users.
    _VERTEX_ONLY_MEDIA_CAPABILITIES: frozenset[str] = frozenset(
        {"image_edit", "image_upscale"}
    )

    #: Default models per media capability, overridable per call and via config.
    _DEFAULT_IMAGE_MODEL = "imagen-4.0-generate-001"        # Vertex only
    _DEFAULT_DEV_IMAGE_MODEL = "gemini-2.5-flash-image"     # Developer API
    _DEFAULT_VIDEO_MODEL = "veo-3.1-generate-preview"
    _DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
    _DEFAULT_EMBED_MODEL = "gemini-embedding-001"

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
        return frozenset(MediaCapability(c) for c in names)

    def media_execution(
        self, capability: "MediaCapability", model: str | None = None
    ) -> "MediaExecution":
        """Return how *capability* completes.

        Veo is a long-running operation; everything else answers synchronously.
        """
        from ..media.models import MediaCapability, MediaExecution

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
                prompt, model=model, n=n, **kwargs
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
            model: Embedding model; defaults to ``gemini-embedding-001``.
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

        try:
            response = await self._client.aio.models.embed_content(
                model=embed_model,
                contents=input_texts,
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
