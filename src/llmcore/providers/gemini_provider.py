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
from typing import Any

logger = logging.getLogger(__name__)

# --- Granular import checks for google-genai and its dependencies ---
google_genai_available = False
try:
    from google import genai
    from google.api_core.exceptions import InvalidArgument, PermissionDenied
    from google.genai import types
    from google.genai.errors import APIError

    google_genai_available = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    APIError = Exception  # type: ignore
    PermissionDenied = Exception  # type: ignore
    InvalidArgument = Exception  # type: ignore

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

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
        if not google_genai_available:
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
            logger.debug("Google Gen AI client initialized successfully.")
        except Exception as e:
            raise ConfigError(f"Google Gen AI configuration failed: {e}")

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
                    metadata["lifecycle_status"] = (
                        card.lifecycle.status
                        if isinstance(card.lifecycle.status, str)
                        else card.lifecycle.status
                    )

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
            raise ProviderError(self.get_name(), f"Failed to list models: {e}")
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
                return self._handle_streaming(
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
                )
            raise ProviderError(self.get_name(), f"Google AI API Error: {e}")
        except PermissionDenied as e:
            logger.error(
                f"Permission denied during Gemini chat: {e}", exc_info=True
            )
            raise ProviderError(self.get_name(), f"Permission denied: {e}")
        except (ContextLengthError, ProviderError):
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during Gemini chat: {e}", exc_info=True
            )
            raise ProviderError(
                self.get_name(), f"An unexpected error occurred: {e}"
            )

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
            return (len(text) + 3) // 4
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
            return (len(text) + 3) // 4

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Counts tokens for a list of messages using the Gemini API."""
        if not self._client:
            logger.warning(
                "Gemini client not available. Approximating message token count."
            )
            total_chars = sum(len(msg.content) for msg in messages)
            return (total_chars + 3 * len(messages)) // 4
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
            total_chars = sum(
                len(part.get("text", ""))
                for content_dict in genai_contents
                for part in content_dict.get("parts", [])
                if "text" in part
            )
            return (total_chars + 3 * len(genai_contents)) // 4

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
