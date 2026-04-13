# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
Accepts context as List[Message].

Tested against ollama-python SDK v0.6.1.

Supports:
- Chat completion (text, streaming)
- Tool/function calling with normalized extraction
- Multimodal content (vision: images via metadata["inline_images"])
- Thinking mode (think parameter with level support)
- Structured output (format parameter: 'json' or JSON Schema)
- Logprobs
- keep_alive control
- Dynamic model discovery with capabilities detection
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ChatResponse, ResponseError, ShowResponse

    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None  # type: ignore
    ResponseError = Exception  # type: ignore
    ChatResponse = None  # type: ignore
    ShowResponse = None  # type: ignore

# Keep tiktoken for token counting
try:
    import tiktoken

    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool, ToolCall
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Ollama models.
# Used as first-tier fallback before model card registry and 4096 default.
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8000,
    "llama3:8b": 8000,
    "llama3:70b": 8000,
    "llama3.2": 131072,
    "llama3.2:latest": 131072,
    "llama3.3": 131072,
    "llama3.3:70b": 131072,
    "qwen3": 262144,
    "qwen3:4b": 262144,
    "qwen3:8b": 262144,
    "qwen3:14b": 262144,
    "qwen3:32b": 262144,
    "qwen2.5": 131072,
    "qwen2.5:latest": 131072,
    "gemma3": 128000,
    "gemma3:4b": 128000,
    "gemma3:12b": 128000,
    "gemma3:27b": 128000,
    "falcon3:3b": 8000,
    "gemma:latest": 8000,
    "gemma:7b": 8000,
    "gemma:2b": 8000,
    "mistral": 8000,
    "mistral:7b": 8000,
    "mixtral": 32000,
    "mixtral:8x7b": 32000,
    "phi3": 4000,
    "phi3:mini": 4000,
    "phi4": 16384,
    "phi4:latest": 16384,
    "deepseek-r1": 131072,
    "deepseek-r1:7b": 131072,
    "deepseek-r1:8b": 131072,
    "deepseek-r1:14b": 131072,
    "codellama": 16000,
    "codellama:7b": 16000,
    "codellama:13b": 16000,
    "codellama:34b": 16000,
    "llama2": 4000,
    "llama2:7b": 4000,
    "llama2:13b": 4000,
    "llama2:70b": 4000,
}
DEFAULT_MODEL = "gemma3:4b"

# Top-level chat parameters that go directly into the chat() call,
# NOT inside the ``options`` dict.  Everything else goes into ``options``.
_TOP_LEVEL_CHAT_PARAMS = frozenset(
    {
        "format",
        "keep_alive",
        "think",
        "logprobs",
        "top_logprobs",
    }
)


class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama using the official ollama library.
    Handles List[Message] context type and standardized tool-calling.
    """

    _client: AsyncClient | None = None
    _encoding: Any | None = None
    tokenizer_name: str

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the OllamaProvider using the official ollama library.

        Args:
            config: Configuration dictionary from ``[providers.ollama]`` containing:
                    'host' (optional): Host URL for the Ollama server.
                    'default_model' (optional): Default Ollama model to use.
                    'timeout' (optional): Request timeout in seconds.
                    'tokenizer' (optional): Tokenizer to use for estimations.
                    'keep_alive' (optional): Default keep_alive duration for requests.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not ollama_available:
            raise ImportError(
                "Ollama library not installed. Please install `ollama` or `llmcore[ollama]`."
            )
        if not tiktoken_available:
            logger.warning("tiktoken library not available. Token counting will use approximation.")

        self.host = config.get("host")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        timeout_val = config.get("timeout")
        self.timeout = float(timeout_val) if timeout_val is not None else None
        self.default_keep_alive = config.get("keep_alive")

        try:
            client_args: dict[str, Any] = {}
            if self.host:
                client_args["host"] = self.host
            if self.timeout is not None:
                client_args["timeout"] = self.timeout
            self._client = AsyncClient(**client_args)
            logger.debug("Ollama AsyncClient initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama AsyncClient: {e}", exc_info=True)
            raise ConfigError(f"Ollama client initialization failed: {e}")

        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base")
        self._encoding = None
        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken_available and tiktoken:
                try:
                    encoding_name = self.tokenizer_name.split("tiktoken_")[1]
                    self._encoding = tiktoken.get_encoding(encoding_name)
                    logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name}.")
                except Exception as e:
                    logger.warning(
                        f"Failed to load tiktoken encoding '{self.tokenizer_name}'. "
                        f"Falling back to approximation. Error: {e}"
                    )
                    self.tokenizer_name = "char_div_4"
            else:
                logger.warning("tiktoken not available. Falling back to character approximation.")
                self.tokenizer_name = "char_div_4"
        elif self.tokenizer_name != "char_div_4":
            logger.warning(
                f"Unsupported tokenizer '{self.tokenizer_name}'. Falling back to approximation."
            )
            self.tokenizer_name = "char_div_4"

        if self.tokenizer_name == "char_div_4":
            logger.info("OllamaProvider using character division for token counting.")

    def get_name(self) -> str:
        """Returns the provider instance name."""
        return self._provider_instance_name or "ollama"

    async def get_models_details(self) -> list[ModelDetails]:
        """
        Asynchronously discovers and returns detailed information about available
        local models by querying the Ollama API.

        Uses ``client.list()`` for the model inventory and, where possible,
        ``client.show()`` to obtain per-model capability flags (tools, vision,
        thinking).
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")
        try:
            list_response = await self._client.list()
            # ListResponse.models is Sequence[ListResponse.Model]
            # Each model has: .model, .size, .digest, .modified_at, .details
            model_list = list_response.models if list_response else []

            details_list = []
            for model_entry in model_list:
                # The model identifier is in the `model` attribute, not `name`.
                model_name = model_entry.model
                if not model_name:
                    continue

                # Extract rich details from the ListResponse.Model.details
                entry_details = model_entry.details
                family = getattr(entry_details, "family", None) if entry_details else None
                families = getattr(entry_details, "families", None) if entry_details else None
                parameter_size = (
                    getattr(entry_details, "parameter_size", None) if entry_details else None
                )
                quantization = (
                    getattr(entry_details, "quantization_level", None) if entry_details else None
                )
                file_size = getattr(model_entry, "size", None)

                # Default capability flags — will be refined via show() if possible
                supports_tools = False
                supports_vision = False
                supports_reasoning = False

                # Attempt to get capabilities from show() API (non-blocking)
                try:
                    show_response = await self._client.show(model_name)
                    capabilities = getattr(show_response, "capabilities", None) or []
                    if capabilities:
                        supports_tools = "tools" in capabilities
                        supports_vision = "vision" in capabilities
                        supports_reasoning = "thinking" in capabilities
                except Exception as e:
                    logger.debug(
                        f"Could not fetch capabilities for '{model_name}': {e}. Using defaults."
                    )

                details = ModelDetails(
                    id=model_name,
                    context_length=self.get_max_context_length(model_name),
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    supports_reasoning=supports_reasoning,
                    provider_name=self.get_name(),
                    family=family,
                    parameter_count=parameter_size,
                    quantization_level=quantization,
                    file_size_bytes=int(file_size) if file_size is not None else None,
                    metadata={
                        "families": families,
                        "digest": getattr(model_entry, "digest", None),
                    },
                )
                details_list.append(details)

            logger.info(f"Discovered {len(details_list)} local Ollama models.")
            return details_list
        except ResponseError as e:
            logger.error(
                f"Ollama API error fetching models: HTTP {e.status_code} - {e.error}",
                exc_info=True,
            )
            raise ProviderError(
                self.get_name(), f"Failed to fetch models from Ollama API: {e.error}"
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error fetching models: {e}")

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """
        Returns a schema of supported inference parameters for Ollama models.

        Covers both the ``options`` dict (sampling, runtime) and top-level
        chat parameters (format, think, logprobs, keep_alive).
        """
        return {
            # --- Sampling / runtime options ---
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer"},
            "num_ctx": {"type": "integer"},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "mirostat": {"type": "integer"},
            "mirostat_tau": {"type": "number"},
            "mirostat_eta": {"type": "number"},
            "num_predict": {"type": "integer"},
            "max_tokens": {"type": "integer", "note": "alias for num_predict"},
            "presence_penalty": {"type": "number"},
            "frequency_penalty": {"type": "number"},
            "repeat_penalty": {"type": "number"},
            "repeat_last_n": {"type": "integer"},
            "tfs_z": {"type": "number"},
            "typical_p": {"type": "number"},
            "penalize_newline": {"type": "boolean"},
            "num_batch": {"type": "integer"},
            "num_gpu": {"type": "integer"},
            "num_thread": {"type": "integer"},
            "num_keep": {"type": "integer"},
            "low_vram": {"type": "boolean"},
            "use_mmap": {"type": "boolean"},
            "use_mlock": {"type": "boolean"},
            # --- Top-level chat parameters ---
            "format": {"type": ["string", "object"], "note": "'json' or JSON Schema"},
            "keep_alive": {"type": ["number", "string"]},
            "think": {"type": ["boolean", "string"], "note": "true or 'low'/'medium'/'high'"},
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer"},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Returns the maximum context length (tokens) for the given Ollama model.

        Resolution order:
            1. Hardcoded DEFAULT_OLLAMA_TOKEN_LIMITS (fast, known models)
            2. Model card registry (covers newer models like qwen3)
            3. Fallback to 4096
        """
        model_name = model or self.default_model
        base_model_name = model_name.split(":")[0]
        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(
            model_name, DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)
        )
        if limit is not None:
            return limit

        # Try the model card registry — it has context info for models
        # not in the hardcoded dict (e.g. qwen3, llama3.2, etc.)
        try:
            from ..model_cards import get_model_card_registry

            registry = get_model_card_registry()
            card_limit = registry.get_context_length(
                "ollama",
                model_name,
                default=0,
            )
            if card_limit > 0:
                logger.debug(
                    f"Context length for Ollama model '{model_name}' from model card: {card_limit}"
                )
                return card_limit
        except Exception as e:
            logger.warning(f"Model card registry lookup failed for '{model_name}': {e}")

        limit = 4096
        logger.warning(
            f"Unknown context length for Ollama model '{model_name}'. Using fallback: {limit}."
        )
        return limit

    # ------------------------------------------------------------------
    # Message Conversion
    # ------------------------------------------------------------------

    def _build_message_payload(self, msg: Message) -> dict[str, Any]:
        """Convert a single llmcore ``Message`` to an Ollama message dict.

        Handles:
        - Standard text messages (system, user, assistant)
        - Tool-result messages (role=tool → includes ``tool_name``)
        - Multimodal images via ``metadata["inline_images"]``

        Image convention (same as Gemini provider):
        - ``metadata["inline_images"]``: list of image sources.  Each element
          may be a ``str`` (file path or base64), ``bytes``, or a dict with
          ``{"data": ..., "mime_type": ...}`` where ``data`` is base64 or bytes.
        """
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        payload: dict[str, Any] = {
            "role": role_str,
            "content": msg.content,
        }

        # Tool-result messages need the function name so the model can
        # correlate the result with its earlier tool_call.
        if role_str == "tool":
            tool_name = msg.metadata.get("tool_name") or msg.metadata.get("name")
            if tool_name:
                payload["tool_name"] = tool_name

        # Multimodal image support
        images = msg.metadata.get("inline_images")
        if images:
            payload["images"] = self._extract_image_values(images)

        return payload

    @staticmethod
    def _extract_image_values(images: list[Any]) -> list[Any]:
        """Normalise a list of image sources into values the SDK accepts.

        The ollama SDK's ``Image`` type serializer accepts:
        - ``str``: base64 string **or** file path
        - ``bytes``: raw image bytes
        - ``pathlib.Path``: file path

        We accept those plus a dict with a ``data`` key for compatibility
        with the Gemini provider's ``inline_images`` convention.
        """
        out: list[Any] = []
        for img in images:
            if isinstance(img, dict):
                # Dict convention: {"data": base64_or_bytes, "mime_type": "image/png"}
                out.append(img.get("data", img))
            else:
                # str (base64 or path) / bytes / Path — pass through to SDK
                out.append(img)
        return out

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
        """
        Sends a chat completion request to the Ollama API.

        Supports tools, thinking mode, structured output (``format``),
        multimodal images, logprobs, and ``keep_alive``.

        Keyword arguments are routed to the correct location:
        - ``format``, ``keep_alive``, ``think``, ``logprobs``, ``top_logprobs``
          go as top-level params to ``client.chat()``.
        - Everything else goes into the ``options`` dict.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")

        model_name = model or self.default_model
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        messages_payload = [self._build_message_payload(msg) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages to send.")

        # --- Alias: max_tokens → num_predict ---
        if "max_tokens" in kwargs:
            kwargs["num_predict"] = kwargs.pop("max_tokens")

        # --- Route kwargs into top-level params vs options dict ---
        top_level_kwargs: dict[str, Any] = {}
        options_kwargs: dict[str, Any] = {}
        supported_params = self.get_supported_parameters()

        for key, value in kwargs.items():
            if key in _TOP_LEVEL_CHAT_PARAMS:
                top_level_kwargs[key] = value
            elif key in supported_params:
                options_kwargs[key] = value
            else:
                logger.warning(
                    f"Parameter '{key}' not in supported parameters for Ollama. "
                    f"Passing through as option anyway."
                )
                options_kwargs[key] = value

        # Apply default keep_alive if configured and not explicitly set
        if "keep_alive" not in top_level_kwargs and self.default_keep_alive is not None:
            top_level_kwargs["keep_alive"] = self.default_keep_alive

        # --- Prepare tools for the ollama library ---
        tools_payload: list[dict[str, Any]] | None = None
        if tools:
            logger.debug(f"Preparing {len(tools)} tools for Ollama request.")
            tools_payload = [
                {
                    "type": "function",
                    "function": tool.model_dump(),
                }
                for tool in tools
            ]

        # --- Logging ---
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
                "options": options_kwargs,
                "tools": tools_payload,
                "tool_choice": tool_choice,
                **top_level_kwargs,
            }
            logger.debug(
                f"RAW LLM REQUEST ({self.get_name()} @ {model_name}): "
                f"{json.dumps(log_data, indent=2, default=str)}"
            )

        try:
            # Build the call kwargs
            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
            }
            if tools_payload:
                call_kwargs["tools"] = tools_payload
            if options_kwargs:
                call_kwargs["options"] = options_kwargs

            # Add top-level params
            for key, value in top_level_kwargs.items():
                call_kwargs[key] = value

            response_or_stream = await self._client.chat(**call_kwargs)

            if stream:
                return self._wrap_stream(response_or_stream)
            else:
                return self._normalize_response(response_or_stream)

        except ResponseError as e:
            error_detail = e.error if hasattr(e, "error") and e.error else str(e)
            logger.error(f"Ollama API error: HTTP {e.status_code} - {error_detail}", exc_info=True)
            if e.status_code == 404:
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found. Pull it with `ollama pull {model_name}`.",
                )
            raise ProviderError(
                self.get_name(), f"Ollama API Error (HTTP {e.status_code}): {error_detail}"
            )
        except ConnectionError as e:
            logger.error(f"Cannot connect to Ollama server: {e}", exc_info=True)
            raise ProviderError(
                self.get_name(),
                f"Cannot connect to Ollama server at {self.host or 'localhost:11434'}. "
                f"Is Ollama running?",
            )
        except Exception as e:
            logger.error(f"Unexpected error during Ollama chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    # ------------------------------------------------------------------
    # Response Normalization
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> dict[str, Any]:
        """Convert a non-streaming ``ChatResponse`` to a plain dict.

        The Ollama SDK returns ``ChatResponse`` Pydantic models.  We
        normalize to a consistent dict format for downstream consumption.

        The Ollama native format is preserved (``message.content``,
        ``message.tool_calls``, ``message.thinking``), keeping it distinct
        from the OpenAI choices-based format used by OpenAI/Gemini providers.
        The ``extract_*`` methods abstract the difference.
        """
        if isinstance(response, dict):
            response_dict = response
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "__dict__"):
            response_dict = dict(response.__dict__)
        else:
            raise ProviderError(self.get_name(), "Invalid non-streaming response format.")

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"RAW LLM RESPONSE ({self.get_name()}): "
                f"{json.dumps(response_dict, indent=2, default=str)}"
            )
        return response_dict

    async def _wrap_stream(self, stream: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Wrap the SDK's async stream, yielding normalized dicts.

        Each ``ChatResponse`` chunk is converted to a dict via
        ``model_dump()`` so that all downstream consumers see dicts,
        matching the ``AsyncGenerator[dict, None]`` return type contract.
        """

        async def _inner() -> AsyncGenerator[dict[str, Any], None]:
            async for chunk in stream:
                # Normalize to dict
                if hasattr(chunk, "model_dump"):
                    chunk_dict = chunk.model_dump()
                elif isinstance(chunk, dict):
                    chunk_dict = chunk
                else:
                    chunk_dict = {"message": {"content": str(chunk)}}

                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RAW LLM STREAM CHUNK ({self.get_name()}): "
                        f"{json.dumps(chunk_dict, default=str)}"
                    )
                yield chunk_dict

        return _inner()

    # ------------------------------------------------------------------
    # Content Extraction
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """
        Extract text content from Ollama non-streaming response.

        Ollama /api/chat response format::

            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": true,
                ...
            }

        Args:
            response: The raw response dictionary from chat_completion().

        Returns:
            The extracted text content.
        """
        try:
            message = response.get("message", {})
            if isinstance(message, dict):
                return message.get("content") or ""
            return ""
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to extract content from Ollama response: {e}")
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """
        Extract text delta from Ollama streaming chunk.

        Ollama streaming format::

            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "The"},
                "done": false
            }

        Each chunk contains incremental content.  The final chunk has
        ``"done": true`` and may have empty content.

        Args:
            chunk: A single streaming chunk dictionary.

        Returns:
            The extracted text delta.
        """
        try:
            if isinstance(chunk, dict):
                message = chunk.get("message", {})
            elif hasattr(chunk, "message"):
                # Handle Pydantic ChatResponse model case (shouldn't happen
                # after _wrap_stream normalization, but kept for safety)
                msg_attr = chunk.message
                if isinstance(msg_attr, dict):
                    message = msg_attr
                elif hasattr(msg_attr, "content"):
                    return msg_attr.content or ""
                else:
                    return ""
            else:
                return ""

            if isinstance(message, dict):
                return message.get("content") or ""
            return ""
        except (KeyError, TypeError, AttributeError):
            return ""

    def extract_thinking_content(self, response: dict[str, Any]) -> str | None:
        """Extract thinking/reasoning content from a non-streaming response.

        When ``think=True`` (or a thinking level) is passed, the model
        populates ``message.thinking`` with its chain-of-thought.

        Args:
            response: The raw response dict from chat_completion().

        Returns:
            The thinking text, or None if not present.
        """
        try:
            message = response.get("message", {})
            if isinstance(message, dict):
                thinking = message.get("thinking")
                return thinking if thinking else None
            return None
        except (KeyError, TypeError):
            return None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an Ollama non-streaming response.

        Ollama tool call format in ``message``::

            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"}
                        }
                    }
                ]
            }

        Note: Ollama tool calls do **not** have an ``id`` field.  We
        generate a UUID to satisfy the ``ToolCall`` model contract.

        Args:
            response: The raw response dict from chat_completion().

        Returns:
            List of ToolCall objects. Empty list if no tool calls.
        """
        tool_calls_out: list[ToolCall] = []
        try:
            message = response.get("message", {})
            if not isinstance(message, dict):
                return tool_calls_out
            raw_calls = message.get("tool_calls")
            if not raw_calls:
                return tool_calls_out
            for tc in raw_calls:
                func = tc.get("function", {})
                arguments = func.get("arguments", {})
                # Arguments from Ollama are already a dict (not JSON string)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                tool_calls_out.append(
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=func.get("name", "unknown"),
                        arguments=arguments,
                    )
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract tool calls from Ollama response: {e}")
        return tool_calls_out

    def extract_usage(self, response: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from Ollama response.

        Ollama reports usage as durations and counts directly on the
        response object, not in a nested ``usage`` dict like OpenAI.

        Maps to OpenAI-compatible field names for uniformity.

        Args:
            response: The raw response dict.

        Returns:
            Dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
        """
        prompt_tokens = response.get("prompt_eval_count") or 0
        completion_tokens = response.get("eval_count") or 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # Ollama-specific timing (nanoseconds)
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_duration": response.get("prompt_eval_duration"),
            "eval_duration": response.get("eval_duration"),
        }

    # ------------------------------------------------------------------
    # Token Counting
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Counts tokens using the configured tokenizer or character approximation."""
        if not self._encoding:
            return (len(text) + 3) // 4
        if not text:
            return 0
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))  # type: ignore

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Approximates token count for a list of messages using tiktoken."""
        if not self._encoding:
            total_chars = sum(
                len(msg.content)
                + len(msg.role.value if hasattr(msg.role, "value") else str(msg.role))
                for msg in messages
            )
            return (total_chars + (len(messages) * 5) + 3) // 4

        num_tokens = 0
        for message in messages:
            num_tokens += 3  # tokens_per_message overhead
            try:
                role_str = (
                    message.role.value if hasattr(message.role, "value") else str(message.role)
                )
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed for message part: {e}. Approximating.")
                role_str_fallback = (
                    message.role.value if hasattr(message.role, "value") else str(message.role)
                )
                num_tokens += (len(role_str_fallback) + len(message.content)) // 4
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Closes the underlying Ollama client session.

        The ollama SDK's ``AsyncClient`` exposes an async ``close()``
        method (inherited from ``BaseClient``) which internally calls
        ``self._client.aclose()`` on the httpx ``AsyncClient``.
        """
        if self._client:
            try:
                await self._client.close()
                logger.info("OllamaProvider client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OllamaProvider client: {e}", exc_info=True)
            finally:
                self._client = None
