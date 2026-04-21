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

Uses ``httpx.AsyncClient`` directly for a lightweight, zero-dependency
implementation (no ``mistralai`` SDK required).

Tested against the Mistral API as of April 2026 (OpenAPI spec v2.4.0).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
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
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default base URL for the Mistral AI API.
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Default model if none is configured.
DEFAULT_MODEL = "mistral-large-latest"

# Fallback context lengths keyed by model-id prefix (longest prefix wins).
# These are used when the model card registry has no entry and the /v1/models
# endpoint was not queried yet.
DEFAULT_MISTRAL_TOKEN_LIMITS: dict[str, int] = {
    # Frontier
    "mistral-large-latest": 262144,
    "mistral-large-3": 262144,
    "mistral-large-2512": 262144,
    "mistral-large-2411": 131072,
    "mistral-large-2407": 131072,
    # Medium
    "mistral-medium-latest": 262144,
    "mistral-medium-3": 262144,
    "mistral-medium-2508": 262144,
    "mistral-medium-2505": 262144,
    # Small
    "mistral-small-latest": 131072,
    "mistral-small-3": 131072,
    "mistral-small-2506": 131072,
    "mistral-small-2503": 131072,
    "mistral-small-2501": 131072,
    # Ministral
    "ministral-3-14b": 262144,
    "ministral-3-8b": 262144,
    "ministral-3-3b": 131072,
    "ministral-8b-latest": 131072,
    "ministral-3b-latest": 131072,
    # Magistral (reasoning)
    "magistral-medium-latest": 262144,
    "magistral-small-latest": 131072,
    "magistral-medium-2509": 262144,
    "magistral-small-2509": 131072,
    # Codestral
    "codestral-latest": 262144,
    "codestral-2508": 262144,
    "codestral-2501": 262144,
    # Devstral
    "devstral-2-latest": 262144,
    "devstral-small-latest": 262144,
    "devstral-medium-latest": 262144,
    # Pixtral
    "pixtral-large-latest": 131072,
    "pixtral-large-2411": 131072,
    "pixtral-12b-2409": 131072,
    # Voxtral (audio)
    "voxtral-mini-latest": 32768,
    "voxtral-small-latest": 32768,
    # OCR
    "mistral-ocr-latest": 0,  # OCR models don't use chat context
    "mistral-ocr-2505": 0,
    "mistral-ocr-2512": 0,
    # Moderation
    "mistral-moderation-latest": 8192,
    "mistral-moderation-2411": 8192,
    "mistral-moderation-2603": 8192,
    # Embedding
    "mistral-embed": 8192,
    "mistral-embed-2312": 8192,
    "mistral-embed-dim128-2510": 8192,
    "mistral-embed-dim256-2510": 8192,
    "codestral-embed-2505": 32768,
    "codestral-embed": 32768,
    # Nemo
    "open-mistral-nemo": 131072,
    "open-mistral-nemo-2407": 131072,
}

# Prefix-based heuristic for future model snapshots not yet in the map.
_MISTRAL_PREFIX_CONTEXT_HEURISTICS: list[tuple[str, int]] = [
    ("mistral-large", 262144),
    ("mistral-medium", 262144),
    ("mistral-small", 131072),
    ("mistral-moderation", 8192),
    ("mistral-ocr", 0),
    ("mistral-embed", 8192),
    ("ministral-3", 262144),
    ("ministral", 131072),
    ("magistral-medium", 262144),
    ("magistral-small", 131072),
    ("magistral", 131072),
    ("codestral-embed", 32768),
    ("codestral", 262144),
    ("devstral", 262144),
    ("pixtral", 131072),
    ("voxtral", 32768),
    ("open-mistral", 131072),
    ("labs-mistral", 131072),
]

# Models that support extended reasoning / thinking mode.
_MISTRAL_REASONING_MODELS = frozenset(
    {
        "magistral-medium-latest",
        "magistral-small-latest",
        "magistral-medium-2509",
        "magistral-small-2509",
        "magistral-medium-2507",
        "magistral-small-2507",
        "magistral-medium-2506",
        "magistral-small-2506",
    }
)


def _is_mistral_reasoning_model(model: str) -> bool:
    """Check whether *model* is a Mistral reasoning model (Magistral)."""
    return model in _MISTRAL_REASONING_MODELS or model.startswith("magistral")


class MistralProvider(BaseProvider):
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
        self.timeout = float(config.get("timeout", 120.0))
        self.max_retries = int(config.get("max_retries", 2))
        self._discovered_context_lengths = {}

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

            # Cache discovered context length
            self._discovered_context_lengths[mid] = ctx_len

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
                    metadata={
                        "owned_by": m.get("owned_by", "mistralai"),
                        "created": m.get("created"),
                        "type": m.get("type", "base"),
                        "capabilities": caps,
                        "aliases": m.get("aliases", []),
                        "description": m.get("description"),
                        "default_model_temperature": m.get("default_model_temperature"),
                        "deprecation": m.get("deprecation"),
                    },
                )
            )
        return result

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        model_name = model or self.default_model
        is_reasoning = _is_mistral_reasoning_model(model_name)

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
        }
        if is_reasoning:
            params["reasoning_effort"] = {
                "type": "string",
                "enum": ["low", "medium", "high"],
            }
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

        # Assistant tool calls for multi-turn
        if role_str == "assistant" and "tool_calls" in metadata:
            msg_dict["tool_calls"] = metadata["tool_calls"]
            if not msg.content:
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

        messages_payload = [self._build_message_payload(msg, model_name) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        # Build tools payload
        tools_payload = None
        if tools:
            tools_payload = [{"type": "function", "function": t.model_dump()} for t in tools]

        # Build request body
        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages_payload,
            "stream": stream,
        }

        if tools_payload:
            body["tools"] = tools_payload
        if tool_choice:
            body["tool_choice"] = tool_choice

        # Merge kwargs (Mistral-specific params like random_seed, safe_prompt, etc.)
        body.update(kwargs)

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM REQUEST (%s): %s",
                self.get_name(),
                json.dumps(body, indent=2, default=str),
            )

        try:
            if stream:
                return self._stream_completion(body, model_name)
            else:
                resp = await self._client.post("/chat/completions", json=body)
                self._raise_for_status(resp, model_name)
                response_dict = resp.json()
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW LLM RESPONSE (%s): %s",
                        self.get_name(),
                        json.dumps(response_dict, indent=2, default=str),
                    )
                return response_dict

        except ProviderError:
            raise
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}")
        except httpx.ConnectError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}")
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.get_name(),
                f"API Error ({e.response.status_code}): {e.response.text}",
            )
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error: {e}")

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
                self._raise_for_status_stream(resp, model_name)

                buffer = ""
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data_str)
                            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "RAW STREAM CHUNK (%s): %s",
                                    self.get_name(),
                                    json.dumps(chunk, default=str),
                                )
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse SSE chunk: %s", data_str[:200])

        return _generator()

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
                return self._stream_fim(body, model_name)
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
                self._raise_for_status_stream(resp, model_name)
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
            return choices[0].get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

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
        if self._encoding:
            return len(self._encoding.encode(text))
        # Rough fallback: ~4 chars per token for Mistral's tokenizer
        return len(text) // 4

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
        voice: str = "alloy",
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate speech via POST /v1/audio/speech.

        Mistral TTS supports:
        - ``voice_id``: preset or custom voice ID
        - ``ref_audio``: base64-encoded audio for zero-shot voice cloning
        - Streaming SSE response (via ``stream=True``)

        The ``voice`` parameter maps to Mistral's ``voice_id`` field.

        Args:
            text: Text to convert to speech.
            voice: Voice identifier (maps to Mistral ``voice_id``).
            model: TTS model (e.g. ``mistral-tts-latest``).
            response_format: Output format (mp3, wav, pcm, flac, opus).
            speed: Not directly supported by Mistral; ignored.
            instructions: Not supported by Mistral; ignored.
            **kwargs: Extra params (``ref_audio`` for voice cloning).

        Returns:
            ``SpeechResult`` with raw audio bytes.
        """
        from ..models_multimodal import SpeechResult

        tts_model = model or kwargs.pop("tts_model", None)

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
                voice=voice,
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
            # File path
            import pathlib

            file_path = pathlib.Path(audio_data)
            files_param = {"file": (file_path.name, open(file_path, "rb"), "audio/wav")}
        elif isinstance(audio_data, bytes):
            files_param = {"file": ("audio.wav", io.BytesIO(audio_data), "audio/wav")}

        if language:
            form_data["language"] = language
        if temperature is not None:
            form_data["temperature"] = str(temperature)
        if diarize:
            form_data["diarize"] = "true"
        if context_bias:
            form_data["context_bias"] = json.dumps(context_bias)
        if timestamp_granularities:
            form_data["timestamp_granularities"] = json.dumps(timestamp_granularities)

        logger.debug(
            "Mistral STT: model=%s, language=%s, diarize=%s",
            stt_model,
            language,
            diarize,
        )

        try:
            # Use a separate request without the default JSON content-type
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            resp = await self._client.post(
                "/audio/transcriptions",
                data=form_data,
                files=files_param,
                headers=headers,
            )
            self._raise_for_status(resp, stt_model)
            result = resp.json()

            # Close file handle if opened
            if files_param and hasattr(files_param.get("file", (None,))[1], "close"):
                try:
                    files_param["file"][1].close()
                except Exception:
                    pass

            text = result.get("text", "")
            detected_lang = result.get("language")
            segments: list[TranscriptionSegment] = []
            for seg in result.get("segments", []):
                segments.append(
                    TranscriptionSegment(
                        text=seg.get("text", ""),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        speaker=seg.get("speaker_id"),
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

    def _raise_for_status(self, resp: Any, model_name: str) -> None:
        """Raise a ProviderError with useful context on HTTP errors."""
        if resp.status_code >= 400:
            try:
                error_body = resp.json()
                error_msg = error_body.get("message") or error_body.get("detail") or resp.text
            except Exception:
                error_msg = resp.text

            status = resp.status_code

            if status == 401:
                raise ProviderError(
                    self.get_name(),
                    f"Authentication error: Invalid or missing API key. {error_msg}",
                )
            if status == 429:
                raise ProviderError(
                    self.get_name(),
                    f"Rate limit exceeded. {error_msg}",
                )
            if status == 400 and any(
                phrase in str(error_msg).lower()
                for phrase in ("context_length", "too long", "token limit")
            ):
                raise ContextLengthError(
                    model_name=model_name,
                    limit=self.get_max_context_length(model_name),
                    actual=0,
                    message=str(error_msg),
                )
            if status == 400 and any(
                phrase in str(error_msg).lower()
                for phrase in ("model not", "does not exist", "invalid model", "not found")
            ):
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found. Default is "
                    f"'{self.default_model}'. Error: {error_msg}",
                )

            raise ProviderError(
                self.get_name(),
                f"API Error ({status}): {error_msg}",
            )

    def _raise_for_status_stream(self, resp: Any, model_name: str) -> None:
        """Raise on HTTP errors during streaming (before body is consumed)."""
        if resp.status_code >= 400:
            raise ProviderError(
                self.get_name(),
                f"Streaming API Error ({resp.status_code})",
            )
