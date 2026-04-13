# src/llmcore/providers/openai_provider.py
"""
OpenAI API provider implementation for the LLMCore library.

Handles interactions with the OpenAI API (GPT models).
Accepts context as List[Message] and supports standardized tool-calling,
multimodal content (vision, audio input), structured output, and
extended token usage extraction.

Tested against openai Python SDK v2.31.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
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
        AuthenticationError as OpenAIAuthError,
    )
    from openai._exceptions import (
        BadRequestError as OpenAIBadRequestError,
    )
    from openai._exceptions import (
        OpenAIError,
    )
    from openai._exceptions import (
        RateLimitError as OpenAIRateLimitError,
    )
    from openai.types.chat import ChatCompletionChunk

    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    OpenAIAPIError = Exception  # type: ignore
    OpenAIAPIStatusError = Exception  # type: ignore
    OpenAIAPIConnectionError = Exception  # type: ignore
    OpenAIAPITimeoutError = Exception  # type: ignore
    OpenAIAuthError = Exception  # type: ignore
    OpenAIBadRequestError = Exception  # type: ignore
    OpenAIRateLimitError = Exception  # type: ignore
    ChatCompletionChunk = None  # type: ignore

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

DEFAULT_OPENAI_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8000,
    "gpt-4-0613": 8000,
    "gpt-4-32k": 32000,
    "gpt-4-32k-0613": 32000,
    "gpt-3.5-turbo-0125": 16000,
    "gpt-3.5-turbo": 16000,
    "gpt-3.5-turbo-1106": 16000,
    "gpt-3.5-turbo-0613": 4000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-3.5-turbo-16k-0613": 16000,
}

_OPENAI_PREFIX_CONTEXT_HEURISTICS: list[tuple[str, int]] = [
    ("gpt-5.4-pro", 1050000),
    ("gpt-5.4-mini", 400000),
    ("gpt-5.4-nano", 400000),
    ("gpt-5.4", 1050000),
    ("gpt-5.3", 400000),
    ("gpt-5.2-pro", 400000),
    ("gpt-5.2", 400000),
    ("gpt-5.1-codex", 400000),
    ("gpt-5.1-mini", 400000),
    ("gpt-5.1", 400000),
    ("gpt-5-nano", 200000),
    ("gpt-5-mini", 200000),
    ("gpt-5", 400000),
    ("o4-mini", 200000),
    ("o3-pro", 200000),
    ("o3-mini", 200000),
    ("o3", 200000),
    ("o1-pro", 200000),
    ("o1-mini", 128000),
    ("o1", 200000),
    ("gpt-4.1-nano", 1048576),
    ("gpt-4.1-mini", 1048576),
    ("gpt-4.1", 1048576),
    ("codex-mini", 200000),
    ("gpt-4o-mini", 128000),
    ("gpt-4o", 128000),
    ("chatgpt-4o", 128000),
]

_warned_unknown_models: set[str] = set()
DEFAULT_MODEL = "gpt-4o"


def _is_reasoning_model(model: str) -> bool:
    """Check if model is an o-series reasoning model."""
    return any(model.startswith(p) for p in ("o1", "o3", "o4"))


def _needs_developer_role(model: str) -> bool:
    """Check if model requires developer role instead of system."""
    return _is_reasoning_model(model)


class OpenAIProvider(BaseProvider):
    """LLMCore provider for the OpenAI API.

    Multimodal via Message.metadata conventions:
    - metadata["content_parts"]: raw content part list (overrides content)
    - metadata["inline_images"]: list of image dicts -> image_url parts
    - metadata["inline_audio"]: list of audio dicts -> input_audio parts
    - metadata["tool_calls"]: assistant tool calls for multi-turn
    - metadata["name"]: participant name
    """

    _client: AsyncOpenAI | None = None
    _encoding: Any | None = None
    _api_key_env_var: str | None = None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads)
        if not openai_available:
            raise ImportError("OpenAI library not installed.")
        if not tiktoken_available:
            raise ImportError("tiktoken library not installed.")

        self._api_key_env_var = config.get("api_key_env_var")
        api_key = config.get("api_key")
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        self.api_key = api_key
        self.base_url = config.get("base_url")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        self.timeout = float(config.get("timeout", 60.0))
        self.max_retries = int(config.get("max_retries", 2))

        if not self.api_key:
            raise ConfigError("OpenAI API key not found. Set OPENAI_API_KEY or configure api_key.")

        try:
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            org = config.get("organization")
            if org:
                client_kwargs["organization"] = org
            project = config.get("project")
            if project:
                client_kwargs["project"] = project
            self._client = AsyncOpenAI(**client_kwargs)
            logger.debug("AsyncOpenAI client initialized.")
        except Exception as e:
            raise ConfigError(f"OpenAI client initialization failed: {e}")

        self._load_tokenizer(self.default_model)

    def _load_tokenizer(self, model_name: str):
        if not tiktoken:
            self._encoding = None
            return
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            fallback = (
                "o200k_base"
                if any(s in model_name for s in ("4o", "gpt-5", "gpt-4.1", "o1", "o3", "o4"))
                else "cl100k_base"
            )
            logger.warning(f"No tiktoken encoding for '{model_name}'. Using '{fallback}'.")
            try:
                self._encoding = tiktoken.get_encoding(fallback)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken: {e}", exc_info=True)
            self._encoding = None

    def get_name(self) -> str:
        return self._provider_instance_name or "openai"

    async def get_models_details(self) -> list[ModelDetails]:
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")
        try:
            models_response = await self._client.models.list()
            result = []
            provider_name = self.get_name()
            try:
                registry = get_model_card_registry()
            except Exception:
                registry = None

            for m in models_response.data:
                mid = m.id
                ctx = self.get_max_context_length(mid)
                supports_tools = False
                supports_vision = False
                if registry:
                    card = registry.get(provider_name, mid)
                    if card and card.capabilities:
                        supports_tools = (
                            card.capabilities.tool_use or card.capabilities.function_calling
                        )
                        supports_vision = card.capabilities.vision
                if not supports_tools:
                    supports_tools = any(
                        mid.startswith(p)
                        for p in (
                            "gpt-3.5-turbo",
                            "gpt-4",
                            "gpt-5",
                            "o1",
                            "o3",
                            "o4",
                            "chatgpt",
                            "codex",
                        )
                    )
                result.append(
                    ModelDetails(
                        id=mid,
                        context_length=ctx,
                        supports_streaming=True,
                        supports_tools=supports_tools,
                        supports_vision=supports_vision,
                        provider_name=provider_name,
                        metadata={"owned_by": m.owned_by, "created": m.created},
                    )
                )
            return result
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Failed to fetch models: {e}")

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        model_name = model or self.default_model
        is_reasoning = _is_reasoning_model(model_name)
        params: dict[str, Any] = {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_completion_tokens": {"type": "integer", "minimum": 1},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "n": {"type": "integer", "minimum": 1},
            "response_format": {"type": "object"},
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer", "minimum": 0, "maximum": 20},
            "logit_bias": {"type": "object"},
            "parallel_tool_calls": {"type": "boolean"},
            "stream_options": {"type": "object"},
            "modalities": {"type": "array", "items": {"type": "string"}},
            "audio": {"type": "object"},
            "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high"]},
            "prediction": {"type": "object"},
            "service_tier": {"type": "string"},
            "store": {"type": "boolean"},
            "metadata": {"type": "object"},
            "user": {"type": "string"},
            "prompt_cache_key": {"type": "string"},
            "prompt_cache_retention": {"type": "string"},
            "web_search_options": {"type": "object"},
            "verbosity": {"type": "string", "enum": ["low", "medium", "high"]},
        }
        if not is_reasoning:
            params["max_tokens"] = {"type": "integer", "minimum": 1}
        return params

    def get_max_context_length(self, model: str | None = None) -> int:
        model_name = model or self.default_model
        limit = DEFAULT_OPENAI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            if "gpt-4o" in model_name:
                limit = 128000
            elif "gpt-4-turbo" in model_name:
                limit = 128000
            elif "gpt-4-32k" in model_name:
                limit = 32768
            elif "gpt-4" in model_name:
                limit = 8192
            elif "gpt-3.5-turbo" in model_name:
                limit = 16385
            else:
                provider_name = self.get_name()
                try:
                    registry = get_model_card_registry()
                    card = registry.get(provider_name, model_name)
                    if card is not None:
                        limit = card.get_context_length()
                    else:
                        for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                            if model_name.startswith(prefix):
                                limit = ctx
                                break
                        if limit is None:
                            limit = 4096
                            if model_name not in _warned_unknown_models:
                                _warned_unknown_models.add(model_name)
                                logger.warning(
                                    "Unknown context for '%s'. Fallback: %d.", model_name, limit
                                )
                except Exception as e:
                    limit = 4096
                    if model_name not in _warned_unknown_models:
                        _warned_unknown_models.add(model_name)
                        logger.warning(
                            "Context lookup failed for '%s' (%s). Fallback: %d.",
                            model_name,
                            e,
                            limit,
                        )
        return limit

    def _build_message_payload(self, msg: Message, model_name: str) -> dict[str, Any]:
        """Build an OpenAI-format message dict from an llmcore Message."""
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        if role_str == "system" and _needs_developer_role(model_name):
            role_str = "developer"

        content: Any = msg.content
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

        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        if role_str == "assistant" and "tool_calls" in metadata:
            msg_dict["tool_calls"] = metadata["tool_calls"]
            if not msg.content:
                msg_dict["content"] = None

        if role_str == "assistant" and "audio" in metadata:
            msg_dict["audio"] = metadata["audio"]

        name = metadata.get("name")
        if name:
            msg_dict["name"] = name

        return msg_dict

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        model_name = model or self.default_model
        supported_params = self.get_supported_parameters(model_name)
        for key in kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for OpenAI provider.")

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        messages_payload = [self._build_message_payload(msg, model_name) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        tools_payload_api = None
        if tools:
            tools_payload_api = [{"type": "function", "function": t.model_dump()} for t in tools]

        api_kwargs = kwargs.copy()
        if tools_payload_api:
            api_kwargs["tools"] = tools_payload_api
        if tool_choice:
            api_kwargs["tool_choice"] = tool_choice

        if _is_reasoning_model(model_name) and "max_tokens" in api_kwargs:
            if "max_completion_tokens" not in api_kwargs:
                api_kwargs["max_completion_tokens"] = api_kwargs.pop("max_tokens")
            else:
                api_kwargs.pop("max_tokens")

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM REQUEST (%s): %s",
                self.get_name(),
                json.dumps(
                    {
                        "model": model_name,
                        "messages": messages_payload,
                        "stream": stream,
                        **api_kwargs,
                    },
                    indent=2,
                    default=str,
                ),
            )

        try:
            resp = await self._client.chat.completions.create(
                model=model_name, messages=messages_payload, stream=stream, **api_kwargs
            )  # type: ignore
            if stream:

                async def stream_wrapper():
                    async for chunk in resp:  # type: ignore
                        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "RAW STREAM CHUNK (%s): %s",
                                self.get_name(),
                                chunk.model_dump_json(),
                            )
                        yield chunk.model_dump(exclude_none=True)

                return stream_wrapper()
            else:
                response_dict = resp.model_dump(exclude_none=True)  # type: ignore
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW LLM RESPONSE (%s): %s",
                        self.get_name(),
                        json.dumps(response_dict, indent=2, default=str),
                    )
                return response_dict

        except OpenAIAPIStatusError as e:
            status = e.status_code
            msg = str(e)
            logger.error(f"OpenAI status error ({status}): {msg}", exc_info=True)
            if status == 400 and "context_length" in msg.lower():
                raise ContextLengthError(
                    provider_name=self.get_name(),
                    model=model_name,
                    max_tokens=self.get_max_context_length(model_name),
                    requested_tokens=None,
                    message=msg,
                )
            # Detect model-not-found errors and provide actionable info.
            # Different providers use different phrasing:
            #   DeepSeek: "Model Not Exist"
            #   OpenAI:   "does not exist" / "model_not_found"
            if status == 400 and any(
                phrase in msg.lower()
                for phrase in (
                    "model not exist",
                    "does not exist",
                    "model_not_found",
                    "invalid model",
                )
            ):
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found on provider "
                    f"'{self.get_name()}'. The provider's default model "
                    f"is '{self.default_model}'. Please verify the model "
                    f"name is correct for this provider's API. "
                    f"Original error: {msg}",
                )
            raise ProviderError(self.get_name(), f"API Error ({status}): {msg}")
        except OpenAIAPITimeoutError as e:
            logger.error(f"OpenAI timeout: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Timeout: {e}")
        except OpenAIAPIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Connection error: {e}")
        except OpenAIAPIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error: {e}")

    def extract_response_content(self, response: dict[str, Any]) -> str:
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract content: {e}")
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
        """Extract tool calls from an OpenAI response, normalized to ToolCall objects."""
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
                        args_dict = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {"_raw": args_str}
                    out.append(
                        ToolCall(
                            id=tc.get("id", ""), name=func.get("name", ""), arguments=args_dict
                        )
                    )
                else:
                    out.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("custom", {}).get("name", tc_type),
                            arguments=tc,
                        )
                    )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract tool calls: {e}")
        return out

    def extract_audio_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Extract audio data from response (when modalities includes audio)."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("audio")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_annotations(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract web search annotations."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return []
            return choices[0].get("message", {}).get("annotations") or []
        except (KeyError, IndexError, TypeError):
            return []

    def extract_refusal(self, response: dict[str, Any]) -> str | None:
        """Extract safety refusal message."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("refusal")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract extended usage details including reasoning/audio/cached tokens."""
        usage = response.get("usage", {})
        if not usage:
            return {}
        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        cd = usage.get("completion_tokens_details", {})
        if cd:
            result["reasoning_tokens"] = cd.get("reasoning_tokens")
            result["audio_output_tokens"] = cd.get("audio_tokens")
            result["accepted_prediction_tokens"] = cd.get("accepted_prediction_tokens")
            result["rejected_prediction_tokens"] = cd.get("rejected_prediction_tokens")
        pd = usage.get("prompt_tokens_details", {})
        if pd:
            result["cached_tokens"] = pd.get("cached_tokens")
            result["audio_input_tokens"] = pd.get("audio_tokens")
        return result

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        if not self._encoding:
            return (len(text) + 3) // 4
        if not text:
            return 0
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        if not self._encoding:
            total = sum(
                len(m.content) + len(m.role.value if hasattr(m.role, "value") else str(m.role))
                for m in messages
            )
            return (total + len(messages) * 15) // 4
        model_name = model or self.default_model
        tpm = 4 if "gpt-3.5-turbo-0301" in model_name else 3
        n = 0
        for m in messages:
            n += tpm
            try:
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                n += len(self._encoding.encode(rv))
                n += len(self._encoding.encode(m.content))
                if m.role == LLMCoreRole.TOOL:
                    n += 5
            except Exception:
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                n += (len(rv) + len(m.content)) // 4
        return n + 3

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.close()
                logger.info("OpenAIProvider closed.")
            except Exception as e:
                logger.error(f"Error closing: {e}", exc_info=True)
            finally:
                self._client = None

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
        """Generate speech audio from text using OpenAI TTS.

        Models: ``tts-1``, ``tts-1-hd``, ``gpt-4o-mini-tts``.
        Voices: alloy, ash, ballad, coral, echo, fable, onyx, nova,
                sage, shimmer, verse, marin, cedar.

        Args:
            text: Text to convert (max 4096 characters).
            voice: Voice identifier.
            model: TTS model.  Defaults to ``tts-1``.
            response_format: mp3, opus, aac, flac, wav, pcm.
            speed: 0.25–4.0 (default 1.0).
            instructions: Voice control instructions
                (gpt-4o-mini-tts only).
            **kwargs: Extra API parameters.

        Returns:
            ``SpeechResult`` with raw audio bytes.
        """
        from ..models_multimodal import SpeechResult

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        tts_model = model or "tts-1"

        api_kwargs: dict[str, Any] = {
            "input": text,
            "model": tts_model,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if instructions is not None:
            api_kwargs["instructions"] = instructions
        api_kwargs.update(kwargs)

        logger.debug(
            "TTS request: model=%s, voice=%s, format=%s, len=%d",
            tts_model,
            voice,
            response_format,
            len(text),
        )

        try:
            response = await self._client.audio.speech.create(**api_kwargs)
            audio_bytes = response.content

            return SpeechResult(
                audio_data=audio_bytes,
                format=response_format,
                model=tts_model,
                voice=voice,
                metadata={},
            )

        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"TTS Error ({e.status_code}): {e}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"TTS timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"TTS connection error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"TTS error: {e}")
        except Exception as e:
            logger.error(f"Unexpected TTS error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"TTS unexpected error: {e}")

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
        """Transcribe audio to text using OpenAI Whisper / GPT-4o-transcribe.

        Models: ``whisper-1``, ``gpt-4o-transcribe``,
                ``gpt-4o-mini-transcribe``, ``gpt-4o-transcribe-diarize``.

        Args:
            audio_data: Raw audio bytes or file path string.
            model: STT model.  Defaults to ``whisper-1``.
            language: Input language hint (ISO-639-1).
            prompt: Optional transcription style guide.
            response_format: json, text, srt, verbose_json, vtt.
            temperature: Sampling temperature.
            timestamp_granularities: ``["word"]`` and/or ``["segment"]``.
            **kwargs: Extra API parameters (e.g. ``chunking_strategy``).

        Returns:
            ``TranscriptionResult`` with transcribed text and segments.
        """
        from ..models_multimodal import TranscriptionResult, TranscriptionSegment

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        stt_model = model or "whisper-1"

        # Build file parameter
        if isinstance(audio_data, str):
            # File path
            import pathlib

            file_path = pathlib.Path(audio_data)
            file_obj: Any = open(file_path, "rb")
            file_name = file_path.name
        else:
            # Raw bytes — wrap in a tuple for the SDK
            import io

            file_obj = ("audio.wav", io.BytesIO(audio_data), "audio/wav")
            file_name = "audio.wav"

        api_kwargs: dict[str, Any] = {
            "file": file_obj,
            "model": stt_model,
        }
        if language is not None:
            api_kwargs["language"] = language
        if prompt is not None:
            api_kwargs["prompt"] = prompt
        if response_format != "json":
            api_kwargs["response_format"] = response_format
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if timestamp_granularities:
            api_kwargs["timestamp_granularities"] = timestamp_granularities
            # verbose_json needed for timestamps
            if response_format == "json":
                api_kwargs["response_format"] = "verbose_json"
        api_kwargs.update(kwargs)

        logger.debug(
            "STT request: model=%s, language=%s, format=%s",
            stt_model,
            language,
            response_format,
        )

        try:
            response = await self._client.audio.transcriptions.create(**api_kwargs)

            # Close file handle if we opened one
            if isinstance(audio_data, str) and hasattr(file_obj, "close"):
                file_obj.close()

            # Parse response — structure depends on format
            if hasattr(response, "text"):
                text = response.text
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)

            # Extract segments if available (verbose_json format)
            segments: list[TranscriptionSegment] = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    segments.append(
                        TranscriptionSegment(
                            text=seg.get("text", "")
                            if isinstance(seg, dict)
                            else getattr(seg, "text", ""),
                            start=seg.get("start", 0.0)
                            if isinstance(seg, dict)
                            else getattr(seg, "start", 0.0),
                            end=seg.get("end", 0.0)
                            if isinstance(seg, dict)
                            else getattr(seg, "end", 0.0),
                        )
                    )

            duration = None
            if hasattr(response, "duration"):
                duration = response.duration

            detected_lang = None
            if hasattr(response, "language"):
                detected_lang = response.language

            return TranscriptionResult(
                text=text,
                language=detected_lang or language,
                duration_seconds=duration,
                segments=segments,
                model=stt_model,
                metadata={},
            )

        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"STT Error ({e.status_code}): {e}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"STT timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"STT connection error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"STT error: {e}")
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Unexpected STT error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"STT unexpected error: {e}")

    # ------------------------------------------------------------------
    # Multimodal: Image Generation
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        response_format: str = "b64_json",
        style: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate images from a text prompt using OpenAI.

        Models: ``dall-e-2``, ``dall-e-3``, ``gpt-image-1``,
                ``gpt-image-1-mini``, ``gpt-image-1.5``.

        Args:
            prompt: Text description of desired image(s).
            model: Image model.  Defaults to ``gpt-image-1``.
            n: Number of images (1–10; dall-e-3 only supports 1).
            size: Dimensions (e.g. ``"1024x1024"``).
            quality: Quality level (standard, hd, low, medium, high, auto).
            response_format: ``"url"`` or ``"b64_json"``.
            style: ``"vivid"`` or ``"natural"`` (dall-e-3).
            **kwargs: Extra params (background, output_format, etc.).

        Returns:
            ``ImageGenerationResult`` with generated images.
        """
        from ..models_multimodal import GeneratedImage, ImageGenerationResult

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        img_model = model or "gpt-image-1"

        api_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model": img_model,
            "n": n,
            "response_format": response_format,
        }
        if size is not None:
            api_kwargs["size"] = size
        if quality is not None:
            api_kwargs["quality"] = quality
        if style is not None:
            api_kwargs["style"] = style
        api_kwargs.update(kwargs)

        logger.debug(
            "Image gen: model=%s, n=%d, size=%s, quality=%s",
            img_model,
            n,
            size,
            quality,
        )

        try:
            response = await self._client.images.generate(**api_kwargs)

            images: list[GeneratedImage] = []
            for img_data in response.data:
                images.append(
                    GeneratedImage(
                        data=getattr(img_data, "b64_json", None),
                        url=getattr(img_data, "url", None),
                        revised_prompt=getattr(img_data, "revised_prompt", None),
                        format=kwargs.get("output_format", "png"),
                    )
                )

            return ImageGenerationResult(
                images=images,
                model=img_model,
                metadata={
                    "created": getattr(response, "created", None),
                },
            )

        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"Image Error ({e.status_code}): {e}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"Image timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"Image connection error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Image error: {e}")
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Unexpected image error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Image unexpected error: {e}")
