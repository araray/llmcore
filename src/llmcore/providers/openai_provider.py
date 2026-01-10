# src/llmcore/providers/openai_provider.py
"""
OpenAI API provider implementation for the LLMCore library.

Handles interactions with the OpenAI API (GPT models).
Accepts context as List[Message] and supports standardized tool-calling.
"""

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

try:
    import openai
    from openai import AsyncOpenAI, OpenAIError
    from openai.types.chat import ChatCompletionChunk

    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    ChatCompletionChunk = None  # type: ignore

try:
    import tiktoken

    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for various OpenAI models
DEFAULT_OPENAI_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
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
DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(BaseProvider):
    """
    LLMCore provider for interacting with the OpenAI API.
    Handles List[Message] context type and standardized tool-calling.
    """

    _client: Optional[AsyncOpenAI] = None
    _encoding: Optional[Any] = None
    _api_key_env_var: Optional[str] = None

    def __init__(self, config: Dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the OpenAIProvider.

        Args:
            config: Configuration dictionary from `[providers.openai]` containing:
                    'api_key' (optional): OpenAI API key.
                    'api_key_env_var' (optional): Environment variable to read the API key from.
                    'base_url' (optional): Custom OpenAI API endpoint URL.
                    'default_model' (optional): Default model to use.
                    'timeout' (optional): Request timeout in seconds.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not openai_available:
            raise ImportError(
                "OpenAI library not installed. Please install `openai` or `llmcore[openai]`."
            )
        if not tiktoken_available:
            raise ImportError(
                "tiktoken library not installed. It is required for the OpenAI provider."
            )

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

        if not self.api_key:
            logger.warning(
                "OpenAI API key not found in config or environment. Provider will likely fail."
            )

        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.debug("AsyncOpenAI client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise ConfigError(f"OpenAI client initialization failed: {e}")

        self._load_tokenizer(self.default_model)

    def _load_tokenizer(self, model_name: str):
        """Loads the tiktoken tokenizer for the specified OpenAI model."""
        if not tiktoken:
            self._encoding = None
            return
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
            logger.debug(f"Loaded tiktoken encoding for model: {model_name}")
        except KeyError:
            logger.warning(
                f"No specific tiktoken encoding for '{model_name}'. Using 'cl100k_base'."
            )
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding for '{model_name}': {e}", exc_info=True)
            self._encoding = None

    def get_name(self) -> str:
        """Returns the provider name: 'openai'."""
        return "openai"

    async def get_models_details(self) -> List[ModelDetails]:
        """
        Dynamically discovers available models from the OpenAI API.
        Note: The OpenAI API does not return context length or tool support flags,
        so these are populated from a static list.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "OpenAI client not initialized.")
        try:
            models_response = await self._client.models.list()
            model_details_list = []
            for model_obj in models_response.data:
                model_id = model_obj.id
                # OpenAI API doesn't return these details, so we use our static map.
                context_length = self.get_max_context_length(model_id)
                supports_tools = (
                    "gpt-3.5-turbo" in model_id or "gpt-4" in model_id or "gpt-4o" in model_id
                )

                details = ModelDetails(
                    id=model_id,
                    context_length=context_length,
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    provider_name=self.get_name(),
                    metadata={"owned_by": model_obj.owned_by, "created": model_obj.created},
                )
                model_details_list.append(details)
            logger.info(f"Discovered {len(model_details_list)} models from OpenAI API.")
            return model_details_list
        except OpenAIError as e:
            logger.error(f"OpenAI API error fetching models: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Failed to fetch models from OpenAI API: {e}")

    def get_supported_parameters(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Returns a schema of supported inference parameters for OpenAI models."""
        return {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_tokens": {"type": "integer", "minimum": 1},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
        }

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given OpenAI model."""
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
                limit = 4096
                logger.warning(
                    f"Unknown context length for OpenAI model '{model_name}'. Using fallback: {limit}."
                )
        return limit

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the OpenAI API with standardized tool support.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "OpenAI client not initialized.")

        supported_params = self.get_supported_parameters()
        for key in kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for OpenAI provider.")

        model_name = model or self.default_model
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        messages_payload: List[Dict[str, Any]] = []
        for msg in context:
            msg_dict = {"role": msg.role.value, "content": msg.content}
            if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_payload.append(msg_dict)

        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages to send.")

        # Prepare tools payload
        tools_payload_api = None
        if tools:
            tools_payload_api = [
                {"type": "function", "function": tool.model_dump()} for tool in tools
            ]

        api_kwargs = kwargs.copy()
        if tools_payload_api:
            api_kwargs["tools"] = tools_payload_api
        if tool_choice:
            api_kwargs["tool_choice"] = tool_choice

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
                **api_kwargs,
            }
            logger.debug(f"RAW LLM REQUEST ({self.get_name()}): {json.dumps(log_data, indent=2)}")

        try:
            response_or_stream = await self._client.chat.completions.create(
                model=model_name,
                messages=messages_payload,  # type: ignore
                stream=stream,
                **api_kwargs,
            )
            if stream:

                async def stream_wrapper():
                    async for chunk in response_or_stream:  # type: ignore
                        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"RAW LLM STREAM CHUNK ({self.get_name()}): {chunk.model_dump_json()}"
                            )
                        yield chunk.model_dump(exclude_none=True)

                return stream_wrapper()
            else:
                response_dict = response_or_stream.model_dump(exclude_none=True)  # type: ignore
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RAW LLM RESPONSE ({self.get_name()}): {json.dumps(response_dict, indent=2)}"
                    )
                return response_dict
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise ProviderError(
                self.get_name(), f"OpenAI API Error (Status {e.status_code}): {e.message}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a text string using tiktoken."""
        if not self._encoding:
            logger.warning("Tiktoken not available. Approximating token count.")
            return (len(text) + 3) // 4
        if not text:
            return 0
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))  # type: ignore

    async def count_message_tokens(
        self, messages: List[Message], model: Optional[str] = None
    ) -> int:
        """Counts tokens for a list of messages using tiktoken, including overhead."""
        if not self._encoding:
            logger.warning("Tiktoken not available. Approximating message token count.")
            total_chars = sum(len(msg.content) + len(msg.role.value) for msg in messages)
            return (total_chars + (len(messages) * 15)) // 4

        model_name = model or self.default_model
        tokens_per_message = 3
        tokens_per_name = 1
        if "gpt-3.5-turbo-0301" in model_name:
            tokens_per_message = 4
            tokens_per_name = -1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            try:
                num_tokens += len(self._encoding.encode(message.role.value))
                num_tokens += len(self._encoding.encode(message.content))
                if message.role == LLMCoreRole.TOOL:
                    # A simplified approximation for tool call overhead
                    num_tokens += 5
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed for message part: {e}. Approximating.")
                num_tokens += (len(message.role.value) + len(message.content)) // 4
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def extract_response_content(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from OpenAI non-streaming response.

        OpenAI format: {"choices": [{"message": {"content": "..."}}]}

        Args:
            response: The raw response dictionary from chat_completion().

        Returns:
            The extracted text content.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return message.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract content from OpenAI response: {e}")
            return ""

    def extract_delta_content(self, chunk: Dict[str, Any]) -> str:
        """
        Extract text delta from OpenAI streaming chunk.

        OpenAI streaming format: {"choices": [{"delta": {"content": "..."}}]}

        Args:
            chunk: A single streaming chunk dictionary.

        Returns:
            The extracted text delta.
        """
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            delta = choices[0].get("delta", {})
            return delta.get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    async def close(self) -> None:
        """Closes the underlying OpenAI client session."""
        if self._client:
            try:
                await self._client.close()
                logger.info("OpenAIProvider client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OpenAIProvider client: {e}", exc_info=True)
            finally:
                self._client = None
