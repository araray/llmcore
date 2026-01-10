# src/llmcore/providers/anthropic_provider.py
"""
Anthropic API provider implementation for the LLMCore library.

Handles interactions with the Anthropic API (Claude models).
Uses the official 'anthropic' Python SDK.
Accepts context as List[Message] and supports standardized tool-calling.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Use the official anthropic library
try:
    import anthropic
    from anthropic import AnthropicError, AsyncAnthropic
    from anthropic.types import MessageParam, TextBlockParam
    from anthropic.types.message_stream_event import MessageStreamEvent

    anthropic_available = True
except ImportError:
    anthropic_available = False
    AsyncAnthropic = None  # type: ignore
    AnthropicError = Exception  # type: ignore
    MessageParam = Dict[str, Any]  # type: ignore
    TextBlockParam = Dict[str, Any]  # type: ignore
    MessageStreamEvent = Any  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths
DEFAULT_ANTHROPIC_TOKEN_LIMITS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
}
DEFAULT_MODEL = "claude-3-haiku-20240307"


class AnthropicProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Anthropic API (Claude models).
    Handles List[Message] context type and standardized tool-calling.
    """

    _client: Optional[AsyncAnthropic] = None
    _sync_client_for_counting: Optional[anthropic.Anthropic] = None
    _api_key_env_var: Optional[str] = None

    def __init__(self, config: Dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the AnthropicProvider.

        Args:
            config: Configuration dictionary from `[providers.anthropic]` containing:
                    'api_key' (optional): Anthropic API key.
                    'api_key_env_var' (optional): Environment variable to read the API key from.
                    'base_url' (optional): Custom Anthropic API endpoint URL.
                    'default_model' (optional): Default model to use.
                    'timeout' (optional): Request timeout in seconds.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not anthropic_available:
            raise ImportError(
                "Anthropic library not installed. Please install `anthropic` or `llmcore[anthropic]`."
            )

        self._api_key_env_var = config.get("api_key_env_var")
        api_key = config.get("api_key")
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.api_key = api_key
        self.base_url = config.get("base_url")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        self.timeout = float(config.get("timeout", 60.0))

        if not self.api_key:
            logger.warning("Anthropic API key not found. Provider will likely fail.")

        try:
            self._client = AsyncAnthropic(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            self._sync_client_for_counting = anthropic.Anthropic(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            logger.debug("Anthropic clients initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic clients: {e}", exc_info=True)
            raise ConfigError(f"Anthropic client initialization failed: {e}")

    def get_name(self) -> str:
        """Returns the provider name: 'anthropic'."""
        return "anthropic"

    async def get_models_details(self) -> List[ModelDetails]:
        """
        Returns a list of `ModelDetails` for known Anthropic models.

        Note: The official Anthropic SDK does not currently provide a public
        method to list models dynamically. This implementation relies on a
        static, internal list of known models and their capabilities.
        """
        logger.warning(
            "AnthropicProvider.get_models_details() returning static list due to SDK limitations."
        )
        details_list = []
        for model_id, context_length in DEFAULT_ANTHROPIC_TOKEN_LIMITS.items():
            # All modern Claude 3 models support tool use.
            supports_tools = "claude-3" in model_id
            details = ModelDetails(
                id=model_id,
                context_length=context_length,
                supports_streaming=True,
                supports_tools=supports_tools,
                provider_name=self.get_name(),
                metadata={},
            )
            details_list.append(details)
        return details_list

    def get_supported_parameters(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Returns a schema of supported inference parameters for Anthropic models."""
        return {
            "max_tokens": {"type": "integer", "minimum": 1},
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer"},
            "stop_sequences": {"type": "array", "items": {"type": "string"}},
        }

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given Anthropic model."""
        model_name = model or self.default_model
        limit = DEFAULT_ANTHROPIC_TOKEN_LIMITS.get(model_name)
        if limit is None:
            limit = 100000
            logger.warning(
                f"Unknown context length for Anthropic model '{model_name}'. Using fallback: {limit}."
            )
        return limit

    def _convert_llmcore_msgs_to_anthropic(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[MessageParam]]:
        """
        Converts a list of LLMCore `Message` objects to the format expected by the Anthropic API.
        """
        anthropic_messages: List[MessageParam] = []
        system_prompt: Optional[str] = None

        processed_messages = list(messages)
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_prompt = processed_messages.pop(0).content

        last_role_added_to_api = None
        for msg in processed_messages:
            anthropic_role_str = ""
            if msg.role == LLMCoreRole.USER:
                anthropic_role_str = "user"
            elif msg.role == LLMCoreRole.ASSISTANT:
                anthropic_role_str = "assistant"
            elif msg.role == LLMCoreRole.TOOL:
                # Anthropic tool results are added to the assistant's turn.
                # This logic should be handled by the caller who constructs the context.
                # For now, we'll skip tool messages here as they are handled differently.
                logger.warning(
                    f"Skipping message with role 'tool' during conversion. Tool results should be part of an assistant message."
                )
                continue
            else:
                logger.warning(f"Skipping message with unmappable role for Anthropic: {msg.role}")
                continue

            if anthropic_role_str == last_role_added_to_api:
                logger.warning(
                    f"Merging consecutive '{anthropic_role_str}' messages for Anthropic API."
                )
                if anthropic_messages and isinstance(anthropic_messages[-1]["content"], list):
                    last_content_block = anthropic_messages[-1]["content"][0]  # type: ignore
                    if last_content_block["type"] == "text":  # type: ignore
                        last_content_block["text"] += f"\n{msg.content}"  # type: ignore
                        continue

            content_block: TextBlockParam = {"type": "text", "text": msg.content}
            anthropic_messages.append({"role": anthropic_role_str, "content": [content_block]})  # type: ignore
            last_role_added_to_api = anthropic_role_str

        return system_prompt, anthropic_messages

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
        Sends a chat completion request to the Anthropic API with standardized tool support.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Anthropic client not initialized.")

        supported_params = self.get_supported_parameters()
        for key in kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for Anthropic provider.")

        model_name = model or self.default_model
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        system_prompt, messages_payload = self._convert_llmcore_msgs_to_anthropic(context)
        if not messages_payload:
            raise ProviderError(
                self.get_name(), "No valid messages to send after context processing."
            )

        api_kwargs = kwargs.copy()
        api_kwargs["max_tokens"] = api_kwargs.pop(
            "max_tokens", 2048
        )  # Anthropic requires max_tokens

        if tools:
            api_kwargs["tools"] = [tool.model_dump() for tool in tools]
        if tool_choice:
            # Anthropic tool_choice is more complex, e.g. {"type": "any"} or {"type": "tool", "name": ...}
            # This is a simplified mapping for now.
            if tool_choice == "auto":
                api_kwargs["tool_choice"] = {"type": "auto"}
            elif tool_choice == "any":
                api_kwargs["tool_choice"] = {"type": "any"}
            else:  # Assume it's a specific tool name
                api_kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "messages": messages_payload,
                "system": system_prompt,
                "stream": stream,
                **api_kwargs,
            }
            logger.debug(f"RAW LLM REQUEST ({self.get_name()}): {json.dumps(log_data, indent=2)}")

        try:
            if stream:
                response_stream = await self._client.messages.stream(
                    model=model_name,
                    messages=messages_payload,  # type: ignore
                    system=system_prompt,
                    **api_kwargs,
                )

                async def stream_wrapper():
                    async for event in response_stream:
                        if self.log_raw_payloads_enabled:
                            logger.debug(
                                f"RAW LLM STREAM CHUNK ({self.get_name()}): {event.model_dump_json()}"
                            )
                        event_dict = event.model_dump(exclude_none=True)
                        # Adapt to a more consistent choice/delta structure if possible
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            event_dict["choices"] = [{"delta": {"content": event.delta.text}}]
                        elif event.type == "message_delta":
                            event_dict["choices"] = [{"finish_reason": event.delta.stop_reason}]
                        yield event_dict

                return stream_wrapper()
            else:
                response = await self._client.messages.create(
                    model=model_name,
                    messages=messages_payload,  # type: ignore
                    system=system_prompt,
                    **api_kwargs,
                )
                response_dict = response.model_dump(exclude_none=True)
                if self.log_raw_payloads_enabled:
                    logger.debug(
                        f"RAW LLM RESPONSE ({self.get_name()}): {json.dumps(response_dict, indent=2)}"
                    )

                # Normalize to OpenAI-like structure for consistency
                main_content = "".join(
                    [
                        block.get("text", "")
                        for block in response_dict.get("content", [])
                        if block.get("type") == "text"
                    ]
                )
                tool_calls = [
                    block
                    for block in response_dict.get("content", [])
                    if block.get("type") == "tool_use"
                ]

                normalized_choices = [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": main_content if not tool_calls else None,
                            "tool_calls": tool_calls if tool_calls else None,
                        },
                        "finish_reason": response_dict.get("stop_reason"),
                    }
                ]

                return {
                    "id": response_dict.get("id"),
                    "model": response_dict.get("model"),
                    "choices": normalized_choices,
                    "usage": response_dict.get("usage"),
                }

        except AnthropicError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise ProviderError(
                self.get_name(), f"API Error (Status: {getattr(e, 'status_code', 'N/A')}): {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the synchronous Anthropic client wrapped in a thread."""
        if not self._sync_client_for_counting:
            logger.warning("Anthropic sync client not available. Approximating token count.")
            return (len(text) + 3) // 4
        if not text:
            return 0
        try:
            return await asyncio.to_thread(self._sync_client_for_counting.count_tokens, text)
        except Exception as e:
            logger.error(f"Failed to count tokens with Anthropic client: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(
        self, messages: List[Message], model: Optional[str] = None
    ) -> int:
        """Approximates token count for a list of messages for Anthropic models."""
        # Anthropic's token counting is complex. A simple sum is a rough approximation.
        # For better accuracy, one would construct the full prompt string.
        prompt_str = ""
        system_prompt, anthropic_msgs_list = self._convert_llmcore_msgs_to_anthropic(messages)
        if system_prompt:
            prompt_str += f"System: {system_prompt}\n\n"
        for msg_param in anthropic_msgs_list:
            role = msg_param["role"]
            content_str = "".join(
                [
                    block.get("text", "")
                    for block in msg_param.get("content", [])
                    if block.get("type") == "text"
                ]
            )
            prompt_str += f"{str(role).capitalize()}: {content_str}\n"
        if prompt_str and anthropic_msgs_list and anthropic_msgs_list[-1]["role"] == "user":
            prompt_str += "Assistant:"

        return await self.count_tokens(prompt_str, model)

    def extract_response_content(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from Anthropic non-streaming response.

        AnthropicProvider normalizes responses to OpenAI format:
        {"choices": [{"message": {"content": "..."}}]}

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
            logger.warning(f"Failed to extract content from Anthropic response: {e}")
            return ""

    def extract_delta_content(self, chunk: Dict[str, Any]) -> str:
        """
        Extract text delta from Anthropic streaming chunk.

        AnthropicProvider normalizes streaming to:
        {"choices": [{"delta": {"content": "..."}}]}

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
        """Closes the underlying Anthropic client sessions."""
        if self._client:
            await self._client.close()
        if self._sync_client_for_counting:
            self._sync_client_for_counting.close()
        logger.info("AnthropicProvider clients closed.")
