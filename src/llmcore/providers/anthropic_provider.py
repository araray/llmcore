# src/llmcore/providers/anthropic_provider.py
"""
Anthropic API provider implementation for the LLMCore library.

Handles interactions with the Anthropic API (Claude models).
Uses the official 'anthropic' Python SDK.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

# Use the official anthropic library
try:
    import anthropic
    from anthropic import AsyncAnthropic, AnthropicError
    from anthropic.types import MessageParam, TextBlockParam, ToolParam # For constructing messages
    anthropic_available = True
except ImportError:
    anthropic_available = False
    AsyncAnthropic = None # type: ignore
    AnthropicError = Exception # type: ignore
    MessageParam = Dict[str, Any] # type: ignore
    TextBlockParam = Dict[str, Any] # type: ignore
    ToolParam = Dict[str, Any] # type: ignore


from ..models import Message, Role
from ..exceptions import ProviderError, ConfigError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Anthropic Claude models
# Source: https://docs.anthropic.com/claude/reference/input-and-output-sizes
# These are approximate and can change. Anthropic models generally have large context windows.
DEFAULT_ANTHROPIC_TOKEN_LIMITS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
    # Add other models as needed
}

# Default model if not specified
DEFAULT_MODEL = "claude-3-haiku-20240307" # A fast and capable model

class AnthropicProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Anthropic API (Claude models).
    """
    _client: Optional[AsyncAnthropic] = None
    _sync_client_for_counting: Optional[anthropic.Anthropic] = None # For synchronous token counting

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AnthropicProvider.

        Args:
            config: Configuration dictionary containing:
                    'api_key' (optional): Anthropic API key. Defaults to env var ANTHROPIC_API_KEY.
                    'base_url' (optional): Custom Anthropic API endpoint URL.
                    'default_model' (optional): Default model (e.g., "claude-3-opus-20240229").
                    'timeout' (optional): Request timeout in seconds (default: 60).
        """
        if not anthropic_available:
            raise ImportError("Anthropic library is not installed. Please install `anthropic` or `llmcore[anthropic]`.")

        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = config.get('base_url') # Allow None for default Anthropic URL
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = float(config.get('timeout', 60.0)) # Anthropic SDK uses httpx.Timeout

        if not self.api_key:
            logger.warning("Anthropic API key not found in config or environment variable ANTHROPIC_API_KEY.")
            # Client initialization will fail if the key is actually needed by the SDK

        try:
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            # Initialize a synchronous client specifically for token counting if needed,
            # as the count_tokens method is synchronous.
            self._sync_client_for_counting = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.debug("AsyncAnthropic client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncAnthropic client: {e}", exc_info=True)
            raise ConfigError(f"Anthropic client initialization failed: {e}")

    def get_name(self) -> str:
        """Returns the provider name."""
        return "anthropic"

    def get_available_models(self) -> List[str]:
        """
        Returns a list of known default models for Anthropic.
        Note: Does not dynamically fetch from the API.
        """
        logger.warning("AnthropicProvider.get_available_models() returning static list.")
        return list(DEFAULT_ANTHROPIC_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length for the given Anthropic model."""
        model_name = model or self.default_model
        limit = DEFAULT_ANTHROPIC_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Fallback for unknown models, Anthropic models usually have large contexts
            limit = 100000
            logger.warning(f"Unknown context length for Anthropic model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    def _convert_context_to_anthropic(
        self,
        messages: List[Message]
    ) -> tuple[Optional[str], List[MessageParam]]:
        """
        Converts LLMCore messages to Anthropic's MessageParam format.
        Separates the system prompt and ensures alternating user/assistant roles.

        Anthropic API expects:
        - Optional `system` parameter for the system prompt.
        - `messages` parameter as a list of dicts, alternating 'user' and 'assistant' roles.
          The first message must be 'user'.
        """
        anthropic_messages: List[MessageParam] = []
        system_prompt: Optional[str] = None

        # Extract system prompt if it's the first message
        processed_messages = list(messages) # Create a mutable copy
        if processed_messages and processed_messages[0].role == Role.SYSTEM:
            system_prompt = processed_messages.pop(0).content
            logger.debug("System prompt extracted for Anthropic request.")

        # Ensure the conversation starts with a user message
        if not processed_messages or processed_messages[0].role != Role.USER:
            # This is a deviation from Anthropic's strict requirement.
            # We might prepend a dummy user message or log a warning.
            # For now, let's log and proceed; the API will likely error out.
            # A more robust solution might involve prepending a generic user prompt
            # if the first message is 'assistant', or erroring early.
            logger.warning(
                "Anthropic conversation should start with a 'user' role. "
                "The current message sequence might be invalid."
            )
            # If the first message is assistant, and there's no user message before it,
            # Anthropic API will error. We could try to fix it here, or let the API handle it.
            # Example fix: if processed_messages and processed_messages[0].role == Role.ASSISTANT:
            #    anthropic_messages.append({"role": "user", "content": [{"type": "text", "text": "(Previous turn)"}]})

        last_role = None
        for msg in processed_messages:
            current_llmcore_role = msg.role
            anthropic_role_str = ""

            if current_llmcore_role == Role.USER:
                anthropic_role_str = "user"
            elif current_llmcore_role == Role.ASSISTANT:
                anthropic_role_str = "assistant"
            else:
                logger.warning(f"Skipping message with unmappable role for Anthropic: {current_llmcore_role}")
                continue

            # Ensure alternating roles. If same as last, merge or skip.
            # Anthropic is strict: user, assistant, user, assistant...
            if anthropic_role_str == last_role:
                logger.warning(
                    f"Duplicate consecutive role '{anthropic_role_str}' found. "
                    "Anthropic API requires alternating roles. Attempting to merge or last message might be dropped by API."
                )
                # A simple merge: append content to the last message if possible.
                # This is a basic heuristic.
                if anthropic_messages and isinstance(anthropic_messages[-1]["content"], list):
                    # Assuming content is a list of blocks, and the first is TextBlockParam
                    last_content_block = anthropic_messages[-1]["content"][0]
                    if last_content_block["type"] == "text": # type: ignore
                        last_content_block["text"] += f"\n{msg.content}" # type: ignore
                        continue # Skip adding a new message dict

            # Create the message structure. Anthropic messages content is a list of blocks.
            # For now, we only support simple text content.
            content_block: TextBlockParam = {"type": "text", "text": msg.content}
            anthropic_messages.append({"role": anthropic_role_str, "content": [content_block]}) # type: ignore
            last_role = anthropic_role_str

        return system_prompt, anthropic_messages


    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Sends a chat completion request to the Anthropic API."""
        if not self._client:
            raise ProviderError(self.get_name(), "Anthropic client not initialized. API key might be missing.")
        if not isinstance(context, list): # Assuming context is List[Message]
            raise ProviderError(self.get_name(), f"AnthropicProvider received unsupported context type: {type(context).__name__}")

        model_name = model or self.default_model
        system_prompt, messages_payload = self._convert_context_to_anthropic(context)

        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after conversion to Anthropic format.")
        if messages_payload[0]['role'] != 'user':
             raise ProviderError(self.get_name(), "Anthropic API requires the first message to be from the 'user' role.")


        # Anthropic specific parameters
        max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_tokens_to_sample", 2048)) # Anthropic uses max_tokens_to_sample

        logger.debug(
            f"Sending request to Anthropic API: model='{model_name}', stream={stream}, "
            f"num_messages={len(messages_payload)}, system_prompt_present={bool(system_prompt)}"
        )

        try:
            if stream:
                response_stream = await self._client.messages.stream(
                    model=model_name,
                    messages=messages_payload, # type: ignore
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                    top_k=kwargs.get("top_k"),
                    stop_sequences=kwargs.get("stop_sequences")
                    # Other Anthropic-specific params can be added here
                )
                logger.debug(f"Processing stream response from Anthropic model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    try:
                        async for event in response_stream:
                            # Convert event to dict for consistent output with other providers
                            # Event types: message_start, content_block_start, content_block_delta,
                            # content_block_stop, message_delta, message_stop
                            event_dict = {}
                            if event.type == "message_start":
                                event_dict = {
                                    "type": "message_start",
                                    "message": {
                                        "id": event.message.id,
                                        "role": event.message.role, # "assistant"
                                        "model": event.message.model,
                                        "usage": event.message.usage.model_dump() if event.message.usage else None
                                    }
                                }
                            elif event.type == "content_block_delta":
                                if event.delta.type == "text_delta":
                                    event_dict = {
                                        "type": "content_block_delta",
                                        "index": event.index,
                                        "delta": {"type": "text", "text": event.delta.text},
                                        # Mimic OpenAI stream structure for text delta
                                        "choices": [{"delta": {"content": event.delta.text}, "index": event.index}]
                                    }
                            elif event.type == "message_delta":
                                # Contains usage updates and stop_reason
                                event_dict = {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": event.delta.stop_reason, "stop_sequence": event.delta.stop_sequence},
                                    "usage": event.usage.model_dump()
                                }
                            elif event.type == "message_stop":
                                event_dict = {"type": "message_stop", "reason": "stop_event"} # Or extract more info
                            # Add other event types if needed

                            if event_dict: # Only yield if we processed the event
                                yield event_dict

                    except anthropic.APIConnectionError as e:
                        logger.error(f"Anthropic API connection error during stream: {e}", exc_info=True)
                        yield {"error": f"Anthropic API Connection Error: {e}", "done": True}
                        raise ProviderError(self.get_name(), f"API Connection Error during stream: {e}")
                    except anthropic.APIStatusError as e:
                        logger.error(f"Anthropic API status error during stream: {e.status_code} - {e.message}", exc_info=True)
                        yield {"error": f"Anthropic API Status Error ({e.status_code}): {e.message}", "done": True}
                        raise ProviderError(self.get_name(), f"API Status Error ({e.status_code}) during stream: {e.message}")
                    except Exception as e:
                        logger.error(f"Unexpected error processing Anthropic stream: {e}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e}", "done": True}
                    finally:
                        logger.debug("Anthropic stream finished.")
                return stream_wrapper()
            else: # Non-streaming
                response = await self._client.messages.create(
                    model=model_name,
                    messages=messages_payload, # type: ignore
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                    top_k=kwargs.get("top_k"),
                    stop_sequences=kwargs.get("stop_sequences")
                )
                logger.debug(f"Processing non-stream response from Anthropic model '{model_name}'")
                # Convert the Pydantic model response to a dictionary
                return response.model_dump()

        except AnthropicError as e: # Catch specific Anthropic errors
            # Try to get status code if available
            status_code = getattr(e, 'status_code', None)
            error_message = str(e)
            logger.error(f"Anthropic API error (Status: {status_code}): {error_message}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error (Status: {status_code}): {error_message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Anthropic timed out after {self.timeout} seconds.")
            raise ProviderError(self.get_name(), f"Request timed out after {self.timeout}s.")
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")


    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the Anthropic client."""
        if not self._sync_client_for_counting:
            logger.warning("Anthropic synchronous client for token counting not initialized.")
            # Fallback approximation
            return (len(text) + 3) // 4 if text else 0

        if not text:
            return 0

        try:
            # The Anthropic library's count_tokens is synchronous
            # model_name = model or self.default_model # Model name not needed for count_tokens
            return self._sync_client_for_counting.count_tokens(text)
        except AnthropicError as e:
            logger.error(f"Anthropic API error during token count: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e}")
        except Exception as e:
            logger.error(f"Failed to count tokens with Anthropic client: {e}", exc_info=True)
            # Fallback approximation
            return (len(text) + 3) // 4


    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Counts tokens for a list of messages using the Anthropic client.
        This involves formatting the messages as a single string or using a more complex method
        if the API provides one for message lists. Anthropic's `count_tokens` takes a simple string.
        We need to simulate the prompt structure.
        """
        if not self._sync_client_for_counting:
            logger.warning("Anthropic synchronous client for token counting not initialized.")
            # Fallback approximation
            total_text_len = sum(len(msg.content) + len(msg.role.value) for msg in messages)
            return (total_text_len + (len(messages) * 5)) // 4 if messages else 0

        # Heuristic: Convert messages to a single string representing the prompt structure.
        # This is an approximation as actual tokenization depends on internal formatting.
        # Anthropic's tokenization is not as publicly detailed as OpenAI's regarding message structures.
        prompt_str = ""
        system_prompt, anthropic_msgs_list = self._convert_context_to_anthropic(messages)

        if system_prompt:
            prompt_str += f"System: {system_prompt}\n\n"

        for msg_param in anthropic_msgs_list:
            role = msg_param["role"]
            content_str = ""
            if isinstance(msg_param["content"], list) and msg_param["content"]:
                # Assuming first block is text
                first_block = msg_param["content"][0]
                if first_block.get("type") == "text":
                    content_str = first_block.get("text", "")

            prompt_str += f"{role.capitalize()}: {content_str}\n"

        # Add a final "Assistant: " to simulate the start of the assistant's response
        if prompt_str and anthropic_msgs_list and anthropic_msgs_list[-1]["role"] == "user":
            prompt_str += "Assistant:"

        if not prompt_str:
            return 0

        return self.count_tokens(prompt_str, model)


    async def close(self) -> None:
        """Closes the underlying Anthropic client session."""
        if self._client:
            try:
                await self._client.close()
                logger.debug("AsyncAnthropic client closed.")
            except Exception as e:
                logger.error(f"Error closing AsyncAnthropic client: {e}", exc_info=True)
        if self._sync_client_for_counting:
            try:
                self._sync_client_for_counting.close()
                logger.debug("Anthropic synchronous client for token counting closed.")
            except Exception as e:
                logger.error(f"Error closing Anthropic synchronous client: {e}", exc_info=True)

        self._client = None
        self._sync_client_for_counting = None
