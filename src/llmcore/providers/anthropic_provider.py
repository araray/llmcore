# src/llmcore/providers/anthropic_provider.py
"""
Anthropic API provider implementation for the LLMCore library.

Handles interactions with the Anthropic API (Claude models).
Uses the official 'anthropic' Python SDK.
Accepts context as List[Message].
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple

# Use the official anthropic library
try:
    import anthropic
    from anthropic import AsyncAnthropic, AnthropicError
    from anthropic.types import MessageParam, TextBlockParam # For constructing messages
    anthropic_available = True
except ImportError:
    anthropic_available = False
    AsyncAnthropic = None # type: ignore [assignment]
    AnthropicError = Exception # type: ignore [assignment]
    MessageParam = Dict[str, Any] # type: ignore [assignment]
    TextBlockParam = Dict[str, Any] # type: ignore [assignment]


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError # MCPError removed
from .base import BaseProvider, ContextPayload # ContextPayload is List[Message]

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
    Handles List[Message] context type.
    """
    _client: Optional[AsyncAnthropic] = None
    _sync_client_for_counting: Optional[anthropic.Anthropic] = None # For token counting

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AnthropicProvider.

        Args:
            config: Configuration dictionary from `[providers.anthropic]` containing:
                    'api_key' (optional): Anthropic API key. Defaults to env var ANTHROPIC_API_KEY.
                    'base_url' (optional): Custom Anthropic API endpoint URL.
                    'default_model' (optional): Default model to use.
                    'timeout' (optional): Request timeout in seconds.
        """
        if not anthropic_available:
            raise ImportError("Anthropic library is not installed. Please install `anthropic` or `llmcore[anthropic]`.")

        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = config.get('base_url')
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = float(config.get('timeout', 60.0))

        if not self.api_key:
            # This is a warning because the SDK might still work if the key is set
            # directly in the environment in a way the SDK picks up but os.environ.get doesn't.
            logger.warning("Anthropic API key not found in config or environment variable ANTHROPIC_API_KEY. "
                           "Ensure it is set for the provider to function.")

        try:
            # Initialize the asynchronous client for API calls
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            # Initialize a synchronous client specifically for token counting, as it's a sync operation
            self._sync_client_for_counting = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.debug("AsyncAnthropic and sync Anthropic clients initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic clients: {e}", exc_info=True)
            raise ConfigError(f"Anthropic client initialization failed: {e}")

    def get_name(self) -> str:
        """Returns the provider name: 'anthropic'."""
        return "anthropic"

    def get_available_models(self) -> List[str]:
        """Returns a static list of known default models for Anthropic."""
        # TODO: Consider implementing dynamic model listing if the Anthropic API supports it easily.
        logger.warning("AnthropicProvider.get_available_models() returning static list. "
                       "Refer to Anthropic documentation for the latest models.")
        return list(DEFAULT_ANTHROPIC_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given Anthropic model."""
        model_name = model or self.default_model
        limit = DEFAULT_ANTHROPIC_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Fallback for unknown models; Anthropic models generally have large context windows.
            limit = 100000 # Default to a common large context window
            logger.warning(f"Unknown context length for Anthropic model '{model_name}'. "
                           f"Using fallback limit: {limit}. Please verify with Anthropic documentation.")
        return limit

    def _convert_llmcore_msgs_to_anthropic(
        self,
        messages: List[Message]
    ) -> tuple[Optional[str], List[MessageParam]]:
        """
        Converts a list of LLMCore `Message` objects to the format expected by the Anthropic API.
        Extracts the system prompt and formats user/assistant messages.

        Args:
            messages: A list of `llmcore.models.Message` objects.

        Returns:
            A tuple containing:
            - An optional system prompt string.
            - A list of `MessageParam` dictionaries for the Anthropic API.
        """
        anthropic_messages: List[MessageParam] = []
        system_prompt: Optional[str] = None

        # Process messages, extracting system prompt first
        processed_messages = list(messages) # Make a copy to modify
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_prompt = processed_messages.pop(0).content
            logger.debug("System prompt extracted for Anthropic request.")

        # Anthropic API requires conversations to start with a 'user' role after system prompt.
        if not processed_messages or processed_messages[0].role != LLMCoreRole.USER:
            # This situation should ideally be handled by the ContextManager ensuring valid sequence.
            # If it still occurs, log a warning. The API call might fail.
            logger.warning("Anthropic conversation (after system prompt) should start with a 'user' role. "
                           "The current sequence might lead to API errors.")

        last_role_added_to_api = None
        for msg in processed_messages:
            anthropic_role_str = ""
            if msg.role == LLMCoreRole.USER:
                anthropic_role_str = "user"
            elif msg.role == LLMCoreRole.ASSISTANT:
                anthropic_role_str = "assistant"
            else:
                logger.warning(f"Skipping message with unmappable role for Anthropic: {msg.role}")
                continue

            # Anthropic API requires alternating user/assistant roles.
            # If consecutive messages have the same role, attempt to merge or log warning.
            if anthropic_role_str == last_role_added_to_api:
                logger.warning(f"Duplicate consecutive role '{anthropic_role_str}' for Anthropic API. "
                               "Attempting to merge content. This might not be ideal for all use cases.")
                if anthropic_messages and isinstance(anthropic_messages[-1]["content"], list):
                    # Assuming content is a list of TextBlockParam
                    last_content_block = anthropic_messages[-1]["content"][0] # type: ignore[index]
                    if last_content_block["type"] == "text": # type: ignore[typeddict-item]
                        last_content_block["text"] += f"\n{msg.content}" # type: ignore[typeddict-item]
                        continue # Skip adding a new message, content merged

            # Create the content block for the message
            content_block: TextBlockParam = {"type": "text", "text": msg.content}
            anthropic_messages.append({"role": anthropic_role_str, "content": [content_block]}) # type: ignore[typeddict-item]
            last_role_added_to_api = anthropic_role_str

        return system_prompt, anthropic_messages


    async def chat_completion(
        self,
        context: ContextPayload, # ContextPayload is List[Message]
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the Anthropic API.

        Args:
            context: The context payload to send, as a list of `llmcore.models.Message` objects.
            model: The specific model identifier to use. Defaults to provider's default.
            stream: If True, returns an async generator of response chunks. Otherwise, returns the full response.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A dictionary for full response or an async generator for streamed response.

        Raises:
            ProviderError: If the API call fails.
            ConfigError: If the provider is not properly configured.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Anthropic client not initialized.")

        model_name = model or self.default_model

        # Context is expected to be List[Message]
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"AnthropicProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        logger.debug("Processing context as List[Message] for Anthropic.")
        system_prompt, messages_payload = self._convert_llmcore_msgs_to_anthropic(context)

        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")

        # Anthropic requires the first message in the 'messages' list to be 'user'
        # (after any system prompt which is passed separately).
        if messages_payload[0]['role'] != 'user':
             logger.warning(f"Anthropic API requires the first message in 'messages' payload to be 'user'. "
                            f"Current first role: {messages_payload[0]['role']}. Attempting to proceed, but API may error.")
             # Optionally, one could try to prepend a dummy user message if it starts with assistant,
             # but this should ideally be handled by the ContextManager.
             # Example:
             # if messages_payload[0]['role'] == 'assistant':
             #     logger.info("Prepending dummy 'user' message as Anthropic requires conversation to start with user.")
             #     messages_payload.insert(0, {"role": "user", "content": [{"type": "text", "text": "(Context provided)"}]})


        # Anthropic uses 'max_tokens' for the response length limit.
        # 'max_tokens_to_sample' was an older parameter name.
        max_tokens_val = kwargs.pop("max_tokens", kwargs.pop("max_tokens_to_sample", 2048))

        logger.debug(
            f"Sending request to Anthropic API: model='{model_name}', stream={stream}, "
            f"num_messages={len(messages_payload)}, system_prompt_present={bool(system_prompt)}"
        )

        try:
            if stream:
                response_stream = await self._client.messages.stream(
                    model=model_name,
                    messages=messages_payload, # type: ignore [arg-type]
                    system=system_prompt,
                    max_tokens=max_tokens_val,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                    top_k=kwargs.get("top_k"),
                    stop_sequences=kwargs.get("stop_sequences")
                )
                logger.debug(f"Processing stream response from Anthropic model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    try:
                        async for event in response_stream:
                            event_dict = {}
                            # Map Anthropic stream events to a more generic OpenAI-like chunk structure if possible
                            if event.type == "message_start":
                                event_dict = {
                                    "type": "message_start",
                                    "message": { # Replicating OpenAI structure for consistency
                                        "id": event.message.id,
                                        "role": event.message.role, # Should be 'assistant'
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
                                        # Mimic OpenAI stream structure for easier processing in LLMCore.chat
                                        "choices": [{"delta": {"content": event.delta.text}, "index": event.index, "finish_reason": None}]
                                    }
                            elif event.type == "message_delta":
                                event_dict = {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": event.delta.stop_reason, "stop_sequence": event.delta.stop_sequence},
                                    "usage": event.usage.model_dump(),
                                    "choices": [{"finish_reason": event.delta.stop_reason}] # Add finish_reason to choices
                                }
                            elif event.type == "message_stop":
                                event_dict = {
                                    "type": "message_stop",
                                    "reason": "stop_event", # Generic stop reason
                                    "choices": [{"finish_reason": "stop"}] # Explicitly mark as stop
                                }

                            if event_dict:
                                yield event_dict
                    except anthropic.APIConnectionError as e_conn:
                        logger.error(f"Anthropic API connection error during stream: {e_conn}", exc_info=True)
                        yield {"error": f"Anthropic API Connection Error: {e_conn}", "done": True}
                        # No need to raise here, error is yielded
                    except anthropic.APIStatusError as e_stat:
                        logger.error(f"Anthropic API status error during stream: {e_stat.status_code} - {e_stat.message}", exc_info=True)
                        yield {"error": f"Anthropic API Status Error ({e_stat.status_code}): {e_stat.message}", "done": True}
                    except Exception as e_stream:
                        logger.error(f"Unexpected error processing Anthropic stream: {e_stream}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e_stream}", "done": True}
                    finally:
                        logger.debug("Anthropic stream finished.")
                return stream_wrapper()
            else: # Non-streaming
                response = await self._client.messages.create(
                    model=model_name,
                    messages=messages_payload, # type: ignore [arg-type]
                    system=system_prompt,
                    max_tokens=max_tokens_val,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                    top_k=kwargs.get("top_k"),
                    stop_sequences=kwargs.get("stop_sequences")
                )
                logger.debug(f"Processing non-stream response from Anthropic model '{model_name}'")
                # Convert Anthropic response to a dictionary similar to OpenAI's for consistency
                # This helps the LLMCore.chat method handle responses more uniformly.
                response_dict = response.model_dump()
                # Extract the primary text content
                main_content = ""
                if response_dict.get("content"):
                    for block in response_dict["content"]:
                        if block.get("type") == "text":
                            main_content += block.get("text", "")

                # Construct an OpenAI-like choices structure
                choices = [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": main_content
                    },
                    "finish_reason": response_dict.get("stop_reason")
                }]
                usage = response_dict.get("usage")

                return {
                    "id": response_dict.get("id"),
                    "model": response_dict.get("model"),
                    "choices": choices,
                    "usage": usage,
                    # Add other relevant fields if needed
                }

        except AnthropicError as e: # Catch base Anthropic SDK error
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

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int: # Changed to async
        """Counts tokens using the Anthropic client (synchronous call wrapped)."""
        if not self._sync_client_for_counting:
            logger.warning("Anthropic sync client for token counting not initialized. Using char approximation.")
            return (len(text) + 3) // 4 # Rough fallback
        if not text:
            return 0
        try:
            # Run the synchronous count_tokens in a thread
            return await asyncio.to_thread(self._sync_client_for_counting.count_tokens, text)
        except AnthropicError as e:
            logger.error(f"Anthropic API error during token count: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e}")
        except Exception as e:
            logger.error(f"Failed to count tokens with Anthropic client: {e}", exc_info=True)
            # Fallback to character approximation if API call fails
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int: # Changed to async
        """
        Counts tokens for a list of LLMCore Messages using the Anthropic client.
        This involves formatting messages as the API would expect them (approximately)
        and then counting tokens on the resulting string.
        """
        if not self._sync_client_for_counting:
            logger.warning("Anthropic sync client for token counting not initialized. Using char approximation.")
            total_text_len = sum(len(msg.content) + len(msg.role.value) for msg in messages)
            # Rough approximation for message overhead
            return (total_text_len + (len(messages) * 5)) // 4

        # Convert LLMCore messages to a string representation similar to how Anthropic might process them.
        # This is an approximation. For exact counts, Anthropic's API might be needed,
        # but that's too slow for frequent context checks.
        prompt_str = ""
        system_prompt, anthropic_msgs_list = self._convert_llmcore_msgs_to_anthropic(messages)

        if system_prompt:
            prompt_str += f"System: {system_prompt}\n\n" # Approximate system prompt formatting

        for msg_param in anthropic_msgs_list:
            role = msg_param["role"]
            content_str = ""
            if isinstance(msg_param["content"], list) and msg_param["content"]:
                first_block = msg_param["content"][0] # type: ignore[index]
                if first_block.get("type") == "text": # type: ignore[typeddict-item]
                    content_str = first_block.get("text", "") # type: ignore[typeddict-item]
            prompt_str += f"{str(role).capitalize()}: {content_str}\n" # e.g., User: Hello\nAssistant: Hi\n

        # Anthropic API might add a final "Assistant:" prompt implicitly.
        if prompt_str and anthropic_msgs_list and anthropic_msgs_list[-1]["role"] == "user":
            prompt_str += "Assistant:" # Simulate the prompt for the assistant's turn

        if not prompt_str:
            return 0

        # Use the async count_tokens method which wraps the sync call
        return await self.count_tokens(prompt_str, model)


    async def close(self) -> None:
        """Closes the underlying Anthropic client sessions."""
        closed_async = False
        if self._client:
            try:
                await self._client.close()
                logger.debug("AsyncAnthropic client closed.")
                closed_async = True
            except Exception as e:
                logger.error(f"Error closing AsyncAnthropic client: {e}", exc_info=True)

        if self._sync_client_for_counting:
            try:
                # The synchronous client's close method is not async
                self._sync_client_for_counting.close()
                logger.debug("Anthropic synchronous client for counting closed.")
            except Exception as e:
                logger.error(f"Error closing Anthropic synchronous client: {e}", exc_info=True)

        self._client = None
        self._sync_client_for_counting = None
