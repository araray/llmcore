# src/llmcore/providers/openai_provider.py
"""
OpenAI API provider implementation for the LLMCore library.

Handles interactions with the OpenAI API (GPT models).
Accepts context as List[Message].
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from enum import Enum # Import Enum for isinstance check


# Removed aiohttp as it's not directly used here; openai SDK handles HTTP.
try:
    import openai
    from openai import AsyncOpenAI, OpenAIError
    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None # type: ignore [assignment]
    OpenAIError = Exception # type: ignore [assignment]

try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None # type: ignore [assignment]


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for various OpenAI models
DEFAULT_OPENAI_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4-turbo": 128000, # Covers gpt-4-turbo-2024-04-09
    "gpt-4-turbo-preview": 128000, # Alias for newer turbo models
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000, # Vision model, context includes image tokens
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-0125": 16385, # Updated 3.5 turbo
    "gpt-3.5-turbo": 16385, # Often points to the latest 3.5 turbo version
    "gpt-3.5-turbo-1106": 16385, # Older 16k variant
    # Older models, less common now but kept for reference
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16385, # Alias for gpt-3.5-turbo-16k-0613
    "gpt-3.5-turbo-16k-0613": 16385,
}
DEFAULT_MODEL = "gpt-4o" # Updated to the latest recommended model


class OpenAIProvider(BaseProvider):
    """
    LLMCore provider for interacting with the OpenAI API.
    Handles List[Message] context type.
    """
    _client: Optional[AsyncOpenAI] = None
    _encoding: Optional[Any] = None # tiktoken encoding object

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OpenAIProvider.

        Args:
            config: Configuration dictionary from `[providers.openai]` containing:
                    'api_key' (optional): OpenAI API key. Defaults to env var OPENAI_API_KEY.
                    'base_url' (optional): Custom OpenAI API endpoint URL.
                    'default_model' (optional): Default model to use (e.g., "gpt-4o").
                    'timeout' (optional): Request timeout in seconds (default: 60).
        """
        if not openai_available:
            raise ImportError("OpenAI library is not installed. Please install `openai` or `llmcore[openai]`.")
        if not tiktoken_available:
            # tiktoken is essential for accurate token counting with OpenAI models.
            raise ImportError("tiktoken library is not installed. Please install `tiktoken` for OpenAI provider.")

        self.api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        self.base_url = config.get('base_url') # Can be None, SDK uses default
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = float(config.get('timeout', 60.0)) # Default timeout 60s

        if not self.api_key:
            # This is a warning because the SDK might still work if the key is set
            # directly in the environment in a way the SDK picks up but os.environ.get doesn't.
            logger.warning("OpenAI API key not found in config or environment variable OPENAI_API_KEY. "
                           "Ensure it is set for the provider to function.")

        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key, # Pass None if not set, SDK handles env var
                base_url=self.base_url, # Pass None for default URL
                timeout=self.timeout,
            )
            logger.debug("AsyncOpenAI client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise ConfigError(f"OpenAI client initialization failed: {e}")

        # Load tokenizer based on the default model for this provider instance
        self._load_tokenizer(self.default_model)

    def _load_tokenizer(self, model_name: str):
        """
        Loads the tiktoken tokenizer for the specified OpenAI model.
        This method is called during initialization and potentially if the model changes.
        """
        if not tiktoken: # Should be caught by __init__ check, but defensive
            logger.error("tiktoken library not available for _load_tokenizer.")
            self._encoding = None
            return
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
            logger.debug(f"Loaded tiktoken encoding for model: {model_name}")
        except KeyError:
            # Fallback for models not explicitly known by tiktoken, e.g., newer ones.
            # cl100k_base is a common encoding for GPT-3.5 and GPT-4 series.
            logger.warning(f"No specific tiktoken encoding found for model '{model_name}'. "
                           "Using default 'cl100k_base'. This might affect token count accuracy for some models.")
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding for model '{model_name}': {e}", exc_info=True)
            self._encoding = None # Ensure encoding is None if loading fails

    def get_name(self) -> str:
        """Returns the provider name: 'openai'."""
        return "openai"

    def get_available_models(self) -> List[str]:
        """
        Returns a static list of known OpenAI models.
        For a dynamic list, an API call would be needed (e.g., client.models.list()).
        """
        # TODO: Consider implementing dynamic model listing via self._client.models.list()
        # if performance allows and it's deemed necessary.
        logger.warning("OpenAIProvider.get_available_models() returning static list. "
                       "Refer to OpenAI documentation for the latest models.")
        return list(DEFAULT_OPENAI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given OpenAI model."""
        model_name = model or self.default_model
        limit = DEFAULT_OPENAI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Fallback logic for models not in the static list (e.g., fine-tuned models or new ones)
            # This is a heuristic. For precise limits, OpenAI's documentation or model details API is needed.
            if "gpt-4o" in model_name: limit = 128000
            elif "gpt-4-turbo" in model_name: limit = 128000
            elif "gpt-4-32k" in model_name: limit = 32768
            elif "gpt-4" in model_name: limit = 8192
            elif "gpt-3.5-turbo-16k" in model_name or "gpt-3.5-turbo-0125" in model_name or "gpt-3.5-turbo-1106" in model_name:
                limit = 16385
            elif "gpt-3.5-turbo" in model_name: # Catches older 4k gpt-3.5-turbo
                limit = 4096
            else: # General fallback
                limit = 4096
                logger.warning(f"Unknown context length for OpenAI model '{model_name}'. "
                               f"Using fallback limit: {limit}. Please verify with OpenAI documentation.")
        return limit

    async def chat_completion(
        self,
        context: ContextPayload, # ContextPayload is List[Message]
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the OpenAI API.
        Accepts context as a list of LLMCore Message objects.

        Args:
            context: The context payload to send, as a list of `llmcore.models.Message` objects.
            model: The specific model identifier to use. Defaults to provider's default.
            stream: If True, returns an async generator of response chunks. Otherwise, returns the full response.
            **kwargs: Additional provider-specific parameters (e.g., temperature, max_tokens).

        Returns:
            A dictionary for full response or an async generator for streamed response.

        Raises:
            ProviderError: If the API call fails.
            ConfigError: If the provider is not properly configured (e.g., API key issue).
        """
        if not self._client:
            # This typically means API key was not configured properly or client init failed.
            raise ProviderError(self.get_name(), "OpenAI client not initialized. "
                                "Ensure API key is set in config or environment (OPENAI_API_KEY).")

        model_name = model or self.default_model

        # Context is expected to be List[Message]
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"OpenAIProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        # Convert LLMCore Messages to OpenAI's expected format
        messages_payload: List[Dict[str, str]] = []
        for msg in context:
            # Ensure role is a string value
            role_str = msg.role.value if isinstance(msg.role, Enum) else str(msg.role)
            messages_payload.append({"role": role_str, "content": msg.content})


        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")

        logger.debug(f"Sending request to OpenAI API: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")

        try:
            # Use the client's chat.completions.create method
            response_or_stream = await self._client.chat.completions.create(
                model=model_name,
                messages=messages_payload, # type: ignore [arg-type] # SDK expects List[ChatCompletionMessageParam]
                stream=stream,
                **kwargs # Pass through other OpenAI specific params
            )

            if stream:
                logger.debug(f"Processing stream response from OpenAI model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    async for chunk in response_or_stream: # type: ignore [misc] # response_or_stream is AsyncStream[ChatCompletionChunk]
                        # Convert Pydantic model chunk to dict for consistent output
                        yield chunk.model_dump(exclude_none=True)
                return stream_wrapper()
            else:
                # For non-streaming, response_or_stream is a ChatCompletion object
                logger.debug(f"Processing non-stream response from OpenAI model '{model_name}'")
                # Convert Pydantic model response to dict
                return response_or_stream.model_dump(exclude_none=True) # type: ignore [union-attr]

        except OpenAIError as e: # Catch specific OpenAI SDK errors
            logger.error(f"OpenAI API error: Status {e.status_code} - {e.message}", exc_info=True)
            # More specific error handling can be added here based on e.type or e.code
            if e.status_code == 401: # Authentication error
                raise ProviderError(self.get_name(), f"Authentication failed (Invalid API Key? Status 401): {e.message}")
            if e.status_code == 429: # Rate limit error
                raise ProviderError(self.get_name(), f"Rate limit exceeded (Status 429): {e.message}")
            raise ProviderError(self.get_name(), f"OpenAI API Error (Status {e.status_code}): {e.message}")
        except asyncio.TimeoutError: # More specific than general Exception for timeouts
            logger.error(f"Request to OpenAI API timed out after {self.timeout} seconds.")
            raise ProviderError(self.get_name(), f"Request timed out after {self.timeout}s.")
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during OpenAI chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Counts tokens for a given text string using the tiktoken tokenizer.
        This method is asynchronous to maintain consistency with the base class,
        though tiktoken itself is synchronous.
        """
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for OpenAIProvider. "
                           "Using rough character-based approximation for token counting.")
            return (len(text) + 3) // 4
        if not text:
            return 0

        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.error(f"Tiktoken encoding failed for text: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Counts tokens for a list of LLMCore Messages using tiktoken,
        including OpenAI's specific overhead per message and role.
        This method is asynchronous for consistency.
        """
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for OpenAIProvider message token counting. "
                           "Using rough character-based approximation.")
            total_chars = 0
            for msg in messages:
                role_str = msg.role.value if isinstance(msg.role, LLMCoreRole) else str(msg.role)
                total_chars += len(msg.content) + len(role_str)
            return (total_chars + (len(messages) * 15)) // 4

        model_name = model or self.default_model

        if model_name in {
            "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model_name == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        elif "gpt-3.5-turbo" in model_name or "gpt-4" in model_name or "gpt-4o" in model_name:
            logger.debug(f"Using modern token counting for model {model_name} (3 tokens_per_message, 1 token_per_name).")
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            logger.warning(f"count_message_tokens() may not be accurate for model {model_name}. "
                           "Using fallback token counting settings (3 tokens_per_message, 1 token_per_name). "
                           "Consult OpenAI documentation for precise counting for this model.")
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            try:
                num_tokens += tokens_per_message
                # Correctly handle message.role whether it's an Enum or a string
                role_str = message.role.value if isinstance(message.role, LLMCoreRole) else str(message.role)
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
                # if message.name: # LLMCore.Message does not have a 'name' field
                #    num_tokens += tokens_per_name
            except Exception as e:
                 logger.error(f"Tiktoken encoding failed for message content/role during count_message_tokens: {e}. "
                              "Using character approximation for this message.")
                 role_str_for_approx = message.role.value if isinstance(message.role, LLMCoreRole) else str(message.role)
                 num_tokens += (len(message.content) + len(role_str_for_approx) + 15) // 4


        num_tokens += 3
        return num_tokens

    async def close(self) -> None:
        """Closes the underlying OpenAI client session if applicable."""
        if self._client:
            try:
                await self._client.close()
                logger.info("OpenAIProvider client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OpenAIProvider client: {e}", exc_info=True)
            finally:
                self._client = None
