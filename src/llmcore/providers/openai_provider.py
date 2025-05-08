# src/llmcore/providers/openai_provider.py
"""
OpenAI API provider implementation for the LLMCore library.

Handles interactions with the OpenAI API (GPT models).
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import aiohttp
try:
    import openai
    from openai import AsyncOpenAI, OpenAIError
    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None # type: ignore
    OpenAIError = Exception # type: ignore

try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None # type: ignore

from ..models import Message, Role
from ..exceptions import ProviderError, ConfigError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common OpenAI models
# Source: https://platform.openai.com/docs/models
DEFAULT_OPENAI_TOKEN_LIMITS = {
    # GPT-4 Turbo models (Input + Output = 128k) - Input limit is typically 128k
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000, # Includes image tokens
    # GPT-4 models
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    # GPT-4o models (Input + Output = 128k)
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    # GPT-3.5 Turbo models (Input + Output = 16k) - Input limit is typically 16k
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-1106": 16385,
    # Older GPT-3.5 Turbo (4k context)
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16385, # 16k version
    # Add other models as needed
}

# Default model if not specified
DEFAULT_MODEL = "gpt-4o"

class OpenAIProvider(BaseProvider):
    """
    LLMCore provider for interacting with the OpenAI API.
    """
    _client: Optional[AsyncOpenAI] = None
    _encoding: Optional[Any] = None # tiktoken encoding object

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OpenAIProvider.

        Args:
            config: Configuration dictionary containing:
                    'api_key' (optional): OpenAI API key. Defaults to env var OPENAI_API_KEY.
                    'base_url' (optional): Custom OpenAI API endpoint URL.
                    'default_model' (optional): Default model to use (e.g., "gpt-4o").
                    'timeout' (optional): Request timeout in seconds (default: 60).
        """
        if not openai_available:
            raise ImportError("OpenAI library is not installed. Please install `openai`.")
        if not tiktoken_available:
            raise ImportError("tiktoken library is not installed. Please install `tiktoken`.")

        self.api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        self.base_url = config.get('base_url') # Allow None for default OpenAI URL
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = float(config.get('timeout', 60.0))

        if not self.api_key:
            # Defer raising error until API call if key might be available through other means
            # (e.g., Azure OpenAI credential management)
            logger.warning("OpenAI API key not found in config or environment variable OPENAI_API_KEY.")
            # self.client remains None

        # Initialize the AsyncOpenAI client
        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url, # Pass None to use default OpenAI URL
                timeout=self.timeout,
            )
            logger.debug("AsyncOpenAI client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise ConfigError(f"OpenAI client initialization failed: {e}")

        # Initialize tokenizer encoding
        self._load_tokenizer(self.default_model)

    def _load_tokenizer(self, model_name: str):
        """Loads the tiktoken tokenizer for the specified model."""
        if not tiktoken: return # Should have been checked in __init__
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
            logger.debug(f"Loaded tiktoken encoding for model: {model_name}")
        except KeyError:
            logger.warning(f"No specific tiktoken encoding found for model '{model_name}'. "
                           f"Using default 'cl100k_base'.")
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding: {e}", exc_info=True)
            self._encoding = None # Indicate tokenizer failure

    def get_name(self) -> str:
        """Returns the provider name."""
        return "openai"

    def get_available_models(self) -> List[str]:
        """
        Returns a list of known default models for OpenAI.
        Note: Does not dynamically fetch from the API.
        """
        # TODO: Implement dynamic fetching via API if needed, requires async handling
        logger.warning("OpenAIProvider.get_available_models() returning static list.")
        return list(DEFAULT_OPENAI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length for the given OpenAI model."""
        model_name = model or self.default_model
        limit = DEFAULT_OPENAI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Attempt to parse model name for known prefixes
            if model_name.startswith("gpt-4o"): limit = 128000
            elif model_name.startswith("gpt-4-turbo"): limit = 128000
            elif model_name.startswith("gpt-4-32k"): limit = 32768
            elif model_name.startswith("gpt-4"): limit = 8192
            elif model_name.startswith("gpt-3.5-turbo-16k"): limit = 16385
            elif model_name.startswith("gpt-3.5-turbo"): limit = 16385 # Default to 16k for newer 3.5
            else:
                limit = 4096 # Fallback default
                logger.warning(f"Unknown context length for OpenAI model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Sends a chat completion request to the OpenAI API."""
        if not self._client:
            raise ProviderError(self.get_name(), "OpenAI client not initialized. API key might be missing.")
        if not isinstance(context, list):
            raise ProviderError(self.get_name(), f"OpenAIProvider received unsupported context type: {type(context).__name__}")

        model_name = model or self.default_model
        messages_payload = [{"role": msg.role.value, "content": msg.content} for msg in context]

        logger.debug(f"Sending request to OpenAI API: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")
        # logger.debug(f"Payload kwargs: {kwargs}")

        try:
            response_stream = await self._client.chat.completions.create(
                model=model_name,
                messages=messages_payload, # type: ignore # Correct type hint for messages
                stream=stream,
                **kwargs
            )

            if stream:
                logger.debug(f"Processing stream response from OpenAI model '{model_name}'")
                # We need to wrap the stream from the openai library
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    async for chunk in response_stream: # type: ignore
                        # Convert the Pydantic model chunk to a dictionary
                        yield chunk.model_dump()
                return stream_wrapper()
            else:
                logger.debug(f"Processing non-stream response from OpenAI model '{model_name}'")
                # Convert the Pydantic model response to a dictionary
                return response_stream.model_dump() # type: ignore

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e.status_code} - {e.message}", exc_info=True)
            # Map specific OpenAI errors to ProviderError if needed
            raise ProviderError(self.get_name(), f"API Error ({e.status_code}): {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to OpenAI timed out after {self.timeout} seconds.")
            raise ProviderError(self.get_name(), f"Request timed out after {self.timeout}s.")
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the tiktoken tokenizer."""
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for OpenAIProvider. Using character approximation.")
            return (len(text) + 3) // 4 # Rough fallback

        if not text:
            return 0

        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.error(f"Tiktoken encoding failed: {e}", exc_info=True)
            # Fallback to approximation if tiktoken fails unexpectedly
            return (len(text) + 3) // 4

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Counts tokens for a list of messages using tiktoken, including OpenAI's overhead.

        Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for token counting. Returning 0.")
            return 0

        model_name = model or self.default_model
        # Adjust model name for known tokenizer variations if necessary
        if model_name in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model_name == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model_name:
            # Covers newer 3.5 models like -1106, -0125
            # logger.debug("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            tokens_per_message = 3
            tokens_per_name = 1
        elif "gpt-4" in model_name:
            # Covers newer GPT-4 models like -turbo, -vision, -o
            # logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            # Fallback or raise error for unknown models
            logger.warning(f"count_message_tokens() may not be accurate for model {model_name}. Using fallback settings.")
            tokens_per_message = 3
            tokens_per_name = 1
            # raise NotImplementedError(f"count_message_tokens() is not implemented for model {model_name}.")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            # Assuming 'name' field is not used in our Message model for now
            # If it were, add tokens_per_name if name exists
            num_tokens += len(self._encoding.encode(message.role.value))
            num_tokens += len(self._encoding.encode(message.content))

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def close(self) -> None:
        """Closes the underlying OpenAI client session if applicable."""
        if self._client:
            try:
                # The openai library >= 1.0 manages its own sessions implicitly
                # but provides close() for explicit cleanup if needed (though often not required).
                # await self._client.close() # This method might not exist or be needed
                logger.debug("OpenAI client does not require explicit closing in recent versions.")
                pass
            except Exception as e:
                logger.error(f"Error closing OpenAI client (may be harmless): {e}", exc_info=True)
        self._client = None # Clear the client reference
