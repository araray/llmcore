# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ResponseError, ChatResponse
    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None # type: ignore
    ResponseError = Exception # type: ignore
    ChatResponse = None # type: ignore


# Keep tiktoken for token counting
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

# Default context lengths for common Ollama models (can be overridden by API in future)
# Source: Often based on the underlying model's known limits (e.g., Llama 3, Mistral)
# Keep this map as the ollama library might not provide context lengths directly.
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8192,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "falcon3:3b": 4096, # Added based on user log context length warning
    "mistral": 8192,         # Mistral 7B v0.1
    "mistral:7b": 8192,
    "mixtral": 32768,        # Mixtral 8x7B
    "mixtral:8x7b": 32768,
    "gemma": 8192,           # Gemma 7B
    "gemma:7b": 8192,
    "gemma:2b": 8192,        # Gemma 2B also often uses 8k context
    "phi3": 4096,            # Phi-3 Mini 4k context default
    "phi3:mini": 4096,
    "codellama": 16384,      # CodeLlama models often have larger contexts
    "codellama:7b": 16384,
    "codellama:13b": 16384,
    "codellama:34b": 16384,
    "llama2": 4096,          # Llama 2 default context
    "llama2:7b": 4096,
    "llama2:13b": 4096,
    "llama2:70b": 4096,
    # Add other common models or rely on dynamic fetching if available
}

# Default model if not specified in config
DEFAULT_MODEL = "llama3"

class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama using the official library.
    """
    _client: Optional[AsyncClient] = None
    _encoding: Optional[Any] = None # tiktoken encoding object

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OllamaProvider using the official ollama library.

        Args:
            config: Configuration dictionary containing:
                    'host' (optional): Hostname/IP of the Ollama server (default: http://localhost:11434).
                                       The library handles adding '/api'.
                    'default_model' (optional): Default Ollama model to use (e.g., "llama3").
                    'timeout' (optional): Request timeout in seconds (default: library default).
                    'tokenizer' (optional): Tokenizer for counting ('tiktoken_cl100k_base',
                                            'tiktoken_p50k_base', 'char_div_4'). Default: 'tiktoken_cl100k_base'.
        """
        if not ollama_available:
            raise ImportError("Ollama library is not installed. Please install `ollama`.")
        if not tiktoken_available:
            # Keep tiktoken requirement for now, as ollama lib doesn't count tokens
            raise ImportError("tiktoken library is not installed but required for token counting. Please install `tiktoken`.")

        # The ollama library uses 'host' which defaults internally if None
        self.host = config.get("host") # Can be None to use library default
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        # Timeout can be passed directly to client methods if needed, or set on client init
        self.timeout = config.get("timeout") # Can be None

        # Initialize the AsyncClient
        try:
            # Pass timeout during client initialization if provided
            client_args = {}
            if self.host:
                client_args['host'] = self.host
            if self.timeout:
                client_args['timeout'] = float(self.timeout)

            self._client = AsyncClient(**client_args)
            logger.debug(f"Ollama AsyncClient initialized (Host: {self.host or 'default'})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama AsyncClient: {e}", exc_info=True)
            raise ConfigError(f"Ollama client initialization failed: {e}")

        # --- Tokenizer Setup (using tiktoken) ---
        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base")
        self._encoding = None
        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken:
                try:
                    encoding_name = self.tokenizer_name.split("tiktoken_")[1]
                    self._encoding = tiktoken.get_encoding(encoding_name)
                    logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name}")
                except Exception as e:
                    logger.warning(f"Failed to load tiktoken encoding '{self.tokenizer_name}'. "
                                   f"Falling back to character approximation. Error: {e}")
                    self.tokenizer_name = "char_div_4" # Fallback explicitly
            else:
                logger.warning("tiktoken library not available. Falling back to character approximation for Ollama.")
                self.tokenizer_name = "char_div_4" # Fallback explicitly

        elif self.tokenizer_name != "char_div_4":
            logger.warning(f"Unsupported Ollama tokenizer '{self.tokenizer_name}'. "
                           "Falling back to 'char_div_4' approximation.")
            self.tokenizer_name = "char_div_4"

        if self.tokenizer_name == "char_div_4":
             logger.info("OllamaProvider using character division approximation for token counting.")
        # --- End Tokenizer Setup ---

    def get_name(self) -> str:
        """Returns the provider name."""
        return "ollama"

    async def _fetch_ollama_models(self) -> List[str]:
        """Fetches the list of models available from the Ollama API using the client."""
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")
        try:
            models_info = await self._client.list()
            models = [m.get("name") for m in models_info.get("models", []) if m.get("name")]
            logger.debug(f"Fetched {len(models)} models from Ollama API via client: {models}")
            return models
        except ResponseError as e:
            logger.warning(f"Failed to fetch models from Ollama API: HTTP {e.status_code} - {e.error}")
            return [] # Return empty list on error
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True)
            return []

    def get_available_models(self) -> List[str]:
        """
        Returns a static list of known default models.
        Note: Use an async method like `_fetch_ollama_models` for dynamic fetching.
        """
        logger.warning("OllamaProvider.get_available_models() returning static list. "
                       "Use an async method for dynamic fetching if needed.")
        return list(DEFAULT_OLLAMA_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the estimated maximum context length for the given Ollama model."""
        # The ollama library doesn't seem to provide this directly yet. Use static map.
        model_name = model or self.default_model
        base_model_name = model_name.split(':')[0] # Base model name (e.g., "llama3" from "llama3:8b")

        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(model_name)
        if limit is None:
            limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)
        if limit is None:
             # Check user log for specific model context length warning
             if model_name == "falcon3:3b": # From user log
                 logger.info(f"Using context length 4096 for Ollama model '{model_name}' based on fallback.")
                 limit = 4096
             else:
                 limit = 4096 # Default fallback
                 logger.warning(f"Unknown context length for Ollama model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Sends a chat completion request to the Ollama API using the ollama library."""
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")
        if not isinstance(context, list) or not all(isinstance(msg, Message) for msg in context):
            raise ProviderError(self.get_name(), f"OllamaProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        model_name = model or self.default_model
        # Ensure msg.role is used directly as it's already a string due to Role(str, Enum)
        messages_payload = [{"role": msg.role, "content": msg.content} for msg in context]


        logger.debug(f"Sending request to Ollama via client: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")

        try:
            # Pass kwargs directly to the 'options' parameter if they exist
            options = kwargs if kwargs else None

            # Call the client's chat method
            response_or_stream_obj: Union[ChatResponse, AsyncGenerator[Dict[str, Any], None]] = await self._client.chat(
                model=model_name,
                messages=messages_payload, # type: ignore
                stream=stream,
                options=options,
                # Timeout can be passed here if needed, overriding client default
                # timeout=self.timeout if self.timeout else None
            )

            if stream:
                logger.debug(f"Processing stream response from Ollama model '{model_name}'")
                # The ollama library stream yields dictionaries directly
                return response_or_stream_obj # type: ignore
            else:
                logger.debug(f"Processing non-stream response from Ollama model '{model_name}'")
                # The non-stream response is an ollama.ChatResponse object. Convert to dict.
                if isinstance(response_or_stream_obj, dict): # Should not happen based on type hint, but defensive
                    return response_or_stream_obj
                if ChatResponse and isinstance(response_or_stream_obj, ChatResponse):
                    # Convert the Pydantic model to a dictionary
                    return response_or_stream_obj.model_dump()
                else:
                    # This case should ideally not be reached if types are correct
                    logger.error(f"Unexpected response type for non-streaming Ollama chat: {type(response_or_stream_obj)}")
                    raise ProviderError(self.get_name(), "Invalid or unexpected response format from Ollama client (non-streaming).")


        except ResponseError as e:
            error_detail = e.error if hasattr(e, 'error') else str(e)
            logger.error(f"Ollama API error: {e.status_code} - {error_detail}", exc_info=True)
            # Check for model not found error specifically
            if error_detail and "model not found" in str(error_detail).lower():
                raise ProviderError(self.get_name(), f"Model '{model_name}' not found. Pull it using 'ollama pull {model_name}'.")
            raise ProviderError(self.get_name(), f"API Error ({e.status_code}): {error_detail}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Ollama timed out (timeout: {self.timeout or 'default'}).")
            raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e:
            # Catch potential connection errors or other unexpected issues
            logger.error(f"Unexpected error during Ollama chat completion: {e}", exc_info=True)
            # Check if it looks like a connection error
            if "connect" in str(e).lower():
                 raise ProviderError(self.get_name(), f"Could not connect to Ollama at {self.host or 'default'}. Is it running? Details: {e}")
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")


    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the configured tokenizer (tiktoken) or approximation."""
        # The ollama library does not provide token counting. Use tiktoken.
        if not self._encoding:
            # Fallback to character approximation if tiktoken failed to load
            return (len(text) + 3) // 4 if text else 0

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
        Counts tokens for a list of messages using the configured method (tiktoken).
        Applies a small heuristic overhead per message.
        """
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for token counting. Using approximation.")
            # Approximate based on character count if no encoder
            total_chars = sum(len(msg.content) for msg in messages)
            return (total_chars + (len(messages) * 15)) // 4 # Rough approximation with overhead

        # Use tiktoken with heuristic overhead
        overhead_per_message = 5 # Rough estimate for role/formatting markers
        total_tokens = 0

        for msg in messages:
            try:
                content_tokens = len(self._encoding.encode(msg.content))
                # Since Role(str, Enum), msg.role is already the string value.
                role_str = str(msg.role) # Ensure it's a string if it wasn't already.
                role_tokens = len(self._encoding.encode(role_str)) # Count role tokens too
                total_tokens += content_tokens + role_tokens + overhead_per_message
            except Exception as e:
                logger.error(f"Tiktoken encoding failed for message content/role: {e}. Using approximation for message.")
                # Use str(msg.role) for approximation as well
                role_str_for_approx = str(msg.role)
                total_tokens += (len(msg.content) + len(role_str_for_approx) + 15) // 4


        total_tokens += 3 # Add a small buffer for the overall structure/prompting

        return total_tokens

    async def close(self) -> None:
        """Closes the underlying Ollama client session if applicable."""
        # The ollama library's AsyncClient uses httpx internally.
        # httpx.AsyncClient should be closed, typically via `await client.aclose()`
        # or by using the client as an async context manager.
        # The ollama.AsyncClient itself does not seem to expose an `aclose()` method directly.
        # It's often expected that the client is closed when it's garbage collected
        # or if the application uses it as an async context manager.
        # Given the error `AttributeError: 'AsyncClient' object has no attribute 'close'`,
        # we should remove the explicit call if the library doesn't provide it.
        if self._client:
            logger.debug("Ollama AsyncClient closure is typically handled by garbage collection "
                         "or by using it as an async context manager. No explicit close action taken by provider.")
            # If ollama.AsyncClient had an `aclose` method, it would be:
            # try:
            #     await self._client.aclose() # Assuming an aclose method exists
            #     logger.debug("Ollama AsyncClient closed.")
            # except Exception as e:
            #     logger.error(f"Error closing Ollama client: {e}", exc_info=True)
        self._client = None # Clear the client reference
