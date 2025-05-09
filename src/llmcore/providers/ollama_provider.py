# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
Accepts context as List[Message].
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from enum import Enum # Import Enum for isinstance check

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ResponseError, ChatResponse # ChatResponse for type checking
    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None # type: ignore [assignment]
    ResponseError = Exception # type: ignore [assignment]
    ChatResponse = None # type: ignore [assignment]

# Keep tiktoken for token counting
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None # type: ignore [assignment]


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError # MCPError removed
from .base import BaseProvider, ContextPayload # ContextPayload is List[Message]

logger = logging.getLogger(__name__)

# Default context lengths for common Ollama models
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8192, "llama3:8b": 8192, "llama3:70b": 8192,
    "gemma3:4b": 4096, # Added from user log in previous version, confirm if still relevant or if gemma is preferred
    "gemma:latest": 8192, "gemma:7b": 8192, "gemma:2b": 8192, # Common Gemma models
    "mistral": 8192, "mistral:7b": 8192, # Common Mistral model
    "mixtral": 32768, "mixtral:8x7b": 32768,
    "phi3": 4096, "phi3:mini": 4096,
    "codellama": 16384, # General codellama, specific sizes below
    "codellama:7b": 16384, "codellama:13b": 16384, "codellama:34b": 16384,
    "llama2": 4096, # General llama2, specific sizes below
    "llama2:7b": 4096, "llama2:13b": 4096, "llama2:70b": 4096,
    # Add other popular models if known context lengths are available
}
DEFAULT_MODEL = "llama3" # A common and capable default for Ollama


class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama using the official ollama library.
    Handles List[Message] context type.
    """
    _client: Optional[AsyncClient] = None
    _encoding: Optional[Any] = None # tiktoken encoding object
    tokenizer_name: str # Added to store the name of the tokenizer for logging

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OllamaProvider using the official ollama library.

        Args:
            config: Configuration dictionary from `[providers.ollama]` containing:
                    'host' (optional): Host URL for the Ollama server (e.g., "http://localhost:11434").
                                       Defaults to the ollama library's default.
                    'default_model' (optional): Default Ollama model to use.
                    'timeout' (optional): Request timeout in seconds.
                    'tokenizer' (optional): Tokenizer to use ('tiktoken_cl100k_base', 'tiktoken_p50k_base', 'char_div_4').
                                            Defaults to 'tiktoken_cl100k_base'.
        """
        if not ollama_available:
            raise ImportError("Ollama library is not installed. Please install `ollama` or `llmcore[ollama]`.")

        # tiktoken is preferred for token counting.
        if not tiktoken_available:
            logger.warning("tiktoken library not available. Ollama token counting will use character approximation.")
            # No need to raise ImportError here if char_div_4 is a fallback.
            # However, if tiktoken is a hard requirement for any configured tokenizer, then raise.

        self.host = config.get("host") # Can be None, ollama library handles default
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        timeout_val = config.get("timeout")
        self.timeout = float(timeout_val) if timeout_val is not None else None # ollama lib handles default timeout if None

        try:
            client_args = {}
            if self.host:
                client_args['host'] = self.host
            if self.timeout is not None: # Pass timeout only if explicitly set
                client_args['timeout'] = self.timeout

            self._client = AsyncClient(**client_args)
            logger.debug(f"Ollama AsyncClient initialized (Host: {self.host or 'default library host'}, Timeout: {self.timeout or 'default library timeout'})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama AsyncClient: {e}", exc_info=True)
            raise ConfigError(f"Ollama client initialization failed: {e}")

        # Tokenizer configuration
        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base")
        self._encoding = None
        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken_available and tiktoken: # Ensure tiktoken is usable
                try:
                    encoding_name = self.tokenizer_name.split("tiktoken_")[1]
                    self._encoding = tiktoken.get_encoding(encoding_name)
                    logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name} for token counting.")
                except Exception as e:
                    logger.warning(f"Failed to load tiktoken encoding '{self.tokenizer_name}'. "
                                   f"Falling back to character approximation for token counting. Error: {e}")
                    self.tokenizer_name = "char_div_4" # Fallback identifier
            else:
                logger.warning(f"tiktoken library not available, but '{self.tokenizer_name}' was configured. "
                               "Falling back to character approximation for token counting.")
                self.tokenizer_name = "char_div_4" # Fallback identifier
        elif self.tokenizer_name != "char_div_4":
            logger.warning(f"Unsupported Ollama tokenizer '{self.tokenizer_name}' configured. "
                           "Falling back to character approximation ('char_div_4') for token counting.")
            self.tokenizer_name = "char_div_4" # Fallback identifier

        if self.tokenizer_name == "char_div_4" and not self._encoding: # Log if char_div_4 is the final choice
            logger.info("OllamaProvider using character division approximation for token counting.")


    def get_name(self) -> str:
        """Returns the provider name: 'ollama'."""
        return "ollama"

    async def _fetch_ollama_models_from_api(self) -> List[str]:
        """Helper to fetch model list from Ollama API."""
        if not self._client:
            # This should not happen if __init__ was successful
            raise ProviderError(self.get_name(), "Ollama client not initialized for fetching models.")
        try:
            models_info = await self._client.list() # Uses the AsyncClient
            # ollama library returns a list of model details dictionaries
            models = [m.get("name") for m in models_info.get("models", []) if m.get("name")] # type: ignore[union-attr]
            logger.debug(f"Fetched {len(models)} models from Ollama API via client: {models}")
            return models
        except ResponseError as e:
            logger.warning(f"Failed to fetch models from Ollama API: HTTP {e.status_code} - {e.error}. "
                           "get_available_models() will return a static list.")
            return [] # Return empty on API error, fallback to static list
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True)
            return [] # Return empty on unexpected error


    def get_available_models(self) -> List[str]:
        """
        Returns a list of available models.
        Tries to fetch from Ollama API, falls back to a static list if API call fails.
        Note: This base class method is synchronous. The async fetch is for internal use or future refactor.
        For now, consistent with other providers, it returns a static list.
        """
        # TODO: Consider making this async and calling _fetch_ollama_models_from_api
        # This would require a change in the BaseProvider interface or how it's used.
        logger.warning("OllamaProvider.get_available_models() currently returning static list. "
                       "For dynamic list, an API call would be needed (see _fetch_ollama_models_from_api).")
        return list(DEFAULT_OLLAMA_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given Ollama model."""
        model_name = model or self.default_model
        # Ollama model names can be like 'llama3:latest' or 'llama3:8b'.
        # Try to find a match for the full name first, then the base model name.
        base_model_name = model_name.split(':')[0] # e.g., 'llama3' from 'llama3:8b'

        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(model_name)
        if limit is None: # If full name not found, try base name
            limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)

        if limit is None:
            # Fallback for unknown models
            # Common Ollama models often have context windows like 4096 or 8192.
            # Some newer models (like certain Llama3 versions) can be much larger.
            # A general safe fallback might be 4096.
            limit = 4096
            logger.warning(f"Unknown context length for Ollama model '{model_name}'. "
                           f"Using fallback limit: {limit}. Verify with 'ollama show {model_name}'.")
        return limit


    async def chat_completion(
        self,
        context: ContextPayload, # ContextPayload is List[Message]
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the Ollama API using the official ollama library.

        Args:
            context: The context payload to send, as a list of `llmcore.models.Message` objects.
            model: The specific Ollama model to use. Defaults to provider's default.
            stream: If True, returns an async generator of response chunks. Otherwise, returns the full response.
            **kwargs: Additional Ollama-specific options (e.g., temperature, top_p).

        Returns:
            A dictionary for full response or an async generator for streamed response.

        Raises:
            ProviderError: If the API call fails or the client is not initialized.
            ConfigError: If the provider is not properly configured.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")

        model_name = model or self.default_model

        # Context is expected to be List[Message]
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"OllamaProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        # Convert LLMCore Messages to Ollama's expected format
        messages_payload: List[Dict[str, str]] = []
        for msg in context:
            # Ollama uses 'system', 'user', 'assistant' roles
            # Ensure msg.role.value is used if msg.role is an Enum, otherwise convert to string
            role_value = msg.role.value if isinstance(msg.role, Enum) else str(msg.role)
            messages_payload.append({"role": role_value, "content": msg.content})


        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")

        logger.debug(f"Sending request to Ollama via client: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")

        try:
            # Pass kwargs directly as 'options' if the ollama library supports it this way,
            # or structure them into an 'options' dictionary if needed.
            # The `ollama.AsyncClient.chat` method takes `options` as a dict.
            ollama_options = kwargs if kwargs else None

            response_or_stream_obj: Union[ChatResponse, AsyncGenerator[Dict[str, Any], None]] = await self._client.chat(
                model=model_name,
                messages=messages_payload, # type: ignore [arg-type] # SDK expects List[MessageRequest]
                stream=stream,
                options=ollama_options,
                # format=kwargs.get("format") # if 'json' format is needed, pass explicitly
            )

            if stream:
                logger.debug(f"Processing stream response from Ollama model '{model_name}'")
                # The ollama library's stream already yields dicts per chunk.
                return response_or_stream_obj # type: ignore [return-value] # It's an AsyncGenerator[MessageResponse, None]
            else:
                # For non-streaming, response_or_stream_obj is a ChatResponse (dict-like)
                logger.debug(f"Processing non-stream response from Ollama model '{model_name}'")
                if isinstance(response_or_stream_obj, dict): # Should be ChatResponse, which is a TypedDict
                    return response_or_stream_obj
                elif ChatResponse and isinstance(response_or_stream_obj, ChatResponse):
                    # If it's the Pydantic model (if library changes), dump it
                    return response_or_stream_obj # It's already a dict
                else:
                    logger.error(f"Unexpected response type for non-streaming Ollama chat: {type(response_or_stream_obj)}")
                    raise ProviderError(self.get_name(), "Invalid response format from Ollama (non-streaming).")

        except ResponseError as e: # Specific error from ollama library
            error_detail = e.error if hasattr(e, 'error') and e.error else str(e)
            logger.error(f"Ollama API error: HTTP {e.status_code} - {error_detail}", exc_info=True)
            if e.status_code == 404 and "model not found" in error_detail.lower():
                 raise ProviderError(self.get_name(), f"Model '{model_name}' not found by Ollama. "
                                     f"Ensure it is pulled: `ollama pull {model_name}`. Details: {error_detail}")
            raise ProviderError(self.get_name(), f"Ollama API Error (HTTP {e.status_code}): {error_detail}")
        except asyncio.TimeoutError: # Catch generic timeout
            logger.error(f"Request to Ollama API timed out (configured timeout: {self.timeout or 'ollama library default'}).")
            raise ProviderError(self.get_name(), f"Request to Ollama API timed out.")
        except Exception as e: # Catch-all for other errors like connection errors
            logger.error(f"Unexpected error during Ollama chat completion: {e}", exc_info=True)
            # Check for common connection error messages
            if "connect" in str(e).lower() or "Connection refused" in str(e):
                raise ProviderError(self.get_name(), f"Could not connect to Ollama server at {self.host or 'default address'}. "
                                    "Is Ollama running? Details: {e}")
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Ollama: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Counts tokens using the configured tokenizer (tiktoken) or character approximation.
        This method is asynchronous to maintain consistency with the base class.
        """
        if not self._encoding: # Using character approximation
            return (len(text) + 3) // 4 if text else 0 # Integer division

        if not text:
            return 0

        try:
            # tiktoken.encode() is synchronous.
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.error(f"Tiktoken encoding failed with '{self.tokenizer_name}': {e}. "
                         "Falling back to character approximation.", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Counts tokens for a list of LLMCore Messages using the configured method (tiktoken or approximation).
        This method is asynchronous for consistency.
        """
        if not self._encoding: # Using character approximation
            logger.debug(f"Using character approximation for message token counting with Ollama (tokenizer: {self.tokenizer_name}).")
            total_chars = sum(len(msg.content) + len(str(msg.role.value if isinstance(msg.role, Enum) else msg.role)) for msg in messages)
            # Add a small overhead per message for role/separators
            return (total_chars + (len(messages) * 5)) // 4

        # Using tiktoken (OpenAI's heuristic is often a good starting point for many models)
        # This heuristic might not be perfectly accurate for all Ollama models.
        tokens_per_message = 3 # Based on OpenAI's typical overhead
        # tokens_per_name_role = 1 # For role token (OpenAI specific, might not apply universally)
        num_tokens = 0

        for message in messages:
            try:
                num_tokens += tokens_per_message
                # Correctly handle message.role whether it's an Enum or a string
                role_str = message.role.value if isinstance(message.role, Enum) else str(message.role)
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                 logger.error(f"Tiktoken encoding failed for message content/role with '{self.tokenizer_name}': {e}. "
                              "Using character approximation for this message.", exc_info=True)
                 # Correctly derive role_str for approximation in the except block
                 role_str_for_approx = message.role.value if isinstance(message.role, Enum) else str(message.role)
                 num_tokens += (len(message.content) + len(role_str_for_approx) + 15) // 4 # Approximation

        num_tokens += 3  # Simulating a prime for the assistant's reply, as in OpenAI's guide
        return num_tokens

    async def close(self) -> None:
        """Closes the underlying Ollama client session if applicable."""
        if self._client:
            logger.debug("Closing OllamaProvider client (AsyncClient)...")
            # The ollama library's AsyncClient uses an httpx.AsyncClient internally.
            # It should have an `aclose` method.
            if hasattr(self._client, 'aclose') and asyncio.iscoroutinefunction(self._client.aclose):
                try:
                    await self._client.aclose()
                    logger.info("OllamaProvider client (AsyncClient) closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing OllamaProvider client (AsyncClient): {e}", exc_info=True)
            else:
                 logger.debug("Ollama AsyncClient does not have an explicit 'aclose' method, or it's not async. "
                              "Closure might be handled by the library's internal httpx client or garbage collection.")
            self._client = None # Dereference
