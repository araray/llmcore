# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
Accepts context as List[Message].
"""

import asyncio
import logging
from enum import Enum  # Import Enum for isinstance check
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Use the official ollama library
try:
    import ollama
    from ollama import (AsyncClient,  # ChatResponse for type checking
                        ChatResponse, ResponseError)
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


from ..exceptions import ConfigError, ProviderError
from ..models import Message
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Ollama models
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8000, "llama3:8b": 8000, "llama3:70b": 8000,
    "gemma3:4b": 128000,
    "falcon3:3b": 8000,
    "gemma:latest": 8000, "gemma:7b": 8000, "gemma:2b": 8000,
    "mistral": 8000, "mistral:7b": 8000,
    "mixtral": 32000, "mixtral:8x7b": 32000,
    "phi3": 4000, "phi3:mini": 4000,
    "codellama": 16000,
    "codellama:7b": 16000, "codellama:13b": 16000, "codellama:34b": 16000,
    "llama2": 4000,
    "llama2:7b": 4000, "llama2:13b": 4000, "llama2:70b": 4000,
}
DEFAULT_MODEL = "gemma3:4b" # A common and capable default for Ollama


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

        if not tiktoken_available:
            logger.warning("tiktoken library not available. Ollama token counting will use character approximation.")

        self.host = config.get("host")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        timeout_val = config.get("timeout")
        self.timeout = float(timeout_val) if timeout_val is not None else None

        try:
            client_args = {}
            if self.host:
                client_args['host'] = self.host
            if self.timeout is not None:
                client_args['timeout'] = self.timeout

            self._client = AsyncClient(**client_args)
            logger.debug(f"Ollama AsyncClient initialized (Host: {self.host or 'default library host'}, Timeout: {self.timeout or 'default library timeout'})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama AsyncClient: {e}", exc_info=True)
            raise ConfigError(f"Ollama client initialization failed: {e}")

        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base")
        self._encoding = None
        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken_available and tiktoken:
                try:
                    encoding_name = self.tokenizer_name.split("tiktoken_")[1]
                    self._encoding = tiktoken.get_encoding(encoding_name)
                    logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name} for token counting.")
                except Exception as e:
                    logger.warning(f"Failed to load tiktoken encoding '{self.tokenizer_name}'. "
                                   f"Falling back to character approximation for token counting. Error: {e}")
                    self.tokenizer_name = "char_div_4"
            else:
                logger.warning(f"tiktoken library not available, but '{self.tokenizer_name}' was configured. "
                               "Falling back to character approximation for token counting.")
                self.tokenizer_name = "char_div_4"
        elif self.tokenizer_name != "char_div_4":
            logger.warning(f"Unsupported Ollama tokenizer '{self.tokenizer_name}' configured. "
                           "Falling back to character approximation ('char_div_4') for token counting.")
            self.tokenizer_name = "char_div_4"

        if self.tokenizer_name == "char_div_4" and not self._encoding:
            logger.info("OllamaProvider using character division approximation for token counting.")


    def get_name(self) -> str:
        """Returns the provider name: 'ollama'."""
        return "ollama"

    async def _fetch_ollama_models_from_api(self) -> List[str]:
        """Helper to fetch model list from Ollama API."""
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized for fetching models.")
        try:
            models_info = await self._client.list()
            models = [m.get("name") for m in models_info.get("models", []) if m.get("name")] # type: ignore[union-attr]
            logger.debug(f"Fetched {len(models)} models from Ollama API via client: {models}")
            return models
        except ResponseError as e:
            logger.warning(f"Failed to fetch models from Ollama API: HTTP {e.status_code} - {e.error}. "
                           "get_available_models() will return a static list.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True)
            return []


    def get_available_models(self) -> List[str]:
        """
        Returns a list of available models.
        Tries to fetch from Ollama API, falls back to a static list if API call fails.
        Note: This base class method is synchronous. The async fetch is for internal use or future refactor.
        For now, consistent with other providers, it returns a static list.
        """
        logger.warning("OllamaProvider.get_available_models() currently returning static list. "
                       "For dynamic list, an API call would be needed (see _fetch_ollama_models_from_api).")
        return list(DEFAULT_OLLAMA_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given Ollama model."""
        model_name = model or self.default_model
        base_model_name = model_name.split(':')[0]

        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(model_name)
        if limit is None:
            limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)

        if limit is None:
            limit = 4096
            logger.warning(f"Unknown context length for Ollama model '{model_name}'. "
                           f"Using fallback limit: {limit}. Verify with 'ollama show {model_name}'.")
        return limit


    async def chat_completion(
        self,
        context: ContextPayload,
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

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"OllamaProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        messages_payload: List[Dict[str, str]] = []
        for msg in context:
            role_value = msg.role.value if isinstance(msg.role, Enum) else str(msg.role)
            messages_payload.append({"role": role_value, "content": msg.content})


        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")

        logger.debug(f"Sending request to Ollama via client: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")

        try:
            ollama_options = kwargs if kwargs else None

            response_or_stream_obj: Union[ChatResponse, AsyncGenerator[Dict[str, Any], None]] = await self._client.chat(
                model=model_name,
                messages=messages_payload, # type: ignore [arg-type]
                stream=stream,
                options=ollama_options,
            )

            if stream:
                logger.debug(f"Processing stream response from Ollama model '{model_name}'")
                return response_or_stream_obj # type: ignore [return-value]
            else:
                logger.debug(f"Processing non-stream response from Ollama model '{model_name}'")
                if ChatResponse and isinstance(response_or_stream_obj, ChatResponse):
                    # Convert Pydantic model (ChatResponse) to dict
                    return response_or_stream_obj.model_dump(exclude_none=True)
                elif isinstance(response_or_stream_obj, dict):
                    # Should ideally be ChatResponse, but handle if it's already a dict
                    return response_or_stream_obj
                else:
                    logger.error(f"Unexpected response type for non-streaming Ollama chat: {type(response_or_stream_obj)}")
                    raise ProviderError(self.get_name(), "Invalid response format from Ollama (non-streaming).")

        except ResponseError as e:
            error_detail = e.error if hasattr(e, 'error') and e.error else str(e)
            logger.error(f"Ollama API error: HTTP {e.status_code} - {error_detail}", exc_info=True)
            if e.status_code == 404 and "model not found" in error_detail.lower():
                 raise ProviderError(self.get_name(), f"Model '{model_name}' not found by Ollama. "
                                     f"Ensure it is pulled: `ollama pull {model_name}`. Details: {error_detail}")
            raise ProviderError(self.get_name(), f"Ollama API Error (HTTP {e.status_code}): {error_detail}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Ollama API timed out (configured timeout: {self.timeout or 'ollama library default'}).")
            raise ProviderError(self.get_name(), f"Request to Ollama API timed out.")
        except Exception as e:
            logger.error(f"Unexpected error during Ollama chat completion: {e}", exc_info=True)
            if "connect" in str(e).lower() or "Connection refused" in str(e):
                raise ProviderError(self.get_name(), f"Could not connect to Ollama server at {self.host or 'default address'}. "
                                    "Is Ollama running? Details: {e}")
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Ollama: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Counts tokens using the configured tokenizer (tiktoken) or character approximation.
        This method is asynchronous to maintain consistency with the base class.
        """
        if not self._encoding:
            return (len(text) + 3) // 4 if text else 0

        if not text:
            return 0

        try:
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
        if not self._encoding:
            logger.debug(f"Using character approximation for message token counting with Ollama (tokenizer: {self.tokenizer_name}).")
            total_chars = sum(len(msg.content) + len(str(msg.role.value if isinstance(msg.role, Enum) else msg.role)) for msg in messages)
            return (total_chars + (len(messages) * 5)) // 4

        tokens_per_message = 3
        num_tokens = 0

        for message in messages:
            try:
                num_tokens += tokens_per_message
                role_str = message.role.value if isinstance(message.role, Enum) else str(message.role)
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                 logger.error(f"Tiktoken encoding failed for message content/role with '{self.tokenizer_name}': {e}. "
                              "Using character approximation for this message.", exc_info=True)
                 role_str_for_approx = message.role.value if isinstance(message.role, Enum) else str(message.role)
                 num_tokens += (len(message.content) + len(role_str_for_approx) + 15) // 4

        num_tokens += 3
        return num_tokens

    async def close(self) -> None:
        """Closes the underlying Ollama client session if applicable."""
        if self._client:
            logger.debug("Closing OllamaProvider client (AsyncClient)...")
            if hasattr(self._client, 'aclose') and asyncio.iscoroutinefunction(self._client.aclose):
                try:
                    await self._client.aclose()
                    logger.info("OllamaProvider client (AsyncClient) closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing OllamaProvider client (AsyncClient): {e}", exc_info=True)
            else:
                 logger.debug("Ollama AsyncClient does not have an explicit 'aclose' method, or it's not async. "
                              "Closure might be handled by the library's internal httpx client or garbage collection.")
            self._client = None
