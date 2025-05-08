# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library.

Interacts with a local Ollama instance via its REST API.
Supports streaming and different API endpoints (/api/chat, /api/generate).
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import aiohttp
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None # Placeholder

from ..models import Message, Role
from ..exceptions import ProviderError, ConfigError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Ollama models (can be overridden by API in future)
# Source: Often based on the underlying model's known limits (e.g., Llama 3, Mistral)
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8192,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
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

class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama.

    Connects to a running Ollama instance via HTTP requests.
    """
    _client_session: Optional[aiohttp.ClientSession] = None

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OllamaProvider.

        Args:
            config: Configuration dictionary containing:
                    'base_url': The base URL of the Ollama API (e.g., "http://localhost:11434").
                    'default_model': The default Ollama model to use (e.g., "llama3").
                    'timeout' (optional): Request timeout in seconds (default: 120).
                    'tokenizer' (optional): Tokenizer to use for counting ('tiktoken_cl100k_base',
                                            'tiktoken_p50k_base', 'char_div_4'). Default: 'tiktoken_cl100k_base'.
        """
        self.base_url = config.get("base_url")
        if not self.base_url:
            raise ConfigError("Ollama provider 'base_url' is required in configuration.")
        # Ensure base_url doesn't end with /api or /
        self.base_url = self.base_url.rstrip('/').replace('/api', '')

        self.default_model = config.get("default_model", "llama3")
        self.timeout = int(config.get("timeout", 120))
        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base")
        self.encoding = None

        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken_available and tiktoken:
                try:
                    encoding_name = self.tokenizer_name.split("tiktoken_")[1]
                    self.encoding = tiktoken.get_encoding(encoding_name)
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


    def get_name(self) -> str:
        """Returns the provider name."""
        return "ollama"

    async def _get_client_session(self) -> aiohttp.ClientSession:
        """Gets or creates an aiohttp ClientSession."""
        if self._client_session is None or self._client_session.closed:
            # You might want to configure the connector with limits, SSL context, etc.
            connector = aiohttp.TCPConnector(limit_per_host=20) # Example limit
            self._client_session = aiohttp.ClientSession(connector=connector)
            logger.debug("Created new aiohttp.ClientSession for OllamaProvider.")
        return self._client_session

    async def close(self) -> None:
        """Closes the underlying aiohttp client session."""
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()
            self._client_session = None
            logger.debug("Closed aiohttp.ClientSession for OllamaProvider.")

    async def _fetch_ollama_models(self) -> List[str]:
        """Fetches the list of models available from the Ollama /api/tags endpoint."""
        session = await self._get_client_session()
        api_url = f"{self.base_url}/api/tags"
        try:
            async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=10)) as response: # Shorter timeout for listing
                response.raise_for_status() # Raise exception for non-2xx status codes
                data = await response.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                logger.debug(f"Fetched {len(models)} models from Ollama API: {models}")
                return models
        except aiohttp.ClientResponseError as e:
            logger.warning(f"Failed to fetch models from Ollama API ({api_url}): HTTP {e.status} - {e.message}")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"Connection error fetching models from Ollama API ({api_url}): {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API ({api_url}): {e}", exc_info=True)
            return []

    def get_available_models(self) -> List[str]:
        """
        Returns a list of known default models.
        Note: Does not dynamically fetch from the API in this synchronous version.
              Use an async method if dynamic fetching is required at runtime.
        """
        # Consider adding dynamic fetching here if needed, but it makes the sync method complex.
        # For now, return the static list.
        logger.warning("OllamaProvider.get_available_models() returning static list. "
                       "Use an async method for dynamic fetching if needed.")
        return list(DEFAULT_OLLAMA_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the estimated maximum context length for the given Ollama model."""
        model_name = model or self.default_model
        # Base model name (e.g., "llama3" from "llama3:8b")
        base_model_name = model_name.split(':')[0]

        # Check specific model, then base model, then default
        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(model_name)
        if limit is None:
            limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)
        if limit is None:
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
        """Sends a chat completion request to the Ollama API."""
        if not isinstance(context, list):
            # Ollama's /api/chat expects a list of messages.
            # If we receive something else (like an MCP object), it needs conversion.
            # For now, raise an error if the context is not the expected list.
            raise ProviderError(self.get_name(), f"OllamaProvider received unsupported context type: {type(context).__name__}")

        model_name = model or self.default_model
        session = await self._get_client_session()
        api_url = f"{self.base_url}/api/chat"

        # Prepare payload for /api/chat
        messages_payload = [{"role": msg.role.value, "content": msg.content} for msg in context]
        payload = {
            "model": model_name,
            "messages": messages_payload,
            "stream": stream,
            "options": kwargs, # Pass kwargs like temperature, top_p under 'options'
        }
        logger.debug(f"Sending request to Ollama ({api_url}) with model '{model_name}', stream={stream}")
        # logger.debug(f"Payload (excluding messages): { {k:v for k,v in payload.items() if k != 'messages'} }")

        try:
            async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                # Check for common errors first
                if response.status == 404:
                    logger.warning(f"Ollama API endpoint not found ({api_url}). Is Ollama running and accessible?")
                    raise ProviderError(self.get_name(), f"API endpoint not found: {api_url}")
                if response.status == 400:
                     error_text = await response.text()
                     logger.error(f"Ollama API Bad Request ({response.status}) for model '{model_name}': {error_text}")
                     # Check if the model might just not be pulled
                     if "model not found" in error_text.lower():
                          raise ProviderError(self.get_name(), f"Model '{model_name}' not found. Pull it using 'ollama pull {model_name}'.")
                     raise ProviderError(self.get_name(), f"API Bad Request: {error_text}")

                # Raise other non-200 errors
                response.raise_for_status()

                if stream:
                    logger.debug(f"Processing stream response from Ollama model '{model_name}'")
                    return self._process_stream(response, model_name)
                else:
                    logger.debug(f"Processing non-stream response from Ollama model '{model_name}'")
                    result_data = await response.json()
                    # Ensure the non-streamed response format is consistent
                    # The actual response might look like:
                    # {'model': 'llama3', 'created_at': '...', 'message': {'role': 'assistant', 'content': '...'}, 'done': True, ...}
                    return result_data

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error accessing Ollama at {self.base_url}: {e}")
            raise ProviderError(self.get_name(), f"Could not connect to Ollama at {self.base_url}. Is it running?")
        except aiohttp.ClientResponseError as e: # Catch non-2xx status codes not handled above
             logger.error(f"Ollama API request failed ({api_url}): HTTP {e.status} - {e.message}")
             raise ProviderError(self.get_name(), f"API request failed: {e.status} - {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Ollama timed out after {self.timeout} seconds ({api_url}).")
            raise ProviderError(self.get_name(), f"Request timed out after {self.timeout}s.")
        except Exception as e:
            logger.error(f"Unexpected error during Ollama chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def _process_stream(self, response: aiohttp.ClientResponse, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Processes the streaming response from Ollama API."""
        buffer = ""
        async for line_bytes in response.content:
            buffer += line_bytes.decode('utf-8', errors='replace')
            # Ollama streams JSON objects separated by newlines
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    try:
                        chunk_data = json.loads(line)
                        # Example chunk: {'model': 'llama3', 'created_at': '...', 'message': {'role': 'assistant', 'content': ' response'}, 'done': False}
                        # Or final chunk: {'model': 'llama3', ..., 'done': True, 'total_duration': ..., 'prompt_eval_count': ..., ...}
                        yield chunk_data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON stream line from Ollama: {line[:100]}...")
                    except Exception as e:
                         logger.error(f"Error processing Ollama stream line: {e} - Line: {line[:100]}...")
        # Process any remaining data in the buffer
        if buffer.strip():
            try:
                chunk_data = json.loads(buffer)
                yield chunk_data
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode final JSON stream buffer from Ollama: {buffer[:100]}...")
            except Exception as e:
                logger.error(f"Error processing final Ollama stream buffer: {e} - Buffer: {buffer[:100]}...")


    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the configured tokenizer or approximation."""
        if not text:
            return 0

        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed for model '{model or self.default_model}': {e}. Falling back.")
                # Fall through to approximation if tiktoken fails unexpectedly

        # Fallback to character approximation
        # Average ~4 chars/token is a rough estimate
        return (len(text) + 3) // 4

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of messages using the configured method."""
        # Ollama token counting is complex and model-dependent.
        # We apply a small overhead per message plus the content tokens.
        # This is a heuristic and may not be perfectly accurate.
        overhead_per_message = 5 # Rough estimate for role/formatting markers
        total_tokens = 0

        for msg in messages:
            content_tokens = self.count_tokens(msg.content, model)
            total_tokens += content_tokens + overhead_per_message

        # Add a small buffer for the overall structure/prompting if needed
        total_tokens += 3 # Similar to OpenAI's final assistant prompt marker

        return total_tokens
