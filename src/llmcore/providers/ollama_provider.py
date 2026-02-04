# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
Accepts context as List[Message].
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from collections.abc import AsyncGenerator

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ChatResponse, ResponseError

    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None  # type: ignore
    ResponseError = Exception  # type: ignore
    ChatResponse = None  # type: ignore

# Keep tiktoken for token counting
try:
    import tiktoken

    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Ollama models
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8000,
    "llama3:8b": 8000,
    "llama3:70b": 8000,
    "gemma3:4b": 128000,
    "falcon3:3b": 8000,
    "gemma:latest": 8000,
    "gemma:7b": 8000,
    "gemma:2b": 8000,
    "mistral": 8000,
    "mistral:7b": 8000,
    "mixtral": 32000,
    "mixtral:8x7b": 32000,
    "phi3": 4000,
    "phi3:mini": 4000,
    "codellama": 16000,
    "codellama:7b": 16000,
    "codellama:13b": 16000,
    "codellama:34b": 16000,
    "llama2": 4000,
    "llama2:7b": 4000,
    "llama2:13b": 4000,
    "llama2:70b": 4000,
}
DEFAULT_MODEL = "gemma3:4b"


class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama using the official ollama library.
    Handles List[Message] context type and standardized tool-calling.
    """

    _client: AsyncClient | None = None
    _encoding: Any | None = None
    tokenizer_name: str

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the OllamaProvider using the official ollama library.

        Args:
            config: Configuration dictionary from `[providers.ollama]` containing:
                    'host' (optional): Host URL for the Ollama server.
                    'default_model' (optional): Default Ollama model to use.
                    'timeout' (optional): Request timeout in seconds.
                    'tokenizer' (optional): Tokenizer to use for estimations.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not ollama_available:
            raise ImportError(
                "Ollama library not installed. Please install `ollama` or `llmcore[ollama]`."
            )
        if not tiktoken_available:
            logger.warning("tiktoken library not available. Token counting will use approximation.")

        self.host = config.get("host")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        timeout_val = config.get("timeout")
        self.timeout = float(timeout_val) if timeout_val is not None else None

        try:
            client_args = {}
            if self.host:
                client_args["host"] = self.host
            if self.timeout is not None:
                client_args["timeout"] = self.timeout
            self._client = AsyncClient(**client_args)
            logger.debug("Ollama AsyncClient initialized.")
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
                    logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name}.")
                except Exception as e:
                    logger.warning(
                        f"Failed to load tiktoken encoding '{self.tokenizer_name}'. Falling back to approximation. Error: {e}"
                    )
                    self.tokenizer_name = "char_div_4"
            else:
                logger.warning("tiktoken not available. Falling back to character approximation.")
                self.tokenizer_name = "char_div_4"
        elif self.tokenizer_name != "char_div_4":
            logger.warning(
                f"Unsupported tokenizer '{self.tokenizer_name}'. Falling back to approximation."
            )
            self.tokenizer_name = "char_div_4"

        if self.tokenizer_name == "char_div_4":
            logger.info("OllamaProvider using character division for token counting.")

    def get_name(self) -> str:
        """Returns the provider name: 'ollama'."""
        return "ollama"

    async def get_models_details(self) -> list[ModelDetails]:
        """
        Asynchronously discovers and returns detailed information about available local models
        by querying the Ollama API.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")
        try:
            models_info = await self._client.list()
            model_list = models_info.get("models", []) if models_info else []

            details_list = []
            for model_data in model_list:
                model_name = model_data.get("name")
                if not model_name:
                    continue

                # Assume tool support is experimental for all Ollama models for now
                details = ModelDetails(
                    id=model_name,
                    context_length=self.get_max_context_length(model_name),
                    supports_streaming=True,
                    supports_tools=True,  # Assuming experimental support
                    provider_name=self.get_name(),
                    metadata={"details": model_data.get("details", {})},
                )
                details_list.append(details)

            logger.info(f"Discovered {len(details_list)} local Ollama models.")
            return details_list
        except ResponseError as e:
            logger.error(
                f"Ollama API error fetching models: HTTP {e.status_code} - {e.error}", exc_info=True
            )
            raise ProviderError(
                self.get_name(), f"Failed to fetch models from Ollama API: {e.error}"
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error fetching models: {e}")

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """
        Returns a schema of supported inference parameters for Ollama models.
        These correspond to the 'options' in the Ollama API.
        """
        return {
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer"},
            "num_ctx": {"type": "integer"},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "mirostat": {"type": "integer"},
            "mirostat_tau": {"type": "number"},
            "mirostat_eta": {"type": "number"},
            "num_predict": {"type": "integer"},
            "max_tokens": {"type": "integer"},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Returns the maximum context length (tokens) for the given Ollama model."""
        model_name = model or self.default_model
        base_model_name = model_name.split(":")[0]
        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(
            model_name, DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)
        )
        if limit is None:
            limit = 4096
            logger.warning(
                f"Unknown context length for Ollama model '{model_name}'. Using fallback: {limit}."
            )
        return limit

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Sends a chat completion request to the Ollama API, with support for tools.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")

        # Pre-flight validation of kwargs
        supported_params = self.get_supported_parameters()
        for key in kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for Ollama provider.")

        model_name = model or self.default_model
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        messages_payload: list[dict[str, str]] = [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content,
            }
            for msg in context
        ]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages to send.")

        if "max_tokens" in kwargs:
            kwargs["num_predict"] = kwargs.pop("max_tokens")

        # Prepare tools for the ollama library
        tools_payload = None
        if tools:
            logger.info(f"Preparing {len(tools)} tools for Ollama request (experimental).")
            tools_payload = [{"type": "function", "function": tool.model_dump()} for tool in tools]

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
                "options": kwargs,
                "tools": tools_payload,
                "tool_choice": tool_choice,
            }
            logger.debug(
                f"RAW LLM REQUEST ({self.get_name()} @ {model_name}): {json.dumps(log_data, indent=2)}"
            )

        try:
            response_or_stream = await self._client.chat(
                model=model_name,
                messages=messages_payload,  # type: ignore
                stream=stream,
                tools=tools_payload,  # type: ignore
                options=kwargs if kwargs else None,
            )
            if stream:

                async def stream_wrapper():
                    async for chunk in response_or_stream:  # type: ignore
                        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"RAW LLM STREAM CHUNK ({self.get_name()}): {json.dumps(chunk)}"
                            )
                        yield chunk

                return stream_wrapper()
            else:
                # Handle both dict and ChatResponse object (Pydantic model)
                if isinstance(response_or_stream, dict):
                    response_dict = response_or_stream
                elif hasattr(response_or_stream, "model_dump"):
                    # ChatResponse is a Pydantic model - convert to dict
                    response_dict = response_or_stream.model_dump()
                elif hasattr(response_or_stream, "__dict__"):
                    # Fallback for other object types
                    response_dict = dict(response_or_stream.__dict__)
                else:
                    raise ProviderError(self.get_name(), "Invalid non-streaming response format.")

                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RAW LLM RESPONSE ({self.get_name()}): {json.dumps(response_dict, indent=2)}"
                    )
                return response_dict

        except ResponseError as e:
            error_detail = e.error if hasattr(e, "error") and e.error else str(e)
            logger.error(f"Ollama API error: HTTP {e.status_code} - {error_detail}", exc_info=True)
            if e.status_code == 404:
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found. Pull it with `ollama pull {model_name}`.",
                )
            raise ProviderError(
                self.get_name(), f"Ollama API Error (HTTP {e.status_code}): {error_detail}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during Ollama chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Counts tokens using the configured tokenizer or character approximation."""
        if not self._encoding:
            return (len(text) + 3) // 4
        if not text:
            return 0
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))  # type: ignore

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Approximates token count for a list of messages using tiktoken."""
        if not self._encoding:
            total_chars = sum(
                len(msg.content)
                + len(
                    msg.role.value
                    if hasattr(msg.role, "value")
                    else str(msg.role)
                    if hasattr(msg.role, "value")
                    else str(msg.role)
                )
                for msg in messages
            )
            return (total_chars + (len(messages) * 5) + 3) // 4

        num_tokens = 0
        for message in messages:
            num_tokens += 3  # tokens_per_message
            try:
                role_str = (
                    message.role.value if hasattr(message.role, "value") else str(message.role)
                )

                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed for message part: {e}. Approximating.")
                role_str_fallback = (
                    message.role.value if hasattr(message.role, "value") else str(message.role)
                )
                num_tokens += (len(role_str_fallback) + len(message.content)) // 4
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """
        Extract text content from Ollama non-streaming response.

        Ollama /api/chat response format (verified from official REST API docs):
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello! How are you?"},
            "done": true,
            "total_duration": 5191566416,
            ...
        }

        The Ollama Python SDK's ChatResponse.model_dump() produces this structure.
        This differs from OpenAI's {"choices": [{"message": {...}}]} format.

        Args:
            response: The raw response dictionary from chat_completion().

        Returns:
            The extracted text content.
        """
        try:
            # Ollama format: response["message"]["content"]
            message = response.get("message", {})
            if isinstance(message, dict):
                return message.get("content") or ""
            return ""
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to extract content from Ollama response: {e}")
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """
        Extract text delta from Ollama streaming chunk.

        Ollama streaming format (verified from official REST API docs):
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "The"},
            "done": false
        }

        Each chunk contains incremental content. The final chunk has "done": true
        and may have empty content.

        Args:
            chunk: A single streaming chunk dictionary (or Pydantic model).

        Returns:
            The extracted text delta.
        """
        try:
            # Handle both dict and Pydantic model formats
            # OllamaProvider.chat_completion() may return ChatResponse Pydantic model
            # which gets converted to dict, but streaming yields raw chunks
            if isinstance(chunk, dict):
                message = chunk.get("message", {})
            elif hasattr(chunk, "message"):
                # Handle Pydantic ChatResponse model case (ollama._types.ChatResponse)
                msg_attr = chunk.message
                if isinstance(msg_attr, dict):
                    message = msg_attr
                elif hasattr(msg_attr, "content"):
                    # Message is a Pydantic model with .content attribute
                    return msg_attr.content or ""
                else:
                    return ""
            else:
                return ""

            if isinstance(message, dict):
                return message.get("content") or ""
            return ""
        except (KeyError, TypeError, AttributeError):
            return ""

    async def close(self) -> None:
        """Closes the underlying Ollama client session."""
        if self._client:
            if hasattr(self._client, "aclose") and asyncio.iscoroutinefunction(self._client.aclose):
                try:
                    await self._client.aclose()
                    logger.info("OllamaProvider client closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing OllamaProvider client: {e}", exc_info=True)
            self._client = None
