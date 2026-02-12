# src/llmcore/providers/base.py
"""
Abstract Base Class for Large Language Model (LLM) Providers.

This module defines the common interface that all specific LLM provider
implementations (e.g., OpenAI, Anthropic, Ollama) must adhere to within
the LLMCore library.

UPDATED: Added instrumentation points for observability metrics and tracing.
UPDATED: Added get_models_details() abstract method for dynamic model discovery.
UPDATED: Added tools and tool_choice parameters to chat_completion() for unified tool-calling.
"""

import abc
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Union

# Import models for type hinting
from ..models import Message, ModelDetails, Tool

# Define a type alias for the context payload that can be passed to providers.
ContextPayload = list[Message]


class BaseProvider(abc.ABC):
    """
    Abstract Base Class for LLM provider integrations.

    Ensures all providers offer a consistent set of core functionalities:
    - Initialization with configuration.
    - Dynamic discovery of model details and supported parameters.
    - Performing chat completions, with standardized support for streaming and tool calling.
    - Counting tokens accurately according to the provider's model.

    UPDATED: Added instrumentation methods for observability integration.
    UPDATED: Added get_models_details() abstract method for dynamic model capability discovery.
    UPDATED: Enhanced chat_completion() method signature with unified tool-calling support.
    """

    log_raw_payloads_enabled: bool

    @abc.abstractmethod
    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initialize the provider with its specific configuration.

        Args:
            config: A dictionary containing provider-specific settings loaded
                    from the main LLMCore configuration (e.g., api_key,
                    base_url, default_model, timeout).
            log_raw_payloads: A boolean flag indicating whether raw request/response
                              payloads should be logged by this provider instance.
        """
        self.log_raw_payloads_enabled = log_raw_payloads
        # Instance name injected by ProviderManager to distinguish
        # OpenAI-compatible providers (deepseek, xai, mistral, etc.)
        self._provider_instance_name: str | None = config.get("_instance_name")

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Return the unique identifier name for this provider.

        Examples: "openai", "anthropic", "ollama", "gemini".

        Returns:
            The provider's name as a string.
        """
        pass

    @abc.abstractmethod
    async def get_models_details(self) -> list[ModelDetails]:
        """
        Asynchronously discover and return detailed information about available models.

        This method should query the provider's API to get a list of models
        and their capabilities, such as context length and feature support.
        Implementations should consider caching the results to avoid excessive API calls.

        Returns:
            A list of `ModelDetails` objects, each describing an available model.
        """
        pass

    @abc.abstractmethod
    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """
        Return a schema describing the supported inference parameters for a model.

        This allows for pre-flight validation of parameters passed to chat_completion.
        The schema can be a simplified JSON Schema-like dictionary.

        Args:
            model: The specific model name. If None, returns parameters for the
                   provider's default model.

        Returns:
            A dictionary describing the supported parameters and their types/constraints.
            Example: {"temperature": {"type": "number"}, "top_p": {"type": "number"}}
        """
        pass

    @abc.abstractmethod
    def get_max_context_length(self, model: str | None = None) -> int:
        """
        Return the maximum context length (in tokens) for a specific model.

        This may use a combination of dynamically discovered data from `get_models_details`
        and internal fallback values.

        Args:
            model: The specific model name. If None, should return the limit
                   for the provider's default model.

        Returns:
            The maximum number of tokens allowed in the context window.
        """
        pass

    @abc.abstractmethod
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
        Perform a chat completion request to the provider's API.

        This is the core method for interacting with the LLM. It sends the
        prepared context and receives the model's response. It now includes
        standardized parameters for tool calling.

        Args:
            context: The context payload to send, as a list of `llmcore.models.Message` objects.
            model: The specific model identifier to use for this completion.
                   If None, the provider's configured default model should be used.
            stream: If True, the method should return an asynchronous generator
                    yielding raw response chunks (dictionaries) from the API.
            tools: An optional list of `llmcore.models.Tool` objects available for the LLM to call.
            tool_choice: An optional string to control how the model uses tools (e.g., "auto", "any").
            **kwargs: Additional provider-specific parameters (e.g., temperature, top_p).
                      These should be validated against `get_supported_parameters`.

        Returns:
            - If stream=False: A dictionary representing the complete API response.
            - If stream=True: An asynchronous generator yielding dictionaries of raw stream chunks.

        Raises:
            ProviderError: For any provider-specific errors (API, connection, etc.).
            ValueError: If an unsupported parameter is passed in `kwargs`.
        """
        pass

    @abc.abstractmethod
    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Asynchronously count the number of tokens for a given text string.

        Args:
            text: The text string to count tokens for.
            model: The specific model context for token counting. If None,
                   uses the provider's default model.

        Returns:
            The number of tokens as an integer.
        """
        pass

    @abc.abstractmethod
    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """
        Asynchronously count the total tokens for a list of messages.

        This should account for the provider's specific formatting overhead
        (e.g., roles, special tokens between messages).

        Args:
            messages: A list of `llmcore.models.Message` objects.
            model: The specific model context for token counting. If None,
                   uses the provider's default model.

        Returns:
            The total estimated token count for the messages.
        """
        pass

    @abc.abstractmethod
    def extract_response_content(self, response: dict[str, Any]) -> str:
        """
        Extract the text content from a non-streaming API response.

        Each provider has a different response structure. This method provides
        a unified interface for extracting the generated text content.

        Args:
            response: The raw response dictionary from chat_completion().

        Returns:
            The extracted text content as a string.

        Raises:
            KeyError: If the expected content field is not present.
        """
        pass

    @abc.abstractmethod
    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """
        Extract the text delta from a streaming response chunk.

        Each provider has a different streaming chunk structure. This method
        provides a unified interface for extracting incremental text content.

        Args:
            chunk: A single chunk dictionary from a streaming response.

        Returns:
            The extracted text delta as a string, or empty string if none.
        """
        pass

    async def close(self) -> None:
        """
        Clean up any resources used by the provider, such as network sessions.
        This is an optional method; providers that do not need explicit cleanup
        can use this default pass-through implementation.
        """
        pass

    # ============================================================================
    # NEW: Observability Instrumentation Methods
    # ============================================================================

    def _record_llm_metrics(
        self,
        model: str,
        duration: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """
        Record metrics for an LLM API request.

        This method should be called by concrete provider implementations
        to record observability metrics for each API call.

        Args:
            model: Model name used for the request
            duration: Request duration in seconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            error: Error type if request failed
            tenant_id: Tenant identifier if available
        """
        try:
            # Extract tenant_id from current request context if not provided
            if tenant_id is None:
                record_llm_request(
                    provider=self.get_name(),
                    model=model,
                    tenant_id=tenant_id,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=error,
                )
        except Exception as e:
            # Don't fail the main operation if metrics recording fails
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to record LLM metrics: {e}")

    def _create_llm_span(self, operation: str, model: str, **attributes):
        """
        Create a tracing span for an LLM operation.

        This method creates a distributed tracing span for LLM operations,
        allowing detailed tracing of API calls across the system.

        Args:
            operation: Name of the operation (e.g., "chat_completion", "token_count")
            model: Model name being used
            **attributes: Additional span attributes

        Returns:
            Span context manager or no-op context manager
        """
        try:
            from ..tracing import create_span, get_tracer

            tracer = get_tracer(f"llmcore.providers.{self.get_name()}")

            span_attributes = {
                "llm.provider": self.get_name(),
                "llm.model": model,
                "llm.operation": operation,
                **attributes,
            }

            return create_span(tracer, f"llm.{operation}", **span_attributes)

        except Exception as e:
            # Return no-op context manager if tracing fails
            import logging
            from contextlib import nullcontext

            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to create LLM span: {e}")
            return nullcontext()

    async def _instrumented_chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Instrumented wrapper for chat_completion that adds observability.

        This method provides a template for concrete providers to add
        observability instrumentation around their chat_completion calls.
        Providers should override this method and call their actual implementation
        within the instrumentation wrapper.

        Args:
            context: The context payload to send
            model: The specific model identifier to use
            stream: If True, return an async generator of chunks
            tools: Optional list of tools available for the LLM
            tool_choice: Optional tool choice strategy
            **kwargs: Additional provider-specific parameters

        Returns:
            API response or async generator of chunks
        """
        actual_model = model or self.default_model if hasattr(self, "default_model") else "unknown"
        start_time = time.time()
        error = None
        input_tokens = None
        output_tokens = None

        # Create tracing span
        span_attributes = {
            "llm.stream": stream,
            "llm.tools_available": len(tools) if tools else 0,
            "llm.tool_choice": tool_choice,
            "llm.context_messages": len(context),
        }

        with self._create_llm_span("chat_completion", actual_model, **span_attributes) as span:
            try:
                # Call the actual implementation
                result = await self.chat_completion(
                    context=context,
                    model=model,
                    stream=stream,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )

                # Extract token counts from response if available
                if not stream and isinstance(result, dict):
                    usage = result.get("usage", {})
                    input_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens")

                return result

            except Exception as e:
                error = type(e).__name__
                if span:
                    from ..tracing import record_span_exception

                    record_span_exception(span, e)
                raise
            finally:
                # Record metrics
                duration = time.time() - start_time
                self._record_llm_metrics(
                    model=actual_model,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=error,
                )

                # Add span attributes
                if span:
                    from ..tracing import add_span_attributes

                    add_span_attributes(
                        span,
                        {
                            "llm.duration_seconds": duration,
                            "llm.input_tokens": input_tokens,
                            "llm.output_tokens": output_tokens,
                            "llm.success": error is None,
                        },
                    )
