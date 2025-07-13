# src/llmcore/providers/base.py
"""
Abstract Base Class for Large Language Model (LLM) Providers.

This module defines the common interface that all specific LLM provider
implementations (e.g., OpenAI, Anthropic, Ollama) must adhere to within
the LLMCore library.
"""

import abc
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Import models for type hinting
from ..models import Message, ModelDetails, Tool

# Define a type alias for the context payload that can be passed to providers.
ContextPayload = List[Message]


class BaseProvider(abc.ABC):
    """
    Abstract Base Class for LLM provider integrations.

    Ensures all providers offer a consistent set of core functionalities:
    - Initialization with configuration.
    - Dynamic discovery of model details and supported parameters.
    - Performing chat completions, with standardized support for streaming and tool calling.
    - Counting tokens accurately according to the provider's model.
    """
    log_raw_payloads_enabled: bool

    @abc.abstractmethod
    def __init__(self, config: Dict[str, Any], log_raw_payloads: bool = False):
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
    async def get_models_details(self) -> List[ModelDetails]:
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
    def get_supported_parameters(self, model: Optional[str] = None) -> Dict[str, Any]:
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
    def get_max_context_length(self, model: Optional[str] = None) -> int:
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
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
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
    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
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
    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
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

    async def close(self) -> None:
         """
         Clean up any resources used by the provider, such as network sessions.
         This is an optional method; providers that do not need explicit cleanup
         can use this default pass-through implementation.
         """
         pass
