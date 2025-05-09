# src/llmcore/providers/base.py
"""
Abstract Base Class for Large Language Model (LLM) Providers.

This module defines the common interface that all specific LLM provider
implementations (e.g., OpenAI, Anthropic, Ollama) must adhere to within
the LLMCore library.
"""

import abc
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING

# Import Message for type hinting
from ..models import Message

# Use TYPE_CHECKING for the MCP import to avoid runtime dependency if SDK not installed
if TYPE_CHECKING:
    try:
        # Attempt to import the specific Context object for type hinting
        from modelcontextprotocol import Context as MCPContextObject
    except ImportError:
        # If import fails, use Any as a fallback type hint
        MCPContextObject = Any
else:
    # At runtime, define MCPContextObject as Any if not checking types
    MCPContextObject = Any

# Define a type alias for the context payload that can be passed to providers
ContextPayload = Union[List[Message], MCPContextObject]


class BaseProvider(abc.ABC):
    """
    Abstract Base Class for LLM provider integrations.

    Ensures all providers offer a consistent set of core functionalities:
    - Initialization with configuration.
    - Getting provider metadata (name, available models, context limits).
    - Performing chat completions (supporting streaming).
    - Counting tokens accurately according to the provider's model.
    """

    @abc.abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider with its specific configuration.

        Args:
            config: A dictionary containing provider-specific settings loaded
                    from the main LLMCore configuration (e.g., api_key,
                    base_url, default_model, timeout).
        """
        pass

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
    def get_available_models(self) -> List[str]:
        """
        Return a list of known or potentially available model names for this provider.

        This might involve an API call in some implementations or return a
        statically defined list based on the provider's known offerings.

        Returns:
            A list of model name strings.
        """
        pass

    @abc.abstractmethod
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Return the maximum context length (in tokens) for a specific model.

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
        context: ContextPayload, # Updated type hint
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Perform a chat completion request to the provider's API.

        This is the core method for interacting with the LLM. It sends the
        prepared context and receives the model's response.

        Args:
            context: The context payload to send. This can be either:
                     - A list of `llmcore.models.Message` objects.
                     - An MCP (Model Context Protocol) Context object, if MCP is enabled
                       and the `modelcontextprotocol` library is installed.
                       Providers need to implement logic to handle this object type.
            model: The specific model identifier to use for this completion.
                   If None, the provider's configured default model should be used.
            stream: If True, the method should return an asynchronous generator
                    yielding raw response chunks (dictionaries) from the API as
                    they arrive. If False, it should return the complete, final
                    response dictionary after the API call finishes.
            **kwargs: Additional provider-specific parameters (e.g., temperature,
                      max_tokens for the response, top_p, stop sequences).

        Returns:
            - If stream=False: A dictionary representing the complete API response.
              The structure depends on the provider (e.g., OpenAI's format).
            - If stream=True: An asynchronous generator yielding dictionaries,
              where each dictionary is a raw chunk received from the streaming API.

        Raises:
            ProviderError: If the API call fails due to issues like authentication,
                           rate limits, network errors, or invalid requests.
            NotImplementedError: If streaming is requested but not supported by the provider,
                                or if MCP context is received but not handled by the provider.
        """
        pass

    @abc.abstractmethod
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count the number of tokens a given text string would consume for a specific model.

        Uses the provider's specific tokenizer or token counting method.
        This method is typically synchronous as tokenizers often operate locally.

        Args:
            text: The text string to count tokens for.
            model: The specific model context for token counting. If None,
                   uses the provider's default model.

        Returns:
            The number of tokens.
        """
        pass

    # --- Updated Method Signature ---
    @abc.abstractmethod
    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Asynchronously count the total number of tokens a list of messages would consume.

        Accounts for the provider's specific formatting overhead (e.g., roles,
        special tokens between messages) in addition to the content tokens.
        This is async because some providers might require an API call for accurate counting.
        Note: This method currently only supports counting tokens for the standard
        `List[Message]` format, not directly for MCP objects. MCP token counting
        might require separate handling or provider-specific methods if the structure
        significantly differs from a simple message list.

        Args:
            messages: A list of `llmcore.models.Message` objects.
            model: The specific model context for token counting. If None,
                   uses the provider's default model.

        Returns:
            The total estimated token count for the messages.
        """
        pass
    # --- End Update ---

    # Optional: Add an async close method if providers need cleanup
    async def close(self) -> None:
         """Clean up resources like network sessions if needed."""
         pass # Default implementation does nothing
