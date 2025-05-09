# src/llmcore/providers/openai_provider.py
"""
OpenAI API provider implementation for the LLMCore library.

Handles interactions with the OpenAI API (GPT models).
Can accept context as List[Message] or MCPContextObject.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING

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

# Conditional MCP imports
if TYPE_CHECKING:
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any
        MCPMessage = Any
        MCPRole = Any
        RetrievedKnowledge = Any
        mcp_library_available = False
else:
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any
        MCPMessage = Any
        MCPRole = Any
        RetrievedKnowledge = Any
        mcp_library_available = False


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError, MCPError
from .base import BaseProvider, ContextPayload # ContextPayload includes MCPContextObject

logger = logging.getLogger(__name__)

# Default context lengths (remains the same)
DEFAULT_OPENAI_TOKEN_LIMITS = {
    "gpt-4o": 128000,"gpt-4o-2024-05-13": 128000,
    "gpt-4-turbo": 128000,"gpt-4-turbo-preview": 128000,"gpt-4-0125-preview": 128000,"gpt-4-1106-preview": 128000,"gpt-4-vision-preview": 128000,
    "gpt-4": 8192,"gpt-4-0613": 8192,"gpt-4-32k": 32768,"gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-0125": 16385,"gpt-3.5-turbo": 16385,"gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-0613": 4096,"gpt-3.5-turbo-16k-0613": 16385,
}
DEFAULT_MODEL = "gpt-4o"

# Mapping from MCP Role Enum to OpenAI Role string (if MCP library is available)
MCP_TO_OPENAI_ROLE_MAP: Dict[Any, str] = {}
if mcp_library_available:
    MCP_TO_OPENAI_ROLE_MAP = {
        MCPRole.SYSTEM: "system",
        MCPRole.USER: "user",
        MCPRole.ASSISTANT: "assistant",
    }

class OpenAIProvider(BaseProvider):
    """
    LLMCore provider for interacting with the OpenAI API.
    Handles both List[Message] and MCPContextObject context types.
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
        self.base_url = config.get('base_url')
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = float(config.get('timeout', 60.0))

        if not self.api_key:
            logger.warning("OpenAI API key not found in config or environment variable OPENAI_API_KEY.")

        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.debug("AsyncOpenAI client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise ConfigError(f"OpenAI client initialization failed: {e}")

        self._load_tokenizer(self.default_model)

    def _load_tokenizer(self, model_name: str):
        """Loads the tiktoken tokenizer for the specified model."""
        # (Tokenizer loading logic remains the same)
        if not tiktoken: return
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
            logger.debug(f"Loaded tiktoken encoding for model: {model_name}")
        except KeyError:
            logger.warning(f"No specific tiktoken encoding found for model '{model_name}'. Using default 'cl100k_base'.")
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding: {e}", exc_info=True)
            self._encoding = None

    def get_name(self) -> str:
        """Returns the provider name."""
        return "openai"

    def get_available_models(self) -> List[str]:
        """Returns a static list of known default models for OpenAI."""
        # (Remains the same)
        logger.warning("OpenAIProvider.get_available_models() returning static list.")
        return list(DEFAULT_OPENAI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length for the given OpenAI model."""
        # (Remains the same)
        model_name = model or self.default_model
        limit = DEFAULT_OPENAI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            if model_name.startswith("gpt-4o"): limit = 128000
            elif model_name.startswith("gpt-4-turbo"): limit = 128000
            elif model_name.startswith("gpt-4-32k"): limit = 32768
            elif model_name.startswith("gpt-4"): limit = 8192
            elif model_name.startswith("gpt-3.5-turbo-16k"): limit = 16385
            elif model_name.startswith("gpt-3.5-turbo"): limit = 16385
            else: limit = 4096; logger.warning(f"Unknown context length for OpenAI model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    def _format_mcp_knowledge(self, knowledge: Optional[List[Any]]) -> Optional[str]:
        """Formats MCP RetrievedKnowledge into a string for system prompt."""
        if not knowledge:
            return None
        parts = ["--- Retrieved Context ---"]
        for item in knowledge:
            source_info = "Unknown Source"
            if item.source_metadata:
                source_info = item.source_metadata.get("source", item.source_metadata.get("doc_id", "Unknown Source"))
            content_snippet = item.content.replace('\n', ' ').strip()
            parts.append(f"\n[Source: {source_info}]\n{content_snippet}")
        parts.append("--- End Context ---")
        return "\n".join(parts)

    async def chat_completion(
        self,
        context: ContextPayload, # Accepts List[Message] or MCPContextObject
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the OpenAI API.
        Handles both standard List[Message] context and MCPContextObject.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "OpenAI client not initialized. API key might be missing.")

        model_name = model or self.default_model
        messages_payload: List[Dict[str, str]] = []
        knowledge_string: Optional[str] = None

        if isinstance(context, list) and all(isinstance(msg, Message) for msg in context):
            logger.debug("Processing context as List[Message] for OpenAI.")
            # Standard path: Convert List[Message] to OpenAI format
            messages_payload = [{"role": str(msg.role), "content": msg.content} for msg in context]

        elif mcp_library_available and isinstance(context, MCPContextObject):
            logger.debug("Processing context as MCPContextObject for OpenAI.")
            # MCP Path: Convert MCP messages and handle knowledge
            if not mcp_library_available: # Double check just in case
                 raise MCPError("MCP library not found at runtime, cannot process MCP context.")

            # Convert MCP messages
            for mcp_msg in context.messages:
                openai_role = MCP_TO_OPENAI_ROLE_MAP.get(mcp_msg.role)
                if openai_role:
                    messages_payload.append({"role": openai_role, "content": mcp_msg.content})
                else:
                    logger.warning(f"Skipping MCP message with unmappable role: {mcp_msg.role}")

            # Format retrieved knowledge for inclusion (best effort)
            knowledge_string = self._format_mcp_knowledge(context.retrieved_knowledge)

        else:
            # Context is neither List[Message] nor a valid MCP object
            raise ProviderError(self.get_name(), f"OpenAIProvider received unsupported context type: {type(context).__name__}.")

        # Prepend formatted knowledge as a system message if present
        if knowledge_string:
            logger.debug("Prepending formatted MCP knowledge as system message.")
            messages_payload.insert(0, {"role": "system", "content": knowledge_string})

        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")

        logger.debug(f"Sending request to OpenAI API: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")

        try:
            response_stream = await self._client.chat.completions.create(
                model=model_name,
                messages=messages_payload, # type: ignore # Correct type hint for messages
                stream=stream,
                **kwargs
            )

            if stream:
                logger.debug(f"Processing stream response from OpenAI model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    async for chunk in response_stream: # type: ignore
                        yield chunk.model_dump()
                return stream_wrapper()
            else:
                logger.debug(f"Processing non-stream response from OpenAI model '{model_name}'")
                return response_stream.model_dump() # type: ignore

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e.status_code} - {e.message}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error ({e.status_code}): {e.message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to OpenAI timed out after {self.timeout} seconds.")
            raise ProviderError(self.get_name(), f"Request timed out after {self.timeout}s.")
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int: # Changed to async
        """Counts tokens using the tiktoken tokenizer."""
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for OpenAIProvider. Using character approximation.")
            return (len(text) + 3) // 4
        if not text: return 0
        try:
            # Tiktoken encode is synchronous
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.error(f"Tiktoken encoding failed: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int: # Changed to async
        """Counts tokens for List[Message] using tiktoken, including OpenAI's overhead."""
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for token counting. Returning 0.")
            return 0

        model_name = model or self.default_model
        # (Overhead calculation logic remains the same)
        if model_name in {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613"}:
            tokens_per_message = 3; tokens_per_name = 1
        elif model_name == "gpt-3.5-turbo-0301":
            tokens_per_message = 4; tokens_per_name = -1
        elif "gpt-3.5-turbo" in model_name or "gpt-4" in model_name or "gpt-4o" in model_name:
            tokens_per_message = 3; tokens_per_name = 1
        else:
            logger.warning(f"count_message_tokens() may not be accurate for model {model_name}. Using fallback settings.")
            tokens_per_message = 3; tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            try:
                num_tokens += tokens_per_message
                role_str = str(message.role)
                # Synchronous encoding
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                 logger.error(f"Tiktoken encoding failed for message content/role: {e}. Using approximation.")
                 role_str_for_approx = str(message.role)
                 num_tokens += (len(message.content) + len(role_str_for_approx) + 15) // 4

        num_tokens += 3 # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def close(self) -> None:
        """Closes the underlying OpenAI client session if applicable."""
        # (Remains the same)
        if self._client:
            try:
                await self._client.close()
                logger.info("OpenAIProvider client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OpenAIProvider client: {e}", exc_info=True)
            finally:
                self._client = None
