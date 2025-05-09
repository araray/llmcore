# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
Can accept context as List[Message] or MCPContextObject.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING

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

# Conditional MCP imports
if TYPE_CHECKING:
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any; MCPMessage = Any; MCPRole = Any; RetrievedKnowledge = Any; mcp_library_available = False
else:
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any; MCPMessage = Any; MCPRole = Any; RetrievedKnowledge = Any; mcp_library_available = False


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError, MCPError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths (remains the same)
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8192,"llama3:8b": 8192,"llama3:70b": 8192, "gemma3:4b": 4096, # Added from user log
    "falcon3:3b": 4096,
    "mistral": 8192,"mistral:7b": 8192,"mixtral": 32768,"mixtral:8x7b": 32768,
    "gemma": 8192,"gemma:7b": 8192,"gemma:2b": 8192,
    "phi3": 4096,"phi3:mini": 4096,
    "codellama": 16384,"codellama:7b": 16384,"codellama:13b": 16384,"codellama:34b": 16384,
    "llama2": 4096,"llama2:7b": 4096,"llama2:13b": 4096,"llama2:70b": 4096,
}
DEFAULT_MODEL = "llama3"

# Mapping from MCP Role Enum to Ollama Role string (if MCP library is available)
MCP_TO_OLLAMA_ROLE_MAP: Dict[Any, str] = {}
if mcp_library_available:
    MCP_TO_OLLAMA_ROLE_MAP = {
        MCPRole.SYSTEM: "system",
        MCPRole.USER: "user",
        MCPRole.ASSISTANT: "assistant",
    }

class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama using the official library.
    Handles both List[Message] and MCPContextObject context types.
    """
    _client: Optional[AsyncClient] = None
    _encoding: Optional[Any] = None # tiktoken encoding object

    def __init__(self, config: Dict[str, Any]):
        """Initializes the OllamaProvider using the official ollama library."""
        # (Initialization logic remains the same)
        if not ollama_available: raise ImportError("Ollama library is not installed.")
        if not tiktoken_available: raise ImportError("tiktoken library required for Ollama token counting.")
        self.host = config.get("host"); self.default_model = config.get("default_model", DEFAULT_MODEL); timeout_val = config.get("timeout"); self.timeout = float(timeout_val) if timeout_val is not None else None
        try:
            client_args = {};
            if self.host: client_args['host'] = self.host
            if self.timeout: client_args['timeout'] = self.timeout
            self._client = AsyncClient(**client_args); logger.debug(f"Ollama AsyncClient initialized (Host: {self.host or 'default'})")
        except Exception as e: logger.error(f"Failed to initialize Ollama AsyncClient: {e}", exc_info=True); raise ConfigError(f"Ollama client initialization failed: {e}")
        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base"); self._encoding = None
        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken:
                try: encoding_name = self.tokenizer_name.split("tiktoken_")[1]; self._encoding = tiktoken.get_encoding(encoding_name); logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name}")
                except Exception as e: logger.warning(f"Failed to load tiktoken encoding '{self.tokenizer_name}'. Falling back to char approx. Error: {e}"); self.tokenizer_name = "char_div_4"
            else: logger.warning("tiktoken library not available. Falling back to char approx."); self.tokenizer_name = "char_div_4"
        elif self.tokenizer_name != "char_div_4": logger.warning(f"Unsupported Ollama tokenizer '{self.tokenizer_name}'. Falling back to 'char_div_4'."); self.tokenizer_name = "char_div_4"
        if self.tokenizer_name == "char_div_4": logger.info("OllamaProvider using char division approximation for token counting.")

    def get_name(self) -> str: return "ollama"
    async def _fetch_ollama_models(self) -> List[str]:
        # (Remains the same)
        if not self._client: raise ProviderError(self.get_name(), "Ollama client not initialized.")
        try: models_info = await self._client.list(); models = [m.get("name") for m in models_info.get("models", []) if m.get("name")]; logger.debug(f"Fetched {len(models)} models from Ollama API via client: {models}"); return models
        except ResponseError as e: logger.warning(f"Failed to fetch models from Ollama API: HTTP {e.status_code} - {e.error}"); return []
        except Exception as e: logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True); return []
    def get_available_models(self) -> List[str]:
        # (Remains the same)
        logger.warning("OllamaProvider.get_available_models() returning static list.")
        return list(DEFAULT_OLLAMA_TOKEN_LIMITS.keys())
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        # (Remains the same)
        model_name = model or self.default_model; base_model_name = model_name.split(':')[0]; limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(model_name);
        if limit is None: limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)
        if limit is None:
            if model_name == "falcon3:3b": limit = 4096; logger.info(f"Using context length 4096 for Ollama model '{model_name}'.")
            elif model_name == "gemma3:4b": limit = 4096; logger.info(f"Using context length 4096 for Ollama model '{model_name}'.") # From user log
            else: limit = 4096; logger.warning(f"Unknown context length for Ollama model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    def _format_mcp_knowledge(self, knowledge: Optional[List[Any]]) -> Optional[str]:
        """Formats MCP RetrievedKnowledge into a string for system prompt."""
        # (Helper function remains the same)
        if not knowledge: return None
        parts = ["--- Retrieved Context ---"]
        for item in knowledge:
            source_info = "Unknown Source";
            if item.source_metadata: source_info = item.source_metadata.get("source", item.source_metadata.get("doc_id", "Unknown Source"))
            content_snippet = item.content.replace('\n', ' ').strip(); parts.append(f"\n[Source: {source_info}]\n{content_snippet}")
        parts.append("--- End Context ---")
        return "\n".join(parts)

    async def chat_completion(
        self,
        context: ContextPayload, # Accepts List[Message] or MCPContextObject
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Sends a chat completion request to the Ollama API using the ollama library."""
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")

        model_name = model or self.default_model
        messages_payload: List[Dict[str, str]] = []
        knowledge_string: Optional[str] = None

        if isinstance(context, list) and all(isinstance(msg, Message) for msg in context):
            logger.debug("Processing context as List[Message] for Ollama.")
            # Standard path: Use LLMCore messages directly
            messages_payload = [{"role": str(msg.role), "content": msg.content} for msg in context]

        elif mcp_library_available and isinstance(context, MCPContextObject):
            logger.debug("Processing context as MCPContextObject for Ollama.")
            if not mcp_library_available: raise MCPError("MCP library not found at runtime.")

            # Convert MCP messages
            for mcp_msg in context.messages:
                ollama_role = MCP_TO_OLLAMA_ROLE_MAP.get(mcp_msg.role)
                if ollama_role:
                    messages_payload.append({"role": ollama_role, "content": mcp_msg.content})
                else:
                    logger.warning(f"Skipping MCP message with unmappable role: {mcp_msg.role}")

            # Format knowledge to prepend as system message
            knowledge_string = self._format_mcp_knowledge(context.retrieved_knowledge)

        else:
            raise ProviderError(self.get_name(), f"OllamaProvider received unsupported context type: {type(context).__name__}.")

        # Prepend formatted knowledge as a system message if present
        if knowledge_string:
            logger.debug("Prepending formatted MCP knowledge as system message for Ollama.")
            # Find if a system message already exists
            existing_system_idx = -1
            for i, msg in enumerate(messages_payload):
                if msg["role"] == "system":
                    existing_system_idx = i
                    break
            if existing_system_idx != -1:
                 # Append knowledge to existing system message
                 messages_payload[existing_system_idx]["content"] = f"{messages_payload[existing_system_idx]['content']}\n\n{knowledge_string}"
            else:
                 # Insert knowledge as the first system message
                 messages_payload.insert(0, {"role": "system", "content": knowledge_string})

        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")


        logger.debug(f"Sending request to Ollama via client: model='{model_name}', stream={stream}, num_messages={len(messages_payload)}")

        try:
            options = kwargs if kwargs else None
            response_or_stream_obj: Union[ChatResponse, AsyncGenerator[Dict[str, Any], None]] = await self._client.chat(
                model=model_name, messages=messages_payload, stream=stream, options=options, # type: ignore
            )

            if stream:
                logger.debug(f"Processing stream response from Ollama model '{model_name}'")
                return response_or_stream_obj # type: ignore
            else:
                logger.debug(f"Processing non-stream response from Ollama model '{model_name}'")
                if isinstance(response_or_stream_obj, dict): return response_or_stream_obj
                if ChatResponse and isinstance(response_or_stream_obj, ChatResponse): return response_or_stream_obj.model_dump()
                else: logger.error(f"Unexpected response type for non-streaming Ollama chat: {type(response_or_stream_obj)}"); raise ProviderError(self.get_name(), "Invalid response format (non-streaming).")

        except ResponseError as e:
            error_detail = e.error if hasattr(e, 'error') else str(e)
            logger.error(f"Ollama API error: {e.status_code} - {error_detail}", exc_info=True)
            if error_detail and "model not found" in str(error_detail).lower(): raise ProviderError(self.get_name(), f"Model '{model_name}' not found. Pull it using 'ollama pull {model_name}'.")
            raise ProviderError(self.get_name(), f"API Error ({e.status_code}): {error_detail}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Ollama timed out (timeout: {self.timeout or 'default'}).")
            raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e:
            logger.error(f"Unexpected error during Ollama chat completion: {e}", exc_info=True)
            if "connect" in str(e).lower(): raise ProviderError(self.get_name(), f"Could not connect to Ollama at {self.host or 'default'}. Is it running? Details: {e}")
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int: # Changed to async
        """Counts tokens using the configured tokenizer (tiktoken) or approximation."""
        if not self._encoding:
            # This part is synchronous, so it's fine within an async method
            return (len(text) + 3) // 4 if text else 0
        if not text:
            return 0
        try:
            # This part is synchronous
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.error(f"Tiktoken encoding failed: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int: # Changed to async
        """Counts tokens for List[Message] using the configured method (tiktoken)."""
        if not self._encoding:
            logger.warning("Tiktoken encoding not available for token counting. Using approximation.")
            total_chars = sum(len(msg.content) for msg in messages)
            return (total_chars + (len(messages) * 15)) // 4 # Synchronous

        overhead_per_message = 5
        total_tokens = 0
        for msg in messages:
            try:
                # Synchronous encoding
                content_tokens = len(self._encoding.encode(msg.content))
                role_str = str(msg.role)
                role_tokens = len(self._encoding.encode(role_str))
                total_tokens += content_tokens + role_tokens + overhead_per_message
            except Exception as e:
                logger.error(f"Tiktoken encoding failed for message content/role: {e}. Using approximation.")
                role_str_for_approx = str(msg.role)
                total_tokens += (len(msg.content) + len(role_str_for_approx) + 15) // 4
        total_tokens += 3
        return total_tokens

    async def close(self) -> None:
        """Closes the underlying Ollama client session if applicable."""
        # (Remains the same)
        if self._client:
            logger.debug("Closing OllamaProvider client...")
            if hasattr(self._client, 'aclose') and asyncio.iscoroutinefunction(self._client.aclose):
                try: await self._client.aclose(); logger.info("OllamaProvider client closed successfully.")
                except Exception as e: logger.error(f"Error closing OllamaProvider client: {e}", exc_info=True)
            else: logger.debug("Ollama AsyncClient does not have 'aclose'. Closure handled by library/GC.")
            self._client = None
