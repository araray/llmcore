# src/llmcore/providers/github_mcp_provider.py
"""
Provider implementation for interacting with a GitHub MCP Server instance.

Sends requests formatted according to the Model Context Protocol (MCP)
to a specified server endpoint.
"""

import asyncio
import logging
import os
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING

import aiohttp

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

# Tiktoken for token counting (assuming underlying model might be OpenAI-like)
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None # type: ignore


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError, MCPError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default endpoint path for github-mcp-server
DEFAULT_MCP_ENDPOINT_PATH = "/v1/mcp/chat/completions"

# Mapping from LLMCore Role Enum to MCP Role Enum (if MCP library is available)
LLMCORE_TO_MCP_ROLE_MAP: Dict[LLMCoreRole, Any] = {}
if mcp_library_available:
    LLMCORE_TO_MCP_ROLE_MAP = {
        LLMCoreRole.SYSTEM: MCPRole.SYSTEM,
        LLMCoreRole.USER: MCPRole.USER,
        LLMCoreRole.ASSISTANT: MCPRole.ASSISTANT,
    }

# Default token limits - These are placeholders. The actual limit depends
# on the underlying model configured in the github-mcp-server instance.
# It's best to configure this per instance.
DEFAULT_GITHUB_MCP_TOKEN_LIMITS = {
    "default": 8192, # Placeholder default
}

class GithubMCPProvider(BaseProvider):
    """
    LLMCore provider for interacting with a github-mcp-server endpoint.

    This provider *always* sends context formatted using the Model Context Protocol.
    It requires the `modelcontextprotocol` library to be installed.
    """
    _session: Optional[aiohttp.ClientSession] = None
    _base_url: str
    _endpoint_path: str
    _timeout: float
    _default_model: str # Model name expected by the MCP server (might just be informational)
    _underlying_model_for_counting: Optional[str] = None # e.g., 'gpt-4' to use tiktoken
    _encoding: Optional[Any] = None # tiktoken encoding object

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GithubMCPProvider.

        Args:
            config: Configuration dictionary containing:
                    'base_url': Base URL of the running github-mcp-server instance (e.g., "http://localhost:8080"). Required.
                    'endpoint_path' (optional): API path for MCP chat completions (default: "/v1/mcp/chat/completions").
                    'default_model' (optional): Informational model name (default: "github-mcp-model").
                    'timeout' (optional): Request timeout in seconds (default: 60).
                    'underlying_model_for_counting' (optional): The name of the underlying model used by the
                                                                server for accurate token counting (e.g., "gpt-4o").
                                                                Required if using tiktoken for counting.
        """
        if not mcp_library_available:
            raise ImportError("GithubMCPProvider requires the 'modelcontextprotocol' library. Install with 'pip install llmcore[mcp]'.")
        # Tiktoken is needed if underlying model specified for counting
        self._underlying_model_for_counting = config.get("underlying_model_for_counting")
        if self._underlying_model_for_counting and not tiktoken_available:
             raise ImportError(f"Tiktoken library required for counting tokens for underlying model '{self._underlying_model_for_counting}'. Install with 'pip install tiktoken'.")

        self._base_url = config.get('base_url')
        if not self._base_url:
            raise ConfigError("GithubMCPProvider requires 'base_url' in its configuration.")

        # Remove trailing slash from base_url if present
        if self._base_url.endswith('/'):
            self._base_url = self._base_url[:-1]

        self._endpoint_path = config.get('endpoint_path', DEFAULT_MCP_ENDPOINT_PATH)
        # Ensure endpoint path starts with a slash
        if not self._endpoint_path.startswith('/'):
            self._endpoint_path = '/' + self._endpoint_path

        self._default_model = config.get('default_model', 'github-mcp-model') # Primarily informational
        self._timeout = float(config.get('timeout', 60.0))

        # Load tokenizer if underlying model specified
        if self._underlying_model_for_counting and tiktoken:
             self._load_tokenizer(self._underlying_model_for_counting)

        # Initialize aiohttp session later in an async context if needed, or create here
        # For simplicity, we might create it per request or manage it within chat_completion
        # Let's initialize it as None and create/manage within chat_completion or via close/init methods.
        logger.info(f"GithubMCPProvider configured for server at {self._base_url}{self._endpoint_path}")

    def _load_tokenizer(self, model_name: str):
        """Loads the tiktoken tokenizer for the specified underlying model."""
        if not tiktoken: return
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
            logger.info(f"Loaded tiktoken encoding for underlying model: {model_name}")
        except KeyError:
            logger.warning(f"No specific tiktoken encoding found for underlying model '{model_name}'. Using default 'cl100k_base'.")
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding for underlying model '{model_name}': {e}", exc_info=True)
            self._encoding = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Gets or creates the aiohttp session."""
        if self._session is None or self._session.closed:
            # You might want to configure connectors, timeouts etc. here
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
            logger.debug("Created new aiohttp.ClientSession for GithubMCPProvider.")
        return self._session

    def get_name(self) -> str:
        return "github_mcp" # Unique name for this provider

    def get_available_models(self) -> List[str]:
        """Returns the configured default model name."""
        # This provider interacts with a specific server, which uses an underlying model.
        # Returning the configured 'default_model' name.
        return [self._default_model]

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Returns the estimated maximum context length.
        This depends heavily on the underlying model configured in the github-mcp-server.
        Ideally, this should be configurable for the provider.
        Using a placeholder default for now.
        """
        # Use the underlying model name if provided for counting, otherwise default
        model_key = self._underlying_model_for_counting or "default"
        limit = DEFAULT_GITHUB_MCP_TOKEN_LIMITS.get(model_key)
        if limit is None:
             limit = 8192 # General fallback
             logger.warning(f"Unknown context length for GithubMCPProvider (underlying model: {model_key}). Using fallback limit: {limit}.")
        return limit

    def _format_mcp_knowledge(self, knowledge: Optional[List[Any]]) -> Optional[List[RetrievedKnowledge]]:
        """Formats LLMCore ContextDocuments into MCP RetrievedKnowledge."""
        if not knowledge or not mcp_library_available:
            return None

        mcp_knowledge: List[RetrievedKnowledge] = []
        for doc in knowledge: # Assuming knowledge here is List[ContextDocument]
            if not isinstance(doc, ContextDocument): continue # Skip if not expected type

            source_meta = {"doc_id": doc.id}
            if isinstance(doc.metadata, dict):
                source = doc.metadata.get("source")
                if source and isinstance(source, str): source_meta["source"] = source
                # Add other simple string metadata fields if needed

            mcp_knowledge.append(RetrievedKnowledge(
                content=doc.content,
                source_metadata=source_meta,
                # score=doc.score # Add score if MCP schema supports it
            ))
        return mcp_knowledge if mcp_knowledge else None


    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None, # Model name might be ignored by the server endpoint
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the configured github-mcp-server endpoint.

        Ensures the context is formatted as an MCP object before sending.
        """
        if not mcp_library_available:
             raise MCPError("GithubMCPProvider requires 'modelcontextprotocol' library, but it's not installed.")

        mcp_context_to_send: Optional[MCPContextObject] = None
        temp_knowledge_for_conversion: Optional[List[ContextDocument]] = None # Store RAG docs if converting List[Message]

        if isinstance(context, list) and all(isinstance(msg, Message) for msg in context):
            logger.debug("Converting List[Message] context to MCP format for GithubMCPProvider.")
            # Need to convert standard List[Message] to MCP format.
            # This logic might overlap with ContextManager, consider refactoring later.
            mcp_messages: List[MCPMessage] = []
            for msg in context:
                mcp_role = LLMCORE_TO_MCP_ROLE_MAP.get(msg.role)
                if mcp_role:
                    mcp_messages.append(MCPMessage(role=mcp_role, content=msg.content))
                else:
                    # Handle potential RAG marker message if ContextManager included it
                    if msg.id == "rag_context":
                         logger.debug("Found RAG marker message, knowledge should be handled separately.")
                         # We need the original RAG docs here. ContextManager doesn't pass them.
                         # This highlights a limitation: if context is List[Message], we lose the
                         # structured RAG info needed for proper MCP conversion here.
                         # Workaround: Assume ContextManager *didn't* include the RAG marker
                         # if the target is this provider, or require context to *always* be MCP.
                         # Let's assume for now RAG info isn't in List[Message] for this provider.
                         pass
                    else:
                         logger.warning(f"Skipping message with unmappable role for MCP conversion: {msg.role}")

            # If RAG was used, the knowledge needs to be added. But we don't have it here
            # if the input was List[Message]. This provider *must* receive MCPContextObject
            # if RAG is involved. Let's raise an error for now if RAG is needed but context isn't MCP.
            # A better fix is for ContextManager to *always* produce MCP if the provider requires it.
            # For now, we construct MCP context without knowledge if input is List[Message].
            mcp_context_to_send = MCPContextObject(messages=mcp_messages)


        elif isinstance(context, MCPContextObject):
            logger.debug("Processing pre-formatted MCPContextObject for GithubMCPProvider.")
            mcp_context_to_send = context # Use the already formatted MCP context
        else:
            raise ProviderError(self.get_name(), f"GithubMCPProvider received unsupported context type: {type(context).__name__}.")

        if not mcp_context_to_send or not mcp_context_to_send.messages:
             raise ProviderError(self.get_name(), "No valid messages found in context to send.")

        # Serialize MCP context to JSON
        try:
            # Use model_dump_json for Pydantic v2 models
            mcp_payload_json_str = mcp_context_to_send.model_dump_json(exclude_none=True)
            mcp_payload_dict = json.loads(mcp_payload_json_str) # Send as dict
        except AttributeError:
             # Fallback for older Pydantic or different MCP object structure
             try:
                 mcp_payload_dict = mcp_context_to_send.dict(exclude_none=True) # type: ignore
             except Exception as json_err:
                 logger.error(f"Failed to serialize MCP context object: {json_err}", exc_info=True)
                 raise MCPError(f"Failed to serialize MCP context: {json_err}")
        except Exception as json_err:
            logger.error(f"Failed to serialize MCP context object: {json_err}", exc_info=True)
            raise MCPError(f"Failed to serialize MCP context: {json_err}")


        target_url = f"{self._base_url}{self._endpoint_path}"
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json" if not stream else "text/event-stream"
        }
        # Add Authorization header if needed (e.g., API key for the MCP server itself)
        # mcp_server_api_key = config.get("mcp_server_api_key")
        # if mcp_server_api_key: request_headers["Authorization"] = f"Bearer {mcp_server_api_key}"

        logger.debug(f"Sending MCP request to {target_url}, stream={stream}")

        session = await self._get_session()
        try:
            async with session.post(
                target_url,
                json=mcp_payload_dict, # Send the MCP payload
                headers=request_headers,
                params={"stream": str(stream).lower()} # Add stream param to URL query
            ) as response:
                response.raise_for_status() # Raise exception for 4xx/5xx errors

                if stream:
                    logger.debug(f"Processing stream response from Github MCP Server")
                    # Stream response line by line (assuming Server-Sent Events)
                    async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                        try:
                            async for line in response.content:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith("data:"):
                                    data_str = line_str[len("data:"):].strip()
                                    if data_str == "[DONE]":
                                        logger.debug("Received stream [DONE] marker.")
                                        break
                                    try:
                                        chunk_dict = json.loads(data_str)
                                        yield chunk_dict
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to decode stream data JSON: {data_str}")
                                elif line_str: # Log other non-empty lines if needed
                                    logger.debug(f"Received non-data line in stream: {line_str}")
                        except aiohttp.ClientPayloadError as e:
                             logger.error(f"Payload error reading stream from {target_url}: {e}", exc_info=True)
                             raise ProviderError(self.get_name(), f"Stream payload error: {e}")
                        except Exception as e:
                             logger.error(f"Unexpected error processing stream from {target_url}: {e}", exc_info=True)
                             # Don't raise here, let the stream end potentially gracefully if possible
                        finally:
                             logger.debug("Github MCP Server stream finished.")
                    return stream_wrapper()
                else: # Non-streaming
                    logger.debug(f"Processing non-stream response from Github MCP Server")
                    response_json = await response.json()
                    # Assume the response format is similar to OpenAI's completion object
                    return response_json

        except aiohttp.ClientResponseError as e:
            logger.error(f"Github MCP Server request failed: {e.status} {e.message}", exc_info=True)
            try:
                error_detail = await e.response.text() # Try to get error body
            except Exception:
                error_detail = e.message
            raise ProviderError(self.get_name(), f"Server Error ({e.status}): {error_detail[:500]}") # Limit error detail length
        except asyncio.TimeoutError:
            logger.error(f"Request to Github MCP Server at {target_url} timed out after {self._timeout} seconds.")
            raise ProviderError(self.get_name(), f"Request timed out after {self._timeout}s.")
        except aiohttp.ClientConnectionError as e:
             logger.error(f"Could not connect to Github MCP Server at {target_url}: {e}", exc_info=True)
             raise ProviderError(self.get_name(), f"Could not connect to server: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Github MCP Server request: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using tiktoken based on the configured underlying model."""
        if not self._encoding:
            logger.warning("Tokenizer for underlying model not loaded. Using character approximation for GithubMCPProvider.")
            return (len(text) + 3) // 4 if text else 0
        if not text: return 0
        try: return len(self._encoding.encode(text))
        except Exception as e:
            logger.error(f"Tiktoken encoding failed for underlying model '{self._underlying_model_for_counting}': {e}", exc_info=True)
            return (len(text) + 3) // 4

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Counts tokens for List[Message] using tiktoken based on the underlying model.
        Note: This counts based on the List[Message] structure, *before* potential
        MCP formatting which might have slightly different overhead.
        """
        if not self._encoding:
            logger.warning("Tokenizer for underlying model not loaded. Using character approximation.")
            total_chars = sum(len(msg.content) for msg in messages)
            return (total_chars + (len(messages) * 15)) // 4 # Rough approximation

        # Use OpenAI's overhead calculation as a reasonable default heuristic
        # unless the underlying model is known to be different.
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            try:
                num_tokens += tokens_per_message
                role_str = str(message.role)
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                 logger.error(f"Tiktoken encoding failed for message: {e}. Using approximation.")
                 role_str_for_approx = str(message.role)
                 num_tokens += (len(message.content) + len(role_str_for_approx) + 15) // 4
        num_tokens += 3 # Prime reply
        return num_tokens

    async def close(self) -> None:
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("GithubMCPProvider aiohttp session closed.")
        self._session = None
