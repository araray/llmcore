# src/llmcore/providers/anthropic_provider.py
"""
Anthropic API provider implementation for the LLMCore library.

Handles interactions with the Anthropic API (Claude models).
Uses the official 'anthropic' Python SDK.
Can accept context as List[Message] or MCPContextObject.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple, TYPE_CHECKING

# Use the official anthropic library
try:
    import anthropic
    from anthropic import AsyncAnthropic, AnthropicError
    from anthropic.types import MessageParam, TextBlockParam, ToolParam # For constructing messages
    anthropic_available = True
except ImportError:
    anthropic_available = False
    AsyncAnthropic = None # type: ignore
    AnthropicError = Exception # type: ignore
    MessageParam = Dict[str, Any] # type: ignore
    TextBlockParam = Dict[str, Any] # type: ignore
    ToolParam = Dict[str, Any] # type: ignore

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
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths (remains the same)
DEFAULT_ANTHROPIC_TOKEN_LIMITS = {
    "claude-3-opus-20240229": 200000,"claude-3-sonnet-20240229": 200000,"claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,"claude-2.0": 100000,"claude-instant-1.2": 100000,
}
DEFAULT_MODEL = "claude-3-haiku-20240307"

# Mapping from MCP Role Enum to Anthropic Role string (if MCP library is available)
MCP_TO_ANTHROPIC_ROLE_MAP: Dict[Any, str] = {}
if mcp_library_available:
    MCP_TO_ANTHROPIC_ROLE_MAP = {
        # MCPRole.SYSTEM handled separately by Anthropic API
        MCPRole.USER: "user",
        MCPRole.ASSISTANT: "assistant",
    }


class AnthropicProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Anthropic API (Claude models).
    Handles both List[Message] and MCPContextObject context types.
    """
    _client: Optional[AsyncAnthropic] = None
    _sync_client_for_counting: Optional[anthropic.Anthropic] = None

    def __init__(self, config: Dict[str, Any]):
        """Initializes the AnthropicProvider."""
        # (Initialization logic remains the same)
        if not anthropic_available:
            raise ImportError("Anthropic library is not installed. Please install `anthropic` or `llmcore[anthropic]`.")

        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = config.get('base_url')
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = float(config.get('timeout', 60.0))

        if not self.api_key:
            logger.warning("Anthropic API key not found in config or environment variable ANTHROPIC_API_KEY.")

        try:
            self._client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            self._sync_client_for_counting = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            logger.debug("AsyncAnthropic client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncAnthropic client: {e}", exc_info=True)
            raise ConfigError(f"Anthropic client initialization failed: {e}")

    def get_name(self) -> str: return "anthropic"
    def get_available_models(self) -> List[str]:
        # (Remains the same)
        logger.warning("AnthropicProvider.get_available_models() returning static list.")
        return list(DEFAULT_ANTHROPIC_TOKEN_LIMITS.keys())
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        # (Remains the same)
        model_name = model or self.default_model
        limit = DEFAULT_ANTHROPIC_TOKEN_LIMITS.get(model_name)
        if limit is None: limit = 100000; logger.warning(f"Unknown context length for Anthropic model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    def _convert_llmcore_msgs_to_anthropic(
        self,
        messages: List[Message]
    ) -> tuple[Optional[str], List[MessageParam]]:
        """Converts LLMCore List[Message] to Anthropic format."""
        # (This internal helper remains largely the same, operating on LLMCore Message objects)
        anthropic_messages: List[MessageParam] = []
        system_prompt: Optional[str] = None
        processed_messages = list(messages)
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_prompt = processed_messages.pop(0).content
            logger.debug("System prompt extracted for Anthropic request from List[Message].")

        if not processed_messages or processed_messages[0].role != LLMCoreRole.USER:
            logger.warning("Anthropic conversation (from List[Message]) should start with a 'user' role.")

        last_role = None
        for msg in processed_messages:
            anthropic_role_str = ""
            if msg.role == LLMCoreRole.USER: anthropic_role_str = "user"
            elif msg.role == LLMCoreRole.ASSISTANT: anthropic_role_str = "assistant"
            else: logger.warning(f"Skipping message with unmappable role for Anthropic: {msg.role}"); continue

            if anthropic_role_str == last_role:
                logger.warning(f"Duplicate consecutive role '{anthropic_role_str}'. Merging content attempt.")
                if anthropic_messages and isinstance(anthropic_messages[-1]["content"], list):
                    last_content_block = anthropic_messages[-1]["content"][0]
                    if last_content_block["type"] == "text": # type: ignore
                        last_content_block["text"] += f"\n{msg.content}" # type: ignore
                        continue

            content_block: TextBlockParam = {"type": "text", "text": msg.content}
            anthropic_messages.append({"role": anthropic_role_str, "content": [content_block]}) # type: ignore
            last_role = anthropic_role_str

        return system_prompt, anthropic_messages

    def _convert_mcp_msgs_to_anthropic(
        self,
        mcp_messages: List[Any] # List[MCPMessage]
    ) -> tuple[Optional[str], List[MessageParam]]:
        """Converts MCPMessage list to Anthropic format."""
        anthropic_messages: List[MessageParam] = []
        system_prompt: Optional[str] = None

        processed_mcp_messages = list(mcp_messages)
        # Extract system prompt if present
        if processed_mcp_messages and processed_mcp_messages[0].role == MCPRole.SYSTEM:
            system_prompt = processed_mcp_messages.pop(0).content
            logger.debug("System prompt extracted for Anthropic request from MCP context.")

        if not processed_mcp_messages or processed_mcp_messages[0].role != MCPRole.USER:
             logger.warning("Anthropic conversation (from MCP) should start with a 'user' role.")

        last_role = None
        for mcp_msg in processed_mcp_messages:
            anthropic_role_str = MCP_TO_ANTHROPIC_ROLE_MAP.get(mcp_msg.role)
            if not anthropic_role_str:
                logger.warning(f"Skipping MCP message with unmappable role for Anthropic: {mcp_msg.role}")
                continue

            if anthropic_role_str == last_role:
                 logger.warning(f"Duplicate consecutive MCP role '{anthropic_role_str}'. Merging content attempt.")
                 if anthropic_messages and isinstance(anthropic_messages[-1]["content"], list):
                     last_content_block = anthropic_messages[-1]["content"][0]
                     if last_content_block["type"] == "text": # type: ignore
                         last_content_block["text"] += f"\n{mcp_msg.content}" # type: ignore
                         continue

            content_block: TextBlockParam = {"type": "text", "text": mcp_msg.content}
            anthropic_messages.append({"role": anthropic_role_str, "content": [content_block]}) # type: ignore
            last_role = anthropic_role_str

        return system_prompt, anthropic_messages

    def _format_mcp_knowledge(self, knowledge: Optional[List[Any]]) -> Optional[str]:
        """Formats MCP RetrievedKnowledge into a string for system prompt."""
        # (Helper function remains the same as in OpenAI provider)
        if not knowledge: return None
        parts = ["--- Retrieved Context ---"]
        for item in knowledge:
            source_info = "Unknown Source"
            if item.source_metadata: source_info = item.source_metadata.get("source", item.source_metadata.get("doc_id", "Unknown Source"))
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
        """Sends a chat completion request to the Anthropic API."""
        if not self._client:
            raise ProviderError(self.get_name(), "Anthropic client not initialized.")

        model_name = model or self.default_model
        messages_payload: List[MessageParam] = []
        system_prompt: Optional[str] = None
        knowledge_string: Optional[str] = None

        if isinstance(context, list) and all(isinstance(msg, Message) for msg in context):
            logger.debug("Processing context as List[Message] for Anthropic.")
            system_prompt, messages_payload = self._convert_llmcore_msgs_to_anthropic(context)

        elif mcp_library_available and isinstance(context, MCPContextObject):
            logger.debug("Processing context as MCPContextObject for Anthropic.")
            if not mcp_library_available: raise MCPError("MCP library not found at runtime.")

            # Convert MCP messages
            system_prompt, messages_payload = self._convert_mcp_msgs_to_anthropic(context.messages)
            # Format knowledge to be added to system prompt
            knowledge_string = self._format_mcp_knowledge(context.retrieved_knowledge)

        else:
            raise ProviderError(self.get_name(), f"AnthropicProvider received unsupported context type: {type(context).__name__}.")

        # Combine original system prompt and knowledge string
        final_system_prompt = system_prompt
        if knowledge_string:
            if final_system_prompt:
                final_system_prompt = f"{final_system_prompt}\n\n{knowledge_string}"
            else:
                final_system_prompt = knowledge_string
            logger.debug("Combined MCP knowledge with system prompt.")

        if not messages_payload:
             raise ProviderError(self.get_name(), "No valid messages to send after context processing.")
        # Anthropic requires first message to be 'user'
        if messages_payload[0]['role'] != 'user':
             # Attempt to fix by adding a dummy user message if it starts with assistant
             if messages_payload[0]['role'] == 'assistant':
                  logger.warning("Prepending dummy 'user' message as Anthropic requires conversation to start with user.")
                  messages_payload.insert(0, {"role": "user", "content": [{"type": "text", "text": "(Context starts)"}]})
             else: # Should not happen if conversion logic is correct
                  raise ProviderError(self.get_name(), "Anthropic API requires the first message to be from the 'user' role after processing.")


        max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_tokens_to_sample", 2048))

        logger.debug(
            f"Sending request to Anthropic API: model='{model_name}', stream={stream}, "
            f"num_messages={len(messages_payload)}, system_prompt_present={bool(final_system_prompt)}"
        )

        try:
            if stream:
                response_stream = await self._client.messages.stream(
                    model=model_name, messages=messages_payload, system=final_system_prompt, # type: ignore
                    max_tokens=max_tokens, temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"), top_k=kwargs.get("top_k"),
                    stop_sequences=kwargs.get("stop_sequences")
                )
                logger.debug(f"Processing stream response from Anthropic model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    # (Stream processing logic remains the same as before)
                    try:
                        async for event in response_stream:
                            event_dict = {}
                            if event.type == "message_start":
                                event_dict = {"type": "message_start", "message": {"id": event.message.id, "role": event.message.role, "model": event.message.model, "usage": event.message.usage.model_dump() if event.message.usage else None}}
                            elif event.type == "content_block_delta":
                                if event.delta.type == "text_delta": event_dict = {"type": "content_block_delta", "index": event.index, "delta": {"type": "text", "text": event.delta.text}, "choices": [{"delta": {"content": event.delta.text}, "index": event.index}]}
                            elif event.type == "message_delta": event_dict = {"type": "message_delta", "delta": {"stop_reason": event.delta.stop_reason, "stop_sequence": event.delta.stop_sequence}, "usage": event.usage.model_dump()}
                            elif event.type == "message_stop": event_dict = {"type": "message_stop", "reason": "stop_event"}
                            if event_dict: yield event_dict
                    except anthropic.APIConnectionError as e: logger.error(f"Anthropic API connection error during stream: {e}", exc_info=True); yield {"error": f"Anthropic API Connection Error: {e}", "done": True}; raise ProviderError(self.get_name(), f"API Connection Error during stream: {e}")
                    except anthropic.APIStatusError as e: logger.error(f"Anthropic API status error during stream: {e.status_code} - {e.message}", exc_info=True); yield {"error": f"Anthropic API Status Error ({e.status_code}): {e.message}", "done": True}; raise ProviderError(self.get_name(), f"API Status Error ({e.status_code}) during stream: {e.message}")
                    except Exception as e: logger.error(f"Unexpected error processing Anthropic stream: {e}", exc_info=True); yield {"error": f"Unexpected stream processing error: {e}", "done": True}
                    finally: logger.debug("Anthropic stream finished.")
                return stream_wrapper()
            else: # Non-streaming
                response = await self._client.messages.create(
                    model=model_name, messages=messages_payload, system=final_system_prompt, # type: ignore
                    max_tokens=max_tokens, temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"), top_k=kwargs.get("top_k"),
                    stop_sequences=kwargs.get("stop_sequences")
                )
                logger.debug(f"Processing non-stream response from Anthropic model '{model_name}'")
                return response.model_dump()

        except AnthropicError as e:
            status_code = getattr(e, 'status_code', None); error_message = str(e)
            logger.error(f"Anthropic API error (Status: {status_code}): {error_message}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error (Status: {status_code}): {error_message}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Anthropic timed out after {self.timeout} seconds.")
            raise ProviderError(self.get_name(), f"Request timed out after {self.timeout}s.")
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    # --- Token Counting Methods (remain the same) ---
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the Anthropic client."""
        # (Remains the same)
        if not self._sync_client_for_counting: logger.warning("Anthropic sync client for token counting not initialized."); return (len(text) + 3) // 4
        if not text: return 0
        try: return self._sync_client_for_counting.count_tokens(text)
        except AnthropicError as e: logger.error(f"Anthropic API error during token count: {e}", exc_info=True); raise ProviderError(self.get_name(), f"API Error during token count: {e}")
        except Exception as e: logger.error(f"Failed to count tokens with Anthropic client: {e}", exc_info=True); return (len(text) + 3) // 4

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for List[Message] using Anthropic client."""
        # Note: This method currently only counts tokens for List[Message] format.
        # Accurate token counting for MCP objects might require adjustments.
        # (Remains the same as before)
        if not self._sync_client_for_counting: logger.warning("Anthropic sync client for token counting not initialized."); total_text_len = sum(len(msg.content) + len(msg.role.value) for msg in messages); return (total_text_len + (len(messages) * 5)) // 4
        prompt_str = ""; system_prompt, anthropic_msgs_list = self._convert_llmcore_msgs_to_anthropic(messages)
        if system_prompt: prompt_str += f"System: {system_prompt}\n\n"
        for msg_param in anthropic_msgs_list:
            role = msg_param["role"]; content_str = ""
            if isinstance(msg_param["content"], list) and msg_param["content"]:
                first_block = msg_param["content"][0]
                if first_block.get("type") == "text": content_str = first_block.get("text", "")
            prompt_str += f"{role.capitalize()}: {content_str}\n"
        if prompt_str and anthropic_msgs_list and anthropic_msgs_list[-1]["role"] == "user": prompt_str += "Assistant:"
        if not prompt_str: return 0
        return self.count_tokens(prompt_str, model)

    async def close(self) -> None:
        """Closes the underlying Anthropic client sessions."""
        # (Remains the same)
        closed_async = False
        if self._client:
            try: await self._client.close(); logger.debug("AsyncAnthropic client closed."); closed_async = True
            except Exception as e: logger.error(f"Error closing AsyncAnthropic client: {e}", exc_info=True)
        if self._sync_client_for_counting:
            try: self._sync_client_for_counting.close(); logger.debug("Anthropic synchronous client closed.")
            except Exception as e: logger.error(f"Error closing Anthropic synchronous client: {e}", exc_info=True)
        self._client = None; self._sync_client_for_counting = None
