# src/llmcore/providers/gemini_provider.py
"""
Google Gemini API provider implementation for the LLMCore library.

Handles interactions with the Google Generative AI API (Gemini models).
Uses the official 'google-generativeai' library.
Can accept context as List[Message] or MCPContextObject.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple, TYPE_CHECKING

# Use the official google-generativeai library
try:
    import google.generativeai as genai
    from google.genai.types import GenerationConfig, ContentDict, HarmCategory, HarmBlockThreshold
    try: from google.api_core import exceptions as google_exceptions
    except ImportError: google_exceptions = None # type: ignore
    gemini_available = True
except ImportError:
    gemini_available = False
    genai = None; GenerationConfig = None; ContentDict = Dict[str, Any]; HarmCategory = None; HarmBlockThreshold = None; google_exceptions = None # type: ignore

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
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-1.5-pro-latest": 1048576,"gemini-1.5-flash-latest": 1048576,
    "gemini-1.0-pro": 32768,"gemini-pro": 32768,
    "gemini-1.0-pro-vision-latest": 16384,"gemini-pro-vision": 16384,
}
DEFAULT_MODEL = "gemini-1.5-pro-latest"

# Mapping from LLMCore Role to Gemini Role string
LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model",
    # System role handled separately
}
# Mapping from MCP Role Enum to Gemini Role string (if MCP library is available)
MCP_TO_GEMINI_ROLE_MAP: Dict[Any, str] = {}
if mcp_library_available:
    MCP_TO_GEMINI_ROLE_MAP = {
        MCPRole.USER: "user",
        MCPRole.ASSISTANT: "model",
        # MCPRole.SYSTEM handled separately
    }


class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API.
    Handles both List[Message] and MCPContextObject context types.
    """
    _model_instance_cache: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        """Initializes the GeminiProvider."""
        # (Initialization logic remains the same)
        if not gemini_available: raise ImportError("Google Generative AI library not installed.")
        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.timeout = config.get('timeout')
        self.safety_settings = self._parse_safety_settings(config.get('safety_settings'))
        if not self.api_key: logger.warning("Google AI API key not found in config or GOOGLE_API_KEY env var.")
        try: genai.configure(api_key=self.api_key); logger.debug("Google Generative AI client configured.")
        except Exception as e: logger.error(f"Failed to configure Google Generative AI client: {e}", exc_info=True); raise ConfigError(f"Google AI client configuration failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[Dict[Any, Any]]:
        # (Remains the same)
        if not settings_config or not HarmCategory or not HarmBlockThreshold: return None
        parsed_settings = {}
        for key, value in settings_config.items():
            try: category = getattr(HarmCategory, key.upper()); threshold = getattr(HarmBlockThreshold, value.upper()); parsed_settings[category] = threshold
            except AttributeError: logger.warning(f"Invalid safety setting category or threshold: {key}={value}. Skipping.")
        logger.debug(f"Parsed safety settings: {parsed_settings}"); return parsed_settings if parsed_settings else None

    def _get_model_instance(self, model_name: str, system_instruction: Optional[str] = None) -> Any:
        """Gets or creates a GenerativeModel instance, caching it (without system instruction). Applies system instruction if provided."""
        # Cache based on model_name only, apply system instruction dynamically
        cache_key = model_name
        if cache_key not in self._model_instance_cache:
            if not genai: raise ProviderError(self.get_name(), "Google Generative AI library not available.")
            try:
                model_instance = genai.GenerativeModel(model_name)
                self._model_instance_cache[cache_key] = model_instance
                logger.debug(f"Created GenerativeModel instance for '{model_name}'")
            except Exception as e:
                logger.error(f"Failed to initialize GenerativeModel for '{model_name}': {e}", exc_info=True)
                raise ProviderError(self.get_name(), f"Failed to get model instance '{model_name}': {e}")

        # Return the cached instance, potentially reconfigured with system instruction
        base_instance = self._model_instance_cache[cache_key]
        if system_instruction:
             # Create a new instance with system instruction based on the cached one's name
             # This avoids modifying the cached instance directly.
             try:
                 return genai.GenerativeModel(base_instance._model_name, system_instruction=system_instruction) # Accessing private _model_name might be fragile
             except Exception as e:
                  logger.error(f"Failed to apply system instruction to model '{model_name}': {e}. Using base instance.")
                  return base_instance # Fallback to base instance
        else:
             return base_instance


    def get_name(self) -> str: return "gemini"
    def get_available_models(self) -> List[str]:
        # (Remains the same)
        logger.warning("GeminiProvider.get_available_models() returning static list.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        # (Remains the same)
        model_name = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            if model_name.startswith("gemini-1.5"): limit = 1048576
            elif model_name.startswith("gemini-1.0-pro-vision"): limit = 16384
            elif model_name.startswith("gemini-1.0-pro") or model_name == "gemini-pro": limit = 32768
            else: limit = 32768; logger.warning(f"Unknown context length for Gemini model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    def _convert_llmcore_msgs_to_gemini(self, messages: List[Message]) -> Tuple[Optional[str], List[ContentDict]]:
        """Converts LLMCore List[Message] to Gemini format."""
        # (Remains largely the same, operates on LLMCore Message)
        gemini_history: List[ContentDict] = []
        system_prompt: Optional[str] = None
        if messages and messages[0].role == LLMCoreRole.SYSTEM:
            system_prompt = messages[0].content; messages = messages[1:]
        last_role = None
        for msg in messages:
            gemini_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not gemini_role: logger.warning(f"Skipping message with unmappable role for Gemini: {msg.role}"); continue
            if gemini_role == last_role:
                if gemini_history: logger.debug(f"Merging consecutive Gemini '{gemini_role}' messages."); gemini_history[-1]['parts'].append({"text": msg.content}) # type: ignore
                else:
                    if gemini_role == "model": logger.warning("Skipping initial 'model' role message."); continue
                    else: gemini_history.append({"role": gemini_role, "parts": [{"text": msg.content}]}); last_role = gemini_role
            else: gemini_history.append({"role": gemini_role, "parts": [{"text": msg.content}]}); last_role = gemini_role
        if gemini_history and gemini_history[-1]['role'] == 'model': logger.warning("Gemini conversation history ends with 'model' role.")
        return system_prompt, gemini_history

    def _convert_mcp_msgs_to_gemini(self, mcp_messages: List[Any]) -> Tuple[Optional[str], List[ContentDict]]:
        """Converts MCPMessage list to Gemini ContentDict format."""
        gemini_history: List[ContentDict] = []
        system_prompt: Optional[str] = None

        processed_mcp_messages = list(mcp_messages)
        if processed_mcp_messages and processed_mcp_messages[0].role == MCPRole.SYSTEM:
            system_prompt = processed_mcp_messages.pop(0).content
            logger.debug("System prompt extracted for Gemini request from MCP context.")

        last_role = None
        for mcp_msg in processed_mcp_messages:
            gemini_role = MCP_TO_GEMINI_ROLE_MAP.get(mcp_msg.role)
            if not gemini_role: logger.warning(f"Skipping MCP message with unmappable role for Gemini: {mcp_msg.role}"); continue

            if gemini_role == last_role:
                if gemini_history: logger.debug(f"Merging consecutive MCP Gemini '{gemini_role}' messages."); gemini_history[-1]['parts'].append({"text": mcp_msg.content}) # type: ignore
                else:
                    if gemini_role == "model": logger.warning("Skipping initial MCP 'model' role message."); continue
                    else: gemini_history.append({"role": gemini_role, "parts": [{"text": mcp_msg.content}]}); last_role = gemini_role
            else: gemini_history.append({"role": gemini_role, "parts": [{"text": mcp_msg.content}]}); last_role = gemini_role

        if gemini_history and gemini_history[-1]['role'] == 'model': logger.warning("Gemini conversation history (from MCP) ends with 'model' role.")
        return system_prompt, gemini_history

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
        """Sends a chat completion request to the Gemini API."""
        if not genai: raise ProviderError(self.get_name(), "Google Generative AI library not available.")

        model_name = model or self.default_model
        gemini_history: List[ContentDict] = []
        system_prompt: Optional[str] = None
        knowledge_string: Optional[str] = None

        if isinstance(context, list) and all(isinstance(msg, Message) for msg in context):
            logger.debug("Processing context as List[Message] for Gemini.")
            system_prompt, gemini_history = self._convert_llmcore_msgs_to_gemini(context)

        elif mcp_library_available and isinstance(context, MCPContextObject):
            logger.debug("Processing context as MCPContextObject for Gemini.")
            if not mcp_library_available: raise MCPError("MCP library not found at runtime.")

            system_prompt, gemini_history = self._convert_mcp_msgs_to_gemini(context.messages)
            knowledge_string = self._format_mcp_knowledge(context.retrieved_knowledge)

        else:
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}.")

        # Combine original system prompt and knowledge string for system_instruction
        final_system_instruction = system_prompt
        if knowledge_string:
            if final_system_instruction:
                final_system_instruction = f"{final_system_instruction}\n\n{knowledge_string}"
            else:
                final_system_instruction = knowledge_string
            logger.debug("Combined MCP knowledge with system instruction for Gemini.")

        # Get model instance, potentially applying the system instruction
        model_instance = self._get_model_instance(model_name, final_system_instruction)

        logger.debug(f"Sending request to Gemini API: model='{model_name}', stream={stream}, num_history={len(gemini_history)}")

        # Prepare GenerationConfig (remains the same)
        gen_config_args = {"temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"), "top_k": kwargs.get("top_k"), "max_output_tokens": kwargs.get("max_tokens"), "stop_sequences": kwargs.get("stop")}
        gen_config_args = {k: v for k, v in gen_config_args.items() if v is not None}
        generation_config = GenerationConfig(**gen_config_args) if gen_config_args else None

        try:
            response_or_iterator = await model_instance.generate_content_async(
                contents=gemini_history, generation_config=generation_config,
                safety_settings=self.safety_settings, stream=stream,
            )

            if stream:
                logger.debug(f"Processing stream response from Gemini model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    # (Stream processing logic remains the same as before)
                    full_response_text = ""
                    try:
                        async for chunk in response_or_iterator: # type: ignore
                            chunk_text = ""; finish_reason = None
                            try: chunk_text = chunk.text; finish_reason = getattr(chunk, 'candidates', [{}])[0].get('finish_reason', None) if hasattr(chunk, 'candidates') else None
                            except ValueError as e: logger.warning(f"ValueError accessing chunk text: {e}. Chunk: {chunk}"); finish_reason = chunk.prompt_feedback.block_reason if chunk.prompt_feedback else None;
                            except Exception as e: logger.error(f"Unexpected error processing stream chunk: {e}. Chunk: {chunk}", exc_info=True); yield {"error": f"Unexpected stream error: {e}"}; continue
                            if finish_reason == "SAFETY": yield {"error": f"Stream stopped due to safety settings: {finish_reason}", "finish_reason": "SAFETY"}; return
                            full_response_text += chunk_text
                            yield {"model": model_name, "message": {"role": "model", "content": chunk_text}, "choices": [{"delta": {"content": chunk_text}, "index": 0}], "done": False, "finish_reason": finish_reason}
                    except google_exceptions.GoogleAPIError as e: logger.error(f"Gemini API error during stream: {e}", exc_info=True); yield {"error": f"Gemini API Error: {e}", "done": True}; raise ProviderError(self.get_name(), f"API Error during stream: {e}")
                    except Exception as e: logger.error(f"Unexpected error processing Gemini stream: {e}", exc_info=True); yield {"error": f"Unexpected stream processing error: {e}", "done": True}
                    finally: logger.debug("Gemini stream finished.")
                return stream_wrapper()
            else: # Non-streaming
                logger.debug(f"Processing non-stream response from Gemini model '{model_name}'")
                # (Non-stream processing logic remains the same)
                try: full_text = response_or_iterator.text # type: ignore
                except ValueError as e: logger.warning(f"Content blocked in Gemini response: {e}. Response: {response_or_iterator}"); finish_reason = response_or_iterator.prompt_feedback.block_reason if response_or_iterator.prompt_feedback else "BLOCKED"; raise ProviderError(self.get_name(), f"Content generation stopped due to safety settings: {finish_reason}") # type: ignore
                except Exception as e: logger.error(f"Error accessing Gemini response content: {e}. Response: {response_or_iterator}", exc_info=True); raise ProviderError(self.get_name(), f"Failed to extract content from response: {e}")
                usage_metadata = None
                if hasattr(response_or_iterator, 'usage_metadata'): usage_metadata = {"prompt_token_count": response_or_iterator.usage_metadata.prompt_token_count, "candidates_token_count": response_or_iterator.usage_metadata.candidates_token_count, "total_token_count": response_or_iterator.usage_metadata.total_token_count} # type: ignore
                result_dict = {"model": model_name, "choices": [{"index": 0, "message": {"role": "model", "content": full_text}, "finish_reason": getattr(response_or_iterator.candidates[0], 'finish_reason', None) if response_or_iterator.candidates else None}], "usage": usage_metadata, "prompt_feedback": {"block_reason": response_or_iterator.prompt_feedback.block_reason} if response_or_iterator.prompt_feedback.block_reason else None} # type: ignore
                return result_dict

        except google_exceptions.GoogleAPIError as e: logger.error(f"Gemini API error: {e}", exc_info=True); raise ProviderError(self.get_name(), f"API Error: {e}")
        except asyncio.TimeoutError: logger.error(f"Request to Gemini timed out."); raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e: logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True); raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    # --- Token Counting Methods (remain the same) ---
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the Gemini API."""
        # (Remains the same)
        if not genai: return 0;
        if not text: return 0; model_name = model or self.default_model; model_instance = self._get_model_instance(model_name)
        try: response = model_instance.count_tokens(text); return response.total_tokens
        except google_exceptions.GoogleAPIError as e: logger.error(f"Gemini API error during token count for model '{model_name}': {e}", exc_info=True); raise ProviderError(self.get_name(), f"API Error during token count: {e}")
        except Exception as e: logger.error(f"Failed to count tokens for model '{model_name}': {e}", exc_info=True); return (len(text) + 3) // 4

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for List[Message] using the Gemini API."""
        # Note: This method currently only counts tokens for List[Message] format.
        # (Remains the same as before)
        if not genai: return 0;
        if not messages: return 0; model_name = model or self.default_model; model_instance = self._get_model_instance(model_name)
        system_prompt, gemini_history = self._convert_llmcore_msgs_to_gemini(messages); contents_to_count = [];
        if system_prompt: pass # System prompt counted separately below
        contents_to_count.extend(gemini_history)
        try:
            response = model_instance.count_tokens(contents_to_count); total_tokens = response.total_tokens
            if system_prompt: system_token_response = model_instance.count_tokens(system_prompt); total_tokens += system_token_response.total_tokens; total_tokens += 5 # Overhead
            return total_tokens
        except google_exceptions.GoogleAPIError as e: logger.error(f"Gemini API error during message token count for model '{model_name}': {e}", exc_info=True); raise ProviderError(self.get_name(), f"API Error during message token count: {e}")
        except Exception as e: logger.error(f"Failed to count message tokens for model '{model_name}': {e}", exc_info=True); total_text = " ".join([msg.content for msg in messages]); return (len(total_text) + 3 * len(messages)) // 4

    async def close(self) -> None:
        """Closes resources associated with the Gemini provider."""
        # (Remains the same)
        logger.debug("GeminiProvider closed (no specific client cleanup needed).")
        self._model_instance_cache.clear()
        pass
