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

# --- Import google-generativeai within __init__ to handle ImportError gracefully ---

# Conditional MCP imports (keep as is)
if TYPE_CHECKING:
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any; MCPMessage = Any; MCPRole = Any; RetrievedKnowledge = Any; mcp_library_available = False
else:
    # Runtime check for MCP library
    try:
        from modelcontextprotocol import Context as MCPContextObject, Message as MCPMessage, Role as MCPRole, RetrievedKnowledge
        mcp_library_available = True
    except ImportError:
        MCPContextObject = Any; MCPMessage = Any; MCPRole = Any; RetrievedKnowledge = Any; mcp_library_available = False


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError, MCPError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Gemini models
# Source: https://ai.google.dev/models/gemini
# Note: These often represent the *total* input+output limit.
# The effective input limit might be slightly smaller.
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-1.5-pro-latest": 1048576,"gemini-1.5-flash-latest": 1048576,
    "gemini-1.0-pro": 32768,"gemini-pro": 32768, # Alias for 1.0 pro
    "gemini-1.0-pro-vision-latest": 16384,"gemini-pro-vision": 16384, # Alias
}
# Default model if not specified in config
DEFAULT_MODEL = "gemini-1.5-pro-latest"

# Mapping from LLMCore Role to Gemini Role string
LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model",
    # System role handled separately by Gemini API (system_instruction)
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
    Requires the `google-generativeai` library.
    """
    _model_instance_cache: Dict[str, Any] = {}
    # Placeholders for library components imported inside __init__
    _genai: Optional[Any] = None
    _google_exceptions: Optional[Any] = None
    _GenerationConfig: Optional[type] = None
    _HarmCategory: Optional[type] = None
    _HarmBlockThreshold: Optional[type] = None
    _Candidate: Optional[type] = None

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiProvider.

        Performs necessary imports internally to handle potential ImportErrors
        if the required 'google-generativeai' library is not installed.

        Args:
            config: Configuration dictionary containing provider-specific settings like
                    'api_key', 'default_model', 'safety_settings'.

        Raises:
            ImportError: If the 'google-generativeai' library cannot be imported.
            ConfigError: If configuration fails (e.g., during genai.configure).
        """
        # --- Perform imports INSIDE init ---
        try:
            import google.generativeai as genai_lib
            # Import necessary types from the library
            from google.generativeai.types import GenerationConfig as GenConfig_lib, ContentDict as ContentDict_lib, HarmCategory as HarmCategory_lib, HarmBlockThreshold as HarmBlockThreshold_lib, Candidate as Candidate_lib
            try:
                # Import exceptions submodule if available
                from google.api_core import exceptions as google_exceptions_lib
            except ImportError:
                google_exceptions_lib = None # Set to None if exceptions submodule not found

            # Assign to instance attributes for use in other methods
            self._genai = genai_lib
            self._google_exceptions = google_exceptions_lib
            self._GenerationConfig = GenConfig_lib
            self._HarmCategory = HarmCategory_lib
            self._HarmBlockThreshold = HarmBlockThreshold_lib
            self._Candidate = Candidate_lib # Store Candidate type for potential use
            logger.debug("Successfully imported google-generativeai inside GeminiProvider.__init__")
        except ImportError as e:
            logger.error("ImportError inside GeminiProvider.__init__: Google Generative AI library (`google-generativeai`) not found or import failed.", exc_info=False)
            raise ImportError("Google Generative AI library (`google-generativeai`) not installed or failed to import. Install with 'pip install llmcore[gemini]'.") from e
        # --- End imports ---

        # Proceed with configuration now that imports are confirmed successful
        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model') or DEFAULT_MODEL
        self.timeout = config.get('timeout') # Note: timeout not directly used by genai client config
        self.safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key:
            logger.warning("Google AI API key not found in config or GOOGLE_API_KEY env var. SDK will rely on default credentials (e.g., ADC) if available.")

        try:
            # Configure the library globally only if an API key is explicitly provided.
            if self.api_key:
                 self._genai.configure(api_key=self.api_key)
                 logger.debug("Google Generative AI client configured with provided API key.")
            else:
                 logger.debug("Google Generative AI client will use environment variables or default credentials (ADC).")
        except Exception as e:
            # Catch errors during the configure() call
            logger.error(f"Failed during Google Generative AI configuration step: {e}", exc_info=True)
            # Allow initialization to continue; API calls will fail later if auth is truly missing/invalid.

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[Dict[Any, Any]]:
        """Parses safety settings from config string values to enum values."""
        # Use self attributes which were set in __init__
        if not settings_config or not self._HarmCategory or not self._HarmBlockThreshold: return None
        parsed_settings = {}
        for key, value in settings_config.items():
            try:
                category = getattr(self._HarmCategory, key.upper())
                threshold = getattr(self._HarmBlockThreshold, value.upper())
                parsed_settings[category] = threshold
            except AttributeError:
                logger.warning(f"Invalid safety setting category or threshold: {key}={value}. Skipping.")
        logger.debug(f"Parsed safety settings: {parsed_settings}")
        return parsed_settings if parsed_settings else None

    def _get_model_instance(self, model_name: str, system_instruction: Optional[str] = None) -> Any:
        """
        Gets or creates a GenerativeModel instance, applying system instruction if provided.

        Caches base model instances (without system instructions). Returns a potentially
        newly configured instance if a system instruction is provided.

        Args:
            model_name: The name of the Gemini model.
            system_instruction: Optional system instruction text.

        Returns:
            An instance of `google.generativeai.GenerativeModel`.

        Raises:
            ProviderError: If the genai library is not available or model initialization fails.
        """
        cache_key = model_name
        # Ensure genai library was imported successfully in __init__
        if not self._genai:
             raise ProviderError(self.get_name(), "Google Generative AI library failed to import or is not available.")

        # Get or create the base model instance (without system instruction)
        if cache_key not in self._model_instance_cache:
            try:
                model_instance = self._genai.GenerativeModel(model_name)
                self._model_instance_cache[cache_key] = model_instance
                logger.debug(f"Created and cached GenerativeModel instance for '{model_name}'")
            except Exception as e:
                logger.error(f"Failed to initialize base GenerativeModel for '{model_name}': {e}", exc_info=True)
                raise ProviderError(self.get_name(), f"Failed to get model instance '{model_name}': {e}")

        base_instance = self._model_instance_cache[cache_key]

        # If a system instruction is provided, create a new instance configured with it
        if system_instruction:
             try:
                 logger.debug(f"Creating temporary GenerativeModel instance for '{model_name}' with system instruction.")
                 # Re-create with system instruction using the model_name.
                 return self._genai.GenerativeModel(model_name, system_instruction=system_instruction)
             except Exception as e:
                  # Log the error but return the base instance as a fallback
                  logger.error(f"Failed to apply system instruction to model '{model_name}': {e}. Using base instance without system instruction.")
                  return base_instance
        else:
             # Return the cached base instance if no system instruction
             return base_instance


    def get_name(self) -> str:
        """Returns the provider name: 'gemini'."""
        return "gemini"

    def get_available_models(self) -> List[str]:
        """Returns a static list of known default models for Gemini."""
        logger.warning("GeminiProvider.get_available_models() returning static list.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Returns the estimated maximum context length (tokens) for the given Gemini model.
        Uses a hardcoded map and falls back to a default if the model is unknown.
        """
        model_name = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Attempt to infer from common prefixes
            if model_name.startswith("gemini-1.5"): limit = 1048576
            elif model_name.startswith("gemini-1.0-pro-vision"): limit = 16384
            elif model_name.startswith("gemini-1.0-pro") or model_name == "gemini-pro": limit = 32768
            else:
                limit = 32768 # Fallback default (for 1.0 pro)
                logger.warning(f"Unknown context length for Gemini model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    # _convert methods remain the same
    def _convert_llmcore_msgs_to_gemini(self, messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Converts LLMCore List[Message] to Gemini's ContentDict history format."""
        gemini_history: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        if messages and messages[0].role == LLMCoreRole.SYSTEM: system_prompt = messages[0].content; messages = messages[1:]
        last_role = None
        for msg in messages:
            gemini_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not gemini_role: logger.warning(f"Skipping message with unmappable role for Gemini: {msg.role}"); continue
            if gemini_role == last_role:
                if gemini_history: logger.debug(f"Merging consecutive Gemini '{gemini_role}' messages."); gemini_history[-1]['parts'].append({"text": msg.content})
                else:
                    if gemini_role == "model": logger.warning("Skipping initial 'model' role message."); continue
                    else: gemini_history.append({"role": gemini_role, "parts": [{"text": msg.content}]}); last_role = gemini_role
            else: gemini_history.append({"role": gemini_role, "parts": [{"text": msg.content}]}); last_role = gemini_role
        if gemini_history and gemini_history[-1]['role'] == 'model': logger.warning("Gemini conversation history ends with 'model' role.")
        return system_prompt, gemini_history

    def _convert_mcp_msgs_to_gemini(self, mcp_messages: List[Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Converts MCPMessage list to Gemini ContentDict history format."""
        gemini_history: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        processed_mcp_messages = list(mcp_messages)
        if processed_mcp_messages and processed_mcp_messages[0].role == MCPRole.SYSTEM: system_prompt = processed_mcp_messages.pop(0).content; logger.debug("System prompt extracted for Gemini request from MCP context.")
        last_role = None
        for mcp_msg in processed_mcp_messages:
            gemini_role = MCP_TO_GEMINI_ROLE_MAP.get(mcp_msg.role)
            if not gemini_role: logger.warning(f"Skipping MCP message with unmappable role for Gemini: {mcp_msg.role}"); continue
            if gemini_role == last_role:
                if gemini_history: logger.debug(f"Merging consecutive MCP Gemini '{gemini_role}' messages."); gemini_history[-1]['parts'].append({"text": mcp_msg.content})
                else:
                    if gemini_role == "model": logger.warning("Skipping initial MCP 'model' role message."); continue
                    else: gemini_history.append({"role": gemini_role, "parts": [{"text": mcp_msg.content}]}); last_role = gemini_role
            else: gemini_history.append({"role": gemini_role, "parts": [{"text": mcp_msg.content}]}); last_role = gemini_role
        if gemini_history and gemini_history[-1]['role'] == 'model': logger.warning("Gemini conversation history (from MCP) ends with 'model' role.")
        return system_prompt, gemini_history

    def _format_mcp_knowledge(self, knowledge: Optional[List[Any]]) -> Optional[str]:
        """Formats MCP RetrievedKnowledge into a string for system prompt."""
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
        """
        Sends a chat completion request to the Gemini API.

        Handles context conversion (List[Message] or MCP), system instructions,
        safety settings, and streaming.

        Args:
            context: The context payload (List[Message] or MCPContextObject).
            model: The specific Gemini model name to use. Defaults to provider default.
            stream: Whether to stream the response.
            **kwargs: Additional parameters for the Gemini API (e.g., temperature, max_tokens).

        Returns:
            A dictionary for non-streaming responses or an async generator of
            dictionaries for streaming responses.

        Raises:
            ProviderError: If the API call fails or the library is unavailable.
            ConfigError: If configuration is invalid.
            MCPError: If MCP processing fails.
        """
        # Use self._genai etc. which were set in __init__
        if not self._genai: raise ProviderError(self.get_name(), "Google Generative AI library not available.")

        model_name = model or self.default_model
        gemini_history: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        knowledge_string: Optional[str] = None

        # Process context based on type
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

        # Combine system prompt and knowledge for the final system instruction
        final_system_instruction = system_prompt
        if knowledge_string:
            if final_system_instruction: final_system_instruction = f"{final_system_instruction}\n\n{knowledge_string}"
            else: final_system_instruction = knowledge_string
            logger.debug("Combined MCP knowledge with system instruction for Gemini.")

        # Get model instance, potentially applying the system instruction
        model_instance = self._get_model_instance(model_name, final_system_instruction)

        logger.debug(f"Sending request to Gemini API: model='{model_name}', stream={stream}, num_history={len(gemini_history)}")

        # Prepare GenerationConfig using the imported type alias
        gen_config_args = {"temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"), "top_k": kwargs.get("top_k"), "max_output_tokens": kwargs.get("max_tokens"), "stop_sequences": kwargs.get("stop")}
        gen_config_args = {k: v for k, v in gen_config_args.items() if v is not None}
        generation_config = self._GenerationConfig(**gen_config_args) if self._GenerationConfig and gen_config_args else None

        try:
            # Check for empty history case again before API call
            if not gemini_history and not final_system_instruction:
                 raise ProviderError(self.get_name(), "Cannot send request with empty history and no system instruction.")

            # Make the API call
            response_or_iterator = await model_instance.generate_content_async(
                contents=gemini_history, generation_config=generation_config,
                safety_settings=self.safety_settings, stream=stream,
            )

            # Process stream or full response
            if stream:
                logger.debug(f"Processing stream response from Gemini model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = ""
                    try:
                        async for chunk in response_or_iterator: # type: ignore
                            chunk_text = ""; finish_reason = None; block_reason = None
                            try:
                                chunk_text = chunk.text
                                if chunk.candidates:
                                     raw_finish_reason = chunk.candidates[0].finish_reason
                                     finish_reason = raw_finish_reason.name if raw_finish_reason else None
                            except ValueError as e:
                                logger.warning(f"ValueError accessing chunk text (likely blocked content): {e}. Chunk: {chunk}")
                                if chunk.prompt_feedback:
                                     raw_block_reason = chunk.prompt_feedback.block_reason
                                     block_reason = raw_block_reason.name if raw_block_reason else None
                                if block_reason: finish_reason = "SAFETY" # Treat block as safety finish
                            except AttributeError as e:
                                logger.error(f"AttributeError processing stream chunk: {e}. Chunk: {chunk}", exc_info=False)
                                yield {"error": f"AttributeError processing stream chunk: {e}"}
                                continue # Skip malformed chunk
                            except Exception as e:
                                logger.error(f"Unexpected error processing stream chunk: {e}. Chunk: {chunk}", exc_info=True)
                                yield {"error": f"Unexpected stream error: {e}"}
                                continue # Skip malformed chunk

                            # Check for safety finish reason
                            if finish_reason == "SAFETY":
                                logger.warning(f"Stream stopped due to safety settings. Reason: {finish_reason or block_reason}")
                                yield {"error": f"Stream stopped due to safety settings: {finish_reason or block_reason}", "finish_reason": "SAFETY"}
                                return # Stop the stream

                            full_response_text += chunk_text
                            # Yield a dictionary mimicking other providers' stream format
                            yield {"model": model_name, "message": {"role": "model", "content": chunk_text}, "choices": [{"delta": {"content": chunk_text}, "index": 0}], "done": False, "finish_reason": finish_reason}
                    # Catch specific Google API errors if possible, otherwise general Exception
                    except Exception as e:
                         if self._google_exceptions and isinstance(e, self._google_exceptions.GoogleAPIError):
                             logger.error(f"Gemini API error during stream: {e}", exc_info=True)
                             yield {"error": f"Gemini API Error: {e}", "done": True}
                         else: # Catch general exceptions during stream processing
                             logger.error(f"Unexpected error processing Gemini stream: {e}", exc_info=True)
                             yield {"error": f"Unexpected stream processing error: {e}", "done": True}
                    finally:
                        logger.debug("Gemini stream finished.")
                return stream_wrapper()
            else: # Non-streaming response
                logger.debug(f"Processing non-stream response from Gemini model '{model_name}'")
                try:
                    # Access response text, handle potential blocking error
                    full_text = response_or_iterator.text # type: ignore
                except ValueError as e:
                    logger.warning(f"Content blocked in Gemini response: {e}.")
                    block_reason_enum = response_or_iterator.prompt_feedback.block_reason if response_or_iterator.prompt_feedback else None # type: ignore
                    finish_reason = block_reason_enum.name if block_reason_enum else "BLOCKED"
                    raise ProviderError(self.get_name(), f"Content generation stopped due to safety settings: {finish_reason}")
                except Exception as e:
                    logger.error(f"Error accessing Gemini response content: {e}.", exc_info=True)
                    raise ProviderError(self.get_name(), f"Failed to extract content from response: {e}")

                # Extract usage and finish reason
                usage_metadata = None
                if hasattr(response_or_iterator, 'usage_metadata'):
                     usage_metadata = {
                          "prompt_token_count": response_or_iterator.usage_metadata.prompt_token_count, # type: ignore
                          "candidates_token_count": response_or_iterator.usage_metadata.candidates_token_count, # type: ignore
                          "total_token_count": response_or_iterator.usage_metadata.total_token_count, # type: ignore
                     }
                resp_finish_reason = None
                if response_or_iterator.candidates: # type: ignore
                    raw_finish_reason = response_or_iterator.candidates[0].finish_reason # type: ignore
                    resp_finish_reason = raw_finish_reason.name if raw_finish_reason else None

                block_reason = None
                if response_or_iterator.prompt_feedback and response_or_iterator.prompt_feedback.block_reason: # type: ignore
                    block_reason = response_or_iterator.prompt_feedback.block_reason.name # type: ignore

                # Construct result dictionary similar to OpenAI format
                result_dict = {
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "model", "content": full_text },
                        "finish_reason": resp_finish_reason
                    }],
                    "usage": usage_metadata,
                    "prompt_feedback": {"block_reason": block_reason}
                }
                return result_dict

        # Catch API errors from the initial generate_content_async call
        except Exception as e:
            # Check if it's a Google API error if the exceptions module was loaded
            if self._google_exceptions and isinstance(e, self._google_exceptions.GoogleAPIError):
                 logger.error(f"Gemini API error: {e}", exc_info=True)
                 raise ProviderError(self.get_name(), f"API Error: {e}")
            elif isinstance(e, asyncio.TimeoutError): # Catch timeout specifically
                 logger.error(f"Request to Gemini timed out.")
                 raise ProviderError(self.get_name(), f"Request timed out.")
            else: # Catch other general exceptions
                 logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
                 raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")


    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a single string using the Gemini API."""
        if not self._genai: return 0 # Library not available
        if not text: return 0
        model_name = model or self.default_model
        try:
            # Get model instance (without system prompt for counting)
            model_instance = self._get_model_instance(model_name)
            response = model_instance.count_tokens(text)
            return response.total_tokens
        # Catch specific Google API errors if possible, otherwise general Exception
        except Exception as e:
            # Check if it's a Google API error if the exceptions module was loaded
            if self._google_exceptions and isinstance(e, self._google_exceptions.GoogleAPIError):
                logger.error(f"Gemini API error during token count for model '{model_name}': {e}", exc_info=True)
                raise ProviderError(self.get_name(), f"API Error during token count: {e}")
            else: # Fallback for general exceptions
                logger.error(f"Failed to count tokens for model '{model_name}': {e}", exc_info=True)
                return (len(text) + 3) // 4 # Fallback approximation

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of LLMCore Messages using the Gemini API."""
        if not self._genai: return 0 # Library not available
        if not messages: return 0
        model_name = model or self.default_model
        total_tokens = 0
        try:
            # Get model instance (without system prompt for counting)
            model_instance = self._get_model_instance(model_name)
            system_prompt, gemini_history = self._convert_llmcore_msgs_to_gemini(messages)

            # Count history tokens only if history is not empty
            if gemini_history:
                response = model_instance.count_tokens(gemini_history)
                total_tokens += response.total_tokens

            # Count system prompt tokens separately if it exists
            if system_prompt:
                 system_response = model_instance.count_tokens(system_prompt)
                 total_tokens += system_response.total_tokens
                 # Add a small heuristic overhead for system prompt integration if needed
                 total_tokens += 5

            # Handle the case where input messages resulted in no content to count
            if not gemini_history and not system_prompt:
                 logger.warning("count_message_tokens called with messages resulting in empty history/system prompt.")
                 return 0

            return total_tokens
        # Catch specific Google API errors if possible, otherwise general Exception
        except Exception as e:
            # Check if it's a Google API error if the exceptions module was loaded
            if self._google_exceptions and isinstance(e, self._google_exceptions.GoogleAPIError):
                # Specifically handle "contents is not specified" which happens with empty history
                if "contents is not specified" in str(e):
                     logger.warning(f"Gemini count_tokens API error (likely empty history) for model '{model_name}'.")
                     # If only system prompt existed, return its count + overhead
                     if system_prompt:
                          try:
                               model_instance_fallback = self._get_model_instance(model_name) # Re-get instance
                               system_response = model_instance_fallback.count_tokens(system_prompt)
                               return system_response.total_tokens + 5
                          except Exception as inner_e: # Fallback if counting system prompt also fails
                               logger.error(f"Failed to count system prompt tokens as fallback: {inner_e}")
                               return (len(system_prompt) + 3) // 4 + 5
                     else:
                          return 0 # No history, no system prompt
                else:
                     # Log other API errors
                     logger.error(f"Gemini API error during message token count for model '{model_name}': {e}", exc_info=True)
                     raise ProviderError(self.get_name(), f"API Error during message token count: {e}")
            else: # Fallback for general exceptions
                 logger.error(f"Failed to count message tokens for model '{model_name}': {e}", exc_info=True)
                 # Fallback approximation
                 total_text = " ".join([msg.content for msg in messages if msg.role != LLMCoreRole.SYSTEM]) # Exclude system prompt from simple approx
                 sys_len = len(system_prompt) if system_prompt else 0
                 return (len(total_text) + sys_len + 3 * len(messages)) // 4

    async def close(self) -> None:
        """Closes resources associated with the Gemini provider (clears cache)."""
        logger.debug("GeminiProvider closed (cleared model instance cache).")
        self._model_instance_cache.clear()
        # No explicit client closing needed for the genai library itself
        pass
