# src/llmcore/providers/gemini_provider.py
"""
Google Gemini API provider implementation for the LLMCore library.

Handles interactions with the Google Generative AI API (Gemini models).
Uses the official 'google-genai' library (v0.8.0+).
Can accept context as List[Message] or MCPContextObject.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple, TYPE_CHECKING

# --- Use the new google-genai library ---
try:
    import google.genai as genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors # Import the errors module
    # Specific error types that might be useful
    from google.generativeai.types import StopCandidateException # For blocked content in stream
    from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError # Base for some API errors
    google_genai_available = True
except ImportError:
    google_genai_available = False
    genai = None # type: ignore
    genai_types = None # type: ignore
    genai_errors = None # type: ignore
    StopCandidateException = Exception # type: ignore
    CoreGoogleAPIError = Exception # type: ignore
# --- End new library import ---

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
from ..exceptions import ProviderError, ConfigError, MCPError, ContextLengthError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Gemini models (remains the same)
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-1.5-pro-latest": 1048576, "gemini-1.5-flash-latest": 1048576,
    "gemini-1.0-pro": 32768, "gemini-pro": 32768, # Alias for 1.0 pro
    "gemini-1.0-pro-vision-latest": 16384, "gemini-pro-vision": 16384, # Alias
    "gemini-2.0-flash-lite": 1048576,
}
# Default model if not specified in config
DEFAULT_MODEL = "gemini-2.0-flash-lite"

# Mapping from LLMCore Role to Gemini Role string (remains the same)
LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model",
    # System role handled separately by Gemini API (system_instruction)
}
# Mapping from MCP Role Enum to Gemini Role string (remains the same)
MCP_TO_GEMINI_ROLE_MAP: Dict[Any, str] = {}
if mcp_library_available:
    MCP_TO_GEMINI_ROLE_MAP = {
        MCPRole.USER: "user",
        MCPRole.ASSISTANT: "model",
        # MCPRole.SYSTEM handled separately
    }


class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API using google-genai.
    Handles both List[Message] and MCPContextObject context types.
    Requires the `google-genai` library.
    """
    _client: Optional[genai.Client] = None # Use the new client type
    _safety_settings: Optional[List[genai_types.SafetySettingDict]] = None # Use new types

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiProvider using the google-genai SDK.

        Args:
            config: Configuration dictionary containing provider-specific settings like
                    'api_key', 'default_model', 'safety_settings'.

        Raises:
            ImportError: If the 'google-genai' library cannot be imported.
            ConfigError: If configuration fails (e.g., during client creation).
        """
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed or failed to import. Install with 'pip install llmcore[gemini]'.")

        # Configuration remains similar
        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model') or DEFAULT_MODEL
        self.timeout = config.get('timeout') # Timeout might be handled differently
        self._safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        # --- Initialize the new genai.Client ---
        try:
            # The client can be initialized without an API key if GOOGLE_API_KEY env var is set
            # or if using Vertex AI credentials.
            client_options = {}
            if self.api_key:
                client_options['api_key'] = self.api_key
            # Add http_options if needed (e.g., for timeout, although direct timeout isn't obvious)
            # http_options = genai_types.HttpOptions(timeout=self.timeout) if self.timeout else None
            # if http_options: client_options['http_options'] = http_options

            self._client = genai.Client(**client_options)
            logger.info("Google Gen AI client initialized successfully.")
            # Optional: Test connection or list models if needed
            # models = self._client.models.list() # Example check
        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client: {e}", exc_info=True)
            raise ConfigError(f"Google Gen AI client initialization failed: {e}")
        # --- End client initialization ---

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[List[genai_types.SafetySettingDict]]:
        """Parses safety settings from config string values to genai_types format."""
        # (Parsing logic remains the same)
        if not settings_config or not genai_types: return None
        parsed_settings: List[genai_types.SafetySettingDict] = []
        for key_str, value_str in settings_config.items():
            try:
                # Convert string keys/values to the enums expected by the SDK
                # The SDK uses strings directly now for categories and thresholds in SafetySettingDict
                # Example: {'category': 'HARM_CATEGORY_SEXUAL', 'threshold': 'BLOCK_LOW_AND_ABOVE'}
                # Validate against known categories/thresholds if possible, or pass strings directly.
                # For simplicity, we pass strings directly, assuming they match SDK expectations.
                category_upper = key_str.upper()
                threshold_upper = value_str.upper()
                # Basic validation (can be expanded)
                if not category_upper.startswith("HARM_CATEGORY_"): raise ValueError("Invalid category format")
                if not threshold_upper.startswith("BLOCK_"): raise ValueError("Invalid threshold format")

                setting: genai_types.SafetySettingDict = {
                    'category': category_upper, # type: ignore # Expects HarmCategory literal
                    'threshold': threshold_upper # type: ignore # Expects HarmBlockThreshold literal
                }
                parsed_settings.append(setting)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Invalid safety setting format: {key_str}={value_str}. Skipping. Error: {e}")
        logger.debug(f"Parsed safety settings: {parsed_settings}")
        return parsed_settings if parsed_settings else None

    def get_name(self) -> str:
        """Returns the provider name: 'gemini'."""
        return "gemini"

    def get_available_models(self) -> List[str]:
        """Returns a static list of known default models for Gemini."""
        # TODO: Potentially implement dynamic listing via self._client.models.list()
        logger.warning("GeminiProvider.get_available_models() returning static list.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the estimated maximum context length (tokens) for the given Gemini model."""
        # TODO: Potentially implement dynamic lookup via self._client.models.get(model).input_token_limit
        # (Logic remains the same)
        model_name = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            if model_name.startswith("gemini-1.5"): limit = 1048576
            elif model_name.startswith("gemini-1.0-pro-vision"): limit = 16384
            elif model_name.startswith("gemini-1.0-pro") or model_name == "gemini-pro": limit = 32768
            else: limit = 32768; logger.warning(f"Unknown context length for Gemini model '{model_name}'. Using fallback limit: {limit}.")
        return limit

    # --- Context Conversion Helpers ---
    def _convert_llmcore_msgs_to_genai_contents(
        self,
        messages: List[Message]
    ) -> Tuple[Optional[str], List[genai_types.ContentDict]]:
        """Converts LLMCore List[Message] to Gemini's ContentDict list and extracts system instruction text."""
        if not genai_types: raise ProviderError(self.get_name(), "google-genai types not available.")

        genai_history: List[genai_types.ContentDict] = []
        system_instruction_text: Optional[str] = None # Store as text

        processed_messages = list(messages)
        # Extract system message first
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_instruction_text = processed_messages[0].content # Store text
            processed_messages = processed_messages[1:]
            logger.debug("System instruction text extracted for Gemini request from List[Message].")

        last_role = None
        for msg in processed_messages:
            genai_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not genai_role:
                logger.warning(f"Skipping message with unmappable role for Gemini: {msg.role}")
                continue

            # Handle consecutive roles by merging
            if genai_role == last_role:
                if genai_history:
                    logger.debug(f"Merging consecutive Gemini '{genai_role}' messages.")
                    # Append text to the parts list of the last message
                    # Ensure 'parts' exists and is a list
                    if isinstance(genai_history[-1].get('parts'), list):
                        genai_history[-1]['parts'].append(genai_types.PartDict(text=msg.content))
                    else: # Should not happen if constructed correctly, but handle defensively
                        genai_history[-1]['parts'] = [genai_types.PartDict(text=msg.content)]
                else:
                    # Cannot start with 'model' role after system instruction
                    if genai_role == "model":
                        logger.warning("Skipping initial 'model' role message after system instruction.")
                        continue
                    else: # First message must be user
                        genai_history.append(genai_types.ContentDict(role=genai_role, parts=[genai_types.PartDict(text=msg.content)]))
                        last_role = genai_role
            else: # Role is different, add new ContentDict
                genai_history.append(genai_types.ContentDict(role=genai_role, parts=[genai_types.PartDict(text=msg.content)]))
                last_role = genai_role

        # Final validation: Must end with 'user' role for generate_content
        if genai_history and genai_history[-1]['role'] == 'model':
            logger.warning("Gemini conversation history ends with 'model' role. API might require 'user' role last.")
            # Optionally append a dummy user message if needed, or let API handle it.

        return system_instruction_text, genai_history # Return text and list

    def _convert_mcp_msgs_to_genai_contents(
        self,
        mcp_messages: List[Any] # List[MCPMessage]
    ) -> Tuple[Optional[str], List[genai_types.ContentDict]]:
        """Converts MCPMessage list to Gemini ContentDict list and extracts system instruction text."""
        if not genai_types: raise ProviderError(self.get_name(), "google-genai types not available.")

        genai_history: List[genai_types.ContentDict] = []
        system_instruction_text: Optional[str] = None # Store as text

        processed_mcp_messages = list(mcp_messages)
        # Extract system message first
        if processed_mcp_messages and processed_mcp_messages[0].role == MCPRole.SYSTEM:
            system_instruction_text = processed_mcp_messages.pop(0).content # Store text
            logger.debug("System instruction text extracted for Gemini request from MCP context.")

        last_role = None
        for mcp_msg in processed_mcp_messages:
            genai_role = MCP_TO_GEMINI_ROLE_MAP.get(mcp_msg.role)
            if not genai_role:
                logger.warning(f"Skipping MCP message with unmappable role for Gemini: {mcp_msg.role}")
                continue

            # Handle consecutive roles by merging
            if genai_role == last_role:
                if genai_history:
                    logger.debug(f"Merging consecutive MCP Gemini '{genai_role}' messages.")
                    if isinstance(genai_history[-1].get('parts'), list):
                        genai_history[-1]['parts'].append(genai_types.PartDict(text=mcp_msg.content))
                    else:
                        genai_history[-1]['parts'] = [genai_types.PartDict(text=mcp_msg.content)]
                else:
                    if genai_role == "model": logger.warning("Skipping initial MCP 'model' role message."); continue
                    else: genai_history.append(genai_types.ContentDict(role=genai_role, parts=[genai_types.PartDict(text=mcp_msg.content)])); last_role = genai_role
            else: # Role is different
                genai_history.append(genai_types.ContentDict(role=genai_role, parts=[genai_types.PartDict(text=mcp_msg.content)]))
                last_role = genai_role

        if genai_history and genai_history[-1]['role'] == 'model':
            logger.warning("Gemini conversation history (from MCP) ends with 'model' role.")

        return system_instruction_text, genai_history # Return text and list

    def _format_mcp_knowledge(self, knowledge: Optional[List[Any]]) -> Optional[str]:
        """Formats MCP RetrievedKnowledge into a string for system instruction."""
        # (Helper function remains the same)
        if not knowledge: return None
        parts = ["--- Retrieved Context ---"]
        for item in knowledge:
            source_info = "Unknown Source";
            if item.source_metadata: source_info = item.source_metadata.get("source", item.source_metadata.get("doc_id", "Unknown Source"))
            content_snippet = item.content.replace('\n', ' ').strip(); parts.append(f"\n[Source: {source_info}]\n{content_snippet}")
        parts.append("--- End Context ---")
        return "\n".join(parts)
    # --- End Context Conversion Helpers ---

    async def chat_completion(
        self,
        context: ContextPayload, # Accepts List[Message] or MCPContextObject
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the Gemini API using google-genai.

        Handles context conversion, system instructions, safety settings, and streaming.
        """
        if not self._client or not genai_types or not genai_errors:
            raise ProviderError(self.get_name(), "Google Gen AI library or types not available.")

        model_name = model or self.default_model
        genai_contents: List[genai_types.ContentDict] = []
        system_instruction_text: Optional[str] = None
        knowledge_string: Optional[str] = None

        # Process context based on type
        if isinstance(context, list) and all(isinstance(msg, Message) for msg in context):
            logger.debug("Processing context as List[Message] for Gemini.")
            system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)
        elif mcp_library_available and isinstance(context, MCPContextObject):
            logger.debug("Processing context as MCPContextObject for Gemini.")
            if not mcp_library_available: raise MCPError("MCP library not found at runtime.")
            system_instruction_text, genai_contents = self._convert_mcp_msgs_to_genai_contents(context.messages)
            knowledge_string = self._format_mcp_knowledge(context.retrieved_knowledge)
        else:
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}.")

        # Combine system instruction text and knowledge string
        final_system_instruction_text: Optional[str] = None
        if system_instruction_text:
            final_system_instruction_text = system_instruction_text
        if knowledge_string:
            if final_system_instruction_text:
                final_system_instruction_text = f"{final_system_instruction_text}\n\n{knowledge_string}"
            else:
                final_system_instruction_text = knowledge_string
            logger.debug("Combined MCP knowledge with system instruction for Gemini.")

        # Prepare GenerationConfig (remains the same)
        gen_config_args = {
            "temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"), "max_output_tokens": kwargs.get("max_tokens"),
            "stop_sequences": kwargs.get("stop_sequences"), "candidate_count": kwargs.get("candidate_count", 1),
        }
        gen_config_args_filtered = {k: v for k, v in gen_config_args.items() if v is not None}
        generation_config: Optional[genai_types.GenerationConfigDict] = None
        if gen_config_args_filtered:
            try: generation_config = genai_types.GenerationConfigDict(**gen_config_args_filtered) # type: ignore
            except TypeError as te: logger.warning(f"Invalid argument for GenerationConfig: {te}. Using default.")

        # --- Prepare contents for API call ---
        # Prepend system instruction text to the contents list if it exists.
        api_contents = list(genai_contents) # Make a copy
        if final_system_instruction_text:
             # The new SDK might support a 'system' role in contents, or use system_instruction param
             # Let's try prepending a 'system' role message if supported by the types
             # UPDATE: Based on testing, system_instruction kwarg is invalid. Prepending is needed.
             logger.debug("Prepending system instruction to contents list for Gemini API call.")
             system_content = genai_types.ContentDict(
                 # Role 'system' might not be valid for all models/API versions.
                 # Using 'user' then 'model' with the instruction might be a safer alternative
                 # if 'system' role causes issues. For now, trying 'system'.
                 role="system", # This role might need verification depending on model support
                 parts=[genai_types.PartDict(text=final_system_instruction_text)]
             )
             # Check if the first message is already a system message (e.g., from MCP conversion)
             if api_contents and api_contents[0].get('role') == 'system':
                 logger.warning("Found existing system message in contents, overwriting with combined instruction.")
                 api_contents[0] = system_content
             else:
                 api_contents.insert(0, system_content)


        logger.debug(f"Sending request to Gemini API: model='{model_name}', stream={stream}, num_contents={len(api_contents)}")

        try:
            # Make the API call using the async client
            response_iterator = await self._client.aio.models.generate_content(
                model=f"models/{model_name}", # Model name often needs 'models/' prefix
                contents=api_contents, # Pass the potentially modified contents list
                # system_instruction=final_system_instruction_text, # REMOVED - Causes TypeError
                generation_config=generation_config,
                safety_settings=self._safety_settings,
                stream=stream,
            )

            # Process stream or full response (Stream wrapper logic remains the same)
            if stream:
                logger.debug(f"Processing stream response from Gemini model '{model_name}'")
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    # (Stream processing logic - unchanged from previous correction)
                    full_response_text = ""
                    try:
                        async for chunk in response_iterator:
                            chunk_text = ""; finish_reason = None; block_reason = None
                            try:
                                chunk_text = chunk.text
                                if chunk.candidates: raw_finish_reason = chunk.candidates[0].finish_reason; finish_reason = raw_finish_reason.name if raw_finish_reason else None
                            except ValueError as e: # Handles blocked content error
                                logger.warning(f"ValueError accessing chunk text (likely blocked content): {e}. Chunk: {chunk}")
                                if chunk.prompt_feedback: raw_block_reason = chunk.prompt_feedback.block_reason; block_reason = raw_block_reason.name if raw_block_reason else None
                                finish_reason = "SAFETY" if block_reason else "ERROR"
                            # --- Updated: Catch StopCandidateException for blocked stream content ---
                            except StopCandidateException as sce:
                                logger.warning(f"Content blocked during stream generation: {sce}")
                                finish_reason = "SAFETY" # Treat as safety stop
                                chunk_text = "" # No text delta if blocked
                            # --- End Update ---
                            except Exception as e: logger.error(f"Unexpected error processing stream chunk: {e}. Chunk: {chunk}", exc_info=True); yield {"error": f"Unexpected stream error: {e}"}; continue

                            if finish_reason == "SAFETY": logger.warning(f"Stream stopped due to safety settings. Reason: {finish_reason or block_reason}"); yield {"error": f"Stream stopped due to safety settings: {finish_reason or block_reason}", "finish_reason": "SAFETY"}; return
                            full_response_text += chunk_text
                            yield {"model": model_name, "message": {"role": "model", "content": chunk_text}, "choices": [{"delta": {"content": chunk_text}, "index": 0}], "done": False, "finish_reason": finish_reason}
                    # --- Updated Error Catching ---
                    except genai_errors.GenerativeAIError as e: # Catch base SDK error
                        logger.error(f"Gemini API error during stream: {e}", exc_info=True)
                        yield {"error": f"Gemini API Error: {e}", "done": True}
                    # --- End Update ---
                    except Exception as e: logger.error(f"Unexpected error processing Gemini stream: {e}", exc_info=True); yield {"error": f"Unexpected stream processing error: {e}", "done": True}
                    finally: logger.debug("Gemini stream finished.")
                return stream_wrapper()
            else: # Non-streaming response
                logger.debug(f"Processing non-stream response from Gemini model '{model_name}'")
                response: genai_types.GenerateContentResponse = response_iterator # type: ignore
                # (Response processing logic - unchanged from previous correction)
                try: full_text = response.text
                except ValueError as e: logger.warning(f"Content blocked in Gemini response: {e}."); block_reason_enum = response.prompt_feedback.block_reason if response.prompt_feedback else None; finish_reason = block_reason_enum.name if block_reason_enum else "BLOCKED"; raise ProviderError(self.get_name(), f"Content generation stopped due to safety settings: {finish_reason}")
                except Exception as e: logger.error(f"Error accessing Gemini response content: {e}.", exc_info=True); raise ProviderError(self.get_name(), f"Failed to extract content from response: {e}")
                usage_metadata = None;
                if hasattr(response, 'usage_metadata') and response.usage_metadata: usage_metadata = {"prompt_token_count": response.usage_metadata.prompt_token_count, "candidates_token_count": response.usage_metadata.candidates_token_count, "total_token_count": response.usage_metadata.total_token_count}
                resp_finish_reason = None;
                if response.candidates: raw_finish_reason = response.candidates[0].finish_reason; resp_finish_reason = raw_finish_reason.name if raw_finish_reason else None
                block_reason = None;
                if response.prompt_feedback and response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason.name
                result_dict = {"model": model_name, "choices": [{"index": 0, "message": { "role": "model", "content": full_text }, "finish_reason": resp_finish_reason}], "usage": usage_metadata, "prompt_feedback": {"block_reason": block_reason}}
                return result_dict

        # --- Updated Error Catching ---
        except genai_errors.GenerativeAIError as e: # Catch base SDK error
            logger.error(f"Gemini API error: {e}", exc_info=True)
            # Check specific types if needed (e.g., PermissionDenied, InvalidArgument)
            # Use isinstance checks for better reliability
            if isinstance(e, genai_errors.PermissionDenied): raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied: {e}")
            # Check for context length error specifically
            if isinstance(e, genai_errors.InvalidArgumentError) and ("context length" in str(e).lower() or "token limit" in str(e).lower()):
                 actual_tokens = await self.count_message_tokens(context, model_name) if isinstance(context, list) else 0
                 limit = self.get_max_context_length(model_name)
                 raise ContextLengthError(model_name=model_name, limit=limit, actual=actual_tokens, message=f"Context length error: {e}")
            # Catch potential StopCandidateException here as well if it bubbles up
            if isinstance(e, StopCandidateException):
                logger.warning(f"Content generation stopped due to safety or other candidate issue: {e}")
                raise ProviderError(self.get_name(), f"Content generation stopped: {e}")
            # Catch other Google API core errors
            if isinstance(e, CoreGoogleAPIError):
                 logger.error(f"Google Core API error during Gemini call: {e}", exc_info=True)
                 raise ProviderError(self.get_name(), f"Google Core API Error: {e}")

            # Fallback for other GenerativeAIError types
            raise ProviderError(self.get_name(), f"API Error: {e}")
        # --- End Update ---
        except asyncio.TimeoutError: logger.error(f"Request to Gemini timed out."); raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e: logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True); raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")


    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a single string using the Gemini API via google-genai."""
        # (Error handling updated)
        if not self._client or not genai_errors: logger.warning("Gemini client/errors not available for token counting. Returning 0."); return 0
        if not text: return 0
        model_name = model or self.default_model
        try:
            response = await self._client.aio.models.count_tokens(model=f"models/{model_name}", contents=[text])
            return response.total_tokens
        except genai_errors.GenerativeAIError as e: # Catch base SDK error
            logger.error(f"Gemini API error during token count for model '{model_name}': {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e}")
        except Exception as e: logger.error(f"Failed to count tokens for model '{model_name}': {e}", exc_info=True); return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of LLMCore Messages using the Gemini API via google-genai."""
        # (Error handling updated, system prompt handling adjusted)
        if not self._client or not genai_errors: logger.warning("Gemini client/errors not available for message token counting. Returning 0."); return 0
        if not messages: return 0
        model_name = model or self.default_model
        try:
            system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(messages)
            contents_to_count: List[Union[str, genai_types.ContentDict]] = list(genai_contents) # Type hint allows str

            # --- Updated System Instruction Handling for Counting ---
            # Prepend system instruction text if it exists, as count_tokens accepts strings
            if system_instruction_text:
                 contents_to_count.insert(0, system_instruction_text) # Prepend the text directly
            # --- End Update ---

            total_tokens = 0
            if contents_to_count:
                 # Ensure contents_to_count matches expected type List[ContentLikable]
                 # which includes str and ContentDict
                 response = await self._client.aio.models.count_tokens(
                     model=f"models/{model_name}",
                     contents=contents_to_count # type: ignore
                 )
                 total_tokens = response.total_tokens
            # No need for separate handling if only system prompt exists, covered above

            return total_tokens
        except genai_errors.GenerativeAIError as e: # Catch base SDK error
            logger.error(f"Gemini API error during message token count for model '{model_name}': {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e}")
        except Exception as e: logger.error(f"Failed to count message tokens for model '{model_name}': {e}", exc_info=True); total_text = " ".join([msg.content for msg in messages]); return (len(total_text) + 3 * len(messages)) // 4

    async def close(self) -> None:
        """Closes resources associated with the Gemini provider."""
        # (Remains the same - no explicit close needed)
        logger.debug("GeminiProvider closed (no explicit client close needed for google-genai).")
        self._client = None # Dereference the client
        pass
