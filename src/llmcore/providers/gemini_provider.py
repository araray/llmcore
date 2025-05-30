# src/llmcore/providers/gemini_provider.py
"""
Google Gemini API provider implementation for the LLMCore library.

Handles interactions with the Google Generative AI API (Gemini models).
Uses the official 'google-genai' library (v1.10.0+).
Accepts context as List[Message].
Follows patterns outlined in the official SDK documentation:
https://googleapis.github.io/python-genai/
"""

import asyncio
import json # Added for logging raw payloads
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# --- Granular import checks for google-genai and its dependencies ---
# This section helps in providing more specific error messages if parts of
# the google-genai SDK or its dependencies are missing or incorrectly installed.
google_genai_base_available = False
google_genai_types_module_available = False # For google.genai.types module
google_api_core_exceptions_available = False

try:
    import google.genai as genai
    from google.genai import errors as genai_errors
    from google.genai import \
        types as genai_types  # This is 'google.generativeai.types' effectively
    from google.genai.errors import \
        APIError as GenAIAPIError  # Specific error from google.genai.errors
    google_genai_base_available = True
    google_genai_types_module_available = True # If 'genai_types' imported, the module is there

    # Define GenAIClientType here if base import is successful
    # These are type aliases for convenience and clarity within this module.
    GenAIClientType = genai.Client
    GenAISafetySettingDictType = genai_types.SafetySettingDict
    GenAIContentDictType = genai_types.ContentDict
    GenAIGenerationConfigType = genai_types.GenerationConfig # CORRECTED: Was GenerateContentConfig
    GenAIPartDictType = genai_types.PartDict
    GenAIGenerateContentResponseType = genai_types.GenerateContentResponse
except ImportError:
    # Fallbacks if 'google.genai' or its submodules fail to import.
    # This allows the code to be parsed but will raise an error at runtime if used.
    genai = None # type: ignore [assignment]
    genai_types = None # type: ignore [assignment]
    genai_errors = None # type: ignore [assignment]
    GenAIAPIError = Exception # type: ignore [assignment] # Fallback to base Exception

    # Fallback type hints for client and config types if SDK is not available
    GenAIClientType = Any
    GenAISafetySettingDictType = Any
    GenAIContentDictType = Any
    GenAIGenerationConfigType = Any
    GenAIPartDictType = Any
    GenAIGenerateContentResponseType = Any

# Try to import exceptions from 'google.api_core.exceptions' as they are
# often used by Google Cloud client libraries, including google-genai.
try:
    # CoreGoogleAPIError is the base for many google cloud client library errors
    from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError
    from google.api_core.exceptions import InvalidArgument, PermissionDenied, FailedPrecondition
    google_api_core_exceptions_available = True
except ImportError:
    CoreGoogleAPIError = Exception # type: ignore [assignment] # Fallback type
    PermissionDenied = Exception   # type: ignore [assignment] # Fallback type
    InvalidArgument = Exception  # type: ignore [assignment] # Fallback type
    FailedPrecondition = Exception  # type: ignore [assignment] # Fallback type

# Overall availability check for critical components of the google-genai SDK
google_genai_available = (
    google_genai_base_available and
    google_genai_types_module_available and
    google_api_core_exceptions_available
)

try:
    if genai_types: # Check if genai_types was successfully imported
        CandidateFinishReason = genai_types.Candidate.FinishReason
    else:
        CandidateFinishReason = None # type: ignore
except AttributeError:
    # Fallback if the structure is different or genai_types is None
    logger.warning("Could not directly access genai_types.Candidate.FinishReason. Finish reason comparisons might be string-based.")
    CandidateFinishReason = None # type: ignore
# --- End granular import checks ---


from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..models import Message
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload # ContextPayload is List[Message]
from ..utils.info import list_class_members

# Default context lengths for common Gemini models.
# These values should be periodically verified against official Google documentation.
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-2.0-flash-lite": 1045000,
    "gemini-2.5-pro-preview-05-06": 2090000,
    "gemini-2.0-flash": 1045000,
    "gemini-2.5-flash-preview-04-17": 1045000,
    "gemini-ultra": 2090000, # If accessible
}
# Default model if not specified in config.
# It's good practice to use a widely available and generally capable model as default.
DEFAULT_MODEL = "gemini-2.0-flash-lite" # Changed to a more recent default

# Mapping from LLMCore Role to Gemini Role string
LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model", # Gemini API uses "model" for assistant/LLM responses
    # System role is handled separately by Gemini API (system_instruction in GenerationConfig)
}


class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API using google-genai.
    Handles List[Message] context type.
    Requires the `google-genai` library.
    """
    _client: Optional[GenAIClientType] = None
    _safety_settings: Optional[List[GenAISafetySettingDictType]] = None # This is passed as safety_settings argument

    def __init__(self, config: Dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the GeminiProvider using the google-genai SDK.

        Args:
            config: Configuration dictionary from `[providers.gemini]` containing:
                    'api_key' (optional): Google AI API key. Defaults to env var GOOGLE_API_KEY.
                    'default_model' (optional): Default Gemini model to use.
                    'safety_settings' (optional): Dictionary for configuring safety settings.
                                                  Example: {"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"}
            log_raw_payloads: Whether to log raw request/response payloads (passed to BaseProvider).
        Raises:
            ImportError: If the 'google-genai' library cannot be imported.
            ConfigError: If configuration fails (e.g., during client creation).
        """
        super().__init__(config, log_raw_payloads) # Pass log_raw_payloads to BaseProvider
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed or failed to import. "
                              "Install with 'pip install llmcore[gemini]'.")

        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model') or DEFAULT_MODEL
        # Timeout is typically handled by the SDK's HTTP client, may not be a direct param for genai.Client
        # self.timeout = config.get('timeout')
        self._safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key:
            # It's a warning because the SDK might pick up credentials from the environment
            # through other means (e.g., gcloud CLI, Vertex AI environment).
            logger.warning("Google API key not found in config or environment (GOOGLE_API_KEY). "
                           "Ensure it is set if not using Vertex AI or other auth methods.")

        try:
            client_options = {}
            if self.api_key:
                client_options['api_key'] = self.api_key

            if genai: # Ensure genai module was imported
                self._client = genai.Client(**client_options)
                logger.info("Google Gen AI client initialized successfully.")
            else:
                # This case should be caught by google_genai_available check, but as a safeguard:
                raise ConfigError("google.genai module not available at runtime despite initial check.")

        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client: {e}", exc_info=True)
            raise ConfigError(f"Google Gen AI client initialization failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[List[GenAISafetySettingDictType]]:
        """
        Parses safety settings from config string values to the format expected by google-genai.
        Example input from config: {"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"}
        Expected output format for SDK: [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}]
        """
        if not settings_config or not genai_types: # Check if genai_types is available
            return None

        parsed_settings: List[GenAISafetySettingDictType] = []
        for key_str, value_str in settings_config.items():
            try:
                # The SDK expects specific string literals for HarmCategory and HarmBlockThreshold.
                # We assume the config provides these strings directly.
                category_upper = key_str.upper()
                threshold_upper = value_str.upper()

                # Basic validation (could be more robust by checking against actual enum values if imported)
                # This is a simple check; the SDK will perform the ultimate validation.
                if not category_upper.startswith("HARM_CATEGORY_"):
                    raise ValueError(f"Invalid harm category format: {category_upper}. Expected 'HARM_CATEGORY_...'")
                if not threshold_upper.startswith("BLOCK_"): # e.g., BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
                    raise ValueError(f"Invalid harm block threshold format: {threshold_upper}. Expected 'BLOCK_...'")

                # Construct the dictionary in the format expected by the SDK
                setting: GenAISafetySettingDictType = {
                    'category': category_upper,  # type: ignore [typeddict-item] # SDK expects specific literals
                    'threshold': threshold_upper # type: ignore [typeddict-item]
                }
                parsed_settings.append(setting)
            except (AttributeError, ValueError) as e: # Catch if genai_types.HarmCategory/HarmBlockThreshold were not found or parsing failed
                logger.warning(f"Invalid safety setting format in config: {key_str}={value_str}. Skipping. Error: {e}")

        logger.debug(f"Parsed safety settings for Gemini API: {parsed_settings}")
        return parsed_settings if parsed_settings else None

    def get_name(self) -> str:
        """Returns the provider name: 'gemini'."""
        return "gemini"

    def get_available_models(self) -> List[str]:
        """
        Returns a list of known/potentially available models for Gemini.
        Ideally, this would query the API. For now, returns a static list.
        """
        # TODO: Implement dynamic model listing using self._client.models.list() if feasible and performant.
        # This would require an async method or running sync in thread.
        logger.warning("GeminiProvider.get_available_models() returning static list. "
                       "Refer to Google AI documentation for the latest models.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Returns the estimated maximum context length (in tokens) for the given Gemini model.
        These are estimates and should be verified with official documentation.
        """
        # TODO: Consider dynamic lookup via self._client.models.get(f"models/{model_name}").input_token_limit
        # This would require an async method or running sync in thread.
        model_name_key = model or self.default_model
        # Try direct match first
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name_key)
        if limit is not None:
            return limit

        # Try matching base model name (e.g., "gemini-1.5-pro" from "gemini-1.5-pro-latest")
        base_model_name = model_name_key.split('-latest')[0] if '-latest' in model_name_key else model_name_key
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(base_model_name)
        if limit is not None:
            return limit

        # General fallbacks based on series
        if "gemini-2.0-flash-lite" in model_name_key:
            limit = 1045000 # 1M tokens typically ## 1048576, but 1045000 to give it some room
        elif "gemini-2.5-pro-preview-05-06" in model_name_key:
            limit = 2090000 #2097152
        elif "gemini-2.0-flash" in model_name_key:
            limit = 1045000
        else: # General fallback for unknown Gemini models
            limit = 1045000 # A common older limit, or a safe bet for unlisted text models
            logger.warning(f"Unknown context length for Gemini model '{model_name_key}'. "
                           f"Using fallback limit: {limit}. Please verify with Google AI documentation.")
        return limit

    def _convert_llmcore_msgs_to_genai_contents(
        self,
        messages: List[Message]
    ) -> Tuple[Optional[str], List[GenAIContentDictType]]:
        """
        Converts a list of LLMCore `Message` objects to Gemini's `ContentDict` list format
        and extracts the system instruction text. Skips messages with empty or whitespace-only content.

        Args:
            messages: A list of `llmcore.models.Message` objects.

        Returns:
            A tuple containing:
            - An optional system instruction string.
            - A list of `ContentDict` for the Gemini API history.
        """
        if not genai_types: # Should be caught by google_genai_available, but defensive check
            raise ProviderError(self.get_name(), "google-genai types not available for message conversion.")

        genai_history: List[GenAIContentDictType] = []
        system_instruction_text: Optional[str] = None

        processed_messages = list(messages) # Make a copy to avoid modifying the original list
        # Extract system message first, as it's handled separately by Gemini API
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_instruction_text = getattr(self, 'system_instruction_text', None)
            logger.debug("System instruction text extracted for Gemini request.")

        last_role_added_to_api = None
        for msg in processed_messages:
            genai_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not genai_role:
                logger.warning(f"Skipping message with unmappable role '{msg.role}' for Gemini.")
                continue

            # Skip messages with empty or whitespace-only content
            if not msg.content or not msg.content.strip():
                logger.warning(f"Skipping message ID '{msg.id}' (role: {msg.role}) with empty or whitespace-only content for Gemini API call.")
                continue

            # Gemini API requires alternating user/model roles.
            # If consecutive messages have the same role after mapping, merge them.
            if genai_role == last_role_added_to_api:
                if genai_history: # Ensure history is not empty
                    logger.debug(f"Merging consecutive Gemini '{genai_role}' messages.")
                    # Append content to the last part of the last message
                    last_genai_msg_parts = genai_history[-1].get('parts')
                    if isinstance(last_genai_msg_parts, list) and last_genai_msg_parts:
                        # Assuming text parts for simplicity. msg.content is now guaranteed non-empty.
                        last_genai_msg_parts[-1]['text'] += f"\n{msg.content}" # type: ignore [typeddict-item]
                    else: # If parts somehow don't exist or are not a list, create a new part
                        genai_history[-1]['parts'] = [genai_types.PartDict(text=msg.content)]
                    continue # Skip adding a new ContentDict entry
                # If history is empty but we have consecutive roles (e.g. two initial user messages),
                # this implies an issue with input or prior logic. For now, just add it.
                # However, Gemini usually expects user -> model -> user ...
                logger.warning(f"Consecutive role '{genai_role}' at the beginning of history. Adding as new message.")


            # Create a new ContentDict for the message
            # For simplicity, assuming all content is text. Multi-modal content would require PartDict for images etc.
            # msg.content is now guaranteed non-empty.
            genai_history.append(genai_types.ContentDict(
                role=genai_role,
                parts=[genai_types.PartDict(text=msg.content)]
            ))
            last_role_added_to_api = genai_role

        # Gemini API expects the last message in history (if any) to be 'user' if the model is to respond.
        if genai_history and genai_history[-1]['role'] == 'model':
            logger.warning("Gemini conversation history ends with 'model' role. "
                           "The API might expect a 'user' role message last for a valid turn.")

        return system_instruction_text, genai_history


    async def chat_completion(
        self,
        context: ContextPayload, # ContextPayload is List[Message]
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the Gemini API using google-genai.

        Args:
            context: The context payload to send, as a list of `llmcore.models.Message` objects.
            model: The specific model identifier to use. Defaults to provider's default.
            stream: If True, returns an async generator of response chunks. Otherwise, returns the full response.
            **kwargs: Additional provider-specific parameters (temperature, top_p, top_k, max_output_tokens, etc.).

        Returns:
            A dictionary for full response or an async generator for streamed response.

        Raises:
            ProviderError: If the API call fails.
            ConfigError: If the provider is not properly configured.
            ContextLengthError: If the context exceeds the model's token limit.
        """
        if not self._client or not genai_types or not genai_errors:
            raise ProviderError(self.get_name(), "Google Gen AI library, types, or errors module not available/initialized.")

        model_name = model or self.default_model
        model_name_for_api = f"models/{model_name}" # Gemini API expects "models/" prefix

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        logger.debug("Processing context as List[Message] for Gemini.")
        system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)

        if not genai_contents:
            # This case can happen if all messages in the context were empty/whitespace after filtering.
            raise ProviderError(self.get_name(), "Cannot make Gemini API call with no valid content after filtering empty messages.")

        # Prepare GenerationConfig for the Gemini API call
        # SDK uses GenerationConfig for these parameters
        gen_config_args = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_output_tokens": kwargs.get("max_tokens"), # Renamed from max_tokens_to_sample
            "stop_sequences": kwargs.get("stop_sequences"),
            "candidate_count": kwargs.get("candidate_count", 1), # Default to 1 candidate
        }
        # Filter out None values as SDK expects actual values or omission for GenerationConfig
        gen_config_args_filtered = {k: v for k, v in gen_config_args.items() if v is not None}

        final_generation_config_obj: Optional[GenAIGenerationConfigType] = None
        if gen_config_args_filtered:
            try:
                # Use the corrected GenAIGenerationConfigType which should be genai_types.GenerationConfig
                final_generation_config_obj = GenAIGenerationConfigType(**gen_config_args_filtered) # type: ignore[arg-type]
            except TypeError as te:
                logger.warning(f"Invalid argument provided for Gemini GenerationConfig: {te}. Some parameters might be ignored.")
                # Attempt to create config with only valid arguments
                valid_config_args = {k: v for k,v in gen_config_args_filtered.items() if k in GenAIGenerationConfigType.__annotations__} # type: ignore
                if valid_config_args:
                    final_generation_config_obj = GenAIGenerationConfigType(**valid_config_args) # type: ignore[arg-type]

        # Add system_instruction to the generation_config_obj
        if system_instruction_text:
            if final_generation_config_obj is None:
                final_generation_config_obj = GenAIGenerationConfigType(system_instruction=system_instruction_text) # type: ignore[arg-type]
            else:
                # If generation_config object exists, reconstruct it with system_instruction
                # This assumes to_dict() method is available and returns a compatible dict.
                current_gen_conf_dict = final_generation_config_obj.to_dict() if hasattr(final_generation_config_obj, 'to_dict') else {}
                current_gen_conf_dict['system_instruction'] = system_instruction_text
                final_generation_config_obj = GenAIGenerationConfigType(**current_gen_conf_dict) # type: ignore[arg-type]

        # Safety settings are passed as a separate argument to the SDK methods
        final_safety_settings = self._safety_settings


        logger.debug(
            f"Sending request to Gemini API: model='{model_name_for_api}', stream={stream}, "
            f"num_contents_passed={len(genai_contents)}"
        )

        # --- MODIFICATION START: Get GenerativeModel instance and use it for calls ---
        try:
            actual_model_object = self._client.models.get(model=model_name_for_api)
        except Exception as e_get_model:
            logger.error(f"Failed to get Gemini model instance '{model_name_for_api}': {e_get_model}", exc_info=True)
            raise ProviderError(self.get_name(), f"Could not retrieve Gemini model '{model_name_for_api}': {e_get_model}")

        # Prepare arguments for the SDK methods on the GenerativeModel instance
        # The `model` key is NOT part of these args, as the model object itself is used.
        sdk_call_args: Dict[str, Any] = {
            "contents": genai_contents,
        }
        if final_generation_config_obj: # Both stream and non-stream methods accept generation_config
            sdk_call_args["generation_config"] = final_generation_config_obj
        if final_safety_settings:
            sdk_call_args["safety_settings"] = final_safety_settings
        # Add tools/tool_config here if self.tools or self.tool_config are used by this provider
        # --- MODIFICATION END ---


        # Log raw request payload if enabled
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            try:
                # Safely convert genai_contents and other parts to loggable format
                loggable_contents = [c.to_dict() if hasattr(c, 'to_dict') else c for c in genai_contents]

                # Log the actual sdk_call_args, but also include model_name_for_api for clarity
                loggable_sdk_call_args = sdk_call_args.copy()
                if "generation_config" in loggable_sdk_call_args and hasattr(loggable_sdk_call_args["generation_config"], 'to_dict'):
                    loggable_sdk_call_args["generation_config"] = loggable_sdk_call_args["generation_config"].to_dict()
                if "safety_settings" in loggable_sdk_call_args and loggable_sdk_call_args["safety_settings"] is not None:
                    loggable_sdk_call_args["safety_settings"] = [s.to_dict() if hasattr(s, 'to_dict') else s for s in loggable_sdk_call_args["safety_settings"]] # type: ignore

                request_log_data = {
                    "model_for_api_call": model_name_for_api, # Which model instance we are using
                    "sdk_method_args": loggable_sdk_call_args, # What's actually passed to the method
                    "stream_flag_for_method": stream # The stream flag for the method call
                }
                # Use default=str for json.dumps to handle non-serializable types like enums if they appear
                logger.debug(f"RAW LLM REQUEST ({self.get_name()} @ {model_name_for_api}): {json.dumps(request_log_data, indent=2, default=str)}")
            except Exception as e_req_log:
                logger.warning(f"Failed to serialize Gemini raw request for logging: {type(e_req_log).__name__} - {str(e_req_log)[:100]}")

        try:
            if stream:
                logger.debug(f"Calling self._client.aio.models.generate_content_stream() for model '{model_name_for_api}' with args: {list(sdk_call_args.keys())}")
                # --- MODIFIED CALL ---
                #response_iterator = await actual_model_object.aio.generate_content_stream(**sdk_call_args)
                #print(sdk_call_args)
                response_iterator = await self._client.aio.models.generate_content_stream(model = model_name_for_api, contents = genai_contents)
                # --- END MODIFIED CALL ---

                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = "" # To accumulate text for final saving if needed
                    try:
                        async for chunk in response_iterator: # chunk is GenerateContentResponse
                            # Log raw stream chunk if enabled
                            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                                try:
                                    # chunk is google.generativeai.types.GenerateContentResponse
                                    # Use its to_dict() method for JSON serialization
                                    raw_chunk_str = json.dumps(chunk.to_dict(), indent=2, default=str)
                                    logger.debug(f"RAW LLM STREAM CHUNK ({self.get_name()} @ {model_name_for_api}): {raw_chunk_str}")
                                except Exception as e_chunk_log:
                                    logger.warning(f"Failed to serialize raw Gemini stream chunk for logging: {type(e_chunk_log).__name__} - {str(e_chunk_log)[:100]}")

                            chunk_text = ""
                            finish_reason_str = None
                            is_blocked = False

                            try:
                                # Accessing chunk.text can raise ValueError if content is blocked
                                chunk_text = chunk.text # This is the primary way to get text from a chunk

                                # Check candidate finish reasons
                                if chunk.candidates:
                                    # Use CandidateFinishReason from genai_types if available
                                    current_finish_reason = chunk.candidates[0].finish_reason
                                    if CandidateFinishReason: # Check if enum was imported
                                        if current_finish_reason != CandidateFinishReason.FINISH_REASON_UNSPECIFIED: # Only set if not unspecified
                                            finish_reason_str = current_finish_reason.name
                                        if current_finish_reason == CandidateFinishReason.RECITATION: is_blocked = True; finish_reason_str = "SAFETY_RECITATION"
                                        elif current_finish_reason == CandidateFinishReason.SAFETY: is_blocked = True; finish_reason_str = "SAFETY" # General safety
                                    else: # Fallback to string name if enum not available
                                        finish_reason_str = str(current_finish_reason) if current_finish_reason else None # Use None if unspecified
                                # Handling cases where prompt feedback indicates blocking, even if candidates list is empty or reason is unspecified
                                elif hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                    is_blocked = True
                                    finish_reason_str = f"PROMPT_BLOCK_{chunk.prompt_feedback.block_reason.name}" # type: ignore
                            except ValueError as ve: # If chunk.text itself raises error (e.g. blocked)
                                logger.warning(f"ValueError accessing chunk text (likely content blocked): {ve}. Chunk details: {chunk!r}")
                                is_blocked = True
                                # Try to get more specific block reason
                                fb = getattr(chunk, "prompt_feedback", None) or (chunk.candidates[0].safety_ratings if chunk.candidates else None) # type: ignore
                                finish_reason_str = (f"SAFETY_BLOCK_{(fb.block_reason.name if hasattr(fb, 'block_reason') and fb.block_reason else 'CONTENT')}" if fb else "SAFETY_UNKNOWN_BLOCK_AT_TEXT_ACCESS")
                            except (InvalidArgument, FailedPrecondition) as api_err: # type: ignore
                                logger.error(f"Gemini API error during stream processing: {api_err}", exc_info=True)
                                raise ProviderError(self.get_name(), f"Gemini API error: {api_err}") from api_err
                            except CoreGoogleAPIError as api_err: # type: ignore
                                logger.error(f"Google Cloud API error during stream processing: {api_err}", exc_info=True)
                                raise ProviderError(self.get_name(), f"Gemini API call failed: {api_err}") from api_err
                            except Exception as e_chunk_proc:
                                logger.error(f"Unexpected error processing stream chunk: {e_chunk_proc}. Chunk: {chunk!r}", exc_info=True)
                                yield {"error": f"Unexpected stream error: {e_chunk_proc}"} # Yield error structure
                                continue # Try to process next chunk

                            if is_blocked:
                                logger.warning(f"Stream chunk blocked due to safety settings. Finish reason: {finish_reason_str}")
                                yield {"error": f"Content blocked by safety settings. Reason: {finish_reason_str}", "finish_reason": "SAFETY", "done": True}
                                return # Stop streaming if content is blocked

                            full_response_text += chunk_text
                            # Mimic OpenAI stream structure for consistency in LLMCore.chat
                            yield {
                                "model": model_name, # Or chunk.model if available and consistent
                                "choices": [{"delta": {"content": chunk_text}, "index": 0, "finish_reason": finish_reason_str}],
                                "usage": None, # Usage typically provided at the end for Gemini stream
                                # Determine 'done' more reliably based on finish_reason
                                "done": finish_reason_str is not None and finish_reason_str not in ["NOT_SET", "FINISH_REASON_UNSPECIFIED", None]
                            }
                    except GenAIAPIError as e_sdk: # Catch SDK's base API error for the stream
                        logger.error(f"Gemini API error during stream iteration: {e_sdk}", exc_info=True)
                        yield {"error": f"Gemini API Error during stream: {e_sdk}", "done": True}
                    except Exception as e_outer_stream:
                        logger.error(f"Unexpected error processing Gemini stream iterator: {e_outer_stream}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e_outer_stream}", "done": True}
                    finally:
                        logger.debug("Gemini stream finished processing.")
                return stream_wrapper()
            else: # Non-streaming
                logger.debug(f"Calling model.aio.generate_content() for model '{model_name_for_api}' with args: {list(sdk_call_args.keys())}")
                # --- MODIFIED CALL ---
                response_obj: GenAIGenerateContentResponseType = await actual_model_object.aio.generate_content(**sdk_call_args)
                # --- END MODIFIED CALL ---
                logger.debug(f"Processing non-stream response from Gemini model '{model_name_for_api}'")

                # Log raw response if enabled
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    try:
                        # response is google.generativeai.types.GenerateContentResponse
                        logger.debug(f"RAW LLM RESPONSE ({self.get_name()} @ {model_name_for_api}): {json.dumps(response_obj.to_dict(), indent=2, default=str)}")
                    except Exception as e_resp_log:
                        logger.warning(f"Failed to serialize Gemini raw response for logging: {type(e_resp_log).__name__} - {str(e_resp_log)[:100]}")

                full_text = ""
                finish_reason_str = None
                is_blocked = False

                try:
                    # Accessing response.text can raise ValueError if underlying content is blocked
                    full_text = response_obj.text # This is the primary way to get aggregated text
                    if response_obj.candidates:
                        # Inspect finish_reason enum instead of catching StopCandidateException
                        current_finish_reason = response_obj.candidates[0].finish_reason
                        if CandidateFinishReason: # Check if enum was imported
                             if current_finish_reason != CandidateFinishReason.FINISH_REASON_UNSPECIFIED: finish_reason_str = current_finish_reason.name
                             if current_finish_reason == CandidateFinishReason.RECITATION: is_blocked = True; finish_reason_str = "SAFETY_RECITATION"
                             elif current_finish_reason == CandidateFinishReason.SAFETY: is_blocked = True; finish_reason_str = "SAFETY"
                        else: # Fallback to string name if enum not available
                            finish_reason_str = str(current_finish_reason) if current_finish_reason else None
                    # Handling cases where prompt feedback indicates blocking
                    elif response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                        is_blocked = True
                        finish_reason_str = f"PROMPT_BLOCK_{response_obj.prompt_feedback.block_reason.name}" # type: ignore
                except ValueError as ve:
                    logger.warning(f"Content blocked in Gemini response: {ve}")
                    is_blocked = True
                    fb = getattr(response_obj, "prompt_feedback", None) or (response_obj.candidates[0].safety_ratings if response_obj.candidates else None) # type: ignore
                    finish_reason_str = (f"SAFETY_BLOCK_{(fb.block_reason.name if hasattr(fb, 'block_reason') and fb.block_reason else 'CONTENT')}" if fb else "SAFETY_UNKNOWN_BLOCK_AT_TEXT_ACCESS")
                except (InvalidArgument, FailedPrecondition) as api_err: # type: ignore
                    logger.error(f"Gemini API invocation error: {api_err}")
                    raise ProviderError(self.get_name(), f"Gemini API error: {api_err}") from api_err
                except CoreGoogleAPIError as api_err: # type: ignore
                    logger.error(f"Google Cloud API error: {api_err}", exc_info=True)
                    raise ProviderError(self.get_name(), f"Gemini API call failed: {api_err}") from None
                except Exception as e_resp_text:
                    logger.error(f"Error extracting Gemini response content: {e_resp_text}", exc_info=True)
                    raise ProviderError(self.get_name(), f"Failed to extract content from Gemini response: {e_resp_text}") from e_resp_text

                if is_blocked:
                    # If content is blocked, raise an error rather than returning potentially misleading empty text.
                    raise ProviderError(self.get_name(), f"Content generation blocked by safety settings. Reason: {finish_reason_str}")

                # Prepare usage data if available
                usage_metadata_dict = None
                if hasattr(response_obj, 'usage_metadata') and response_obj.usage_metadata:
                    usage_metadata_dict = {
                        "prompt_token_count": response_obj.usage_metadata.prompt_token_count,
                        "candidates_token_count": response_obj.usage_metadata.candidates_token_count, # Sum if multiple candidates
                        "total_token_count": response_obj.usage_metadata.total_token_count
                    }

                # Construct an OpenAI-like response dictionary for consistency
                result_dict = {
                    "id": response_obj.candidates[0].citation_metadata.citation_sources[0].uri if response_obj.candidates and response_obj.candidates[0].citation_metadata and response_obj.candidates[0].citation_metadata.citation_sources else None, # Example of trying to get an ID
                    "model": model_name, # The requested model name
                    "choices": [{"index": 0, "message": {"role": "model", "content": full_text}, "finish_reason": finish_reason_str}],
                    "usage": usage_metadata_dict,
                    "prompt_feedback": { # Include prompt feedback if available
                        "block_reason": response_obj.prompt_feedback.block_reason.name if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason else None, # type: ignore
                        "safety_ratings": [rating.to_dict() for rating in response_obj.prompt_feedback.safety_ratings] if response_obj.prompt_feedback else [] # type: ignore
                    }
                }
                return result_dict

        except GenAIAPIError as e_sdk: # Catch google.genai.errors.APIError and its children
            logger.error(f"Gemini SDK API error: {e_sdk}", exc_info=True)
            # Specific error handling based on SDK error types
            if isinstance(e_sdk, genai_errors.PermissionDeniedError): # type: ignore[attr-defined] # Check against actual class if imported
                raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied for Gemini: {e_sdk}") from e_sdk
            if isinstance(e_sdk, genai_errors.InvalidArgumentError): # type: ignore[attr-defined]
                # Check if it's a context length error
                if "context length" in str(e_sdk).lower() or "token limit" in str(e_sdk).lower() or "user input is too long" in str(e_sdk).lower():
                    actual_tokens = 0
                    try:
                        actual_tokens = await self.count_message_tokens(context, model_name)
                    except Exception:
                        pass # type: ignore
                    raise ContextLengthError(model_name=model_name, limit=self.get_max_context_length(model_name), actual=actual_tokens, message=f"Context length error with Gemini: {e_sdk}") from None
                raise ProviderError(self.get_name(), f"Invalid Argument for Gemini API: {e_sdk}") from e_sdk
            # Handle other CoreGoogleAPIError types if they are distinct and relevant
            if isinstance(e_sdk, CoreGoogleAPIError): # If it's a more general Google API error
                 logger.error(f"Google Core API error during Gemini call: {e_sdk}", exc_info=True)
                 raise ProviderError(self.get_name(), f"Google Core API Error: {e_sdk}") from e_sdk
            # Fallback for other GenAIAPIError
            raise ProviderError(self.get_name(), f"Gemini API Error: {e_sdk}")
        except asyncio.TimeoutError: # Catch standard asyncio timeout
            logger.error(f"Request to Gemini API timed out.")
            raise ProviderError(self.get_name(), f"Request timed out.") from asyncio.TimeoutError
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Gemini provider: {e}") from None


    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a single string using the Gemini API (google-genai)."""
        if not self._client or not genai_errors: # Check for genai_errors as well
            logger.warning("Gemini client or errors module not available for token counting. Returning rough approximation.")
            return (len(text) + 3) // 4 # Simple character-based approximation
        if not text:
            return 0

        model_name_for_api = f"models/{model or self.default_model}"
        try:
            # --- MODIFIED CALL: Use actual_model_object ---
            actual_model_object = self._client.models.get_model(model_name_for_api)
            response = await actual_model_object.aio.count_tokens(contents=[text])
            # --- END MODIFIED CALL ---
            return response.total_tokens
        except GenAIAPIError as e_sdk: # Catch google.genai.errors.APIError
            logger.error(f"Gemini API error during token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            # Fallback to character approximation if API call fails
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of LLMCore Messages using the Gemini API (google-genai)."""
        if not self._client or not genai_errors: # Check for genai_errors
            logger.warning("Gemini client or errors module not available for message token counting. Approx.")
            total_text_len = sum(len(msg.content) for msg in messages)
            return (total_text_len + 3 * len(messages)) // 4 # Approximation
        if not messages:
            return 0

        model_name_for_api = f"models/{model or self.default_model}"
        try:
            # --- MODIFIED CALL: Use actual_model_object ---
            actual_model_object = self._client.models.get(model=model_name_for_api)
            # list_class_members(actual_model_object)
            system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(messages)

            # Prepare contents for the count_tokens API
            contents_to_count: List[Union[str, GenAIContentDictType]] = []
            if system_instruction_text:
                # The count_tokens API can take a list of strings or ContentDicts.
                # For system instruction, it's usually just text.
                contents_to_count.append(system_instruction_text)

            contents_to_count.extend(genai_contents) # Add the formatted message history

            if not contents_to_count: # If all messages were filtered out (e.g., empty)
                return 0

            #response = await actual_model_object.aio.compute_tokens(contents=contents_to_count) # type: ignore[arg-type]
            #response = await actual_model_object.aio.compute_tokens(contents=contents_to_count)
            response = await self._client.aio.models.count_tokens(
                model = actual_model_object.name,
                contents = contents_to_count,
            )
            # --- END MODIFIED CALL ---
            return response.total_tokens
        except GenAIAPIError as e_sdk: # Catch google.genai.errors.APIError
            logger.error(f"Gemini API error during message token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count message tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            # Fallback approximation
            total_text_len = sum(len(msg.content) for msg in messages)
            if system_instruction_text: # Add system instruction length if it exists
                total_text_len += len(system_instruction_text)
            return (total_text_len + 3 * (len(messages) + (1 if system_instruction_text else 0))) // 4


    async def close(self) -> None:
        """Closes resources associated with the Gemini provider."""
        # The google-genai SDK client typically does not require an explicit close method
        # as it manages its underlying HTTP connections.
        logger.debug("GeminiProvider closed (google-genai client typically does not require explicit close).")
        self._client = None # Allow garbage collection
        pass
