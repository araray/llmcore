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
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple

# --- Granular import checks for google-genai and its dependencies ---
google_genai_base_available = False
google_genai_types_module_available = False # For google.genai.types module
google_api_core_exceptions_available = False

try:
    import google.genai as genai # type: ignore
    from google.genai import types as genai_types # type: ignore # This is 'google.generativeai.types' effectively
    from google.genai import errors as genai_errors # type: ignore
    from google.genai.types import StopCandidateException # type: ignore
    from google.generativeai.types import GenerateContentResponse as GenAIGenerateContentResponseType # type: ignore
    google_genai_base_available = True
    google_genai_types_module_available = True # If 'genai_types' imported, the module is there

    # Define GenAIClientType here if base import is successful
    # GenAIClientType = genai.Client # genai.Client is not used directly; genai.configure sets global state
    GenAISafetySettingDictType = genai_types.SafetySettingDict
    GenAIContentDictType = genai_types.ContentDict
    GenAIGenerationConfigType = genai_types.GenerationConfig # Corrected type
    GenAIPartDictType = genai_types.PartDict

except ImportError:
    # Fallbacks if 'google.genai' or its submodules fail
    genai = None # type: ignore [assignment]
    genai_types = None # type: ignore [assignment]
    genai_errors = None # type: ignore [assignment]
    StopCandidateException = Exception # type: ignore [assignment]
    GenAIGenerateContentResponseType = Any # type: ignore [assignment]


    # Fallback type hints for client and config types
    # GenAIClientType = Any # Not directly used
    GenAISafetySettingDictType = Any
    GenAIContentDictType = Any
    GenAIGenerationConfigType = Any
    GenAIPartDictType = Any


# Try to import exceptions from 'google.api_core.exceptions'
try:
    # CoreGoogleAPIError is the base for many google cloud client library errors
    from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError, PermissionDenied, InvalidArgument # type: ignore
    google_api_core_exceptions_available = True
except ImportError:
    CoreGoogleAPIError = Exception # type: ignore [assignment] # Fallback type
    PermissionDenied = Exception   # type: ignore [assignment] # Fallback type
    InvalidArgument = Exception  # type: ignore [assignment] # Fallback type

# Overall availability depends on all critical parts
google_genai_available = (
    google_genai_base_available and
    google_genai_types_module_available and
    google_api_core_exceptions_available
)
# --- End granular import checks ---


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError, ContextLengthError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Gemini models
# Updated to user-provided values.
# Default context lengths for common Gemini models
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-2.0-flash-lite": 1000000,
    "gemini-2.5-pro-preview-05-06": 2000000,
    "gemini-2.0-flash": 1000000,
    "gemini-2.5-flash-preview-04-17": 1000000,
}
# Default model if not specified in config
DEFAULT_MODEL = "gemini-2.0-flash-lite" # Updated to a common, capable default

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
    # _client is not a client instance but stores the result of genai.configure() which is None.
    # API calls use the globally configured genai module.
    _client: None = None # Explicitly None as genai.configure() returns None
    _safety_settings: Optional[List[GenAISafetySettingDictType]] = None # This is passed as safety_settings argument
    config: Dict[str, Any] # To store the provider-specific config for fallback_context_length

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiProvider using the google-genai SDK.

        Args:
            config: Configuration dictionary from `[providers.gemini]` containing:
                    'api_key' (optional): Google AI API key. Defaults to env var GOOGLE_API_KEY.
                    'default_model' (optional): Default Gemini model to use.
                    'safety_settings' (optional): Dictionary for configuring safety settings.
                                                  Example: {"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"}
                    'fallback_context_length' (optional): Integer fallback token limit for this provider.
        Raises:
            ImportError: If the 'google-genai' library cannot be imported.
            ConfigError: If configuration fails (e.g., during client creation).
        """
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed or failed to import. "
                              "Install with 'pip install llmcore[gemini]'.")

        self.config = config # Store the passed config for later access

        self.api_key = self.config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = self.config.get('default_model') or DEFAULT_MODEL
        self._safety_settings = self._parse_safety_settings(self.config.get('safety_settings'))

        if not self.api_key:
            logger.warning("Google API key not found in config or environment (GOOGLE_API_KEY). "
                           "Ensure it is set if not using Vertex AI or other auth methods.")

        try:
            client_options = {}
            if self.api_key:
                client_options['api_key'] = self.api_key

            # transport can be 'rest' or 'grpc' (grpc might need google-auth)
            # client_options['transport'] = self.config.get('transport', 'rest')


            if genai: # Ensure genai module was imported
                # genai.configure() sets global configuration for the library
                genai.configure(**client_options) # type: ignore
                self._client = None # Explicitly set to None as genai.configure returns None
                logger.info("Google Gen AI client (module) configured successfully.")
            else:
                raise ConfigError("google.genai module not available at runtime despite initial check.")

        except Exception as e:
            logger.error(f"Failed to configure Google Gen AI client (module): {e}", exc_info=True)
            raise ConfigError(f"Google Gen AI client (module) configuration failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[List[GenAISafetySettingDictType]]:
        """
        Parses safety settings from config string values to the format expected by google-genai.
        Example input: {"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"}
        """
        if not settings_config or not genai_types: # Check if genai_types is available
            return None

        parsed_settings: List[GenAISafetySettingDictType] = []
        for key_str, value_str in settings_config.items():
            try:
                category_upper = key_str.upper()
                threshold_upper = value_str.upper()

                if not category_upper.startswith("HARM_CATEGORY_"):
                    raise ValueError(f"Invalid harm category format: {category_upper}")

                valid_thresholds = {"BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"}
                if threshold_upper not in valid_thresholds:
                    raise ValueError(f"Invalid harm block threshold format: {threshold_upper}. Must be one of {valid_thresholds}")

                setting: GenAISafetySettingDictType = {
                    'category': category_upper,  # type: ignore [typeddict-item]
                    'threshold': threshold_upper # type: ignore [typeddict-item]
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
        """
        Returns a list of known/potentially available models for Gemini.
        For now, returns a static list. Dynamic fetching can be added.
        """
        logger.warning("GeminiProvider.get_available_models() returning static list. "
                       "Refer to Google AI documentation for the latest models.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Returns the maximum context length (in tokens) for the given Gemini model.
        Attempts dynamic lookup first, then falls back to predefined limits,
        then to provider-specific configuration, and finally to a hardcoded default.
        """
        model_name = model or self.default_model

        # Note: genai.get_model expects model name without "models/" prefix for some operations,
        # but for direct API model listing or info, it might need "models/".
        # The google-genai SDK's GenerativeModel class takes it without "models/".
        # For genai.get_model, it should be prefixed.
        model_name_for_api_get_model = f"models/{model_name}"

        # 1. Attempt dynamic lookup
        if genai: # Check if genai module is available
            try:
                logger.debug(f"Attempting to dynamically fetch context length for Gemini model: {model_name_for_api_get_model}")
                # genai.get_model is a synchronous call
                model_info = genai.get_model(model_name_for_api_get_model)
                if model_info and hasattr(model_info, 'input_token_limit') and model_info.input_token_limit:
                    logger.info(f"Dynamically fetched context length for {model_name_for_api_get_model}: {model_info.input_token_limit}")
                    return model_info.input_token_limit
                else:
                    logger.warning(f"Dynamic lookup for {model_name_for_api_get_model} did not return input_token_limit or model_info was None.")
            except Exception as e:
                logger.warning(f"Failed to dynamically fetch context length for {model_name_for_api_get_model}: {e}. Falling back.")

        # 2. Fallback to predefined limits (using model_name without "models/" prefix)
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None and ":" in model_name: # Check for base model name if variant (e.g. gemini-1.5-pro-latest:001)
            base_model_name = model_name.split(":")[0]
            limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(base_model_name)
            if limit is not None:
                 logger.debug(f"Using predefined context length for base model {base_model_name} of {model_name}: {limit}")


        if limit is not None:
            logger.debug(f"Using predefined context length for {model_name}: {limit}")
            return limit

        # 3. Fallback to provider-specific config (if user adds it to their TOML)
        # self.config here refers to the dictionary passed during __init__
        provider_config_fallback = self.config.get("fallback_context_length")
        if provider_config_fallback is not None:
            try:
                fallback_val = int(provider_config_fallback)
                logger.info(f"Using fallback_context_length from provider config for {model_name}: {fallback_val}")
                return fallback_val
            except ValueError:
                logger.warning(f"Invalid fallback_context_length in provider config for {model_name}: '{provider_config_fallback}'. Must be an integer. Ignoring.")

        # 4. Final hardcoded fallback (a general safe value for many models)
        hardcoded_fallback_limit = 8192 # A conservative general fallback
        logger.warning(f"Unknown context length for Gemini model '{model_name}' and no provider config fallback. "
                       f"Using hardcoded fallback limit: {hardcoded_fallback_limit}. Please verify with Google AI documentation or configure 'fallback_context_length'.")
        return hardcoded_fallback_limit

    def _convert_llmcore_msgs_to_genai_contents(
        self,
        messages: List[Message]
    ) -> Tuple[Optional[str], List[GenAIContentDictType]]:
        """
        Converts a list of LLMCore `Message` objects to Gemini's `ContentDict` list format
        and extracts the system instruction text.

        Args:
            messages: A list of `llmcore.models.Message` objects.

        Returns:
            A tuple containing:
            - An optional system instruction string.
            - A list of `ContentDict` for the Gemini API history.
        """
        if not genai_types:
            raise ProviderError(self.get_name(), "google-genai types not available for message conversion.")

        genai_history: List[GenAIContentDictType] = []
        system_instruction_text: Optional[str] = None

        processed_messages = list(messages)
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_instruction_text = processed_messages.pop(0).content
            logger.debug("System instruction text extracted for Gemini request.")

        last_role_added_to_api = None
        for msg in processed_messages:
            genai_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not genai_role:
                logger.warning(f"Skipping message with unmappable role '{msg.role}' for Gemini.")
                continue

            if genai_role == last_role_added_to_api:
                if genai_history:
                    logger.debug(f"Merging consecutive Gemini '{genai_role}' messages.")
                    last_genai_msg_parts = genai_history[-1].get('parts')
                    if isinstance(last_genai_msg_parts, list) and last_genai_msg_parts:
                        if 'text' in last_genai_msg_parts[-1] and isinstance(last_genai_msg_parts[-1]['text'], str):
                            last_genai_msg_parts[-1]['text'] += f"\n{msg.content}"
                        else:
                            last_genai_msg_parts[-1]['text'] = msg.content
                    else:
                        genai_history[-1]['parts'] = [genai_types.PartDict(text=msg.content)]
                    continue
                logger.warning(f"Consecutive role '{genai_role}' at the beginning of history. Adding as new message.")

            genai_history.append(genai_types.ContentDict(
                role=genai_role,
                parts=[genai_types.PartDict(text=msg.content)]
            ))
            last_role_added_to_api = genai_role

        if genai_history and genai_history[-1]['role'] == 'model':
            logger.warning("Gemini conversation history ends with 'model' role. "
                           "The API might expect a 'user' role message last for a valid turn.")

        return system_instruction_text, genai_history


    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the Gemini API using google-genai.
        """
        if not genai or not genai_types or not genai_errors:
            raise ProviderError(self.get_name(), "Google Gen AI library, types, or errors module not available/initialized.")

        model_name = model or self.default_model

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        logger.debug("Processing context as List[Message] for Gemini.")
        system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)

        if not genai_contents:
            raise ProviderError(self.get_name(), "Cannot make Gemini API call with no content.")

        gen_config_args = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_output_tokens": kwargs.get("max_tokens"),
            "stop_sequences": kwargs.get("stop_sequences"),
            "candidate_count": kwargs.get("candidate_count", 1),
        }
        gen_config_args_filtered = {k: v for k, v in gen_config_args.items() if v is not None}

        final_generation_config: Optional[GenAIGenerationConfigType] = None
        if gen_config_args_filtered:
            try:
                final_generation_config = genai_types.GenerationConfig(**gen_config_args_filtered) # type: ignore[arg-type]
            except TypeError as te:
                logger.warning(f"Invalid argument provided for Gemini GenerationConfig: {te}. Some parameters might be ignored.")
                valid_config_args = {k: v for k,v in gen_config_args_filtered.items() if k in genai_types.GenerationConfig.__annotations__} # type: ignore
                if valid_config_args:
                    final_generation_config = genai_types.GenerationConfig(**valid_config_args) # type: ignore[arg-type]

        model_init_args = {}
        if system_instruction_text:
            model_init_args["system_instruction"] = system_instruction_text
        if self._safety_settings:
            model_init_args["safety_settings"] = self._safety_settings

        try:
            generative_model_instance = genai.GenerativeModel(
                model_name=model_name,
                **model_init_args
            )
        except Exception as model_init_e:
            logger.error(f"Failed to initialize Gemini GenerativeModel for '{model_name}': {model_init_e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Failed to initialize Gemini model '{model_name}': {model_init_e}")


        logger.debug(
            f"Sending request to Gemini API: model='{model_name}', stream={stream}, "
            f"num_contents_passed={len(genai_contents)}"
        )

        try:
            if stream:
                logger.debug(f"Calling generate_content(stream=True) for model '{model_name}'")
                response_iterator = await generative_model_instance.generate_content_async( # type: ignore
                    contents=genai_contents,
                    generation_config=final_generation_config,
                    stream=True
                )

                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = ""
                    try:
                        async for chunk in response_iterator:
                            chunk_text = ""
                            finish_reason_str = None
                            is_blocked = False

                            try:
                                chunk_text = chunk.text
                                if chunk.candidates and chunk.candidates[0].finish_reason:
                                    finish_reason_str = chunk.candidates[0].finish_reason.name
                            except ValueError as ve:
                                logger.warning(f"ValueError accessing chunk text (likely blocked by safety settings): {ve}. Chunk: {chunk}")
                                is_blocked = True
                                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                    finish_reason_str = f"SAFETY_BLOCK_{chunk.prompt_feedback.block_reason.name}"
                                else:
                                    finish_reason_str = "SAFETY_UNKNOWN_BLOCK"
                            except StopCandidateException as sce:
                                logger.warning(f"Content generation stopped by StopCandidateException (safety filter): {sce}")
                                is_blocked = True
                                finish_reason_str = "SAFETY_CANDIDATE_STOP"
                            except Exception as e_chunk:
                                logger.error(f"Unexpected error processing stream chunk: {e_chunk}. Chunk: {chunk}", exc_info=True)
                                yield {"error": f"Unexpected stream error: {e_chunk}"}
                                continue

                            if is_blocked:
                                logger.warning(f"Stream chunk blocked due to safety settings. Finish reason: {finish_reason_str}")
                                yield {"error": f"Content blocked by safety settings. Reason: {finish_reason_str}", "finish_reason": "SAFETY"}
                                return

                            full_response_text += chunk_text or ""
                            yield {
                                "model": model_name,
                                "choices": [{"delta": {"content": chunk_text or ""}, "index": 0, "finish_reason": finish_reason_str}],
                                "usage": None,
                                "done": finish_reason_str is not None and finish_reason_str != "NOT_SET"
                            }
                    except genai_errors.APIError as e_sdk:
                        logger.error(f"Gemini API error during stream: {e_sdk}", exc_info=True)
                        yield {"error": f"Gemini API Error: {e_sdk}", "done": True}
                    except Exception as e_outer_stream:
                        logger.error(f"Unexpected error processing Gemini stream: {e_outer_stream}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e_outer_stream}", "done": True}
                    finally:
                        logger.debug("Gemini stream finished.")
                return stream_wrapper()
            else:
                logger.debug(f"Calling generate_content() for model '{model_name}'")
                response: GenAIGenerateContentResponseType = await generative_model_instance.generate_content_async( # type: ignore
                    contents=genai_contents,
                    generation_config=final_generation_config
                )

                logger.debug(f"Processing non-stream response from Gemini model '{model_name}'")

                full_text = ""
                finish_reason_str = None
                is_blocked = False

                try:
                    full_text = response.text
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason_str = response.candidates[0].finish_reason.name
                except ValueError as ve:
                    logger.warning(f"Content blocked in Gemini response (ValueError accessing .text): {ve}")
                    is_blocked = True
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        finish_reason_str = f"SAFETY_BLOCK_{response.prompt_feedback.block_reason.name}"
                    else:
                        finish_reason_str = "SAFETY_UNKNOWN_BLOCK"
                except StopCandidateException as sce:
                    logger.warning(f"Content generation stopped by StopCandidateException (safety filter): {sce}")
                    is_blocked = True
                    finish_reason_str = "SAFETY_CANDIDATE_STOP"
                except Exception as e_resp_text:
                    logger.error(f"Error accessing Gemini response content (response.text): {e_resp_text}.", exc_info=True)
                    raise ProviderError(self.get_name(), f"Failed to extract content from Gemini response: {e_resp_text}")

                if is_blocked:
                    raise ProviderError(self.get_name(), f"Content generation blocked by safety settings. Reason: {finish_reason_str}")

                usage_metadata_dict = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage_metadata_dict = {
                        "prompt_token_count": response.usage_metadata.prompt_token_count,
                        "candidates_token_count": response.usage_metadata.candidates_token_count,
                        "total_token_count": response.usage_metadata.total_token_count
                    }

                result_dict = {
                    "id": response.candidates[0].citation_metadata.citation_sources[0].uri if response.candidates and response.candidates[0].citation_metadata and response.candidates[0].citation_metadata.citation_sources else None, # type: ignore
                    "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "model", "content": full_text}, "finish_reason": finish_reason_str}],
                    "usage": usage_metadata_dict,
                    "prompt_feedback": {
                        "block_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback and response.prompt_feedback.block_reason else None,
                        "safety_ratings": [rating.to_dict() for rating in response.prompt_feedback.safety_ratings] if response.prompt_feedback else []
                    }
                }
                return result_dict

        except genai_errors.APIError as e_sdk: # type: ignore
            logger.error(f"Gemini API error: {e_sdk}", exc_info=True)
            if isinstance(e_sdk, PermissionDenied):
                raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied for Gemini: {e_sdk}")
            if isinstance(e_sdk, InvalidArgument):
                if "context length" in str(e_sdk).lower() or "token limit" in str(e_sdk).lower() or "user input is too long" in str(e_sdk).lower():
                    actual_tokens = 0
                    try: actual_tokens = await self.count_message_tokens(context, model_name)
                    except Exception: pass
                    limit = self.get_max_context_length(model_name)
                    raise ContextLengthError(model_name=model_name, limit=limit, actual=actual_tokens, message=f"Context length error with Gemini: {e_sdk}")
                raise ProviderError(self.get_name(), f"Invalid Argument for Gemini API: {e_sdk}")
            if isinstance(e_sdk, CoreGoogleAPIError):
                 logger.error(f"Google Core API error during Gemini call: {e_sdk}", exc_info=True)
                 raise ProviderError(self.get_name(), f"Google Core API Error: {e_sdk}")
            raise ProviderError(self.get_name(), f"Gemini API Error: {e_sdk}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Gemini API timed out.")
            raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e:
            logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Gemini provider: {e}")


    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a single string using the Gemini API (google-genai)."""
        if not genai or not genai_errors:
            logger.warning("Gemini client or errors module not available for token counting. Returning rough approximation.")
            return (len(text) + 3) // 4
        if not text:
            return 0

        model_name_for_api = model or self.default_model
        try:
            counter_model_instance = genai.GenerativeModel(model_name=model_name_for_api)
            response = await counter_model_instance.count_tokens_async(contents=[text]) # type: ignore
            return response.total_tokens
        except genai_errors.APIError as e_sdk: # type: ignore
            logger.error(f"Gemini API error during token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of LLMCore Messages using the Gemini API (google-genai)."""
        if not genai or not genai_errors:
            logger.warning("Gemini client or errors module not available for message token counting. Returning rough approximation.")
            total_text_len = sum(len(msg.content) for msg in messages)
            return (total_text_len + 3 * len(messages)) // 4
        if not messages:
            return 0

        model_name_for_api = model or self.default_model
        try:
            system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(messages)

            contents_to_count: List[Union[str, GenAIContentDictType]] = []
            if system_instruction_text:
                contents_to_count.append(system_instruction_text)

            contents_to_count.extend(genai_contents)

            if not contents_to_count:
                return 0

            counter_model_instance = genai.GenerativeModel(model_name=model_name_for_api)
            response = await counter_model_instance.count_tokens_async(contents=contents_to_count) # type: ignore[arg-type]
            return response.total_tokens
        except genai_errors.APIError as e_sdk: # type: ignore
            logger.error(f"Gemini API error during message token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count message tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            total_text_len = sum(len(msg.content) for msg in messages)
            if system_instruction_text:
                total_text_len += len(system_instruction_text)
            return (total_text_len + 3 * (len(messages) + (1 if system_instruction_text else 0))) // 4


    async def close(self) -> None:
        """Closes resources associated with the Gemini provider."""
        logger.debug("GeminiProvider closed (google-genai client typically does not require explicit close).")
        pass
