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
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# --- Granular import checks for google-genai and its dependencies ---
google_genai_base_available = False
google_genai_types_module_available = False
google_api_core_exceptions_available = False

try:
    import google.genai as genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
    from google.genai.errors import APIError as GenAIAPIError
    google_genai_base_available = True
    google_genai_types_module_available = True

    GenAIClientType = genai.Client
    GenAISafetySettingDictType = genai_types.SafetySettingDict
    GenAIContentDictType = genai_types.ContentDict
    GenAIGenerationConfigType = genai_types.GenerationConfig # CORRECTED: Was GenerateContentConfig
    GenAICountTokensConfig = genai_types.CountTokensConfig # For count_tokens
    GenAIHttpOptions = genai_types.HttpOptions # For timeout configuration
    GenAIPartDictType = genai_types.PartDict
    GenAIGenerateContentResponseType = genai_types.GenerateContentResponse
except ImportError:
    genai = None # type: ignore
    genai_types = None # type: ignore
    genai_errors = None # type: ignore
    GenAIAPIError = Exception # type: ignore

    GenAIClientType = Any
    GenAISafetySettingDictType = Any
    GenAIContentDictType = Any
    GenAIGenerationConfigType = Any
    GenAICountTokensConfig = Any # type: ignore
    GenAIHttpOptions = Any # type: ignore
    GenAIPartDictType = Any
    GenAIGenerateContentResponseType = Any

try:
    from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError
    from google.api_core.exceptions import InvalidArgument, PermissionDenied, FailedPrecondition
    google_api_core_exceptions_available = True
except ImportError:
    CoreGoogleAPIError = Exception # type: ignore
    PermissionDenied = Exception   # type: ignore
    InvalidArgument = Exception  # type: ignore
    FailedPrecondition = Exception # type: ignore

google_genai_available = (
    google_genai_base_available and
    google_genai_types_module_available and
    google_api_core_exceptions_available
)

try:
    if genai_types:
        CandidateFinishReason = genai_types.Candidate.FinishReason
    else:
        CandidateFinishReason = None # type: ignore
except AttributeError:
    logger.warning("Could not directly access genai_types.Candidate.FinishReason. Finish reason comparisons might be string-based.")
    CandidateFinishReason = None # type: ignore
# --- End granular import checks ---


from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..models import Message
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-2.0-flash-lite": 1045000,
    "gemini-2.5-pro-preview-05-06": 2090000,
    "gemini-2.0-flash": 1045000,
    "gemini-2.5-flash-preview-04-17": 1045000,
    "gemini-ultra": 2090000,
}
DEFAULT_MODEL = "gemini-2.0-flash-lite"

LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model",
}


class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API using google-genai.
    Handles List[Message] context type.
    Requires the `google-genai` library.
    """
    _client: Optional[GenAIClientType] = None
    _safety_settings: Optional[List[GenAISafetySettingDictType]] = None

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
        super().__init__(config, log_raw_payloads)
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed or failed to import. "
                              "Install with 'pip install llmcore[gemini]'.")

        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model') or DEFAULT_MODEL
        self._safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key:
            logger.warning("Google API key not found in config or environment (GOOGLE_API_KEY). "
                           "Ensure it is set if not using Vertex AI or other auth methods.")

        try:
            client_options = {}
            if self.api_key:
                client_options['api_key'] = self.api_key

            if genai:
                self._client = genai.Client(**client_options)
                logger.info("Google Gen AI client initialized successfully.")
            else:
                raise ConfigError("google.genai module not available at runtime despite initial check.")

        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client: {e}", exc_info=True)
            raise ConfigError(f"Google Gen AI client initialization failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[List[GenAISafetySettingDictType]]:
        """
        Parses safety settings from config string values to the format expected by google-genai.
        """
        if not settings_config or not genai_types:
            return None

        parsed_settings: List[GenAISafetySettingDictType] = []
        for key_str, value_str in settings_config.items():
            try:
                category_upper = key_str.upper()
                threshold_upper = value_str.upper()

                if not category_upper.startswith("HARM_CATEGORY_"):
                    raise ValueError(f"Invalid harm category format: {category_upper}. Expected 'HARM_CATEGORY_...'")
                if not threshold_upper.startswith("BLOCK_"):
                    raise ValueError(f"Invalid harm block threshold format: {threshold_upper}. Expected 'BLOCK_...'")

                setting: GenAISafetySettingDictType = {
                    'category': category_upper,  # type: ignore
                    'threshold': threshold_upper # type: ignore
                }
                parsed_settings.append(setting)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Invalid safety setting format in config: {key_str}={value_str}. Skipping. Error: {e}")

        logger.debug(f"Parsed safety settings for Gemini API: {parsed_settings}")
        return parsed_settings if parsed_settings else None

    def get_name(self) -> str:
        return "gemini"

    def get_available_models(self) -> List[str]:
        logger.warning("GeminiProvider.get_available_models() returning static list. "
                       "Refer to Google AI documentation for the latest models.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        model_name_key = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name_key)
        if limit is not None:
            return limit

        base_model_name = model_name_key.split('-latest')[0] if '-latest' in model_name_key else model_name_key
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(base_model_name)
        if limit is not None:
            return limit

        if "gemini-2.0-flash-lite" in model_name_key: limit = 1045000
        elif "gemini-2.5-pro-preview-05-06" in model_name_key: limit = 2090000
        elif "gemini-2.0-flash" in model_name_key: limit = 1045000
        else:
            limit = 1045000
            logger.warning(f"Unknown context length for Gemini model '{model_name_key}'. "
                           f"Using fallback limit: {limit}. Please verify with Google AI documentation.")
        return limit

    def _convert_llmcore_msgs_to_genai_contents(
        self,
        messages: List[Message]
    ) -> Tuple[Optional[str], List[GenAIContentDictType]]:
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
            if not msg.content or not msg.content.strip():
                logger.warning(f"Skipping message ID '{msg.id}' (role: {msg.role}) with empty content for Gemini.")
                continue

            if genai_role == last_role_added_to_api:
                if genai_history:
                    logger.debug(f"Merging consecutive Gemini '{genai_role}' messages.")
                    last_genai_msg_parts = genai_history[-1].get('parts')
                    if isinstance(last_genai_msg_parts, list) and last_genai_msg_parts:
                        last_genai_msg_parts[-1]['text'] += f"\n{msg.content}" # type: ignore
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
            logger.warning("Gemini conversation history ends with 'model' role.")
        return system_instruction_text, genai_history

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        if not self._client or not genai_types or not genai_errors:
            raise ProviderError(self.get_name(), "Google Gen AI library, types, or errors module not available/initialized.")

        model_name = model or self.default_model
        model_name_for_api = f"models/{model_name}"

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        logger.debug("Processing context as List[Message] for Gemini.")
        system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)

        if not genai_contents:
            raise ProviderError(self.get_name(), "Cannot make Gemini API call with no valid content after filtering empty messages.")

        # --- Rationale Block for System Instruction & Timeout Handling (Recommendations 1 & 4) ---
        # Pre-state (System Instruction): System instruction text was wrapped in a ContentDict with role="user".
        # Limitation (System Instruction): google.genai.types.GenerationConfig.system_instruction accepts a direct string.
        # Decision Path (System Instruction): Verified SDK documentation (google-geanai_api_modules.html, types.GenerationConfig)
        #                                   confirms `str` is a valid type for system_instruction.
        # Post-state (System Instruction): Pass system_instruction_text directly as a string to GenerationConfig.
        #
        # Pre-state (Timeout): No per-request timeout handling. Relied on SDK/HTTP client defaults.
        # Limitation (Timeout): Lack of fine-grained timeout control for specific requests.
        # Decision Path (Timeout): google.genai.types.GenerationConfig accepts `http_options: HttpOptions`.
        #                          google.genai.types.HttpOptions accepts `timeout: float`.
        # Post-state (Timeout): Extract 'timeout' from kwargs. If valid, create HttpOptions
        #                       and pass it to GenerationConfig. Remove 'timeout' from kwargs
        #                       to prevent it from being passed elsewhere.
        # --- End Rationale Block ---

        gen_config_params: Dict[str, Any] = {
            k: kwargs.get(k) for k in ["temperature", "top_p", "top_k", "candidate_count"] if kwargs.get(k) is not None
        }
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            gen_config_params["max_output_tokens"] = kwargs["max_tokens"]
        if "stop_sequences" in kwargs and kwargs["stop_sequences"] is not None:
            gen_config_params["stop_sequences"] = kwargs["stop_sequences"]

        if system_instruction_text:
            gen_config_params["system_instruction"] = system_instruction_text # Direct string assignment

        # Handle timeout kwarg for HttpOptions
        timeout_kwarg = kwargs.pop("timeout", None) # Extract and remove from general kwargs
        if timeout_kwarg is not None:
            if isinstance(timeout_kwarg, (int, float)) and timeout_kwarg > 0:
                if GenAIHttpOptions: # Check if the type was imported successfully
                    gen_config_params["http_options"] = GenAIHttpOptions(timeout=float(timeout_kwarg))
                    logger.debug(f"Applying per-request timeout of {timeout_kwarg}s via HttpOptions for Gemini.")
                else:
                    logger.warning("GenAIHttpOptions type not available, cannot apply per-request timeout.")
            else:
                logger.warning(f"Invalid timeout value '{timeout_kwarg}' in chat kwargs for Gemini, ignoring.")

        final_generation_config_obj: Optional[GenAIGenerationConfigType] = None
        if gen_config_params:
            try:
                final_generation_config_obj = GenAIGenerationConfigType(**gen_config_params) # type: ignore
            except TypeError as te:
                logger.warning(f"Invalid argument for Gemini GenerationConfig: {te}. Some parameters might be ignored.")
                valid_config_args = {k: v for k,v in gen_config_params.items() if k in GenAIGenerationConfigType.__annotations__} # type: ignore
                if valid_config_args:
                    final_generation_config_obj = GenAIGenerationConfigType(**valid_config_args) # type: ignore

        final_safety_settings = self._safety_settings
        if "safety_settings" in kwargs:
            kwarg_safety_settings = self._parse_safety_settings(kwargs["safety_settings"])
            if kwarg_safety_settings is not None:
                final_safety_settings = kwarg_safety_settings
                logger.debug(f"Overriding provider safety settings with those from kwargs: {final_safety_settings}")

        sdk_call_args: Dict[str, Any] = {"contents": genai_contents}
        if final_generation_config_obj:
            sdk_call_args["generation_config"] = final_generation_config_obj
        if final_safety_settings:
            sdk_call_args["safety_settings"] = final_safety_settings

        logger.debug(
            f"Sending request to Gemini API: model='{model_name_for_api}', stream={stream}, "
            f"num_contents_passed={len(genai_contents)}"
        )

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            try:
                loggable_sdk_call_args = sdk_call_args.copy()
                if "contents" in loggable_sdk_call_args:
                    loggable_sdk_call_args["contents"] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in loggable_sdk_call_args["contents"]]
                if "generation_config" in loggable_sdk_call_args and hasattr(loggable_sdk_call_args["generation_config"], 'to_dict'):
                    loggable_sdk_call_args["generation_config"] = loggable_sdk_call_args["generation_config"].to_dict()
                if "safety_settings" in loggable_sdk_call_args and loggable_sdk_call_args["safety_settings"] is not None:
                    loggable_sdk_call_args["safety_settings"] = [s.to_dict() if hasattr(s, 'to_dict') else s for s in loggable_sdk_call_args["safety_settings"]] # type: ignore

                request_log_data = {"model_for_api_call": model_name_for_api, "sdk_method_args": loggable_sdk_call_args, "stream_flag_for_method": stream}
                logger.debug(f"RAW LLM REQUEST ({self.get_name()} @ {model_name_for_api}): {json.dumps(request_log_data, indent=2, default=str)}")
            except Exception as e_req_log:
                logger.warning(f"Failed to serialize Gemini raw request for logging: {type(e_req_log).__name__} - {str(e_req_log)[:100]}")

        try:
            if stream:
                logger.debug(f"Calling self._client.aio.models.generate_content_stream() for model '{model_name_for_api}' with args: {list(sdk_call_args.keys())}")
                response_iterator = await self._client.aio.models.generate_content_stream(model=model_name_for_api, **sdk_call_args)
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = ""
                    try:
                        async for chunk in response_iterator:
                            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                                try: logger.debug(f"RAW LLM STREAM CHUNK ({self.get_name()} @ {model_name_for_api}): {json.dumps(chunk.to_dict(), indent=2, default=str)}")
                                except Exception as e_chunk_log: logger.warning(f"Failed to serialize raw Gemini stream chunk for logging: {type(e_chunk_log).__name__} - {str(e_chunk_log)[:100]}")
                            chunk_text = ""; finish_reason_str = None; is_blocked = False
                            try:
                                chunk_text = chunk.text
                                if chunk.candidates:
                                    current_finish_reason = chunk.candidates[0].finish_reason
                                    if CandidateFinishReason:
                                        if current_finish_reason != CandidateFinishReason.FINISH_REASON_UNSPECIFIED: finish_reason_str = current_finish_reason.name
                                        if current_finish_reason == CandidateFinishReason.RECITATION: is_blocked = True; finish_reason_str = "SAFETY_RECITATION"
                                        elif current_finish_reason == CandidateFinishReason.SAFETY: is_blocked = True; finish_reason_str = "SAFETY"
                                    else: finish_reason_str = str(current_finish_reason) if current_finish_reason else None
                                elif hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                    is_blocked = True; finish_reason_str = f"PROMPT_BLOCK_{chunk.prompt_feedback.block_reason.name}" # type: ignore
                            except ValueError as ve:
                                logger.warning(f"ValueError accessing chunk text (likely content blocked): {ve}. Chunk details: {chunk!r}")
                                is_blocked = True
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
                                yield {"error": f"Unexpected stream error: {e_chunk_proc}"}; continue
                            if is_blocked:
                                logger.warning(f"Stream chunk blocked due to safety settings. Finish reason: {finish_reason_str}")
                                yield {"error": f"Content blocked by safety settings. Reason: {finish_reason_str}", "finish_reason": "SAFETY", "done": True}; return
                            full_response_text += chunk_text
                            yield {"model": model_name, "choices": [{"delta": {"content": chunk_text}, "index": 0, "finish_reason": finish_reason_str}], "usage": None, "done": finish_reason_str is not None and finish_reason_str not in ["NOT_SET", "FINISH_REASON_UNSPECIFIED", None]}
                    except GenAIAPIError as e_sdk:
                        logger.error(f"Gemini API error during stream iteration: {e_sdk}", exc_info=True)
                        yield {"error": f"Gemini API Error during stream: {e_sdk}", "done": True}
                    except Exception as e_outer_stream:
                        logger.error(f"Unexpected error processing Gemini stream iterator: {e_outer_stream}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e_outer_stream}", "done": True}
                    finally: logger.debug("Gemini stream finished processing.")
                return stream_wrapper()
            else: # Non-streaming
                logger.debug(f"Calling self._client.aio.models.generate_content() for model '{model_name_for_api}' with args: {list(sdk_call_args.keys())}")
                response_obj: GenAIGenerateContentResponseType = await self._client.aio.models.generate_content(model=model_name_for_api, **sdk_call_args)
                logger.debug(f"Processing non-stream response from Gemini model '{model_name_for_api}'")
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    try: logger.debug(f"RAW LLM RESPONSE ({self.get_name()} @ {model_name_for_api}): {json.dumps(response_obj.to_dict(), indent=2, default=str)}")
                    except Exception as e_resp_log: logger.warning(f"Failed to serialize Gemini raw response for logging: {type(e_resp_log).__name__} - {str(e_resp_log)[:100]}")
                full_text = ""; finish_reason_str = None; is_blocked = False
                try:
                    full_text = response_obj.text
                    if response_obj.candidates:
                        current_finish_reason = response_obj.candidates[0].finish_reason
                        if CandidateFinishReason:
                             if current_finish_reason != CandidateFinishReason.FINISH_REASON_UNSPECIFIED: finish_reason_str = current_finish_reason.name
                             if current_finish_reason == CandidateFinishReason.RECITATION: is_blocked = True; finish_reason_str = "SAFETY_RECITATION"
                             elif current_finish_reason == CandidateFinishReason.SAFETY: is_blocked = True; finish_reason_str = "SAFETY"
                        else: finish_reason_str = str(current_finish_reason) if current_finish_reason else None
                    elif response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                        is_blocked = True; finish_reason_str = f"PROMPT_BLOCK_{response_obj.prompt_feedback.block_reason.name}" # type: ignore
                except ValueError as ve:
                    logger.warning(f"Content blocked in Gemini response: {ve}")
                    is_blocked = True
                    fb = getattr(response_obj, "prompt_feedback", None) or (response_obj.candidates[0].safety_ratings if response_obj.candidates else None) # type: ignore
                    finish_reason_str = (f"SAFETY_BLOCK_{(fb.block_reason.name if hasattr(fb, 'block_reason') and fb.block_reason else 'CONTENT')}" if fb else "SAFETY_UNKNOWN_BLOCK_AT_TEXT_ACCESS")
                except (InvalidArgument, FailedPrecondition) as api_err: # type: ignore
                    logger.error(f"Gemini API invocation error: {api_err}"); raise ProviderError(self.get_name(), f"Gemini API error: {api_err}") from api_err
                except CoreGoogleAPIError as api_err: # type: ignore
                    logger.error(f"Google Cloud API error: {api_err}", exc_info=True); raise ProviderError(self.get_name(), f"Gemini API call failed: {api_err}") from None
                except Exception as e_resp_text:
                    logger.error(f"Error extracting Gemini response content: {e_resp_text}", exc_info=True); raise ProviderError(self.get_name(), f"Failed to extract content from Gemini response: {e_resp_text}") from e_resp_text
                if is_blocked: raise ProviderError(self.get_name(), f"Content generation blocked by safety settings. Reason: {finish_reason_str}")
                usage_metadata_dict = None
                if hasattr(response_obj, 'usage_metadata') and response_obj.usage_metadata:
                    usage_metadata_dict = {"prompt_token_count": response_obj.usage_metadata.prompt_token_count, "candidates_token_count": response_obj.usage_metadata.candidates_token_count, "total_token_count": response_obj.usage_metadata.total_token_count}
                result_dict = {"id": response_obj.candidates[0].citation_metadata.citation_sources[0].uri if response_obj.candidates and response_obj.candidates[0].citation_metadata and response_obj.candidates[0].citation_metadata.citation_sources else None, "model": model_name, "choices": [{"index": 0, "message": {"role": "model", "content": full_text}, "finish_reason": finish_reason_str}], "usage": usage_metadata_dict, "prompt_feedback": {"block_reason": response_obj.prompt_feedback.block_reason.name if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason else None, "safety_ratings": [rating.to_dict() for rating in response_obj.prompt_feedback.safety_ratings] if response_obj.prompt_feedback else []}} # type: ignore
                return result_dict
        except GenAIAPIError as e_sdk:
            logger.error(f"Gemini SDK API error: {e_sdk}", exc_info=True)
            if isinstance(e_sdk, genai_errors.PermissionDeniedError): # type: ignore
                raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied for Gemini: {e_sdk}") from e_sdk
            if isinstance(e_sdk, genai_errors.InvalidArgumentError): # type: ignore
                # --- Rationale Block for Context Length Error Detection (Recommendation 2) ---
                # Pre-state: String matching for "context length", "token limit", "user input is too long".
                # Limitation: Fragile due to potential changes in API error messages.
                # Decision Path: No specific structured error type for token limits was readily identifiable
                #                in the provided SDK documentation snippets for `google.genai.errors` or
                #                `google.api_core.exceptions`.
                # Post-state: Retain string matching as a heuristic, but add a comment acknowledging
                #             its fragility and the desirability of a more structured error from the SDK.
                # --- End Rationale Block ---
                if "context length" in str(e_sdk).lower() or \
                   "token limit" in str(e_sdk).lower() or \
                   "user input is too long" in str(e_sdk).lower() or \
                   "resource has been exhausted" in str(e_sdk).lower(): # Added another common phrase
                    # Note: The string matching for context length errors is fragile.
                    # A more robust solution would be a specific error code or type from the SDK.
                    actual_tokens = 0; limit_from_provider = self.get_max_context_length(model_name)
                    try: actual_tokens = await self.count_message_tokens(context, model_name)
                    except Exception: pass # type: ignore
                    raise ContextLengthError(model_name=model_name, limit=limit_from_provider, actual=actual_tokens, message=f"Context length error with Gemini: {e_sdk}") from None
                raise ProviderError(self.get_name(), f"Invalid Argument for Gemini API: {e_sdk}") from e_sdk
            if isinstance(e_sdk, CoreGoogleAPIError): # type: ignore
                 logger.error(f"Google Core API error during Gemini call: {e_sdk}", exc_info=True)
                 raise ProviderError(self.get_name(), f"Google Core API Error: {e_sdk}") from e_sdk
            raise ProviderError(self.get_name(), f"Gemini API Error: {e_sdk}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Gemini API timed out.")
            raise ProviderError(self.get_name(), f"Request timed out.") from asyncio.TimeoutError
        except Exception as e:
            logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Gemini provider: {e}") from None

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        if not self._client or not genai_errors:
            logger.warning("Gemini client or errors module not available for token counting. Returning rough approximation.")
            return (len(text) + 3) // 4
        if not text: return 0
        model_name_for_api = f"models/{model or self.default_model}"
        try:
            response = await self._client.aio.models.count_tokens(model=model_name_for_api, contents=[text])
            return response.total_tokens
        except GenAIAPIError as e_sdk:
            logger.error(f"Gemini API error during token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        if not self._client or not genai_types or not genai_errors or not GenAICountTokensConfig:
            logger.warning("Gemini client, types, errors, or CountTokensConfig not available for message token counting. Approximating.")
            total_text_len = sum(len(msg.content) for msg in messages)
            return (total_text_len + 3 * len(messages)) // 4
        if not messages: return 0

        model_name_for_api = f"models/{model or self.default_model}"
        try:
            # --- Rationale Block for System Instruction Token Counting (Recommendation 3) ---
            # Pre-state: System instruction was added as a user message to `contents_to_count`.
            # Limitation: This is an approximation and may not perfectly match the API's internal accounting.
            # Decision Path: The `google.genai.types.CountTokensConfig` class supports a
            #                `system_instruction` field (accepting `str` or `Content`).
            #                The `client.models.count_tokens` method accepts a `config` parameter
            #                of this type.
            # Post-state: Use `CountTokensConfig` to pass `system_instruction_text` directly
            #             to the SDK's `count_tokens` method for potentially more accurate counting.
            # --- End Rationale Block ---
            system_instruction_text, genai_contents_for_counting = self._convert_llmcore_msgs_to_genai_contents(messages)

            sdk_count_tokens_args: Dict[str, Any] = {"contents": genai_contents_for_counting} # type: ignore

            if system_instruction_text:
                # Pass system_instruction directly as a string within CountTokensConfig
                count_config_obj = GenAICountTokensConfig(system_instruction=system_instruction_text) # type: ignore
                sdk_count_tokens_args["config"] = count_config_obj
                logger.debug(f"Using CountTokensConfig with system_instruction for model '{model_name_for_api}'.")
            elif not genai_contents_for_counting: # If only system instruction was present and now it's handled by config
                logger.debug(f"No regular contents to count for model '{model_name_for_api}' after extracting system instruction. "
                             f"If system_instruction_text is present, its tokens will be counted via config.")
                # If only system_instruction_text exists, and genai_contents_for_counting is empty,
                # we still need to make the call if system_instruction_text is to be counted.
                if not system_instruction_text: return 0 # No content at all
                # Ensure 'contents' is not empty if config is the only thing, API might require it.
                # However, SDK docs for count_tokens show 'contents' is optional if other fields are set.
                # Let's ensure contents is at least an empty list if system_instruction is primary.
                if "contents" not in sdk_count_tokens_args or not sdk_count_tokens_args["contents"]:
                    sdk_count_tokens_args["contents"] = [] # API might need it even if empty

            if not sdk_count_tokens_args.get("contents") and not sdk_count_tokens_args.get("config"):
                logger.debug(f"No contents or system instruction to count for model '{model_name_for_api}'. Returning 0 tokens.")
                return 0

            response = await self._client.aio.models.count_tokens(
                model=model_name_for_api,
                **sdk_count_tokens_args
            )
            return response.total_tokens
        except GenAIAPIError as e_sdk:
            logger.error(f"Gemini API error during message token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count message tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            total_text_len = sum(len(msg.content) for msg in messages)
            if system_instruction_text: total_text_len += len(system_instruction_text)
            return (total_text_len + 3 * (len(messages) + (1 if system_instruction_text else 0))) // 4

    async def close(self) -> None:
        logger.debug("GeminiProvider closed (google-genai client typically does not require explicit close).")
        self._client = None
