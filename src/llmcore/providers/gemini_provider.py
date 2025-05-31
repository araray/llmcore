# src/llmcore/providers/gemini_provider.py
"""
Google Gemini API provider implementation for the LLMCore library.

Handles interactions with the Google Generative AI API (Gemini models).
Uses the official 'google-genai' library (v1.10.0+).
Accepts context as List[Message].
System instructions are prepended to the 'contents' list as a 'user' message.
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
    GenAIGenerationConfigType = genai_types.GenerationConfig
    GenAICountTokensConfigType = genai_types.CountTokensConfig
    GenAIHttpOptionsType = genai_types.HttpOptions
    GenAIPartDictType = genai_types.PartDict
    GenAIGenerateContentResponseType = genai_types.GenerateContentResponse
    # GenAIContentType and GenAIPartType are not strictly needed here anymore
    # as system_instruction is directly embedded in contents.

except ImportError:
    genai = None # type: ignore
    genai_types = None # type: ignore
    genai_errors = None # type: ignore
    GenAIAPIError = Exception # type: ignore

    GenAIClientType = Any
    GenAISafetySettingDictType = Any
    GenAIContentDictType = Any
    GenAIGenerationConfigType = Any # type: ignore
    GenAICountTokensConfigType = Any # type: ignore
    GenAIHttpOptionsType = Any # type: ignore
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
    "gemini-1.5-flash-latest": 1048576,
    "gemini-1.5-pro-latest": 1048576,
    "gemini-1.0-pro": 30720,
    "gemini-2.0-flash-001": 1048576,
    "gemini-2.5-pro-preview-05-06": 2097152,
    "gemini-ultra": 2097152,
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
        (Docstring unchanged)
        """
        super().__init__(config, log_raw_payloads)
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed or failed to import. "
                              "Install with 'pip install llmcore[gemini]'.")

        self.api_key = config.get('api_key')
        self.default_model = config.get('default_model') or DEFAULT_MODEL
        self.fallback_context_length = int(config.get('fallback_context_length', 1048576))
        self._safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key and not os.environ.get('GOOGLE_API_KEY'):
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
                raise ConfigError("google.genai module is unexpectedly None at client initialization.")

        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client: {e}", exc_info=True)
            raise ConfigError(f"Google Gen AI client initialization failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[List[GenAISafetySettingDictType]]:
        """
        Parses safety settings from config.
        (Implementation unchanged)
        """
        if not settings_config: return None
        if not genai_types: logger.warning("google.genai.types not available. Cannot parse safety settings."); return None
        parsed_settings: List[GenAISafetySettingDictType] = []
        for key_str, value_str in settings_config.items():
            try:
                category_upper = key_str.upper(); threshold_upper = value_str.upper()
                if not category_upper.startswith("HARM_CATEGORY_"): raise ValueError(f"Invalid harm category: {category_upper}")
                if not threshold_upper.startswith("BLOCK_"): raise ValueError(f"Invalid harm block threshold: {threshold_upper}")
                parsed_settings.append({'category': category_upper, 'threshold': threshold_upper}) # type: ignore
            except (AttributeError, ValueError) as e: logger.warning(f"Invalid safety setting: {key_str}={value_str}. Error: {e}")
        logger.debug(f"Parsed safety settings: {parsed_settings}")
        return parsed_settings if parsed_settings else None

    def get_name(self) -> str:
        """Returns the provider name: 'gemini'."""
        return "gemini"

    def get_available_models(self) -> List[str]:
        """Returns a static list of known/common Gemini models."""
        logger.warning("GeminiProvider.get_available_models() returning static list.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given Gemini model."""
        # (Implementation unchanged)
        model_name_key = model or self.default_model; limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name_key)
        if limit is None:
            base_model_name = model_name_key.split('-latest')[0] if '-latest' in model_name_key else model_name_key
            base_model_name = base_model_name.split('-001')[0] if '-001' in base_model_name else base_model_name
            limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(base_model_name)
        if limit is None:
            limit = self.fallback_context_length
            logger.warning(f"Unknown context length for Gemini model '{model_name_key}'. Using fallback: {limit}.")
        return limit

    def _convert_llmcore_msgs_to_genai_contents(
        self,
        messages: List[Message]
        # Removed for_token_counting flag
    ) -> Tuple[Optional[str], List[GenAIContentDictType]]: # System instruction text in tuple is now always None
        """
        Converts LLMCore messages to Gemini's `contents` format.
        System instructions (if any, as the first SYSTEM role message) are
        prepended to the `contents` list as a 'user' role message.
        The first element of the returned tuple (system_instruction_text) will now always be None.
        """
        if not genai_types:
            raise ProviderError(self.get_name(), "google-genai types module not available for message conversion.")

        genai_history: List[GenAIContentDictType] = []
        system_instruction_text: Optional[str] = None # Will remain None
        processed_messages = list(messages)

        # Extract system instruction and prepend it to genai_history as a 'user' message
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            actual_system_instruction_text = processed_messages.pop(0).content
            logger.debug("System instruction extracted and will be prepended to contents.")
            genai_history.append(genai_types.ContentDict(
                role="user", # System instructions are passed as user role in contents for public API
                parts=[genai_types.PartDict(text=actual_system_instruction_text)]
            ))
            logger.debug("Prepended system instruction as 'user' message to contents.")

        last_role_added_to_api = genai_history[-1]['role'] if genai_history and 'role' in genai_history[-1] else None

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
                    logger.debug(f"Merging consecutive Gemini '{genai_role}' messages. Original msg ID: {msg.id}")
                    last_genai_msg_parts = genai_history[-1].get('parts')
                    if isinstance(last_genai_msg_parts, list) and last_genai_msg_parts and 'text' in last_genai_msg_parts[-1]:
                        last_genai_msg_parts[-1]['text'] += f"\n{msg.content}" # type: ignore
                    else:
                        genai_history[-1]['parts'] = [genai_types.PartDict(text=msg.content)]
                    continue
                else:
                    logger.warning(f"Logical inconsistency: last_role_added_to_api='{last_role_added_to_api}' but genai_history is empty.")
            genai_history.append(genai_types.ContentDict(role=genai_role, parts=[genai_types.PartDict(text=msg.content)]))
            last_role_added_to_api = genai_role
        if genai_history and genai_history[-1]['role'] == 'model': # Check after all messages processed
            logger.warning("Gemini conversation history for generation ends with 'model' role.")
        return None, genai_history # Always return None for system_instruction_text now

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Sends a chat completion request to the Google Gemini API.
        System instructions are now always part of the 'context' (prepended as a user message).
        """
        if not self._client or not genai_types or not genai_errors or not GenAIGenerationConfigType or not GenAIHttpOptionsType:
            raise ProviderError(self.get_name(), "Google Gen AI library, types, or specific config classes not available/initialized.")

        model_name = model or self.default_model
        model_name_for_api = f"models/{model_name}"

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        # _system_instruction_text will be None, genai_contents will have system message prepended
        _system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)

        if not genai_contents: # This means original context was empty or only had a system message that got prepended.
                               # If system message was prepended, genai_contents will not be empty.
                               # This check is now primarily for if the original context (sans system) was empty.
            raise ProviderError(self.get_name(), "Cannot make Gemini API call with no effective user/model messages in contents.")

        # Prepare GenerationConfig: system_instruction is NOT set here anymore.
        gen_config_params: Dict[str, Any] = {
            k: kwargs.get(k) for k in ["temperature", "top_p", "top_k", "candidate_count", "seed", "frequency_penalty", "presence_penalty", "response_mime_type", "response_schema"] if kwargs.get(k) is not None
        }
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            gen_config_params["max_output_tokens"] = kwargs["max_tokens"]
        if "stop_sequences" in kwargs and kwargs["stop_sequences"] is not None:
            gen_config_params["stop_sequences"] = kwargs["stop_sequences"]

        timeout_kwarg = kwargs.pop("timeout", None)
        if timeout_kwarg is not None:
            if isinstance(timeout_kwarg, (int, float)) and timeout_kwarg > 0:
                gen_config_params["http_options"] = GenAIHttpOptionsType(timeout=float(timeout_kwarg))
                logger.debug(f"Applying per-request timeout of {timeout_kwarg}s via HttpOptions for Gemini generate_content.")
            else:
                logger.warning(f"Invalid timeout value '{timeout_kwarg}' in chat kwargs for Gemini, ignoring.")

        final_generation_config_obj: Optional[GenAIGenerationConfigType] = None
        if gen_config_params: # Only create if there are params for it
            try:
                final_generation_config_obj = GenAIGenerationConfigType(**gen_config_params)
            except Exception as e_val: # Catch Pydantic validation errors or TypeErrors
                 logger.error(f"Error creating Gemini GenerationConfig: {e_val}. Params: {gen_config_params}", exc_info=True)
                 raise ConfigError(f"Failed to create GenerationConfig: {e_val}")

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
            # safety_settings is a top-level parameter for the SDK methods
            sdk_call_args["safety_settings"] = final_safety_settings
        if "tools" in kwargs and kwargs["tools"] is not None: sdk_call_args["tools"] = kwargs["tools"]
        if "tool_config" in kwargs and kwargs["tool_config"] is not None: sdk_call_args["tool_config"] = kwargs["tool_config"]

        logger.debug(
            f"Sending request to Gemini API: model='{model_name_for_api}', stream={stream}, "
            f"num_contents_passed={len(genai_contents)}"
        )

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            try:
                loggable_sdk_call_args = json.loads(json.dumps(sdk_call_args, default=str))
                request_log_data = {"model_for_api_call": model_name_for_api, "sdk_method_args": loggable_sdk_call_args, "stream_flag_for_method": stream}
                logger.debug(f"RAW LLM REQUEST ({self.get_name()} @ {model_name_for_api}): {json.dumps(request_log_data, indent=2, default=str)}")
            except Exception as e_req_log:
                logger.warning(f"Failed to serialize Gemini raw request for logging: {type(e_req_log).__name__} - {str(e_req_log)[:100]}")

        # The rest of the try-except block for API call and response processing
        # remains largely the same as in gemini_provider_py_v2,
        # as the Pydantic error was during GenerationConfig creation.
        try:
            if stream:
                # (Stream handling logic unchanged from previous version)
                logger.debug(f"Calling self._client.aio.models.generate_content_stream() for model '{model_name_for_api}' with args: {list(sdk_call_args.keys())}")
                response_iterator = await self._client.aio.models.generate_content_stream(model=model_name_for_api, **sdk_call_args) # type: ignore
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text_stream = ""
                    try:
                        async for chunk in response_iterator:
                            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                                try: raw_chunk_str = chunk.model_dump_json(indent=2) if hasattr(chunk, "model_dump_json") else str(chunk); logger.debug(f"RAW LLM STREAM CHUNK ({self.get_name()} @ {model_name_for_api}): {raw_chunk_str}")
                                except Exception as e_chunk_log: logger.warning(f"Failed to serialize raw Gemini stream chunk for logging: {type(e_chunk_log).__name__} - {str(e_chunk_log)[:100]} - Chunk type: {type(chunk).__name__}")
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
                                if hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                    is_blocked = True; block_reason_name = chunk.prompt_feedback.block_reason.name if hasattr(chunk.prompt_feedback.block_reason, 'name') else str(chunk.prompt_feedback.block_reason); finish_reason_str = f"PROMPT_BLOCK_{block_reason_name}"
                            except ValueError as ve:
                                logger.warning(f"ValueError accessing chunk text (likely content blocked): {ve}. Chunk details: {chunk!r}"); is_blocked = True
                                fb = getattr(chunk, "prompt_feedback", None) or (chunk.candidates[0].safety_ratings if chunk.candidates and hasattr(chunk.candidates[0], 'safety_ratings') else None); block_reason_detail = "CONTENT_BLOCKED" # type: ignore
                                if fb:
                                    if hasattr(fb, 'block_reason') and fb.block_reason: block_reason_detail = fb.block_reason.name if hasattr(fb.block_reason, 'name') else str(fb.block_reason)
                                    elif isinstance(fb, list) and fb: block_reason_detail = "SAFETY_RATINGS_TRIGGERED"
                                finish_reason_str = f"SAFETY_BLOCK_{block_reason_detail}"
                            except (InvalidArgument, FailedPrecondition) as api_err: logger.error(f"Gemini API error during stream processing: {api_err}", exc_info=True); raise ProviderError(self.get_name(), f"Gemini API error: {api_err}") from api_err # type: ignore
                            except CoreGoogleAPIError as api_err: logger.error(f"Google Cloud API error during stream processing: {api_err}", exc_info=True); raise ProviderError(self.get_name(), f"Gemini API call failed: {api_err}") from api_err # type: ignore
                            except Exception as e_chunk_proc: logger.error(f"Unexpected error processing stream chunk: {e_chunk_proc}. Chunk: {chunk!r}", exc_info=True); yield {"error": f"Unexpected stream error: {e_chunk_proc}"}; continue
                            if is_blocked: logger.warning(f"Stream chunk blocked due to safety settings. Finish reason: {finish_reason_str}"); yield {"error": f"Content blocked by safety settings. Reason: {finish_reason_str}", "finish_reason": "SAFETY", "done": True}; return
                            full_response_text_stream += chunk_text
                            yield {"model": model_name, "choices": [{"delta": {"content": chunk_text}, "index": 0, "finish_reason": finish_reason_str}], "usage": None, "done": finish_reason_str is not None and finish_reason_str not in ["NOT_SET", "FINISH_REASON_UNSPECIFIED", None]}
                    except GenAIAPIError as e_sdk: logger.error(f"Gemini API error during stream iteration: {e_sdk}", exc_info=True); yield {"error": f"Gemini API Error during stream: {e_sdk}", "done": True} # type: ignore
                    except Exception as e_outer_stream: logger.error(f"Unexpected error processing Gemini stream iterator: {e_outer_stream}", exc_info=True); yield {"error": f"Unexpected stream processing error: {e_outer_stream}", "done": True}
                    finally: logger.debug(f"Gemini stream finished processing. Full streamed text length: {len(full_response_text_stream)}")
                return stream_wrapper()
            else: # Non-streaming
                # (Non-streaming logic unchanged from previous version)
                logger.debug(f"Calling self._client.aio.models.generate_content() for model '{model_name_for_api}' with args: {list(sdk_call_args.keys())}")
                response_obj: GenAIGenerateContentResponseType = await self._client.aio.models.generate_content(model=model_name_for_api, **sdk_call_args) # type: ignore
                logger.debug(f"Processing non-stream response from Gemini model '{model_name_for_api}'")

                # --- Diagnostic Logging ---
                gemini_provider_logger_level_full = logger.getEffectiveLevel()
                logger.debug(
                    f"GEMINI_PROVIDER_FULL_RESPONSE_DEBUG_INFO: "
                    f"log_raw_payloads_enabled={self.log_raw_payloads_enabled}, "
                    f"logger.name={logger.name}, "
                    f"logger.level_int={gemini_provider_logger_level_full}, "
                    f"logger.level_name={logging.getLevelName(gemini_provider_logger_level_full)}, "
                    f"logger.isEnabledFor(DEBUG)={logger.isEnabledFor(logging.DEBUG)}"
                )
                # --- End Diagnostic Logging ---
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(f"RAW LLM RESPONSE ({self.get_name()} @ {model_name_for_api}): {response_obj.model_dump_json(indent=2) if hasattr(response_obj, 'model_dump_json') else str(response_obj)}")
                    except Exception as e_resp_log: logger.warning(f"Failed to serialize Gemini raw response for logging: {type(e_resp_log).__name__} - {str(e_resp_log)[:100]} - Response type: {type(response_obj).__name__}")
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
                    if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                        is_blocked = True; block_reason_name = response_obj.prompt_feedback.block_reason.name if hasattr(response_obj.prompt_feedback.block_reason, 'name') else str(response_obj.prompt_feedback.block_reason); finish_reason_str = f"PROMPT_BLOCK_{block_reason_name}"
                except ValueError as ve:
                    logger.warning(f"Content blocked in Gemini response: {ve}"); is_blocked = True
                    fb = getattr(response_obj, "prompt_feedback", None) or (response_obj.candidates[0].safety_ratings if response_obj.candidates and hasattr(response_obj.candidates[0], 'safety_ratings') else None); block_reason_detail = "CONTENT_BLOCKED" # type: ignore
                    if fb:
                        if hasattr(fb, 'block_reason') and fb.block_reason: block_reason_detail = fb.block_reason.name if hasattr(fb.block_reason, 'name') else str(fb.block_reason)
                        elif isinstance(fb, list) and fb: block_reason_detail = "SAFETY_RATINGS_TRIGGERED"
                    finish_reason_str = f"SAFETY_BLOCK_{block_reason_detail}"
                except (InvalidArgument, FailedPrecondition) as api_err: logger.error(f"Gemini API invocation error: {api_err}"); raise ProviderError(self.get_name(), f"Gemini API error: {api_err}") from api_err # type: ignore
                except CoreGoogleAPIError as api_err: logger.error(f"Google Cloud API error: {api_err}", exc_info=True); raise ProviderError(self.get_name(), f"Gemini API call failed: {api_err}") from None # type: ignore
                except Exception as e_resp_text: logger.error(f"Error extracting Gemini response content: {e_resp_text}", exc_info=True); raise ProviderError(self.get_name(), f"Failed to extract content from Gemini response: {e_resp_text}") from e_resp_text
                if is_blocked: raise ProviderError(self.get_name(), f"Content generation blocked by safety settings. Reason: {finish_reason_str}")
                usage_metadata_dict = None
                if hasattr(response_obj, 'usage_metadata') and response_obj.usage_metadata:
                    usage_metadata_dict = {"prompt_token_count": response_obj.usage_metadata.prompt_token_count, "candidates_token_count": response_obj.usage_metadata.candidates_token_count, "total_token_count": response_obj.usage_metadata.total_token_count}
                result_dict = {"id": response_obj.candidates[0].citation_metadata.citation_sources[0].uri if response_obj.candidates and response_obj.candidates[0].citation_metadata and response_obj.candidates[0].citation_metadata.citation_sources else None, "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": finish_reason_str}], "usage": usage_metadata_dict, "prompt_feedback": {"block_reason": response_obj.prompt_feedback.block_reason.name if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason else None, "safety_ratings": [rating.model_dump() for rating in response_obj.prompt_feedback.safety_ratings] if response_obj.prompt_feedback and hasattr(response_obj.prompt_feedback, 'safety_ratings') else []}} # type: ignore
                return result_dict
        except GenAIAPIError as e_sdk: # type: ignore
            # (Error handling for GenAIAPIError unchanged from previous version)
            logger.error(f"Gemini SDK API error: {e_sdk}", exc_info=True)
            if isinstance(e_sdk, genai_errors.PermissionDeniedError): raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied for Gemini: {e_sdk}") from e_sdk # type: ignore
            if isinstance(e_sdk, genai_errors.InvalidArgumentError): # type: ignore
                error_str_lower = str(e_sdk).lower()
                if "context length" in error_str_lower or "token limit" in error_str_lower or "user input is too long" in error_str_lower or "resource has been exhausted" in error_str_lower:
                    actual_tokens = 0; limit_from_provider = self.get_max_context_length(model_name)
                    try: actual_tokens = await self.count_message_tokens(context, model_name)
                    except Exception: pass
                    raise ContextLengthError(model_name=model_name, limit=limit_from_provider, actual=actual_tokens, message=f"Context length error with Gemini: {e_sdk}") from None
                raise ProviderError(self.get_name(), f"Invalid Argument for Gemini API: {e_sdk}") from e_sdk
            if isinstance(e_sdk, CoreGoogleAPIError): logger.error(f"Google Core API error during Gemini call: {e_sdk}", exc_info=True); raise ProviderError(self.get_name(), f"Google Core API Error: {e_sdk}") from e_sdk # type: ignore
            raise ProviderError(self.get_name(), f"Gemini API Error: {e_sdk}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Gemini API timed out.")
            raise ProviderError(self.get_name(), f"Request timed out.") from asyncio.TimeoutError
        except Exception as e:
            logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Gemini provider: {e}") from None


    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Counts tokens for a single text string using the Gemini API.
        (Implementation unchanged)
        """
        if not self._client or not genai_errors:
            logger.warning("Gemini client/errors not available for token counting. Approximating.")
            return (len(text) + 3) // 4
        if not text: return 0
        model_name_for_api = f"models/{model or self.default_model}"
        logger.debug(f"Counting tokens for single text (len: {len(text)}) with Gemini model '{model_name_for_api}'")
        try:
            response = await self._client.aio.models.count_tokens(model=model_name_for_api, contents=[text])
            return response.total_tokens
        except GenAIAPIError as e_sdk: # type: ignore
            logger.error(f"Gemini API error during token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """
        Counts tokens for LLMCore Messages using Gemini API. System instructions are prepended to contents.
        (Implementation unchanged, relies on modified _convert_llmcore_msgs_to_genai_contents)
        """
        if not self._client or not genai_types or not genai_errors:
            logger.warning("Gemini client, types, or errors module not available for message token counting. Approximating.")
            total_text_len = sum(len(msg.content) for msg in messages)
            return (total_text_len + 3 * len(messages)) // 4
        if not messages: return 0

        model_name_for_api = f"models/{model or self.default_model}"
        logger.debug(f"Counting message tokens for Gemini model '{model_name_for_api}'")
        # _system_instruction_text will be None from the modified _convert method
        _system_instruction_text, final_contents_for_counting = \
            self._convert_llmcore_msgs_to_genai_contents(messages) # No for_token_counting flag needed
        if not final_contents_for_counting:
            logger.debug(f"No contents to count for model '{model_name_for_api}' after processing. Returning 0 tokens.")
            return 0
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            try:
                loggable_contents = [c.model_dump() if hasattr(c, 'model_dump') else c for c in final_contents_for_counting]
                logger.debug(f"RAW COUNT_TOKENS REQUEST ({self.get_name()} @ {model_name_for_api}): "
                             f"{json.dumps({'model': model_name_for_api, 'contents': loggable_contents}, indent=2, default=str)}")
            except Exception as e_log: logger.warning(f"Failed to serialize contents for count_tokens logging: {e_log}")
        try:
            response = await self._client.aio.models.count_tokens(model=model_name_for_api, contents=final_contents_for_counting)
            logger.debug(f"Gemini count_tokens API returned total_tokens: {response.total_tokens}")
            return response.total_tokens
        except GenAIAPIError as e_sdk: # type: ignore
            logger.error(f"Gemini API error during message token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e_sdk}")
        except Exception as e:
            logger.error(f"Failed to count message tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            total_text_len = sum(len(part['text']) for content_dict in final_contents_for_counting for part in content_dict.get('parts',[])) # type: ignore
            return (total_text_len + 3 * len(final_contents_for_counting)) // 4

    async def close(self) -> None:
        """Closes the Gemini client (if applicable)."""
        # (Implementation unchanged)
        logger.debug("GeminiProvider closed (google-genai client typically does not require explicit async close).")
        self._client = None
