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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

# --- Granular import checks for google-genai and its dependencies ---
google_genai_base_available = False
google_genai_types_module_available = False # For google.genai.types module
google_api_core_exceptions_available = False

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import \
        types as genai_types  # This is 'google.generativeai.types' effectively
    from google.genai.errors import \
        APIError as GenAIAPIError  # Specific error from google.genai.errors
    google_genai_base_available = True
    google_genai_types_module_available = True # If 'genai_types' imported, the module is there

    # Define GenAIClientType here if base import is successful
    GenAIClientType = genai.Client
    GenAISafetySettingDictType = genai_types.SafetySettingDict
    GenAIContentDictType = genai_types.ContentDict
    GenAIGenerationConfigType = genai_types.GenerateContentConfig
    GenAIPartDictType = genai_types.PartDict
    GenAIGenerateContentResponseType = genai_types.GenerateContentResponse
except ImportError:
    # Fallbacks if 'google.genai' or its submodules fail
    genai = None # type: ignore [assignment]
    genai_types = None # type: ignore [assignment]
    genai_errors = None # type: ignore [assignment]
    GenAIAPIError = Exception # type: ignore [assignment]

    # Fallback type hints for client and config types
    GenAIClientType = Any
    GenAISafetySettingDictType = Any
    GenAIContentDictType = Any
    GenAIGenerationConfigType = Any
    GenAIPartDictType = Any
    GenAIGenerateContentResponseType = Any

# Try to import exceptions from 'google.api_core.exceptions'
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

# Overall availability depends on all critical parts
google_genai_available = (
    google_genai_base_available and
    google_genai_types_module_available and
    google_api_core_exceptions_available
)
from google.ai.generativelanguage_v1.types import Candidate
# --- End granular import checks ---


from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..models import Message
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

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
    _client: Optional[GenAIClientType] = None
    _safety_settings: Optional[List[GenAISafetySettingDictType]] = None # This is passed as safety_settings argument

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiProvider using the google-genai SDK.

        Args:
            config: Configuration dictionary from `[providers.gemini]` containing:
                    'api_key' (optional): Google AI API key. Defaults to env var GOOGLE_API_KEY.
                    'default_model' (optional): Default Gemini model to use.
                    'safety_settings' (optional): Dictionary for configuring safety settings.
                                                  Example: {"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"}
        Raises:
            ImportError: If the 'google-genai' library cannot be imported.
            ConfigError: If configuration fails (e.g., during client creation).
        """
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed or failed to import. "
                              "Install with 'pip install llmcore[gemini]'.")

        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model') or DEFAULT_MODEL
        # Timeout is typically handled by the SDK's HTTP client, may not be a direct param for genai.Client
        # self.timeout = config.get('timeout')
        self._safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key:
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
        Example input: {"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"}
        """
        if not settings_config or not genai_types: # Check if genai_types is available
            return None

        parsed_settings: List[GenAISafetySettingDictType] = []
        for key_str, value_str in settings_config.items():
            try:
                # Ensure keys and values match the SDK's expected enum-like strings or types
                # The SDK might expect HarmCategory and HarmBlockThreshold enums or their string representations.
                # For simplicity, we assume string representations are accepted.
                category_upper = key_str.upper()
                threshold_upper = value_str.upper()

                # Basic validation (could be more robust by checking against actual enum values if imported)
                if not category_upper.startswith("HARM_CATEGORY_"):
                    raise ValueError(f"Invalid harm category format: {category_upper}")
                if not threshold_upper.startswith("BLOCK_"): # e.g., BLOCK_NONE, BLOCK_ONLY_HIGH
                    raise ValueError(f"Invalid harm block threshold format: {threshold_upper}")

                setting: GenAISafetySettingDictType = {
                    'category': category_upper,  # type: ignore [typeddict-item] # SDK expects specific literals
                    'threshold': threshold_upper # type: ignore [typeddict-item]
                }
                parsed_settings.append(setting)
            except (AttributeError, ValueError) as e: # Catch if genai_types.HarmCategory/HarmBlockThreshold were not found or parsing failed
                logger.warning(f"Invalid safety setting format: {key_str}={value_str}. Skipping. Error: {e}")

        logger.debug(f"Parsed safety settings: {parsed_settings}")
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
        logger.warning("GeminiProvider.get_available_models() returning static list. "
                       "Refer to Google AI documentation for the latest models.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Returns the estimated maximum context length (in tokens) for the given Gemini model.
        """
        # TODO: Consider dynamic lookup via self._client.models.get(f"models/{model_name}").input_token_limit
        model_name = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Attempt to infer from model name patterns if not in the static list
            if model_name.startswith("gemini-1.5"):
                limit = 1048576
            elif model_name.startswith("gemini-1.0-pro-vision") or model_name == "gemini-pro-vision":
                limit = 16384
            elif model_name.startswith("gemini-1.0-pro") or model_name == "gemini-pro":
                limit = 32768
            else: # General fallback
                limit = 32768
                logger.warning(f"Unknown context length for Gemini model '{model_name}'. "
                               f"Using fallback limit: {limit}. Please verify with Google AI documentation.")
        return limit

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
        if not genai_types: # Should be caught by google_genai_available, but defensive check
            raise ProviderError(self.get_name(), "google-genai types not available for message conversion.")

        genai_history: List[GenAIContentDictType] = []
        system_instruction_text: Optional[str] = None

        processed_messages = list(messages) # Make a copy
        # Extract system message first, as it's handled separately by Gemini API
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_instruction_text = processed_messages.pop(0).content
            logger.debug("System instruction text extracted for Gemini request.")

        last_role_added_to_api = None
        for msg in processed_messages:
            genai_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not genai_role:
                logger.warning(f"Skipping message with unmappable role '{msg.role}' for Gemini.")
                continue

            # Gemini API requires alternating user/model roles.
            # If consecutive messages have the same role after mapping, merge them.
            if genai_role == last_role_added_to_api:
                if genai_history: # Ensure history is not empty
                    logger.debug(f"Merging consecutive Gemini '{genai_role}' messages.")
                    # Append content to the last part of the last message
                    last_genai_msg_parts = genai_history[-1].get('parts')
                    if isinstance(last_genai_msg_parts, list) and last_genai_msg_parts:
                        # Assuming text parts for simplicity
                        last_genai_msg_parts[-1]['text'] += f"\n{msg.content}" # type: ignore[typeddict-item]
                    else: # If parts somehow don't exist or are not a list, create a new part
                        genai_history[-1]['parts'] = [genai_types.PartDict(text=msg.content)]
                    continue # Skip adding a new ContentDict entry
                # If history is empty but we have consecutive roles (e.g. two initial user messages),
                # this implies an issue with input or prior logic. For now, just add it.
                # However, Gemini usually expects user -> model -> user ...
                logger.warning(f"Consecutive role '{genai_role}' at the beginning of history. Adding as new message.")


            # Create a new ContentDict for the message
            # For simplicity, assuming all content is text. Multi-modal content would require PartDict for images etc.
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

        # Context is expected to be List[Message]
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        logger.debug("Processing context as List[Message] for Gemini.")
        system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)

        if not genai_contents:
            # This case should ideally not happen if there's a user message.
            raise ProviderError(self.get_name(), "Cannot make Gemini API call with no content.")

        # Prepare GenerationConfig
        # SDK uses GenerateContentConfig for these parameters
        gen_config_args = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_output_tokens": kwargs.get("max_tokens"), # Renamed from max_tokens_to_sample
            "stop_sequences": kwargs.get("stop_sequences"),
            "candidate_count": kwargs.get("candidate_count", 1), # Default to 1 candidate
        }
        # Filter out None values as SDK expects actual values or omission
        gen_config_args_filtered = {k: v for k, v in gen_config_args.items() if v is not None}

        # This will be the 'config' argument for the SDK call
        final_generation_config_obj: Optional[GenAIGenerationConfigType] = None
        if gen_config_args_filtered:
            try:
                final_generation_config_obj = genai_types.GenerateContentConfig(**gen_config_args_filtered) # type: ignore[arg-type]
            except TypeError as te:
                logger.warning(f"Invalid argument provided for Gemini GenerationConfig: {te}. Some parameters might be ignored.")
                valid_config_args = {k: v for k,v in gen_config_args_filtered.items() if k in genai_types.GenerateContentConfig.__annotations__}
                if valid_config_args:
                    final_generation_config_obj = genai_types.GenerateContentConfig(**valid_config_args) # type: ignore[arg-type]

        # Add system_instruction to the generation_config_obj
        if system_instruction_text:
            if final_generation_config_obj is None:
                final_generation_config_obj = genai_types.GenerateContentConfig(system_instruction=system_instruction_text) # type: ignore[arg-type]
            else:
                # If generation_config object exists, reconstruct it with system_instruction
                current_gen_conf_dict = final_generation_config_obj.to_dict() if hasattr(final_generation_config_obj, 'to_dict') else {}
                current_gen_conf_dict['system_instruction'] = system_instruction_text
                final_generation_config_obj = genai_types.GenerateContentConfig(**current_gen_conf_dict) # type: ignore[arg-type]

        # Safety settings are passed as a separate argument to the SDK methods
        final_safety_settings = self._safety_settings


        logger.debug(
            f"Sending request to Gemini API: model='{model_name_for_api}', stream={stream}, "
            f"num_contents_passed={len(genai_contents)}"
        )

        try:
            if stream:
                logger.debug(f"Calling generate_content_stream() for model '{model_name_for_api}'")
                # Pass config and safety_settings as named arguments
                call_args = {
                    "model": model_name_for_api,
                    "contents": genai_contents,
                }
                if final_generation_config_obj:
                    call_args["config"] = final_generation_config_obj
                if final_safety_settings:
                    call_args["safety_settings"] = final_safety_settings

                response_iterator = await self._client.aio.models.generate_content_stream(**call_args)

                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = ""
                    try:
                        async for chunk in response_iterator:
                            chunk_text = ""
                            finish_reason_str = None
                            is_blocked = False

                            try:
                                # Accessing chunk.text can still raise ValueError if blocked at HTTP/text layer
                                chunk_text = chunk.text

                                if chunk.candidates:
                                    fr = chunk.candidates[0].finish_reason
                                    # Normal stop on provided or natural stop sequence
                                    if fr == Candidate.FinishReason.STOP_SEQUENCE:
                                        finish_reason_str = fr.name
                                    # Safety “recitation” block at candidate level
                                    elif fr == Candidate.FinishReason.RECITATION:
                                        is_blocked = True
                                        finish_reason_str = "SAFETY_RECITATION"
                                    else:
                                        finish_reason_str = fr.name
                                else:
                                    # No candidates at all → prompt was blocked before candidate generation
                                    is_blocked = True
                                    fb = getattr(chunk, "prompt_feedback", None)
                                    if fb and fb.block_reason:
                                        finish_reason_str = f"SAFETY_BLOCK_{fb.block_reason.name}"
                                    else:
                                        finish_reason_str = "SAFETY_UNKNOWN_BLOCK"

                                yield {"text": chunk_text, "finish_reason": finish_reason_str}

                            except ValueError as ve:
                                # HTTP/text‐layer block (e.g. network‐level filter)
                                logger.warning(f"ValueError accessing chunk text (blocked?): {ve}. Chunk: {chunk!r}")
                                is_blocked = True
                                fb = getattr(chunk, "prompt_feedback", None)
                                finish_reason_str = (
                                    f"SAFETY_BLOCK_{fb.block_reason.name}"
                                    if fb and fb.block_reason
                                    else "SAFETY_UNKNOWN_BLOCK"
                                )
                                yield {"text": None, "finish_reason": finish_reason_str}

                            except (InvalidArgument, FailedPrecondition) as api_err:
                                # Bad request, safety precondition failures, etc.
                                logger.error(f"API error: {api_err}", exc_info=True)
                                raise  # or: raise ProviderError(...) from api_err

                            except CoreGoogleAPIError as api_err:
                                # Catch-all for other HTTP/gRPC errors (rate limits, server errors, etc.)
                                logger.error(f"Cloud API call failed: {api_err}", exc_info=True)
                                raise  # or: raise ProviderError(...) from api_err

                            except Exception as e_chunk:
                                # Unexpected error in processing this chunk—skip it but continue stream
                                logger.error(f"Unexpected error processing stream chunk: {e_chunk}. Chunk: {chunk!r}", exc_info=True)
                                yield {"error": f"Unexpected stream error: {e_chunk}"}
                                continue

                            if is_blocked:
                                logger.warning(f"Stream chunk blocked due to safety settings. Finish reason: {finish_reason_str}")
                                yield {"error": f"Content blocked by safety settings. Reason: {finish_reason_str}", "finish_reason": "SAFETY"}
                                return # Stop streaming if content is blocked

                            full_response_text += chunk_text
                            # Mimic OpenAI stream structure for consistency
                            yield {
                                "model": model_name, # Or chunk.model if available
                                "choices": [{"delta": {"content": chunk_text}, "index": 0, "finish_reason": finish_reason_str}],
                                "usage": None, # Usage typically provided at the end for Gemini
                                "done": finish_reason_str is not None and finish_reason_str != "NOT_SET" # Heuristic for done
                            }
                    except GenAIAPIError as e_sdk: # Catch SDK's base API error
                        logger.error(f"Gemini API error during stream: {e_sdk}", exc_info=True)
                        yield {"error": f"Gemini API Error: {e_sdk}", "done": True}
                    except Exception as e_outer_stream:
                        logger.error(f"Unexpected error processing Gemini stream: {e_outer_stream}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e_outer_stream}", "done": True}
                    finally:
                        logger.debug("Gemini stream finished.")
                return stream_wrapper()
            else: # Non-streaming
                logger.debug(f"Calling generate_content() for model '{model_name_for_api}'")
                # Pass config and safety_settings as named arguments
                call_args = {
                    "model": model_name_for_api,
                    "contents": genai_contents,
                }
                if final_generation_config_obj:
                    call_args["config"] = final_generation_config_obj
                if final_safety_settings:
                    call_args["safety_settings"] = final_safety_settings

                response: GenAIGenerateContentResponseType = await self._client.aio.models.generate_content(**call_args)

                logger.debug(f"Processing non-stream response from Gemini model '{model_name_for_api}'")

                full_text = ""
                finish_reason_str = None
                is_blocked = False

                try:
                    # Accessing response.text can still raise ValueError if underlying content is blocked
                    full_text = response.text
                    if response.candidates:
                        # Inspect finish_reason enum instead of catching StopCandidateException
                        fr = response.candidates[0].finish_reason
                        if fr == Candidate.FinishReason.STOP_SEQUENCE:
                            finish_reason_str = fr.name
                        elif fr == Candidate.FinishReason.RECITATION:
                            # e.g. safety recitation block
                            is_blocked = True
                            finish_reason_str = "SAFETY_RECITATION"
                        else:
                            finish_reason_str = fr.name
                    else:
                        # No candidates → prompt was fully blocked
                        fb = response.prompt_feedback
                        is_blocked = True
                        if fb and fb.block_reason:
                            finish_reason_str = f"SAFETY_BLOCK_{fb.block_reason.name}"
                        else:
                            finish_reason_str = "SAFETY_UNKNOWN_BLOCK"

                except ValueError as ve:
                    logger.warning(f"Content blocked in Gemini response: {ve}")
                    is_blocked = True
                    fb = getattr(response, "prompt_feedback", None)
                    finish_reason_str = (
                        f"SAFETY_BLOCK_{fb.block_reason.name}"
                        if fb and fb.block_reason
                        else "SAFETY_UNKNOWN_BLOCK"
                    )

                except (InvalidArgument, FailedPrecondition) as api_err:
                    logger.error(f"API invocation error: {api_err}")
                    # Chain the original API error for full context:
                    raise ProviderError(self.get_name(), f"Gemini API error: {api_err}") from api_err

                except CoreGoogleAPIError as api_err:
                    logger.error(f"Cloud API error: {api_err}", exc_info=True)
                    # Suppress the original cause if you don't want the long stack:
                    raise ProviderError(self.get_name(), f"Gemini API call failed: {api_err}") from None

                except Exception as e_resp_text:
                    logger.error(f"Error extracting Gemini response: {e_resp_text}", exc_info=True)
                    # Chain to preserve the unexpected exception’s traceback:
                    raise ProviderError(self.get_name(), f"Failed to extract content: {e_resp_text}") from e_resp_text


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
                    "id": response.candidates[0].citation_metadata.citation_sources[0].uri if response.candidates and response.candidates[0].citation_metadata and response.candidates[0].citation_metadata.citation_sources else None,
                    "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "model", "content": full_text}, "finish_reason": finish_reason_str}],
                    "usage": usage_metadata_dict,
                    "prompt_feedback": {
                        "block_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback and response.prompt_feedback.block_reason else None,
                        "safety_ratings": [rating.to_dict() for rating in response.prompt_feedback.safety_ratings] if response.prompt_feedback else []
                    }
                }
                return result_dict

        except GenAIAPIError as e_sdk:
            logger.error(f"Gemini API error: {e_sdk}", exc_info=True)
            if isinstance(e_sdk, genai_errors.PermissionDeniedError): # type: ignore[attr-defined]
                raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied for Gemini: {e_sdk}") from e_sdk
            if isinstance(e_sdk, genai_errors.InvalidArgumentError): # type: ignore[attr-defined]
                if "context length" in str(e_sdk).lower() or "token limit" in str(e_sdk).lower() or "user input is too long" in str(e_sdk).lower():
                    actual_tokens = 0
                    try: actual_tokens = await self.count_message_tokens(context, model_name) # type: ignore[arg-type]
                    except Exception: pass
                    limit = self.get_max_context_length(model_name)
                    raise ContextLengthError(model_name=model_name, limit=limit, actual=actual_tokens, message=f"Context length error with Gemini: {e_sdk}") from None
                raise ProviderError(self.get_name(), f"Invalid Argument for Gemini API: {e_sdk}") from e_sdk
            if isinstance(e_sdk, CoreGoogleAPIError):
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
        """Counts tokens for a single string using the Gemini API (google-genai)."""
        if not self._client or not genai_errors:
            logger.warning("Gemini client or errors module not available for token counting. Returning rough approximation.")
            return (len(text) + 3) // 4
        if not text:
            return 0

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
        """Counts tokens for a list of LLMCore Messages using the Gemini API (google-genai)."""
        if not self._client or not genai_errors:
            logger.warning("Gemini client or errors module not available for message token counting. Returning rough approximation.")
            total_text_len = sum(len(msg.content) for msg in messages)
            return (total_text_len + 3 * len(messages)) // 4
        if not messages:
            return 0

        model_name_for_api = f"models/{model or self.default_model}"
        try:
            system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(messages)

            contents_to_count: List[Union[str, GenAIContentDictType]] = []
            if system_instruction_text:
                contents_to_count.append(system_instruction_text)

            contents_to_count.extend(genai_contents)

            if not contents_to_count:
                return 0

            response = await self._client.aio.models.count_tokens(model=model_name_for_api, contents=contents_to_count) # type: ignore[arg-type]
            return response.total_tokens
        except GenAIAPIError as e_sdk:
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
        self._client = None
        pass
