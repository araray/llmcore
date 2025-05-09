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

# --- Use the new google-genai library ---
try:
    import google.genai as genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors # Import the errors module
    from google.generativeai.types import StopCandidateException # Specific exception for blocked content
    from google.genai.errors import APIError as GenAIAPIError # Use correct base error
    from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError # Base for some API errors

    # --- Define type aliases for hinting ---
    GenAIClientType = genai.Client
    GenAISafetySettingDictType = genai_types.SafetySettingDict
    GenAIContentDictType = genai_types.ContentDict
    GenAIGenerationConfigType = genai_types.GenerateContentConfig
    GenAIPartDictType = genai_types.PartDict
    GenAIGenerateContentResponseType = genai_types.GenerateContentResponse
    # --- End Define type aliases ---
    google_genai_available = True
except ImportError:
    google_genai_available = False
    genai = None # type: ignore [assignment]
    genai_types = None # type: ignore [assignment]
    genai_errors = None # type: ignore [assignment]
    StopCandidateException = Exception # type: ignore [assignment]
    GenAIAPIError = Exception # type: ignore [assignment]
    CoreGoogleAPIError = Exception # type: ignore [assignment]
    # --- Define fallback type hints ---
    GenAIClientType = Any # Use Any as fallback type hint
    GenAISafetySettingDictType = Any
    GenAIContentDictType = Any
    GenAIGenerationConfigType = Any
    GenAIPartDictType = Any
    GenAIGenerateContentResponseType = Any
    # --- End Define fallback type hints ---


from ..models import Message, Role as LLMCoreRole
from ..exceptions import ProviderError, ConfigError, ContextLengthError # MCPError removed
from .base import BaseProvider, ContextPayload # ContextPayload is List[Message]

logger = logging.getLogger(__name__)

# Default context lengths for common Gemini models
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-1.5-pro-latest": 1048576,
    "gemini-1.5-flash-latest": 1048576,
    "gemini-1.0-pro": 32768, # Often refers to gemini-1.0-pro-001 or similar
    "gemini-pro": 32768, # Alias for 1.0 pro
    "gemini-1.0-pro-vision-latest": 16384, # Check specific vision model names
    "gemini-pro-vision": 16384, # Alias
    "gemini-2.0-flash-lite": 1048576, # Placeholder, verify actual model name and limits
}
# Default model if not specified in config
DEFAULT_MODEL = "gemini-1.5-flash-latest" # Updated to a common, capable default

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
    _safety_settings: Optional[List[GenAISafetySettingDictType]] = None

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

        # Context is expected to be List[Message]
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}. Expected List[Message].")

        logger.debug("Processing context as List[Message] for Gemini.")
        system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(context)

        # The last message in `genai_contents` is effectively the current user prompt.
        # The rest of `genai_contents` (if any) becomes the history.
        current_prompt_content: List[GenAIContentDictType]
        history_contents: List[GenAIContentDictType] = []

        if genai_contents:
            # The new SDK's `generate_content` takes the full conversation as `contents`.
            # The last message in `contents` is treated as the current prompt.
            # So, we pass `genai_contents` directly.
            current_prompt_content = genai_contents # The whole list is passed as `contents`
            # No separate history needed for the `generate_content` method's `contents` argument.
        else:
            # This case should ideally not happen if there's a user message.
            logger.warning("No content derived from LLMCore messages for Gemini API call.")
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

        generation_config_obj: Optional[GenAIGenerationConfigType] = None
        if gen_config_args_filtered:
            try:
                generation_config_obj = genai_types.GenerateContentConfig(**gen_config_args_filtered) # type: ignore[arg-type]
            except TypeError as te:
                logger.warning(f"Invalid argument provided for Gemini GenerationConfig: {te}. Some parameters might be ignored.")
                # Attempt to create with valid args only if possible, or log and proceed with None
                valid_config_args = {k: v for k,v in gen_config_args_filtered.items() if k in genai_types.GenerateContentConfig.__annotations__}
                if valid_config_args:
                    generation_config_obj = genai_types.GenerateContentConfig(**valid_config_args) # type: ignore[arg-type]


        # Combine system_instruction, safety_settings into the config object
        # The new SDK structure uses a single `config` parameter in `generate_content` which is `GenerateContentConfig`
        # and `system_instruction` is a parameter of `GenerateContentConfig`.
        # `safety_settings` is a direct parameter to `generate_content`.

        final_sdk_config_params: Dict[str, Any] = {}
        if generation_config_obj:
            # If GenerateContentConfig is a Pydantic model, convert to dict for merging if needed,
            # or pass the object directly if the SDK method expects it.
            # The `generate_content` method takes `generation_config` (the object) and `safety_settings` separately.
            final_sdk_config_params['generation_config'] = generation_config_obj

        if system_instruction_text:
            # System instruction is now part of generation_config in the new SDK structure
            if 'generation_config' not in final_sdk_config_params or final_sdk_config_params['generation_config'] is None:
                final_sdk_config_params['generation_config'] = genai_types.GenerateContentConfig(system_instruction=system_instruction_text) # type: ignore[arg-type]
            else:
                # If generation_config object exists, try to set system_instruction on it
                # This assumes GenerateContentConfig is mutable or can be reconstructed
                current_gen_conf_dict = final_sdk_config_params['generation_config'].to_dict() if hasattr(final_sdk_config_params['generation_config'], 'to_dict') else {}
                current_gen_conf_dict['system_instruction'] = system_instruction_text
                final_sdk_config_params['generation_config'] = genai_types.GenerateContentConfig(**current_gen_conf_dict) # type: ignore[arg-type]


        if self._safety_settings:
            final_sdk_config_params['safety_settings'] = self._safety_settings

        logger.debug(
            f"Sending request to Gemini API: model='models/{model_name}', stream={stream}, "
            f"num_contents_passed={len(current_prompt_content)}"
        )

        try:
            # Use the asynchronous client's model interface
            async_model_interface = self._client.aio.models.get(f"models/{model_name}")

            if stream:
                logger.debug(f"Calling generate_content(stream=True) for model 'models/{model_name}'")
                # The `generate_content` method with stream=True returns an async iterator.
                response_iterator = await async_model_interface.generate_content(
                    contents=current_prompt_content, # Pass the full conversation history
                    stream=True,
                    **final_sdk_config_params # Pass generation_config and safety_settings
                )

                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = ""
                    try:
                        async for chunk in response_iterator:
                            chunk_text = ""
                            finish_reason_str = None
                            is_blocked = False

                            try:
                                # Accessing chunk.text can raise ValueError if content is blocked
                                chunk_text = chunk.text
                                if chunk.candidates and chunk.candidates[0].finish_reason:
                                    finish_reason_str = chunk.candidates[0].finish_reason.name
                            except ValueError as ve: # Often indicates blocked content
                                logger.warning(f"ValueError accessing chunk text (likely blocked by safety settings): {ve}. Chunk: {chunk}")
                                is_blocked = True
                                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                    finish_reason_str = f"SAFETY_BLOCK_{chunk.prompt_feedback.block_reason.name}"
                                else:
                                    finish_reason_str = "SAFETY_UNKNOWN_BLOCK"
                            except StopCandidateException as sce: # Explicit exception for blocked candidate
                                logger.warning(f"Content generation stopped by StopCandidateException (safety filter): {sce}")
                                is_blocked = True
                                finish_reason_str = "SAFETY_CANDIDATE_STOP"
                            except Exception as e_chunk:
                                logger.error(f"Unexpected error processing stream chunk: {e_chunk}. Chunk: {chunk}", exc_info=True)
                                yield {"error": f"Unexpected stream error: {e_chunk}"}
                                continue # Skip this problematic chunk

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
                logger.debug(f"Calling generate_content(stream=False) for model 'models/{model_name}'")
                response: GenAIGenerateContentResponseType = await async_model_interface.generate_content(
                    contents=current_prompt_content, # Pass the full conversation history
                    stream=False,
                    **final_sdk_config_params # Pass generation_config and safety_settings
                )

                logger.debug(f"Processing non-stream response from Gemini model 'models/{model_name}'")

                full_text = ""
                finish_reason_str = None
                is_blocked = False

                try:
                    # Accessing response.text can raise ValueError if content is blocked
                    full_text = response.text
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason_str = response.candidates[0].finish_reason.name
                except ValueError as ve: # Often indicates blocked content
                    logger.warning(f"Content blocked in Gemini response (ValueError accessing .text): {ve}")
                    is_blocked = True
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        finish_reason_str = f"SAFETY_BLOCK_{response.prompt_feedback.block_reason.name}"
                    else:
                        finish_reason_str = "SAFETY_UNKNOWN_BLOCK"
                    # full_text will remain empty or as partially extracted if error occurred mid-extraction
                except StopCandidateException as sce: # Explicit exception for blocked candidate
                    logger.warning(f"Content generation stopped by StopCandidateException (safety filter): {sce}")
                    is_blocked = True
                    finish_reason_str = "SAFETY_CANDIDATE_STOP"
                except Exception as e_resp_text:
                    logger.error(f"Error accessing Gemini response content (response.text): {e_resp_text}.", exc_info=True)
                    raise ProviderError(self.get_name(), f"Failed to extract content from Gemini response: {e_resp_text}")

                if is_blocked:
                    # Raise a provider error if the whole response is blocked
                    raise ProviderError(self.get_name(), f"Content generation blocked by safety settings. Reason: {finish_reason_str}")

                # Extract usage metadata if available
                usage_metadata_dict = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage_metadata_dict = {
                        "prompt_token_count": response.usage_metadata.prompt_token_count,
                        "candidates_token_count": response.usage_metadata.candidates_token_count, # Sum if multiple candidates
                        "total_token_count": response.usage_metadata.total_token_count
                    }

                # Mimic OpenAI response structure
                result_dict = {
                    "id": response.candidates[0].citation_metadata.citation_sources[0].uri if response.candidates and response.candidates[0].citation_metadata and response.candidates[0].citation_metadata.citation_sources else None, # Example ID source
                    "model": model_name, # Or from response if available
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "model", # Gemini uses "model" for assistant
                            "content": full_text
                        },
                        "finish_reason": finish_reason_str
                    }],
                    "usage": usage_metadata_dict,
                    "prompt_feedback": { # Include prompt feedback if available
                        "block_reason": response.prompt_feedback.block_reason.name if response.prompt_feedback and response.prompt_feedback.block_reason else None,
                        "safety_ratings": [rating.to_dict() for rating in response.prompt_feedback.safety_ratings] if response.prompt_feedback else []
                    }
                }
                return result_dict

        except GenAIAPIError as e_sdk: # Catch SDK's base API error
            logger.error(f"Gemini API error: {e_sdk}", exc_info=True)
            # Check for specific error types if needed (e.g., PermissionDenied, InvalidArgumentError)
            if isinstance(e_sdk, genai_errors.PermissionDeniedError): # type: ignore[attr-defined]
                raise ProviderError(self.get_name(), f"API Key Invalid or Permission Denied for Gemini: {e_sdk}")
            if isinstance(e_sdk, genai_errors.InvalidArgumentError): # type: ignore[attr-defined]
                # Check if it's a context length error
                if "context length" in str(e_sdk).lower() or "token limit" in str(e_sdk).lower() or "user input is too long" in str(e_sdk).lower():
                    # Attempt to count tokens to provide more specific error
                    actual_tokens = 0
                    try:
                        actual_tokens = await self.count_message_tokens(context, model_name) # type: ignore[arg-type]
                    except Exception: # Ignore errors during this recount
                        pass
                    limit = self.get_max_context_length(model_name)
                    raise ContextLengthError(model_name=model_name, limit=limit, actual=actual_tokens, message=f"Context length error with Gemini: {e_sdk}")
                raise ProviderError(self.get_name(), f"Invalid Argument for Gemini API: {e_sdk}")
            # StopCandidateException is now handled within stream/non-stream blocks
            if isinstance(e_sdk, CoreGoogleAPIError): # Catch underlying google-api-core errors
                 logger.error(f"Google Core API error during Gemini call: {e_sdk}", exc_info=True)
                 raise ProviderError(self.get_name(), f"Google Core API Error: {e_sdk}")
            raise ProviderError(self.get_name(), f"Gemini API Error: {e_sdk}")
        except asyncio.TimeoutError:
            logger.error(f"Request to Gemini API timed out.")
            raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred with Gemini provider: {e}")


    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a single string using the Gemini API (google-genai)."""
        if not self._client or not genai_errors: # Check genai_errors as well
            logger.warning("Gemini client or errors module not available for token counting. Returning rough approximation.")
            return (len(text) + 3) // 4 # Fallback
        if not text:
            return 0

        model_name_for_api = f"models/{model or self.default_model}"
        try:
            # Use the asynchronous client's model interface for counting
            async_model_interface = self._client.aio.models.get(model_name_for_api)
            response = await async_model_interface.count_tokens(contents=[text]) # Pass content as a list
            return response.total_tokens
        except GenAIAPIError as e_sdk: # Catch SDK's base API error
            logger.error(f"Gemini API error during token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e_sdk}")
        except Exception as e: # Fallback for other errors
            logger.error(f"Failed to count tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            # Fallback to character approximation if API call fails
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of LLMCore Messages using the Gemini API (google-genai)."""
        if not self._client or not genai_errors: # Check genai_errors
            logger.warning("Gemini client or errors module not available for message token counting. Returning rough approximation.")
            total_text_len = sum(len(msg.content) for msg in messages)
            return (total_text_len + 3 * len(messages)) // 4 # Fallback
        if not messages:
            return 0

        model_name_for_api = f"models/{model or self.default_model}"
        try:
            system_instruction_text, genai_contents = self._convert_llmcore_msgs_to_genai_contents(messages)

            # The `count_tokens` method accepts a list of `Content` objects or strings.
            # We need to construct the `contents` argument carefully.
            # If there's a system instruction, it's usually counted as part of the overall prompt.
            # The SDK might handle this implicitly if system_instruction is part of a config object,
            # but for direct counting, we might need to include it.

            contents_to_count: List[Union[str, GenAIContentDictType]] = []
            if system_instruction_text:
                # Prepending system instruction as a simple string for counting.
                # More accurate counting might involve structuring it as a specific Content object if the API expects that.
                contents_to_count.append(system_instruction_text)

            contents_to_count.extend(genai_contents) # Add the user/model messages

            if not contents_to_count: # If only a system message was present and it was empty, or no messages
                return 0

            async_model_interface = self._client.aio.models.get(model_name_for_api)
            response = await async_model_interface.count_tokens(contents=contents_to_count) # type: ignore[arg-type]
            return response.total_tokens
        except GenAIAPIError as e_sdk: # Catch SDK's base API error
            logger.error(f"Gemini API error during message token count for model '{model_name_for_api}': {e_sdk}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e_sdk}")
        except Exception as e: # Fallback for other errors
            logger.error(f"Failed to count message tokens for model '{model_name_for_api}' using Gemini API: {e}", exc_info=True)
            # Fallback to character approximation
            total_text_len = sum(len(msg.content) for msg in messages)
            if system_instruction_text: # Add system instruction length to approximation
                total_text_len += len(system_instruction_text)
            return (total_text_len + 3 * (len(messages) + (1 if system_instruction_text else 0))) // 4


    async def close(self) -> None:
        """Closes resources associated with the Gemini provider."""
        # The google-genai SDK's client does not typically require an explicit close method
        # as it manages HTTP sessions internally, often per-request or with a shared session
        # that's garbage collected.
        logger.debug("GeminiProvider closed (google-genai client typically does not require explicit close).")
        self._client = None # Dereference the client to allow GC
        pass
