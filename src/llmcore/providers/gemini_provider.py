# src/llmcore/providers/gemini_provider.py
"""
Google Gemini API provider implementation for the LLMCore library.

Handles interactions with the Google Generative AI API (Gemini models).
Uses the official 'google-generativeai' library.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple

# Use the official google-generativeai library
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, ContentDict, HarmCategory, HarmBlockThreshold
    # Import specific exceptions if available, otherwise use a general one
    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        google_exceptions = None # type: ignore
    gemini_available = True
except ImportError:
    gemini_available = False
    genai = None # type: ignore
    GenerationConfig = None # type: ignore
    ContentDict = Dict[str, Any] # type: ignore
    HarmCategory = None # type: ignore
    HarmBlockThreshold = None # type: ignore
    google_exceptions = None # type: ignore

from ..models import Message, Role
from ..exceptions import ProviderError, ConfigError
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# Default context lengths for common Gemini models
# Source: https://ai.google.dev/models/gemini
# Note: These often represent the *total* input+output limit.
# The effective input limit might be slightly smaller.
DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-1.5-pro-latest": 1048576, # 1M context window (can be up to 2M)
    "gemini-1.5-flash-latest": 1048576, # 1M context window
    "gemini-1.0-pro": 32768, # 32k context (input: 30720 tokens, output: 2048 tokens)
    "gemini-pro": 32768, # Alias for 1.0 pro
    "gemini-1.0-pro-vision-latest": 16384, # 12k text + 4k image/video
    "gemini-pro-vision": 16384, # Alias
    # Older models might have different limits
}

# Default model if not specified in config
DEFAULT_MODEL = "gemini-1.5-pro-latest"

# Mapping from LLMCore Role to Gemini Role
GEMINI_ROLE_MAP = {
    Role.USER: "user",
    Role.ASSISTANT: "model",
    # System role is handled separately in Gemini API
}

class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API.
    """
    _model_instance_cache: Dict[str, Any] = {} # Cache for GenerativeModel instances

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiProvider.

        Args:
            config: Configuration dictionary containing:
                    'api_key' (optional): Google AI API key. Defaults to env var GOOGLE_API_KEY.
                    'default_model' (optional): Default model (e.g., "gemini-1.5-pro-latest").
                    'timeout' (optional): Request timeout (not directly used by genai client, handled internally).
                    'safety_settings' (optional): Dictionary for content safety settings.
        """
        if not gemini_available:
            raise ImportError("Google Generative AI library not installed. Please install `google-generativeai`.")

        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        # Timeout is generally handled by the underlying library's transport layer
        self.timeout = config.get('timeout') # Store if needed for future use
        self.safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key:
            logger.warning("Google AI API key not found in config or environment variable GOOGLE_API_KEY. "
                           "Initialization will likely fail if authentication is required.")
            # Let the genai library handle the error later if the key is truly needed

        try:
            genai.configure(api_key=self.api_key)
            logger.debug("Google Generative AI client configured.")
        except Exception as e:
            logger.error(f"Failed to configure Google Generative AI client: {e}", exc_info=True)
            # Don't raise here, allow potential use without API key for some scenarios?
            # Or raise ConfigError? Let's raise for clarity.
            raise ConfigError(f"Google AI client configuration failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[Dict[Any, Any]]:
        """Parses safety settings from config string values to enum values."""
        if not settings_config or not HarmCategory or not HarmBlockThreshold:
            return None

        parsed_settings = {}
        for key, value in settings_config.items():
            try:
                category = getattr(HarmCategory, key.upper())
                threshold = getattr(HarmBlockThreshold, value.upper())
                parsed_settings[category] = threshold
            except AttributeError:
                logger.warning(f"Invalid safety setting category or threshold: {key}={value}. Skipping.")
        logger.debug(f"Parsed safety settings: {parsed_settings}")
        return parsed_settings if parsed_settings else None

    def _get_model_instance(self, model_name: str) -> Any:
        """Gets or creates a GenerativeModel instance, caching it."""
        if model_name in self._model_instance_cache:
            return self._model_instance_cache[model_name]

        if not genai: # Check again in case of import issues
             raise ProviderError(self.get_name(), "Google Generative AI library not available.")

        try:
            # System instruction is handled separately during content generation
            model_instance = genai.GenerativeModel(model_name)
            self._model_instance_cache[model_name] = model_instance
            logger.debug(f"Created GenerativeModel instance for '{model_name}'")
            return model_instance
        except Exception as e:
            logger.error(f"Failed to initialize GenerativeModel for '{model_name}': {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Failed to get model instance '{model_name}': {e}")

    def get_name(self) -> str:
        """Returns the provider name."""
        return "gemini"

    def get_available_models(self) -> List[str]:
        """
        Returns a static list of known default models for Gemini.
        Note: Does not dynamically fetch from the API.
        """
        # TODO: Implement dynamic fetching via API (genai.list_models()) if needed, requires async handling
        logger.warning("GeminiProvider.get_available_models() returning static list.")
        return list(DEFAULT_GEMINI_TOKEN_LIMITS.keys())

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length for the given Gemini model."""
        model_name = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Attempt to parse model name for known prefixes
            if model_name.startswith("gemini-1.5"): limit = 1048576
            elif model_name.startswith("gemini-1.0-pro-vision"): limit = 16384
            elif model_name.startswith("gemini-1.0-pro") or model_name == "gemini-pro": limit = 32768
            else:
                limit = 32768 # Fallback default (for 1.0 pro)
                logger.warning(f"Unknown context length for Gemini model '{model_name}'. Using fallback limit: {limit}.")
        # Gemini limits often include output tokens. We return the advertised limit,
        # but context management should reserve space.
        return limit

    def _convert_context_to_gemini(self, messages: List[Message]) -> Tuple[Optional[str], List[ContentDict]]:
        """
        Converts LLMCore messages to Gemini's ContentDict format.
        Separates the system prompt. Handles alternating user/model roles.
        """
        gemini_history: List[ContentDict] = []
        system_prompt: Optional[str] = None

        # Extract system prompt first
        if messages and messages[0].role == Role.SYSTEM:
            system_prompt = messages[0].content
            messages = messages[1:] # Remove system prompt from main history

        last_role = None
        for msg in messages:
            gemini_role = GEMINI_ROLE_MAP.get(msg.role)
            if not gemini_role:
                logger.warning(f"Skipping message with unmappable role for Gemini: {msg.role}")
                continue

            # Ensure alternating roles (user -> model -> user ...)
            if gemini_role == last_role:
                # Merge consecutive messages of the same role (heuristic)
                if gemini_history:
                    logger.debug(f"Merging consecutive Gemini '{gemini_role}' messages.")
                    gemini_history[-1]['parts'].append({"text": msg.content}) # type: ignore
                else:
                    # Cannot start with 'model' role if history is empty
                    if gemini_role == "model":
                         logger.warning("Skipping initial 'model' role message in Gemini history.")
                         continue
                    else: # Start with user message
                         gemini_history.append({"role": gemini_role, "parts": [{"text": msg.content}]})
                         last_role = gemini_role
            else:
                # Add new message with the correct role
                gemini_history.append({"role": gemini_role, "parts": [{"text": msg.content}]})
                last_role = gemini_role

        # Gemini API requires the conversation to end with a 'user' role message
        # If it ends with 'model', we might need to drop the last model message or handle differently.
        # For now, let the API handle potential errors if the turn structure is invalid.
        if gemini_history and gemini_history[-1]['role'] == 'model':
            logger.warning("Gemini conversation history ends with 'model' role. This might cause issues.")

        return system_prompt, gemini_history

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Sends a chat completion request to the Gemini API."""
        if not genai:
            raise ProviderError(self.get_name(), "Google Generative AI library not available.")
        if not isinstance(context, list):
            raise ProviderError(self.get_name(), f"GeminiProvider received unsupported context type: {type(context).__name__}")

        model_name = model or self.default_model
        model_instance = self._get_model_instance(model_name)

        system_prompt, gemini_history = self._convert_context_to_gemini(context)

        logger.debug(f"Sending request to Gemini API: model='{model_name}', stream={stream}, num_history={len(gemini_history)}")

        # Prepare GenerationConfig
        gen_config_args = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_output_tokens": kwargs.get("max_tokens"), # Map max_tokens
            "stop_sequences": kwargs.get("stop"),
            # candidate_count is not directly mapped from typical kwargs
        }
        # Filter out None values
        gen_config_args = {k: v for k, v in gen_config_args.items() if v is not None}
        generation_config = GenerationConfig(**gen_config_args) if gen_config_args else None

        try:
            # Handle system prompt separately if provided
            model_to_use = model_instance
            if system_prompt:
                 model_to_use = genai.GenerativeModel(
                     model_name,
                     system_instruction=system_prompt
                 )
                 logger.debug("Applied system prompt to Gemini model instance.")

            # Start the chat session (history is passed directly)
            # Note: The library manages history internally if you use chat.send_message,
            # but we pass the full history for stateless operation per call.
            response_or_iterator = await model_to_use.generate_content_async(
                contents=gemini_history,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                stream=stream,
            )

            if stream:
                logger.debug(f"Processing stream response from Gemini model '{model_name}'")
                # Wrap the stream to yield dictionaries
                async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
                    full_response_text = ""
                    try:
                        async for chunk in response_or_iterator: # type: ignore
                            # Extract text and potentially other info from the chunk
                            # Chunk structure: GenerateContentResponse(text='...', candidates=[...], prompt_feedback=...)
                            chunk_text = ""
                            try:
                                # Accessing chunk.text directly is often the simplest way
                                chunk_text = chunk.text
                            except ValueError as e:
                                # Handle potential errors like blocked content within the stream
                                logger.warning(f"ValueError accessing chunk text (content might be blocked): {e}. Chunk: {chunk}")
                                # Check for finish reason due to safety
                                finish_reason = chunk.prompt_feedback.block_reason if chunk.prompt_feedback else None
                                if finish_reason:
                                     yield {"error": f"Stream stopped due to safety settings: {finish_reason}", "finish_reason": "SAFETY"}
                                     return # Stop the stream
                                else:
                                     yield {"error": f"Error accessing stream chunk content: {e}"}
                                     continue # Skip this chunk
                            except Exception as e:
                                logger.error(f"Unexpected error processing stream chunk: {e}. Chunk: {chunk}", exc_info=True)
                                yield {"error": f"Unexpected stream error: {e}"}
                                continue # Skip this chunk


                            full_response_text += chunk_text
                            # Yield a dictionary similar to other providers' stream chunks
                            yield {
                                "model": model_name,
                                # Mimic OpenAI/Ollama stream structure where possible
                                "message": {"role": "model", "content": chunk_text},
                                "choices": [{"delta": {"content": chunk_text}, "index": 0}], # Mimic OpenAI
                                "done": False, # Assume not done unless finish reason appears
                                # Include finish reason if available in the chunk (usually at the end)
                                "finish_reason": getattr(chunk, 'candidates', [{}])[0].get('finish_reason', None) if hasattr(chunk, 'candidates') else None,
                                # TODO: Add token counts if available in the chunk
                            }
                    except google_exceptions.GoogleAPIError as e:
                        logger.error(f"Gemini API error during stream: {e}", exc_info=True)
                        yield {"error": f"Gemini API Error: {e}", "done": True}
                        raise ProviderError(self.get_name(), f"API Error during stream: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error processing Gemini stream: {e}", exc_info=True)
                        yield {"error": f"Unexpected stream processing error: {e}", "done": True}
                        # Don't raise here, allow potential cleanup in calling code
                    finally:
                         # Yield a final "done" message if needed, although finish_reason might cover this
                         logger.debug("Gemini stream finished.")
                         # The library might handle the final state implicitly.
                         # We could yield a final dict with done=True if necessary.
                         # yield {"done": True, "full_content": full_response_text}
                         pass

                return stream_wrapper()
            else:
                logger.debug(f"Processing non-stream response from Gemini model '{model_name}'")
                # Handle potential errors in the full response
                try:
                    # Accessing response.text raises ValueError if content is blocked
                    full_text = response_or_iterator.text # type: ignore
                except ValueError as e:
                     logger.warning(f"Content blocked in Gemini response: {e}. Response: {response_or_iterator}")
                     finish_reason = response_or_iterator.prompt_feedback.block_reason if response_or_iterator.prompt_feedback else "BLOCKED" # type: ignore
                     # Return an error structure or raise an exception
                     raise ProviderError(self.get_name(), f"Content generation stopped due to safety settings: {finish_reason}")
                except Exception as e:
                     logger.error(f"Error accessing Gemini response content: {e}. Response: {response_or_iterator}", exc_info=True)
                     raise ProviderError(self.get_name(), f"Failed to extract content from response: {e}")


                # Construct a dictionary similar to other providers' non-streaming output
                # Extract usage metadata if available
                usage_metadata = None
                if hasattr(response_or_iterator, 'usage_metadata'):
                     usage_metadata = { # type: ignore
                          "prompt_token_count": response_or_iterator.usage_metadata.prompt_token_count, # type: ignore
                          "candidates_token_count": response_or_iterator.usage_metadata.candidates_token_count, # type: ignore
                          "total_token_count": response_or_iterator.usage_metadata.total_token_count, # type: ignore
                     }

                result_dict = {
                    "model": model_name,
                    # Mimic OpenAI structure
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "model", # Gemini uses 'model'
                            "content": full_text,
                        },
                        "finish_reason": getattr(response_or_iterator.candidates[0], 'finish_reason', None) if response_or_iterator.candidates else None, # type: ignore
                        # TODO: Add safety ratings if needed: response.candidates[0].safety_ratings
                    }],
                    "usage": usage_metadata,
                    # Include prompt feedback if relevant
                    "prompt_feedback": {"block_reason": response_or_iterator.prompt_feedback.block_reason} if response_or_iterator.prompt_feedback.block_reason else None, # type: ignore
                }
                return result_dict

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error: {e}")
        except asyncio.TimeoutError: # Catch timeout if applicable at this level
            logger.error(f"Request to Gemini timed out.")
            raise ProviderError(self.get_name(), f"Request timed out.")
        except Exception as e:
            logger.error(f"Unexpected error during Gemini chat completion: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens using the Gemini API."""
        if not genai: return 0 # Library not available
        if not text: return 0

        model_name = model or self.default_model
        model_instance = self._get_model_instance(model_name)
        try:
            # Use count_tokens method from the model instance
            response = model_instance.count_tokens(text)
            return response.total_tokens
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API error during token count for model '{model_name}': {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during token count: {e}")
        except Exception as e:
            logger.error(f"Failed to count tokens for model '{model_name}': {e}", exc_info=True)
            # Fallback approximation (less accurate)
            return (len(text) + 3) // 4

    def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of messages using the Gemini API."""
        if not genai: return 0
        if not messages: return 0

        model_name = model or self.default_model
        model_instance = self._get_model_instance(model_name)

        # Convert messages to the format expected by count_tokens (list of strings or ContentDict)
        # Using ContentDict structure is more accurate as it reflects roles.
        system_prompt, gemini_history = self._convert_context_to_gemini(messages)

        # The count_tokens method can take the history structure
        contents_to_count = []
        if system_prompt:
             # System prompt counting might need specific handling or is implicitly included
             # For now, let's count it separately and add.
             # contents_to_count.append(system_prompt) # count_tokens might not accept system prompt directly
             pass
        contents_to_count.extend(gemini_history)


        try:
            # Count tokens for the history
            response = model_instance.count_tokens(contents_to_count)
            total_tokens = response.total_tokens

            # Add tokens for system prompt separately if it wasn't included above
            if system_prompt:
                 system_token_response = model_instance.count_tokens(system_prompt)
                 total_tokens += system_token_response.total_tokens
                 # Add a small overhead for system prompt integration if needed (heuristic)
                 total_tokens += 5

            return total_tokens
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API error during message token count for model '{model_name}': {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error during message token count: {e}")
        except Exception as e:
            logger.error(f"Failed to count message tokens for model '{model_name}': {e}", exc_info=True)
            # Fallback approximation
            total_text = " ".join([msg.content for msg in messages])
            return (len(total_text) + 3 * len(messages)) // 4

    async def close(self) -> None:
        """Closes resources associated with the Gemini provider."""
        # The google-generativeai library doesn't seem to require explicit client closing.
        # Connections are typically managed by the underlying transport (like gRPC or HTTP).
        logger.debug("GeminiProvider closed (no specific client cleanup needed).")
        self._model_instance_cache.clear() # Clear model cache
        pass
