import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# --- Granular import checks for google-genai and its dependencies ---
google_genai_available = False
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    from google.api_core.exceptions import PermissionDenied, InvalidArgument
    google_genai_available = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    APIError = Exception  # type: ignore
    PermissionDenied = Exception  # type: ignore
    InvalidArgument = Exception  # type: ignore

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..models import Message, ModelDetails, Tool
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

DEFAULT_GEMINI_TOKEN_LIMITS = {
    "gemini-1.5-flash-latest": 1048576,
    "gemini-1.5-pro-latest": 1048576,
    "gemini-1.0-pro": 30720,
    "gemini-ultra": 2097152,
}
DEFAULT_MODEL = "gemini-1.5-flash-latest"

LLMCORE_TO_GEMINI_ROLE_MAP = {
    LLMCoreRole.USER: "user",
    LLMCoreRole.ASSISTANT: "model",
}


class GeminiProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Google Gemini API using google-genai.
    Handles List[Message] context type and standardized tool-calling.
    """
    _client: Optional[Any] = None  # Use Any for genai.Client
    _api_key_env_var: Optional[str] = None
    _safety_settings: Optional[List[Dict[str, Any]]] = None

    def __init__(self, config: Dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the GeminiProvider using the google-genai SDK.

        Args:
            config: Configuration dictionary from `[providers.gemini]` containing:
                    'api_key' (optional): Google AI API key.
                    'api_key_env_var' (optional): Environment variable to read the API key from.
                    'default_model' (optional): Default model to use.
                    'safety_settings' (optional): Dictionary for configuring safety settings.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not google_genai_available:
            raise ImportError("Google Gen AI library (`google-genai`) not installed. Install with 'pip install llmcore[gemini]'.")

        self._api_key_env_var = config.get('api_key_env_var')
        api_key = config.get('api_key')
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            api_key = os.environ.get('GOOGLE_API_KEY')

        self.api_key = api_key
        self.default_model = config.get('default_model', DEFAULT_MODEL)
        self.fallback_context_length = int(config.get('fallback_context_length', 1048576))
        self._safety_settings = self._parse_safety_settings(config.get('safety_settings'))

        if not self.api_key:
            logger.warning("Google API key not found. Ensure it is set if not using other auth methods.")

        try:
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Google Gen AI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client: {e}", exc_info=True)
            raise ConfigError(f"Google Gen AI configuration failed: {e}")

    def _parse_safety_settings(self, settings_config: Optional[Dict[str, str]]) -> Optional[List[Dict[str, Any]]]:
        """Parses safety settings from config into the format expected by the SDK."""
        if not settings_config:
            return None
        parsed_settings = []
        for key, value in settings_config.items():
            try:
                # Basic validation, SDK will do the rest
                category = types.HarmCategory[key.upper()]
                threshold = types.HarmBlockThreshold[value.upper()]
                parsed_settings.append({"category": category, "threshold": threshold})
            except (KeyError, AttributeError):
                logger.warning(f"Invalid safety setting: {key}={value}. Skipping.")
        return parsed_settings if parsed_settings else None

    def get_name(self) -> str:
        """Returns the provider name: 'gemini'."""
        return "gemini"

    async def get_models_details(self) -> List[ModelDetails]:
        """Dynamically discovers available models from the Google AI API."""
        details_list = []
        try:
            # The google-genai library doesn't have an async list_models, so we run sync in thread
            models = await asyncio.to_thread(genai.list_models)
            for model in models:
                # Check if the model supports the 'generateContent' method
                if 'generateContent' in model.supported_generation_methods:
                    details = ModelDetails(
                        id=model.name.replace("models/", ""),
                        context_length=model.input_token_limit,
                        supports_streaming=True,
                        supports_tools='functionCalling' in model.supported_generation_methods,
                        provider_name=self.get_name(),
                        metadata={"display_name": model.display_name, "version": model.version}
                    )
                    details_list.append(details)
            logger.info(f"Discovered {len(details_list)} supported models from Google AI.")
        except Exception as e:
            logger.error(f"Failed to list models from Google AI: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Failed to list models: {e}")
        return details_list

    def get_supported_parameters(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Returns a schema of supported GenerationConfig parameters for Gemini."""
        return {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer"},
            "candidate_count": {"type": "integer"},
            "max_output_tokens": {"type": "integer"},
            "stop_sequences": {"type": "array", "items": {"type": "string"}},
            "response_mime_type": {"type": "string"},
        }

    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """Returns the maximum context length (tokens) for the given Gemini model."""
        model_name = model or self.default_model
        limit = DEFAULT_GEMINI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            limit = self.fallback_context_length
            logger.warning(f"Unknown context length for Gemini model '{model_name}'. Using fallback: {limit}.")
        return limit

    def _convert_llmcore_msgs_to_genai_contents(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Converts LLMCore messages to Gemini's `contents` format."""
        genai_history = []
        system_instruction_text = None

        processed_messages = list(messages)
        if processed_messages and processed_messages[0].role == LLMCoreRole.SYSTEM:
            system_instruction_text = processed_messages.pop(0).content

        last_role = None
        for msg in processed_messages:
            genai_role = LLMCORE_TO_GEMINI_ROLE_MAP.get(msg.role)
            if not genai_role:
                continue

            if genai_role == last_role:
                if genai_history:
                    genai_history[-1]['parts'][-1]['text'] += f"\n{msg.content}"
                    continue

            genai_history.append({"role": genai_role, "parts": [{"text": msg.content}]})
            last_role = genai_role

        return genai_history, system_instruction_text

    async def chat_completion(
        self,
        context: ContextPayload,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Sends a chat completion request to the Google Gemini API."""
        if not self._client:
            raise ProviderError(self.get_name(), "Gemini client not initialized.")

        supported_params = self.get_supported_parameters()
        for key in kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for Gemini provider.")

        model_name = model or self.default_model

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        genai_contents, system_instruction_text = self._convert_llmcore_msgs_to_genai_contents(context)
        if not genai_contents:
            raise ProviderError(self.get_name(), "No valid messages to send.")

        generation_config = types.GenerationConfig(**kwargs) if kwargs else None
        if system_instruction_text:
            generation_config.system_instruction = system_instruction_text

        function_declarations = [types.FunctionDeclaration.from_dict(tool.model_dump()) for tool in tools] if tools else None
        tools_payload = [types.Tool(function_declarations=function_declarations)] if function_declarations else None

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {"model": model_name, "contents": genai_contents, "stream": stream, "generation_config": kwargs, "tools": tools_payload}
            logger.debug(f"RAW LLM REQUEST ({self.get_name()}): {json.dumps(log_data, indent=2, default=str)}")

        try:
            if stream:
                response = await self._client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=genai_contents,
                    config=generation_config,
                    tools=tools_payload,
                    safety_settings=self._safety_settings
                )
                async def stream_wrapper():
                    async for chunk in response:
                        if self.log_raw_payloads_enabled:
                            logger.debug(f"RAW LLM STREAM CHUNK ({self.get_name()}): {chunk}")
                        yield {"choices": [{"delta": {"content": chunk.text}}]}
                return stream_wrapper()
            else:
                response = await self._client.aio.models.generate_content(
                    model=model_name,
                    contents=genai_contents,
                    config=generation_config,
                    tools=tools_payload,
                    safety_settings=self._safety_settings
                )
                response_dict = {
                    "id": None,  # Gemini API doesn't provide a top-level ID in this response
                    "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": response.text}, "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None}],
                    "usage": {"prompt_token_count": response.usage_metadata.prompt_token_count, "candidates_token_count": response.usage_metadata.candidates_token_count, "total_token_count": response.usage_metadata.total_token_count}
                }
                if self.log_raw_payloads_enabled:
                    logger.debug(f"RAW LLM RESPONSE ({self.get_name()}): {json.dumps(response_dict, indent=2)}")
                return response_dict
        except (APIError, InvalidArgument) as e:
            logger.error(f"Google AI API error: {e}", exc_info=True)
            if "context length" in str(e).lower():
                raise ContextLengthError(model_name=model_name, message=str(e))
            raise ProviderError(self.get_name(), f"Google AI API Error: {e}")
        except PermissionDenied as e:
            logger.error(f"Permission denied during Gemini chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Permission denied: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Gemini chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts tokens for a text string using the Gemini API."""
        if not self._client:
            logger.warning("Gemini client not available. Approximating token count.")
            return (len(text) + 3) // 4
        if not text:
            return 0

        target_model = model or self.default_model
        try:
            response = await self._client.aio.models.count_tokens(contents=[text], model=target_model)
            return response.total_tokens
        except Exception as e:
            logger.error(f"Failed to count tokens with Gemini API for model '{target_model}': {e}", exc_info=True)
            return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: List[Message], model: Optional[str] = None) -> int:
        """Counts tokens for a list of messages using the Gemini API."""
        if not self._client:
            logger.warning("Gemini client not available. Approximating message token count.")
            total_chars = sum(len(msg.content) for msg in messages)
            return (total_chars + 3 * len(messages)) // 4
        if not messages:
            return 0

        target_model = model or self.default_model
        genai_contents, _ = self._convert_llmcore_msgs_to_genai_contents(messages)
        try:
            response = await self._client.aio.models.count_tokens(contents=genai_contents, model=target_model)
            return response.total_tokens
        except Exception as e:
            logger.error(f"Failed to count message tokens with Gemini API for model '{target_model}': {e}", exc_info=True)
            total_chars = sum(len(part['text']) for content_dict in genai_contents for part in content_dict.get('parts',[]))
            return (total_chars + 3 * len(genai_contents)) // 4

    async def close(self) -> None:
        """The google-genai library does not require explicit client closing."""
        logger.debug("GeminiProvider closed (no explicit client cleanup needed).")
        self._client = None
