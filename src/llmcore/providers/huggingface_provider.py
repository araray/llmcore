# src/llmcore/providers/huggingface_provider.py
"""
Hugging Face Inference API provider implementation for the LLMCore library.

Uses the ``huggingface_hub`` Python package (``AsyncInferenceClient``) to
interact with Hugging Face's Inference API, supporting:

- Chat completion (streaming + non-streaming, OpenAI-compatible format)
- Tool / function calling
- Vision (multimodal image + text inputs)
- Text-to-Speech (TTS)
- Speech-to-Text / Automatic Speech Recognition (STT / ASR)
- Embeddings (feature extraction)
- Image generation (text-to-image)
- Document Question Answering

The Inference API routes requests to models hosted on the HF Hub or to
third-party providers (Together, Sambanova, Replicate, etc.) via the HF
routing infrastructure.

Requires: ``pip install huggingface_hub>=0.28``
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

try:
    from huggingface_hub import AsyncInferenceClient
    from huggingface_hub.errors import (
        BadRequestError as HFBadRequestError,
    )
    from huggingface_hub.errors import (
        HfHubHTTPError,
    )
    from huggingface_hub.errors import (
        InferenceTimeoutError as HFTimeoutError,
    )
    from huggingface_hub.inference._generated.types import (
        ChatCompletionOutput,
        ChatCompletionStreamOutput,
    )

    huggingface_hub_available = True
except ImportError:
    huggingface_hub_available = False
    AsyncInferenceClient = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore
    HFBadRequestError = Exception  # type: ignore
    HFTimeoutError = Exception  # type: ignore
    ChatCompletionOutput = None  # type: ignore
    ChatCompletionStreamOutput = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from ..models_multimodal import (
    GeneratedImage,
    ImageGenerationResult,
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback context-length table for well-known HF-hosted models
# ---------------------------------------------------------------------------
DEFAULT_HF_TOKEN_LIMITS: dict[str, int] = {
    # Meta Llama
    "meta-llama/Llama-3.3-70B-Instruct": 131072,
    "meta-llama/Llama-3.1-70B-Instruct": 131072,
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
    "meta-llama/Meta-Llama-3-70B-Instruct": 8192,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": 131072,
    "meta-llama/Llama-3.2-3B-Instruct": 131072,
    "meta-llama/Llama-3.2-1B-Instruct": 131072,
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3": 32768,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "mistralai/Mistral-Small-24B-Instruct-2501": 32768,
    # Qwen
    "Qwen/Qwen2.5-72B-Instruct": 131072,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 131072,
    "Qwen/QwQ-32B": 131072,
    # Google
    "google/gemma-2-27b-it": 8192,
    "google/gemma-2-9b-it": 8192,
    # Microsoft
    "microsoft/Phi-3.5-mini-instruct": 131072,
    "microsoft/phi-4": 16384,
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 131072,
    "deepseek-ai/DeepSeek-V3-0324": 131072,
    # Embeddings
    "sentence-transformers/all-MiniLM-L6-v2": 512,
    "BAAI/bge-large-en-v1.5": 512,
    # TTS
    "facebook/mms-tts-eng": 4096,
    "OuteAI/OuteTTS-0.3-500M": 4096,
    # ASR
    "openai/whisper-large-v3": 448,
    "openai/whisper-large-v3-turbo": 448,
    # Image generation
    "stabilityai/stable-diffusion-xl-base-1.0": 77,
    "black-forest-labs/FLUX.1-dev": 512,
}

# Approximate tokens-per-character ratio for rough estimation
# when no tokenizer is available.
_APPROX_CHARS_PER_TOKEN = 4.0


class HuggingFaceProvider(BaseProvider):
    """LLMCore provider for the Hugging Face Inference API.

    Configuration keys (in ``[providers.huggingface]``):

    - ``api_key`` / ``api_key_env_var``: HF API token (``hf_...``)
    - ``default_model``: Default chat model ID (HF Hub model ID)
    - ``timeout``: Request timeout in seconds (default: 120)
    - ``provider``: HF routing provider (e.g. ``"hf-inference"``,
      ``"together"``, ``"sambanova"``)
    - ``bill_to``: HF organization to bill usage to
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads)

        if not huggingface_hub_available:
            raise ConfigError(
                "HuggingFace provider requires the 'huggingface_hub' package. "
                "Install with: pip install huggingface_hub>=0.28"
            )

        # Resolve API key
        self._api_key = config.get("api_key") or os.environ.get(
            config.get("api_key_env_var", "HF_TOKEN"), ""
        )
        if not self._api_key:
            # Also try the conventional HF env vars
            self._api_key = os.environ.get("HF_TOKEN", "") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN", ""
            )
        if not self._api_key:
            raise ConfigError(
                "HuggingFace provider requires an API key. "
                "Set HF_TOKEN environment variable or configure "
                "api_key / api_key_env_var in [providers.huggingface]."
            )

        self.default_model: str = config.get(
            "default_model", "meta-llama/Llama-3.3-70B-Instruct"
        )
        self._timeout: float = float(config.get("timeout", 120))
        self._hf_provider: str | None = config.get("provider")
        self._bill_to: str | None = config.get("bill_to")

        # Build the async inference client
        self._client = AsyncInferenceClient(
            token=self._api_key,
            timeout=self._timeout,
            provider=self._hf_provider,
            bill_to=self._bill_to,
        )

        # Cache for dynamically discovered context lengths
        self._discovered_context_lengths: dict[str, int] = {}

        logger.info(
            "HuggingFace provider initialized (default_model=%s, provider=%s)",
            self.default_model,
            self._hf_provider or "auto",
        )

    def get_name(self) -> str:
        if self._provider_instance_name:
            return self._provider_instance_name
        return "huggingface"

    async def get_models_details(self) -> list[ModelDetails]:
        """Return model details from the model card registry.

        HF Hub has hundreds of thousands of models; we return only the
        models for which we have model cards, rather than calling the
        full Hub listing API.
        """
        registry = get_model_card_registry()
        cards = registry.get_provider_cards("huggingface")
        results: list[ModelDetails] = []
        for card in cards.values():
            results.append(
                ModelDetails(
                    id=card.model_id,
                    provider_name="huggingface",
                    display_name=card.display_name or card.model_id,
                    context_length=card.get_context_length(),
                    max_output_tokens=card.context.get("max_output_tokens")
                    if card.context
                    else None,
                    supports_streaming=card.capabilities.streaming
                    if card.capabilities
                    else True,
                    supports_tools=card.capabilities.tool_calling
                    if card.capabilities
                    else False,
                    supports_vision=card.capabilities.vision
                    if card.capabilities
                    else False,
                    family=card.architecture.get("family")
                    if card.architecture
                    else None,
                    model_type=card.model_type or "chat",
                    metadata={"source": "model_card"},
                )
            )
        return results

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {
            "temperature": {"type": "number", "min": 0, "max": 2.0},
            "top_p": {"type": "number", "min": 0, "max": 1.0},
            "max_tokens": {"type": "integer", "min": 1},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "frequency_penalty": {"type": "number", "min": -2.0, "max": 2.0},
            "presence_penalty": {"type": "number", "min": -2.0, "max": 2.0},
            "top_k": {"type": "integer"},
            "response_format": {"type": "object"},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        model_name = model or self.default_model

        # 1. Check model card registry
        try:
            registry = get_model_card_registry()
            card = registry.get("huggingface", model_name)
            if card:
                ctx = card.get_context_length()
                if ctx and ctx > 0:
                    return ctx
        except Exception:
            pass

        # 2. Cached discovery
        if model_name in self._discovered_context_lengths:
            return self._discovered_context_lengths[model_name]

        # 3. Fallback table
        if model_name in DEFAULT_HF_TOKEN_LIMITS:
            return DEFAULT_HF_TOKEN_LIMITS[model_name]

        # 4. Default
        return 8192

    # ------------------------------------------------------------------
    # Message preparation (OpenAI-compatible format)
    # ------------------------------------------------------------------

    def _build_message_payload(self, msg: Message) -> dict[str, Any]:
        """Convert an llmcore Message into an OpenAI-compatible message dict."""
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        content: Any = msg.content

        # Handle multimodal content parts
        if "content_parts" in metadata:
            content = metadata["content_parts"]
        elif "inline_images" in metadata:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"type": "text", "text": msg.content})
            for img in metadata.get("inline_images", []):
                if isinstance(img, str):
                    parts.append(
                        {"type": "image_url", "image_url": {"url": img}}
                    )
                elif isinstance(img, dict):
                    img_p: dict[str, Any] = {"url": img.get("url", "")}
                    if "detail" in img:
                        img_p["detail"] = img["detail"]
                    parts.append({"type": "image_url", "image_url": img_p})
            content = parts

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        # Tool result messages
        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        # Assistant messages with tool calls
        if role_str == "assistant" and "tool_calls" in metadata:
            msg_dict["tool_calls"] = metadata["tool_calls"]
            if not msg.content:
                msg_dict["content"] = None

        # Optional name field
        name = metadata.get("name")
        if name:
            msg_dict["name"] = name

        return msg_dict

    # ------------------------------------------------------------------
    # Chat Completion
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        model_name = model or self.default_model

        # Validate context
        if not (
            isinstance(context, list)
            and all(isinstance(msg, Message) for msg in context)
        ):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        # Build messages
        messages_payload = [self._build_message_payload(msg) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        # Build tools payload
        tools_payload = None
        if tools:
            tools_payload = [
                {"type": "function", "function": t.model_dump()} for t in tools
            ]

        # Build API kwargs
        api_kwargs: dict[str, Any] = {}
        for k in (
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
        ):
            if k in kwargs:
                api_kwargs[k] = kwargs[k]

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM REQUEST (%s): model=%s, messages=%d, stream=%s, tools=%d",
                self.get_name(),
                model_name,
                len(messages_payload),
                stream,
                len(tools_payload) if tools_payload else 0,
            )

        try:
            response = await self._client.chat_completion(
                messages=messages_payload,
                model=model_name,
                stream=stream,
                tools=tools_payload,
                tool_choice=tool_choice,
                **api_kwargs,
            )

            if stream:
                return self._stream_wrapper(response)
            else:
                # ChatCompletionOutput extends dict, so we can use it directly.
                # But nested objects might not be plain dicts in all versions,
                # so we normalize via _normalize_response.
                response_dict = self._normalize_hf_response(response)

                if self.log_raw_payloads_enabled and logger.isEnabledFor(
                    logging.DEBUG
                ):
                    logger.debug(
                        "RAW LLM RESPONSE (%s): %s",
                        self.get_name(),
                        json.dumps(response_dict, indent=2, default=str),
                    )

                return response_dict

        except HFTimeoutError as e:
            raise ProviderError(
                self.get_name(), f"Request timed out: {e}"
            ) from e
        except HFBadRequestError as e:
            raise ProviderError(
                self.get_name(), f"Bad request: {e}"
            ) from e
        except HfHubHTTPError as e:
            raise ProviderError(
                self.get_name(), f"HTTP error: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                self.get_name(), f"Chat completion failed: {e}"
            ) from e

    async def _stream_wrapper(
        self, response: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Wrap HF streaming response into dicts for llmcore."""
        try:
            async for chunk in response:
                chunk_dict = self._normalize_hf_response(chunk)
                if self.log_raw_payloads_enabled and logger.isEnabledFor(
                    logging.DEBUG
                ):
                    logger.debug(
                        "RAW STREAM CHUNK (%s): %s",
                        self.get_name(),
                        json.dumps(chunk_dict, default=str),
                    )
                yield chunk_dict
        except HfHubHTTPError as e:
            raise ProviderError(
                self.get_name(), f"Stream error: {e}"
            ) from e

    def _normalize_hf_response(self, obj: Any) -> dict[str, Any]:
        """Recursively convert HF BaseInferenceType objects to plain dicts.

        HF's generated types extend ``dict`` and call
        ``self.update(asdict(self))`` in ``__post_init__``, so they
        should already be dict-like.  But nested dataclass fields may
        still be ``BaseInferenceType`` instances in some SDK versions.
        This method ensures we get a fully plain dict tree.
        """
        if isinstance(obj, dict):
            return {k: self._normalize_hf_response(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._normalize_hf_response(item) for item in obj]
        else:
            return obj

    # ------------------------------------------------------------------
    # Response extraction (OpenAI-compatible format)
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to extract content: %s", e)
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from HF response (OpenAI-compatible format)."""
        out: list[ToolCall] = []
        try:
            choices = response.get("choices", [])
            if not choices:
                return out
            raw_calls = choices[0].get("message", {}).get("tool_calls")
            if not raw_calls:
                return out
            for tc in raw_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    if isinstance(args_str, str):
                        args_dict = json.loads(args_str)
                    elif isinstance(args_str, dict):
                        args_dict = args_str
                    else:
                        args_dict = {"_raw": str(args_str)}
                except (json.JSONDecodeError, TypeError):
                    args_dict = {"_raw": str(args_str)}
                out.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        arguments=args_dict,
                    )
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to extract tool calls: %s", e)
        return out

    # ------------------------------------------------------------------
    # Token counting (approximate)
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Approximate token count using character-based heuristic.

        HF models use many different tokenizers; for accurate counting
        the model's specific tokenizer would need to be loaded locally.
        This heuristic gives a reasonable upper-bound estimate.
        """
        return max(1, int(len(text) / _APPROX_CHARS_PER_TOKEN))

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        total = 0
        for msg in messages:
            # Role overhead ~4 tokens
            total += 4
            if msg.content:
                total += await self.count_tokens(msg.content, model)
            metadata = msg.metadata or {}
            if "content_parts" in metadata:
                for part in metadata["content_parts"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += await self.count_tokens(
                            part.get("text", ""), model
                        )
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        total += 1000  # Rough image token estimate
        # Message framing overhead
        total += 3
        return total

    # ------------------------------------------------------------------
    # Multimodal: Text-to-Speech (TTS)
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str = "alloy",
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> SpeechResult:
        """Generate speech audio from text via HF Inference API."""
        tts_model = model or "facebook/mms-tts-eng"

        try:
            audio_bytes: bytes = await self._client.text_to_speech(
                text=text,
                model=tts_model,
                **kwargs,
            )
            return SpeechResult(
                audio_data=audio_bytes,
                format=response_format,
                model=tts_model,
                voice=voice,
                metadata={"provider": "huggingface"},
            )
        except HfHubHTTPError as e:
            raise ProviderError(
                self.get_name(), f"TTS failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Multimodal: Speech-to-Text (STT / ASR)
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_data: bytes | str,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe audio to text via HF Inference API (ASR)."""
        asr_model = model or "openai/whisper-large-v3-turbo"

        try:
            result = await self._client.automatic_speech_recognition(
                audio=audio_data,
                model=asr_model,
            )

            # Build segments from chunks if available
            segments: list[TranscriptionSegment] = []
            if hasattr(result, "chunks") and result.chunks:
                for chunk in result.chunks:
                    ts = chunk.get("timestamp", [0.0, 0.0]) if isinstance(chunk, dict) else [0.0, 0.0]
                    text_val = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    segments.append(
                        TranscriptionSegment(
                            text=text_val,
                            start=ts[0] if len(ts) > 0 else 0.0,
                            end=ts[1] if len(ts) > 1 else 0.0,
                        )
                    )

            return TranscriptionResult(
                text=result.text if hasattr(result, "text") else str(result),
                language=language,
                segments=segments,
                model=asr_model,
                metadata={"provider": "huggingface"},
            )
        except HfHubHTTPError as e:
            raise ProviderError(
                self.get_name(), f"ASR failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Multimodal: Image Generation
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        response_format: str = "b64_json",
        style: str | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate images from text via HF Inference API."""
        img_model = model or "black-forest-labs/FLUX.1-dev"

        # Parse size into dimensions
        width, height = 1024, 1024
        if size:
            try:
                parts = size.split("x")
                width, height = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass

        try:
            import base64
            from io import BytesIO

            images: list[GeneratedImage] = []
            for _ in range(n):
                img = await self._client.text_to_image(
                    prompt=prompt,
                    model=img_model,
                    width=width,
                    height=height,
                    **kwargs,
                )
                # HF returns a PIL Image; convert to base64
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                images.append(
                    GeneratedImage(data=b64_data, format="png")
                )

            return ImageGenerationResult(
                images=images,
                model=img_model,
                metadata={"provider": "huggingface"},
            )
        except HfHubHTTPError as e:
            raise ProviderError(
                self.get_name(), f"Image generation failed: {e}"
            ) from e
        except ImportError as e:
            raise ProviderError(
                self.get_name(),
                f"Image generation requires PIL: {e}",
            ) from e

    # ------------------------------------------------------------------
    # Embeddings (Feature Extraction)
    # ------------------------------------------------------------------

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create text embeddings via HF Inference API (feature extraction)."""
        embed_model = model or "sentence-transformers/all-MiniLM-L6-v2"
        texts = [input_texts] if isinstance(input_texts, str) else input_texts

        try:
            data_items = []
            for idx, text in enumerate(texts):
                embedding = await self._client.feature_extraction(
                    text=text,
                    model=embed_model,
                )
                # HF returns numpy array; convert to list
                emb_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                # Flatten if nested (HF sometimes returns [1, dim] shape)
                if (
                    isinstance(emb_list, list)
                    and len(emb_list) > 0
                    and isinstance(emb_list[0], list)
                ):
                    emb_list = emb_list[0]
                data_items.append(
                    {
                        "object": "embedding",
                        "embedding": emb_list,
                        "index": idx,
                    }
                )

            total_tokens = 0
            for t in texts:
                total_tokens += await self.count_tokens(t)

            return {
                "object": "list",
                "data": data_items,
                "model": embed_model,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                },
            }
        except HfHubHTTPError as e:
            raise ProviderError(
                self.get_name(), f"Embedding failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug("Error closing HF client: %s", e)
