# src/llmcore/providers/ollama_provider.py
"""
Ollama provider implementation for the LLMCore library using the official ollama library.

Interacts with a local Ollama instance.
Supports streaming and different API endpoints (/api/chat, /api/generate).
Accepts context as List[Message].

Audited against ollama-python 0.6.3 and a live server (0.32.14) on
2026-11-16:

- The server loads a model with ITS default context (4096 on this 6 GiB GPU;
  docs: 4k under 24 GiB VRAM, 32k under 48, 256k above), not the trained
  window, and silently truncates longer prompts. ``num_ctx`` in config is sent
  with every request; without it, the effective window is read from
  ``/api/ps`` and :meth:`get_max_context_length` reports that.
- Tool loops: assistant turns are replayed with their ``tool_calls`` and
  ``thinking`` (both were dropped), tool results carry ``tool_name``
  (resolved from the call id), and tool calls keep the server's ``id``.
- ``effort`` maps to ``think`` (false for "none").

Supports:
- Chat completion (text, streaming)
- Tool/function calling with normalized extraction
- Multimodal content (vision: images via metadata["inline_images"])
- Thinking mode (think parameter with level support)
- Structured output (format parameter: 'json' or JSON Schema)
- Logprobs
- keep_alive control
- Dynamic model discovery with capabilities detection
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

# Use the official ollama library
try:
    import ollama
    from ollama import AsyncClient, ChatResponse, ResponseError, ShowResponse

    ollama_available = True
except ImportError:
    ollama_available = False
    AsyncClient = None  # type: ignore
    ResponseError = Exception  # type: ignore
    ChatResponse = None  # type: ignore
    ShowResponse = None  # type: ignore

# Keep tiktoken for token counting
try:
    import tiktoken

    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool, ToolCall
from ..tokens import EstimateCounter as _EstimateCounter
from .base import BaseProvider, ContextPayload

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover
    httpx_available = False
    httpx = None  # type: ignore

logger = logging.getLogger(__name__)

# Default context lengths for common Ollama models.
# Used as first-tier fallback before model card registry and 4096 default.
DEFAULT_OLLAMA_TOKEN_LIMITS = {
    "llama3": 8192,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "llama3.2": 131072,
    "llama3.2:latest": 131072,
    "llama3.3": 131072,
    "llama3.3:70b": 131072,
    # qwen3:32b reports 40960 (live server, 2026-11-16); the 262144 here was
    # wrong. Other qwen3 tags are left to their cards.
    "qwen3": 40960,
    "qwen3:32b": 40960,
    "qwen3.5": 262144,
    "qwen2.5": 131072,
    "qwen2.5:latest": 131072,
    "gemma3": 131072,
    "gemma3:4b": 131072,
    "gemma3:12b": 131072,
    "gemma3:27b": 131072,
    "falcon3:3b": 8000,
    "gemma:latest": 8000,
    "gemma:7b": 8000,
    "gemma:2b": 8000,
    "mistral": 8000,
    "mistral:7b": 8000,
    "mixtral": 32000,
    "mixtral:8x7b": 32000,
    "phi3": 4000,
    "phi3:mini": 4000,
    "phi4": 16384,
    "phi4:latest": 16384,
    "deepseek-r1": 131072,
    "deepseek-r1:7b": 131072,
    "deepseek-r1:8b": 131072,
    "deepseek-r1:14b": 131072,
    "codellama": 16000,
    "codellama:7b": 16000,
    "codellama:13b": 16000,
    "codellama:34b": 16000,
    "llama2": 4000,
    "llama2:7b": 4000,
    "llama2:13b": 4000,
    "llama2:70b": 4000,
}
#: A small current model with tools, thinking and vision that fits a 6 GiB GPU.
#: (gemma3:4b and llama3, the old code/config defaults, have no tool support.)
DEFAULT_MODEL = "qwen3.5:4b"

# Top-level chat parameters that go directly into the chat() call,
# NOT inside the ``options`` dict.  Everything else goes into ``options``.
_TOP_LEVEL_CHAT_PARAMS = frozenset(
    {
        "format",
        "keep_alive",
        "think",
        "logprobs",
        "top_logprobs",
    }
)


class OllamaProvider(BaseProvider):
    """
    LLMCore provider for interacting with Ollama using the official ollama library.
    Handles List[Message] context type and standardized tool-calling.
    """

    _client: AsyncClient | None = None
    _encoding: Any | None = None
    tokenizer_name: str

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the OllamaProvider using the official ollama library.

        Args:
            config: Configuration dictionary from ``[providers.ollama]`` containing:
                    'host' (optional): Host URL for the Ollama server.
                    'default_model' (optional): Default Ollama model to use.
                    'timeout' (optional): Request timeout in seconds.
                    'tokenizer' (optional): Tokenizer to use for estimations.
                    'keep_alive' (optional): Default keep_alive duration for requests.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not ollama_available:
            raise ImportError(
                "Ollama library not installed. Please install `ollama` or `llmcore[ollama]`."
            )
        if not tiktoken_available:
            logger.warning("tiktoken library not available. Token counting will use approximation.")

        self.host = config.get("host")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        # The context the server allocates. Without it Ollama uses its own
        # default (4096 on small GPUs) and silently truncates longer prompts.
        try:
            self.num_ctx: int | None = int(config["num_ctx"]) if config.get("num_ctx") else None
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid Ollama num_ctx %r.", config.get("num_ctx"))
            self.num_ctx = None
        # ollama.com cloud (and authenticated proxies) need a bearer token.
        self._api_key = config.get("api_key") or os.environ.get(
            config.get("api_key_env_var") or "OLLAMA_API_KEY", ""
        )
        #: model -> trained window reported by the server (tags/show)
        self._server_ctx: dict[str, int] = {}
        #: model -> context the server actually loaded it with (/api/ps)
        self._effective_ctx: dict[str, int] = {}
        #: model -> thinking levels from /api/show (server >= 0.34.3)
        self._thinking_levels: dict[str, list[str]] = {}
        self._warned_ctx: set[str] = set()
        timeout_val = config.get("timeout")
        self.timeout = float(timeout_val) if timeout_val is not None else None
        self.default_keep_alive = config.get("keep_alive")
        # Model discovery is expensive (one show() per local model); cache the
        # result briefly so per-request context-length lookups don't re-scan
        # the whole model inventory. TTL <= 0 disables the cache.
        try:
            self._models_cache_ttl = float(config.get("models_cache_ttl", 60.0))
        except (TypeError, ValueError):
            self._models_cache_ttl = 60.0
        self._models_details_cache: tuple[float, list[ModelDetails]] | None = None

        try:
            client_args: dict[str, Any] = {}
            if self.host:
                client_args["host"] = self.host
            if self.timeout is not None:
                client_args["timeout"] = self.timeout
            if self._api_key:
                client_args["headers"] = {"Authorization": f"Bearer {self._api_key}"}
            self._client = AsyncClient(**client_args)
            logger.debug("Ollama AsyncClient initialized.")
            # llmcore's rule is to call the API directly and fall back to the
            # vendor SDK. Ollama's REST surface is small and stable, so the
            # direct path is available with `backend = "httpx"`; the SDK stays
            # the default because this provider's tests mock AsyncClient and it
            # handles host resolution and NDJSON framing already.
            self._backend = self._resolve_backend(config.get("backend"))
            self._http: Any = None
        except Exception as e:
            logger.error(f"Failed to initialize Ollama AsyncClient: {e}", exc_info=True)
            raise ConfigError(f"Ollama client initialization failed: {e}")

        self.tokenizer_name = config.get("tokenizer", "tiktoken_cl100k_base")
        self._encoding = None
        if self.tokenizer_name.startswith("tiktoken_"):
            if tiktoken_available and tiktoken:
                try:
                    encoding_name = self.tokenizer_name.split("tiktoken_")[1]
                    self._encoding = tiktoken.get_encoding(encoding_name)
                    logger.info(f"OllamaProvider using tiktoken encoding: {encoding_name}.")
                except Exception as e:
                    logger.warning(
                        f"Failed to load tiktoken encoding '{self.tokenizer_name}'. "
                        f"Falling back to approximation. Error: {e}"
                    )
                    self.tokenizer_name = "char_div_4"
            else:
                logger.warning("tiktoken not available. Falling back to character approximation.")
                self.tokenizer_name = "char_div_4"
        elif self.tokenizer_name != "char_div_4":
            logger.warning(
                f"Unsupported tokenizer '{self.tokenizer_name}'. Falling back to approximation."
            )
            self.tokenizer_name = "char_div_4"

        if self.tokenizer_name == "char_div_4":
            logger.info("OllamaProvider using character division for token counting.")

    def get_name(self) -> str:
        """Returns the provider instance name."""
        return self._provider_instance_name or "ollama"

    async def get_models_details(self) -> list[ModelDetails]:
        """
        Asynchronously discovers and returns detailed information about available
        local models by querying the Ollama API.

        Uses ``client.list()`` for the model inventory and, where possible,
        ``client.show()`` to obtain per-model capability flags (tools, vision,
        thinking). The ``show()`` calls run concurrently: with many local
        models a sequential scan multiplies the per-request timeout (N models
        x timeout on an unresponsive server), which stalled callers for tens
        of minutes. Results are cached for ``models_cache_ttl`` seconds.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")
        if self._models_details_cache is not None:
            cached_at, cached = self._models_details_cache
            if time.monotonic() - cached_at < self._models_cache_ttl:
                return list(cached)
        try:
            if self._backend == "httpx":
                resp = await self._get_http().get("/api/tags")
                if resp.status_code >= 400:
                    self._raise_direct_status(resp.status_code, resp.text, "tags")
                list_response = resp.json()
            else:
                list_response = await self._client.list()
            # The SDK returns ListResponse (.models); the direct path a dict
            # ({"models": [...]}) - which used to fail with AttributeError.
            model_list = _field(list_response, "models") or []
            model_entries = [entry for entry in model_list if _field(entry, "model")]

            async def _fetch_capabilities(entry: Any) -> tuple[bool, bool, bool]:
                """Flags from the tags entry (servers >= 0.34.1 list them),
                else from show(); defaults on failure."""
                model_name = _field(entry, "model")
                capabilities = _field(entry, "capabilities")
                if not isinstance(capabilities, (list, tuple)) or not capabilities:
                    try:
                        show = await self._show(model_name)
                        capabilities = show.get("capabilities") or []
                    except Exception as e:
                        logger.debug(
                            f"Could not fetch capabilities for '{model_name}': {e}. Using defaults."
                        )
                        return False, False, False
                return (
                    "tools" in capabilities,
                    "vision" in capabilities,
                    "thinking" in capabilities,
                )

            capability_flags = await asyncio.gather(
                *(_fetch_capabilities(entry) for entry in model_entries)
            )
            for entry in model_entries:
                window = _field(_field(entry, "details"), "context_length")
                if isinstance(window, int) and window > 0:
                    self._server_ctx[_field(entry, "model")] = window
            await self._observe_effective_context()

            details_list = []
            for model_entry, flags in zip(model_entries, capability_flags, strict=True):
                model_name = _field(model_entry, "model")
                supports_tools, supports_vision, supports_reasoning = flags

                # Extract rich details from the ListResponse.Model.details
                entry_details = _field(model_entry, "details")
                family = _field(entry_details, "family")
                families = _field(entry_details, "families")
                parameter_size = _field(entry_details, "parameter_size")
                quantization = _field(entry_details, "quantization_level")
                file_size = _field(model_entry, "size")

                details = ModelDetails(
                    id=model_name,
                    context_length=self.get_max_context_length(model_name),
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    supports_reasoning=supports_reasoning,
                    provider_name=self.get_name(),
                    family=family,
                    parameter_count=parameter_size,
                    quantization_level=quantization,
                    file_size_bytes=int(file_size) if file_size is not None else None,
                    metadata={
                        "families": families,
                        "digest": _field(model_entry, "digest"),
                        "trained_context_length": self._server_ctx.get(model_name),
                        "effective_context_length": self._effective_ctx.get(model_name),
                        "embedding": "embedding" in (_field(model_entry, "capabilities") or []),
                    },
                )
                details_list.append(details)

            logger.info(f"Discovered {len(details_list)} local Ollama models.")
            if self._models_cache_ttl > 0:
                self._models_details_cache = (time.monotonic(), list(details_list))
            return details_list
        except ResponseError as e:
            logger.error(
                f"Ollama API error fetching models: HTTP {e.status_code} - {e.error}",
                exc_info=True,
            )
            raise ProviderError(
                self.get_name(), f"Failed to fetch models from Ollama API: {e.error}"
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching models from Ollama API: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error fetching models: {e}")

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """
        Returns a schema of supported inference parameters for Ollama models.

        Covers both the ``options`` dict (sampling, runtime) and top-level
        chat parameters (format, think, logprobs, keep_alive).
        """
        return {
            # --- Sampling / runtime options ---
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer"},
            "num_ctx": {"type": "integer"},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "mirostat": {"type": "integer"},
            "mirostat_tau": {"type": "number"},
            "mirostat_eta": {"type": "number"},
            "num_predict": {"type": "integer"},
            "max_tokens": {"type": "integer", "note": "alias for num_predict"},
            "presence_penalty": {"type": "number"},
            "frequency_penalty": {"type": "number"},
            "repeat_penalty": {"type": "number"},
            "repeat_last_n": {"type": "integer"},
            "tfs_z": {"type": "number"},
            "typical_p": {"type": "number"},
            "penalize_newline": {"type": "boolean"},
            "num_batch": {"type": "integer"},
            "num_gpu": {"type": "integer"},
            "num_thread": {"type": "integer"},
            "num_keep": {"type": "integer"},
            "low_vram": {"type": "boolean"},
            "use_mmap": {"type": "boolean"},
            "use_mlock": {"type": "boolean"},
            # --- Top-level chat parameters ---
            "format": {"type": ["string", "object"], "note": "'json' or JSON Schema"},
            "keep_alive": {"type": ["number", "string"]},
            "think": {"type": ["boolean", "string"], "note": "true or 'low'/'medium'/'high'"},
            # llmcore's cross-provider knob: "none" -> think=false, else a
            # level the model lists (/api/show thinking.values) or true.
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            },
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer"},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """The context the server will actually give ``model``.

        Resolution order:
            1. ``num_ctx`` from config (sent with every request), capped at
               the trained window when that is known.
            2. The window the server loaded the model with (``/api/ps``) --
               its default is 4096 on small GPUs, whatever the model's training.
            3. The trained window the server reports (``/api/tags``).
            4. Model card registry, then DEFAULT_OLLAMA_TOKEN_LIMITS.
            5. Fallback to 4096.
        """
        model_name = model or self.default_model
        trained = self._server_ctx.get(model_name)
        if self.num_ctx:
            return min(self.num_ctx, trained) if trained else self.num_ctx
        if model_name in self._effective_ctx:
            return self._effective_ctx[model_name]
        if trained:
            return trained

        # Try the model card registry — it has context info for models
        # not in the hardcoded dict (e.g. qwen3, llama3.2, etc.)
        try:
            from ..model_cards import get_model_card_registry

            registry = get_model_card_registry()
            card_limit = registry.get_context_length(
                "ollama",
                model_name,
                default=0,
            )
            if card_limit > 0:
                logger.debug(
                    f"Context length for Ollama model '{model_name}' from model card: {card_limit}"
                )
                return card_limit
        except Exception as e:
            logger.warning(f"Model card registry lookup failed for '{model_name}': {e}")

        base_model_name = model_name.split(":")[0]
        limit = DEFAULT_OLLAMA_TOKEN_LIMITS.get(
            model_name, DEFAULT_OLLAMA_TOKEN_LIMITS.get(base_model_name)
        )
        if limit is not None:
            return limit

        limit = 4096
        logger.warning(
            f"Unknown context length for Ollama model '{model_name}'. Using fallback: {limit}."
        )
        return limit

    # ------------------------------------------------------------------
    # Message Conversion
    # ------------------------------------------------------------------

    def _build_message_payload(
        self, msg: Message, call_names: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Convert a single llmcore ``Message`` to an Ollama message dict.

        Handles:
        - Standard text messages (system, user, assistant)
        - Tool-result messages (role=tool → includes ``tool_name``)
        - Multimodal images via ``metadata["inline_images"]``

        Image convention (same as Gemini provider):
        - ``metadata["inline_images"]``: list of image sources.  Each element
          may be a ``str`` (file path or base64), ``bytes``, or a dict with
          ``{"data": ..., "mime_type": ...}`` where ``data`` is base64 or bytes.
        """
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        payload: dict[str, Any] = {
            "role": role_str,
            "content": msg.content,
        }

        # Tool-result messages need the function name so the model can
        # correlate the result with its earlier tool_call; resolved from the
        # call id when the caller did not name it.
        if role_str == "tool":
            tool_name = (
                msg.metadata.get("tool_name")
                or msg.metadata.get("name")
                or (call_names or {}).get(str(msg.tool_call_id or ""))
            )
            if tool_name:
                payload["tool_name"] = tool_name

        # Assistant turns go back with their tool calls and thinking (both
        # were dropped, so a tool loop resent a turn with no calls).
        if role_str == "assistant":
            calls = getattr(msg, "tool_calls", None) or msg.metadata.get("tool_calls") or []
            if calls:
                payload["tool_calls"] = [_ollama_tool_call(c) for c in calls]
            thinking = msg.metadata.get("thinking") or msg.metadata.get("reasoning_content")
            if thinking:
                payload["thinking"] = thinking

        # Multimodal image support
        images = msg.metadata.get("inline_images")
        if images:
            payload["images"] = self._extract_image_values(images)

        return payload

    @staticmethod
    def _extract_image_values(images: list[Any]) -> list[Any]:
        """Normalise a list of image sources into values the SDK accepts.

        The ollama SDK's ``Image`` type serializer accepts:
        - ``str``: base64 string **or** file path
        - ``bytes``: raw image bytes
        - ``pathlib.Path``: file path

        We accept those plus a dict with a ``data`` key for compatibility
        with the Gemini provider's ``inline_images`` convention.
        """
        out: list[Any] = []
        for img in images:
            if isinstance(img, dict):
                # Dict convention: {"data": base64_or_bytes, "mime_type": "image/png"}
                out.append(img.get("data", img))
            else:
                # str (base64 or path) / bytes / Path — pass through to SDK
                out.append(img)
        return out

    # ------------------------------------------------------------------
    # Chat Completion
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport. ``"sdk"`` by default; ``"httpx"`` is opt-in."""
        req = (requested or "sdk").lower()
        if req not in ("sdk", "httpx"):
            logger.warning("Unknown Ollama backend '%s'; using the SDK.", req)
            return "sdk"
        if req == "httpx" and not httpx_available:
            logger.warning("httpx is not installed; using the Ollama SDK.")
            return "sdk"
        return req

    def _direct_base_url(self) -> str:
        """Return the server root for the direct transport.

        Mirrors the SDK (config ``host``, then ``OLLAMA_HOST``, then
        localhost) so switching transports does not change the server.
        """
        host = self.host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host.rstrip("/")

    def _get_http(self) -> Any:
        """Return the lazily-built direct HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._direct_base_url(),
                timeout=self.timeout if self.timeout is not None else 120.0,
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else None,
            )
        return self._http

    def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
        """Map a direct failure onto the same errors the SDK path raises.

        Raises:
            ProviderError: Always.
        """
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"Model '{model_name}' not found. Pull it with `ollama pull {model_name}`.",
                model_name=model_name,
                status_code=status,
            )
        raise ProviderError(
            self.get_name(),
            f"Ollama API Error (HTTP {status}): {body}",
            model_name=model_name,
            status_code=status,
            retryable=status >= 500,
        )

    async def _direct_chat(
        self, call_kwargs: dict[str, Any], model_name: str, *, stream: bool
    ) -> Any:
        """POST ``/api/chat``, returning a dict or an NDJSON line iterator.

        Returns the same shapes the SDK returns at this seam — a mapping for a
        single reply, an async iterator of mappings when streaming — because
        ``_normalize_response`` and ``_wrap_stream`` consume both and already
        accept plain dicts.
        """
        body = {**call_kwargs, "stream": stream}
        if not stream:
            try:
                resp = await self._get_http().post("/api/chat", json=body)
            except httpx.ConnectError as exc:
                raise ProviderError(
                    self.get_name(),
                    f"Cannot connect to Ollama server at {self._direct_base_url()}. "
                    f"Is Ollama running?",
                    model_name=model_name,
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderError(
                    self.get_name(), f"Ollama transport error: {exc}", model_name=model_name
                ) from exc
            if resp.status_code >= 400:
                self._raise_direct_status(resp.status_code, resp.text, model_name)
            return resp.json()
        return self._direct_chat_stream(body, model_name)

    async def _direct_chat_stream(
        self, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield NDJSON chat chunks from ``/api/chat``.

        Ollama streams newline-delimited JSON rather than server-sent events, so
        there is no ``data:`` prefix to strip — each line is a whole object.
        """
        client = self._get_http()
        try:
            async with client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_direct_status(resp.status_code, resp.text, model_name)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        logger.warning("Skipping unparseable Ollama NDJSON line.")
        except ProviderError:
            raise
        except httpx.ConnectError as exc:
            raise ProviderError(
                self.get_name(),
                f"Cannot connect to Ollama server at {self._direct_base_url()}. "
                f"Is Ollama running?",
                model_name=model_name,
            ) from exc

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Sends a chat completion request to the Ollama API.

        Supports tools, thinking mode, structured output (``format``),
        multimodal images, logprobs, and ``keep_alive``.

        Keyword arguments are routed to the correct location:
        - ``format``, ``keep_alive``, ``think``, ``logprobs``, ``top_logprobs``
          go as top-level params to ``client.chat()``.
        - Everything else goes into the ``options`` dict.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Ollama client not initialized.")

        model_name = model or self.default_model
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        call_names = {
            str(_call_id(c)): _call_name(c)
            for msg in context
            for c in (getattr(msg, "tool_calls", None) or (msg.metadata or {}).get("tool_calls") or [])
            if _call_id(c)
        }
        messages_payload = []
        pending: list[str] = []  # names of the last assistant turn's calls, in order
        for msg in context:
            payload = self._build_message_payload(msg, call_names)
            if payload["role"] == "assistant":
                pending = [f["function"]["name"] for f in payload.get("tool_calls") or []]
            elif payload["role"] == "tool":
                # The SDK drops call ids, so results that cannot be matched by
                # id are matched by position, the order Ollama returns them in.
                if "tool_name" not in payload and pending:
                    payload["tool_name"] = pending[0]
                if pending and payload.get("tool_name") in pending:
                    pending.remove(payload["tool_name"])
            messages_payload.append(payload)
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages to send.")

        kwargs = dict(kwargs)
        # --- Alias: max_tokens → num_predict ---
        if "max_tokens" in kwargs:
            kwargs["num_predict"] = kwargs.pop("max_tokens")
        if self.num_ctx and "num_ctx" not in kwargs:
            kwargs["num_ctx"] = self.num_ctx
        effort = kwargs.pop("effort", None)
        if effort is not None and "think" not in kwargs:
            kwargs["think"] = self._think_for(model_name, str(effort).strip().lower())

        # --- Route kwargs into top-level params vs options dict ---
        top_level_kwargs: dict[str, Any] = {}
        options_kwargs: dict[str, Any] = {}
        supported_params = self.get_supported_parameters()

        for key, value in kwargs.items():
            if key in _TOP_LEVEL_CHAT_PARAMS:
                top_level_kwargs[key] = value
            elif key in supported_params:
                options_kwargs[key] = value
            else:
                logger.warning(
                    f"Parameter '{key}' not in supported parameters for Ollama. "
                    f"Passing through as option anyway."
                )
                options_kwargs[key] = value

        # Apply default keep_alive if configured and not explicitly set
        if "keep_alive" not in top_level_kwargs and self.default_keep_alive is not None:
            top_level_kwargs["keep_alive"] = self.default_keep_alive

        # --- Prepare tools for the ollama library ---
        tools_payload: list[dict[str, Any]] | None = None
        if tools:
            logger.debug(f"Preparing {len(tools)} tools for Ollama request.")
            tools_payload = [
                {
                    "type": "function",
                    "function": tool.model_dump(),
                }
                for tool in tools
            ]

        # --- Logging ---
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
                "options": options_kwargs,
                "tools": tools_payload,
                "tool_choice": tool_choice,
                **top_level_kwargs,
            }
            logger.debug(
                f"RAW LLM REQUEST ({self.get_name()} @ {model_name}): "
                f"{json.dumps(log_data, indent=2, default=str)}"
            )

        try:
            # Build the call kwargs
            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
            }
            if tools_payload:
                call_kwargs["tools"] = tools_payload
            if options_kwargs:
                call_kwargs["options"] = options_kwargs

            # Add top-level params
            for key, value in top_level_kwargs.items():
                call_kwargs[key] = value

            if self._backend == "httpx":
                response_or_stream = await self._direct_chat(
                    call_kwargs, model_name, stream=bool(stream)
                )
            else:
                response_or_stream = await self._client.chat(**call_kwargs)

            if stream:
                return await self._wrap_stream(response_or_stream, model_name)
            normalized = self._normalize_response(response_or_stream)
            await self._observe_effective_context(model_name)
            return normalized

        except ProviderError:
            # Already mapped by the direct transport. Without this, the generic
            # handler below re-wraps it as "An unexpected error occurred", which
            # buries the actionable message and makes the two transports report
            # the same condition differently.
            raise
        except ResponseError as e:
            error_detail = e.error if hasattr(e, "error") and e.error else str(e)
            logger.error(f"Ollama API error: HTTP {e.status_code} - {error_detail}", exc_info=True)
            if e.status_code == 404:
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found. Pull it with `ollama pull {model_name}`.",
                )
            raise ProviderError(
                self.get_name(), f"Ollama API Error (HTTP {e.status_code}): {error_detail}"
            )
        except ConnectionError as e:
            logger.error(f"Cannot connect to Ollama server: {e}", exc_info=True)
            raise ProviderError(
                self.get_name(),
                f"Cannot connect to Ollama server at {self.host or 'localhost:11434'}. "
                f"Is Ollama running?",
            )
        except Exception as e:
            logger.error(f"Unexpected error during Ollama chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    # ------------------------------------------------------------------
    # Response Normalization
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> dict[str, Any]:
        """Convert a non-streaming ``ChatResponse`` to a plain dict.

        The Ollama SDK returns ``ChatResponse`` Pydantic models.  We
        normalize to a consistent dict format for downstream consumption.

        The Ollama native format is preserved (``message.content``,
        ``message.tool_calls``, ``message.thinking``), keeping it distinct
        from the OpenAI choices-based format used by OpenAI/Gemini providers.
        The ``extract_*`` methods abstract the difference.
        """
        if isinstance(response, dict):
            response_dict = response
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "__dict__"):
            response_dict = dict(response.__dict__)
        else:
            raise ProviderError(self.get_name(), "Invalid non-streaming response format.")

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"RAW LLM RESPONSE ({self.get_name()}): "
                f"{json.dumps(response_dict, indent=2, default=str)}"
            )
        return response_dict

    async def _wrap_stream(
        self, stream: Any, model_name: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Wrap the SDK's async stream, yielding normalized dicts.

        Each ``ChatResponse`` chunk is converted to a dict via
        ``model_dump()`` so that all downstream consumers see dicts,
        matching the ``AsyncGenerator[dict, None]`` return type contract.
        The final (``done``) chunk also carries the whole turn's thinking as
        ``message.replay_thinking``, for :meth:`assistant_replay_metadata`.
        """

        async def _inner() -> AsyncGenerator[dict[str, Any], None]:
            thinking: list[str] = []
            async for chunk in stream:
                # Normalize to dict
                if hasattr(chunk, "model_dump"):
                    chunk_dict = chunk.model_dump()
                elif isinstance(chunk, dict):
                    chunk_dict = chunk
                else:
                    chunk_dict = {"message": {"content": str(chunk)}}

                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RAW LLM STREAM CHUNK ({self.get_name()}): "
                        f"{json.dumps(chunk_dict, default=str)}"
                    )
                message = chunk_dict.get("message")
                if isinstance(message, dict):
                    if message.get("thinking"):
                        thinking.append(message["thinking"])
                    if chunk_dict.get("done") and thinking:
                        message["replay_thinking"] = "".join(thinking)
                yield chunk_dict
            if model_name:
                await self._observe_effective_context(model_name)

        return _inner()

    # ------------------------------------------------------------------
    # Server state: effective context, thinking levels
    # ------------------------------------------------------------------

    async def _show(self, model_name: str) -> dict[str, Any]:
        """``/api/show`` as a dict, on the configured transport.

        The SDK drops fields it does not model (``thinking`` levels among
        them), so those are only known with ``backend = "httpx"``.
        """
        if self._backend == "httpx":
            resp = await self._get_http().post("/api/show", json={"model": model_name})
            if resp.status_code >= 400:
                self._raise_direct_status(resp.status_code, resp.text, model_name)
            shown: Any = resp.json()
        else:
            raw = await self._client.show(model_name)
            dumped = raw.model_dump() if hasattr(raw, "model_dump") else raw
            shown = dumped if isinstance(dumped, dict) else {
                "capabilities": getattr(raw, "capabilities", None) or [],
                "model_info": getattr(raw, "modelinfo", None) or {},
            }
        levels = (shown.get("thinking") or {}).get("values") if isinstance(shown.get("thinking"), dict) else None
        if levels:
            self._thinking_levels[model_name] = [str(v) for v in levels]
        # "model_info" on the wire, "modelinfo" in the SDK's dump.
        info = shown.get("model_info") or shown.get("modelinfo") or {}
        for key, value in (info.items() if isinstance(info, dict) else []):
            if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                self._server_ctx.setdefault(model_name, value)
        return dict(shown)

    async def _observe_effective_context(self, model_name: str | None = None) -> None:
        """Record the context the server loaded models with (``/api/ps``).

        Warns once per model when that is smaller than its trained window,
        which is when Ollama truncates prompts silently.
        """
        if self.num_ctx or (model_name and model_name in self._effective_ctx):
            return
        try:
            if self._backend == "httpx":
                resp = await asyncio.wait_for(self._get_http().get("/api/ps"), timeout=5)
                running = (resp.json().get("models") or []) if resp.status_code < 400 else []
            else:
                running = _field(await asyncio.wait_for(self._client.ps(), timeout=5), "models") or []
        except Exception as e:
            logger.debug("Could not read Ollama /api/ps: %s", e)
            return
        for entry in running:
            name = _field(entry, "model") or _field(entry, "name")
            window = _field(entry, "context_length")
            if not name or not isinstance(window, int) or window <= 0:
                continue
            self._effective_ctx[name] = window
            trained = self._server_ctx.get(name)
            if trained and window < trained and name not in self._warned_ctx:
                self._warned_ctx.add(name)
                logger.warning(
                    "Ollama runs '%s' with a %d-token context (server default), not its "
                    "trained %d; longer prompts are truncated silently. Set num_ctx in "
                    "[providers.ollama] (or OLLAMA_CONTEXT_LENGTH on the server).",
                    name,
                    window,
                    trained,
                )

    def _think_for(self, model_name: str, effort: str) -> bool | str:
        """``think`` for an effort level: False for "none", else the nearest
        level the model declares (``/api/show`` thinking.values), else True."""
        if effort == "none":
            return False
        levels = self._thinking_levels.get(model_name)
        if levels:
            from .openai_provider import nearest_effort

            return nearest_effort(effort, levels) or True
        return True

    # ------------------------------------------------------------------
    # Content Extraction
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """
        Extract text content from Ollama non-streaming response.

        Ollama /api/chat response format::

            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": true,
                ...
            }

        Args:
            response: The raw response dictionary from chat_completion().

        Returns:
            The extracted text content.
        """
        try:
            message = response.get("message", {})
            if isinstance(message, dict):
                return message.get("content") or ""
            return ""
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to extract content from Ollama response: {e}")
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """
        Extract text delta from Ollama streaming chunk.

        Ollama streaming format::

            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "The"},
                "done": false
            }

        Each chunk contains incremental content.  The final chunk has
        ``"done": true`` and may have empty content.

        Args:
            chunk: A single streaming chunk dictionary.

        Returns:
            The extracted text delta.
        """
        try:
            if isinstance(chunk, dict):
                message = chunk.get("message", {})
            elif hasattr(chunk, "message"):
                # Handle Pydantic ChatResponse model case (shouldn't happen
                # after _wrap_stream normalization, but kept for safety)
                msg_attr = chunk.message
                if isinstance(msg_attr, dict):
                    message = msg_attr
                elif hasattr(msg_attr, "content"):
                    return msg_attr.content or ""
                else:
                    return ""
            else:
                return ""

            if isinstance(message, dict):
                return message.get("content") or ""
            return ""
        except (KeyError, TypeError, AttributeError):
            return ""

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """Alias of :meth:`extract_thinking_content` (the name other providers use)."""
        return self.extract_thinking_content(response)

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """The thinking delta of a streaming chunk."""
        message = chunk.get("message") if isinstance(chunk, dict) else None
        return (message or {}).get("thinking") or None if isinstance(message, dict) else None

    def assistant_replay_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """The thinking to send back with the next assistant turn.

        Ollama's tool-calling docs append the whole assistant message
        (thinking, content, tool_calls). Tool calls travel in
        ``Message.tool_calls``; this returns the thinking. For a stream, pass
        the final (``done``) chunk.
        """
        message = response.get("message") if isinstance(response, dict) else None
        if not isinstance(message, dict):
            return {}
        thinking = message.get("replay_thinking") or message.get("thinking")
        return {"thinking": thinking} if thinking else {}

    def extract_thinking_content(self, response: dict[str, Any]) -> str | None:
        """Extract thinking/reasoning content from a non-streaming response.

        When ``think=True`` (or a thinking level) is passed, the model
        populates ``message.thinking`` with its chain-of-thought.

        Args:
            response: The raw response dict from chat_completion().

        Returns:
            The thinking text, or None if not present.
        """
        try:
            message = response.get("message", {})
            if isinstance(message, dict):
                thinking = message.get("thinking")
                return thinking if thinking else None
            return None
        except (KeyError, TypeError):
            return None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an Ollama non-streaming response.

        Ollama tool call format in ``message``::

            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"}
                        }
                    }
                ]
            }

        Note: Ollama tool calls do **not** have an ``id`` field.  We
        generate a UUID to satisfy the ``ToolCall`` model contract.

        Args:
            response: The raw response dict from chat_completion().

        Returns:
            List of ToolCall objects. Empty list if no tool calls.
        """
        tool_calls_out: list[ToolCall] = []
        try:
            message = response.get("message", {})
            if not isinstance(message, dict):
                return tool_calls_out
            raw_calls = message.get("tool_calls")
            if not raw_calls:
                return tool_calls_out
            for tc in raw_calls:
                func = tc.get("function", {})
                arguments = func.get("arguments", {})
                # Arguments from Ollama are already a dict (not JSON string)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                tool_calls_out.append(
                    ToolCall(
                        # Servers since 2026 send an id ("call_..."); keep it
                        # so tool results can be matched to their call.
                        id=str(tc.get("id") or uuid.uuid4()),
                        name=func.get("name", "unknown"),
                        arguments=arguments,
                    )
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract tool calls from Ollama response: {e}")
        return tool_calls_out

    def extract_usage(self, response: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from Ollama response.

        Ollama reports usage as durations and counts directly on the
        response object, not in a nested ``usage`` dict like OpenAI.

        Maps to OpenAI-compatible field names for uniformity.

        Args:
            response: The raw response dict.

        Returns:
            Dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
        """
        prompt_tokens = response.get("prompt_eval_count") or 0
        completion_tokens = response.get("eval_count") or 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": response.get("prompt_eval_cached_count"),
            "finish_reason": response.get("done_reason"),
            # Ollama-specific timing (nanoseconds)
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_duration": response.get("prompt_eval_duration"),
            "eval_duration": response.get("eval_duration"),
        }

    # ------------------------------------------------------------------
    # Token Counting
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Counts tokens using the configured tokenizer or character approximation."""
        if not text:
            return 0
        if not self._encoding:
            return _EstimateCounter().count(text)
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))  # type: ignore

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Approximates token count for a list of messages using tiktoken."""
        if not self._encoding:
            counter = _EstimateCounter()
            total = sum(
                counter.count(msg.content)
                + counter.count(msg.role.value if hasattr(msg.role, "value") else str(msg.role))
                for msg in messages
            )
            return total + len(messages) * 2 + 1

        num_tokens = 0
        for message in messages:
            num_tokens += 3  # tokens_per_message overhead
            try:
                role_str = (
                    message.role.value if hasattr(message.role, "value") else str(message.role)
                )
                num_tokens += len(self._encoding.encode(role_str))
                num_tokens += len(self._encoding.encode(message.content))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed for message part: {e}. Approximating.")
                role_str_fallback = (
                    message.role.value if hasattr(message.role, "value") else str(message.role)
                )
                num_tokens += (len(role_str_fallback) + len(message.content)) // 4
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Closes the underlying Ollama client session.

        The ollama SDK's ``AsyncClient`` API varies across versions: some
        releases expose an async ``close()`` (which internally calls
        ``aclose()`` on the wrapped httpx ``AsyncClient``), while others expose
        no ``close()`` at all and only the inner ``httpx.AsyncClient`` (stored
        as ``_client``). We therefore call ``close()`` when it is available and
        otherwise fall back to ``aclose()`` on the inner httpx client, so
        cleanup works across ollama versions instead of raising
        ``AttributeError``.
        """
        if self._client:
            try:
                client_close = getattr(self._client, "close", None)
                if callable(client_close):
                    await client_close()
                else:
                    inner = getattr(self._client, "_client", None)
                    inner_aclose = getattr(inner, "aclose", None)
                    if callable(inner_aclose):
                        await inner_aclose()
                logger.info("OllamaProvider client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OllamaProvider client: {e}", exc_info=True)
            finally:
                self._client = None


def _field(obj: Any, name: str) -> Any:
    """A field from an SDK model or a plain dict (the two transports)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _call_id(call: Any) -> str | None:
    if isinstance(call, dict):
        return call.get("id")
    return getattr(call, "id", None)


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        func = call.get("function") if isinstance(call.get("function"), dict) else call
        return str(func.get("name") or "")
    return str(getattr(call, "name", "") or "")


def _ollama_tool_call(call: Any) -> dict[str, Any]:
    """An OpenAI-shaped (or ToolCall) call as Ollama expects it back:
    ``{"function": {"name", "arguments": <dict>}}`` (arguments are objects,
    not JSON strings)."""
    if not isinstance(call, dict):
        call = {"id": getattr(call, "id", None),
                "function": {"name": getattr(call, "name", ""), "arguments": getattr(call, "arguments", {})}}
    func = dict(call.get("function") or {})
    arguments = func.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            arguments = {"raw": arguments}
    out: dict[str, Any] = {"function": {"name": func.get("name", ""), "arguments": arguments}}
    if func.get("index") is not None:
        out["function"]["index"] = func["index"]
    if call.get("id"):
        out["id"] = call["id"]
    return out
