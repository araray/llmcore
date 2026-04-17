# src/llmcore/providers/vllm_provider.py
"""
vLLM provider implementation for the LLMCore library.

Integrates with self-hosted `vLLM <https://github.com/vllm-project/vllm>`_
inference servers. vLLM exposes an OpenAI-compatible surface at
``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``, and
``/v1/models``, so this provider **subclasses** :class:`OpenAIProvider`
and overrides only what differs:

- **Authentication**: vLLM's ``--api-key`` is an optional shared secret
  with no canonical environment variable. We honour ``api_key`` /
  ``api_key_env_var`` in config and fall back to the literal string
  ``"EMPTY"`` (the OpenAI SDK refuses empty keys; vLLM accepts any
  string when ``--api-key`` is not set).
- **Base URL**: required. There is no default because vLLM is
  self-hosted. A missing ``base_url`` raises :class:`ConfigError`.
- **Model discovery**: ``/v1/models`` returns vLLM's
  :class:`ModelCard` with a ``max_model_len`` field. We fetch it via
  :mod:`httpx` directly to guarantee access to that extra field
  regardless of OpenAI SDK version, and cache the result.
- **Context length**: delegates to the discovered cache first,
  ignoring the OpenAI model-name heuristics in the parent (they are
  irrelevant for arbitrary vLLM-served models).
- **Supported parameters**: adds vLLM-specific knobs
  (``structured_outputs``, ``guided_json``, ``guided_regex``,
  ``guided_choice``, ``guided_grammar``, ``prompt_logprobs``,
  ``kv_transfer_params``, ``min_p``, ``top_k``, ``repetition_penalty``,
  etc.) and declares ``extra_body`` so :meth:`chat_completion` can
  forward non-OpenAI fields as top-level JSON keys through the SDK.
  Removes OpenAI-only params (``service_tier``, ``prompt_cache_key``,
  ``web_search_options``, ``verbosity``, ``prediction``) that vLLM
  ignores or rejects.
- **Tokenisation**: vLLM serves arbitrary models whose tokenisers are
  not ``tiktoken``-compatible. :meth:`count_tokens` and
  :meth:`count_message_tokens` fall back to character-based
  approximation, consistent with how :class:`OllamaProvider` behaves
  when no tokenizer is available. This is advisory (used for context
  budgeting, not billing); exact counts via vLLM's ``/tokenize``
  endpoint are deferred to a later PR.
- **chat_completion**: inherited from :class:`OpenAIProvider`, with a
  lightweight override that repackages vLLM-specific kwargs into
  ``extra_body`` so they bypass the parent's kwarg whitelist and reach
  the server as top-level JSON fields.

Tool-calling support cannot be detected from the client: it depends on
whether the server was launched with ``--enable-auto-tool-choice`` and
the matching ``--tool-call-parser`` for the model family. We publish
a best-effort heuristic on ``ModelDetails.supports_tools`` based on the
model ID substring, biased toward ``True`` so callers aren't blocked.
Errors from unsupported combinations surface via the inherited
:class:`ProviderError` path.

Multi-instance config is supported out of the box: any ``[providers.*]``
section with ``type = "vllm"`` is mapped to this class by
:class:`ProviderManager`, and ``get_name()`` returns the section name
(via ``_instance_name``) so metrics and logs remain distinguishable.

Tested against vLLM commit ``219bb5b8c`` (any vLLM ≥ v0.8 with the
OpenAI-compatible entrypoint should work).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import ContextPayload

# Deferred import mirrors the OpenRouter pattern and guards against
# the ``openai`` SDK being absent at import time.
try:
    from .openai_provider import OpenAIProvider, openai_available
except ImportError:
    OpenAIProvider = None  # type: ignore[assignment,misc]
    openai_available = False

logger = logging.getLogger(__name__)

#: Default model served by vLLM when the user does not specify one in
#: config. Chosen as a small, broadly-available instruct model; users
#: almost always override.
DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Set of kwargs that, when passed to :meth:`chat_completion`, should
#: be moved from top-level ``**kwargs`` into ``extra_body`` so the
#: OpenAI SDK forwards them as top-level JSON keys to vLLM. These are
#: the vLLM-specific extensions the OpenAI wire protocol does not know
#: about. Every key here MUST also be declared in
#: :meth:`get_supported_parameters` so the parent's whitelist check
#: passes.
_VLLM_EXTRA_BODY_KEYS: frozenset[str] = frozenset(
    {
        "structured_outputs",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
        "guided_decoding_backend",
        "prompt_logprobs",
        "kv_transfer_params",
        "min_p",
        "top_k",
        "repetition_penalty",
        "length_penalty",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "add_generation_prompt",
        "continue_final_message",
        "echo",
        "include_stop_str_in_output",
        "min_tokens",
        "ignore_eos",
        "chat_template",
        "chat_template_kwargs",
        "mm_processor_kwargs",
    }
)

#: OpenAI-only parameters that vLLM silently ignores or rejects.
#: Stripped from the inherited ``get_supported_parameters`` output so
#: users get a clear ``ValueError`` instead of a confusing server
#: response.
_OPENAI_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "service_tier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "web_search_options",
        "verbosity",
        "prediction",
        "audio",  # vLLM chat endpoint does not accept audio output parity
        "modalities",
    }
)

#: Model-name substrings that suggest tool-calling support. Advisory
#: only; vLLM's actual tool-calling depends on server launch flags.
_TOOL_CAPABLE_SUBSTRINGS: tuple[str, ...] = (
    "llama-3",
    "llama3",
    "qwen2.5",
    "qwen3",
    "mistral",
    "mixtral",
    "hermes",
    "command-r",
    "deepseek",
    "granite",
    "functionary",
)

#: Model-name substrings that suggest vision/multimodal support.
_VISION_CAPABLE_SUBSTRINGS: tuple[str, ...] = (
    "-vl",
    "-vision",
    "llava",
    "pixtral",
    "molmo",
    "internvl",
    "phi-3.5-vision",
    "phi-4-multimodal",
)


class VLLMProvider(OpenAIProvider):  # type: ignore[misc,valid-type]
    """LLMCore provider for self-hosted vLLM inference servers.

    Extends :class:`OpenAIProvider` because vLLM's chat, completion,
    embedding, and model-listing endpoints are byte-compatible with
    OpenAI's. The parent handles all hot-path logic (message
    serialisation, streaming, tool-call extraction, error mapping,
    observability). This subclass overrides only:

    - :meth:`__init__` — validates ``base_url``, resolves API key,
      initialises the vLLM-specific model cache.
    - :meth:`get_name` — returns ``"vllm"`` or the configured instance
      name.
    - :meth:`get_models_details` — queries ``/v1/models`` via httpx
      to capture ``max_model_len``.
    - :meth:`get_max_context_length` — uses the discovered cache
      instead of OpenAI model heuristics.
    - :meth:`get_supported_parameters` — adds vLLM kwargs, removes
      OpenAI-only kwargs, declares ``extra_body``.
    - :meth:`chat_completion` — repackages vLLM-specific kwargs into
      ``extra_body`` before delegating to the parent.
    - :meth:`count_tokens` / :meth:`count_message_tokens` — character
      approximation (tiktoken is wrong for arbitrary vLLM models).

    Configuration example (``[providers.vllm]``)::

        [providers.vllm]
        base_url = "http://localhost:8000/v1"
        default_model = "meta-llama/Llama-3.1-8B-Instruct"
        # api_key or api_key_env_var — only if vLLM launched with --api-key
        timeout = 240

    Multi-instance is supported — use ``type = "vllm"`` in any
    ``[providers.*]`` section::

        [providers.vllm_prod]
        type = "vllm"
        base_url = "http://vllm.prod.internal:8000/v1"
        api_key_env_var = "VLLM_PROD_TOKEN"
        default_model = "meta-llama/Llama-3.3-70B-Instruct"
    """

    #: Per-model cache of ``max_model_len`` discovered from
    #: ``/v1/models``. Populated on :meth:`get_models_details` call;
    #: consulted by :meth:`get_max_context_length`. Process-local;
    #: stale if the vLLM server is restarted with a different model.
    _vllm_model_cache: dict[str, int]

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialise the vLLM provider.

        Args:
            config: Configuration dictionary from ``[providers.vllm]``.

                Required:
                    ``base_url``: URL of the vLLM server, including the
                    ``/v1`` suffix (e.g. ``http://localhost:8000/v1``).

                Optional:
                    ``api_key``: Shared secret if the server was launched
                        with ``--api-key``. Defaults to ``"EMPTY"``.
                    ``api_key_env_var``: Name of an environment variable
                        to read the API key from.
                    ``default_model``: HuggingFace repo id or local name
                        of the model the server is serving. Defaults to
                        :data:`DEFAULT_VLLM_MODEL`.
                    ``timeout``: Request timeout in seconds. Defaults to
                        240 (local cold-starts can be slow).
                    ``fallback_context_length``: Hard cap on context
                        length used when ``max_model_len`` is not yet
                        discovered.

            log_raw_payloads: If True, request/response payloads are
                logged at DEBUG level (inherited from
                :class:`OpenAIProvider`).

        Raises:
            ConfigError: If ``base_url`` is missing from config.
            ImportError: If the ``openai`` Python SDK is not installed.
        """
        # --- 1. base_url is required; no sensible default exists. ---
        # Checked before the ImportError so users with config issues
        # get the actionable error first.
        base_url = config.get("base_url")
        if not base_url:
            raise ConfigError(
                "vLLM provider requires 'base_url' in config (e.g. "
                "'http://localhost:8000/v1'). There is no default because "
                "vLLM is self-hosted."
            )

        if OpenAIProvider is None or not openai_available:
            raise ImportError(
                "OpenAI library not installed. vLLM provider requires the "
                "'openai' package (reused as an OpenAI-compatible client). "
                "Install with 'pip install llmcore[openai]' or 'pip install openai'."
            )

        # --- 2. Resolve API key with documented fallbacks. ---
        api_key = config.get("api_key")
        api_key_env_var = config.get("api_key_env_var")
        if not api_key and api_key_env_var:
            api_key = os.environ.get(api_key_env_var)
        if not api_key:
            # vLLM accepts any non-empty string when --api-key is not set;
            # the OpenAI SDK refuses empty keys, so use a literal sentinel.
            api_key = "EMPTY"

        # --- 3. Build the OpenAI-shaped config for the parent. ---
        # Use a copy so we don't mutate the caller's dict, and keep every
        # key the caller provided so config helpers (``_instance_name``,
        # ``organization``, ``project``, etc.) flow through unchanged.
        openai_config: dict[str, Any] = {
            **config,
            "api_key": api_key,
            "base_url": base_url,
            "default_model": config.get("default_model", DEFAULT_VLLM_MODEL),
            "timeout": float(config.get("timeout", 240.0)),
        }

        # Set the instance name eagerly so it is available even if a
        # test patches the parent's __init__ (which is what normally
        # populates ``self._provider_instance_name`` via BaseProvider).
        # The parent __init__ will re-assign this from the same config
        # key, so production behaviour is unchanged.
        self._provider_instance_name = config.get("_instance_name")

        # Parent __init__ builds the ``AsyncOpenAI`` client with our
        # base_url. It also loads a tiktoken encoding, which is wrong
        # for arbitrary vLLM-served models — we clear it below.
        super().__init__(openai_config, log_raw_payloads=log_raw_payloads)

        # --- 4. Post-parent initialisation. ---
        # tiktoken is irrelevant for arbitrary vLLM-served models; clear
        # the encoding so count_tokens falls back to approximation rather
        # than silently producing GPT-specific token counts.
        self._encoding = None

        # Stash the user-supplied fallback context length (if any) and
        # allocate the model cache.
        self._fallback_context_length: int | None = None
        raw_fallback = config.get("fallback_context_length")
        if raw_fallback is not None:
            try:
                self._fallback_context_length = int(raw_fallback)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid 'fallback_context_length' in vLLM config: %r",
                    raw_fallback,
                )
        self._vllm_model_cache = {}

        logger.info(
            "VLLMProvider initialised: base_url=%s, default_model=%s, instance=%s",
            base_url,
            openai_config["default_model"],
            self._provider_instance_name or "vllm",
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the instance name, or ``"vllm"`` if unset."""
        return self._provider_instance_name or "vllm"

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models from vLLM's ``/v1/models`` endpoint.

        Uses :mod:`httpx` directly rather than the ``openai`` SDK's
        :meth:`AsyncOpenAI.models.list` because vLLM's
        :class:`ModelCard` exposes a non-OpenAI ``max_model_len``
        field that the SDK is not guaranteed to preserve across
        versions.

        Populates :attr:`_vllm_model_cache` as a side effect so
        :meth:`get_max_context_length` can consult it without an extra
        round trip.

        Returns:
            A list of :class:`ModelDetails`, one per model served. The
            ``context_length`` field reflects the server's
            ``max_model_len`` when available, falling back to 4096.

        Raises:
            ProviderError: If the server is unreachable, returns a
                non-2xx status, or emits malformed JSON.
        """
        url = f"{self.base_url.rstrip('/')}/models"
        headers: dict[str, str] = {}
        # Only send the Authorization header when a real token is set;
        # sending ``Bearer EMPTY`` to a no-auth server is harmless but
        # sending any header to a strictly-validated proxy may not be.
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.get_name(),
                f"Failed to list vLLM models: HTTP {e.response.status_code} from {url}",
            ) from e
        except httpx.HTTPError as e:
            raise ProviderError(
                self.get_name(),
                f"Transport error listing vLLM models at {url}: {e}",
            ) from e
        except ValueError as e:
            # .json() emits ValueError on malformed payload
            raise ProviderError(
                self.get_name(),
                f"Malformed /v1/models response from {url}: {e}",
            ) from e

        results: list[ModelDetails] = []
        model_entries = data.get("data", []) if isinstance(data, dict) else []
        for entry in model_entries:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("id")
            if not mid or not isinstance(mid, str):
                continue

            ctx_raw = entry.get("max_model_len")
            ctx: int | None = None
            if isinstance(ctx_raw, int) and ctx_raw > 0:
                ctx = ctx_raw
            elif ctx_raw is not None:
                # Tolerate string-typed max_model_len from custom proxies.
                try:
                    ctx_int = int(ctx_raw)
                    if ctx_int > 0:
                        ctx = ctx_int
                except (TypeError, ValueError):
                    pass

            if ctx is not None:
                self._vllm_model_cache[mid] = ctx

            results.append(
                ModelDetails(
                    id=mid,
                    provider_name=self.get_name(),
                    context_length=ctx if ctx is not None else 4096,
                    supports_streaming=True,
                    supports_tools=self._heuristic_supports_tools(mid),
                    supports_vision=self._heuristic_supports_vision(mid),
                    model_type="chat",
                    metadata={
                        "owned_by": entry.get("owned_by"),
                        "created": entry.get("created"),
                        "max_model_len": ctx,
                        "root": entry.get("root"),
                        "parent": entry.get("parent"),
                    },
                )
            )
        return results

    @staticmethod
    def _heuristic_supports_tools(model_id: str) -> bool:
        """Advisory: does the model likely support tool calling?

        vLLM's actual tool-calling requires ``--enable-auto-tool-choice``
        and a matching ``--tool-call-parser`` at server launch, both
        invisible to clients. This heuristic biases toward ``True`` for
        known tool-capable families so callers aren't blocked; the
        server's real response wins on mismatch.
        """
        m = model_id.lower()
        return any(sub in m for sub in _TOOL_CAPABLE_SUBSTRINGS)

    @staticmethod
    def _heuristic_supports_vision(model_id: str) -> bool:
        """Advisory: does the model likely accept image input?"""
        m = model_id.lower()
        return any(sub in m for sub in _VISION_CAPABLE_SUBSTRINGS)

    # ------------------------------------------------------------------
    # Context length
    # ------------------------------------------------------------------

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context window for a vLLM-served model.

        Four-tier resolution:

        1. Cached ``max_model_len`` from a prior
           :meth:`get_models_details` call.
        2. User-supplied ``fallback_context_length`` from config.
        3. Model card registry lookup (best-effort).
        4. ``4096`` with a warning log (emitted at most once per model).

        Args:
            model: Target model id. If None, uses ``self.default_model``.

        Returns:
            The context length in tokens.
        """
        model_name = model or self.default_model

        # 1. Dynamically discovered value.
        cached = self._vllm_model_cache.get(model_name)
        if cached is not None:
            return cached

        # 2. Config override.
        if self._fallback_context_length is not None:
            return self._fallback_context_length

        # 3. Registry fallback.
        try:
            from ..model_cards.registry import get_model_card_registry

            registry = get_model_card_registry()
            card = registry.get(self.get_name(), model_name)
            if card is not None:
                try:
                    return card.get_context_length()
                except Exception:  # pragma: no cover - defensive
                    pass
        except Exception:  # pragma: no cover - registry may not be loaded
            pass

        # 4. Last resort.
        logger.warning(
            "Unknown context length for vLLM model '%s'. Falling back to "
            "4096. Call get_models_details() before generation to discover "
            "the authoritative value, or set fallback_context_length in "
            "config.",
            model_name,
        )
        return 4096

    # ------------------------------------------------------------------
    # Supported parameters
    # ------------------------------------------------------------------

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the parameter schema for the vLLM chat endpoint.

        Starts from :meth:`OpenAIProvider.get_supported_parameters` and:

        - Removes OpenAI-only keys (see :data:`_OPENAI_ONLY_KEYS`).
        - Adds vLLM-specific generation knobs.
        - Declares ``extra_body`` so :meth:`chat_completion` can smuggle
          vLLM extensions past the parent's whitelist check.

        Args:
            model: Target model id; currently unused for vLLM since the
                schema is the same across all served models. Accepted
                for signature compatibility.

        Returns:
            A dict describing the union of OpenAI-standard and
            vLLM-specific kwargs. Values are JSON-Schema-like snippets.
        """
        params = super().get_supported_parameters(model)

        # Strip parameters vLLM does not implement.
        for key in _OPENAI_ONLY_KEYS:
            params.pop(key, None)

        # Declare ``extra_body`` so ``chat_completion`` can pass
        # vLLM-specific top-level JSON fields through the SDK without
        # tripping the parent's whitelist.
        params.setdefault("extra_body", {"type": "object"})

        # vLLM-specific kwargs. Each must be accepted by the parent's
        # whitelist check (or moved into ``extra_body`` by our override
        # of ``chat_completion``).
        vllm_extensions: dict[str, dict[str, Any]] = {
            # Structured-output constraints (exactly one of these may be
            # set; vLLM enforces the mutual exclusion server-side).
            "structured_outputs": {
                "type": "object",
                "description": (
                    "vLLM StructuredOutputsParams. One of: json (schema), "
                    "regex, choice (list), grammar (lark/gbnf), "
                    "json_object (bool), structural_tag."
                ),
            },
            # Legacy flat guided_* knobs still accepted by vLLM.
            "guided_json": {
                "type": ["object", "string"],
                "description": "JSON schema (dict or stringified) for constrained decoding.",
            },
            "guided_regex": {"type": "string"},
            "guided_choice": {"type": "array", "items": {"type": "string"}},
            "guided_grammar": {"type": "string"},
            "guided_decoding_backend": {
                "type": "string",
                "enum": ["xgrammar", "guidance", "outlines", "lm-format-enforcer"],
            },
            # Advanced sampling.
            "prompt_logprobs": {
                "type": "integer",
                "minimum": 0,
                "maximum": 32,
                "description": "Return per-token logprobs over the prompt.",
            },
            "min_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_k": {"type": "integer"},
            "repetition_penalty": {"type": "number"},
            "length_penalty": {"type": "number"},
            "min_tokens": {"type": "integer", "minimum": 0},
            "ignore_eos": {"type": "boolean"},
            "include_stop_str_in_output": {"type": "boolean"},
            # Tokenisation / chat-template plumbing.
            "skip_special_tokens": {"type": "boolean"},
            "spaces_between_special_tokens": {"type": "boolean"},
            "add_generation_prompt": {"type": "boolean"},
            "continue_final_message": {"type": "boolean"},
            "echo": {"type": "boolean"},
            "chat_template": {"type": "string"},
            "chat_template_kwargs": {"type": "object"},
            # Multimodal processor kwargs.
            "mm_processor_kwargs": {"type": "object"},
            # Disaggregated-serving hint (advanced).
            "kv_transfer_params": {"type": "object"},
        }
        params.update(vllm_extensions)
        return params

    # ------------------------------------------------------------------
    # Chat completion override
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
        """Dispatch a chat completion to vLLM's OpenAI-compatible endpoint.

        vLLM-specific kwargs (see :data:`_VLLM_EXTRA_BODY_KEYS`) are
        moved into ``extra_body`` so the OpenAI SDK forwards them as
        top-level JSON fields. All other logic — message payload
        construction, streaming, tool-call extraction, error mapping,
        observability — is inherited unchanged from
        :class:`OpenAIProvider`.

        Args:
            context: List of :class:`~llmcore.models.Message`.
            model: Target model id; falls back to ``self.default_model``.
            stream: If True, returns an async generator of chunk dicts.
            tools: Optional tool definitions.
            tool_choice: Optional tool-choice directive.
            **kwargs: Standard generation params + any vLLM extension
                declared in :meth:`get_supported_parameters`.

        Returns:
            A dict (non-streaming) or async generator of dicts
            (streaming), in OpenAI-normalised shape.

        Raises:
            ProviderError: On API / connection / timeout errors.
            ValueError: If a kwarg is not declared in
                :meth:`get_supported_parameters`.
            ContextLengthError: When vLLM reports the request exceeds
                ``max_model_len`` (mapping inherited from parent).
        """
        # Pop vLLM-specific keys from kwargs and merge into extra_body.
        # The parent's whitelist check will then see only
        # OpenAI-standard keys plus ``extra_body`` itself.
        extra_body: dict[str, Any] = {}
        caller_extra = kwargs.pop("extra_body", None)
        if isinstance(caller_extra, dict):
            extra_body.update(caller_extra)

        for key in list(kwargs.keys()):
            if key in _VLLM_EXTRA_BODY_KEYS:
                extra_body[key] = kwargs.pop(key)

        if extra_body:
            kwargs["extra_body"] = extra_body

        return await super().chat_completion(
            context,
            model=model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Token counting (approximation)
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Approximate token count for raw text.

        vLLM serves arbitrary models whose tokenisers are not
        ``tiktoken``-compatible, so we fall back to a 4-character
        heuristic. This is advisory (used for context budgeting, not
        billing). Exact counts via vLLM's ``/tokenize`` endpoint are
        deferred to a follow-up PR.
        """
        if not text:
            return 0
        return (len(text) + 3) // 4

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Approximate token count for a message list.

        Mirrors the non-tiktoken branch of
        :meth:`OpenAIProvider.count_message_tokens`.
        """
        if not messages:
            return 0
        total = 0
        for m in messages:
            role_str = m.role.value if hasattr(m.role, "value") else str(m.role)
            total += len(m.content) + len(role_str)
        # ~15 overhead tokens per message, divided by 4 chars/token.
        return (total + len(messages) * 15) // 4
