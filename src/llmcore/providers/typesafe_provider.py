# src/llmcore/providers/typesafe_provider.py
"""TypeSafe.ai System One provider for llmcore.

TypeSafe.ai (https://docs.typesafe.ai) is **not** a chat/completion API. Its
System One models (``jev-*``) evaluate a piece of *state* — a string, a JSON
object, or an array — against a map of *typed questions* and return structured,
calibrated answers that code can consume directly:

* **noul** — a yes/no question; the answer is the probability of *yes*.
* **choice** — pick one option from a set you define; the answer carries the
  chosen option, a probability for every option, and a ``confidence``.
* **score** — rate the state along an ordered rubric; the answer carries a
  probability-weighted score, the legend, per-level probabilities, and a
  ``confidence``.

The provider therefore mirrors the Deepgram (speech) precedent: it satisfies
the :class:`~llmcore.providers.base.BaseProvider` contract honestly, but its
real surface is :meth:`TypeSafeProvider.system_one`. A thin **chat bridge**
lets generic callers reach it through ``LLMCore.chat(..., provider_name=
"typesafe", questions={...})``: the conversation becomes the ``state`` and
the answers come back as a JSON string.

Transport is plain ``httpx`` against the two REST endpoints
(``POST /v1/systemone``, ``GET /v1/models``); no vendor SDK is required.
Retries for ``408/429/5xx/529`` honour ``retry-after-ms`` / ``Retry-After``
and use exponential backoff with jitter, matching the official SDK defaults.

Configuration (``[providers.typesafe]``)::

    api_key                 # or env TYPESAFE_API_KEY (api_key_env_var overrides the name)
    base_url                # default https://api.typesafe.ai (or env TYPESAFE_BASE_URL)
    default_model           # default "jev-latest"          (or env TYPESAFE_DEFAULT_MODEL)
    timeout                 # seconds per HTTP operation (default 30)
    max_retries             # retries after the first attempt (default 2)
    retry_backoff_initial   # seconds (default 0.5)
    retry_backoff_max       # seconds (default 5.0)
    fallback_context_length # tokens reported when no model card matches (default 65536)

Example::

    from llmcore.providers.typesafe_provider import Choice, Noul, Score

    provider = llm._provider_manager.get_provider("typesafe")
    result = await provider.system_one(
        state="Hi, my payouts have been failing for 3 days. Please help ASAP.",
        questions={
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments, invoicing, refunds",
                          "technical": "Bugs, outages, integrations"},
            ),
            "frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
            "is_urgent": Noul(instructions="Does this convey urgency?"),
        },
    )
    result.choices["department"].choice      # "technical"
    result.scores["frustration"].score       # 1.6
    result.nouls["is_urgent"].noul           # 0.92
    result.usage.input_tokens                # 312

Docs: https://docs.typesafe.ai  (API reference: https://docs.typesafe.ai/api)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import BaseProvider, ContextPayload, flatten_tool_messages_for_text_protocol

try:  # httpx is the only transport dependency (``pip install llmcore[typesafe]``).
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover - exercised via the ImportError test
    httpx = None  # type: ignore[assignment]
    httpx_available = False

try:  # Optional: tiktoken gives a closer token estimate than the char heuristic.
    import tiktoken

    tiktoken_available = True
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    tiktoken_available = False

logger = logging.getLogger(__name__)

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "JSONContent",
    "Noul",
    "NoulAnswer",
    "Score",
    "ScoreAnswer",
    "SystemOneResult",
    "SystemOneUsage",
    "TypeSafeModelInfo",
    "TypeSafeProvider",
    "normalize_questions",
]

# =============================================================================
# Constants
# =============================================================================

_PROVIDER_NAME = "typesafe"
_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_MODEL = "jev-latest"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_BACKOFF_INITIAL = 0.5
_DEFAULT_BACKOFF_MAX = 5.0
_DEFAULT_BACKOFF_JITTER = 0.25
#: Hard ceiling for a server-requested ``Retry-After`` wait (seconds).
_MAX_RETRY_AFTER = 60.0
#: Jev 1.13: 64k tokens per request (state + all questions).
_DEFAULT_FALLBACK_CONTEXT = 65536
#: Jev 1.13: 32k tokens for the state plus the single longest question.
_STATE_PLUS_QUESTION_LIMIT = 32768

_SYSTEM_ONE_PATH = "/v1/systemone"
_MODELS_PATH = "/v1/models"

_ENV_API_KEY = "TYPESAFE_API_KEY"
_ENV_BASE_URL = "TYPESAFE_BASE_URL"
_ENV_DEFAULT_MODEL = "TYPESAFE_DEFAULT_MODEL"

_REQUEST_ID_HEADER = "x-typesafe-request-id"
_RETRY_AFTER_HEADER = "retry-after"
_RETRY_AFTER_MS_HEADER = "retry-after-ms"
_RETRY_COUNT_HEADER = "X-TypeSafe-Retry-Count"

#: Statuses retried in-provider (mirrors the official SDK: 408, 429, 5xx incl. 529).
_RETRY_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504, 529})
#: Statuses that are never retried and map to ``retryable=False``.
_NON_RETRYABLE_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 422})

_QUESTION_TYPES: frozenset[str] = frozenset({"noul", "choice", "score"})
_MAX_ERROR_BODY = 500

#: JSON content accepted for ``state``, ``instructions`` and criteria descriptions.
JSONContent = str | dict[str, Any] | list[Any]


def _llmcore_version() -> str:
    """Return the installed llmcore version for the ``User-Agent`` header."""
    try:
        return _pkg_version("llmcore")
    except PackageNotFoundError:  # pragma: no cover - source-tree runs
        return "unknown"


# =============================================================================
# Question builders (request side)
# =============================================================================


class _Question(BaseModel):
    """Common base for the three question builders.

    Unknown fields are rejected so typos surface locally instead of as a 422.
    Optional fields left at ``None`` are omitted from the wire form; ``None``
    values *inside* ``criteria`` (an undescribed choice) are preserved.
    """

    model_config = ConfigDict(extra="forbid")

    instructions: JSONContent | None = None
    """What the model should decide, as text, a JSON object, or an array."""

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the JSON dict sent in the ``questions`` map."""
        return {k: v for k, v in self.model_dump(mode="json").items() if v is not None}


class Noul(_Question):
    """A yes/no question. The answer is the probability (0-1) that the answer is *yes*.

    Args:
        instructions: The yes/no question or statement to evaluate.
        criteria: Optional ``{"true": ..., "false": ...}`` descriptions of what
            a yes (value near 1) and a no (value near 0) mean.

    See https://docs.typesafe.ai/primitives/noul
    """

    type: Literal["noul"] = "noul"
    criteria: dict[str, JSONContent | None] | None = None

    @field_validator("criteria")
    @classmethod
    def _check_criteria(
        cls, value: dict[str, JSONContent | None] | None
    ) -> dict[str, JSONContent | None] | None:
        if value is None:
            return None
        unknown = set(value) - {"true", "false"}
        if unknown:
            raise ValueError(
                f"Noul criteria accept only the keys 'true' and 'false' (got {sorted(unknown)})."
            )
        return value


class Choice(_Question):
    """Pick one option from a set you define.

    Args:
        instructions: What the model should decide.
        criteria: Map of option name → description (``None`` when the name is
            self-explanatory). At least one option is required; include a
            "none of the above" option when nothing may fit.

    The answer carries ``choice`` (highest-probability option), ``probabilities``
    for every option, and ``confidence``. See https://docs.typesafe.ai/primitives/choice
    """

    type: Literal["choice"] = "choice"
    criteria: dict[str, JSONContent | None]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, value: dict[str, JSONContent | None]) -> dict[str, JSONContent | None]:
        if not value:
            raise ValueError("Choice criteria must define at least one option.")
        return value


class Score(_Question):
    """Rate the state along an ordered rubric.

    Args:
        instructions: What the model should rate.
        criteria: Ordered level descriptions; the position is the score,
            starting at 0. The API expects at least two levels.

    The answer carries the probability-weighted ``score`` (may fall between
    levels), the ``legend``, per-level ``probabilities``, and ``confidence``.
    See https://docs.typesafe.ai/primitives/score
    """

    type: Literal["score"] = "score"
    criteria: list[JSONContent]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, value: list[JSONContent]) -> list[JSONContent]:
        if not value:
            raise ValueError(
                "Score criteria must define at least one level (two or more recommended)."
            )
        return value


QuestionLike = Noul | Choice | Score | Mapping[str, Any]


def normalize_questions(questions: Mapping[str, QuestionLike]) -> dict[str, dict[str, Any]]:
    """Validate a questions map and convert builders to their wire dicts.

    Raw dicts are accepted (and passed through, so future fields survive) as
    long as they carry a known ``type`` and the criteria that type requires.

    Args:
        questions: Map of question id → :class:`Noul` / :class:`Choice` /
            :class:`Score` or an equivalent dict.

    Returns:
        A new dict with plain JSON-serialisable question dicts.

    Raises:
        ValueError: On an empty map, an unknown question type, or missing/empty
            criteria for ``choice``/``score`` questions.
    """
    if not questions:
        raise ValueError("questions must contain at least one question.")
    wire: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Question ids must be non-empty strings.")
        if isinstance(question, (Noul, Choice, Score)):
            wire[name] = question.to_wire()
            continue
        if not isinstance(question, Mapping):
            raise ValueError(
                f"Question {name!r} must be a Noul/Choice/Score or a dict with a 'type' key "
                f"(got {type(question).__name__})."
            )
        qtype = question.get("type")
        if qtype not in _QUESTION_TYPES:
            raise ValueError(
                f"Question {name!r} has unsupported type {qtype!r}; expected one of "
                f"{sorted(_QUESTION_TYPES)}."
            )
        if qtype in ("choice", "score"):
            criteria = question.get("criteria")
            if not criteria:
                raise ValueError(f"Question {name!r} ({qtype}) requires non-empty 'criteria'.")
            if qtype == "choice" and not isinstance(criteria, Mapping):
                raise ValueError(
                    f"Question {name!r}: choice criteria must be a mapping of option → description."
                )
            if qtype == "score" and isinstance(criteria, (str, bytes, Mapping)):
                raise ValueError(
                    f"Question {name!r}: score criteria must be an ordered list of levels."
                )
        wire[name] = dict(question)
    return wire


# =============================================================================
# Answer models (response side)
# =============================================================================


class NoulAnswer(BaseModel):
    """Answer to a :class:`Noul` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"] = "noul"
    noul: float
    """Probability of *yes* / *true*, from 0 to 1 (≈0.5 means genuinely unsure)."""


class ChoiceAnswer(BaseModel):
    """Answer to a :class:`Choice` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"] = "choice"
    choice: str
    """The highest-probability option."""
    probabilities: dict[str, float]
    """Every option mapped to its probability (sums to ≈1)."""
    confidence: float
    """How certain the model is (0-1), derived from the distribution's shape."""


class ScoreAnswer(BaseModel):
    """Answer to a :class:`Score` question.

    ``legend`` and ``probabilities`` are keyed by the integer level (the wire
    format uses string keys; they are coerced here for ergonomic use).
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["score"] = "score"
    score: float
    """Probability-weighted score across the levels; may fall between levels."""
    legend: dict[int, JSONContent]
    """Each level mapped back to its description."""
    probabilities: dict[int, float]
    """Each level mapped to its probability (sums to ≈1)."""
    confidence: float
    """How certain the model is (0-1), derived from the distribution's shape."""


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
"""An answer to a single question, discriminated on ``type``."""


class SystemOneUsage(BaseModel):
    """Token usage for one ``/v1/systemone`` request. Output tokens are free."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Input + output tokens (missing values count as 0)."""
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class TypeSafeModelInfo(BaseModel):
    """One entry from ``GET /v1/models``."""

    model_config = ConfigDict(extra="allow")

    name: str
    """Model id or alias accepted by the ``model`` field (e.g. ``jev-latest``)."""
    description: str | None = None
    release_date: str | None = None


class SystemOneResult(BaseModel):
    """Typed result of :meth:`TypeSafeProvider.system_one`.

    Attributes:
        model: The versioned model id that answered (e.g. ``jev-1.13.0``), even
            when the request used an alias.
        answers: Every answer keyed by the question id you chose.
        usage: Token usage.
        request_id: The ``x-typesafe-request-id`` response header, if present.
        raw: The raw decoded JSON body (includes answer types this version of
            llmcore does not model, which are dropped from ``answers``).
    """

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: SystemOneUsage = Field(default_factory=SystemOneUsage)
    request_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Yes/no answers keyed by question id."""
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Choice answers keyed by question id."""
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Score answers keyed by question id."""
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}

    def answers_dict(self) -> dict[str, dict[str, Any]]:
        """Answers as plain JSON-serialisable dicts (string keys, wire shape)."""
        return {k: v.model_dump(mode="json") for k, v in self.answers.items()}

    def answers_json(self, **dumps_kwargs: Any) -> str:
        """Answers serialised as a JSON string (used by the chat bridge)."""
        dumps_kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.answers_dict(), **dumps_kwargs)


# =============================================================================
# Error helpers
# =============================================================================


def _extract_error_message(body: Any) -> str | None:
    """Pull a human-readable message out of a TypeSafe error body.

    Handles the shapes the API is known to emit: ``{"detail": {"message": ...}}``
    (401), ``{"detail": [{"loc": [...], "msg": ...}, ...]}`` (422 validation),
    plus the generic ``error`` / ``message`` / string forms.
    """
    if isinstance(body, str):
        return body or None
    if not isinstance(body, dict):
        return None
    error, message, detail = body.get("error"), body.get("message"), body.get("detail")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(message, str):
        return message
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, list):
        parts: list[str] = []
        for entry in detail:
            if not isinstance(entry, dict) or not isinstance(entry.get("msg"), str):
                continue
            loc = entry.get("loc")
            path = (
                ".".join(str(item) for item in loc if item != "body")
                if isinstance(loc, list)
                else ""
            )
            parts.append(f"{path}: {entry['msg']}" if path else entry["msg"])
        return "; ".join(parts) or None
    return None


def _parse_retry_after(headers: Any) -> float | None:
    """Return the server-requested wait in seconds from ``retry-after-ms`` / ``Retry-After``.

    ``retry-after-ms`` wins when present. ``Retry-After`` may be an integer
    number of seconds or an HTTP-date. Returns ``None`` when neither header is
    usable.
    """
    if headers is None:
        return None
    raw_ms = headers.get(_RETRY_AFTER_MS_HEADER)
    if raw_ms is not None:
        try:
            value = float(str(raw_ms).strip() or "0")
            if math.isfinite(value) and value >= 0:
                return value / 1000.0
        except ValueError:
            pass
    raw = headers.get(_RETRY_AFTER_HEADER)
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        value = float(text or "0")
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(text).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None
    if math.isfinite(value) and value >= 0:
        return value
    return None


# =============================================================================
# Provider
# =============================================================================


class TypeSafeProvider(BaseProvider):
    """TypeSafe.ai System One provider (typed judgments, not chat).

    Satisfies the :class:`~llmcore.providers.base.BaseProvider` contract for a
    decision provider: :meth:`system_one` and :meth:`list_models` are the real
    surface; :meth:`chat_completion` is a bridge that requires a ``questions``
    kwarg and returns the answers as JSON content; token/context methods return
    documented heuristics.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the provider.

        Args:
            config: The ``[providers.typesafe]`` configuration dictionary. Keys:

                * ``api_key`` / ``api_key_env_var`` — API key, or the env var
                  holding it (default ``TYPESAFE_API_KEY``).
                * ``base_url`` — API root (default ``https://api.typesafe.ai``;
                  env ``TYPESAFE_BASE_URL`` is honoured when unset).
                * ``default_model`` — default ``jev-latest`` (env
                  ``TYPESAFE_DEFAULT_MODEL`` is honoured when unset).
                * ``timeout`` — seconds per HTTP operation (default 30).
                * ``max_retries`` — retries after the first attempt (default 2).
                * ``retry_backoff_initial`` / ``retry_backoff_max`` — backoff
                  window in seconds (defaults 0.5 / 5.0).
                * ``fallback_context_length`` — context length reported when no
                  model card matches (default 65536).
                * ``headers`` — optional extra headers sent on every request.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ImportError: If ``httpx`` is not installed.
            ConfigError: If no API key can be resolved or a numeric knob is invalid.
        """
        super().__init__(config, log_raw_payloads)
        if not httpx_available:
            raise ImportError(
                "The 'httpx' package is required for the TypeSafe.ai provider. "
                "Install with 'pip install llmcore[typesafe]'."
            )

        # --- Credentials: config > named env var > TYPESAFE_API_KEY ---
        env_var = config.get("api_key_env_var") or _ENV_API_KEY
        api_key = config.get("api_key") or (os.environ.get(env_var) or "").strip()
        if not api_key and env_var != _ENV_API_KEY:
            api_key = (os.environ.get(_ENV_API_KEY) or "").strip()
        if not api_key:
            raise ConfigError(
                "TypeSafe.ai API key not found. Set the TYPESAFE_API_KEY environment "
                "variable or configure 'api_key' / 'api_key_env_var' in [providers.typesafe]."
            )
        self._api_key: str = api_key

        # --- Endpoint / model (config > env > default) ---
        base_url = config.get("base_url") or (os.environ.get(_ENV_BASE_URL) or "").strip()
        self._base_url: str = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        default_model = (
            config.get("default_model") or (os.environ.get(_ENV_DEFAULT_MODEL) or "").strip()
        )
        self.default_model: str = default_model or _DEFAULT_MODEL

        # --- Transport / retry knobs ---
        self._timeout: float = self._positive_float(
            config.get("timeout", _DEFAULT_TIMEOUT), "timeout"
        )
        self._max_retries: int = self._non_negative_int(
            config.get("max_retries", _DEFAULT_MAX_RETRIES), "max_retries"
        )
        self._backoff_initial: float = self._non_negative_float(
            config.get("retry_backoff_initial", _DEFAULT_BACKOFF_INITIAL), "retry_backoff_initial"
        )
        self._backoff_max: float = self._non_negative_float(
            config.get("retry_backoff_max", _DEFAULT_BACKOFF_MAX), "retry_backoff_max"
        )
        self.fallback_context_length: int = int(
            config.get("fallback_context_length", _DEFAULT_FALLBACK_CONTEXT)
        )
        self._default_headers: dict[str, str] = {
            str(k): str(v) for k, v in (config.get("headers") or {}).items()
        }

        # Honesty flags for generic chat-surface consumers.
        self.supports_streaming: bool = False
        self.supports_tools: bool = False

        self._http: Any | None = None
        self._encoding: Any | None = None
        if tiktoken_available:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:  # pragma: no cover - encoding download issues
                logger.debug("tiktoken unavailable for TypeSafe token estimates: %s", exc)

        logger.debug(
            "TypeSafe provider initialised (instance=%s, base_url=%s, model=%s, retries=%d).",
            self._provider_instance_name or _PROVIDER_NAME,
            self._base_url,
            self.default_model,
            self._max_retries,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"[providers.typesafe] {name} must be a number (got {value!r})."
            ) from exc
        if not math.isfinite(result) or result <= 0:
            raise ConfigError(
                f"[providers.typesafe] {name} must be a positive, finite number of seconds."
            )
        return result

    @staticmethod
    def _non_negative_float(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"[providers.typesafe] {name} must be a number (got {value!r})."
            ) from exc
        if not math.isfinite(result) or result < 0:
            raise ConfigError(f"[providers.typesafe] {name} must be a non-negative, finite number.")
        return result

    @staticmethod
    def _non_negative_int(value: Any, name: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"[providers.typesafe] {name} must be an integer (got {value!r})."
            ) from exc
        if result < 0:
            raise ConfigError(f"[providers.typesafe] {name} must be >= 0.")
        return result

    # ------------------------------------------------------------------
    # BaseProvider: identity & capabilities
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the provider instance name (or ``"typesafe"``)."""
        return self._provider_instance_name or _PROVIDER_NAME

    @property
    def base_url(self) -> str:
        """The API root this instance talks to."""
        return self._base_url

    async def get_models_details(self) -> list[ModelDetails]:
        """Return known TypeSafe models as :class:`ModelDetails`.

        Sourced from the builtin model-card registry (provider ``"typesafe"``);
        each card's aliases (``jev-latest``, ``jev-preview``) are emitted as
        their own entries so they show up in ``LLMCore.get_available_models()``.
        Falls back to a static list if the registry yields nothing. No network
        call is made — use :meth:`list_models` for the live listing.
        """
        details: list[ModelDetails] = []
        try:
            from ..model_cards.registry import get_model_card_registry

            registry = get_model_card_registry()
            registry.load()
            for summary in registry.list_cards(provider=_PROVIDER_NAME):
                card = registry.get(_PROVIDER_NAME, summary.model_id)
                aliases = list(getattr(card, "aliases", []) or [])
                tags = list(getattr(summary, "tags", []) or [])
                base_meta = {
                    "tags": tags,
                    "status": getattr(summary, "status", None),
                    "aliases": aliases,
                    "question_types": sorted(_QUESTION_TYPES),
                }
                details.append(
                    ModelDetails(
                        id=summary.model_id,
                        provider_name=self.get_name(),
                        display_name=summary.display_name,
                        context_length=summary.context_length,
                        supports_streaming=False,
                        supports_tools=False,
                        family=summary.model_id.split("-", 1)[0] or None,
                        model_type=summary.model_type,
                        metadata=base_meta,
                    )
                )
                for alias in aliases:
                    details.append(
                        ModelDetails(
                            id=alias,
                            provider_name=self.get_name(),
                            display_name=f"{summary.display_name or summary.model_id} ({alias})",
                            context_length=summary.context_length,
                            supports_streaming=False,
                            supports_tools=False,
                            family=alias.split("-", 1)[0] or None,
                            model_type=summary.model_type,
                            metadata={**base_meta, "alias_of": summary.model_id},
                        )
                    )
        except Exception as exc:  # registry is best-effort
            logger.debug("TypeSafe model-card lookup failed: %s", exc)

        if details:
            return details

        return [
            ModelDetails(
                id=model_id,
                provider_name=self.get_name(),
                context_length=self.fallback_context_length,
                supports_streaming=False,
                supports_tools=False,
                family="jev",
                model_type="decision",
            )
            for model_id in ("jev-1.13.0", "jev-latest", "jev-preview")
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the parameters accepted by :meth:`chat_completion` / ``LLMCore.chat``.

        ``LLMCore.chat`` validates every ``provider_kwargs`` key against this
        map, so it lists exactly the bridge's knobs.

        Args:
            model: Unused (parameters are not model-scoped); interface parity.
        """
        return {
            "questions": {
                "type": "object",
                "required": True,
                "description": (
                    "Map of question id -> Noul | Choice | Score builder or an equivalent "
                    "{'type': 'noul'|'choice'|'score', 'instructions', 'criteria'} dict."
                ),
            },
            "state": {
                "type": ["string", "object", "array"],
                "description": (
                    "Explicit state to evaluate. When omitted, the conversation context is "
                    "sent as an array of {role, content} entries."
                ),
            },
            "timeout": {"type": "number", "description": "Per-call timeout override (seconds)."},
            "extra_body": {"type": "object", "description": "Extra top-level request fields."},
            "extra_headers": {"type": "object", "description": "Extra HTTP headers for this call."},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the request token budget for ``model``.

        Reads ``context.max_input_tokens`` from the model card (aliases resolve),
        otherwise ``fallback_context_length``. Note the tighter per-question
        rule: state + the single longest question must fit in 32k tokens.
        """
        target = model or self.default_model
        try:
            from ..model_cards.registry import get_model_card_registry

            registry = get_model_card_registry()
            registry.load()
            card = registry.get(_PROVIDER_NAME, target)
            if card is not None:
                return int(card.context.max_input_tokens)
        except Exception as exc:  # best-effort
            logger.debug("TypeSafe context-length lookup failed for %s: %s", target, exc)
        return self.fallback_context_length

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _get_http(self) -> Any:
        """Return (lazily creating) the shared ``httpx.AsyncClient``."""
        if self._http is None:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "User-Agent": f"llmcore/{_llmcore_version()}",
                **self._default_headers,
            }
            self._http = httpx.AsyncClient(
                base_url=self._base_url, headers=headers, timeout=self._timeout
            )
        return self._http

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for retry number ``attempt`` (0-based)."""
        if self._backoff_initial <= 0 or self._backoff_max <= 0:
            return 0.0
        exponential = min(self._backoff_initial * (2**attempt), self._backoff_max)
        jitter = exponential * random.uniform(0, _DEFAULT_BACKOFF_JITTER)
        return max(0.0, round(exponential - jitter, 3))

    def _retry_delay(self, headers: Any, attempt: int) -> float:
        """Server-requested wait (``retry-after-ms`` / ``Retry-After``) or backoff."""
        server = _parse_retry_after(headers)
        if server is not None:
            return min(server, _MAX_RETRY_AFTER)
        return self._backoff_delay(attempt)

    def _map_error(
        self, response: Any, *, model: str | None, method: str, path: str
    ) -> ProviderError:
        """Map an unsuccessful HTTP response to a :class:`ProviderError`."""
        status = int(response.status_code)
        headers = response.headers
        request_id = headers.get(_REQUEST_ID_HEADER) if headers is not None else None
        try:
            body: Any = response.json()
        except Exception:
            body = getattr(response, "text", "") or None
        detail = _extract_error_message(body)
        if detail is None and body is not None:
            raw = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
            detail = raw[:_MAX_ERROR_BODY] + ("…" if len(raw) > _MAX_ERROR_BODY else "")

        if status == 401:
            message = (
                "Authentication failed (401). Check TYPESAFE_API_KEY / "
                "[providers.typesafe].api_key."
            )
        elif status == 403:
            message = "Permission denied (403)."
        elif status == 404:
            message = f"Not found (404): {method} {path}."
        elif status == 422:
            message = "Request failed validation (422)."
        elif status == 429:
            message = "Rate limit exceeded (429)."
        elif status == 529:
            message = "TypeSafe is temporarily overloaded (529)."
        elif status >= 500:
            message = f"TypeSafe server error ({status})."
        else:
            message = f"TypeSafe API error ({status})."
        if detail:
            message = f"{message} {detail}"
        if request_id:
            message = f"{message} (request_id={request_id})"

        retryable: bool | None
        if status in _NON_RETRYABLE_STATUSES:
            retryable = False
        elif status in _RETRY_STATUSES:
            retryable = True
        else:
            retryable = None  # let ProviderError infer from the status code
        retry_after = _parse_retry_after(headers) if retryable else None

        return ProviderError(
            self.get_name(),
            message,
            model_name=model,
            status_code=status,
            retryable=retryable,
            retry_after_seconds=retry_after,
            headers=dict(headers) if headers is not None else None,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - SDK-parity kwarg name
        extra_headers: Mapping[str, str] | None = None,
        model: str | None = None,
    ) -> Any:
        """Send one request with in-provider retries; return the successful response.

        Retries ``408/429/5xx/529`` responses, timeouts and connection errors up
        to ``max_retries`` times, sleeping for the server-requested wait or an
        exponential backoff between attempts.

        Raises:
            ProviderError: For non-retryable statuses immediately, or the last
                error once retries are exhausted.
        """
        client = self._get_http()
        attempts = self._max_retries + 1
        last_error: ProviderError | None = None
        for attempt in range(attempts):
            headers: dict[str, str] = dict(extra_headers or {})
            if attempt:
                headers[_RETRY_COUNT_HEADER] = str(attempt)
            kwargs: dict[str, Any] = {}
            if json_body is not None:
                kwargs["json"] = json_body
            if headers:
                kwargs["headers"] = headers
            if timeout is not None:
                kwargs["timeout"] = timeout
            try:
                response = await client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    self.get_name(),
                    f"TypeSafe request timed out after {timeout or self._timeout}s "
                    f"({method} {path}).",
                    model_name=model,
                    retryable=True,
                    original_exception=exc,
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
                raise last_error from exc
            except httpx.HTTPError as exc:
                last_error = ProviderError(
                    self.get_name(),
                    f"TypeSafe connection error ({method} {path}): {exc}",
                    model_name=model,
                    retryable=True,
                    original_exception=exc,
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
                raise last_error from exc

            if response.status_code < 400:
                return response

            error = self._map_error(response, model=model, method=method, path=path)
            if response.status_code in _RETRY_STATUSES and attempt < attempts - 1:
                delay = self._retry_delay(response.headers, attempt)
                logger.warning(
                    "TypeSafe %s %s returned %s; retrying in %.2fs (attempt %d/%d).",
                    method,
                    path,
                    response.status_code,
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                last_error = error
                await asyncio.sleep(delay)
                continue
            raise error

        assert last_error is not None  # loop always sets it before falling through
        raise last_error

    # ------------------------------------------------------------------
    # System One surface
    # ------------------------------------------------------------------

    async def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, QuestionLike],
        *,
        model: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - SDK-parity kwarg name
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> SystemOneResult:
        """Evaluate ``state`` against typed ``questions`` (``POST /v1/systemone``).

        Args:
            state: The content to evaluate — a string, a JSON object, or an
                array. Text only (pre-process images/audio into text first).
            questions: Map of question id → :class:`Noul` / :class:`Choice` /
                :class:`Score` (or equivalent dicts). All questions see the same
                state and are evaluated independently, in parallel.
            model: Model id or alias; defaults to ``default_model``.
            timeout: Per-call timeout override in seconds.
            extra_headers: Extra HTTP headers for this call.
            extra_body: Extra top-level request fields (forward compatibility).

        Returns:
            A :class:`SystemOneResult` with typed answers, usage and request id.

        Raises:
            ValueError: On invalid ``state``/``questions`` before any request.
            ProviderError: On HTTP/transport errors (``status_code`` and
                ``retryable`` set) or an unexpected response shape.
        """
        if state is None:
            raise ValueError("state is required (a string, JSON object, or array).")
        wire_questions = normalize_questions(questions)
        used_model = model or self.default_model
        body: dict[str, Any] = {"state": state, "model": used_model, "questions": wire_questions}
        if extra_body:
            body.update(dict(extra_body))
        if self.log_raw_payloads_enabled:
            logger.debug("TypeSafe request body: %s", json.dumps(body, ensure_ascii=False)[:4000])

        response = await self._request(
            "POST",
            _SYSTEM_ONE_PATH,
            json_body=body,
            timeout=timeout,
            extra_headers=extra_headers,
            model=used_model,
        )
        result = self._parse_system_one(response, model=used_model)
        if self.log_raw_payloads_enabled:
            logger.debug(
                "TypeSafe response body: %s", json.dumps(result.raw, ensure_ascii=False)[:4000]
            )
        return result

    def _parse_system_one(self, response: Any, *, model: str) -> SystemOneResult:
        """Decode and validate a ``/v1/systemone`` response body."""
        try:
            data = response.json()
        except Exception as exc:
            raise ProviderError(
                self.get_name(),
                f"TypeSafe returned a non-JSON body: {getattr(response, 'text', '')[:200]!r}",
                model_name=model,
                status_code=int(getattr(response, "status_code", 200)),
                retryable=False,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                self.get_name(),
                "TypeSafe returned an unexpected response shape (expected a JSON object).",
                model_name=model,
                retryable=False,
            )
        answers = data.get("answers")
        known: dict[str, Any] = {}
        if isinstance(answers, dict):
            for name, raw_answer in answers.items():
                if isinstance(raw_answer, dict) and raw_answer.get("type") in _QUESTION_TYPES:
                    known[name] = raw_answer
                else:
                    # Forward compatibility: keep unknown answer kinds in ``raw`` only.
                    logger.warning(
                        "Ignoring TypeSafe answer %r with unrecognised type %r.",
                        name,
                        raw_answer.get("type") if isinstance(raw_answer, dict) else None,
                    )
        headers = getattr(response, "headers", None)
        request_id = headers.get(_REQUEST_ID_HEADER) if headers is not None else None
        try:
            return SystemOneResult.model_validate(
                {
                    "model": data.get("model") or model,
                    "answers": known,
                    "usage": data.get("usage") or {},
                    "request_id": request_id,
                    "raw": data,
                }
            )
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0] if exc.errors() else {}
            location = ".".join(str(p) for p in first.get("loc", ()))
            raise ProviderError(
                self.get_name(),
                f"TypeSafe response failed validation at {location or '<root>'}: "
                f"{first.get('msg', exc)}",
                model_name=model,
                retryable=False,
                original_exception=exc,
            ) from exc

    async def list_models(
        self,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - SDK-parity kwarg name
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[TypeSafeModelInfo]:
        """Return the models/aliases the account can use (``GET /v1/models``).

        The endpoint currently lists aliases (``jev-latest``, ``jev-preview``);
        versioned ids such as ``jev-1.13.0`` are accepted by ``model`` whether
        or not they appear here.
        """
        response = await self._request(
            "GET", _MODELS_PATH, timeout=timeout, extra_headers=extra_headers
        )
        try:
            data = response.json()
        except Exception as exc:
            raise ProviderError(
                self.get_name(),
                "TypeSafe /v1/models returned a non-JSON body.",
                retryable=False,
            ) from exc
        entries = data.get("models", []) if isinstance(data, dict) else []
        models: list[TypeSafeModelInfo] = []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name"):
                try:
                    models.append(TypeSafeModelInfo.model_validate(entry))
                except ValidationError as exc:
                    logger.debug("Skipping malformed TypeSafe model entry %r: %s", entry, exc)
        return models

    # ------------------------------------------------------------------
    # BaseProvider: chat bridge
    # ------------------------------------------------------------------

    @staticmethod
    def _role_str(role: Any) -> str:
        return str(getattr(role, "value", role))

    def _context_to_state(self, context: ContextPayload) -> list[dict[str, Any]]:
        """Turn the conversation into an array of ``{"role", "content"}`` entries.

        Tool-role messages are flattened into text first (TypeSafe has no tool
        protocol); empty messages are skipped.
        """
        flattened = flatten_tool_messages_for_text_protocol(list(context or []))
        state: list[dict[str, Any]] = []
        for msg in flattened:
            content = getattr(msg, "content", None)
            if content is None or (isinstance(content, str) and not content.strip()):
                continue
            state.append({"role": self._role_str(getattr(msg, "role", "user")), "content": content})
        return state

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat bridge: evaluate the conversation with ``questions=`` and return JSON answers.

        TypeSafe does not generate text, stream, or call tools. The bridge
        requires a ``questions`` kwarg (see :func:`normalize_questions`); the
        conversation (or an explicit ``state`` kwarg) is the state. The result
        is an OpenAI-shaped dict whose assistant ``content`` is the answers map
        serialised as JSON, so ``LLMCore.chat(...)`` returns that JSON string.

        Args:
            context: The conversation messages.
            model: Model id/alias override.
            stream: Must be ``False``.
            tools: Must be empty/``None``.
            tool_choice: Must be ``None``.
            **kwargs: ``questions`` (required), ``state``, ``timeout``,
                ``extra_body``, ``extra_headers``.

        Raises:
            ProviderError: (400, non-retryable) when streaming/tools are
                requested or ``questions`` is missing; otherwise as
                :meth:`system_one`.
        """
        if stream:
            raise ProviderError(
                self.get_name(),
                "TypeSafe.ai does not stream — it returns structured answers in one "
                "response. Call chat_completion(stream=False) or system_one().",
                model_name=model,
                status_code=400,
                retryable=False,
            )
        if tools or tool_choice:
            raise ProviderError(
                self.get_name(),
                "TypeSafe.ai does not support tool calling. Model the decision as "
                "Choice/Score/Noul questions instead (see the function_calling cookbook).",
                model_name=model,
                status_code=400,
                retryable=False,
            )
        questions = kwargs.get("questions")
        if not questions:
            raise ProviderError(
                self.get_name(),
                "TypeSafe.ai is a typed-judgment provider, not a chat model. Pass "
                "questions={...} (Noul/Choice/Score) to chat()/chat_completion(), or call "
                "provider.system_one(state, questions) directly.",
                model_name=model,
                status_code=400,
                retryable=False,
            )
        unsupported = set(kwargs) - set(self.get_supported_parameters(model))
        if unsupported:
            raise ValueError(
                f"Unsupported parameter(s) {sorted(unsupported)} for TypeSafe provider. "
                f"Supported: {sorted(self.get_supported_parameters(model))}."
            )
        state = kwargs.get("state")
        if state is None:
            state = self._context_to_state(context)

        result = await self.system_one(
            state,
            questions,
            model=model,
            timeout=kwargs.get("timeout"),
            extra_headers=kwargs.get("extra_headers"),
            extra_body=kwargs.get("extra_body"),
        )
        usage = result.usage
        prompt_tokens = usage.input_tokens or 0
        completion_tokens = usage.output_tokens or 0
        return {
            "id": f"typesafe-{result.request_id or uuid.uuid4().hex}",
            "object": "systemone.result",
            "created": int(time.time()),
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answers_json()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "typesafe": result.raw,
        }

    # ------------------------------------------------------------------
    # BaseProvider: response helpers
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Return the answers JSON string from a chat-bridge response dict."""
        try:
            content = response["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """TypeSafe never streams; there are no deltas."""
        return ""

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Return usage in the OpenAI-style shape plus TypeSafe's native names."""
        usage = response.get("usage") if isinstance(response, dict) else None
        if not usage:
            return {}
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "input_tokens": prompt,
            "output_tokens": completion,
        }

    def extract_tool_calls(self, response: dict[str, Any]) -> list[Any]:
        """TypeSafe never emits tool calls."""
        return []

    def extract_finish_reason(self, response: dict[str, Any]) -> str | None:
        """Bridge responses always finish with ``"stop"``."""
        try:
            return response["choices"][0].get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError, AttributeError):
            return "stop"

    # ------------------------------------------------------------------
    # BaseProvider: tokens
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Estimate tokens with tiktoken ``cl100k_base`` when available, else ``len/4``.

        TypeSafe does not publish its tokenizer; this is a documented
        approximation for budgeting against the 64k / 32k limits.
        """
        if not text:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:  # pragma: no cover - fall through to heuristic
                pass
        return max(1, len(text) // 4)

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Sum :meth:`count_tokens` over message contents plus a small per-message overhead."""
        total = 0
        for msg in messages or []:
            content = getattr(msg, "content", "") or ""
            total += await self.count_tokens(str(content), model) + 4
        return total

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Cheap readiness check (no network call); logs the configured endpoint."""
        logger.debug(
            "TypeSafe provider ready (instance=%s, base_url=%s, model=%s).",
            self.get_name(),
            self._base_url,
            self.default_model,
        )

    async def close(self) -> None:
        """Close the underlying ``httpx`` client (best-effort, never raises)."""
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as exc:
                logger.error("Error closing TypeSafe HTTP client: %s", exc)
            self._http = None
        logger.info("TypeSafeProvider closed.")
