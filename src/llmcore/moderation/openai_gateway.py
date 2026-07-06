# src/llmcore/moderation/openai_gateway.py
"""
OpenAI moderation gateway (plan SF-1 / DDS-06).

Calls the OpenAI ``/moderations`` endpoint through the same ``openai`` SDK
the chat provider uses (see :mod:`llmcore.providers.openai_provider` for
the client-construction pattern this mirrors: ``api_key`` /
``api_key_env_var`` / ``OPENAI_API_KEY`` resolution, ``timeout`` and
``max_retries`` on the ``AsyncOpenAI`` client).

All backend failures — timeouts, connection faults, API errors, malformed
responses — surface as :class:`llmcore.exceptions.ModerationError` so the
policy's fail-safe can convert them into BLOCK decisions.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

try:
    from openai import AsyncOpenAI
    from openai._exceptions import (
        APIConnectionError as OpenAIAPIConnectionError,
    )
    from openai._exceptions import (
        APIStatusError as OpenAIAPIStatusError,
    )
    from openai._exceptions import (
        APITimeoutError as OpenAIAPITimeoutError,
    )
    from openai._exceptions import (
        OpenAIError,
    )

    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    OpenAIAPIStatusError = Exception  # type: ignore
    OpenAIAPIConnectionError = Exception  # type: ignore
    OpenAIAPITimeoutError = Exception  # type: ignore

from ..exceptions import ConfigError, ModerationError
from .models import ModerationAction, ModerationResult

__all__ = ["DEFAULT_MODERATION_MODEL", "OpenAIModerationGateway"]

logger = logging.getLogger(__name__)

#: Current omni moderation alias per the installed openai SDK
#: (``openai.types.ModerationModel``).
DEFAULT_MODERATION_MODEL = "omni-moderation-latest"

_GATEWAY_NAME = "openai"


def _normalize_category(name: str) -> str:
    """Normalize an API category name to snake_case.

    The moderation API uses names like ``"self-harm/intent"``; the SDK's
    pydantic models use ``self_harm_intent``. Policies key thresholds on
    the snake_case form (TOML-friendly), so both spellings map to it.
    """
    return name.replace("/", "_").replace("-", "_")


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """Coerce an SDK pydantic model / plain dict to a mapping (or empty)."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
        except Exception:
            return {}
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


class OpenAIModerationGateway:
    """:class:`~llmcore.moderation.gateway.ModerationGateway` backed by OpenAI.

    Config keys (all optional except an API key from *some* source):
        api_key: Explicit API key (highest precedence).
        api_key_env_var: Environment variable to read the key from.
            ``OPENAI_API_KEY`` is the final fallback either way.
        model: Moderation model (default ``omni-moderation-latest``).
        base_url: Alternative endpoint (e.g. a compatible proxy).
        timeout: Request timeout in seconds (default 30.0).
        max_retries: SDK-level retries (default 2).

    Raises:
        ImportError: When the ``openai`` package is not installed.
        ConfigError: When no API key can be resolved or the client cannot
            be constructed. Construction failures are loud on purpose —
            an enabled moderation gateway must never silently degrade.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = config or {}
        if not openai_available:
            raise ImportError("OpenAI library not installed (required for moderation gateway).")

        api_key = config.get("api_key")
        api_key_env_var = config.get("api_key_env_var")
        if not api_key and api_key_env_var:
            api_key = os.environ.get(str(api_key_env_var))
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError(
                "OpenAI API key not found for moderation. "
                "Set OPENAI_API_KEY or configure [moderation] api_key / api_key_env_var."
            )

        self.model = str(config.get("model") or DEFAULT_MODERATION_MODEL)
        self.timeout = float(config.get("timeout", 30.0))
        self.max_retries = int(config.get("max_retries", 2))

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        base_url = config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url
        try:
            self._client = AsyncOpenAI(**client_kwargs)
        except Exception as e:
            raise ConfigError(f"OpenAI moderation client initialization failed: {e}") from e

    def get_name(self) -> str:
        return _GATEWAY_NAME

    async def check(self, text: str, *, context: str = "") -> ModerationResult:
        """Classify *text* via the OpenAI moderations endpoint.

        Args:
            text: Content to classify.
            context: Surface tag, used only for logging.

        Returns:
            The normalized :class:`ModerationResult`.

        Raises:
            ModerationError: On any backend failure (timeout, connection,
                API error, malformed response). Callers going through
                :func:`llmcore.moderation.moderate` get fail-safe handling
                automatically.
        """
        try:
            response = await self._client.moderations.create(model=self.model, input=text)
        except OpenAIAPITimeoutError as e:
            raise ModerationError(_GATEWAY_NAME, f"Timeout: {e}") from e
        except OpenAIAPIConnectionError as e:
            raise ModerationError(_GATEWAY_NAME, f"Connection error: {e}") from e
        except OpenAIAPIStatusError as e:
            raise ModerationError(
                _GATEWAY_NAME, f"API Error ({e.status_code}): {e}"
            ) from e
        except OpenAIError as e:
            raise ModerationError(_GATEWAY_NAME, f"Error: {e}") from e
        except Exception as e:
            raise ModerationError(_GATEWAY_NAME, f"Unexpected error: {e}") from e

        logger.debug("Moderation check (model=%s, context=%r, chars=%d)", self.model, context, len(text))
        return self._to_result(response)

    def _to_result(self, response: Any) -> ModerationResult:
        """Map a ``ModerationCreateResponse`` (or dict equivalent) to a result.

        Raises:
            ModerationError: When the response carries no results entry.
        """
        if isinstance(response, Mapping):
            results = response.get("results")
            response_model = response.get("model", "")
        else:
            results = getattr(response, "results", None)
            response_model = getattr(response, "model", "")
        if not results:
            raise ModerationError(_GATEWAY_NAME, "Malformed moderation response: no results.")
        first = results[0]

        if isinstance(first, Mapping):
            flagged = bool(first.get("flagged", False))
            raw_scores = first.get("category_scores")
            raw_flags = first.get("categories")
        else:
            flagged = bool(getattr(first, "flagged", False))
            raw_scores = getattr(first, "category_scores", None)
            raw_flags = getattr(first, "categories", None)

        categories: dict[str, float] = {}
        for name, score in _as_mapping(raw_scores).items():
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                categories[_normalize_category(str(name))] = float(score)

        flagged_categories = tuple(
            sorted(
                _normalize_category(str(name))
                for name, is_flagged in _as_mapping(raw_flags).items()
                if is_flagged
            )
        )

        return ModerationResult(
            flagged=flagged,
            categories=categories,
            flagged_categories=flagged_categories,
            action_hint=ModerationAction.BLOCK if flagged else None,
            provider=_GATEWAY_NAME,
            model=str(response_model or self.model),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                await client.close()
            except Exception as e:
                logger.debug("Error closing moderation client: %s", e)
            finally:
                self._client = None
