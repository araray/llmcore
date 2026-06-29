"""Token-counting utilities for LLMCore.

LLMCore owns model-aware token accounting because providers, model names,
context windows, and agent budgeting all live in this package. The public
``count_tokens`` helper prefers tiktoken when available and falls back to a
deterministic character-ratio estimate for offline or minimal environments.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_DEFAULT_ENCODING = "cl100k_base"
_FRONTIER_ENCODING = "o200k_base"
_O200K_MODEL_SUBSTRINGS = ("gpt-5", "gpt-4.1", "4o", "o1", "o3", "o4")
_warned_keys: set[str] = set()


class TokenCounter(Protocol):
    """Protocol implemented by token counters."""

    def count(self, text: str | None) -> int:
        """Return the estimated or exact token count for ``text``."""


class EstimateCounter:
    """Character-ratio token estimator used when tiktoken is unavailable."""

    def __init__(self, chars_per_token: int = _CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than zero")
        self.chars_per_token = chars_per_token

    def count(self, text: str | None) -> int:
        """Estimate token count using ceiling division over character length."""
        if not text:
            return 0
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token


class TiktokenCounter:
    """Token counter backed by a tiktoken encoding."""

    def __init__(self, model: str | None = None) -> None:
        encoding = _load_encoding(model)
        if encoding is None:
            raise RuntimeError("tiktoken is not available; use EstimateCounter instead")
        self.model = model
        self._encoding = encoding

    def count(self, text: str | None) -> int:
        """Count tokens with the selected tiktoken encoding."""
        if not text:
            return 0
        return len(self._encoding.encode(text))


def count_tokens(text: str | None, model: str | None = None) -> int:
    """Count tokens for ``text`` with precise and approximate fallbacks."""

    return get_counter(model).count(text)


def get_counter(model: str | None = None) -> TokenCounter:
    """Return the best available counter for ``model``."""

    encoding = _load_encoding(model)
    if encoding is None:
        return EstimateCounter()

    counter = object.__new__(TiktokenCounter)
    counter.model = model
    counter._encoding = encoding
    return counter


def _load_encoding(model: str | None) -> Any | None:
    tiktoken = _load_tiktoken()
    if tiktoken is None:
        _warn_once(
            "tiktoken-missing",
            "tiktoken is not available; using character-based token estimation",
        )
        return None

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            fallback_name = _fallback_encoding_name(model)
            _warn_once(
                f"unknown-model:{model}:{fallback_name}",
                "No tiktoken encoding for model %r; using %s",
                model,
                fallback_name,
            )
            return _get_encoding_or_none(tiktoken, fallback_name)
        except Exception as exc:  # pragma: no cover - defensive around third-party code
            _warn_once(
                f"model-load-failed:{model}",
                "Failed to load tiktoken encoding for model %r (%s); using estimate",
                model,
                exc,
            )
            return None

    return _get_encoding_or_none(tiktoken, _DEFAULT_ENCODING)


def _get_encoding_or_none(tiktoken: Any, encoding_name: str) -> Any | None:
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # pragma: no cover - defensive around third-party code
        _warn_once(
            f"encoding-load-failed:{encoding_name}",
            "Failed to load tiktoken encoding %r (%s); using estimate",
            encoding_name,
            exc,
        )
        return None


def _load_tiktoken() -> Any | None:
    try:
        return importlib.import_module("tiktoken")
    except Exception:
        return None


def _fallback_encoding_name(model: str | None) -> str:
    if model:
        lowered = model.lower()
        if any(marker in lowered for marker in _O200K_MODEL_SUBSTRINGS):
            return _FRONTIER_ENCODING
    return _DEFAULT_ENCODING


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(message, *args)


__all__ = [
    "EstimateCounter",
    "TiktokenCounter",
    "TokenCounter",
    "count_tokens",
    "get_counter",
]
