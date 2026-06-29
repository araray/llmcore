"""Canonical error mapping for the bridge (spec §9).

One place translates llmcore's exception hierarchy into the wire-neutral
:class:`LlmcoreError` proto, and maps the resulting category to a gRPC status
code and an HTTP status. ``BridgeError`` is the single exception type the
transport layers know about; each transport renders it in its own idiom
(gRPC: ``abort`` + trailing ``llmcore-error-bin`` metadata; HTTP: JSON body).
"""

from __future__ import annotations

import asyncio
from typing import Any

import grpc

from ._generated.llmcore.v1 import errors_pb2
from .redact import redact

__all__ = [
    "BridgeError",
    "grpc_status_for",
    "http_status_for",
    "invalid_argument",
    "map_exception",
    "to_llmcore_error",
    "unsupported",
]

_Cat = errors_pb2.ErrorCategory

# --- llmcore exception classes (imported defensively for version tolerance) ---
try:  # pragma: no cover - import shape depends on installed llmcore
    from llmcore.exceptions import (
        ConfigError,
        ContextError,
        ContextLengthError,
        EmbeddingError,
        LLMCoreError,
        ProviderError,
        SearchProviderError,
        SessionNotFoundError,
        StorageError,
    )
except Exception:  # llmcore not importable (e.g. docs build) — define sentinels
    class _Missing(Exception):
        ...

    ConfigError = ContextError = ContextLengthError = EmbeddingError = _Missing  # type: ignore
    LLMCoreError = ProviderError = SearchProviderError = _Missing  # type: ignore
    SessionNotFoundError = StorageError = _Missing  # type: ignore


class BridgeError(Exception):
    """A mapped, transport-neutral error carrying an :class:`LlmcoreError` proto.

    Attributes:
        proto: The populated ``llmcore.v1.LlmcoreError`` message.
    """

    def __init__(self, proto: "errors_pb2.LlmcoreError") -> None:
        self.proto = proto
        super().__init__(proto.message or proto.code or "bridge error")

    @property
    def category(self) -> int:
        return self.proto.category

    @property
    def code(self) -> str:
        return self.proto.code


def _provider_subcode(http_status: int | None, retryable: bool) -> str:
    if http_status == 429:
        return "provider.rate_limited"
    if http_status == 401:
        return "provider.unauthenticated"
    if http_status == 403:
        return "provider.permission_denied"
    if http_status in (408, 504):
        return "provider.deadline_exceeded"
    return "provider.unavailable" if retryable else "provider.error"


def to_llmcore_error(exc: BaseException) -> "errors_pb2.LlmcoreError":
    """Translate any exception into a populated ``LlmcoreError`` proto.

    The mapping is ordered most-specific-first so subclasses (e.g.
    ``SessionNotFoundError`` < ``StorageError``) resolve correctly.
    """
    err = errors_pb2.LlmcoreError()
    err.message = redact(str(exc))
    err.retryable = False

    # Context length (subclass of ContextError) first.
    if isinstance(exc, ContextLengthError):
        err.category = _Cat.ERROR_CATEGORY_CONTEXT_LENGTH
        err.code = "context.too_long"
        err.http_status = 413
        return err
    if isinstance(exc, ContextError):
        err.category = _Cat.ERROR_CATEGORY_CONTEXT
        err.code = "context.error"
        err.http_status = 400
        return err
    # SessionNotFound (subclass of StorageError) before StorageError.
    if isinstance(exc, SessionNotFoundError):
        err.category = _Cat.ERROR_CATEGORY_NOT_FOUND
        err.code = "not_found.session"
        err.http_status = 404
        return err
    if isinstance(exc, StorageError):
        err.category = _Cat.ERROR_CATEGORY_STORAGE
        err.code = "storage.unavailable"
        err.http_status = 503
        err.retryable = True
        return err
    if isinstance(exc, EmbeddingError):
        err.category = _Cat.ERROR_CATEGORY_EMBEDDING
        err.code = "embedding.error"
        err.http_status = 500
        return err
    if isinstance(exc, SearchProviderError):
        err.category = _Cat.ERROR_CATEGORY_SEARCH
        err.code = "search.error"
        err.http_status = 502
        return err
    if isinstance(exc, ConfigError):
        err.category = _Cat.ERROR_CATEGORY_CONFIG
        err.code = "config.invalid"
        err.http_status = 400
        return err
    if isinstance(exc, ProviderError):
        err.category = _Cat.ERROR_CATEGORY_PROVIDER
        provider = getattr(exc, "provider_name", None)
        model = getattr(exc, "model_name", None)
        status = getattr(exc, "status_code", None)
        retryable = bool(getattr(exc, "retryable", False))
        retry_after = getattr(exc, "retry_after_seconds", None)
        if provider:
            err.provider = str(provider)
        if model:
            err.model = str(model)
        if isinstance(status, int):
            err.http_status = status
        err.retryable = retryable
        if isinstance(retry_after, (int, float)):
            err.retry_after_ms = float(retry_after) * 1000.0
        err.code = _provider_subcode(status if isinstance(status, int) else None, retryable)
        return err
    if isinstance(exc, NotImplementedError):
        err.category = _Cat.ERROR_CATEGORY_UNSUPPORTED
        err.code = "unsupported.capability"
        err.http_status = 501
        return err
    if isinstance(exc, asyncio.CancelledError):
        err.category = _Cat.ERROR_CATEGORY_CANCELLED
        err.code = "cancelled"
        err.http_status = 499
        return err
    if isinstance(exc, LLMCoreError):
        err.category = _Cat.ERROR_CATEGORY_INTERNAL
        err.code = "internal"
        err.http_status = 500
        return err
    # Unknown / unexpected.
    err.category = _Cat.ERROR_CATEGORY_INTERNAL
    err.code = "internal"
    err.http_status = 500
    return err


# Category -> default gRPC status (overridable by provider http_status).
_GRPC_BY_CATEGORY = {
    _Cat.ERROR_CATEGORY_CONFIG: grpc.StatusCode.FAILED_PRECONDITION,
    _Cat.ERROR_CATEGORY_CONTEXT_LENGTH: grpc.StatusCode.INVALID_ARGUMENT,
    _Cat.ERROR_CATEGORY_CONTEXT: grpc.StatusCode.INVALID_ARGUMENT,
    _Cat.ERROR_CATEGORY_INVALID_ARGUMENT: grpc.StatusCode.INVALID_ARGUMENT,
    _Cat.ERROR_CATEGORY_EMBEDDING: grpc.StatusCode.INTERNAL,
    _Cat.ERROR_CATEGORY_STORAGE: grpc.StatusCode.UNAVAILABLE,
    _Cat.ERROR_CATEGORY_NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    _Cat.ERROR_CATEGORY_SEARCH: grpc.StatusCode.INTERNAL,
    _Cat.ERROR_CATEGORY_UNSUPPORTED: grpc.StatusCode.UNIMPLEMENTED,
    _Cat.ERROR_CATEGORY_CANCELLED: grpc.StatusCode.CANCELLED,
    _Cat.ERROR_CATEGORY_INTERNAL: grpc.StatusCode.INTERNAL,
}


def grpc_status_for(err: "errors_pb2.LlmcoreError") -> grpc.StatusCode:
    """Map an ``LlmcoreError`` to a gRPC status code."""
    if err.category == _Cat.ERROR_CATEGORY_PROVIDER:
        status = err.http_status if err.HasField("http_status") else None
        if status == 429:
            return grpc.StatusCode.RESOURCE_EXHAUSTED
        if status == 401:
            return grpc.StatusCode.UNAUTHENTICATED
        if status == 403:
            return grpc.StatusCode.PERMISSION_DENIED
        if status in (408, 504):
            return grpc.StatusCode.DEADLINE_EXCEEDED
        return grpc.StatusCode.UNAVAILABLE if err.retryable else grpc.StatusCode.INTERNAL
    return _GRPC_BY_CATEGORY.get(err.category, grpc.StatusCode.INTERNAL)


# Category -> default HTTP status (used when no provider http_status present).
_HTTP_BY_CATEGORY = {
    _Cat.ERROR_CATEGORY_PROVIDER: 502,
    _Cat.ERROR_CATEGORY_CONFIG: 400,
    _Cat.ERROR_CATEGORY_CONTEXT_LENGTH: 413,
    _Cat.ERROR_CATEGORY_CONTEXT: 400,
    _Cat.ERROR_CATEGORY_INVALID_ARGUMENT: 400,
    _Cat.ERROR_CATEGORY_EMBEDDING: 500,
    _Cat.ERROR_CATEGORY_STORAGE: 503,
    _Cat.ERROR_CATEGORY_NOT_FOUND: 404,
    _Cat.ERROR_CATEGORY_SEARCH: 502,
    _Cat.ERROR_CATEGORY_UNSUPPORTED: 501,
    _Cat.ERROR_CATEGORY_CANCELLED: 499,
    _Cat.ERROR_CATEGORY_INTERNAL: 500,
}


def http_status_for(err: "errors_pb2.LlmcoreError") -> int:
    """Map an ``LlmcoreError`` to an HTTP status code."""
    if err.HasField("http_status") and err.http_status:
        return int(err.http_status)
    return _HTTP_BY_CATEGORY.get(err.category, 500)


def map_exception(exc: BaseException) -> BridgeError:
    """Wrap any exception into a :class:`BridgeError`.

    ``asyncio.CancelledError`` is re-raised rather than wrapped so cancellation
    propagates to the transport layer untouched.
    """
    if isinstance(exc, BridgeError):
        return exc
    if isinstance(exc, asyncio.CancelledError):  # pragma: no cover - re-raised
        raise exc
    return BridgeError(to_llmcore_error(exc))


def _make(category: int, code: str, message: str, http_status: int) -> BridgeError:
    err = errors_pb2.LlmcoreError(
        category=category, code=code, message=redact(message), http_status=http_status
    )
    return BridgeError(err)


def unsupported(message: str) -> BridgeError:
    """Construct an UNSUPPORTED ``BridgeError`` (capability not available)."""
    return _make(_Cat.ERROR_CATEGORY_UNSUPPORTED, "unsupported.capability", message, 501)


def invalid_argument(message: str, details: dict[str, Any] | None = None) -> BridgeError:
    """Construct an INVALID_ARGUMENT ``BridgeError`` (bad request at the edge)."""
    err = errors_pb2.LlmcoreError(
        category=_Cat.ERROR_CATEGORY_INVALID_ARGUMENT,
        code="invalid_argument",
        message=redact(message),
        http_status=400,
    )
    return BridgeError(err)


def not_found(message: str, *, code: str = "not_found") -> BridgeError:
    """Construct a NOT_FOUND ``BridgeError`` (e.g. a missing context item)."""
    return _make(_Cat.ERROR_CATEGORY_NOT_FOUND, code, message, 404)
