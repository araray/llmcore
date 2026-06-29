"""Unit tests for the canonical error mapping (no transport)."""

from __future__ import annotations

import grpc
import pytest

from llmcore.bridge import errors
from llmcore.bridge._generated.llmcore.v1 import errors_pb2
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

Cat = errors_pb2.ErrorCategory
SC = grpc.StatusCode

CASES = [
    (ProviderError("p", "rl", status_code=429, retry_after_seconds=2.0),
     Cat.ERROR_CATEGORY_PROVIDER, "provider.rate_limited", 429, SC.RESOURCE_EXHAUSTED, True),
    (ProviderError("p", "auth", status_code=401),
     Cat.ERROR_CATEGORY_PROVIDER, "provider.unauthenticated", 401, SC.UNAUTHENTICATED, False),
    (ProviderError("p", "perm", status_code=403),
     Cat.ERROR_CATEGORY_PROVIDER, "provider.permission_denied", 403, SC.PERMISSION_DENIED, False),
    (ProviderError("p", "boom", status_code=500, retryable=True),
     Cat.ERROR_CATEGORY_PROVIDER, "provider.unavailable", 500, SC.UNAVAILABLE, True),
    (ConfigError("bad"),
     Cat.ERROR_CATEGORY_CONFIG, "config.invalid", 400, SC.FAILED_PRECONDITION, False),
    (ContextLengthError("m", 10, 20, "too long"),
     Cat.ERROR_CATEGORY_CONTEXT_LENGTH, "context.too_long", 413, SC.INVALID_ARGUMENT, False),
    (ContextError("ctx"),
     Cat.ERROR_CATEGORY_CONTEXT, "context.error", 400, SC.INVALID_ARGUMENT, False),
    (SessionNotFoundError("sess"),
     Cat.ERROR_CATEGORY_NOT_FOUND, "not_found.session", 404, SC.NOT_FOUND, False),
    (StorageError("db"),
     Cat.ERROR_CATEGORY_STORAGE, "storage.unavailable", 503, SC.UNAVAILABLE, True),
    (EmbeddingError("e", "x"),
     Cat.ERROR_CATEGORY_EMBEDDING, "embedding.error", 500, SC.INTERNAL, False),
    (SearchProviderError("p", "x"),
     Cat.ERROR_CATEGORY_SEARCH, "search.error", 502, SC.INTERNAL, False),
    (NotImplementedError("nope"),
     Cat.ERROR_CATEGORY_UNSUPPORTED, "unsupported.capability", 501, SC.UNIMPLEMENTED, False),
    (LLMCoreError("kaboom"),
     Cat.ERROR_CATEGORY_INTERNAL, "internal", 500, SC.INTERNAL, False),
    (ValueError("unexpected"),
     Cat.ERROR_CATEGORY_INTERNAL, "internal", 500, SC.INTERNAL, False),
]


@pytest.mark.parametrize("exc, cat, code, http, grpc_code, retryable", CASES)
def test_mapping(exc, cat, code, http, grpc_code, retryable):
    err = errors.to_llmcore_error(exc)
    assert err.category == cat
    assert err.code == code
    assert err.http_status == http
    assert err.retryable is retryable
    assert errors.grpc_status_for(err) == grpc_code
    assert errors.http_status_for(err) == http


def test_retry_after_ms_conversion():
    err = errors.to_llmcore_error(ProviderError("p", "rl", status_code=429, retry_after_seconds=2.0))
    assert err.retry_after_ms == pytest.approx(2000.0)


def test_message_is_redacted():
    fake_key = "sk-" + "ABCDEFGHIJKLMNOP1234"
    err = errors.to_llmcore_error(LLMCoreError(f"key {fake_key} leaked"))
    assert fake_key not in err.message
    assert "REDACTED" in err.message


def test_bridge_error_carries_proto():
    be = errors.unsupported("no embed")
    assert isinstance(be, errors.BridgeError)
    assert be.code == "unsupported.capability"
    assert errors.http_status_for(be.proto) == 501
