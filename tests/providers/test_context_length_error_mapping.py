# tests/providers/test_context_length_error_mapping.py
"""
Regression tests for provider -> :class:`ContextLengthError` mapping.

``ContextLengthError`` takes ``(model_name, limit, actual, message)``. The
OpenAI, DeepSeek and Z.ai providers used to construct it with a different,
never-supported keyword set (``provider_name`` / ``model`` / ``max_tokens`` /
``requested_tokens``), so every context-overflow response raised an opaque
``TypeError`` from inside the exception constructor instead of the
``ContextLengthError`` callers catch. The bug survived because no test drove
those error branches.

This module locks the contract down two ways:

1. A static check over the whole package: every ``ContextLengthError(...)``
   call site must use keywords the constructor actually accepts. This covers
   providers that are not exercised below, and any added later.
2. Behavioural tests that drive the real ``chat_completion()`` error path of
   each previously-broken provider and assert the mapped exception carries the
   model name and the model's context limit.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmcore.exceptions import ContextLengthError
from llmcore.models import Message, Role

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "llmcore"

#: Keywords ``ContextLengthError.__init__`` actually accepts.
ACCEPTED_KWARGS = frozenset(inspect.signature(ContextLengthError).parameters)


# ---------------------------------------------------------------------------
# 1. Static contract: no call site may pass unsupported keywords
# ---------------------------------------------------------------------------


def _context_length_error_call_sites() -> list[tuple[pathlib.Path, int, set[str]]]:
    """Return every ``ContextLengthError(...)`` call in the package.

    Returns:
        ``(path, lineno, keyword_names)`` for each call site found.
    """
    sites: list[tuple[pathlib.Path, int, set[str]]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "ContextLengthError":
                continue
            sites.append(
                (path, node.lineno, {kw.arg for kw in node.keywords if kw.arg is not None})
            )
    return sites


class TestContextLengthErrorCallSites:
    def test_call_sites_were_found(self):
        """Guard the guard: the AST scan must actually find the call sites."""
        sites = _context_length_error_call_sites()
        assert len(sites) >= 5, f"expected several call sites, found {len(sites)}"

    def test_every_call_site_uses_supported_keywords(self):
        """No provider may pass keywords ``ContextLengthError`` does not accept.

        This is the exact defect that broke OpenAI, DeepSeek and Z.ai: the
        unsupported keywords raised ``TypeError`` at the ``raise`` statement.
        """
        offenders = [
            f"{path.relative_to(SRC_ROOT)}:{lineno} -> "
            f"{sorted(kwargs - ACCEPTED_KWARGS)}"
            for path, lineno, kwargs in _context_length_error_call_sites()
            if not kwargs <= ACCEPTED_KWARGS
        ]
        assert not offenders, (
            "ContextLengthError call sites with unsupported keywords "
            f"(accepted: {sorted(ACCEPTED_KWARGS)}):\n  " + "\n  ".join(offenders)
        )

    def test_constructor_rejects_the_legacy_keywords(self):
        """Pin the failure mode so the signature cannot silently absorb them."""
        with pytest.raises(TypeError):
            ContextLengthError(
                provider_name="openai",  # type: ignore[call-arg]
                model="gpt-4o",
                max_tokens=128_000,
                requested_tokens=None,
                message="context_length exceeded",
            )

    def test_supported_keywords_populate_the_exception(self):
        err = ContextLengthError(
            model_name="gpt-4o", limit=128_000, actual=0, message="context_length exceeded"
        )
        assert err.model_name == "gpt-4o"
        assert err.limit == 128_000
        assert err.actual == 0
        assert "gpt-4o" in str(err)


# ---------------------------------------------------------------------------
# 2. Behavioural: drive each previously-broken provider's error path
# ---------------------------------------------------------------------------


def _require_real_sdk_exception(module: Any) -> type[BaseException]:
    """Return *module*'s bound ``APIStatusError``, skipping if it was stubbed.

    ``tests/providers/test_openai_provider.py`` installs ``MagicMock``
    placeholders for the ``openai`` package into ``sys.modules`` at import
    time. If that happens before the provider under test is first imported,
    the provider binds a mock instead of the real exception class and no
    ``except`` clause can ever match it. Skip explicitly in that case instead
    of failing on another module's import-time side effect — the static call
    site check above still guards the actual defect unconditionally.
    """
    exc_cls = getattr(module, "OpenAIAPIStatusError", None)
    if not (isinstance(exc_cls, type) and issubclass(exc_cls, BaseException)):
        pytest.skip(
            f"{module.__name__} bound a stubbed 'openai' SDK in this process; "
            "the real APIStatusError class is required to drive its error path"
        )
    return exc_cls


def _api_status_error(module: Any, message: str, status: int = 400) -> Exception:
    """Build the ``APIStatusError`` instance *module*'s handler catches.

    The providers bind the class at import time, so instantiating the bound
    attribute keeps the ``except`` clause matching whatever is installed.
    """
    exc_cls = _require_real_sdk_exception(module)
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return exc_cls(message, response=response, body=None)


CONTEXT_OVERFLOW_MESSAGES = {
    # Each provider sniffs the body with its own phrasing test.
    "openai": "This model's maximum context_length is 128000 tokens.",
    "deepseek": "This model's maximum context_length is 131072 tokens.",
    "zai": "Input context length exceeds the model limit.",
}

USER_TURN = [Message(role=Role.USER, content="hello" * 10)]


class TestOpenAIProviderMapping:
    @pytest.fixture
    def provider(self):
        from llmcore.providers import openai_provider

        with patch.object(openai_provider, "AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            return openai_provider.OpenAIProvider(
                {"api_key": "sk-test", "default_model": "gpt-4o"}
            )

    async def test_context_overflow_maps_to_context_length_error(self, provider):
        from llmcore.providers import openai_provider

        provider._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(openai_provider, CONTEXT_OVERFLOW_MESSAGES["openai"])
        )
        with pytest.raises(ContextLengthError) as exc:
            await provider.chat_completion(USER_TURN)

        assert exc.value.model_name == "gpt-4o"
        assert exc.value.limit == provider.get_max_context_length("gpt-4o")
        assert "context_length" in str(exc.value)

    async def test_other_400s_are_not_context_length_errors(self, provider):
        from llmcore.exceptions import ProviderError
        from llmcore.providers import openai_provider

        provider._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(openai_provider, "Invalid value for 'temperature'.")
        )
        with pytest.raises(ProviderError) as exc:
            await provider.chat_completion(USER_TURN)
        assert not isinstance(exc.value, ContextLengthError)


class TestDeepSeekProviderMapping:
    @pytest.fixture
    def provider(self):
        from llmcore.providers import deepseek_provider

        with patch.object(deepseek_provider, "AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            return deepseek_provider.DeepSeekProvider(
                {"api_key": "sk-test", "default_model": "deepseek-v4-pro"}
            )

    async def test_context_overflow_maps_to_context_length_error(self, provider):
        from llmcore.providers import deepseek_provider

        provider._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(
                deepseek_provider, CONTEXT_OVERFLOW_MESSAGES["deepseek"]
            )
        )
        with pytest.raises(ContextLengthError) as exc:
            await provider.chat_completion(USER_TURN)

        assert exc.value.model_name == "deepseek-v4-pro"
        assert exc.value.limit == provider.get_max_context_length("deepseek-v4-pro")


class TestZaiProviderMapping:
    @pytest.fixture
    def provider(self):
        from llmcore.providers import zai_provider

        with patch.object(zai_provider, "AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            return zai_provider.ZaiProvider(
                {"api_key": "test-key", "default_model": "glm-5.2", "backend": "openai"}
            )

    async def test_context_overflow_maps_to_context_length_error(self, provider):
        from llmcore.providers import zai_provider

        provider._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(zai_provider, CONTEXT_OVERFLOW_MESSAGES["zai"])
        )
        with pytest.raises(ContextLengthError) as exc:
            await provider.chat_completion(USER_TURN)

        assert exc.value.model_name == "glm-5.2"
        assert exc.value.limit == provider.get_max_context_length("glm-5.2")

    async def test_auth_failure_is_still_a_provider_error(self, provider):
        from llmcore.exceptions import ProviderError
        from llmcore.providers import zai_provider

        provider._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(zai_provider, "invalid api key", status=401)
        )
        with pytest.raises(ProviderError) as exc:
            await provider.chat_completion(USER_TURN)
        assert not isinstance(exc.value, ContextLengthError)
