# tests/providers/test_zai_current_api.py
"""ZaiProvider against the current API and the real zai-sdk 0.2.3 signature.

- zai-sdk's create() has a fixed signature without stream_options/user, so
  every stream on the default sdk backend raised TypeError; the old tests
  mocked create() with a bare MagicMock (which accepts anything);
- GLM-5.3 always thinks and takes only low/high/max (code 1210, measured);
- the out-of-balance answer is a 429 with code 1113 (measured).
"""

from __future__ import annotations

import functools
from typing import Any
from unittest.mock import MagicMock

import pytest

zai = pytest.importorskip("zai")

from llmcore.exceptions import ProviderError  # noqa: E402
from llmcore.models import Message, Role  # noqa: E402
from llmcore.providers.zai_provider import ZaiProvider  # noqa: E402

CTX = [Message(role=Role.USER, content="hi")]


def _sdk_provider(**cfg: Any) -> tuple[ZaiProvider, list[dict[str, Any]]]:
    """A provider whose SDK create() has the real signature and records calls."""
    p = ZaiProvider({"api_key": "id.secret", "backend": "sdk", **cfg})
    real = p._sdk_client.chat.completions.create
    calls: list[dict[str, Any]] = []

    @functools.wraps(real)
    def create(*args: Any, **kwargs: Any) -> Any:
        import inspect

        inspect.signature(real).bind(*args, **kwargs)  # TypeError, like the SDK
        calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([MagicMock(model_dump=lambda exclude_none=True: {
                "choices": [{"index": 0, "delta": {"reasoning_content": "think"}, "finish_reason": "stop"}]})])
        return MagicMock(model_dump=lambda exclude_none=True: {"choices": [{"message": {"content": "ok"}}]})

    p._sdk_client.chat.completions.create = create
    return p, calls


async def test_sdk_streaming_survives_the_real_signature():
    p, calls = _sdk_provider()
    out = [c async for c in await p.chat_completion(CTX, "glm-5.3", stream=True)]
    assert calls[0]["extra_body"] == {"stream_options": {"include_usage": True}}
    assert p.assistant_replay_metadata(out[-1]) == {"reasoning_content": "think"}


@pytest.mark.parametrize(
    ("model", "kwargs", "thinking", "effort"),
    [("glm-5.3", {"thinking": "disabled"}, {"type": "enabled"}, "high"),
     ("glm-5.3", {"effort": "medium"}, {"type": "enabled"}, "high"),
     ("glm-5.3", {"effort": "minimal"}, {"type": "enabled"}, "low"),
     ("glm-5.2", {"effort": "none"}, {"type": "disabled"}, None),
     ("glm-5.2", {"effort": "medium"}, {"type": "enabled"}, "medium")],
)
async def test_glm_53_always_thinks_with_its_own_levels(model, kwargs, thinking, effort):
    p, calls = _sdk_provider()
    await p.chat_completion(CTX, model, **kwargs)
    assert calls[0]["thinking"] == thinking
    assert calls[0].get("reasoning_effort") == effort


async def test_clear_thinking_and_greedy_decoding():
    p, calls = _sdk_provider(clear_thinking=False)
    await p.chat_completion(CTX, "glm-5.3", temperature=0)
    assert calls[0]["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert calls[0]["do_sample"] is False and "temperature" not in calls[0]


async def test_native_search_adds_the_builtin_tool():
    p, calls = _sdk_provider()
    await p.chat_completion(CTX, "glm-5.3", native_search=True)
    assert calls[0]["tools"][-1]["type"] == "web_search"


def test_balance_and_validation_errors_are_explained():
    p = ZaiProvider({"api_key": "id.secret", "backend": "httpx"})
    exc = type("E", (Exception,), {"status_code": 429})(
        '{"error": {"code": "1113", "message": "Insufficient balance or no resource package."}}')
    with pytest.raises(ProviderError, match="insufficient balance") as info:
        p._raise_chat_error(exc, "glm-5.3")
    assert info.value.retryable is False


def test_window_exceeded_finish_reason_is_length():
    p = ZaiProvider({"api_key": "id.secret", "backend": "httpx"})
    assert p.extract_finish_reason({"choices": [{"finish_reason": "model_context_window_exceeded"}]}) == "length"
