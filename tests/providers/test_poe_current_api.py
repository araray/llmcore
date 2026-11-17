# tests/providers/test_poe_current_api.py
"""PoeProvider against the current API (live-verified 2026-11-16).

- /v1/models declares each bot's custom parameters with JSON schemas; reasoning
  is spelled per bot (reasoning_effort / output_effort / thinking_level /
  thinking_budget / enable_thinking) and effort= has to follow it;
- top-level reasoning_effort is honoured (minimal: 0 reasoning tokens, high:
  52 on gpt-5.4-nano);
- an over-long message is a 413 ("Your message is too long");
- the native protocol's roles are system/user/bot/tool.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("openai")

from llmcore.models import Message, Role  # noqa: E402
from llmcore.providers.poe_provider import PoeProvider  # noqa: E402

LISTING = [
    {"id": "gpt-5.4-nano", "context_window": {"context_length": 400000, "max_output_tokens": 128000},
     "parameters": [{"name": "web_search", "schema": {"type": "boolean"}},
                    {"name": "reasoning_effort", "schema": {"enum": ["none", "low", "medium", "high", "xhigh", "max"]}}]},
    {"id": "gemini-3-flash", "parameters": [{"name": "thinking_level", "schema": {"enum": ["minimal", "low", "medium", "high"]}}]},
    {"id": "claude-haiku-4.5", "parameters": [{"name": "thinking_budget", "schema": {"type": "number", "minimum": 0, "maximum": 63999}}]},
    {"id": "qwen3.5-plus", "parameters": [{"name": "enable_thinking", "schema": {"type": "boolean"}}]},
    {"id": "gpt-4o-mini", "parameters": []},
]


@pytest.fixture
def provider() -> PoeProvider:
    p = PoeProvider({"api_key": "k"})
    p._models_cache = [dict(m) for m in LISTING]
    return p


@pytest.mark.parametrize(
    ("bot", "effort", "expected"),
    [("gpt-5.4-nano", "minimal", {"reasoning_effort": "low"}),  # tie breaks upward
     ("gpt-5.4-nano", "high", {"reasoning_effort": "high"}),
     ("gemini-3-flash", "none", {"thinking_level": "minimal"}),
     ("claude-haiku-4.5", "high", {"thinking_budget": 47999}),
     ("claude-haiku-4.5", "none", {"thinking_budget": 0}),
     ("qwen3.5-plus", "none", {"enable_thinking": False}),
     ("unlisted-bot", "high", {"reasoning_effort": "high"})],
)
def test_effort_follows_each_bots_parameter(provider, bot, effort, expected):
    assert provider._bot_parameters(bot, {"effort": effort})["extra_body"] == expected


def test_effort_without_a_reasoning_parameter_is_dropped(provider):
    assert "extra_body" not in provider._bot_parameters("gpt-4o-mini", {"effort": "high"})


def test_native_search_only_where_declared(provider):
    assert provider.supports_native_search("GPT-5.4-Nano")  # ids are case-insensitive
    assert not provider.supports_native_search("gpt-4o-mini")
    shaped = provider._bot_parameters("gpt-5.4-nano", {"native_search": True, "extra_body": {"verbosity": "low"}})
    assert shaped["extra_body"] == {"verbosity": "low", "web_search": True}


def test_listing_drives_context_and_413_is_overflow(provider):
    assert provider.get_max_context_length("gpt-5.4-nano") == 400000
    assert provider._is_context_overflow(413, "Your message is too long.", None)
    assert not provider._is_context_overflow(400, "bad request", None)


def test_ignored_parameters_are_stripped(provider):
    params = provider.get_supported_parameters("gpt-5.4-nano")
    for name in ("response_format", "seed", "store", "metadata", "prediction"):
        assert name not in params
    assert "effort" in params and "native_search" in params


async def test_native_backend_maps_assistant_to_bot(provider, monkeypatch):
    pytest.importorskip("fastapi_poe")
    from llmcore.providers import poe_provider as module

    seen: dict[str, Any] = {}

    async def fake_get_bot_response(messages, **kwargs):
        seen["roles"] = [m.role for m in messages]
        seen["params"] = messages[-1].parameters

        class Partial:
            text = "Ana"
            is_suggested_reply = False
            is_replace_response = False

        yield Partial()

    monkeypatch.setattr(module, "get_bot_response", fake_get_bot_response)
    provider._backend = "native"
    history = [Message(role=Role.USER, content="I am Ana."), Message(role=Role.ASSISTANT, content="Hi Ana!"),
               Message(role=Role.USER, content="Name?")]
    out = await provider.chat_completion(history, "gpt-5.4-nano", effort="low")
    assert seen["roles"] == ["user", "bot", "user"]
    assert seen["params"] == {"reasoning_effort": "low"}
    assert out["choices"][0]["message"]["content"] == "Ana"
