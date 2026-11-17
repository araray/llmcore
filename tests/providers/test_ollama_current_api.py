# tests/providers/test_ollama_current_api.py
"""OllamaProvider against a live 0.32 server and ollama-python 0.6.3 (2026-11-16).

- the server loads a model with its own default context (4096 on a 6 GiB GPU:
  /api/ps), not the trained window, and truncates longer prompts silently;
- a tool loop needs the assistant turn back with its tool_calls (arguments as
  objects) and thinking; tool results carry tool_name;
- tool calls arrive with an id ("call_...") the SDK's models drop;
- the direct backend's listing is a dict, which the old code read as an object.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("ollama")

from llmcore.models import Message, Role  # noqa: E402
from llmcore.providers.ollama_provider import OllamaProvider  # noqa: E402

CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})}}


@pytest.fixture
def provider() -> OllamaProvider:
    p = OllamaProvider({})
    p._client = MagicMock()
    p._client.chat = AsyncMock(return_value={"message": {"role": "assistant", "content": "ok"}, "done": True})
    p._client.ps = AsyncMock(return_value={"models": []})
    return p


def _sent(p: OllamaProvider) -> dict[str, Any]:
    return p._client.chat.await_args.kwargs


async def test_tool_loop_replays_calls_thinking_and_names(provider):
    history = [
        Message(role=Role.USER, content="Weather?"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[CALL], metadata={"thinking": "use the tool"}),
        Message(role=Role.TOOL, content="22C", tool_call_id="call_1"),
    ]
    await provider.chat_completion(history)
    assistant, tool = _sent(provider)["messages"][1:]
    assert assistant["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "Lisbon"}}, "id": "call_1"}]
    assert assistant["thinking"] == "use the tool"
    assert tool["tool_name"] == "get_weather"


async def test_unmatched_results_are_named_by_position(provider):
    calls = [{"id": None, "function": {"name": "a", "arguments": "{}"}},
             {"id": None, "function": {"name": "b", "arguments": "{}"}}]
    history = [Message(role=Role.USER, content="go"), Message(role=Role.ASSISTANT, content="", tool_calls=calls),
               Message(role=Role.TOOL, content="1"), Message(role=Role.TOOL, content="2")]
    await provider.chat_completion(history)
    assert [m.get("tool_name") for m in _sent(provider)["messages"][2:]] == ["a", "b"]


async def test_num_ctx_and_effort(provider):
    provider.num_ctx = 16384
    await provider.chat_completion([Message(role=Role.USER, content="hi")], effort="none", max_tokens=5)
    sent = _sent(provider)
    assert sent["options"] == {"num_predict": 5, "num_ctx": 16384}
    assert sent["think"] is False


async def test_effective_context_is_observed_and_reported(provider, caplog):
    provider._server_ctx["qwen3.5:4b"] = 262144
    provider._client.ps = AsyncMock(return_value={"models": [{"model": "qwen3.5:4b", "context_length": 4096}]})
    await provider.chat_completion([Message(role=Role.USER, content="hi")], model="qwen3.5:4b")
    assert provider.get_max_context_length("qwen3.5:4b") == 4096
    assert "truncated silently" in caplog.text


def test_server_tool_call_ids_are_kept(provider):
    response = {"message": {"tool_calls": [{"id": "call_x", "function": {"name": "f", "arguments": {}}}]}}
    assert provider.extract_tool_calls(response)[0].id == "call_x"


async def test_stream_final_chunk_carries_the_thinking(provider):
    async def chunks():
        yield {"message": {"role": "assistant", "content": "", "thinking": "use "}, "done": False}
        yield {"message": {"role": "assistant", "content": "", "thinking": "the tool"}, "done": False}
        yield {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}

    provider._client.chat = AsyncMock(return_value=chunks())
    out = [c async for c in await provider.chat_completion([Message(role=Role.USER, content="hi")], stream=True)]
    assert provider.assistant_replay_metadata(out[-1]) == {"thinking": "use the tool"}


async def test_direct_backend_lists_dict_responses(provider):
    provider._backend = "httpx"
    tags = {"models": [{"model": "qwen3.5:4b", "capabilities": ["completion", "tools", "vision", "thinking"],
                        "details": {"family": "qwen35", "context_length": 262144}}]}
    http = MagicMock()
    http.get = AsyncMock(side_effect=[MagicMock(status_code=200, json=lambda: tags),
                                      MagicMock(status_code=200, json=lambda: {"models": []})])
    provider._http = http
    (details,) = await provider.get_models_details()
    assert details.supports_tools and details.supports_vision and details.supports_reasoning
    assert provider.get_max_context_length("qwen3.5:4b") == 262144
