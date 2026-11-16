# tests/providers/test_gemini_tool_loops.py
"""Gemini tool loops, thought signatures, effort and usage (google-genai 3.x).

Measured live on 2026-11-16:

- ``types.FunctionDeclaration.from_dict`` does not exist, so every call with
  tools raised AttributeError;
- assistant tool calls were never replayed, and Gemini 3+ answers 400
  ("Function call is missing a thought_signature") to a replayed call
  without its signature. Only the first parallel call carries one;
- gemini-embedding-2 returns ONE vector for a plain list of strings.

These tests use real google-genai types; only the network call is mocked.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("google.genai")
from google.genai import types  # noqa: E402

from llmcore.models import Message, Role, Tool  # noqa: E402
from llmcore.providers.gemini_provider import _SKIP_SIGNATURE, GeminiProvider  # noqa: E402

WEATHER = Tool(name="get_weather", description="Weather for a city",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}})
SIG = base64.b64encode(b"real-signature").decode()


def _provider() -> GeminiProvider:
    p = GeminiProvider({"api_key": "k"})
    p._client = MagicMock()
    return p


def _response(*parts: types.Part, usage: dict | None = None) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)),
                                    finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**(usage or {})),
    )


def _call(name: str, args: dict, call_id: str, signature: bytes | None = None) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id),
                      thought_signature=signature)


class TestToolDeclarationsAndReplay:
    async def test_tools_are_real_function_declarations(self):
        p = _provider()
        p._client.aio.models.generate_content = AsyncMock(return_value=_response(types.Part(text="ok")))
        await p.chat_completion([Message(role=Role.USER, content="hi")], "gemini-3.8-flash", tools=[WEATHER])
        config = p._client.aio.models.generate_content.call_args.kwargs["config"]
        decl = config.tools[0].function_declarations[0]
        assert decl.name == "get_weather"
        assert decl.parameters_json_schema == WEATHER.parameters

    async def test_parallel_calls_keep_their_signature_and_replay_verbatim(self):
        p = _provider()
        p._client.aio.models.generate_content = AsyncMock(side_effect=[
            _response(_call("get_weather", {"city": "Lisbon"}, "c1", b"real-signature"),
                      _call("get_weather", {"city": "Porto"}, "c2")),
            _response(types.Part(text="22C in both")),
        ])
        q = [Message(role=Role.USER, content="Lisbon and Porto?")]
        first = await p.chat_completion(q, "gemini-3.8-flash", tools=[WEATHER])
        calls = first["choices"][0]["message"]["tool_calls"]
        assert calls[0]["extra_content"]["google"]["thought_signature"] == SIG
        assert "extra_content" not in calls[1]  # only the first parallel call is signed
        meta = p.assistant_replay_metadata(first)
        history = q + [Message(role=Role.ASSISTANT, content="", tool_calls=calls, metadata=meta)] + [
            Message(role=Role.TOOL, content="22C", tool_call_id=c["id"]) for c in calls]
        await p.chat_completion(history, "gemini-3.8-flash", tools=[WEATHER])
        contents = p._client.aio.models.generate_content.call_args.kwargs["contents"]
        model_turn, results = contents[1], contents[2]
        assert model_turn["role"] == "model"
        assert model_turn["parts"][0]["thought_signature"] == SIG
        # Results name the function (not the call id) and carry the id.
        assert results["parts"][0]["function_response"] == {
            "name": "get_weather", "response": {"result": "22C"}, "id": "c1"}

    def test_tool_calls_alone_rebuild_the_signed_turn(self):
        p = _provider()
        calls = [{"id": "c1", "type": "function",
                  "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})},
                  "extra_content": {"google": {"thought_signature": SIG}}}]
        parts = p._model_turn_parts(Message(role=Role.ASSISTANT, content="", tool_calls=calls))
        assert parts == [{"function_call": {"name": "get_weather", "args": {"city": "Lisbon"}, "id": "c1"},
                          "thought_signature": SIG}]

    def test_foreign_history_gets_the_documented_placeholder(self):
        p = _provider()
        calls = [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                 {"id": "y", "type": "function", "function": {"name": "g", "arguments": "{}"}}]
        parts = p._model_turn_parts(Message(role=Role.ASSISTANT, content="", tool_calls=calls))
        assert parts[0]["thought_signature"] == _SKIP_SIGNATURE
        assert "thought_signature" not in parts[1]


class TestStreaming:
    async def test_streamed_calls_finish_reason_usage_and_replay_parts(self):
        chunks = [
            _response(types.Part(text="Checking ")),
            _response(_call("get_weather", {"city": "Lisbon"}, "c1", b"real-signature"),
                      usage={"prompt_token_count": 5, "candidates_token_count": 3, "thoughts_token_count": 7,
                             "total_token_count": 15}),
        ]

        async def gen():
            for c in chunks:
                yield c

        p = _provider()
        p._client.aio.models.generate_content_stream = AsyncMock(return_value=gen())
        out = [c async for c in await p.chat_completion(
            [Message(role=Role.USER, content="?")], "gemini-3.8-flash", tools=[WEATHER], stream=True)]
        tool_deltas = [tc for c in out for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]
        assert tool_deltas[0]["index"] == 0 and tool_deltas[0]["function"]["name"] == "get_weather"
        final = out[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        assert final["usage"]["completion_tokens"] == 10  # thinking billed as output
        parts = p.assistant_replay_metadata(final)["gemini_parts"]
        assert parts[0] == {"text": "Checking "} and parts[1]["thought_signature"] == SIG


class TestEffortUsageEmbeddings:
    @pytest.mark.parametrize(
        ("model", "effort", "expected"),
        [("gemini-3.5-flash-lite", "none", {"thinking_level": "minimal"}),
         ("gemini-3.8-flash", "none", {"thinking_level": "low"}),   # no "minimal" on 3.8/4
         ("gemini-4-hydrogen", "max", {"thinking_level": "xhigh"}),
         ("gemini-4-carbon", "max", {"thinking_level": "high"}),
         ("gemini-2.5-flash", "none", {"thinking_budget": 0})],
    )
    def test_effort_mapping(self, model, effort, expected):
        assert GeminiProvider._effort_to_thinking(effort, model) == expected

    async def test_effort_and_display_reach_thinking_config(self):
        p = _provider()
        p._client.aio.models.generate_content = AsyncMock(return_value=_response(types.Part(text="ok")))
        await p.chat_completion([Message(role=Role.USER, content="hi")], "gemini-3.8-flash",
                                effort="high", thinking_display="summarized")
        thinking = p._client.aio.models.generate_content.call_args.kwargs["config"].thinking_config
        assert thinking.thinking_level.value.lower() == "high" and thinking.include_thoughts is True

    async def test_embeddings_one_vector_per_text(self):
        p = _provider()
        p._client.aio.models.embed_content = AsyncMock(return_value=MagicMock(
            embeddings=[MagicMock(values=[0.1]), MagicMock(values=[0.2])]))
        await p.create_embeddings(["hello", "world"])
        contents = p._client.aio.models.embed_content.call_args.kwargs["contents"]
        assert [c.parts[0].text for c in contents] == ["hello", "world"]

    def test_context_overflow_is_specific(self):
        assert GeminiProvider._is_context_overflow(
            "The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).")
        assert not GeminiProvider._is_context_overflow("max_output_tokens must be <= 65536")

    def test_developer_api_client_uses_v1beta(self):
        p = GeminiProvider({"api_key": "k"})
        assert p._client._api_client._http_options.api_version == "v1beta"
