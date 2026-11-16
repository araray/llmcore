# tests/providers/test_openai_responses.py
"""The OpenAI Responses wire module (``providers/openai_responses.py``).

Pure functions, so these tests pin the conversions directly. Payloads mirror
what Command Code's gateway returned for ``gpt-6-luna`` on 2026-11-15.
"""

from __future__ import annotations

import pytest

from llmcore.models import Message, Role, Tool
from llmcore.providers import openai_responses as rsp

WEATHER = Tool(
    name="get_weather",
    description="Weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
REASONING_ITEM = {"id": "rs_1", "type": "reasoning", "summary": [], "content": [],
                  "encrypted_content": "gAAAA..."}


class TestRequest:
    def test_messages_become_instructions_and_items(self):
        msgs = [
            Message(role=Role.SYSTEM, content="Be terse."),
            Message(role=Role.SYSTEM, content="Answer in English."),
            Message(role=Role.USER, content="Weather?"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[{"id": "call_1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}}],
                metadata={"reasoning_items": [REASONING_ITEM]},
            ),
            Message(role=Role.TOOL, content="22C", tool_call_id="call_1"),
        ]
        instructions, items = rsp.messages_to_input(msgs)
        assert instructions == "Be terse.\n\nAnswer in English."
        assert items == [
            {"role": "user", "content": "Weather?"},
            REASONING_ITEM,  # echoed ahead of the turn it belongs to
            {"type": "function_call", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\":\"Lisbon\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "22C"},
        ]

    def test_images(self):
        msg = Message(role=Role.USER, content="what is this", metadata={"inline_images": [
            "https://x/cat.png", {"data": "AAAA", "media_type": "image/jpeg", "detail": "low"}]})
        _, items = rsp.messages_to_input([msg])
        assert items[0]["content"] == [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": "https://x/cat.png"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA", "detail": "low"},
        ]

    def test_parameters_translate(self):
        body = rsp.build_responses_request(
            "gpt-6-luna",
            [Message(role=Role.USER, content="hi")],
            tools=[WEATHER],
            tool_choice="get_weather",
            max_tokens=100,
            effort="high",
            response_format={"type": "json_schema", "json_schema": {
                "name": "x", "strict": True, "schema": {"type": "object"}}},
        )
        assert body["tools"] == [{"type": "function", "name": "get_weather",
                                  "description": "Weather for a city",
                                  "parameters": WEATHER.parameters}]
        assert body["tool_choice"] == {"type": "function", "name": "get_weather"}
        assert body["max_output_tokens"] == 100
        assert body["reasoning"] == {"effort": "high"}
        assert body["text"] == {"format": {"type": "json_schema", "name": "x",
                                           "schema": {"type": "object"}, "strict": True}}
        # Stateless by default, with the reasoning chain returned to the caller.
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]

    def test_explicit_responses_fields_win(self):
        body = rsp.build_responses_request(
            "m", [Message(role=Role.USER, content="hi")],
            reasoning={"effort": "low", "summary": "auto"}, effort="high", store=True,
            max_tokens=5, max_output_tokens=9,
        )
        assert body["reasoning"] == {"effort": "low", "summary": "auto"}
        assert body["store"] is True and "include" not in body
        assert body["max_output_tokens"] == 9

    @pytest.mark.parametrize(
        ("given", "wanted"),
        [("auto", "auto"), ("none", "none"), ("required", "required"), ("any", "required"),
         ({"type": "function", "function": {"name": "f"}}, {"type": "function", "name": "f"})],
    )
    def test_tool_choice(self, given, wanted):
        assert rsp.tool_choice_to_responses(given) == wanted


class TestResponse:
    def test_text_reasoning_and_usage(self):
        out = rsp.normalize_response({
            "id": "resp_1", "model": "gpt-6-luna", "status": "completed",
            "output": [
                {**REASONING_ITEM, "summary": [{"type": "summary_text", "text": "91 = 7*13"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "No", "annotations": []}]},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 26, "total_tokens": 46,
                      "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens_details":
                                               {"ttl_12h_tokens": 2, "ttl_30m_tokens": 3}},
                      "output_tokens_details": {"reasoning_tokens": 19}},
        })
        message = out["choices"][0]["message"]
        assert message["content"] == "No"
        assert message["reasoning"] == "91 = 7*13"
        assert message["reasoning_items"][0]["encrypted_content"] == "gAAAA..."
        assert out["choices"][0]["finish_reason"] == "stop"
        usage = out["usage"]
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (20, 26)
        assert usage["prompt_tokens_details"] == {"cached_tokens": 4, "cache_write_tokens": 5}
        assert usage["completion_tokens_details"] == {"reasoning_tokens": 19}

    def test_function_calls(self):
        out = rsp.normalize_response({"status": "completed", "output": [
            {"type": "function_call", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\":\"Lisbon\"}"}]})
        choice = out["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"][0]["id"] == "call_1"

    def test_incomplete_is_length(self):
        out = rsp.normalize_response({"status": "incomplete",
                                      "incomplete_details": {"reason": "max_output_tokens"},
                                      "output": [REASONING_ITEM]})
        assert out["choices"][0]["finish_reason"] == "length"

    def test_failed_raises_with_partial(self):
        with pytest.raises(rsp.ResponsesFailed) as info:
            rsp.normalize_response({"status": "failed", "error": {"code": "server_error",
                                                                  "message": "boom"},
                                    "output": [{"type": "message", "content": [
                                        {"type": "output_text", "text": "par"}]}]})
        assert info.value.error["code"] == "server_error"
        assert info.value.partial == "par"


class TestStream:
    def test_text_tool_calls_and_final_chunk(self):
        n = rsp.ResponsesStreamNormalizer()
        events = [
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-6-luna"}},
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.output_item.added", "output_index": 1, "item": {
                "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"c"},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "ity\":1}"},
            {"type": "response.completed", "response": {
                "id": "resp_1", "status": "completed",
                "output": [REASONING_ITEM, {"type": "function_call", "call_id": "call_1",
                                            "name": "get_weather", "arguments": "{\"city\":1}"}],
                "usage": {"input_tokens": 3, "output_tokens": 4}}},
        ]
        chunks = [c for e in events for c in n.feed(e)]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert deltas[0]["reasoning"] == "thinking"
        assert deltas[1]["content"] == "Hi"
        assert deltas[2]["tool_calls"][0] == {"index": 0, "id": "call_1", "type": "function",
                                              "function": {"name": "get_weather", "arguments": ""}}
        args = "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas[3:5])
        assert args == "{\"city\":1}"
        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        assert final["usage"]["total_tokens"] == 7
        assert final["choices"][0]["delta"]["reasoning_items"] == [REASONING_ITEM]
        assert all(c["id"] == "resp_1" for c in chunks)

    def test_failure_events_raise(self):
        n = rsp.ResponsesStreamNormalizer()
        n.feed({"type": "response.output_text.delta", "delta": "par"})
        with pytest.raises(rsp.ResponsesFailed) as info:
            n.feed({"type": "response.failed", "response": {"error": {"code": "server_error"}}})
        assert info.value.partial == "par"
        with pytest.raises(rsp.ResponsesFailed):
            rsp.ResponsesStreamNormalizer().feed({"type": "error", "message": "x", "code": "y"})

    def test_unknown_events_are_ignored(self):
        assert rsp.ResponsesStreamNormalizer().feed({"type": "response.content_part.added"}) == []
