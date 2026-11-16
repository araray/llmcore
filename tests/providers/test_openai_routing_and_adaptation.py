# tests/providers/test_openai_routing_and_adaptation.py
"""OpenAIProvider: Responses routing and learned per-model parameter rules.

The error bodies are the ones OpenAI returned live on 2026-11-16 across
gpt-4.1, gpt-5, gpt-5.1, gpt-5.4, gpt-6 and o3. Each model accepts a different
set of parameters, so the provider learns a model's rules from its first
rejection instead of carrying a table that goes stale with every release.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from llmcore.exceptions import ContextLengthError, ProviderError
from llmcore.models import Message, Role
from llmcore.providers.openai_provider import OpenAIProvider, nearest_effort

BASE = "https://api.openai.com/v1"
CTX = [Message(role=Role.SYSTEM, content="Be terse."), Message(role=Role.USER, content="hi")]


def _provider(**cfg: Any) -> OpenAIProvider:
    return OpenAIProvider({"api_key": "sk-test", "transport": "httpx", **cfg})


def _chat(content: str = "ok") -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _resp(text: str = "ok", **extra: Any) -> dict[str, Any]:
    body = {"id": "resp_1", "object": "response", "status": "completed", "model": "m",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}
    body.update(extra)
    return body


def _err(status: int, message: str, code: str | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error",
                                                  "param": None, "code": code}})


def _sent(route: respx.Route, i: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[i].request.content)


class TestRouting:
    @pytest.mark.parametrize(
        ("model", "route"),
        [("gpt-6-luna", "responses"), ("gpt-5.4-pro", "responses"), ("o3", "responses"),
         ("gpt-5.3-codex", "responses"), ("gpt-4.1-mini", "chat"), ("gpt-4o", "chat"),
         ("gpt-5-search-api", "chat"), ("gpt-audio", "chat")],
    )
    @respx.mock
    async def test_auto_policy(self, model, route):
        chat = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_chat()))
        resp = respx.post(f"{BASE}/responses").mock(return_value=httpx.Response(200, json=_resp()))
        await _provider().chat_completion(CTX, model)
        assert (resp.call_count, chat.call_count) == ((1, 0) if route == "responses" else (0, 1))

    @respx.mock
    async def test_parameters_pick_the_route(self):
        chat = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_chat()))
        resp = respx.post(f"{BASE}/responses").mock(return_value=httpx.Response(200, json=_resp()))
        p = _provider()
        await p.chat_completion(CTX, "gpt-6-luna", seed=7)  # Chat-only
        await p.chat_completion(CTX, "gpt-4.1-mini", reasoning_summary="auto")  # Responses-only
        await p.chat_completion(CTX, "gpt-6-luna", responses=False)  # explicit
        assert chat.call_count == 2 and resp.call_count == 1

    @respx.mock
    async def test_compatible_endpoints_never_use_responses(self):
        chat = respx.post("https://compat.example/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat()))
        await _provider(base_url="https://compat.example/v1").chat_completion(CTX, "gpt-6-luna")
        assert chat.call_count == 1

    @respx.mock
    async def test_responses_request_shape(self):
        route = respx.post(f"{BASE}/responses").mock(return_value=httpx.Response(200, json=_resp()))
        await _provider().chat_completion(CTX, "gpt-6-luna", max_tokens=99, effort="low")
        body = _sent(route)
        assert body["instructions"] == "Be terse."
        assert body["max_output_tokens"] == 99
        assert body["reasoning"] == {"effort": "low"}
        assert body["store"] is False

    @respx.mock
    async def test_native_search_uses_the_responses_web_search_tool(self):
        route = respx.post(f"{BASE}/responses").mock(return_value=httpx.Response(200, json=_resp()))
        await _provider().chat_completion(CTX, "gpt-4.1-mini", native_search=True)
        assert _sent(route)["tools"] == [{"type": "web_search"}]

    @respx.mock
    async def test_chat_streams_ask_for_usage(self):
        route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"}))
        stream = await _provider().chat_completion(CTX, "gpt-4.1-mini", stream=True)
        assert [c async for c in stream]
        assert _sent(route)["stream_options"] == {"include_usage": True}


class TestLearnedRules:
    @respx.mock
    async def test_unsupported_sampling_is_dropped_and_remembered(self):
        route = respx.post(f"{BASE}/responses").mock(side_effect=[
            _err(400, "Unsupported value: 'temperature' does not support 0.2 with this model. "
                      "Only the default (1) value is supported."),
            httpx.Response(200, json=_resp()),
            httpx.Response(200, json=_resp()),
        ])
        p = _provider()
        await p.chat_completion(CTX, "gpt-6-luna", temperature=0.2)
        assert "temperature" in _sent(route, 0) and "temperature" not in _sent(route, 1)
        await p.chat_completion(CTX, "gpt-6-luna", temperature=0.2)
        assert route.call_count == 3 and "temperature" not in _sent(route, 2)

    @respx.mock
    async def test_effort_moves_to_the_nearest_supported_level(self):
        chat = respx.post(f"{BASE}/chat/completions").mock(side_effect=[
            _err(400, "Unsupported value: 'reasoning_effort' does not support 'minimal' with this "
                      "model. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."),
            httpx.Response(200, json=_chat()),
        ])
        await _provider().chat_completion(CTX, "gpt-6-luna", reasoning_effort="minimal", responses=False)
        assert _sent(chat, 1)["reasoning_effort"] == "low"  # tie between none and low goes up

    @respx.mock
    async def test_effort_on_responses_with_nameless_wording(self):
        route = respx.post(f"{BASE}/responses").mock(side_effect=[
            _err(400, "Unsupported value: 'none' is not supported with the 'gpt-5-nano' model. "
                      "Supported values are: 'minimal', 'low', 'medium', and 'high'."),
            httpx.Response(200, json=_resp()),
        ])
        await _provider().chat_completion(CTX, "gpt-5-nano", effort="none")
        assert _sent(route, 1)["reasoning"] == {"effort": "minimal"}

    @respx.mock
    async def test_unrecognized_argument_is_dropped(self):
        chat = respx.post(f"{BASE}/chat/completions").mock(side_effect=[
            _err(400, "Unrecognized request argument supplied: reasoning_effort"),
            httpx.Response(200, json=_chat()),
        ])
        await _provider().chat_completion(CTX, "gpt-4.1-mini", effort="high")
        assert "reasoning_effort" not in _sent(chat, 1)

    @respx.mock
    async def test_responses_only_model_moves_there(self):
        chat = respx.post(f"{BASE}/chat/completions").mock(return_value=_err(
            404, "This model is only supported in v1/responses and not in v1/chat/completions."))
        resp = respx.post(f"{BASE}/responses").mock(return_value=httpx.Response(200, json=_resp()))
        p = _provider()
        await p.chat_completion(CTX, "some-new-model")
        await p.chat_completion(CTX, "some-new-model")
        assert chat.call_count == 1 and resp.call_count == 2

    @respx.mock
    async def test_strict_policy_fails_instead(self):
        route = respx.post(f"{BASE}/responses").mock(return_value=_err(
            400, "Unsupported parameter: 'top_p' is not supported with this model."))
        with pytest.raises(ProviderError, match="top_p"):
            await _provider(param_policy="strict").chat_completion(CTX, "gpt-6-luna", top_p=0.5)
        assert route.call_count == 1

    @pytest.mark.parametrize(
        ("requested", "supported", "chosen"),
        [("minimal", ["none", "low", "medium", "high"], "low"),
         ("none", ["minimal", "low", "medium", "high"], "minimal"),
         ("max", ["low", "medium", "high", "xhigh"], "xhigh"),
         ("xhigh", ["low", "medium", "high"], "high")],
    )
    def test_nearest_effort(self, requested, supported, chosen):
        assert nearest_effort(requested, supported) == chosen


class TestErrorsAndShapes:
    @respx.mock
    async def test_context_overflow_on_both_routes(self):
        respx.post(f"{BASE}/chat/completions").mock(return_value=_err(
            400, "This model's maximum context length is 1047576 tokens.", "context_length_exceeded"))
        respx.post(f"{BASE}/responses").mock(return_value=_err(
            400, "Your input exceeds the context window of this model.", "context_length_exceeded"))
        p = _provider()
        for model in ("gpt-4.1-mini", "gpt-6-luna"):
            with pytest.raises(ContextLengthError):
                await p.chat_completion(CTX, model)

    @respx.mock
    async def test_extra_body_is_merged_on_the_direct_transport(self):
        route = respx.post("https://compat.example/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat()))
        p = _provider(base_url="https://compat.example/v1")
        await p._chat_via_httpx("m", [{"role": "user", "content": "hi"}], False,
                                {"extra_body": {"search_parameters": {"mode": "auto"}}})
        body = _sent(route)
        assert body["search_parameters"] == {"mode": "auto"} and "extra_body" not in body

    def test_context_windows(self):
        p = _provider()
        # Maximum input tokens, measured live from OpenAI's overflow errors.
        assert p.get_max_context_length("gpt-4.1-mini") == 1047576  # once 8,192
        assert p.get_max_context_length("gpt-6-luna") == 922000
        assert p.get_max_context_length("gpt-5.4-mini") == 272000
        assert p.get_max_context_length("gpt-5.4") == 922000

    def test_configured_limit_wording_is_a_context_error(self):
        p = _provider()
        text = ("Input tokens exceed the configured limit of 272000 tokens. "
                "Your messages resulted in 1300007 tokens.")
        assert p._is_context_overflow(400, text, None)

    async def test_image_requests_omit_response_format_for_gpt_image_models(self):
        p = OpenAIProvider({"api_key": "sk-test"})
        item = MagicMock(b64_json="AAAA", url=None, revised_prompt=None)
        p._client = MagicMock()
        p._client.images.generate = AsyncMock(return_value=MagicMock(data=[item], created=1))
        await p.generate_image("a red dot")
        kwargs = p._client.images.generate.call_args.kwargs
        assert kwargs["model"] == "gpt-image-2" and "response_format" not in kwargs
        await p.generate_image("a red dot", model="dall-e-3")
        assert p._client.images.generate.call_args.kwargs["response_format"] == "b64_json"
