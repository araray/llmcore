# tests/providers/test_openai_direct_transport.py
"""The direct (httpx) transport on ``OpenAIProvider`` (plan §5.8).

``OpenAIProvider`` is the base class for ``deepinfra``, ``vllm``, ``poe`` and
``openrouter``, so one direct transport gives **five** providers a dual
approach. That leverage is also the risk: anything that changes default
behaviour here changes it for all five at once.

Two properties are therefore load-bearing:

* **the default must stay ``sdk``** — those suites mock ``AsyncOpenAI``, and
  flipping the default would route five providers straight past their own
  tests, which is exactly how the Z.ai SDK backend broke 21 tests; and
* **the transports must be interchangeable** at the dict boundary, because
  every ``extract_*`` method downstream parses those dicts rather than SDK
  objects.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from llmcore.exceptions import ContextLengthError, ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.providers.openai_provider import OpenAIProvider

BASE_URL = "https://api.openai.com/v1"
CONFIG: dict[str, Any] = {
    "api_key": "sk-test",
    "_instance_name": "openai",
    "default_model": "gpt-4o-mini",
}
CTX = [Message(role=Role.USER, content="hello")]


def _direct(**overrides: Any) -> OpenAIProvider:
    return OpenAIProvider({**CONFIG, "transport": "httpx", **overrides})


def _completion(content: str = "hi") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _sse(*chunks: dict[str, Any], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


# ---------------------------------------------------------------------------
# Selection — the regression guard for five providers
# ---------------------------------------------------------------------------


class TestTransportSelection:
    def test_default_is_the_sdk(self):
        """Five providers' suites mock AsyncOpenAI. This must not change."""
        assert OpenAIProvider(dict(CONFIG))._transport == "sdk"

    def test_httpx_is_opt_in(self):
        assert _direct()._transport == "httpx"

    def test_unknown_transport_falls_back_to_the_sdk(self):
        assert OpenAIProvider({**CONFIG, "transport": "carrier-pigeon"})._transport == "sdk"

    def test_missing_httpx_falls_back_to_the_sdk(self):
        with patch("llmcore.providers.openai_provider.httpx_available", False):
            assert _direct()._transport == "sdk"

    def test_the_key_is_transport_not_backend(self):
        """OpenRouter and Poe already use `backend` for native-SDK selection;
        overloading it would make one of the two settings unreachable."""
        p = OpenAIProvider({**CONFIG, "backend": "httpx"})
        assert p._transport == "sdk"

    def test_defaults_exist_without_init(self):
        """Tests build stubs with object.__new__; the attribute must resolve."""
        stub = object.__new__(OpenAIProvider)
        assert stub._transport == "sdk"
        assert stub._http is None
        assert stub._direct_headers == {}

    def test_header_dicts_are_per_instance(self):
        """A shared class-level dict would leak one provider's headers into all."""
        a = _direct(default_headers={"X-A": "1"})
        b = _direct()
        assert a._direct_headers == {"X-A": "1"}
        assert b._direct_headers == {}
        assert OpenAIProvider._direct_headers == {}


class TestBaseUrl:
    def test_defaults_to_the_public_api(self):
        assert _direct()._direct_base_url() == BASE_URL

    def test_honours_a_configured_base_url(self):
        p = _direct(base_url="https://api.deepinfra.com/v1/openai/")
        assert p._direct_base_url() == "https://api.deepinfra.com/v1/openai"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class TestDirectRequests:
    @respx.mock
    async def test_non_streaming_returns_the_response_dict(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion("hi"))
        )
        p = _direct()
        result = await p.chat_completion(CTX, model="gpt-4o-mini")
        assert result["choices"][0]["message"]["content"] == "hi"
        assert p.extract_response_content(result) == "hi"
        await p.close()

    @respx.mock
    async def test_request_body_shape(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion())
        )
        p = _direct()
        await p.chat_completion(CTX, model="gpt-4o-mini", temperature=0.5, max_tokens=64)
        body = json.loads(route.calls[0].request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["stream"] is False
        assert body["temperature"] == 0.5 and body["max_tokens"] == 64
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
        await p.close()

    @respx.mock
    async def test_tools_are_sent_in_openai_shape(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion())
        )
        p = _direct()
        tool = Tool(name="get_weather", description="Get weather",
                    parameters={"type": "object", "properties": {}})
        await p.chat_completion(CTX, model="gpt-4o-mini", tools=[tool], tool_choice="auto")
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert body["tool_choice"] == "auto"
        await p.close()

    @respx.mock
    async def test_reasoning_model_parameters_reach_the_body(self):
        """Request shaping happens before the transport branch, so the direct
        path inherits parameter validation and naming rather than reimplementing
        them. Reasoning models take ``max_completion_tokens``; ``max_tokens`` is
        not in their supported set at all."""
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion())
        )
        p = _direct(default_model="o1-mini")
        await p.chat_completion(CTX, model="o1-mini", max_completion_tokens=100)
        body = json.loads(route.calls[0].request.content)
        assert body["max_completion_tokens"] == 100
        assert "max_tokens" not in body
        await p.close()

    @respx.mock
    async def test_unsupported_parameters_are_rejected_before_any_request(self):
        """Validation is shared, so the direct transport rejects the same
        parameters the SDK path does — and sends nothing."""
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion())
        )
        p = _direct(default_model="o1-mini")
        with pytest.raises(ValueError, match="Unsupported parameter"):
            await p.chat_completion(CTX, model="o1-mini", max_tokens=100)
        assert route.call_count == 0
        await p.close()

    @respx.mock
    async def test_custom_headers_are_sent(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion())
        )
        p = _direct(default_headers={"HTTP-Referer": "https://ex.com", "X-Title": "llmcore"})
        await p.chat_completion(CTX, model="gpt-4o-mini")
        headers = route.calls[0].request.headers
        assert headers["http-referer"] == "https://ex.com"
        assert headers["x-title"] == "llmcore"
        await p.close()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestDirectStreaming:
    @respx.mock
    async def test_sse_chunks_become_dicts(self):
        chunks = [
            {"choices": [{"delta": {"content": "he"}, "index": 0}]},
            {"choices": [{"delta": {"content": "llo"}, "index": 0}]},
        ]
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, content=_sse(*chunks))
        )
        p = _direct()
        gen = await p.chat_completion(CTX, model="gpt-4o-mini", stream=True)
        got = [p.extract_delta_content(c) async for c in gen]
        assert "".join(got) == "hello"
        await p.close()

    @respx.mock
    async def test_done_sentinel_ends_the_stream(self):
        """`[DONE]` is a sentinel, not JSON; parsing it would raise on the last
        chunk of every successful stream."""
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200, content=_sse({"choices": [{"delta": {"content": "x"}}]})
            )
        )
        p = _direct()
        gen = await p.chat_completion(CTX, model="gpt-4o-mini", stream=True)
        assert len([c async for c in gen]) == 1
        await p.close()

    @respx.mock
    async def test_unparseable_chunks_are_skipped_not_fatal(self):
        body = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"
            b"data: {not json\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, content=body)
        )
        p = _direct()
        gen = await p.chat_completion(CTX, model="gpt-4o-mini", stream=True)
        got = [p.extract_delta_content(c) async for c in gen]
        assert "".join(got) == "ab"
        await p.close()

    @respx.mock
    async def test_non_data_lines_are_ignored(self):
        body = b": keep-alive\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"z\"}}]}\n\ndata: [DONE]\n\n"
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, content=body)
        )
        p = _direct()
        gen = await p.chat_completion(CTX, model="gpt-4o-mini", stream=True)
        assert len([c async for c in gen]) == 1
        await p.close()

    @respx.mock
    async def test_stream_errors_map_before_any_chunk(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(401, text="bad key")
        )
        p = _direct()
        gen = await p.chat_completion(CTX, model="gpt-4o-mini", stream=True)
        with pytest.raises(ProviderError, match="API Error"):
            [c async for c in gen]
        await p.close()


# ---------------------------------------------------------------------------
# Error parity — a dual transport that reports failures differently is not dual
# ---------------------------------------------------------------------------


class TestErrorParity:
    @respx.mock
    async def test_context_length_raises_the_same_typed_error(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                400, text='{"error":{"message":"maximum context_length exceeded"}}'
            )
        )
        p = _direct()
        with pytest.raises(ContextLengthError) as exc:
            await p.chat_completion(CTX, model="gpt-4o-mini")
        assert exc.value.model_name == "gpt-4o-mini"
        await p.close()

    @respx.mock
    @pytest.mark.parametrize(
        "body",
        [
            '{"error":{"message":"The model does not exist"}}',
            '{"error":{"code":"model_not_found"}}',
            '{"error":{"message":"Model Not Exist"}}',
            '{"error":{"message":"invalid model"}}',
        ],
    )
    async def test_model_not_found_is_actionable(self, body):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(400, text=body)
        )
        p = _direct()
        with pytest.raises(ProviderError) as exc:
            await p.chat_completion(CTX, model="nope")
        message = str(exc.value)
        assert "not found on provider" in message
        assert "gpt-4o-mini" in message, "should name the provider's default model"
        await p.close()

    @respx.mock
    @pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
    async def test_other_statuses_become_provider_errors(self, status):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(status, text="boom")
        )
        p = _direct()
        with pytest.raises(ProviderError, match=f"API Error \\({status}\\)"):
            await p.chat_completion(CTX, model="gpt-4o-mini")
        await p.close()

    @respx.mock
    async def test_timeouts_are_reported_as_timeouts(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            side_effect=httpx.ConnectTimeout("too slow")
        )
        p = _direct()
        with pytest.raises(ProviderError, match="Timeout"):
            await p.chat_completion(CTX, model="gpt-4o-mini")
        await p.close()

    @respx.mock
    async def test_connection_errors_are_reported_as_such(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        p = _direct()
        with pytest.raises(ProviderError, match="Connection error"):
            await p.chat_completion(CTX, model="gpt-4o-mini")
        await p.close()


# ---------------------------------------------------------------------------
# Interchangeability
# ---------------------------------------------------------------------------


class TestInterchangeability:
    """Both transports must hand the extractors the same dicts."""

    @respx.mock
    async def test_extractors_work_identically(self):
        payload = _completion("shared")
        payload["choices"][0]["message"]["tool_calls"] = [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Lisbon"}'}}
        ]
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        direct = _direct()
        via_http = await direct.chat_completion(CTX, model="gpt-4o-mini")

        sdk = OpenAIProvider(dict(CONFIG))
        response = MagicMock()
        response.model_dump.return_value = payload
        sdk._client = MagicMock()
        sdk._client.chat.completions.create = AsyncMock(return_value=response)
        via_sdk = await sdk.chat_completion(CTX, model="gpt-4o-mini")

        assert via_http == via_sdk
        for provider, result in ((direct, via_http), (sdk, via_sdk)):
            assert provider.extract_response_content(result) == "shared"
            assert [c.name for c in provider.extract_tool_calls(result)] == ["get_weather"]
            assert provider.extract_usage_details(result)["total_tokens"] == 2
        await direct.close()


class TestModelListing:
    @respx.mock
    async def test_models_are_listed_over_the_direct_transport(self):
        respx.get(f"{BASE_URL}/models").mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]}
            )
        )
        p = _direct()
        details = await p.get_models_details()
        assert {d.id for d in details} == {"gpt-4o-mini", "gpt-4o"}
        assert all(d.provider_name == "openai" for d in details)
        await p.close()

    @respx.mock
    async def test_listing_errors_map(self):
        respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(401, text="nope"))
        p = _direct()
        with pytest.raises(ProviderError):
            await p.get_models_details()
        await p.close()


class TestLifecycle:
    @respx.mock
    async def test_close_releases_the_direct_client(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion())
        )
        p = _direct()
        await p.chat_completion(CTX, model="gpt-4o-mini")
        assert p._http is not None
        await p.close()
        assert p._http is None

    async def test_close_is_safe_when_nothing_was_built(self):
        p = _direct()
        await p.close()
        assert p._http is None


# ---------------------------------------------------------------------------
# The leverage: four subclasses inherit this
# ---------------------------------------------------------------------------


class TestSubclassInheritance:
    @pytest.mark.parametrize(
        ("module", "cls_name", "config", "expected_base"),
        [
            ("deepinfra_provider", "DeepInfraProvider", {"api_key": "k"},
             "https://api.deepinfra.com/v1/openai"),
            ("openrouter_provider", "OpenRouterProvider", {"api_key": "k"},
             "https://openrouter.ai/api/v1"),
            ("poe_provider", "PoeProvider", {"api_key": "k"}, "https://api.poe.com/v1"),
            ("vllm_provider", "VLLMProvider",
             {"api_key": "k", "base_url": "http://localhost:8000/v1"},
             "http://localhost:8000/v1"),
        ],
    )
    def test_subclasses_accept_the_direct_transport(
        self, module, cls_name, config, expected_base
    ):
        import importlib

        cls = getattr(importlib.import_module(f"llmcore.providers.{module}"), cls_name)
        p = cls({**config, "_instance_name": "x", "transport": "httpx"})
        assert p._transport == "httpx"
        assert p._direct_base_url() == expected_base

    @pytest.mark.parametrize(
        ("module", "cls_name", "config"),
        [
            ("deepinfra_provider", "DeepInfraProvider", {"api_key": "k"}),
            ("openrouter_provider", "OpenRouterProvider", {"api_key": "k"}),
            ("poe_provider", "PoeProvider", {"api_key": "k"}),
            ("vllm_provider", "VLLMProvider",
             {"api_key": "k", "base_url": "http://localhost:8000/v1"}),
        ],
    )
    def test_subclasses_still_default_to_the_sdk(self, module, cls_name, config):
        """The regression that would silently bypass four other test suites."""
        import importlib

        cls = getattr(importlib.import_module(f"llmcore.providers.{module}"), cls_name)
        assert cls({**config, "_instance_name": "x"})._transport == "sdk"

    def test_openrouter_headers_reach_the_direct_transport(self):
        """Both transports must identify the app to OpenRouter the same way."""
        from llmcore.providers.openrouter_provider import OpenRouterProvider

        p = OpenRouterProvider(
            {"api_key": "k", "_instance_name": "openrouter",
             "app_url": "https://ex.com", "app_title": "llmcore", "transport": "httpx"}
        )
        assert p._direct_headers == {"HTTP-Referer": "https://ex.com", "X-Title": "llmcore"}
