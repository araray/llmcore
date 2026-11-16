# tests/providers/test_commandcode_provider.py
"""Unit tests for the Command Code provider (``commandcode_provider.py``).

Command Code is a gateway with three wire protocols behind one key, so most of
what can go wrong is routing: a model sent to a route that does not serve it,
an error reported under the wrong provider name, or a gateway quirk that a
plain OpenAI-compatible client would let through. Every quirk tested here was
observed against the live API (2026-11-15). The fixtures reproduce the
observed bodies.

The gateway is mocked with ``respx``, so the default (httpx) transport is
exercised end to end without a network.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, ContextLengthError, ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.providers import commandcode_provider as cc
from llmcore.providers.commandcode_provider import (
    CommandCodeCatalog,
    CommandCodeModel,
    CommandCodeProvider,
    build_commandcode_error,
    guess_route,
    route_for_endpoints,
)
from llmcore.providers.typesafe_provider import Noul

ROOT = "https://api.commandcode.ai/provider"
V1 = f"{ROOT}/v1"
KEY = "user_test_key"

CATALOG = [
    {"id": "claude-haiku-5-5", "object": "model", "created": 1, "owned_by": "command-code",
     "name": "Claude Haiku 5.5", "context_length": 1000000, "supported_endpoints": ["/messages"]},
    {"id": "moonshotai/Kimi-K3", "object": "model", "created": 1, "owned_by": "command-code",
     "name": "Kimi K3", "context_length": 1000000,
     "supported_endpoints": ["/chat/completions", "/responses"]},
    {"id": "thinkingmachines/inkling-2-small:free", "object": "model", "created": 1,
     "owned_by": "command-code", "name": "Inkling 2 Small", "context_length": 1048576,
     "supported_endpoints": ["/chat/completions"]},
    {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek V4 Flash", "context_length": 1000000,
     "supported_endpoints": ["/chat/completions", "/responses"]},
    {"id": "deepseek/deepseek-v4-flash-fast", "name": "DeepSeek V4 Flash Fast",
     "context_length": 1000000, "supported_endpoints": ["/chat/completions"]},
    {"id": "gpt-6-luna", "name": "GPT-6 Luna", "context_length": 1050000,
     "supported_endpoints": ["/chat/completions", "/responses"]},
]
TRACE = {"x-trace-id": "trace123", "x-powered-by": "Hono"}
CTX = [Message(role=Role.SYSTEM, content="Be terse."), Message(role=Role.USER, content="hi")]


def _chat_body(content: str = "391", **extra: Any) -> dict[str, Any]:
    body = {
        "id": "gen-1", "object": "chat.completion", "created": 1,
        "model": "thinkingmachines/inkling-2-small:free", "provider": "Thinking Machines",
        "choices": [{
            "index": 0, "finish_reason": "stop", "native_finish_reason": "stop",
            "message": {
                "role": "assistant", "content": content, "refusal": None,
                "reasoning": "17*23 = 391.",
                "reasoning_details": [{"type": "reasoning.text", "text": "17*23 = 391.",
                                       "format": "unknown", "index": 0}],
            },
        }],
        "usage": {
            "prompt_tokens": 32, "completion_tokens": 30, "total_tokens": 62,
            "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
    }
    body.update(extra)
    return body


def _messages_body(text: str = "Hello there!") -> dict[str, Any]:
    return {
        "model": "claude-haiku-5-5", "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 17, "output_tokens": 7, "cache_creation_input_tokens": 3,
                  "cache_read_input_tokens": 11},
    }


def _sse(*chunks: dict[str, Any], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def _chunk(content: str = "", **choice: Any) -> dict[str, Any]:
    return {"id": "gen-1", "object": "chat.completion.chunk", "provider": "Thinking Machines",
            "choices": [{"index": 0, "delta": {"content": content, "role": "assistant"},
                         "finish_reason": None, **choice}]}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*cc.API_KEY_ENV_VARS, cc.ENV_ZDR, cc.ENV_BASE_URL):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def provider() -> CommandCodeProvider:
    return CommandCodeProvider({"api_key": KEY})


@pytest.fixture
def gateway():
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{V1}/models").mock(
            return_value=httpx.Response(200, json={"object": "list", "data": CATALOG})
        )
        yield mock


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.parametrize(
        ("endpoints", "route"),
        [
            (["/messages"], "messages"),
            (["/chat/completions", "/responses"], "chat"),
            (["/chat/completions"], "chat"),
            (["/systemone"], "system_one"),
            (["/responses"], "responses"),
            ([], None),
        ],
    )
    def test_endpoints_pick_the_lane(self, endpoints, route):
        assert route_for_endpoints("x/y", endpoints) == route

    @pytest.mark.parametrize(
        ("model", "policy", "route"),
        [
            ("gpt-6-luna", "openai", "responses"),
            ("moonshotai/Kimi-K3", "openai", "chat"),
            ("moonshotai/Kimi-K3", "all", "responses"),
            ("gpt-6-luna", "never", "chat"),
        ],
    )
    def test_responses_policy(self, model, policy, route):
        assert route_for_endpoints(model, ["/chat/completions", "/responses"], policy) == route

    @pytest.mark.parametrize(
        ("model", "route"),
        [("claude-opus-5-5", "messages"), ("typesafe/jev", "system_one"),
         ("moonshotai/Kimi-K3", "chat"), ("gpt-6-sol", "chat")],
    )
    def test_family_guess_when_nothing_else_knows(self, model, route):
        assert guess_route(model) == route

    async def test_catalog_routes_each_model(self, provider, gateway):
        await provider.catalog.ensure()
        assert provider.route_for("claude-haiku-5-5") == "messages"
        assert provider.route_for("moonshotai/Kimi-K3") == "chat"

    async def test_undrivable_model_is_refused_clearly(self, provider):
        provider.catalog.load_rows([{"id": "x/r", "supported_endpoints": ["/embeddings"]}])
        with pytest.raises(ProviderError, match="only on"):
            provider.route_for("x/r")

    def test_policy_config_and_per_call_override(self):
        p = CommandCodeProvider({"api_key": KEY, "responses_api": "never"})
        p.catalog.load_rows(CATALOG)
        assert p.route_for("gpt-6-luna") == "chat"
        assert p.route_for("gpt-6-luna", responses=True) == "responses"
        assert CommandCodeProvider({"api_key": KEY, "responses_api": "bogus"}).responses_policy == "openai"

    async def test_catalog_outage_falls_back_to_cards_then_families(self, provider):
        with respx.mock() as mock:
            route = mock.get(f"{V1}/models").mock(return_value=httpx.Response(503))
            assert await provider.catalog.ensure() is False
            # A second call inside the retry window does not hit the network again.
            assert await provider.catalog.ensure() is False
            assert route.call_count == 1
        # The bundled commandcode cards know this one's route...
        assert provider.route_for("moonshotai/Kimi-K3") == "chat"
        # ...and family names cover the rest.
        assert provider.route_for("claude-brand-new-9") == "messages"


class TestNameResolution:
    def _catalog(self) -> CommandCodeCatalog:
        cat = CommandCodeCatalog(base_url=ROOT)
        cat.load_rows(CATALOG)
        return cat

    def test_exact_case_insensitive_and_short_names(self):
        cat = self._catalog()
        assert cat.resolve("claude-haiku-5-5") == "claude-haiku-5-5"
        assert cat.resolve("MOONSHOTAI/KIMI-K3") == "moonshotai/Kimi-K3"
        assert cat.resolve("kimi-k3") == "moonshotai/Kimi-K3"
        assert cat.resolve("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"

    def test_unknown_names_pass_through_for_the_gateway_to_judge(self):
        assert self._catalog().resolve("nope/nope") == "nope/nope"

    def test_ambiguous_short_name_raises_with_candidates(self):
        cat = CommandCodeCatalog(base_url=ROOT)
        cat.load_rows([{"id": "a/same"}, {"id": "b/same"}])
        with pytest.raises(ConfigError, match="a/same, b/same"):
            cat.resolve("same")

    def test_catalog_rejects_an_unknown_shape(self):
        async def run():
            with respx.mock() as mock:
                mock.get(f"{V1}/models").mock(return_value=httpx.Response(200, json={"x": 1}))
                await CommandCodeCatalog(base_url=ROOT).refresh()

        import asyncio

        with pytest.raises(ProviderError, match="no `data` list"):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_key_lookup_order(self, monkeypatch):
        monkeypatch.setenv("COMMANDCODE_API_KEY", "third")
        assert CommandCodeProvider({}).api_key == "third"
        monkeypatch.setenv("COMMAND_CODE_API_KEY", "second")
        assert CommandCodeProvider({}).api_key == "second"
        monkeypatch.setenv("CMD_API_KEY", "first")
        assert CommandCodeProvider({}).api_key == "first"
        monkeypatch.setenv("MY_KEY", "named")
        assert CommandCodeProvider({"api_key_env_var": "MY_KEY"}).api_key == "named"
        assert CommandCodeProvider({"api_key": "explicit"}).api_key == "explicit"

    def test_missing_key_names_every_variable(self):
        with pytest.raises(ConfigError) as info:
            CommandCodeProvider({})
        for name in cc.API_KEY_ENV_VARS:
            assert name in str(info.value)

    def test_documented_v1_base_url_is_accepted(self, monkeypatch):
        assert CommandCodeProvider({"api_key": KEY, "base_url": f"{V1}/"}).base_url == ROOT
        monkeypatch.setenv("CMD_BASE_URL", "https://staging.example/provider")
        assert CommandCodeProvider({"api_key": KEY}).base_url == "https://staging.example/provider"

    def test_zdr_from_env_and_config(self, monkeypatch):
        assert CommandCodeProvider({"api_key": KEY}).zdr is False
        monkeypatch.setenv("CMD_ZDR", "1")
        assert CommandCodeProvider({"api_key": KEY}).zdr is True
        assert CommandCodeProvider({"api_key": KEY, "zdr": False}).zdr is False

    def test_packaged_defaults_leave_cmd_zdr_in_charge(self, monkeypatch):
        """Regression (PR #59 review): the packaged section once set `zdr = false`.

        Package defaults are merged into the provider config before it is
        built, and the provider reads CMD_ZDR only when no `zdr` key is
        present, so that line silently disabled the documented env switch.
        Building from the shipped section is the only way to catch it.
        """
        import tomllib
        from pathlib import Path

        import llmcore

        default_config = Path(llmcore.__file__).parent / "config" / "default_config.toml"
        section = dict(tomllib.loads(default_config.read_text())["providers"]["commandcode"])
        assert "zdr" not in section
        monkeypatch.setenv("CMD_ZDR", "1")
        assert CommandCodeProvider({**section, "api_key": KEY}).zdr is True
        monkeypatch.delenv("CMD_ZDR")
        assert CommandCodeProvider({**section, "api_key": KEY}).zdr is False

    def test_lanes_report_the_outer_instance_name(self):
        p = CommandCodeProvider({"api_key": KEY, "_instance_name": "cc-work"})
        for route in ("chat", "responses", "messages", "system_one"):
            assert p._lane(route, False).get_name() == "cc-work"

    def test_transports(self):
        direct = CommandCodeProvider({"api_key": KEY})
        assert direct._lane("chat", False)._transport == "httpx"
        assert direct._lane("messages", False)._backend == "httpx"
        sdk = CommandCodeProvider({"api_key": KEY, "transport": "sdk"})
        chat, msgs = sdk._lane("chat", False), sdk._lane("messages", False)
        assert chat._transport == "sdk" and msgs._backend == "sdk"
        # The official SDK clients exist and point at the gateway, not the vendors.
        assert str(chat._client.base_url).rstrip("/") == V1
        assert str(msgs._client.base_url).rstrip("/") == ROOT

    def test_sdk_transport_carries_the_zdr_header(self):
        sdk = CommandCodeProvider({"api_key": KEY, "transport": "sdk", "zdr": True})
        assert sdk._lane("messages", True)._client.default_headers[cc.ZDR_HEADER] == "1"
        assert sdk._lane("chat", True)._client.default_headers[cc.ZDR_HEADER] == "1"

    def test_supported_parameters_follow_the_route(self, provider):
        provider.catalog.load_rows(CATALOG)
        chat = provider.get_supported_parameters("kimi-k3")
        claude = provider.get_supported_parameters("claude-haiku-5-5")
        assert {"effort", "reasoning", "max_tokens", "zdr"} <= set(chat)
        assert {"thinking", "effort", "output_config", "zdr"} <= set(claude)

    def test_context_from_catalog_then_cards(self, provider):
        provider.catalog.load_rows([{"id": "x/y", "context_length": 4242,
                                     "supported_endpoints": ["/chat/completions"]}])
        assert provider.get_max_context_length("x/y") == 4242
        assert provider.get_max_context_length("gpt-6-sol") >= 400000  # from its card


# ---------------------------------------------------------------------------
# Chat Completions lane
# ---------------------------------------------------------------------------


class TestChatLane:
    async def test_request_shape(self, provider, gateway):
        route = gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_body(), headers=TRACE)
        )
        await provider.chat_completion(CTX, "kimi-k3", effort="high", max_tokens=50)
        req = route.calls.last.request
        body = json.loads(req.content)
        assert req.headers["authorization"] == f"Bearer {KEY}"
        assert body["model"] == "moonshotai/Kimi-K3"  # short name resolved
        assert body["messages"][0]["role"] == "system"  # never "developer"
        assert body["reasoning_effort"] == "high"
        assert "effort" not in body
        assert cc.ZDR_HEADER not in req.headers

    async def test_response_parsing(self, provider, gateway):
        gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_body(), headers=TRACE)
        )
        resp = await provider.chat_completion(CTX, "inkling-2-small:free")
        assert provider.extract_response_content(resp) == "391"
        assert provider.extract_reasoning_content(resp) == "17*23 = 391."
        assert provider.extract_reasoning_details(resp)[0]["type"] == "reasoning.text"
        usage = provider.extract_usage_details(resp)
        assert usage["cached_tokens"] == 4
        assert usage["cache_write_tokens"] == 7
        assert usage["reasoning_tokens"] == 25
        assert usage["trace_id"] == "trace123"
        assert usage["upstream"] == "Thinking Machines"
        assert usage["route"] == "chat"
        assert provider.last_call_info()["trace_id"] == "trace123"

    async def test_reasoning_details_are_echoed_back(self, provider, gateway):
        route = gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_body())
        )
        details = [{"type": "reasoning.text", "text": "t"}]
        history = [
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="a", metadata={"reasoning_details": details}),
            Message(role=Role.USER, content="q2"),
        ]
        await provider.chat_completion(history, "kimi-k3")
        sent = json.loads(route.calls.last.request.content)["messages"]
        assert sent[1]["reasoning_details"] == details

    async def test_tools(self, provider, gateway):
        body = _chat_body(content=None)
        body["choices"][0]["message"]["tool_calls"] = [{
            "type": "function", "index": 0, "id": "functions.get_weather:0",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon\"}"},
        }]
        route = gateway.post(f"{V1}/chat/completions").mock(return_value=httpx.Response(200, json=body))
        tool = Tool(name="get_weather", description="w",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}})
        resp = await provider.chat_completion(CTX, "kimi-k3", tools=[tool])
        calls = provider.extract_tool_calls(resp)
        assert calls[0].name == "get_weather" and calls[0].arguments == {"city": "Lisbon"}
        assert json.loads(route.calls.last.request.content)["tools"][0]["function"]["name"] == "get_weather"

    async def test_inband_error_raises_instead_of_returning_partial_output(self, provider, gateway):
        body = _chat_body(content='{"a": ')
        body["choices"][0].update({
            "finish_reason": "error",
            "error": {"code": 502, "message": "Upstream error from Thinking Machines: parse",
                      "metadata": {"error_type": "provider_unavailable"}},
        })
        gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(200, json=body, headers=TRACE)
        )
        with pytest.raises(ProviderError) as info:
            await provider.chat_completion(CTX, "kimi-k3")
        err = info.value
        assert err.status_code == 502 and err.retryable is True
        assert err.failure_kind == "server"
        assert err.partial_content == '{"a": '
        assert "discarded 6 chars" in str(err) and "trace123" in str(err)

    async def test_streaming(self, provider, gateway):
        route = gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=_sse(_chunk("3"), _chunk("91"), _chunk("", finish_reason="stop")),
                headers={"content-type": "text/event-stream", **TRACE},
            )
        )
        stream = await provider.chat_completion(CTX, "kimi-k3", stream=True)
        text = "".join([provider.extract_delta_content(c) async for c in stream])
        assert text == "391"
        assert json.loads(route.calls.last.request.content)["stream_options"] == {"include_usage": True}

    async def test_stream_inband_error_raises_after_partial(self, provider, gateway):
        bad = _chunk("", finish_reason="error")
        bad["choices"][0]["error"] = {"code": 502, "message": "boom"}
        gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(200, content=_sse(_chunk("par"), bad),
                                        headers={"content-type": "text/event-stream"})
        )
        stream = await provider.chat_completion(CTX, "kimi-k3", stream=True)
        got: list[str] = []
        with pytest.raises(ProviderError, match="mid-response") as info:
            async for chunk in stream:
                got.append(provider.extract_delta_content(chunk))
        assert got == ["par"]
        assert info.value.partial_content == "par"

    async def test_stream_failure_before_first_byte_raises_at_call_time(self, provider, gateway):
        gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "slow down",
                                                             "type": "rate_limit_error"}})
        )
        with pytest.raises(ProviderError) as info:
            await provider.chat_completion(CTX, "kimi-k3", stream=True)
        assert info.value.failure_kind == "rate_limit"


# ---------------------------------------------------------------------------
# Messages lane
# ---------------------------------------------------------------------------


class TestMessagesLane:
    async def test_request_shape_and_parsing(self, provider, gateway):
        route = gateway.post(f"{V1}/messages").mock(
            return_value=httpx.Response(200, json=_messages_body(), headers=TRACE)
        )
        resp = await provider.chat_completion(CTX, "claude-haiku-5-5", effort="low", max_tokens=64)
        req = route.calls.last.request
        body = json.loads(req.content)
        assert req.headers["x-api-key"] == KEY
        assert req.headers["anthropic-version"]
        assert body["system"] == "Be terse."
        assert body["output_config"]["effort"] == "low"  # inherited effort mapping
        assert provider.extract_response_content(resp) == "Hello there!"
        usage = provider.extract_usage_details(resp)
        assert usage["cached_tokens"] == 11 and usage["cache_write_tokens"] == 3
        assert usage["trace_id"] == "trace123" and usage["route"] == "messages"

    async def test_streaming(self, provider, gateway):
        events = [
            {"type": "message_start", "message": {"id": "m", "model": "claude-haiku-5-5",
                                                  "usage": {"input_tokens": 5, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]
        gateway.post(f"{V1}/messages").mock(
            return_value=httpx.Response(200, content=_sse(*events, done=False),
                                        headers={"content-type": "text/event-stream"})
        )
        stream = await provider.chat_completion(CTX, "claude-haiku-5-5", stream=True)
        assert "".join([provider.extract_delta_content(c) async for c in stream]) == "Hello"

    async def test_stale_catalog_heals_itself(self, provider, gateway):
        """A model the catalog put on the wrong route is re-routed after one refresh."""
        await provider.catalog.ensure()
        provider.catalog._models["claude-haiku-5-5"] = CommandCodeModel(
            id="claude-haiku-5-5", endpoints=("/chat/completions",)
        )
        gateway.post(f"{V1}/chat/completions").mock(return_value=httpx.Response(400, json={
            "error": {"message": 'Model "claude-haiku-5-5" must be called via /provider/v1/messages '
                                 "(Anthropic Messages shape).",
                      "type": "invalid_request_error", "param": "model", "code": "unsupported_model"}
        }))
        messages = gateway.post(f"{V1}/messages").mock(
            return_value=httpx.Response(200, json=_messages_body("ok"))
        )
        resp = await provider.chat_completion(CTX, "claude-haiku-5-5")
        assert provider.extract_response_content(resp) == "ok"
        assert resp["commandcode"]["route"] == "messages"
        assert messages.call_count == 1


# ---------------------------------------------------------------------------
# Zero data retention
# ---------------------------------------------------------------------------


class TestZdr:
    async def test_config_sends_the_header_on_every_lane(self, gateway):
        p = CommandCodeProvider({"api_key": KEY, "zdr": True})
        chat = gateway.post(f"{V1}/chat/completions").mock(return_value=httpx.Response(200, json=_chat_body()))
        msgs = gateway.post(f"{V1}/messages").mock(return_value=httpx.Response(200, json=_messages_body()))
        await p.chat_completion(CTX, "kimi-k3")
        await p.chat_completion(CTX, "claude-haiku-5-5")
        assert chat.calls.last.request.headers[cc.ZDR_HEADER] == "1"
        assert msgs.calls.last.request.headers[cc.ZDR_HEADER] == "1"

    async def test_per_call_override(self, provider, gateway):
        route = gateway.post(f"{V1}/chat/completions").mock(return_value=httpx.Response(200, json=_chat_body()))
        await provider.chat_completion(CTX, "kimi-k3", zdr=True)
        assert route.calls.last.request.headers[cc.ZDR_HEADER] == "1"
        await provider.chat_completion(CTX, "kimi-k3")
        assert cc.ZDR_HEADER not in route.calls.last.request.headers

    async def test_no_zdr_upstream(self, provider, gateway):
        gateway.post(f"{V1}/chat/completions").mock(return_value=httpx.Response(422, json={
            "error": {"message": "This model has no zero-data-retention upstream.",
                      "type": "invalid_request_error", "code": "cmd_zdr_no_providers"}
        }))
        with pytest.raises(ProviderError, match="no ZDR upstream") as info:
            await provider.chat_completion(CTX, "inkling-2-small:free", zdr=True)
        assert info.value.failure_kind == "model_not_found"
        assert info.value.retryable is False


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "error", "kind", "retryable", "phrase"),
        [
            (400, {"message": 'Model "nope" is not supported on this endpoint.',
                   "code": "unsupported_model"}, "model_not_found", False, "does not serve"),
            (401, {"message": "Invalid 'Authorization' header or token.",
                   "type": "authentication_error", "code": "UNAUTHORIZED"}, "auth", False, "CMD_API_KEY"),
            (403, {"message": "upgrade", "code": "upgrade_required"}, "auth", False, "Go"),
            (403, {"message": "region", "code": "model_region_unavailable"},
             "model_not_found", False, "region"),
            (403, {"message": "no", "code": "content_policy_refused"}, "refusal", False, "refused"),
            (413, {"message": "big", "code": "request_too_large"}, "bad_request", False, "32 MiB"),
            (429, {"message": "slow", "type": "rate_limit_error"}, "rate_limit", True, "backoff"),
            (402, {"message": "You've reached your 5-hour usage limit."},
             "insufficient_credit", False, "limit"),
            (502, {"message": "down", "code": "upstream_unavailable"}, "server", True, "not charged"),
            (500, {"message": "oops", "type": "server_error"}, "server", True, "unavailable"),
        ],
    )
    def test_table(self, status, error, kind, retryable, phrase):
        err = build_commandcode_error("commandcode", status, {"error": error}, model="m",
                                      headers=httpx.Headers(TRACE))
        assert err.failure_kind == kind
        assert err.retryable is retryable
        assert err.status_code == status
        assert phrase in str(err)
        assert "trace_id=trace123" in str(err)

    def test_double_encoded_message_is_unwrapped_with_routing(self):
        inner = {
            "error": {"message": "This model does not support structured output.",
                      "type": "invalid_request_error"},
            "providerMetadata": {"gateway": {"routing": {
                "finalProvider": "stealth",
                "modelAttempts": [{"providerAttempts": [{"provider": "stealth", "statusCode": 400}]}],
            }}},
        }
        body = {"error": {"message": json.dumps(inner), "type": "invalid_request_error"}}
        err = build_commandcode_error("commandcode", 400, body, model="m")
        assert "does not support structured output" in str(err)
        assert "providerMetadata" not in str(err)
        assert "stealth=400" in str(err)
        assert err.gateway_routing["finalProvider"] == "stealth"

    def test_anthropic_and_router_shapes(self):
        anth = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        assert "slow" in str(build_commandcode_error("cc", 429, anth, model="m"))
        hono = {"success": False, "status": 404, "message": "404 Not found.",
                "cause": "GET /x is not a registered API route"}
        err = build_commandcode_error("cc", 404, hono, model="m")
        assert "not a registered API route" in str(err)

    def test_context_overflow_is_a_context_length_error(self):
        err = build_commandcode_error(
            "cc", 400, {"error": {"message": "prompt is too long: context length exceeded"}}, model="m"
        )
        assert isinstance(err, ContextLengthError)

    def test_routing_honours_the_hint(self):
        from llmcore.routing.models import FailureKind, classify_failure

        region = build_commandcode_error(
            "cc", 403, {"error": {"message": "x", "code": "model_region_unavailable"}}, model="m"
        )
        # Without the hint a 403 would bench the whole provider as an auth failure.
        assert classify_failure(region)[0] is FailureKind.MODEL_NOT_FOUND
        plain = ProviderError("x", "forbidden", status_code=403)
        assert classify_failure(plain)[0] is FailureKind.AUTH

    def test_sdk_exceptions_are_remapped(self):
        class FakeStatusError(Exception):
            status_code = 422
            body = {"message": "no zdr", "code": "cmd_zdr_no_providers"}  # noqa: RUF012
            response = httpx.Response(422, headers=TRACE)

        try:
            try:
                raise FakeStatusError("422")
            except FakeStatusError:
                # No `from`: this is exactly how the base providers wrap SDK
                # errors, leaving the SDK exception reachable as __context__.
                raise ProviderError("commandcode", "API Error (422): no zdr")  # noqa: B904
        except ProviderError as wrapped:
            mapped = cc._from_sdk_exception("commandcode", wrapped, model="m", route="chat")
        assert mapped is not None
        assert mapped.failure_kind == "model_not_found"
        assert mapped.trace_id == "trace123"


# ---------------------------------------------------------------------------
# System One, discovery, registration
# ---------------------------------------------------------------------------


class TestSystemOne:
    async def test_round_trip(self, provider):
        with respx.mock() as mock:
            route = mock.post(f"{V1}/systemone").mock(return_value=httpx.Response(200, json={
                "model": "typesafe/jev", "answers": {"done": {"type": "noul", "noul": 0.59}},
                "usage": {"input_tokens": 284, "output_tokens": 20},
            }, headers=TRACE))
            res = await provider.system_one("tests pass", {"done": Noul(instructions="Finished?")})
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "typesafe/jev"
        assert body["questions"]["done"]["instructions"] == "Finished?"
        assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
        assert res.answers_dict()["done"]["noul"] == 0.59

    async def test_errors_use_the_gateway_mapping(self, provider):
        with respx.mock() as mock:
            mock.post(f"{V1}/systemone").mock(return_value=httpx.Response(422, json={
                "error": {"message": "no zdr", "code": "cmd_zdr_no_providers"}}, headers=TRACE))
            with pytest.raises(ProviderError) as info:
                await provider.system_one("s", {"q": Noul(instructions="?")}, zdr=True)
        assert info.value.failure_kind == "model_not_found"
        assert "trace123" in str(info.value)

    async def test_chat_with_a_decision_model_points_to_system_one(self, provider):
        provider.catalog.load_rows([{"id": "typesafe/jev", "supported_endpoints": ["/systemone"]}])
        with pytest.raises(ProviderError, match="system_one"):
            await provider.chat_completion(CTX, "typesafe/jev")


class TestDiscovery:
    async def test_models_details_carry_routes_and_card_capabilities(self, provider, gateway):
        details = {d.id: d for d in await provider.get_models_details()}
        kimi = details["moonshotai/Kimi-K3"]
        assert kimi.provider_name == "commandcode"
        assert kimi.metadata["route"] == "chat"
        assert kimi.context_length == 1000000
        assert kimi.supports_tools is True  # from the commandcode card
        assert details["claude-haiku-5-5"].metadata["route"] == "messages"


class TestRegistration:
    def test_provider_map_and_aliases(self):
        from llmcore.providers.manager import _PROVIDER_INSTANCE_ALIASES, PROVIDER_MAP

        for name in ("commandcode", "command_code", "command-code"):
            assert PROVIDER_MAP[name] is CommandCodeProvider
        assert _PROVIDER_INSTANCE_ALIASES["command-code"] == "commandcode"

    def test_dynamic_targets_find_the_key(self, monkeypatch):
        from llmcore.providers.manager import ProviderManager

        monkeypatch.setenv("COMMAND_CODE_API_KEY", "k")
        key, tried = ProviderManager._discover_credential("commandcode", None)
        assert key == "k"
        # Tried in order, stopping at the first one that is set.
        assert tried == ["CMD_API_KEY", "COMMAND_CODE_API_KEY"]


# ---------------------------------------------------------------------------
# Responses lane
# ---------------------------------------------------------------------------


def _responses_body(text: str = "No", **extra: Any) -> dict[str, Any]:
    body = {
        "id": "resp_1", "object": "response", "model": "gpt-6-luna", "status": "completed",
        "output": [
            {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": text, "annotations": []}]},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 26, "total_tokens": 46,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 19}},
    }
    body.update(extra)
    return body


class TestResponsesLane:
    async def test_gpt_goes_to_responses_by_default(self, provider, gateway):
        route = gateway.post(f"{V1}/responses").mock(
            return_value=httpx.Response(200, json=_responses_body(), headers=TRACE)
        )
        resp = await provider.chat_completion(CTX, "gpt-6-luna", effort="low", max_tokens=99)
        body = json.loads(route.calls.last.request.content)
        assert body["instructions"] == "Be terse."
        assert body["input"] == [{"role": "user", "content": "hi"}]
        assert body["reasoning"] == {"effort": "low"}
        assert body["max_output_tokens"] == 99
        assert body["store"] is False
        assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
        assert provider.extract_response_content(resp) == "No"
        assert provider.extract_reasoning_items(resp)[0]["encrypted_content"] == "enc"
        usage = provider.extract_usage_details(resp)
        assert usage["reasoning_tokens"] == 19 and usage["route"] == "responses"
        assert usage["trace_id"] == "trace123"

    async def test_per_call_chat_override(self, provider, gateway):
        chat = gateway.post(f"{V1}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_body())
        )
        await provider.chat_completion(CTX, "gpt-6-luna", responses=False)
        assert chat.call_count == 1

    async def test_failed_status_raises(self, provider, gateway):
        gateway.post(f"{V1}/responses").mock(return_value=httpx.Response(200, json=_responses_body(
            status="failed", error={"code": "server_error", "message": "upstream died"})))
        with pytest.raises(ProviderError, match="upstream died") as info:
            await provider.chat_completion(CTX, "gpt-6-luna")
        assert info.value.retryable is True and info.value.failure_kind == "server"

    async def test_http_errors_use_the_gateway_mapping(self, provider, gateway):
        gateway.post(f"{V1}/responses").mock(return_value=httpx.Response(429, json={
            "error": {"message": "slow", "type": "rate_limit_error"}}, headers=TRACE))
        with pytest.raises(ProviderError) as info:
            await provider.chat_completion(CTX, "gpt-6-luna")
        assert info.value.failure_kind == "rate_limit" and info.value.trace_id == "trace123"

    async def test_streaming(self, provider, gateway):
        events = [
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-6-luna"}},
            {"type": "response.output_text.delta", "delta": "N"},
            {"type": "response.output_text.delta", "delta": "o"},
            {"type": "response.completed", "response": _responses_body()},
        ]
        sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        gateway.post(f"{V1}/responses").mock(return_value=httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}))
        stream = await provider.chat_completion(CTX, "gpt-6-luna", stream=True)
        chunks = [c async for c in stream]
        assert "".join(provider.extract_delta_content(c) for c in chunks) == "No"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    async def test_tool_loop_round_trip(self, provider, gateway):
        route = gateway.post(f"{V1}/responses").mock(
            return_value=httpx.Response(200, json=_responses_body("22C, sunny"))
        )
        items = [{"id": "rs_1", "type": "reasoning", "encrypted_content": "enc"}]
        history = [
            Message(role=Role.USER, content="Weather in Lisbon?"),
            Message(role=Role.ASSISTANT, content="", metadata={"reasoning_items": items},
                    tool_calls=[{"id": "call_1", "type": "function", "function": {
                        "name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}}]),
            Message(role=Role.TOOL, content="22C sunny", tool_call_id="call_1"),
        ]
        tool = Tool(name="get_weather", description="w",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}})
        await provider.chat_completion(history, "gpt-6-luna", tools=[tool])
        sent = json.loads(route.calls.last.request.content)
        assert [i.get("type", i.get("role")) for i in sent["input"]] == [
            "user", "reasoning", "function_call", "function_call_output"]
        assert sent["tools"][0]["name"] == "get_weather"

    async def test_extra_tools_pass_through(self, provider, gateway):
        route = gateway.post(f"{V1}/responses").mock(
            return_value=httpx.Response(200, json=_responses_body())
        )
        await provider.chat_completion(CTX, "gpt-6-luna", extra_tools=[{"type": "web_search"}])
        assert json.loads(route.calls.last.request.content)["tools"] == [{"type": "web_search"}]

    def test_parameters(self, provider):
        provider.catalog.load_rows(CATALOG)
        params = provider.get_supported_parameters("gpt-6-luna")
        assert {"reasoning", "text", "extra_tools", "responses", "zdr"} <= set(params)
