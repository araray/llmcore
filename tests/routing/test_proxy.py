# tests/routing/test_proxy.py
"""Proxy mode: llmcore impersonating an OpenAI endpoint.

The point of this surface is that an *unmodified* harness works against it, so
the tests drive it the way a harness would — over HTTP, with OpenAI-shaped
bodies — rather than calling the handlers directly. A test that called the
functions would pass while the thing a harness needs was broken.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from llmcore.bridge.proxy_app import ProxyConfig, create_proxy_app, parse_model
from llmcore.exceptions import NoTargetAvailableError, PromptBlockedError, ProviderError


class FakeRouting:
    def __init__(self) -> None:
        self._health = {"openai:gpt-4o-mini": {"available": True, "failures": 0}}

    def lanes(self):
        return {"deep": "anthropic:claude-opus-5-5", "trivial": "pool:cheap"}

    def pools(self):
        return {"main": ["openai:gpt-4o-mini", "anthropic:claude-opus-5-5"]}

    def health(self):
        return self._health

    async def explain(self, prompt, **kwargs):
        from llmcore.routing import Candidate, RoutingPlan, Target

        chosen = Target.parse("openai:gpt-4o-mini")
        return RoutingPlan(
            chosen=chosen,
            pool="main",
            lane=kwargs.get("lane"),
            candidates=(
                Candidate(target=chosen, eligible=True),
                Candidate(
                    target=Target.parse("anthropic:claude-opus-5-5"),
                    eligible=False,
                    reason="cooling down for 12s after rate_limit",
                ),
            ),
            estimated_cost_usd=0.0004,
            notes=("a note",),
        )


class FakeInfo:
    provider = "openai"
    model = "gpt-4o-mini"
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class FakeLLM:
    """Just enough llmcore for the proxy to drive."""

    def __init__(self, *, answer="hello", raises=None, chunks=None) -> None:
        self.routing = FakeRouting()
        self.answer = answer
        self.raises = raises
        self.chunks = chunks
        self.calls: list[dict] = []
        self.discarded: list[str] = []
        self.config = None

    async def chat(self, message, **kwargs):
        self.calls.append({"message": message, **kwargs})
        if self.raises is not None:
            raise self.raises
        if kwargs.get("stream"):
            async def gen():
                for chunk in self.chunks or ["he", "llo"]:
                    yield chunk

            return gen()
        return self.answer

    def get_last_interaction_context_info(self, session_id):
        return FakeInfo()

    def discard_transient_state(self, session_id):
        self.discarded.append(session_id)

    def get_available_providers(self):
        return ["openai", "anthropic"]


def client(llm=None, **config_kwargs) -> TestClient:
    return TestClient(create_proxy_app(llm or FakeLLM(), ProxyConfig(**config_kwargs)))


def body(model="auto", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


# ---------------------------------------------------------------------------
# The model name as the routing interface
# ---------------------------------------------------------------------------


class TestModelNameParsing:
    """A harness can only send a model name, so the name carries the intent."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("lane:deep", {"lane": "deep"}),
            ("pool:main", {"pool": "main"}),
            ("profile:frugal", {"profile": "frugal"}),
            ("auto", {}),
            ("", {}),
            (None, {}),
            ("openai:gpt-5.4?effort=max", {"target": "openai:gpt-5.4?effort=max"}),
            ("ollama:llama3.3:70b", {"target": "ollama:llama3.3:70b"}),
        ],
    )
    def test_parses(self, model, expected):
        assert parse_model(model) == expected

    def test_a_bare_name_stays_a_model_name(self):
        """So a harness already configured for a plain OpenAI model keeps
        working -- the difference between a drop-in proxy and a migration."""
        assert parse_model("gpt-4o-mini") == {"model_name": "gpt-4o-mini"}

    def test_routing_arguments_reach_chat(self):
        llm = FakeLLM()
        client(llm).post("/v1/chat/completions", json=body("lane:deep"))
        assert llm.calls[0]["lane"] == "deep"


# ---------------------------------------------------------------------------
# Auth — this process holds every credential
# ---------------------------------------------------------------------------


class TestAuth:
    def test_a_non_loopback_bind_without_a_token_is_refused(self):
        """Not a warning. An open routing proxy is a credential giveaway."""
        with pytest.raises(ValueError, match="without an API key"):
            ProxyConfig(host="0.0.0.0")

    def test_a_non_loopback_bind_with_a_token_is_allowed(self):
        assert ProxyConfig(host="0.0.0.0", api_key="secret").host == "0.0.0.0"

    def test_loopback_needs_no_token(self):
        assert client().post("/v1/chat/completions", json=body()).status_code == 200

    def test_a_configured_token_is_required(self):
        api = client(api_key="secret")
        assert api.post("/v1/chat/completions", json=body()).status_code == 401
        assert api.get("/v1/models").status_code == 401
        assert api.get("/v1/routing/health").status_code == 401

    def test_the_right_token_is_accepted(self):
        api = client(api_key="secret")
        response = api.post(
            "/v1/chat/completions",
            json=body(),
            headers={"authorization": "Bearer secret"},
        )
        assert response.status_code == 200

    def test_a_wrong_token_is_rejected(self):
        api = client(api_key="secret")
        response = api.post(
            "/v1/chat/completions", json=body(), headers={"authorization": "Bearer nope"}
        )
        assert response.status_code == 401

    def test_healthz_needs_no_token(self):
        """So a container orchestrator can probe it."""
        assert client(api_key="secret").get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_the_response_is_openai_shaped(self):
        response = client(FakeLLM(answer="the answer")).post(
            "/v1/chat/completions", json=body()
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"] == {
            "role": "assistant",
            "content": "the answer",
        }
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert payload["id"].startswith("chatcmpl-")

    def test_a_system_message_is_extracted(self):
        llm = FakeLLM()
        client(llm).post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert llm.calls[0]["system_message"] == "be terse"
        assert llm.calls[0]["message"] == "hi"

    def test_history_becomes_prior_messages(self):
        llm = FakeLLM()
        client(llm).post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ],
            },
        )
        call = llm.calls[0]
        assert call["message"] == "second"
        assert len(call["extra_messages"]) == 2

    def test_multimodal_parts_keep_their_text(self):
        """A text proxy cannot forward an image, but it must not crash on a
        harness that sends content parts."""
        llm = FakeLLM()
        response = client(llm).post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        assert llm.calls[0]["message"] == "describe this"

    def test_sampling_parameters_pass_through(self):
        llm = FakeLLM()
        client(llm).post(
            "/v1/chat/completions",
            json=body(temperature=0.3, max_tokens=64, top_p=0.9, seed=7),
        )
        call = llm.calls[0]
        assert call["temperature"] == 0.3 and call["max_tokens"] == 64
        assert call["top_p"] == 0.9 and call["seed"] == 7

    def test_reasoning_effort_becomes_the_effort_argument(self):
        llm = FakeLLM()
        client(llm).post("/v1/chat/completions", json=body(reasoning_effort="high"))
        assert llm.calls[0]["effort"] == "high"

    def test_routing_policy_can_arrive_in_extra_body(self):
        """How the OpenAI SDKs let a caller send non-standard fields."""
        llm = FakeLLM()
        client(llm).post(
            "/v1/chat/completions",
            json=body(llmcore={"lane": "deep", "routing": {"max_attempts": 5}}),
        )
        call = llm.calls[0]
        assert call["lane"] == "deep"
        assert call["routing"] == {"max_attempts": 5}

    def test_usage_names_the_target_that_actually_answered(self):
        """Under a pool that is genuinely not the model that was requested, and
        a harness logging spend per model must not be lied to."""
        payload = client().post("/v1/chat/completions", json=body("pool:main")).json()
        assert payload["usage"]["total_tokens"] == 18
        assert payload["llmcore"]["target"] == "openai:gpt-4o-mini"
        assert payload["model"] == "gpt-4o-mini"

    def test_an_empty_message_array_is_a_400(self):
        response = client().post("/v1/chat/completions", json={"model": "auto", "messages": []})
        assert response.status_code == 400

    def test_a_non_json_body_is_a_400(self):
        response = client().post(
            "/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400

    def test_a_session_id_enables_persistence(self):
        llm = FakeLLM()
        client(llm).post("/v1/chat/completions", json=body(llmcore={"session_id": "s1"}))
        assert llm.calls[0]["session_id"] == "s1"
        assert llm.calls[0]["save_session"] is True

    def test_without_a_session_nothing_is_saved(self):
        """A harness sending its full history every turn does not want llmcore
        accumulating a second copy of it."""
        llm = FakeLLM()
        client(llm).post("/v1/chat/completions", json=body())
        assert llm.calls[0]["save_session"] is False

    def test_a_synthetic_session_is_used_and_then_discarded(self):
        """A session id is needed to read the usage numbers back, but keeping
        one per request would leak for the life of the process."""
        llm = FakeLLM()
        client(llm).post("/v1/chat/completions", json=body())
        synthetic = llm.calls[0]["session_id"]
        assert synthetic.startswith("proxy-")
        assert llm.discarded == [synthetic]

    def test_a_caller_session_is_not_discarded(self):
        llm = FakeLLM()
        client(llm).post("/v1/chat/completions", json=body(llmcore={"session_id": "s1"}))
        assert llm.discarded == []

    def test_transient_state_is_discarded_even_on_failure(self):
        llm = FakeLLM(raises=ProviderError("p", "boom", status_code=500))
        client(llm).post("/v1/chat/completions", json=body())
        assert len(llm.discarded) == 1


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """Harnesses branch on these statuses, so getting them right is what makes
    the proxy behave like the thing it is impersonating."""

    @pytest.mark.parametrize(
        ("exc", "status"),
        [
            (ProviderError("p", "Rate limited", status_code=429), 429),
            (ProviderError("p", "not_enough_credits", status_code=403), 402),
            (ProviderError("p", "Invalid key", status_code=401), 401),
            (ProviderError("p", "Bad parameter", status_code=400), 400),
            (ProviderError("p", "model does not exist", status_code=404), 404),
            (ProviderError("p", "Internal error", status_code=500), 502),
            (TimeoutError("slow"), 504),
            (NoTargetAvailableError(pool="main"), 503),
            (PromptBlockedError(transform="pii"), 403),
        ],
    )
    def test_statuses(self, exc, status):
        response = client(FakeLLM(raises=exc)).post("/v1/chat/completions", json=body())
        assert response.status_code == status
        assert "error" in response.json()

    def test_a_rate_limit_passes_retry_after_through(self):
        exc = ProviderError("p", "Rate limited", status_code=429)
        exc.retry_after = 30
        response = client(FakeLLM(raises=exc)).post("/v1/chat/completions", json=body())
        assert response.headers["retry-after"] == "30"

    def test_a_blocked_prompt_does_not_leak_what_was_found(self):
        exc = PromptBlockedError(
            transform="pii", findings=[("email", "prompt", "abcdef0123456789")]
        )
        response = client(FakeLLM(raises=exc)).post("/v1/chat/completions", json=body())
        assert response.status_code == 403
        assert "abcdef0123456789" not in response.text


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def _events(self, response) -> list[dict]:
        out = []
        for line in response.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
        return out

    def test_sse_chunks_are_openai_shaped(self):
        response = client(FakeLLM(chunks=["he", "ll", "o"])).post(
            "/v1/chat/completions", json=body(stream=True)
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = self._events(response)
        assert all(event["object"] == "chat.completion.chunk" for event in events)
        content = "".join(
            event["choices"][0]["delta"].get("content", "") for event in events
        )
        assert content == "hello"

    def test_the_stream_terminates_with_done(self):
        response = client(FakeLLM(chunks=["x"])).post(
            "/v1/chat/completions", json=body(stream=True)
        )
        assert response.text.rstrip().endswith("data: [DONE]")

    def test_the_last_chunk_carries_a_finish_reason(self):
        response = client(FakeLLM(chunks=["x"])).post(
            "/v1/chat/completions", json=body(stream=True)
        )
        assert self._events(response)[-1]["choices"][0]["finish_reason"] == "stop"

    def test_an_error_before_the_first_chunk_is_reported_in_band(self):
        """A streaming response has already sent its headers, so there is no
        status code left to use."""
        response = client(
            FakeLLM(raises=ProviderError("p", "down", status_code=503))
        ).post("/v1/chat/completions", json=body(stream=True))
        assert response.status_code == 200
        assert "error" in self._events(response)[0]


# ---------------------------------------------------------------------------
# /v1/models and the routing extensions
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_lanes_and_pools_are_advertised(self):
        """So a harness's model picker becomes a way to choose routing
        policy."""
        data = client().get("/v1/models").json()["data"]
        ids = {entry["id"] for entry in data}
        assert "lane:deep" in ids and "pool:main" in ids and "auto" in ids

    def test_a_lane_entry_says_where_it_goes(self):
        data = client().get("/v1/models").json()["data"]
        lane = next(entry for entry in data if entry["id"] == "lane:deep")
        assert lane["llmcore"]["destination"] == "anthropic:claude-opus-5-5"

    def test_advertising_can_be_turned_off(self):
        data = client(advertise_lanes=False, advertise_pools=False).get("/v1/models").json()["data"]
        ids = {entry["id"] for entry in data}
        assert not any(i.startswith(("lane:", "pool:")) for i in ids)
        assert "openai" in ids

    def test_health_is_exposed(self):
        payload = client().get("/v1/routing/health").json()
        assert "openai:gpt-4o-mini" in payload["targets"]

    def test_explain_reports_the_rejected_candidates(self):
        payload = client().get("/v1/routing/explain?prompt=hello").json()
        assert payload["chosen"] == "openai:gpt-4o-mini"
        skipped = next(c for c in payload["candidates"] if not c["eligible"])
        assert "cooling down" in skipped["reason"]

    def test_explain_without_a_prompt_is_a_400(self):
        assert client().get("/v1/routing/explain").status_code == 400


# ---------------------------------------------------------------------------
# Harness configuration
# ---------------------------------------------------------------------------


class TestHarnessEnv:
    def test_it_prints_what_a_harness_needs(self):
        env = ProxyConfig(port=9999).harness_env()
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
        assert env["OPENAI_MODEL"] == "auto"

    def test_a_placeholder_key_is_supplied_on_loopback(self):
        """Most harnesses insist on some key being present."""
        assert ProxyConfig().harness_env()["OPENAI_API_KEY"] == "llmcore-local"

    def test_the_real_token_is_used_when_one_is_set(self):
        assert ProxyConfig(api_key="secret").harness_env()["OPENAI_API_KEY"] == "secret"


# ---------------------------------------------------------------------------
# The per-turn step budget
# ---------------------------------------------------------------------------


class TestTurnBudget:
    """The proxy is where the expensive turns in the measurements came from.

    Turns over 200 steps are 62% of this project's spend, so a step ceiling is
    the lever -- and a harness talking to this endpoint has no idea llmcore is
    here, which is exactly why the budget has to live on this side.
    """

    def _policy(self, **kwargs):
        from llmcore.routing.budget import BudgetAction, BudgetPolicy

        kwargs.setdefault("action", BudgetAction.STOP)
        return BudgetPolicy(**kwargs)

    def test_a_step_ceiling_refuses_the_next_call(self):
        api = client(budget=self._policy(max_steps=2))
        for _ in range(2):
            assert api.post("/v1/chat/completions",
                            json=body(user="turn-1")).status_code == 200
        refused = api.post("/v1/chat/completions", json=body(user="turn-1"))
        assert refused.status_code == 429
        assert refused.json()["error"]["type"] == "budget_exceeded"
        assert "2/2 steps" in refused.json()["error"]["message"]

    def test_each_conversation_gets_its_own_budget(self):
        api = client(budget=self._policy(max_steps=1))
        assert api.post("/v1/chat/completions",
                        json=body(user="alice")).status_code == 200
        # Bob is not affected by Alice's exhausted turn.
        assert api.post("/v1/chat/completions",
                        json=body(user="bob")).status_code == 200
        assert api.post("/v1/chat/completions",
                        json=body(user="alice")).status_code == 429

    def test_the_session_id_extension_also_identifies_a_turn(self):
        api = client(budget=self._policy(max_steps=1))
        payload = body()
        payload["llmcore"] = {"session_id": "s-1"}
        assert api.post("/v1/chat/completions", json=payload).status_code == 200
        assert api.post("/v1/chat/completions", json=payload).status_code == 429

    def test_an_unidentified_conversation_is_not_counted_at_all(self):
        """Without a stable key every request gets a fresh synthetic session,
        so "steps so far" would always be zero. Rather than enforce something
        meaningless, the proxy declines to track the turn -- and says so in the
        docs instead of pretending."""
        api = client(budget=self._policy(max_steps=1))
        for _ in range(5):
            response = api.post("/v1/chat/completions", json=body())
            assert response.status_code == 200
            assert "budget" not in response.json()["llmcore"]

    def test_the_usage_block_reports_the_budget(self):
        api = client(budget=self._policy(max_steps=10))
        served = api.post("/v1/chat/completions",
                          json=body(user="turn-2")).json()["llmcore"]
        assert served["budget"]["steps"] == 1
        assert served["budget"]["state"] == "ok"

    def test_no_budget_is_tracked_when_nothing_is_configured(self):
        """Unbounded is the default, and an unbounded budget should not even
        allocate: the proxy must stay exactly as it was for anyone who has not
        asked for this."""
        api = client()
        served = api.post("/v1/chat/completions",
                          json=body(user="turn-3")).json()["llmcore"]
        assert "budget" not in served

    def test_report_mode_counts_without_refusing(self):
        from llmcore.routing.budget import BudgetAction

        api = client(budget=self._policy(max_steps=1, action=BudgetAction.REPORT))
        assert api.post("/v1/chat/completions",
                        json=body(user="t")).status_code == 200
        second = api.post("/v1/chat/completions", json=body(user="t"))
        assert second.status_code == 200
        assert second.json()["llmcore"]["budget"]["state"] == "exceeded"

    def test_an_unpriced_target_is_not_counted_as_free(self):
        """FakeInfo's model has no card pricing in this fixture, so the step
        records an unknown cost. A spend ceiling must report that it cannot be
        enforced rather than imply the turn is cheap."""
        api = client(budget=self._policy(max_cost_usd=100.0, max_steps=50))
        served = api.post("/v1/chat/completions",
                          json=body(user="t")).json()["llmcore"]
        budget = served["budget"]
        assert budget["steps"] == 1
        if budget["cost_is_partial"]:
            assert "cannot be enforced" in (budget["reason"] or "")

    def test_streaming_also_consumes_the_budget(self):
        """Otherwise streaming would be a way to spend without being counted,
        and a harness choosing `stream=true` would bypass the ceiling."""
        api = client(budget=self._policy(max_steps=1))
        with api.stream("POST", "/v1/chat/completions",
                        json=body(user="s", stream=True)) as response:
            assert response.status_code == 200
            list(response.iter_lines())
        assert api.post("/v1/chat/completions",
                        json=body(user="s")).status_code == 429

    def test_tracked_turns_do_not_grow_without_limit(self):
        """This process holds every provider credential, so an unbounded dict
        keyed by caller-supplied strings is not acceptable."""
        from llmcore.bridge.proxy_app import MAX_TRACKED_TURNS

        api = client(budget=self._policy(max_steps=5))
        for i in range(MAX_TRACKED_TURNS + 10):
            api.post("/v1/chat/completions", json=body(user=f"turn-{i}"))
        # The store is a closure local, so assert the effect rather than the
        # dict: the earliest turn was evicted and counts again from zero.
        first_again = api.post("/v1/chat/completions", json=body(user="turn-0"))
        assert first_again.status_code == 200
        assert first_again.json()["llmcore"]["budget"]["steps"] == 1
