# tests/providers/test_deepgram_agent.py
"""Voice-Agent + text-intelligence + auth/account tests (fake sockets/clients).

Covers Part 4:

* ``open_voice_agent`` / ``run_voice_agent`` (``agent.v1``), including
  settings assembly, event mapping, audio frames, runtime steering
  (inject/update), and automatic function-call dispatch.
* ``analyze_text`` (``read.v1``), ``grant_token`` (``auth.v1``), ``get_projects``
  (``manage.v1``).

All network surfaces are faked. The agent fake records every ``send_*`` and
replays a scripted event stream (the agent protocol has no close frame, so the
fake simply ends after its scripted messages).
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

import pytest

from llmcore.exceptions import ProviderError
from llmcore.models_multimodal import VoiceAgentEventType
from llmcore.providers.deepgram_provider import DeepgramProvider, deepgram_available

from .test_deepgram_provider import _install_fake_client
from .test_deepgram_streaming import _bind_connect, _Holder

# Optional extra: skip the whole module when ``deepgram-sdk`` is not installed
# (the agent paths build/validate real ``AgentV1Settings`` and message types).
pytestmark = pytest.mark.skipif(
    not deepgram_available,
    reason="deepgram-sdk not installed (optional extra: pip install llmcore[deepgram])",
)

# Minimal but complete agent config so AgentV1Settings.model_validate succeeds
# (``audio`` and ``agent`` are required by the SDK model).
_AGENT_CONFIG: dict[str, Any] = {
    "agent": {
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 24000},
            "output": {"encoding": "linear16", "sample_rate": 24000},
        },
        "listen": {"provider": {"type": "deepgram", "model": "nova-3"}},
        "think": {"provider": {"type": "open_ai", "model": "gpt-4o-mini"}},
        "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
    }
}


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any] | None = None
) -> DeepgramProvider:
    _install_fake_client(monkeypatch)
    cfg: dict[str, Any] = {"api_key": "dg-test-key", "_instance_name": "deepgram"}
    if config:
        cfg.update(config)
    return DeepgramProvider(cfg)


def _set_attr(root: Any, path: list[str], value: Any) -> None:
    """Set ``value`` at a dotted attribute path, creating namespaces as needed."""
    node = root
    for a in path[:-1]:
        child = getattr(node, a, None)
        if child is None:
            child = _Holder()
            setattr(node, a, child)
        node = child
    setattr(node, path[-1], value)


class _FakeResp:
    """Response stand-in whose ``model_dump`` returns a pre-baked nested dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


# ---------------------------------------------------------------------------
# Fake agent socket
# ---------------------------------------------------------------------------


class _FakeAgentSocket:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.settings: Any = None
        self.sent_audio: list[bytes] = []
        self.injected_user: list[str] = []
        self.injected_agent: list[str] = []
        self.prompts: list[str] = []
        self.thinks: list[Any] = []
        self.speaks: list[Any] = []
        self.function_responses: list[tuple[str | None, str, str]] = []
        self.keepalives = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def send_settings(self, message: Any) -> None:
        self.settings = message

    async def send_media(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def send_inject_user_message(self, message: Any) -> None:
        self.injected_user.append(message.content)

    async def send_inject_agent_message(self, message: Any) -> None:
        self.injected_agent.append(message.message)

    async def send_update_prompt(self, message: Any) -> None:
        self.prompts.append(message.prompt)

    async def send_update_think(self, message: Any) -> None:
        self.thinks.append(message)

    async def send_update_speak(self, message: Any) -> None:
        self.speaks.append(message)

    async def send_function_call_response(self, message: Any) -> None:
        self.function_responses.append((message.id, message.name, message.content))

    async def send_keep_alive(self, message: Any = None) -> None:
        self.keepalives += 1

    async def __aiter__(self):
        for m in self.messages:
            await asyncio.sleep(0)
            yield m


# ---------------------------------------------------------------------------
# open_voice_agent / event mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_voice_agent_sends_settings_and_maps_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch, _AGENT_CONFIG)
    socket = _FakeAgentSocket(
        [
            {"type": "Welcome", "request_id": "rid"},
            {"type": "ConversationText", "role": "assistant", "content": "Hello!"},
            b"\x01\x02",  # audio frame
        ]
    )
    _bind_connect(p.client, ["agent", "v1", "connect"], socket)

    events = []
    async with p.open_voice_agent(prompt="You are helpful.", greeting="Hi!") as session:
        async for ev in session:
            events.append(ev)

    # Settings were sent and carry the (never-defaulted) prompt + greeting.
    assert socket.settings is not None
    dumped = socket.settings.model_dump()
    assert dumped["type"] == "Settings"
    assert dumped["agent"]["think"]["prompt"] == "You are helpful."
    assert dumped["agent"]["greeting"] == "Hi!"
    assert dumped["audio"]["input"]["encoding"] == "linear16"

    types = [e.type for e in events]
    assert types == [
        VoiceAgentEventType.WELCOME,
        VoiceAgentEventType.CONVERSATION_TEXT,
        VoiceAgentEventType.AUDIO,
    ]
    assert events[1].content == "Hello!"
    assert events[2].audio == b"\x01\x02"


@pytest.mark.asyncio
async def test_voice_agent_session_steering_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch, _AGENT_CONFIG)
    socket = _FakeAgentSocket([])
    _bind_connect(p.client, ["agent", "v1", "connect"], socket)

    async with p.open_voice_agent() as session:
        await session.send_audio(b"mic")
        await session.inject_user_message("hi there")
        await session.inject_agent_message("Welcome aboard.")
        await session.update_prompt("new system prompt")
        await session.update_think({"provider": {"type": "open_ai", "model": "gpt-4o"}})
        await session.update_speak({"provider": {"type": "deepgram", "model": "aura-asteria-en"}})
        await session.respond_to_function_call("f1", "get_weather", "sunny")
        await session.keepalive()

    assert socket.sent_audio == [b"mic"]
    assert socket.injected_user == ["hi there"]
    assert socket.injected_agent == ["Welcome aboard."]
    assert socket.prompts == ["new system prompt"]
    assert socket.thinks[0].model_dump()["think"]["provider"]["model"] == "gpt-4o"
    assert socket.speaks[0].model_dump()["speak"]["provider"]["model"] == "aura-asteria-en"
    assert socket.function_responses == [("f1", "get_weather", "sunny")]
    assert socket.keepalives == 1


# ---------------------------------------------------------------------------
# run_voice_agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_voice_agent_auto_answers_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch, _AGENT_CONFIG)
    script = [
        {"type": "Welcome"},
        {
            "type": "FunctionCallRequest",
            "functions": [
                {
                    "id": "f1",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                    "client_side": True,
                }
            ],
        },
        {"type": "ConversationText", "role": "assistant", "content": "It's sunny."},
    ]
    socket = _FakeAgentSocket(script)
    _bind_connect(p.client, ["agent", "v1", "connect"], socket)

    seen: list[Any] = []

    async def _handler(call: Any) -> str:
        assert call.name == "get_weather"
        assert call.arguments == {"city": "SF"}
        return "72F and sunny"

    async def _on_event(ev: Any) -> None:
        seen.append(ev.type)

    async def _audio():
        yield b"mic"

    events = [
        e
        async for e in p.run_voice_agent(
            _audio(), on_event=_on_event, function_handler=_handler, prompt="be helpful"
        )
    ]

    assert socket.function_responses == [("f1", "get_weather", "72F and sunny")]
    assert socket.sent_audio == [b"mic"]
    assert [e.type for e in events] == [
        VoiceAgentEventType.WELCOME,
        VoiceAgentEventType.FUNCTION_CALL_REQUEST,
        VoiceAgentEventType.CONVERSATION_TEXT,
    ]
    assert seen == [e.type for e in events]  # on_event saw every event


@pytest.mark.asyncio
async def test_run_voice_agent_producer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch, _AGENT_CONFIG)
    # Long-but-finite script so the receive loop stays alive until the producer
    # fails on its second pull.
    socket = _FakeAgentSocket([{"type": "Welcome"}] + [{"type": "AgentThinking"}] * 50)
    _bind_connect(p.client, ["agent", "v1", "connect"], socket)

    async def _bad_audio():
        yield b"a"
        raise RuntimeError("mic died")

    with pytest.raises(ProviderError) as ei:
        _ = [e async for e in p.run_voice_agent(_bad_audio())]
    assert "producer" in str(ei.value).lower()


# ---------------------------------------------------------------------------
# analyze_text
# ---------------------------------------------------------------------------


def _analyze_response() -> _FakeResp:
    return _FakeResp(
        {
            "metadata": {"request_id": "rid", "models": ["read-model"]},
            "results": {
                "summary": {"text": "A short summary."},
                "topics": {
                    "segments": [
                        {"text": "weather", "topics": [{"topic": "weather", "confidence": 0.9}]}
                    ]
                },
                "intents": {"segments": [{"intents": [{"intent": "ask_weather"}]}]},
                "sentiments": {"average": {"sentiment": "positive"}},
            },
        }
    )


@pytest.mark.asyncio
async def test_analyze_text_with_text(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_analyze(*, request: Any, **kwargs: Any) -> Any:
        captured["request"] = request
        captured.update(kwargs)
        return _analyze_response()

    _set_attr(p.client, ["read", "v1", "text", "analyze"], _fake_analyze)

    result = await p.analyze_text(
        "How's the weather?", summarize=True, topics=True, sentiment=True, intents=True
    )
    assert captured["request"] == {"text": "How's the weather?"}
    assert captured["summarize"] is True
    assert result.summary == "A short summary."
    assert result.topics == [
        {"text": "weather", "topics": [{"topic": "weather", "confidence": 0.9}]}
    ]
    assert result.intents == [{"intents": [{"intent": "ask_weather"}]}]
    assert result.sentiments == {"average": {"sentiment": "positive"}}
    assert result.request_id == "rid"
    assert result.model == "read-model"


@pytest.mark.asyncio
async def test_analyze_text_with_url(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_analyze(*, request: Any, **kwargs: Any) -> Any:
        captured["request"] = request
        return _analyze_response()

    _set_attr(p.client, ["read", "v1", "text", "analyze"], _fake_analyze)
    _ = await p.analyze_text(url="https://example.com/article.txt", summarize=True)
    assert captured["request"] == {"url": "https://example.com/article.txt"}


@pytest.mark.asyncio
async def test_analyze_text_requires_exactly_one_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _make_provider(monkeypatch)
    with pytest.raises(ProviderError) as ei_none:
        await p.analyze_text()
    assert ei_none.value.status_code == 400
    with pytest.raises(ProviderError):
        await p.analyze_text("text", url="https://x")


# ---------------------------------------------------------------------------
# grant_token + get_projects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grant_token(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_grant(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeResp({"access_token": "tok-123", "expires_in": 30.0})

    _set_attr(p.client, ["auth", "v1", "tokens", "grant"], _fake_grant)

    out = await p.grant_token(ttl_seconds=30)
    assert out["access_token"] == "tok-123"
    assert out["expires_in"] == 30.0
    assert captured["ttl_seconds"] == 30

    # Without ttl, no ttl kwarg is forwarded.
    captured.clear()
    await p.grant_token()
    assert "ttl_seconds" not in captured


@pytest.mark.asyncio
async def test_get_projects(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _make_provider(monkeypatch)

    async def _fake_list() -> Any:
        return _FakeResp({"projects": [{"project_id": "p1", "name": "Default"}]})

    _set_attr(p.client, ["manage", "v1", "projects", "list"], _fake_list)
    out = await p.get_projects()
    assert out["projects"][0]["project_id"] == "p1"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
