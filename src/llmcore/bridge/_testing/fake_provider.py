"""A deterministic ``BaseProvider`` for the real-``LLMCore`` integration test.

Registered into ``llmcore.providers.manager.PROVIDER_MAP`` under the type key
``"fake"`` so a ``[providers.fake]`` config section makes ``LLMCore.create()``
wire a fully offline provider. Used only by ``test_integration_real_llmcore.py``
(skip-guarded on confy availability); never imported by the bridge runtime.

The streaming chunk / non-streaming response shapes match what this provider's
own ``extract_*`` methods consume, so ``LLMCore.chat`` works end-to-end.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from llmcore.models import Message, ModelDetails
from llmcore.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """Offline echo provider with OpenAI-shaped raw responses."""

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        super().__init__(config, log_raw_payloads=log_raw_payloads)
        self._config = config
        self.default_model = config.get("default_model", "fake-1")

    def get_name(self) -> str:
        return self._provider_instance_name or "fake"

    async def get_models_details(self) -> list[ModelDetails]:
        return [
            ModelDetails(
                id="fake-1",
                provider_name=self.get_name(),
                display_name="Fake Model 1",
                context_length=8192,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                model_type="chat",
            )
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {"temperature": True, "top_p": True, "max_tokens": True}

    def get_max_context_length(self, model: str | None = None) -> int:
        return 8192

    async def count_tokens(self, text: str | None, model: str | None = None) -> int:
        return len((text or "").split())

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        return sum(len((m.content or "").split()) for m in messages)

    def _last_user_text(self, context: Any) -> str:
        # ``context`` is the prepared prompt (list[Message] | str | dict-ish).
        try:
            if isinstance(context, str):
                return context
            if isinstance(context, list) and context:
                last = context[-1]
                return getattr(last, "content", None) or (
                    last.get("content", "") if isinstance(last, dict) else ""
                )
        except Exception:
            pass
        return ""

    async def chat_completion(
        self,
        context: Any,
        model: str | None = None,
        stream: bool = False,
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        text = "echo: " + self._last_user_text(context)
        if not stream:
            return {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "model": model or self.default_model,
                "usage": {
                    "prompt_tokens": len(self._last_user_text(context).split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(self._last_user_text(context).split())
                    + len(text.split()),
                },
            }

        async def gen() -> AsyncGenerator[dict[str, Any], None]:
            for i in range(0, len(text), 8):
                yield {"choices": [{"delta": {"content": text[i : i + 8]}}]}

        return gen()

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        try:
            return chunk["choices"][0]["delta"].get("content", "") or ""
        except (KeyError, IndexError, AttributeError):
            return ""


def register_fake_provider() -> None:
    """Register :class:`FakeProvider` under PROVIDER_MAP['fake'] (idempotent)."""
    from llmcore.providers import manager

    manager.PROVIDER_MAP["fake"] = FakeProvider


class FakeAudioProvider:
    """Deterministic Tier-2 audio surface for bridge conformance tests (B3).

    Intentionally **not** a :class:`BaseProvider` subclass: it implements only
    the audio methods the bridge ``AudioService`` actually calls, so it can be
    returned by ``FakeFacade.get_provider()`` and drive the gRPC/WebSocket audio
    handlers fully offline. Crucially, it yields the *real*
    ``llmcore.models_multimodal`` pydantic events, so the bridge's
    ``event -> proto`` mapping is exercised identically to a real provider.

    ``transcribe_stream`` consumes the inbound audio iterable and, for each
    non-empty chunk, treats the decoded bytes as one "word": it emits an
    ``INTERIM`` event per chunk, then a single ``FINAL`` event whose ``text`` is
    the space-joined words, then an ``UTTERANCE_END``. This is fully
    deterministic and lets the e2e assert ``final.text == " ".join(words)``.
    """

    def get_name(self) -> str:
        return "fake"

    async def transcribe_stream(
        self, audio: Any = None, **kwargs: Any
    ) -> "AsyncGenerator[Any, None]":
        from llmcore.models_multimodal import StreamEventType, TranscriptionStreamEvent

        words: list[str] = []
        if audio is not None:
            async for chunk in audio:
                if not chunk:
                    continue
                if isinstance(chunk, (bytes, bytearray)):
                    word = bytes(chunk).decode("utf-8", "replace")
                else:
                    word = str(chunk)
                words.append(word)
                yield TranscriptionStreamEvent(
                    type=StreamEventType.INTERIM,
                    text=word,
                    is_final=False,
                    provider="fake",
                )
        yield TranscriptionStreamEvent(
            type=StreamEventType.FINAL,
            text=" ".join(words),
            is_final=True,
            speech_final=True,
            provider="fake",
        )
        yield TranscriptionStreamEvent(
            type=StreamEventType.UTTERANCE_END,
            text="",
            provider="fake",
        )

    async def stream_speech(
        self, text: Any, **kwargs: Any
    ) -> "AsyncGenerator[bytes, None]":
        """Deterministic streaming TTS: emit each text piece as UTF-8 bytes.

        Mirrors ``DeepgramProvider.stream_speech``'s dual-mode signature: a plain
        ``str`` yields a single chunk; an async iterable of text pieces yields
        one chunk per non-empty piece. Lets the SynthesizeStream e2e assert
        ``b"".join(out) == "".join(pieces).encode()``.
        """
        if isinstance(text, str):
            if text:
                yield text.encode("utf-8")
            return
        async for piece in text:
            if piece:
                yield piece.encode("utf-8")

    @asynccontextmanager
    async def open_voice_agent(
        self,
        *,
        settings: Any = None,
        prompt: str | None = None,
        functions: list[Any] | None = None,
        greeting: str | None = None,
        **kwargs: Any,
    ) -> "AsyncGenerator[FakeVoiceAgentSession, None]":
        """Open a deterministic voice-agent session (offline).

        Mirrors ``DeepgramProvider.open_voice_agent`` (an async context manager
        yielding a session). The session reacts to inbound calls by enqueueing
        mapped :class:`~llmcore.models_multimodal.VoiceAgentEvent`s.
        """
        session = FakeVoiceAgentSession()
        try:
            yield session
        finally:
            session.close()


#: Sentinel enqueued by :meth:`FakeVoiceAgentSession.close` to end iteration.
_CLOSE_SENTINEL = object()


def _va_event(type_name: str, **fields: Any) -> Any:
    """Build a fake ``VoiceAgentEvent`` of the named type (``provider='fake'``)."""
    from llmcore.models_multimodal import VoiceAgentEvent, VoiceAgentEventType

    return VoiceAgentEvent(type=VoiceAgentEventType(type_name), provider="fake", **fields)


class FakeVoiceAgentSession:
    """Deterministic, offline analogue of ``DeepgramVoiceAgentSession``.

    Exposes the public surface the bridge drives (``send_audio`` / ``inject_*``
    / ``update_*`` / ``respond_to_function_call`` / ``keepalive`` / ``__aiter__``)
    plus a fake-only :meth:`close` the bridge calls to end the event stream
    (real sessions have no close frame). Inbound calls enqueue mapped
    ``VoiceAgentEvent``s; iteration drains the queue (FIFO) until close, so the
    duplex e2e sees a deterministic sequence regardless of task interleaving.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False
        # Lifecycle: a WELCOME is available immediately on open.
        self._queue.put_nowait(_va_event("welcome"))

    # -- inbound (driven by the bridge pump) ------------------------------ #
    async def send_audio(self, chunk: bytes) -> None:
        if chunk:
            self._queue.put_nowait(_va_event("audio", audio=b"agent:" + bytes(chunk)))

    async def inject_user_message(self, content: str) -> None:
        self._queue.put_nowait(
            _va_event("conversation_text", role="user", content=content)
        )

    async def inject_agent_message(self, message: str) -> None:
        self._queue.put_nowait(
            _va_event("conversation_text", role="assistant", content=message)
        )

    async def update_prompt(self, prompt: str) -> None:
        self._queue.put_nowait(_va_event("prompt_updated"))

    async def update_think(self, think: dict) -> None:
        self._queue.put_nowait(_va_event("think_updated"))

    async def update_speak(self, speak: dict) -> None:
        self._queue.put_nowait(_va_event("speak_updated"))

    async def respond_to_function_call(
        self, function_id: str, name: str, content: str
    ) -> None:
        self._queue.put_nowait(
            _va_event(
                "conversation_text",
                role="assistant",
                content=f"fn-result:{function_id}={content}",
            )
        )

    async def keepalive(self) -> None:
        return None

    def close(self) -> None:
        """End the event stream (fake-only affordance for the bridge)."""
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_CLOSE_SENTINEL)

    # -- outbound --------------------------------------------------------- #
    async def __aiter__(self):
        while True:
            event = await self._queue.get()
            if event is _CLOSE_SENTINEL:
                yield _va_event("close")
                return
            yield event
