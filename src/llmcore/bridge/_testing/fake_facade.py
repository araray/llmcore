"""A deterministic, offline ``LLMCoreFacade`` for conformance testing.

Behaviour is fully reproducible and requires no provider keys or network:

* ``chat`` / ``chat_with_usage`` echo the prompt as ``"echo: <message>"`` and
  compute token usage by whitespace splitting. The streaming form yields fixed
  8-character slices so ``"".join(chunks) == unary_text`` exactly.
* Scriptable triggers (carried in the message) exercise non-happy paths:
  - ``__error__:<key>``     — raise a mapped llmcore exception at call setup.
  - ``__error_mid__:<key>`` — yield one chunk, then raise mid-stream.
  - ``__cancel__``          — yield two chunks, then await forever (cancellation).

Supported ``<key>`` values: ``provider_rate_limited``, ``provider_unauthorized``,
``config``, ``context_length``, ``not_found``, ``embedding``, ``storage``,
``unsupported``, ``internal``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator

from llmcore.exceptions import (
    ConfigError,
    ContextLengthError,
    EmbeddingError,
    LLMCoreError,
    ProviderError,
    SessionNotFoundError,
    StorageError,
)
from llmcore.models import ModelDetails
from llmcore.usage import ChatUsage

_ERROR_PREFIX = "__error__:"
_ERROR_MID_PREFIX = "__error_mid__:"
_CANCEL_TOKEN = "__cancel__"


def fake_count_tokens(text: str | None, model: str | None = None) -> int:
    """Deterministic token counter: number of whitespace-delimited words."""
    return len((text or "").split())


def _raise_for_key(key: str) -> None:
    key = key.strip()
    if key == "provider_rate_limited":
        raise ProviderError(
            "fake", "rate limited", model_name="fake-1", status_code=429, retry_after_seconds=2.0
        )
    if key == "provider_unauthorized":
        raise ProviderError("fake", "invalid api key", model_name="fake-1", status_code=401)
    if key == "config":
        raise ConfigError("bad configuration value")
    if key == "context_length":
        raise ContextLengthError("fake-1", 8192, 9000, "prompt too long")
    if key == "not_found":
        raise SessionNotFoundError("sess-does-not-exist")
    if key == "embedding":
        raise EmbeddingError("embed-1", "embedding backend failure")
    if key == "storage":
        raise StorageError("session store unavailable")
    if key == "unsupported":
        raise NotImplementedError("provider does not support this capability")
    if key == "internal":
        raise LLMCoreError("unexpected internal failure")
    raise LLMCoreError(f"unknown error trigger: {key}")


class FakeFacade:
    """Deterministic stand-in implementing the ``LLMCoreFacade`` protocol."""

    PROVIDER = "fake"
    MODEL = "fake-1"

    def __init__(self) -> None:
        # Tier-1 in-memory store (offline). Keyed by session id; values are real
        # ``llmcore.models.ChatSession`` objects so the bridge conversion helpers
        # exercise the same code path as a live ``LLMCore``.
        self._sessions: dict[str, Any] = {}
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    # -- chat ------------------------------------------------------------- #
    async def chat_with_usage(self, message: str, **kw: Any) -> tuple[str, ChatUsage]:
        if message.startswith(_ERROR_PREFIX):
            _raise_for_key(message[len(_ERROR_PREFIX):])
        text = "echo: " + message
        usage = ChatUsage(
            prompt_tokens=fake_count_tokens(message),
            completion_tokens=fake_count_tokens(text),
            total_tokens=fake_count_tokens(message) + fake_count_tokens(text),
            provider=kw.get("provider_name") or self.PROVIDER,
            model=kw.get("model_name") or self.MODEL,
        )
        return text, usage

    async def chat(
        self, message: str, *, stream: bool = False, **kw: Any
    ) -> "str | AsyncGenerator[str, None]":
        if message.startswith(_ERROR_PREFIX):
            _raise_for_key(message[len(_ERROR_PREFIX):])
        text = "echo: " + message
        if not stream:
            return text
        return self._stream(message, text)

    def _stream(self, message: str, text: str) -> AsyncGenerator[str, None]:
        async def gen() -> AsyncGenerator[str, None]:
            if message.strip() == _CANCEL_TOKEN:
                yield text[:4]
                yield text[4:8]
                await asyncio.Event().wait()  # never set -> cancellation point
                return
            if message.startswith(_ERROR_MID_PREFIX):
                yield text[:4]
                _raise_for_key(message[len(_ERROR_MID_PREFIX):])
                return
            for i in range(0, len(text), 8):
                yield text[i : i + 8]

        return gen()

    # -- cost / catalog --------------------------------------------------- #
    def estimate_cost(
        self,
        provider_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> Any:
        from llmcore.models import CostEstimate

        in_rate, out_rate = 1.0, 2.0  # USD per 1M tokens
        input_cost = prompt_tokens / 1_000_000 * in_rate
        output_cost = completion_tokens / 1_000_000 * out_rate
        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            currency="USD",
            pricing_source="model_card",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            input_price_per_million=in_rate,
            output_price_per_million=out_rate,
            model_id=model_name,
            provider=provider_name,
        )

    def get_provider_details(self, provider_name: str | None = None) -> ModelDetails:
        return ModelDetails(
            id=self.MODEL,
            provider_name=provider_name or self.PROVIDER,
            display_name="Fake Model 1",
            context_length=8192,
            max_output_tokens=4096,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
            supports_reasoning=False,
            family="Fake",
            model_type="chat",
            metadata={"vendor": "fake"},
        )

    def get_available_providers(self) -> list[str]:
        return [self.PROVIDER]

    def get_models_for_provider(self, provider_name: str) -> list[str]:
        if provider_name in (self.PROVIDER, "", None):
            return ["fake-1", "fake-2"]
        return []

    # -- providers / audio (Tier-2) -------------------------------------- #
    def get_provider(self, provider_name: str | None = None) -> Any:
        """Return a deterministic, offline audio-capable provider.

        See :class:`FakeAudioProvider`. Returned regardless of audio mode; the
        Tier-2 *capability* (and the handler's UNIMPLEMENTED short-circuit) is
        gated separately by :meth:`supports_audio`.
        """
        from .fake_provider import FakeAudioProvider

        return FakeAudioProvider()

    def supports_audio(self) -> bool:
        """Tier-2 enablement gate for the fake bridge.

        Read dynamically from ``LLMCORE_BRIDGE_FAKE_AUDIO`` so a test can toggle
        audio on/off against an already-constructed server/fixture. Off by
        default, which keeps the Tier-0 capability set byte-identical and the
        five Tier-0 language clients' "tier2.audio rejected" assertions valid.
        """
        return os.getenv("LLMCORE_BRIDGE_FAKE_AUDIO") == "1"

    # -- Tier-1: sessions & context items -------------------------------- #
    def supports_sessions(self) -> bool:
        """Tier-1 enablement gate for the fake bridge.

        Read dynamically from ``LLMCORE_BRIDGE_FAKE_SESSIONS`` so a test can
        toggle sessions on/off against an already-constructed fixture. Off by
        default, which keeps the Tier-0 capability set byte-identical for the
        existing conformance suites.
        """
        return os.getenv("LLMCORE_BRIDGE_FAKE_SESSIONS") == "1"

    def _require(self, session_id: str) -> Any:
        s = self._sessions.get(session_id)
        if s is None:
            raise SessionNotFoundError(session_id)
        return s

    async def create_session(
        self,
        session_id: str | None = None,
        name: str | None = None,
        system_message: str | None = None,
    ) -> Any:
        from llmcore.models import ChatSession, Message, Role

        sid = session_id or self._next_id("sess")
        if sid in self._sessions:
            raise StorageError(f"session already exists: {sid}")
        messages = []
        if system_message:
            messages.append(Message(session_id=sid, role=Role.SYSTEM, content=system_message))
        session = ChatSession(id=sid, name=name, messages=messages)
        self._sessions[sid] = session
        return session

    async def get_session(self, session_id: str) -> Any:
        return self._require(session_id)

    async def list_sessions(self, limit: int | None = None) -> list[Any]:
        sessions = list(self._sessions.values())
        if limit is not None:
            sessions = sessions[:limit]
        return sessions

    async def delete_session(self, session_id: str) -> None:
        self._require(session_id)
        del self._sessions[session_id]

    async def update_session_name(self, session_id: str, new_name: str) -> None:
        self._require(session_id).name = new_name

    async def fork_session(
        self,
        session_id: str,
        *,
        new_name: str | None = None,
        from_message_id: str | None = None,
        message_ids: list[str] | None = None,
        message_range: tuple[int, int] | None = None,
        include_context_items: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from llmcore.models import ChatSession

        src = self._require(session_id)
        messages = list(src.messages)
        if message_ids:
            wanted = set(message_ids)
            messages = [m for m in messages if m.id in wanted]
        elif message_range is not None:
            start, end = message_range
            messages = messages[start:end]
        elif from_message_id is not None:
            ids = [m.id for m in messages]
            if from_message_id in ids:
                messages = messages[: ids.index(from_message_id) + 1]
        new_id = self._next_id("fork")
        fork = ChatSession(
            id=new_id,
            name=new_name or (src.name and f"{src.name} (fork)"),
            messages=[m.model_copy() for m in messages],
            context_items=[ci.model_copy() for ci in src.context_items] if include_context_items else [],
            metadata=dict(metadata or {}),
        )
        self._sessions[new_id] = fork
        return new_id

    async def clone_session(
        self,
        session_id: str,
        new_name: str | None = None,
        *,
        include_messages: bool = True,
        include_context_items: bool = True,
    ) -> str:
        from llmcore.models import ChatSession

        src = self._require(session_id)
        new_id = self._next_id("clone")
        clone = ChatSession(
            id=new_id,
            name=new_name or (src.name and f"{src.name} (clone)"),
            messages=[m.model_copy() for m in src.messages] if include_messages else [],
            context_items=[ci.model_copy() for ci in src.context_items] if include_context_items else [],
            metadata=dict(src.metadata),
        )
        self._sessions[new_id] = clone
        return new_id

    async def delete_messages(self, session_id: str, message_ids: list[str]) -> int:
        session = self._require(session_id)
        wanted = set(message_ids)
        before = len(session.messages)
        session.messages = [m for m in session.messages if m.id not in wanted]
        return before - len(session.messages)

    async def get_messages_by_range(
        self, session_id: str, start_index: int, end_index: int
    ) -> list[Any]:
        return self._require(session_id).messages[start_index:end_index]

    async def add_context_item(
        self,
        session_id: str,
        content: str,
        item_type: Any = None,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from llmcore.models import ContextItem, ContextItemType

        session = self._require(session_id)
        item = ContextItem(
            type=item_type or ContextItemType.USER_TEXT,
            content=content,
            source_id=source_id,
            tokens=fake_count_tokens(content),
            metadata=dict(metadata or {}),
        )
        session.context_items.append(item)
        return item.id

    async def get_context_item(self, session_id: str, item_id: str) -> Any:
        session = self._require(session_id)
        for item in session.context_items:
            if item.id == item_id:
                return item
        return None

    async def remove_context_item(self, session_id: str, item_id: str) -> bool:
        session = self._require(session_id)
        before = len(session.context_items)
        session.context_items = [ci for ci in session.context_items if ci.id != item_id]
        return len(session.context_items) < before

    # -- lifecycle -------------------------------------------------------- #
    async def reload_config(self) -> None:
        return None

    async def close(self) -> None:
        return None
