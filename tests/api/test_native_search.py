# tests/api/test_native_search.py
"""Tests for the ``native_search`` chat option gating in :class:`LLMCore`.

These exercise the api-layer routing added for provider-native web search
(plan §4/F9 dependency): ``LLMCore.chat`` must forward ``native_search=True``
to ``chat_completion`` only when the resolved provider advertises a native
search surface (``provider.supports_native_search``); otherwise it is a silent,
byte-identical no-op. The provider-side payload mapping is covered separately in
``tests/providers``.

Fully offline: a fake provider records the ``native_search`` kwarg it receives.
"""

from __future__ import annotations

from typing import Any

from llmcore import LLMCore
from llmcore.models import Message
from llmcore.providers.base import BaseProvider

_FAKE_REPLY = "ok"


class _NativeSearchFakeProvider(BaseProvider):
    """Offline provider that records whether ``native_search`` reached it."""

    def __init__(self, supports: bool):
        super().__init__({}, False)
        self.default_model = "fake-model-1"
        self._supports = supports
        self.received_native_search: Any = "UNSET"

    def get_name(self) -> str:
        return "fake"

    def supports_native_search(self, model: str | None = None) -> bool:
        return self._supports

    async def get_models_details(self) -> list[Any]:
        return []

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        return 8192

    async def chat_completion(
        self,
        context: Any,
        model: str | None = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: str | None = None,
        native_search: Any = "UNSET",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.received_native_search = native_search
        return {"content": _FAKE_REPLY}

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len((text or "").split()))

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        return max(1, sum(len((getattr(m, "content", "") or "").split()) for m in messages))

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return chunk.get("content", "")


async def _make_llm(tmp_path, supports: bool):
    llm = await LLMCore.create(
        config_overrides={
            "storage": {
                "vector": {"type": ""},
                "session": {"type": "json", "path": str(tmp_path / "sessions")},
            }
        }
    )
    fake = _NativeSearchFakeProvider(supports=supports)
    llm._provider_manager._providers["fake"] = fake
    llm._provider_manager._default_provider_name = "fake"
    return llm, fake


class TestNativeSearchGating:
    async def test_forwarded_when_provider_supports(self, tmp_path):
        llm, fake = await _make_llm(tmp_path, supports=True)
        try:
            await llm.chat(
                message="latest news?",
                provider_name="fake",
                save_session=False,
                native_search=True,
            )
            assert fake.received_native_search is True
        finally:
            await llm.close()

    async def test_noop_when_provider_unsupported(self, tmp_path):
        llm, fake = await _make_llm(tmp_path, supports=False)
        try:
            await llm.chat(
                message="latest news?",
                provider_name="fake",
                save_session=False,
                native_search=True,
            )
            # native_search must NOT be forwarded -> default sentinel untouched.
            assert fake.received_native_search == "UNSET"
        finally:
            await llm.close()

    async def test_default_off_not_forwarded(self, tmp_path):
        llm, fake = await _make_llm(tmp_path, supports=True)
        try:
            await llm.chat(
                message="hi",
                provider_name="fake",
                save_session=False,
            )
            assert fake.received_native_search == "UNSET"
        finally:
            await llm.close()

    async def test_chat_with_usage_threads_native_search(self, tmp_path):
        llm, fake = await _make_llm(tmp_path, supports=True)
        try:
            await llm.chat_with_usage(
                message="latest news?",
                provider_name="fake",
                save_session=False,
                native_search=True,
            )
            assert fake.received_native_search is True
        finally:
            await llm.close()
