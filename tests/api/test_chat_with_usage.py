# tests/api/test_chat_with_usage.py
"""Tests for the :meth:`LLMCore.chat_with_usage` usage-returning surface.

This suite validates the Part 3 §3.2.2 "Option A" usage surface that lets a
caller meter token consumption per call without enabling session
persistence. It covers:

* :class:`llmcore.ChatUsage` value semantics (naming aliases, availability,
  construction from introspection details, immutability).
* The signature/protocol contract a metering caller (e.g. Convergence's
  ``LLMBackendWithUsage``) relies on.
* End-to-end behaviour against a fully **offline** fake provider: token
  counts populated, provider/model recorded, ephemeral-session cleanup,
  caller-supplied-session passthrough, and streaming rejection.

No real network calls are made: a :class:`FakeProvider` is injected into the
provider manager and vector storage is disabled.

Phase: v0.9 — llmcore usage surface (Convergence Spec Part 3 §3.2, gate #10).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from llmcore import ChatUsage, LLMCore
from llmcore.models import ContextPreparationDetails, Message
from llmcore.providers.base import BaseProvider

# ---------------------------------------------------------------------------
# Offline fake provider
# ---------------------------------------------------------------------------

_FAKE_REPLY = "Hello from the fake provider."


class FakeProvider(BaseProvider):
    """A minimal, fully-offline :class:`BaseProvider`.

    Token counting is deterministic (whitespace word count) so assertions on
    prompt/completion token values are stable, and ``chat_completion`` returns
    a canned dict — no network is ever touched.
    """

    def __init__(self, config: dict[str, Any] | None = None, log_raw_payloads: bool = False):
        super().__init__(config or {}, log_raw_payloads)
        self.default_model = "fake-model-1"
        self.chat_completion_calls = 0

    def get_name(self) -> str:
        return "fake"

    async def get_models_details(self) -> list[Any]:
        return []

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        # No provider-specific kwargs accepted (keeps chat()'s validation simple).
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.chat_completion_calls += 1
        return {"content": _FAKE_REPLY}

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len((text or "").split()))

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        total = 0
        for m in messages:
            total += len((getattr(m, "content", "") or "").split())
        return max(1, total)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return chunk.get("content", "")


@pytest.fixture
async def fake_llm(tmp_path):
    """An :class:`LLMCore` wired to an injected :class:`FakeProvider`.

    Vector storage is disabled and session storage points at a temp dir, so
    construction and ``chat()`` run entirely offline.
    """
    llm = await LLMCore.create(
        config_overrides={
            "storage": {
                "vector": {"type": ""},
                "session": {"type": "json", "path": str(tmp_path / "sessions")},
            }
        }
    )
    fake = FakeProvider()
    # Inject as a resolvable, default provider (manager keys are lower-cased).
    llm._provider_manager._providers["fake"] = fake
    llm._provider_manager._default_provider_name = "fake"
    try:
        yield llm, fake
    finally:
        await llm.close()


# ---------------------------------------------------------------------------
# ChatUsage value object
# ---------------------------------------------------------------------------


class TestChatUsageValue:
    def test_in_out_aliases(self):
        u = ChatUsage(prompt_tokens=1200, completion_tokens=400, total_tokens=1600)
        assert u.tokens_in == 1200
        assert u.tokens_out == 400
        assert u.is_available is True

    def test_empty_is_unavailable(self):
        u = ChatUsage()
        assert u.tokens_in is None
        assert u.tokens_out is None
        assert u.total_tokens is None
        assert u.provider is None
        assert u.model is None
        assert u.is_available is False

    def test_zero_completion_is_still_available(self):
        # 0 is a real count (empty response), not "unavailable".
        u = ChatUsage(prompt_tokens=10, completion_tokens=0)
        assert u.is_available is True
        assert u.tokens_out == 0

    def test_from_context_details_maps_fields(self):
        d = ContextPreparationDetails(final_token_count=11)
        d.provider = "openai"
        d.model = "gpt-4o"
        d.prompt_tokens = 11
        d.completion_tokens = 7
        d.total_tokens = 18
        u = ChatUsage.from_context_details(d)
        assert (u.tokens_in, u.tokens_out, u.total_tokens) == (11, 7, 18)
        assert u.provider == "openai"
        assert u.model == "gpt-4o"

    def test_from_none_is_empty(self):
        assert ChatUsage.from_context_details(None) == ChatUsage()

    def test_frozen(self):
        u = ChatUsage(prompt_tokens=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            u.prompt_tokens = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Signature / protocol contract (the cross-repo metering contract, gate #10)
# ---------------------------------------------------------------------------


class TestUsageSurfaceContract:
    def test_chat_with_usage_is_async_coroutine(self):
        import inspect

        assert inspect.iscoroutinefunction(LLMCore.chat_with_usage)

    def test_signature_covers_bridge_kwargs(self):
        import inspect

        params = set(inspect.signature(LLMCore.chat_with_usage).parameters)
        # The exact keyword set Convergence's bridge passes (Part 3 §3.2.3).
        required = {
            "message",
            "system_message",
            "provider_name",
            "model_name",
            "enable_rag",
            "save_session",
        }
        assert required <= params, required - params

    def test_chat_str_contract_unchanged(self):
        # chat() still advertises a non-tuple return; we did not break callers.
        import inspect

        ret = inspect.signature(LLMCore.chat).return_annotation
        assert "tuple" not in str(ret)


# ---------------------------------------------------------------------------
# End-to-end behaviour (offline)
# ---------------------------------------------------------------------------


class TestChatWithUsageBehaviour:
    async def test_returns_text_and_usage(self, fake_llm):
        llm, fake = fake_llm
        text, usage = await llm.chat_with_usage(
            message="Count these words please",
            provider_name="fake",
            save_session=False,
        )
        assert text == _FAKE_REPLY
        assert isinstance(usage, ChatUsage)
        assert usage.is_available
        assert usage.tokens_in is not None and usage.tokens_in > 0
        # Completion is the canned reply: 5 whitespace tokens.
        assert usage.tokens_out == len(_FAKE_REPLY.split())
        assert usage.total_tokens == usage.tokens_in + usage.tokens_out
        assert usage.provider == "fake"
        assert usage.model == "fake-model-1"
        assert fake.chat_completion_calls == 1

    async def test_default_provider_used_when_none(self, fake_llm):
        llm, _ = fake_llm  # the fake is the configured default provider
        text, usage = await llm.chat_with_usage(message="hi", save_session=False)
        assert text == _FAKE_REPLY
        assert usage.provider == "fake"

    async def test_ephemeral_session_leaves_no_residue(self, fake_llm):
        llm, _ = fake_llm
        before_info = dict(llm._transient_last_interaction_info_cache)
        before_raw = dict(llm._transient_last_raw_response_cache)
        before_sess = dict(llm._transient_sessions_cache)
        await llm.chat_with_usage(message="hello world", save_session=False)
        # No new ephemeral keys remain in any transient cache.
        assert llm._transient_last_interaction_info_cache == before_info
        assert llm._transient_last_raw_response_cache == before_raw
        assert llm._transient_sessions_cache == before_sess

    async def test_caller_session_id_is_preserved(self, fake_llm):
        llm, _ = fake_llm
        sid = "real-session-xyz"
        _, usage = await llm.chat_with_usage(
            message="remember me", session_id=sid, save_session=False
        )
        assert usage.is_available
        # Caller-owned session id is NOT cleaned up: introspection still works.
        details = llm.get_last_interaction_context_info(sid)
        assert details is not None
        assert details.prompt_tokens == usage.tokens_in

    async def test_stream_kwarg_is_rejected(self, fake_llm):
        llm, _ = fake_llm
        with pytest.raises(ValueError):
            await llm.chat_with_usage(message="x", stream=True)  # type: ignore[call-arg]

    async def test_unsupported_provider_kwarg_rejected(self, fake_llm):
        # Same validation as chat(): unknown provider kwargs raise ValueError.
        llm, _ = fake_llm
        with pytest.raises(ValueError):
            await llm.chat_with_usage(
                message="x", provider_name="fake", temperature=0.5
            )

    async def test_usage_duck_types_for_convergence_bridge(self, fake_llm):
        # Convergence's bridge reads .tokens_in / .tokens_out off the object;
        # assert that drop-in path holds end-to-end.
        llm, _ = fake_llm
        _, usage = await llm.chat_with_usage(message="abc def", save_session=False)
        assert hasattr(usage, "tokens_in") and hasattr(usage, "tokens_out")
        assert isinstance(usage.tokens_in, int) and isinstance(usage.tokens_out, int)
