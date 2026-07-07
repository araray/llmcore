"""Regression tests locking the R-2 first-class ``Message.tool_calls`` mapping.

The OpenAI-compatible providers (deepseek/mistral/kimi/zai) were fixed to honor
a first-class ``Message.tool_calls`` field on an assistant message, not only the
legacy ``metadata["tool_calls"]`` channel. Without this, a caller using the
native tool-role protocol emits a ``role="tool"`` result whose preceding
assistant turn carries its ``tool_calls`` in the first-class field; if the
provider ignored that field, the wire payload dropped ``tool_calls`` and the
API rejected the request with a 400 ("agentic tool use 400s" bug).

These tests exercise each provider's ``_build_message_payload`` and assert:

- a first-class ``Message(role="assistant", tool_calls=[...])`` is mapped to
  ``msg_dict["tool_calls"]``;
- a ``Message(role="tool", tool_call_id=...)`` is mapped to
  ``msg_dict["tool_call_id"]``;
- the legacy ``metadata["tool_calls"]`` channel STILL maps (back-compat);
- the first-class field takes precedence over the legacy channel.

Hermetic: provider clients are patched out at construction; no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from llmcore.models import Message, Role

# Canonical OpenAI-normalized tool_calls shape.
_TOOL_CALLS: list[dict[str, Any]] = [
    {
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "list_dir", "arguments": '{"path": "."}'},
    }
]
_TOOL_CALL_ID = "call_abc123"


# ---------------------------------------------------------------------------
# Provider factories — each constructs the real provider with its client(s)
# patched out so construction is network-free. ``_build_message_payload`` does
# not touch the client, so a mocked client is sufficient.
# ---------------------------------------------------------------------------
def _make_openai():
    with (
        patch("llmcore.providers.openai_provider.AsyncOpenAI"),
        patch("llmcore.providers.openai_provider.tiktoken"),
    ):
        from llmcore.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            {"api_key": "sk-test-000", "default_model": "gpt-4o"},
            log_raw_payloads=False,
        )


def _make_deepseek():
    with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
        from llmcore.providers.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(
            {"api_key": "sk-test-000", "default_model": "deepseek-chat"},
            log_raw_payloads=False,
        )


def _make_mistral():
    with patch("llmcore.providers.mistral_provider.httpx"):
        from llmcore.providers.mistral_provider import MistralProvider

        return MistralProvider(
            {"api_key": "sk-test-000", "default_model": "mistral-large-latest"},
            log_raw_payloads=False,
        )


def _make_kimi():
    with (
        patch("llmcore.providers.kimi_provider.AsyncOpenAI"),
        patch("llmcore.providers.kimi_provider.httpx"),
    ):
        from llmcore.providers.kimi_provider import KimiProvider

        return KimiProvider(
            {"api_key": "sk-test-000", "default_model": "kimi-k2.6"},
            log_raw_payloads=False,
        )


def _make_zai():
    with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
        from llmcore.providers.zai_provider import ZaiProvider

        # Force the openai-compat backend so only AsyncOpenAI needs patching.
        return ZaiProvider(
            {"api_key": "test-000", "default_model": "glm-4.6", "backend": "openai"},
            log_raw_payloads=False,
        )


# (factory, model_name-or-None). openai/mistral take a ``model_name`` positional
# argument on ``_build_message_payload``; the others do not.
_PROVIDERS = [
    pytest.param(_make_openai, "gpt-4o", id="openai"),
    pytest.param(_make_deepseek, None, id="deepseek"),
    pytest.param(_make_mistral, "mistral-large-latest", id="mistral"),
    pytest.param(_make_kimi, None, id="kimi"),
    pytest.param(_make_zai, None, id="zai"),
]


def _build(provider: Any, msg: Message, model_name: str | None) -> dict[str, Any]:
    """Call the provider's payload builder, bridging the signature difference."""
    if model_name is None:
        return provider._build_message_payload(msg)
    return provider._build_message_payload(msg, model_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory, model_name", _PROVIDERS)
def test_first_class_tool_calls_on_assistant_are_mapped(factory, model_name):
    """A first-class ``Message.tool_calls`` (NOT metadata) reaches the wire."""
    provider = factory()
    msg = Message(role=Role.ASSISTANT, content="", tool_calls=_TOOL_CALLS)

    payload = _build(provider, msg, model_name)

    assert payload["role"] == "assistant"
    assert payload["tool_calls"] == _TOOL_CALLS


@pytest.mark.parametrize("factory, model_name", _PROVIDERS)
def test_tool_role_message_carries_tool_call_id(factory, model_name):
    """A ``role="tool"`` result maps ``tool_call_id`` onto the wire dict."""
    provider = factory()
    msg = Message(role=Role.TOOL, content="dir listing", tool_call_id=_TOOL_CALL_ID)

    payload = _build(provider, msg, model_name)

    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == _TOOL_CALL_ID


@pytest.mark.parametrize("factory, model_name", _PROVIDERS)
def test_legacy_metadata_tool_calls_still_mapped(factory, model_name):
    """Back-compat: the legacy ``metadata["tool_calls"]`` channel still maps."""
    provider = factory()
    msg = Message(
        role=Role.ASSISTANT,
        content="",
        metadata={"tool_calls": _TOOL_CALLS},
    )

    payload = _build(provider, msg, model_name)

    assert payload["role"] == "assistant"
    assert payload["tool_calls"] == _TOOL_CALLS


@pytest.mark.parametrize("factory, model_name", _PROVIDERS)
def test_first_class_tool_calls_take_precedence_over_metadata(factory, model_name):
    """When both channels are set, the first-class field wins."""
    provider = factory()
    legacy = [
        {
            "id": "call_legacy",
            "type": "function",
            "function": {"name": "other_tool", "arguments": "{}"},
        }
    ]
    msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=_TOOL_CALLS,
        metadata={"tool_calls": legacy},
    )

    payload = _build(provider, msg, model_name)

    assert payload["tool_calls"] == _TOOL_CALLS
