# tests/providers/test_responses_shared.py
"""The shared Responses wire module and the replay contract on OpenAI."""

from __future__ import annotations

from llmcore.models import Message, Role
from llmcore.providers import openai_responses as rsp


def test_extra_body_is_merged_into_the_wire_body():
    # A raw HTTP body has no "extra_body" field. Posted as-is, servers that
    # ignore unknown fields (vLLM) dropped the vendor parameters inside it.
    body = rsp.build_responses_request(
        "m", [Message(role=Role.USER, content="hi")],
        extra_body={"structured_outputs": {"choice": ["a", "b"]}}, top_k=5,
    )
    assert "extra_body" not in body
    assert body["structured_outputs"] == {"choice": ["a", "b"]}
    assert body["top_k"] == 5


def test_openai_replays_reasoning_items_from_message_and_stream():
    import pytest

    pytest.importorskip("openai")
    from llmcore.providers.openai_provider import OpenAIProvider

    p = OpenAIProvider({"api_key": "k"})
    item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}
    message = {"choices": [{"message": {"content": "", "reasoning_items": [item]}}]}
    final_chunk = {"choices": [{"delta": {"reasoning_items": [item]}, "finish_reason": "stop"}]}
    assert p.assistant_replay_metadata(message) == {"reasoning_items": [item]}
    assert p.assistant_replay_metadata(final_chunk) == {"reasoning_items": [item]}
    assert p.assistant_replay_metadata({"choices": [{"message": {"content": "hi"}}]}) == {}
