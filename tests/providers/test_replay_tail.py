# tests/providers/test_replay_tail.py
"""carry_replay_to_tail: replay data stays on the last chunk of a stream."""

from llmcore.providers.base import carry_replay_to_tail


def test_trailing_usage_chunk_gets_the_replay():
    state: dict = {}
    finish = {"choices": [{"index": 0, "delta": {"content": "", "replay_reasoning_content": "r"},
                           "finish_reason": "stop"}]}
    usage = {"choices": [], "usage": {"total_tokens": 3}}
    carry_replay_to_tail({"choices": [{"index": 0, "delta": {"content": "a"}}]}, state)
    carry_replay_to_tail(finish, state)
    carry_replay_to_tail(usage, state)
    assert usage["choices"] == [{"index": 0, "delta": {"replay_reasoning_content": "r"}, "finish_reason": None}]


def test_nothing_to_carry_leaves_chunks_alone():
    state: dict = {}
    usage = {"choices": [], "usage": {}}
    carry_replay_to_tail(usage, state)
    assert usage["choices"] == []
