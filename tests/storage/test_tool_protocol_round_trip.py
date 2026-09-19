# tests/storage/test_tool_protocol_round_trip.py
"""Persistence round-trip tests for the Message tool-protocol fields (R-2).

The SQLite and PostgreSQL message tables predate ``tool_call_id`` /
``tool_calls``; the fields are folded into the metadata JSON on save and
popped back out on load. These tests cover the fold/unfold helpers and an
end-to-end SQLite round trip (JSON storage round-trips via model_dump_json
and needs no folding).
"""

from __future__ import annotations

import pytest

from llmcore.models import ChatSession, Message, Role
from llmcore.storage.base_session import (
    TOOL_CALL_ID_METADATA_KEY,
    TOOL_CALLS_METADATA_KEY,
    embed_tool_fields_in_metadata,
    extract_tool_fields_from_metadata,
)

_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
    }
]


class TestEmbedExtractHelpers:
    def test_plain_message_metadata_unchanged(self):
        msg = Message(role=Role.USER, content="Hi", metadata={"provider": "openai"})
        metadata = embed_tool_fields_in_metadata(msg)
        assert metadata == {"provider": "openai"}
        assert TOOL_CALL_ID_METADATA_KEY not in metadata
        assert TOOL_CALLS_METADATA_KEY not in metadata

    def test_embed_does_not_mutate_message(self):
        msg = Message(role=Role.TOOL, content="ok", tool_call_id="call_1")
        embed_tool_fields_in_metadata(msg)
        assert msg.metadata == {}

    def test_round_trip_via_metadata(self):
        assistant = Message(role=Role.ASSISTANT, content="", tool_calls=_CALLS)
        tool_msg = Message(role=Role.TOOL, content='{"temp": 72}', tool_call_id="call_1")

        for original in (assistant, tool_msg):
            dumped = original.model_dump()
            dumped["metadata"] = embed_tool_fields_in_metadata(original)
            dumped.pop("tool_call_id")
            dumped.pop("tool_calls")
            extract_tool_fields_from_metadata(dumped)
            restored = Message.model_validate(dumped)
            assert restored.tool_call_id == original.tool_call_id
            assert restored.tool_calls == original.tool_calls
            assert restored.metadata == original.metadata

    def test_embed_tool_call_id_optional(self):
        msg = Message(role=Role.TOOL, content="ok", tool_call_id="call_1", tool_calls=None)
        metadata = embed_tool_fields_in_metadata(msg, embed_tool_call_id=False)
        assert TOOL_CALL_ID_METADATA_KEY not in metadata

    def test_extract_never_overwrites_column_value(self):
        msg_dict = {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "from_column",
            "metadata": {TOOL_CALL_ID_METADATA_KEY: "from_metadata"},
        }
        extract_tool_fields_from_metadata(msg_dict)
        assert msg_dict["tool_call_id"] == "from_column"
        assert TOOL_CALL_ID_METADATA_KEY not in msg_dict["metadata"]


@pytest.mark.asyncio
async def test_sqlite_session_round_trips_tool_protocol_fields(tmp_path):
    """A full save/load cycle preserves tool_call_id and tool_calls."""
    aiosqlite = pytest.importorskip("aiosqlite")  # noqa: F841

    from llmcore.storage.sqlite_session import SqliteSessionStorage

    storage = SqliteSessionStorage()
    await storage.initialize({"path": str(tmp_path / "sessions.db")})
    try:
        session = ChatSession(id="tool-proto-session")
        session.messages = [
            Message(
                role=Role.USER, content="Weather in NYC?", session_id=session.id, metadata={"k": 1}
            ),
            Message(role=Role.ASSISTANT, content="", session_id=session.id, tool_calls=_CALLS),
            Message(
                role=Role.TOOL,
                content='{"temp": 72}',
                session_id=session.id,
                tool_call_id="call_1",
            ),
            Message(role=Role.ASSISTANT, content="It is 72F.", session_id=session.id),
        ]
        await storage.save_session(session)

        loaded = await storage.get_session(session.id)
        assert loaded is not None
        assert [m.role for m in loaded.messages] == ["user", "assistant", "tool", "assistant"]
        assert loaded.messages[0].metadata == {"k": 1}
        assert loaded.messages[1].tool_calls == _CALLS
        assert loaded.messages[1].metadata == {}
        assert loaded.messages[2].tool_call_id == "call_1"
        assert loaded.messages[2].metadata == {}
        assert loaded.messages[3].tool_calls is None
    finally:
        await storage.close()
