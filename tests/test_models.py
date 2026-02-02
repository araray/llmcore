# tests/test_models.py
"""
Comprehensive tests for the llmcore.models module.

Tests all core data models including Message, Role, ContextItem,
Episode, ChatSession, and other fundamental types.
"""

import uuid

import pytest

from llmcore.models import (
    AgentState,
    ChatSession,
    ContextItem,
    ContextItemType,
    ContextPreset,
    Episode,
    EpisodeType,
    Message,
    Role,
    Tool,
    ToolCall,
    ToolResult,
)


class TestRole:
    """Tests for the Role enum."""

    def test_role_values(self):
        """Test that all expected roles exist."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_case_insensitive(self):
        """Test case-insensitive role matching."""
        assert Role("SYSTEM") == Role.SYSTEM
        assert Role("System") == Role.SYSTEM
        assert Role("system") == Role.SYSTEM

    def test_role_agent_alias(self):
        """Test that 'agent' maps to ASSISTANT."""
        assert Role("agent") == Role.ASSISTANT


class TestMessage:
    """Tests for the Message model."""

    def test_create_basic_message(self):
        """Test creating a basic message."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.id is not None

    def test_message_auto_id(self):
        """Test that message gets auto-generated ID."""
        msg = Message(role=Role.USER, content="Test")
        assert msg.id is not None
        uuid.UUID(msg.id)  # Should be valid UUID

    def test_message_timestamp(self):
        """Test that timestamp is set."""
        msg = Message(role=Role.USER, content="Test")
        assert msg.timestamp is not None

    def test_message_with_tokens(self):
        """Test message with token count."""
        msg = Message(role=Role.USER, content="Test", tokens=42)
        assert msg.tokens == 42

    def test_message_with_metadata(self):
        """Test message with metadata."""
        meta = {"source": "api", "version": 2}
        msg = Message(role=Role.USER, content="Test", metadata=meta)
        assert msg.metadata == meta

    def test_message_serialization(self):
        """Test message serialization."""
        msg = Message(role=Role.USER, content="Test")
        data = msg.model_dump()
        assert "role" in data
        assert "content" in data


class TestContextItemType:
    """Tests for ContextItemType enum."""

    def test_all_types_exist(self):
        """Test that all expected types exist."""
        assert ContextItemType.HISTORY_MESSAGE is not None
        assert ContextItemType.USER_TEXT is not None
        assert ContextItemType.USER_FILE is not None
        assert ContextItemType.RAG_SNIPPET is not None

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert ContextItemType("USER_TEXT") == ContextItemType.USER_TEXT


class TestContextItem:
    """Tests for ContextItem model."""

    def test_create_basic_item(self):
        """Test creating a basic context item."""
        item = ContextItem(type=ContextItemType.USER_TEXT, content="Some text")
        assert item.type == "user_text"
        assert item.content == "Some text"

    def test_item_with_tokens(self):
        """Test item with token counts."""
        item = ContextItem(
            type=ContextItemType.RAG_SNIPPET,
            content="Text",
            tokens=100,
            original_tokens=150,
            is_truncated=True
        )
        assert item.tokens == 100
        assert item.original_tokens == 150


class TestEpisodeType:
    """Tests for EpisodeType enum."""

    def test_all_types_exist(self):
        """Test that all episode types exist."""
        assert EpisodeType.THOUGHT is not None
        assert EpisodeType.ACTION is not None
        assert EpisodeType.OBSERVATION is not None


class TestEpisode:
    """Tests for Episode model."""

    def test_create_basic_episode(self):
        """Test creating a basic episode."""
        ep = Episode(
            session_id="session_123",
            event_type=EpisodeType.THOUGHT,
            data={"content": "Thinking"}
        )
        assert ep.session_id == "session_123"
        assert ep.event_type == "thought"


class TestChatSession:
    """Tests for ChatSession model."""

    def test_create_basic_session(self):
        """Test creating a basic chat session."""
        session = ChatSession(name="Test Session")
        assert session.name == "Test Session"
        assert session.id is not None
        assert session.messages == []

    def test_session_with_messages(self):
        """Test session with messages."""
        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.ASSISTANT, content="Hi there!")
        session = ChatSession(name="Test", messages=[msg1, msg2])
        assert len(session.messages) == 2


class TestTool:
    """Tests for Tool model."""

    def test_create_basic_tool(self):
        """Test creating a basic tool definition."""
        tool = Tool(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}}
        )
        assert tool.name == "search"


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        """Test creating a tool call request."""
        call = ToolCall(id="call_123", name="search", arguments={"query": "hello"})
        assert call.id == "call_123"
        assert call.name == "search"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_create_successful_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(tool_call_id="call_123", content="Search results")
        assert result.tool_call_id == "call_123"


class TestAgentState:
    """Tests for AgentState model."""

    def test_create_basic_state(self):
        """Test creating a basic agent state."""
        state = AgentState(goal="Find information")
        assert state.goal == "Find information"
        assert state.plan == []


class TestContextPreset:
    """Tests for ContextPreset model."""

    def test_create_basic_preset(self):
        """Test creating a basic preset."""
        preset = ContextPreset(name="Default Preset", description="A default preset")
        assert preset.name == "Default Preset"


class TestModelSerialization:
    """Tests for model serialization/deserialization."""

    def test_message_round_trip(self):
        """Test message serialization round trip."""
        original = Message(role=Role.USER, content="Test message", tokens=42)
        data = original.model_dump()
        restored = Message.model_validate(data)
        assert restored.content == original.content
        assert restored.tokens == original.tokens


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content(self):
        """Test message with empty content."""
        msg = Message(role=Role.USER, content="")
        assert msg.content == ""

    def test_unicode_content(self):
        """Test message with Unicode content."""
        content = "Hello 世界 🌍 مرحبا"
        msg = Message(role=Role.USER, content=content)
        assert msg.content == content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
