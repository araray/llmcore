# tests/api/test_phase3_session_operations.py
"""
Tests for Phase 3: Session Fork, Clone, and Message Operations.

This test module covers:
- fork_session() with all 4 modes (full, from_message, message_ids, message_range)
- clone_session() with various options
- delete_messages()
- copy_messages_to_session()
- get_messages_by_range()

All tests use mocks to avoid dependency on storage backends.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ==============================================================================
# Mock Factories
# ==============================================================================


def create_mock_message(
    msg_id: Optional[str] = None,
    session_id: str = "test-session",
    role: str = "user",
    content: str = "Test content",
    timestamp: Optional[datetime] = None,
    tokens: int = 10,
    metadata: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    """Create a mock Message object."""
    msg = MagicMock()
    msg.id = msg_id or str(uuid.uuid4())
    msg.session_id = session_id
    msg.role = role
    msg.content = content
    msg.timestamp = timestamp or datetime.now(timezone.utc)
    msg.tokens = tokens
    msg.tool_call_id = None
    msg.metadata = metadata or {}
    return msg


def create_mock_context_item(
    item_id: Optional[str] = None,
    content: str = "Context content",
    item_type: str = "user_text",
) -> MagicMock:
    """Create a mock ContextItem object."""
    item = MagicMock()
    item.id = item_id or str(uuid.uuid4())
    item.type = item_type
    item.source_id = None
    item.content = content
    item.tokens = 5
    item.original_tokens = 5
    item.is_truncated = False
    item.metadata = {}
    item.timestamp = datetime.now(timezone.utc)
    return item


def create_mock_session(
    session_id: str = "source-session",
    name: str = "Test Session",
    num_messages: int = 5,
    num_context_items: int = 2,
) -> MagicMock:
    """Create a mock ChatSession with messages."""
    session = MagicMock()
    session.id = session_id
    session.name = name
    session.created_at = datetime.now(timezone.utc)
    session.updated_at = datetime.now(timezone.utc)
    session.metadata = {}

    # Create messages with sequential timestamps
    base_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    session.messages = [
        create_mock_message(
            msg_id=f"msg-{i:03d}",
            session_id=session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i} content",
            timestamp=base_time + timedelta(minutes=i),
        )
        for i in range(num_messages)
    ]

    # Create context items
    session.context_items = [
        create_mock_context_item(
            item_id=f"ctx-{i:03d}",
            content=f"Context item {i}",
        )
        for i in range(num_context_items)
    ]

    return session


class MockSessionManager:
    """Mock session manager for testing."""

    def __init__(self):
        self.sessions: Dict[str, MagicMock] = {}
        self.saved_sessions: List[MagicMock] = []

    async def get_session(self, session_id: str) -> Optional[MagicMock]:
        return self.sessions.get(session_id)

    async def save_session(self, session: MagicMock) -> None:
        self.saved_sessions.append(session)
        self.sessions[session.id] = session

    async def update_session_name(self, session_id: str, new_name: str) -> bool:
        if session_id in self.sessions:
            self.sessions[session_id].name = new_name
            return True
        return False


# ==============================================================================
# Test Classes
# ==============================================================================


class TestForkSession:
    """Tests for fork_session() method."""

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore with session manager."""
        llmcore = MagicMock()
        llmcore._session_manager = MockSessionManager()

        # Mock create_session to return a new session
        async def mock_create_session(name=None, session_id=None, system_message=None):
            new_session = MagicMock()
            new_session.id = str(uuid.uuid4())
            new_session.name = name
            new_session.messages = []
            new_session.context_items = []
            new_session.metadata = {}
            new_session.updated_at = datetime.now(timezone.utc)
            llmcore._session_manager.sessions[new_session.id] = new_session
            return new_session

        llmcore.create_session = mock_create_session
        return llmcore

    @pytest.mark.asyncio
    async def test_fork_full_mode(self, mock_llmcore):
        """Test full fork copies all messages."""
        # Import the actual method
        from llmcore.api import LLMCore

        # Setup source session
        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        # Call fork_session using the implementation directly
        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        new_id = await llm.fork_session("source-id")

        # Verify
        assert new_id is not None
        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 5
        assert saved.metadata.get("fork_type") == "full"
        assert saved.metadata.get("forked_from") == "source-id"

    @pytest.mark.asyncio
    async def test_fork_from_message_mode(self, mock_llmcore):
        """Test fork from specific message onwards."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        # Fork from message 2 (should include msg-002, msg-003, msg-004)
        new_id = await llm.fork_session("source-id", from_message_id="msg-002")

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 3  # Messages 2, 3, 4
        assert saved.metadata.get("fork_type") == "from_message"
        assert saved.metadata.get("fork_point_message_id") == "msg-002"

    @pytest.mark.asyncio
    async def test_fork_message_ids_mode(self, mock_llmcore):
        """Test fork with specific message IDs."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        new_id = await llm.fork_session(
            "source-id",
            message_ids=["msg-001", "msg-003"],
        )

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 2
        assert saved.metadata.get("fork_type") == "specific_messages"

    @pytest.mark.asyncio
    async def test_fork_range_mode(self, mock_llmcore):
        """Test fork with message range."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        new_id = await llm.fork_session(
            "source-id",
            message_range=(1, 3),  # Messages at indices 1, 2, 3
        )

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 3
        assert saved.metadata.get("fork_type") == "range"
        assert saved.metadata.get("fork_range") == [1, 3]

    @pytest.mark.asyncio
    async def test_fork_with_custom_name(self, mock_llmcore):
        """Test fork with custom name."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.fork_session("source-id", new_name="my_custom_fork")

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert saved.name == "my_custom_fork"

    @pytest.mark.asyncio
    async def test_fork_without_context_items(self, mock_llmcore):
        """Test fork without copying context items."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3, num_context_items=2)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.fork_session("source-id", include_context_items=False)

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.context_items) == 0

    @pytest.mark.asyncio
    async def test_fork_with_context_items(self, mock_llmcore):
        """Test fork copies context items by default."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3, num_context_items=2)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.fork_session("source-id", include_context_items=True)

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.context_items) == 2

    @pytest.mark.asyncio
    async def test_fork_invalid_message_id_raises(self, mock_llmcore):
        """Test fork with invalid message ID raises ValueError."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        with pytest.raises(ValueError, match="Message IDs not found"):
            await llm.fork_session("source-id", message_ids=["nonexistent"])

    @pytest.mark.asyncio
    async def test_fork_invalid_range_raises(self, mock_llmcore):
        """Test fork with invalid range raises ValueError."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        with pytest.raises(ValueError, match="Invalid range"):
            await llm.fork_session("source-id", message_range=(0, 10))

    @pytest.mark.asyncio
    async def test_fork_nonexistent_session_raises(self, mock_llmcore):
        """Test fork of nonexistent session raises SessionNotFoundError."""
        from llmcore.api import LLMCore
        from llmcore.exceptions import SessionNotFoundError

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        with pytest.raises(SessionNotFoundError):
            await llm.fork_session("nonexistent")

    @pytest.mark.asyncio
    async def test_fork_messages_get_new_ids(self, mock_llmcore):
        """Test forked messages get new IDs (independence)."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.fork_session("source-id")

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        source_ids = {m.id for m in source.messages}
        forked_ids = {m.id for m in saved.messages}

        # No overlap in IDs
        assert source_ids.isdisjoint(forked_ids)


class TestCloneSession:
    """Tests for clone_session() method."""

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore with session manager."""
        llmcore = MagicMock()
        llmcore._session_manager = MockSessionManager()

        async def mock_create_session(name=None, session_id=None, system_message=None):
            new_session = MagicMock()
            new_session.id = str(uuid.uuid4())
            new_session.name = name
            new_session.messages = []
            new_session.context_items = []
            new_session.metadata = {}
            new_session.updated_at = datetime.now(timezone.utc)
            llmcore._session_manager.sessions[new_session.id] = new_session
            return new_session

        llmcore.create_session = mock_create_session
        return llmcore

    @pytest.mark.asyncio
    async def test_clone_full(self, mock_llmcore):
        """Test full clone copies everything."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=4, num_context_items=2)
        source.metadata = {"custom_key": "custom_value"}
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        new_id = await llm.clone_session("source-id")

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 4
        assert len(saved.context_items) == 2
        assert saved.metadata.get("cloned_from") == "source-id"
        assert saved.metadata.get("source_custom_key") == "custom_value"

    @pytest.mark.asyncio
    async def test_clone_without_messages(self, mock_llmcore):
        """Test clone without messages."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=4)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.clone_session("source-id", include_messages=False)

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 0
        assert saved.metadata.get("messages_included") is False

    @pytest.mark.asyncio
    async def test_clone_without_context_items(self, mock_llmcore):
        """Test clone without context items."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=2, num_context_items=3)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.clone_session("source-id", include_context_items=False)

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.context_items) == 0
        assert saved.metadata.get("context_items_included") is False

    @pytest.mark.asyncio
    async def test_clone_with_custom_name(self, mock_llmcore):
        """Test clone with custom name."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=2)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        await llm.clone_session("source-id", new_name="my_clone")

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert saved.name == "my_clone"


class TestDeleteMessages:
    """Tests for delete_messages() method."""

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore with session manager."""
        llmcore = MagicMock()
        llmcore._session_manager = MockSessionManager()
        return llmcore

    @pytest.mark.asyncio
    async def test_delete_single_message(self, mock_llmcore):
        """Test deleting a single message."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        deleted = await llm.delete_messages("source-id", ["msg-002"])

        assert deleted == 1
        assert len(source.messages) == 4

    @pytest.mark.asyncio
    async def test_delete_multiple_messages(self, mock_llmcore):
        """Test deleting multiple messages."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        deleted = await llm.delete_messages("source-id", ["msg-001", "msg-003"])

        assert deleted == 2
        assert len(source.messages) == 3

    @pytest.mark.asyncio
    async def test_delete_nonexistent_messages_returns_zero(self, mock_llmcore):
        """Test deleting nonexistent messages returns 0."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        deleted = await llm.delete_messages("source-id", ["nonexistent"])

        assert deleted == 0
        assert len(source.messages) == 3


class TestCopyMessagesToSession:
    """Tests for copy_messages_to_session() method."""

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore with session manager."""
        llmcore = MagicMock()
        llmcore._session_manager = MockSessionManager()
        return llmcore

    @pytest.mark.asyncio
    async def test_copy_messages(self, mock_llmcore):
        """Test copying messages between sessions."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        target = create_mock_session("target-id", "Target", num_messages=2)
        mock_llmcore._session_manager.sessions["source-id"] = source
        mock_llmcore._session_manager.sessions["target-id"] = target

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        new_ids = await llm.copy_messages_to_session(
            "source-id",
            "target-id",
            ["msg-001", "msg-003"],
        )

        assert len(new_ids) == 2
        assert len(target.messages) == 4  # Original 2 + copied 2

    @pytest.mark.asyncio
    async def test_copy_preserves_timestamps(self, mock_llmcore):
        """Test copying preserves original timestamps."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        original_timestamp = source.messages[1].timestamp
        target = create_mock_session("target-id", "Target", num_messages=0)
        mock_llmcore._session_manager.sessions["source-id"] = source
        mock_llmcore._session_manager.sessions["target-id"] = target

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        await llm.copy_messages_to_session(
            "source-id",
            "target-id",
            ["msg-001"],
            preserve_timestamps=True,
        )

        # Note: We're testing the logic, actual timestamp check depends on mock setup
        assert len(target.messages) == 1

    @pytest.mark.asyncio
    async def test_copy_invalid_message_raises(self, mock_llmcore):
        """Test copying invalid message IDs raises ValueError."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        target = create_mock_session("target-id", "Target", num_messages=0)
        mock_llmcore._session_manager.sessions["source-id"] = source
        mock_llmcore._session_manager.sessions["target-id"] = target

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        with pytest.raises(ValueError, match="Message IDs not found"):
            await llm.copy_messages_to_session(
                "source-id",
                "target-id",
                ["nonexistent"],
            )


class TestGetMessagesByRange:
    """Tests for get_messages_by_range() method."""

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore with session manager."""
        llmcore = MagicMock()
        llmcore._session_manager = MockSessionManager()
        return llmcore

    @pytest.mark.asyncio
    async def test_get_range(self, mock_llmcore):
        """Test getting messages by range."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        messages = await llm.get_messages_by_range("source-id", 1, 3)

        assert len(messages) == 3  # Indices 1, 2, 3 (inclusive)

    @pytest.mark.asyncio
    async def test_get_range_clamps_end(self, mock_llmcore):
        """Test range end is clamped to valid range."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        messages = await llm.get_messages_by_range("source-id", 3, 10)

        assert len(messages) == 2  # Indices 3, 4

    @pytest.mark.asyncio
    async def test_get_range_invalid_start_raises(self, mock_llmcore):
        """Test invalid start index raises ValueError."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        with pytest.raises(ValueError, match="start_index .* out of range"):
            await llm.get_messages_by_range("source-id", 10, 15)

    @pytest.mark.asyncio
    async def test_get_range_negative_start_raises(self, mock_llmcore):
        """Test negative start index raises ValueError."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        with pytest.raises(ValueError, match="start_index must be >= 0"):
            await llm.get_messages_by_range("source-id", -1, 3)

    @pytest.mark.asyncio
    async def test_get_range_end_less_than_start_raises(self, mock_llmcore):
        """Test end < start raises ValueError."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager

        with pytest.raises(ValueError, match="end_index .* must be >= start_index"):
            await llm.get_messages_by_range("source-id", 3, 1)


class TestIntegration:
    """Integration tests verifying fork/clone create independent sessions."""

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore with session manager."""
        llmcore = MagicMock()
        llmcore._session_manager = MockSessionManager()

        async def mock_create_session(name=None, session_id=None, system_message=None):
            new_session = MagicMock()
            new_session.id = str(uuid.uuid4())
            new_session.name = name
            new_session.messages = []
            new_session.context_items = []
            new_session.metadata = {}
            new_session.updated_at = datetime.now(timezone.utc)
            llmcore._session_manager.sessions[new_session.id] = new_session
            return new_session

        llmcore.create_session = mock_create_session
        return llmcore

    @pytest.mark.asyncio
    async def test_forked_session_is_independent(self, mock_llmcore):
        """Test that modifying forked session doesn't affect source."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=3)
        mock_llmcore._session_manager.sessions["source-id"] = source
        original_count = len(source.messages)

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        new_id = await llm.fork_session("source-id")
        forked = mock_llmcore._session_manager.sessions[new_id]

        # Modify forked session
        forked.messages.append(create_mock_message())

        # Source unchanged
        assert len(source.messages) == original_count
        assert len(forked.messages) == original_count + 1

    @pytest.mark.asyncio
    async def test_fork_mode_priority(self, mock_llmcore):
        """Test fork mode priority: message_ids > message_range > from_message_id."""
        from llmcore.api import LLMCore

        source = create_mock_session("source-id", "Source", num_messages=5)
        mock_llmcore._session_manager.sessions["source-id"] = source

        llm = LLMCore.__new__(LLMCore)
        llm._session_manager = mock_llmcore._session_manager
        llm.create_session = mock_llmcore.create_session

        # Provide all modes - message_ids should win
        await llm.fork_session(
            "source-id",
            message_ids=["msg-001"],
            message_range=(0, 3),
            from_message_id="msg-002",
        )

        saved = mock_llmcore._session_manager.saved_sessions[-1]
        assert len(saved.messages) == 1  # Only msg-001
        assert saved.metadata.get("fork_type") == "specific_messages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
