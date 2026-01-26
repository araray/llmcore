# tests/storage/test_postgres_session_storage.py
"""
Comprehensive tests for PostgresSessionStorage Phase 4 methods.

This module tests the 7 methods implemented in Phase 4:
- save_context_preset()
- get_context_preset()
- list_context_presets()
- delete_context_preset()
- rename_context_preset()
- add_episode()
- get_episodes()

Tests cover both tenant mode (SQLAlchemy session) and legacy mode (psycopg pool).
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

# Standard imports - package must be installed or PYTHONPATH=src set
from llmcore.models import (
    ContextPreset, ContextPresetItem, ContextItemType,
    Episode, EpisodeType
)
from llmcore.exceptions import SessionStorageError
from llmcore.storage.postgres_session_storage import PostgresSessionStorage


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_preset() -> ContextPreset:
    """Create a sample ContextPreset for testing."""
    items = [
        ContextPresetItem(
            item_id="item-001",
            type=ContextItemType.PRESET_TEXT_CONTENT,
            content="You are a helpful assistant.",
            source_identifier="system",
            metadata={"priority": "high"}
        ),
        ContextPresetItem(
            item_id="item-002",
            type=ContextItemType.USER_FILE,
            content="File content here",
            source_identifier="file:///test.txt",
            metadata={}
        ),
    ]
    return ContextPreset(
        name="test_preset",
        description="A test preset for debugging",
        items=items,
        created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        metadata={"author": "test_user"}
    )


@pytest.fixture
def sample_episode() -> Episode:
    """Create a sample Episode for testing."""
    return Episode(
        episode_id="ep-001",
        session_id="session-abc",
        timestamp=datetime(2025, 1, 10, 14, 30, 0, tzinfo=timezone.utc),
        event_type=EpisodeType.ACTION,
        data={"action": "search", "query": "test query", "result_count": 5}
    )


@pytest.fixture
def mock_tenant_session() -> AsyncMock:
    """Create a mock SQLAlchemy AsyncSession for tenant mode testing."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_pool_connection():
    """Create a mock psycopg pool and connection for legacy mode testing."""
    mock_cursor = AsyncMock()
    mock_cursor.rowcount = 1
    mock_cursor.fetchone = AsyncMock(return_value=None)
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.__aiter__ = lambda self: self
    mock_cursor.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.cursor = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_cursor),
        __aexit__=AsyncMock(return_value=None)
    ))
    mock_conn.transaction = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=None)
    ))
    mock_conn.row_factory = None

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_conn),
        __aexit__=AsyncMock(return_value=None)
    ))

    return mock_pool, mock_conn, mock_cursor


@pytest.fixture
def storage_tenant_mode(mock_tenant_session) -> PostgresSessionStorage:
    """Create a PostgresSessionStorage instance configured for tenant mode."""
    storage = PostgresSessionStorage()
    storage._tenant_session = mock_tenant_session
    storage._pool = None
    storage._context_presets_table = "context_presets"
    storage._context_preset_items_table = "context_preset_items"
    storage._episodes_table = "episodes"
    return storage


@pytest.fixture
def storage_legacy_mode(mock_pool_connection):
    """Create a PostgresSessionStorage instance configured for legacy mode."""
    mock_pool, mock_conn, mock_cursor = mock_pool_connection
    storage = PostgresSessionStorage()
    storage._tenant_session = None
    storage._pool = mock_pool
    storage._context_presets_table = "context_presets"
    storage._context_preset_items_table = "context_preset_items"
    storage._episodes_table = "episodes"
    return storage, mock_conn, mock_cursor


# =============================================================================
# TEST: save_context_preset()
# =============================================================================

class TestSaveContextPresetTenantMode:
    """Tests for save_context_preset in tenant mode."""

    @pytest.mark.asyncio
    async def test_save_preset_executes_upsert(self, storage_tenant_mode, sample_preset, mock_tenant_session):
        """Verify preset upsert SQL is executed."""
        await storage_tenant_mode.save_context_preset(sample_preset)

        # Should have called execute multiple times (upsert, delete items, insert items)
        assert mock_tenant_session.execute.call_count >= 2
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_save_preset_with_items(self, storage_tenant_mode, sample_preset, mock_tenant_session):
        """Verify preset items are saved."""
        await storage_tenant_mode.save_context_preset(sample_preset)

        # Commit should be called
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_save_preset_without_items(self, storage_tenant_mode, mock_tenant_session):
        """Verify preset without items can be saved."""
        empty_preset = ContextPreset(name="empty_preset", items=[])
        await storage_tenant_mode.save_context_preset(empty_preset)

        # Should still commit
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_save_preset_error_handling(self, storage_tenant_mode, sample_preset, mock_tenant_session):
        """Verify error handling when save fails."""
        mock_tenant_session.execute.side_effect = Exception("DB connection lost")

        with pytest.raises(SessionStorageError) as exc_info:
            await storage_tenant_mode.save_context_preset(sample_preset)

        assert "Failed to save context preset" in str(exc_info.value)


class TestSaveContextPresetLegacyMode:
    """Tests for save_context_preset in legacy mode."""

    @pytest.mark.asyncio
    async def test_save_preset_legacy_mode(self, storage_legacy_mode, sample_preset):
        """Verify preset is saved in legacy mode."""
        storage, mock_conn, mock_cursor = storage_legacy_mode
        # Patch Jsonb at the module level
        with patch("llmcore.storage.postgres_session_storage.Jsonb", MagicMock()):
            await storage.save_context_preset(sample_preset)

        # Execute should have been called on the connection
        assert mock_conn.execute.called


# =============================================================================
# TEST: get_context_preset()
# =============================================================================

class TestGetContextPresetTenantMode:
    """Tests for get_context_preset in tenant mode."""

    @pytest.mark.asyncio
    async def test_get_preset_returns_preset(self, storage_tenant_mode, sample_preset, mock_tenant_session):
        """Verify preset is retrieved correctly."""
        # Mock the result
        preset_row = MagicMock()
        preset_row._mapping = {
            "name": sample_preset.name,
            "description": sample_preset.description,
            "created_at": sample_preset.created_at,
            "updated_at": sample_preset.updated_at,
            "metadata": json.dumps(sample_preset.metadata)
        }

        item_rows = []
        for item in sample_preset.items:
            item_mock = MagicMock()
            item_mock._mapping = {
                "item_id": item.item_id,
                "preset_name": sample_preset.name,
                "type": str(item.type),
                "content": item.content,
                "source_identifier": item.source_identifier,
                "metadata": json.dumps(item.metadata)
            }
            item_rows.append(item_mock)

        # First execute returns preset row
        result1 = MagicMock()
        result1.fetchone.return_value = preset_row

        # Second execute returns items
        result2 = MagicMock()
        result2.fetchall.return_value = item_rows

        mock_tenant_session.execute.side_effect = [result1, result2]

        result = await storage_tenant_mode.get_context_preset(sample_preset.name)

        assert result is not None
        assert result.name == sample_preset.name
        assert len(result.items) == len(sample_preset.items)

    @pytest.mark.asyncio
    async def test_get_preset_not_found(self, storage_tenant_mode, mock_tenant_session):
        """Verify None returned when preset not found."""
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.get_context_preset("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_preset_error_handling(self, storage_tenant_mode, mock_tenant_session):
        """Verify error handling when get fails."""
        mock_tenant_session.execute.side_effect = Exception("Query failed")

        with pytest.raises(SessionStorageError) as exc_info:
            await storage_tenant_mode.get_context_preset("test")

        assert "Failed to retrieve context preset" in str(exc_info.value)


# =============================================================================
# TEST: list_context_presets()
# =============================================================================

class TestListContextPresetsTenantMode:
    """Tests for list_context_presets in tenant mode."""

    @pytest.mark.asyncio
    async def test_list_presets_returns_metadata(self, storage_tenant_mode, mock_tenant_session):
        """Verify list returns preset metadata."""
        row1 = MagicMock()
        row1._mapping = {
            "name": "preset1",
            "description": "First preset",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "metadata": "{}",
            "item_count": 3
        }
        row2 = MagicMock()
        row2._mapping = {
            "name": "preset2",
            "description": "Second preset",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "metadata": '{"key": "value"}',
            "item_count": 1
        }

        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row1, row2]
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.list_context_presets()

        assert len(result) == 2
        assert result[0]["name"] == "preset1"
        assert result[1]["name"] == "preset2"
        assert result[0]["item_count"] == 3

    @pytest.mark.asyncio
    async def test_list_presets_empty(self, storage_tenant_mode, mock_tenant_session):
        """Verify empty list returned when no presets exist."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.list_context_presets()

        assert result == []


# =============================================================================
# TEST: delete_context_preset()
# =============================================================================

class TestDeleteContextPresetTenantMode:
    """Tests for delete_context_preset in tenant mode."""

    @pytest.mark.asyncio
    async def test_delete_existing_preset(self, storage_tenant_mode, mock_tenant_session):
        """Verify existing preset is deleted."""
        result_mock = MagicMock()
        result_mock.rowcount = 1
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.delete_context_preset("test_preset")

        assert result is True
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_delete_nonexistent_preset(self, storage_tenant_mode, mock_tenant_session):
        """Verify False returned when preset doesn't exist."""
        # First call for items deletion, second for preset deletion
        items_result = MagicMock()
        items_result.rowcount = 0

        preset_result = MagicMock()
        preset_result.rowcount = 0

        mock_tenant_session.execute.side_effect = [items_result, preset_result]

        result = await storage_tenant_mode.delete_context_preset("nonexistent")

        assert result is False


# =============================================================================
# TEST: rename_context_preset()
# =============================================================================

class TestRenameContextPresetTenantMode:
    """Tests for rename_context_preset in tenant mode."""

    @pytest.mark.asyncio
    async def test_rename_same_name(self, storage_tenant_mode):
        """Verify renaming to same name returns True immediately."""
        result = await storage_tenant_mode.rename_context_preset("test", "test")
        assert result is True

    @pytest.mark.asyncio
    async def test_rename_invalid_name(self, storage_tenant_mode):
        """Verify ValueError for invalid new name."""
        with pytest.raises(ValueError):
            await storage_tenant_mode.rename_context_preset("old", "invalid/name")

    @pytest.mark.asyncio
    async def test_rename_new_name_exists(self, storage_tenant_mode, mock_tenant_session):
        """Verify False when new name already exists."""
        # First execute returns existing row (new name already taken)
        result_mock = MagicMock()
        result_mock.fetchone.return_value = MagicMock()  # Row exists
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.rename_context_preset("old", "existing")

        assert result is False

    @pytest.mark.asyncio
    async def test_rename_old_name_not_found(self, storage_tenant_mode, mock_tenant_session):
        """Verify False when old name doesn't exist."""
        # First check (new name exists) returns None
        check_result = MagicMock()
        check_result.fetchone.return_value = None

        # Second check (get old preset) returns None
        preset_result = MagicMock()
        preset_result.fetchone.return_value = None

        mock_tenant_session.execute.side_effect = [check_result, preset_result]

        result = await storage_tenant_mode.rename_context_preset("nonexistent", "new_name")

        assert result is False


# =============================================================================
# TEST: add_episode()
# =============================================================================

class TestAddEpisodeTenantMode:
    """Tests for add_episode in tenant mode."""

    @pytest.mark.asyncio
    async def test_add_episode_success(self, storage_tenant_mode, sample_episode, mock_tenant_session):
        """Verify episode is added successfully."""
        await storage_tenant_mode.add_episode(sample_episode)

        assert mock_tenant_session.execute.called
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_add_episode_error_handling(self, storage_tenant_mode, sample_episode, mock_tenant_session):
        """Verify error handling when add fails."""
        mock_tenant_session.execute.side_effect = Exception("Insert failed")

        with pytest.raises(SessionStorageError) as exc_info:
            await storage_tenant_mode.add_episode(sample_episode)

        assert "Failed to add episode" in str(exc_info.value)


class TestAddEpisodeLegacyMode:
    """Tests for add_episode in legacy mode."""

    @pytest.mark.asyncio
    async def test_add_episode_legacy_mode(self, storage_legacy_mode, sample_episode):
        """Verify episode is added in legacy mode."""
        storage, mock_conn, mock_cursor = storage_legacy_mode
        # Patch Jsonb at the module level
        with patch("llmcore.storage.postgres_session_storage.Jsonb", MagicMock()):
            await storage.add_episode(sample_episode)

        assert mock_conn.execute.called


# =============================================================================
# TEST: get_episodes()
# =============================================================================

class TestGetEpisodesTenantMode:
    """Tests for get_episodes in tenant mode."""

    @pytest.mark.asyncio
    async def test_get_episodes_success(self, storage_tenant_mode, sample_episode, mock_tenant_session):
        """Verify episodes are retrieved correctly."""
        row = MagicMock()
        row._mapping = {
            "episode_id": sample_episode.episode_id,
            "session_id": sample_episode.session_id,
            "timestamp": sample_episode.timestamp,
            "event_type": str(sample_episode.event_type),
            "data": json.dumps(sample_episode.data)
        }

        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row]
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.get_episodes("session-abc")

        assert len(result) == 1
        assert result[0].episode_id == sample_episode.episode_id
        assert result[0].event_type == sample_episode.event_type

    @pytest.mark.asyncio
    async def test_get_episodes_empty(self, storage_tenant_mode, mock_tenant_session):
        """Verify empty list when no episodes exist."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.get_episodes("nonexistent-session")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_episodes_pagination(self, storage_tenant_mode, mock_tenant_session):
        """Verify pagination parameters are passed correctly."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_tenant_session.execute.return_value = result_mock

        await storage_tenant_mode.get_episodes("session-abc", limit=50, offset=10)

        # Verify execute was called with correct parameters
        call_args = mock_tenant_session.execute.call_args
        params = call_args[0][1]  # Second positional arg is the params dict
        assert params["limit"] == 50
        assert params["offset"] == 10

    @pytest.mark.asyncio
    async def test_get_episodes_error_handling(self, storage_tenant_mode, mock_tenant_session):
        """Verify error handling when get fails."""
        mock_tenant_session.execute.side_effect = Exception("Query failed")

        with pytest.raises(SessionStorageError) as exc_info:
            await storage_tenant_mode.get_episodes("session-abc")

        assert "Failed to retrieve episodes" in str(exc_info.value)


# =============================================================================
# TEST: API Compatibility and Backward Compatibility
# =============================================================================

class TestAPICompatibility:
    """Tests for API backward compatibility."""

    @pytest.mark.asyncio
    async def test_storage_instantiation(self):
        """Verify storage can be instantiated without errors."""
        storage = PostgresSessionStorage()
        assert storage is not None

    def test_method_signatures_exist(self):
        """Verify all required methods exist on the class."""
        storage = PostgresSessionStorage()

        assert hasattr(storage, 'save_context_preset')
        assert hasattr(storage, 'get_context_preset')
        assert hasattr(storage, 'list_context_presets')
        assert hasattr(storage, 'delete_context_preset')
        assert hasattr(storage, 'rename_context_preset')
        assert hasattr(storage, 'add_episode')
        assert hasattr(storage, 'get_episodes')

        # All should be async methods
        import asyncio
        assert asyncio.iscoroutinefunction(storage.save_context_preset)
        assert asyncio.iscoroutinefunction(storage.get_context_preset)
        assert asyncio.iscoroutinefunction(storage.list_context_presets)
        assert asyncio.iscoroutinefunction(storage.delete_context_preset)
        assert asyncio.iscoroutinefunction(storage.rename_context_preset)
        assert asyncio.iscoroutinefunction(storage.add_episode)
        assert asyncio.iscoroutinefunction(storage.get_episodes)


# =============================================================================
# TEST: Multiple Episode Types
# =============================================================================

class TestEpisodeTypes:
    """Tests for different episode types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", [
        EpisodeType.THOUGHT,
        EpisodeType.ACTION,
        EpisodeType.OBSERVATION,
        EpisodeType.USER_INTERACTION,
        EpisodeType.AGENT_REFLECTION,
    ])
    async def test_episode_type_variations(self, storage_tenant_mode, mock_tenant_session, event_type):
        """Verify all episode types can be saved."""
        episode = Episode(
            episode_id=f"ep-{event_type.value}",
            session_id="session-test",
            event_type=event_type,
            data={"type_test": True}
        )

        await storage_tenant_mode.add_episode(episode)

        assert mock_tenant_session.execute.called
        assert mock_tenant_session.commit.called


# =============================================================================
# TEST: Regression Tests
# =============================================================================

class TestRegressionPhase4:
    """Regression tests for Phase 4 bug fixes."""

    @pytest.mark.asyncio
    async def test_save_preset_not_stub(self, storage_tenant_mode, sample_preset, mock_tenant_session):
        """Verify save_context_preset is not a stub (previously just 'pass')."""
        await storage_tenant_mode.save_context_preset(sample_preset)

        # The stub version did nothing - now it should call execute
        assert mock_tenant_session.execute.called, "save_context_preset should execute SQL, not be a stub"

    @pytest.mark.asyncio
    async def test_get_preset_not_always_none(self, storage_tenant_mode, sample_preset, mock_tenant_session):
        """Verify get_context_preset doesn't always return None."""
        # Setup mock to return valid preset data
        preset_row = MagicMock()
        preset_row._mapping = {
            "name": sample_preset.name,
            "description": sample_preset.description,
            "created_at": sample_preset.created_at,
            "updated_at": sample_preset.updated_at,
            "metadata": "{}"
        }
        result1 = MagicMock()
        result1.fetchone.return_value = preset_row

        result2 = MagicMock()
        result2.fetchall.return_value = []
        mock_tenant_session.execute.side_effect = [result1, result2]

        result = await storage_tenant_mode.get_context_preset(sample_preset.name)

        assert result is not None, "get_context_preset should not always return None"

    @pytest.mark.asyncio
    async def test_list_presets_not_always_empty(self, storage_tenant_mode, mock_tenant_session):
        """Verify list_context_presets doesn't always return []."""
        row = MagicMock()
        row._mapping = {
            "name": "test",
            "description": "test desc",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "metadata": "{}",
            "item_count": 0
        }
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row]
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.list_context_presets()

        assert len(result) > 0, "list_context_presets should not always return []"

    @pytest.mark.asyncio
    async def test_delete_preset_not_always_false(self, storage_tenant_mode, mock_tenant_session):
        """Verify delete_context_preset doesn't always return False."""
        result_mock = MagicMock()
        result_mock.rowcount = 1
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.delete_context_preset("test")

        assert result is True, "delete_context_preset should not always return False"

    @pytest.mark.asyncio
    async def test_add_episode_not_stub(self, storage_tenant_mode, sample_episode, mock_tenant_session):
        """Verify add_episode is not a stub (previously just 'pass')."""
        await storage_tenant_mode.add_episode(sample_episode)

        assert mock_tenant_session.execute.called, "add_episode should execute SQL, not be a stub"

    @pytest.mark.asyncio
    async def test_get_episodes_not_always_empty(self, storage_tenant_mode, sample_episode, mock_tenant_session):
        """Verify get_episodes doesn't always return []."""
        row = MagicMock()
        row._mapping = {
            "episode_id": sample_episode.episode_id,
            "session_id": sample_episode.session_id,
            "timestamp": sample_episode.timestamp,
            "event_type": str(sample_episode.event_type),
            "data": json.dumps(sample_episode.data)
        }
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row]
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.get_episodes(sample_episode.session_id)

        assert len(result) > 0, "get_episodes should not always return []"


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_preset_with_empty_metadata(self, storage_tenant_mode, mock_tenant_session):
        """Verify preset with empty metadata can be saved."""
        preset = ContextPreset(name="no_metadata", items=[], metadata={})
        await storage_tenant_mode.save_context_preset(preset)
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_preset_with_special_characters_in_description(self, storage_tenant_mode, mock_tenant_session):
        """Verify preset with special characters in description can be saved."""
        preset = ContextPreset(
            name="special_preset",
            description="Description with 'quotes', \"double quotes\", and emoji 🎉",
            items=[]
        )
        await storage_tenant_mode.save_context_preset(preset)
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_episode_with_complex_data(self, storage_tenant_mode, mock_tenant_session):
        """Verify episode with complex nested data can be saved."""
        episode = Episode(
            episode_id="ep-complex",
            session_id="session-test",
            event_type=EpisodeType.OBSERVATION,
            data={
                "nested": {"deep": {"value": 42}},
                "list": [1, 2, 3],
                "string": "test",
                "null": None,
                "bool": True
            }
        )
        await storage_tenant_mode.add_episode(episode)
        assert mock_tenant_session.commit.called

    @pytest.mark.asyncio
    async def test_get_episodes_with_large_offset(self, storage_tenant_mode, mock_tenant_session):
        """Verify get_episodes handles large offset gracefully."""
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_tenant_session.execute.return_value = result_mock

        result = await storage_tenant_mode.get_episodes("session", limit=10, offset=999999)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
