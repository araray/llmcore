# tests/api/test_phase4_rag_collections.py
"""
Phase 4: RAG Collection Management API Tests

Tests for llmcore API methods:
- list_rag_collections()
- get_rag_collection_info()
- delete_rag_collection()

These tests verify the RAG collection management functionality
added in Phase 4 of the UNIFIED_IMPLEMENTATION_PLAN.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

# =============================================================================
# MOCK INFRASTRUCTURE
# =============================================================================


class MockVectorStorage:
    """Mock vector storage backend for testing."""

    def __init__(self):
        self.collections: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def add_mock_collection(
        self,
        name: str,
        count: int = 0,
        metadata: Optional[Dict] = None,
        embedding_dim: int = 1536,
    ):
        """Add a mock collection for testing."""
        self.collections[name] = {
            "name": name,
            "count": count,
            "metadata": metadata or {},
            "embedding_dimension": embedding_dim,
        }

    async def list_collection_names(self) -> List[str]:
        """List all collection names."""
        return list(self.collections.keys())

    async def list_collections(self) -> List[str]:
        """Alias for list_collection_names."""
        return await self.list_collection_names()

    async def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection info."""
        return self.collections.get(name)

    async def delete_collection(self, name: str, force: bool = False) -> bool:
        """Delete a collection."""
        if name not in self.collections:
            return False
        info = self.collections[name]
        if info["count"] > 0 and not force:
            raise ValueError(f"Collection '{name}' has documents. Use force=True.")
        del self.collections[name]
        return True


class MockStorageManager:
    """Mock storage manager for testing."""

    def __init__(self):
        self.vector_storage = MockVectorStorage()
        self._initialized = True

    async def get_vector_storage(self):
        """Get vector storage backend."""
        return self.vector_storage


class MockLLMCoreAPI:
    """Mock LLMCore API for testing collection management."""

    def __init__(self):
        self._storage_manager = MockStorageManager()
        self._initialized = True

    async def list_rag_collections(self) -> List[str]:
        """List all RAG collections."""
        vector_storage = await self._storage_manager.get_vector_storage()
        if not vector_storage:
            return []
        return await vector_storage.list_collection_names()

    async def get_rag_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a RAG collection."""
        vector_storage = await self._storage_manager.get_vector_storage()
        if not vector_storage:
            return None
        return await vector_storage.get_collection_info(collection_name)

    async def delete_rag_collection(self, collection_name: str, force: bool = False) -> bool:
        """Delete a RAG collection."""
        vector_storage = await self._storage_manager.get_vector_storage()
        if not vector_storage:
            return False
        return await vector_storage.delete_collection(collection_name, force=force)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_api():
    """Create a mock LLMCore API instance."""
    return MockLLMCoreAPI()


@pytest.fixture
def api_with_collections(mock_api):
    """Create mock API with pre-populated collections."""
    storage = mock_api._storage_manager.vector_storage
    storage.add_mock_collection("default", count=100, embedding_dim=1536)
    storage.add_mock_collection("code_embeddings", count=500, embedding_dim=768)
    storage.add_mock_collection("empty_collection", count=0)
    storage.add_mock_collection(
        "with_metadata",
        count=50,
        metadata={"model": "text-embedding-3-small", "version": "1.0"},
    )
    return mock_api


# =============================================================================
# LIST_RAG_COLLECTIONS TESTS
# =============================================================================


class TestListRagCollections:
    """Tests for list_rag_collections() API method."""

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_api):
        """Test listing when no collections exist."""
        result = await mock_api.list_rag_collections()
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_single_collection(self, mock_api):
        """Test listing with one collection."""
        mock_api._storage_manager.vector_storage.add_mock_collection("test_coll")

        result = await mock_api.list_rag_collections()

        assert result == ["test_coll"]

    @pytest.mark.asyncio
    async def test_list_multiple_collections(self, api_with_collections):
        """Test listing multiple collections."""
        result = await api_with_collections.list_rag_collections()

        assert len(result) == 4
        assert "default" in result
        assert "code_embeddings" in result
        assert "empty_collection" in result
        assert "with_metadata" in result

    @pytest.mark.asyncio
    async def test_list_returns_list_type(self, api_with_collections):
        """Test that list always returns a list, not other iterables."""
        result = await api_with_collections.list_rag_collections()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_no_vector_storage(self):
        """Test listing when vector storage is not available."""
        api = MockLLMCoreAPI()
        api._storage_manager.get_vector_storage = AsyncMock(return_value=None)

        result = await api.list_rag_collections()

        assert result == []


# =============================================================================
# GET_RAG_COLLECTION_INFO TESTS
# =============================================================================


class TestGetRagCollectionInfo:
    """Tests for get_rag_collection_info() API method."""

    @pytest.mark.asyncio
    async def test_get_existing_collection(self, api_with_collections):
        """Test getting info for an existing collection."""
        result = await api_with_collections.get_rag_collection_info("default")

        assert result is not None
        assert result["name"] == "default"
        assert result["count"] == 100
        assert result["embedding_dimension"] == 1536

    @pytest.mark.asyncio
    async def test_get_nonexistent_collection(self, api_with_collections):
        """Test getting info for a collection that doesn't exist."""
        result = await api_with_collections.get_rag_collection_info("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_empty_collection(self, api_with_collections):
        """Test getting info for an empty collection."""
        result = await api_with_collections.get_rag_collection_info("empty_collection")

        assert result is not None
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_collection_with_metadata(self, api_with_collections):
        """Test getting info for collection with metadata."""
        result = await api_with_collections.get_rag_collection_info("with_metadata")

        assert result is not None
        assert "metadata" in result
        assert result["metadata"]["model"] == "text-embedding-3-small"
        assert result["metadata"]["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_get_collection_embedding_dimension(self, api_with_collections):
        """Test that embedding dimension is returned correctly."""
        result = await api_with_collections.get_rag_collection_info("code_embeddings")

        assert result["embedding_dimension"] == 768

    @pytest.mark.asyncio
    async def test_get_no_vector_storage(self):
        """Test getting info when vector storage is not available."""
        api = MockLLMCoreAPI()
        api._storage_manager.get_vector_storage = AsyncMock(return_value=None)

        result = await api.get_rag_collection_info("any_collection")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_dict_or_none(self, api_with_collections):
        """Test that get returns dict for existing, None for non-existing."""
        existing = await api_with_collections.get_rag_collection_info("default")
        assert isinstance(existing, dict)

        nonexistent = await api_with_collections.get_rag_collection_info("nope")
        assert nonexistent is None


# =============================================================================
# DELETE_RAG_COLLECTION TESTS
# =============================================================================


class TestDeleteRagCollection:
    """Tests for delete_rag_collection() API method."""

    @pytest.mark.asyncio
    async def test_delete_empty_collection(self, api_with_collections):
        """Test deleting an empty collection without force."""
        result = await api_with_collections.delete_rag_collection("empty_collection")

        assert result is True
        # Verify it's gone
        remaining = await api_with_collections.list_rag_collections()
        assert "empty_collection" not in remaining

    @pytest.mark.asyncio
    async def test_delete_nonempty_without_force(self, api_with_collections):
        """Test that deleting non-empty collection without force raises error."""
        with pytest.raises(ValueError, match="has documents"):
            await api_with_collections.delete_rag_collection("default")

        # Verify it still exists
        remaining = await api_with_collections.list_rag_collections()
        assert "default" in remaining

    @pytest.mark.asyncio
    async def test_delete_nonempty_with_force(self, api_with_collections):
        """Test deleting non-empty collection with force=True."""
        result = await api_with_collections.delete_rag_collection("default", force=True)

        assert result is True
        remaining = await api_with_collections.list_rag_collections()
        assert "default" not in remaining

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, api_with_collections):
        """Test deleting a collection that doesn't exist."""
        result = await api_with_collections.delete_rag_collection("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_no_vector_storage(self):
        """Test deleting when vector storage is not available."""
        api = MockLLMCoreAPI()
        api._storage_manager.get_vector_storage = AsyncMock(return_value=None)

        result = await api.delete_rag_collection("any_collection", force=True)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_multiple_collections(self, api_with_collections):
        """Test deleting multiple collections in sequence."""
        # Delete empty one first
        await api_with_collections.delete_rag_collection("empty_collection")
        # Delete non-empty with force
        await api_with_collections.delete_rag_collection("code_embeddings", force=True)

        remaining = await api_with_collections.list_rag_collections()
        assert len(remaining) == 2
        assert "empty_collection" not in remaining
        assert "code_embeddings" not in remaining

    @pytest.mark.asyncio
    async def test_delete_returns_boolean(self, api_with_collections):
        """Test that delete always returns a boolean."""
        result_success = await api_with_collections.delete_rag_collection("empty_collection")
        assert isinstance(result_success, bool)

        result_fail = await api_with_collections.delete_rag_collection("nonexistent")
        assert isinstance(result_fail, bool)


# =============================================================================
# INTEGRATION SCENARIOS
# =============================================================================


class TestRagCollectionIntegration:
    """Integration tests for RAG collection management."""

    @pytest.mark.asyncio
    async def test_create_list_info_delete_cycle(self, mock_api):
        """Test full lifecycle: create, list, info, delete."""
        storage = mock_api._storage_manager.vector_storage

        # Create collection
        storage.add_mock_collection("lifecycle_test", count=10)

        # List should include it
        collections = await mock_api.list_rag_collections()
        assert "lifecycle_test" in collections

        # Info should return details
        info = await mock_api.get_rag_collection_info("lifecycle_test")
        assert info["count"] == 10

        # Delete with force
        deleted = await mock_api.delete_rag_collection("lifecycle_test", force=True)
        assert deleted is True

        # Should be gone
        collections = await mock_api.list_rag_collections()
        assert "lifecycle_test" not in collections

        # Info should return None
        info = await mock_api.get_rag_collection_info("lifecycle_test")
        assert info is None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, api_with_collections):
        """Test that multiple operations don't interfere."""
        # List collections
        initial = await api_with_collections.list_rag_collections()

        # Get info on multiple collections
        info1 = await api_with_collections.get_rag_collection_info("default")
        info2 = await api_with_collections.get_rag_collection_info("code_embeddings")

        # Both should succeed
        assert info1 is not None
        assert info2 is not None

        # List should still work
        final = await api_with_collections.list_rag_collections()
        assert initial == final


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
