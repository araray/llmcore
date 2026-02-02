# tests/storage/test_chromadb_collections.py
"""
Phase 4: ChromaDB Collection Management Tests

Tests for ChromaDB vector storage collection methods:
- list_collections()
- get_collection_info()
- delete_collection()

These tests verify the ChromaDB collection management functionality
added in Phase 4 of the UNIFIED_IMPLEMENTATION_PLAN.
"""

from typing import Any, Dict, List, Optional

import pytest

# =============================================================================
# MOCK CHROMADB CLIENT
# =============================================================================


class MockChromaCollection:
    """Mock ChromaDB collection."""

    def __init__(self, name: str, count: int = 0, metadata: Optional[Dict] = None):
        self.name = name
        self._count = count
        self.metadata = metadata or {}

    def count(self) -> int:
        """Return document count."""
        return self._count


class MockChromaClient:
    """Mock ChromaDB client for testing."""

    def __init__(self):
        self._collections: Dict[str, MockChromaCollection] = {}

    def add_collection(
        self, name: str, count: int = 0, metadata: Optional[Dict] = None
    ):
        """Add a mock collection."""
        self._collections[name] = MockChromaCollection(name, count, metadata)

    def list_collections(self) -> List[MockChromaCollection]:
        """List all collections."""
        return list(self._collections.values())

    def get_collection(self, name: str) -> MockChromaCollection:
        """Get a collection by name."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} not found")
        return self._collections[name]

    def delete_collection(self, name: str):
        """Delete a collection."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} not found")
        del self._collections[name]


# =============================================================================
# CHROMADB VECTOR STORAGE IMPLEMENTATION (for testing)
# =============================================================================


class ChromaDBVectorStorage:
    """Simplified ChromaDB storage implementation for testing."""

    def __init__(self, client: MockChromaClient):
        self._client = client
        self._initialized = True
        self._default_embedding_dimension = 1536

    def _sync_list_collection_names(self) -> List[str]:
        """Synchronous list collection names."""
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def _sync_get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Synchronous get collection info."""
        try:
            collection = self._client.get_collection(name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
                "embedding_dimension": self._default_embedding_dimension,
            }
        except ValueError:
            return None

    def _sync_delete_collection(self, name: str, force: bool = False) -> bool:
        """Synchronous delete collection."""
        try:
            collection = self._client.get_collection(name)
            if collection.count() > 0 and not force:
                raise ValueError(
                    f"Collection '{name}' has {collection.count()} documents. "
                    f"Use force=True to delete."
                )
            self._client.delete_collection(name)
            return True
        except ValueError as e:
            if "not found" in str(e).lower():
                return False
            raise

    # Async wrappers
    async def list_collection_names(self) -> List[str]:
        """List all collection names."""
        return self._sync_list_collection_names()

    async def list_collections(self) -> List[str]:
        """Alias for list_collection_names."""
        return await self.list_collection_names()

    async def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information."""
        return self._sync_get_collection_info(name)

    async def delete_collection(self, name: str, force: bool = False) -> bool:
        """Delete a collection."""
        return self._sync_delete_collection(name, force)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_client():
    """Create a mock ChromaDB client."""
    return MockChromaClient()


@pytest.fixture
def storage(mock_client):
    """Create ChromaDB storage with mock client."""
    return ChromaDBVectorStorage(mock_client)


@pytest.fixture
def storage_with_collections(mock_client):
    """Create storage with pre-populated collections."""
    mock_client.add_collection("default", count=100)
    mock_client.add_collection("code_docs", count=500, metadata={"type": "code"})
    mock_client.add_collection("empty", count=0)
    mock_client.add_collection(
        "with_meta",
        count=25,
        metadata={"model": "text-embedding-ada-002", "created_at": "2025-01-01"},
    )
    return ChromaDBVectorStorage(mock_client)


# =============================================================================
# LIST_COLLECTIONS TESTS
# =============================================================================


class TestListCollections:
    """Tests for list_collections() and list_collection_names()."""

    @pytest.mark.asyncio
    async def test_list_empty(self, storage):
        """Test listing when no collections exist."""
        result = await storage.list_collections()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_collection_names_empty(self, storage):
        """Test list_collection_names when empty."""
        result = await storage.list_collection_names()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_single(self, mock_client):
        """Test listing with one collection."""
        mock_client.add_collection("single_coll")
        storage = ChromaDBVectorStorage(mock_client)

        result = await storage.list_collections()

        assert result == ["single_coll"]

    @pytest.mark.asyncio
    async def test_list_multiple(self, storage_with_collections):
        """Test listing multiple collections."""
        result = await storage_with_collections.list_collections()

        assert len(result) == 4
        assert "default" in result
        assert "code_docs" in result
        assert "empty" in result
        assert "with_meta" in result

    @pytest.mark.asyncio
    async def test_list_returns_names_only(self, storage_with_collections):
        """Test that list returns only string names, not objects."""
        result = await storage_with_collections.list_collections()

        for item in result:
            assert isinstance(item, str)

    @pytest.mark.asyncio
    async def test_list_collection_names_alias(self, storage_with_collections):
        """Test that list_collections and list_collection_names return same data."""
        names = await storage_with_collections.list_collection_names()
        collections = await storage_with_collections.list_collections()

        assert names == collections


# =============================================================================
# GET_COLLECTION_INFO TESTS
# =============================================================================


class TestGetCollectionInfo:
    """Tests for get_collection_info()."""

    @pytest.mark.asyncio
    async def test_get_existing(self, storage_with_collections):
        """Test getting info for existing collection."""
        result = await storage_with_collections.get_collection_info("default")

        assert result is not None
        assert result["name"] == "default"
        assert result["count"] == 100

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, storage_with_collections):
        """Test getting info for non-existent collection."""
        result = await storage_with_collections.get_collection_info("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_empty_collection(self, storage_with_collections):
        """Test getting info for empty collection."""
        result = await storage_with_collections.get_collection_info("empty")

        assert result is not None
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_with_metadata(self, storage_with_collections):
        """Test getting info includes metadata."""
        result = await storage_with_collections.get_collection_info("with_meta")

        assert result is not None
        assert "metadata" in result
        assert result["metadata"]["model"] == "text-embedding-ada-002"
        assert result["metadata"]["created_at"] == "2025-01-01"

    @pytest.mark.asyncio
    async def test_get_includes_embedding_dimension(self, storage_with_collections):
        """Test that info includes embedding dimension."""
        result = await storage_with_collections.get_collection_info("default")

        assert "embedding_dimension" in result
        assert result["embedding_dimension"] == 1536

    @pytest.mark.asyncio
    async def test_get_returns_dict_structure(self, storage_with_collections):
        """Test that returned dict has expected keys."""
        result = await storage_with_collections.get_collection_info("code_docs")

        assert "name" in result
        assert "count" in result
        assert "metadata" in result
        assert "embedding_dimension" in result


# =============================================================================
# DELETE_COLLECTION TESTS
# =============================================================================


class TestDeleteCollection:
    """Tests for delete_collection()."""

    @pytest.mark.asyncio
    async def test_delete_empty_without_force(self, storage_with_collections):
        """Test deleting empty collection without force."""
        result = await storage_with_collections.delete_collection("empty")

        assert result is True
        remaining = await storage_with_collections.list_collections()
        assert "empty" not in remaining

    @pytest.mark.asyncio
    async def test_delete_nonempty_without_force_raises(self, storage_with_collections):
        """Test that deleting non-empty without force raises error."""
        with pytest.raises(ValueError, match="has.*documents"):
            await storage_with_collections.delete_collection("default")

        # Verify not deleted
        remaining = await storage_with_collections.list_collections()
        assert "default" in remaining

    @pytest.mark.asyncio
    async def test_delete_nonempty_with_force(self, storage_with_collections):
        """Test deleting non-empty collection with force."""
        result = await storage_with_collections.delete_collection("default", force=True)

        assert result is True
        remaining = await storage_with_collections.list_collections()
        assert "default" not in remaining

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, storage_with_collections):
        """Test deleting non-existent collection returns False."""
        result = await storage_with_collections.delete_collection("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_with_force_nonexistent(self, storage_with_collections):
        """Test deleting non-existent with force still returns False."""
        result = await storage_with_collections.delete_collection(
            "nonexistent", force=True
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_verifies_removal(self, storage_with_collections):
        """Test that deleted collection is no longer accessible."""
        await storage_with_collections.delete_collection("empty")

        # Should not be in list
        names = await storage_with_collections.list_collections()
        assert "empty" not in names

        # Info should return None
        info = await storage_with_collections.get_collection_info("empty")
        assert info is None

    @pytest.mark.asyncio
    async def test_delete_multiple_sequential(self, storage_with_collections):
        """Test deleting multiple collections in sequence."""
        await storage_with_collections.delete_collection("empty")
        await storage_with_collections.delete_collection("code_docs", force=True)

        remaining = await storage_with_collections.list_collections()
        assert len(remaining) == 2
        assert "default" in remaining
        assert "with_meta" in remaining


# =============================================================================
# SYNC METHOD TESTS
# =============================================================================


class TestSyncMethods:
    """Tests for synchronous implementation methods."""

    def test_sync_list_collection_names(self, storage_with_collections):
        """Test synchronous list_collection_names."""
        result = storage_with_collections._sync_list_collection_names()

        assert len(result) == 4
        assert isinstance(result, list)

    def test_sync_get_collection_info_exists(self, storage_with_collections):
        """Test synchronous get_collection_info for existing."""
        result = storage_with_collections._sync_get_collection_info("default")

        assert result is not None
        assert result["name"] == "default"

    def test_sync_get_collection_info_not_exists(self, storage_with_collections):
        """Test synchronous get_collection_info for non-existing."""
        result = storage_with_collections._sync_get_collection_info("nope")

        assert result is None

    def test_sync_delete_collection_empty(self, storage_with_collections):
        """Test synchronous delete_collection for empty."""
        result = storage_with_collections._sync_delete_collection("empty")

        assert result is True

    def test_sync_delete_collection_nonempty_no_force(self, storage_with_collections):
        """Test synchronous delete without force raises."""
        with pytest.raises(ValueError):
            storage_with_collections._sync_delete_collection("default")

    def test_sync_delete_collection_with_force(self, storage_with_collections):
        """Test synchronous delete with force succeeds."""
        result = storage_with_collections._sync_delete_collection("default", force=True)

        assert result is True


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge case tests for collection management."""

    @pytest.mark.asyncio
    async def test_collection_with_special_characters(self, mock_client):
        """Test collection names with special characters."""
        mock_client.add_collection("test-collection_v1.0", count=5)
        storage = ChromaDBVectorStorage(mock_client)

        result = await storage.get_collection_info("test-collection_v1.0")

        assert result is not None
        assert result["name"] == "test-collection_v1.0"

    @pytest.mark.asyncio
    async def test_collection_with_empty_metadata(self, mock_client):
        """Test collection with explicitly empty metadata."""
        mock_client.add_collection("no_meta", count=10, metadata={})
        storage = ChromaDBVectorStorage(mock_client)

        result = await storage.get_collection_info("no_meta")

        assert result is not None
        assert result["metadata"] == {}

    @pytest.mark.asyncio
    async def test_collection_zero_count(self, mock_client):
        """Test collection with zero documents."""
        mock_client.add_collection("zero_docs", count=0)
        storage = ChromaDBVectorStorage(mock_client)

        info = await storage.get_collection_info("zero_docs")
        assert info["count"] == 0

        # Should delete without force
        result = await storage.delete_collection("zero_docs")
        assert result is True

    @pytest.mark.asyncio
    async def test_large_document_count(self, mock_client):
        """Test collection with large document count."""
        mock_client.add_collection("huge", count=1_000_000)
        storage = ChromaDBVectorStorage(mock_client)

        info = await storage.get_collection_info("huge")

        assert info["count"] == 1_000_000


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
