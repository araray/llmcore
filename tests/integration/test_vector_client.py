# tests/integration/test_vector_client.py
"""
Tests for LLMCore Vector Client Integration.

Phase 3 (SYMBIOSIS): Tests for the LLMCoreVectorClient adapter that enables
SemantiScan to delegate vector operations to LLMCore.

These tests verify:
- Client initialization and configuration
- Chunk conversion and document handling
- Query operations with filtering
- Async-to-sync bridging
- Protocol compliance
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockChunk:
    """Mock SemantiScan Chunk for testing."""

    id: str
    content: str
    metadata: Dict[str, Any]

    @classmethod
    def create(cls, content: str = "Test content", **metadata):
        return cls(
            id=f"chunk_{uuid.uuid4().hex[:8]}",
            content=content,
            metadata=metadata,
        )


@dataclass
class MockContextDocument:
    """Mock ContextDocument for testing."""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    score: float = 0.0


class MockVectorStorage:
    """Mock vector storage backend for testing."""

    def __init__(self):
        self.documents = {}
        self.collections = {}

    async def batch_upsert_documents(
        self,
        documents,
        collection_name,
        context=None,
        batch_config=None,
    ):
        if collection_name not in self.collections:
            self.collections[collection_name] = {}

        for doc in documents:
            self.collections[collection_name][doc.id] = doc

        return MagicMock(success_count=len(documents), total=len(documents))

    async def similarity_search(
        self,
        query_embedding,
        k,
        collection_name,
        context=None,
        search_config=None,
    ):
        if collection_name not in self.collections:
            return []

        docs = list(self.collections[collection_name].values())[:k]
        return docs

    async def create_collection(
        self,
        name,
        vector_dimension=None,
        if_not_exists=True,
        context=None,
    ):
        if name not in self.collections:
            self.collections[name] = {}
        return True


class MockStorageManager:
    """Mock storage manager for testing."""

    def __init__(self):
        self.vector_storage = MockVectorStorage()


class MockLLMCore:
    """Mock LLMCore instance for testing."""

    def __init__(self):
        self._storage_manager = MockStorageManager()
        self._documents_added = []

    async def add_documents_to_vector_store(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        self._documents_added.extend(documents)
        return [doc.get("id", str(uuid.uuid4())) for doc in documents]

    async def search_vector_store(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[MockContextDocument]:
        return [
            MockContextDocument(
                id=f"doc_{i}",
                content=f"Result content {i}",
                metadata={"source": f"file_{i}.py"},
                score=1.0 - (i * 0.1),
            )
            for i in range(min(k, 5))
        ]


# =============================================================================
# TESTS FOR CHUNK ADAPTER
# =============================================================================


class TestChunkAdapter:
    """Tests for the ChunkAdapter utility class."""

    def test_chunk_to_document_basic(self):
        """Test basic chunk to document conversion."""
        from llmcore.integration.vector_client import ChunkAdapter

        chunk = MockChunk.create(
            content="def hello(): pass",
            file_path="/src/main.py",
            repo_name="test-repo",
            commit_hash="abc123",
        )
        embedding = [0.1, 0.2, 0.3]

        doc = ChunkAdapter.chunk_to_document(chunk, embedding)

        assert doc["id"] == chunk.id
        assert doc["content"] == chunk.content
        assert doc["embedding"] == embedding
        assert doc["metadata"]["file_path"] == "/src/main.py"
        assert doc["metadata"]["repo_name"] == "test-repo"
        assert doc["metadata"]["commit_hash"] == "abc123"

    def test_chunk_to_document_filters_complex_metadata(self):
        """Test that complex metadata types are converted to strings."""
        from llmcore.integration.vector_client import ChunkAdapter

        chunk = MockChunk.create(
            content="test",
            file_path="/test.py",
            complex_value={"nested": "dict"},  # Should be filtered or converted
        )
        embedding = [0.1]

        doc = ChunkAdapter.chunk_to_document(chunk, embedding)

        # Core metadata should be preserved
        assert doc["metadata"]["file_path"] == "/test.py"
        # Complex values not in core/simple keys should be omitted
        assert "complex_value" not in doc["metadata"]

    def test_chunk_to_document_handles_missing_fields(self):
        """Test handling of chunks with missing metadata fields."""
        from llmcore.integration.vector_client import ChunkAdapter

        @dataclass
        class MinimalChunk:
            content: str

        chunk = MinimalChunk(content="minimal content")
        embedding = [0.5]

        doc = ChunkAdapter.chunk_to_document(chunk, embedding)

        assert doc["content"] == "minimal content"
        assert doc["embedding"] == embedding
        assert "id" in doc  # Should have auto-generated ID

    def test_document_to_result_format(self):
        """Test conversion to ChromaDB-compatible result format."""
        from llmcore.integration.vector_client import ChunkAdapter

        result = ChunkAdapter.document_to_result(
            doc_id="doc_123",
            content="Result content",
            metadata={"source": "test.py"},
            distance=0.25,
        )

        assert result["id"] == "doc_123"
        assert result["document"] == "Result content"
        assert result["metadata"]["source"] == "test.py"
        assert result["distance"] == 0.25


# =============================================================================
# TESTS FOR CONFIGURATION
# =============================================================================


class TestLLMCoreVectorClientConfig:
    """Tests for LLMCoreVectorClientConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from llmcore.integration import LLMCoreVectorClientConfig

        config = LLMCoreVectorClientConfig()

        assert config.collection_name == "codebase_default"
        assert config.user_id is None
        assert config.namespace is None
        assert config.batch_size == 100
        assert config.retry_count == 3
        assert config.enable_hybrid_search is False

    def test_custom_config(self):
        """Test custom configuration values."""
        from llmcore.integration import LLMCoreVectorClientConfig

        config = LLMCoreVectorClientConfig(
            collection_name="my_collection",
            user_id="user_123",
            namespace="project_alpha",
            batch_size=50,
            enable_hybrid_search=True,
            vector_weight=0.8,
        )

        assert config.collection_name == "my_collection"
        assert config.user_id == "user_123"
        assert config.namespace == "project_alpha"
        assert config.batch_size == 50
        assert config.enable_hybrid_search is True
        assert config.vector_weight == 0.8

    def test_get_full_collection_name_without_namespace(self):
        """Test collection name without namespace."""
        from llmcore.integration import LLMCoreVectorClientConfig

        config = LLMCoreVectorClientConfig(collection_name="test")

        assert config.get_full_collection_name() == "test"

    def test_get_full_collection_name_with_namespace(self):
        """Test collection name with namespace prefix."""
        from llmcore.integration import LLMCoreVectorClientConfig

        config = LLMCoreVectorClientConfig(
            collection_name="test",
            namespace="proj",
        )

        assert config.get_full_collection_name() == "proj_test"


# =============================================================================
# TESTS FOR CLIENT INITIALIZATION
# =============================================================================


class TestLLMCoreVectorClientInit:
    """Tests for LLMCoreVectorClient initialization."""

    @pytest.mark.asyncio
    async def test_create_client_basic(self):
        """Test basic client creation."""
        from llmcore.integration import LLMCoreVectorClient, LLMCoreVectorClientConfig

        mock_llmcore = MockLLMCore()
        config = LLMCoreVectorClientConfig(collection_name="test_collection")

        client = await LLMCoreVectorClient.create(mock_llmcore, config)

        assert client is not None
        assert client.collection_name == "test_collection"
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_create_client_with_user_isolation(self):
        """Test client creation with user isolation."""
        from llmcore.integration import LLMCoreVectorClient, LLMCoreVectorClientConfig

        mock_llmcore = MockLLMCore()
        config = LLMCoreVectorClientConfig(
            collection_name="shared_collection",
            user_id="user_123",
        )

        client = await LLMCoreVectorClient.create(mock_llmcore, config)

        assert client.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_create_client_default_config(self):
        """Test client creation with default configuration."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()

        client = await LLMCoreVectorClient.create(mock_llmcore)

        assert client.collection_name == "codebase_default"


# =============================================================================
# TESTS FOR ADD_CHUNKS OPERATION
# =============================================================================


class TestLLMCoreVectorClientAddChunks:
    """Tests for add_chunks operation."""

    @pytest.mark.asyncio
    async def test_add_chunks_basic(self):
        """Test basic add_chunks operation."""
        from llmcore.integration import LLMCoreVectorClient, LLMCoreVectorClientConfig

        mock_llmcore = MockLLMCore()
        config = LLMCoreVectorClientConfig(collection_name="test")
        client = await LLMCoreVectorClient.create(mock_llmcore, config)

        chunks = [
            MockChunk.create(content="chunk 1", file_path="/a.py"),
            MockChunk.create(content="chunk 2", file_path="/b.py"),
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        # This should not raise
        client.add_chunks(chunks, embeddings)

        # Verify documents were stored
        storage = mock_llmcore._storage_manager.vector_storage
        assert len(storage.collections.get("test", {})) == 2

    @pytest.mark.asyncio
    async def test_add_chunks_empty_list(self):
        """Test add_chunks with empty lists."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        # Should not raise, just log warning
        client.add_chunks([], [])

    @pytest.mark.asyncio
    async def test_add_chunks_mismatched_lengths(self):
        """Test add_chunks with mismatched chunk/embedding counts."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        chunks = [MockChunk.create(content="chunk")]
        embeddings = [[0.1], [0.2]]  # More embeddings than chunks

        # Should not raise, just skip
        client.add_chunks(chunks, embeddings)


# =============================================================================
# TESTS FOR QUERY OPERATION
# =============================================================================


class TestLLMCoreVectorClientQuery:
    """Tests for query operation."""

    @pytest.mark.asyncio
    async def test_query_basic(self):
        """Test basic query operation."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        # Add some documents first
        storage = mock_llmcore._storage_manager.vector_storage
        storage.collections["codebase_default"] = {
            "doc_1": MockContextDocument(
                id="doc_1",
                content="Test content 1",
                metadata={"file_path": "/test.py"},
                score=0.9,
            ),
        }

        query_embedding = [0.5, 0.5]
        results = client.query(query_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "doc_1"
        assert "document" in results[0]
        assert "metadata" in results[0]

    @pytest.mark.asyncio
    async def test_query_with_filter(self):
        """Test query with metadata filter."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        query_embedding = [0.5]
        where_filter = {"repo_name": "test-repo"}

        # Should not raise
        results = client.query(query_embedding, top_k=10, where_filter=where_filter)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_empty_collection(self):
        """Test query on empty collection."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        query_embedding = [0.5]
        results = client.query(query_embedding, top_k=5)

        assert results == []


# =============================================================================
# TESTS FOR PROTOCOL COMPLIANCE
# =============================================================================


class TestProtocolCompliance:
    """Tests for VectorClientProtocol compliance."""

    @pytest.mark.asyncio
    async def test_implements_protocol(self):
        """Test that LLMCoreVectorClient implements VectorClientProtocol."""
        from llmcore.integration import LLMCoreVectorClient, VectorClientProtocol

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        # Check protocol compliance via isinstance
        assert isinstance(client, VectorClientProtocol)

    @pytest.mark.asyncio
    async def test_has_required_methods(self):
        """Test that client has all required protocol methods."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        assert hasattr(client, "add_chunks")
        assert callable(client.add_chunks)

        assert hasattr(client, "query")
        assert callable(client.query)


# =============================================================================
# TESTS FOR EXTENDED FEATURES
# =============================================================================


class TestExtendedFeatures:
    """Tests for extended features beyond ChromaDB compatibility."""

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Test hybrid search feature."""
        from llmcore.integration import LLMCoreVectorClient, LLMCoreVectorClientConfig

        mock_llmcore = MockLLMCore()

        # Add hybrid_search to mock storage
        async def mock_hybrid_search(*args, **kwargs):
            return []

        mock_llmcore._storage_manager.vector_storage.hybrid_search = mock_hybrid_search

        config = LLMCoreVectorClientConfig(enable_hybrid_search=True)
        client = await LLMCoreVectorClient.create(mock_llmcore, config)

        # Should return results (empty in this case)
        results = await client.hybrid_search(
            query_text="search term",
            query_embedding=[0.5],
            top_k=10,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_count_operation(self):
        """Test count operation."""
        from llmcore.integration import LLMCoreVectorClient

        mock_llmcore = MockLLMCore()
        client = await LLMCoreVectorClient.create(mock_llmcore)

        # Add some documents
        storage = mock_llmcore._storage_manager.vector_storage
        storage.collections["codebase_default"] = {
            f"doc_{i}": MockContextDocument(
                id=f"doc_{i}",
                content=f"Content {i}",
                metadata={},
            )
            for i in range(5)
        }

        # Add get_collection_info to mock
        async def mock_get_info(name, context=None):
            from dataclasses import dataclass

            @dataclass
            class Info:
                document_count: int = 5

            return Info()

        storage.get_collection_info = mock_get_info

        count = client.count()
        assert count == 5


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
