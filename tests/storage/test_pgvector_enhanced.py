# tests/storage/test_pgvector_enhanced.py
"""
Tests for Enhanced PgVector Storage (Phase 2 - NEXUS).

Tests cover:
- Initialization and configuration
- HNSW index optimization
- Document CRUD operations
- Batch upsert operations
- Similarity search with filters
- Hybrid search (vector + full-text)
- Multi-user isolation
- Collection management
- Connection pool lifecycle

Integration tests require PostgreSQL with pgvector extension.
Configure via environment variables:
    LLMCORE_TEST_PG_HOST: PostgreSQL host
    LLMCORE_TEST_PG_PORT: PostgreSQL port
    LLMCORE_TEST_PG_USER: PostgreSQL user
    LLMCORE_TEST_PG_PASSWORD: PostgreSQL password
    LLMCORE_TEST_PG_DATABASE: PostgreSQL database
"""

import asyncio
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import from installed llmcore package first, then fallback to direct path
try:
    from llmcore.storage.abstraction import (
        StorageContext,
        HNSWConfig,
        VectorSearchConfig,
        BatchConfig,
        BatchResult,
    )
    ABSTRACTION_AVAILABLE = True
except ImportError:
    # Fallback: add storage module to path for direct imports
    _storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
    if str(_storage_path) not in sys.path:
        sys.path.insert(0, str(_storage_path))
    try:
        from abstraction import (
            StorageContext,
            HNSWConfig,
            VectorSearchConfig,
            BatchConfig,
            BatchResult,
        )
        ABSTRACTION_AVAILABLE = True
    except ImportError:
        ABSTRACTION_AVAILABLE = False

# Try to import pgvector_enhanced
PGVECTOR_ENHANCED_AVAILABLE = False
try:
    from llmcore.storage.pgvector_enhanced import (
        EnhancedPgVectorStorage,
        CollectionInfo,
        HybridSearchResult,
        DEFAULT_VECTORS_TABLE,
        DEFAULT_COLLECTIONS_TABLE,
    )
    PGVECTOR_ENHANCED_AVAILABLE = True
except ImportError as e:
    # Try direct import if llmcore not installed
    try:
        _storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
        if str(_storage_path) not in sys.path:
            sys.path.insert(0, str(_storage_path))
        from pgvector_enhanced import (
            EnhancedPgVectorStorage,
            CollectionInfo,
            HybridSearchResult,
            DEFAULT_VECTORS_TABLE,
            DEFAULT_COLLECTIONS_TABLE,
        )
        PGVECTOR_ENHANCED_AVAILABLE = True
    except ImportError as e2:
        # Module not available - tests will skip
        EnhancedPgVectorStorage = None
        CollectionInfo = None
        HybridSearchResult = None
        DEFAULT_VECTORS_TABLE = "vectors"
        DEFAULT_COLLECTIONS_TABLE = "vector_collections"

        # Print import error for debugging
        import warnings
        warnings.warn(f"pgvector_enhanced not available: {e2}")


# =============================================================================
# MOCK MODELS (in case llmcore.models not available)
# =============================================================================

@dataclass
class MockContextDocument:
    """Mock ContextDocument for testing."""
    id: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: Optional[float] = None


# Try to import real ContextDocument, fallback to mock
try:
    from llmcore.models import ContextDocument
except ImportError:
    ContextDocument = MockContextDocument  # type: ignore


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    pool.connection = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    random.seed(42)
    return [random.random() for _ in range(384)]


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    random.seed(42)
    docs = []
    for i in range(5):
        docs.append(ContextDocument(
            id=f"doc_{i}",
            content=f"This is test document number {i} with content about topic{i}.",
            metadata={"source": "test", "index": i, "category": f"cat_{i % 2}"},
            embedding=[random.random() for _ in range(384)],
        ))
    return docs


@pytest.fixture
def user_context():
    """Create a user storage context."""
    return StorageContext(user_id="test_user_123", namespace="test")


@pytest.fixture
def system_context():
    """Create a system storage context."""
    return StorageContext.system_context()


# =============================================================================
# UNIT TESTS (No database required)
# =============================================================================

# Skip all unit tests if pgvector_enhanced module not available
pytestmark_pgvector = pytest.mark.skipif(
    not PGVECTOR_ENHANCED_AVAILABLE,
    reason="pgvector_enhanced module not available (missing dependencies)"
)


@pytestmark_pgvector
class TestEnhancedPgVectorStorageInit:
    """Tests for initialization without database."""

    def test_default_attributes(self):
        """Test default attribute values."""
        storage = EnhancedPgVectorStorage()
        assert storage._pool is None
        assert storage._vectors_table == DEFAULT_VECTORS_TABLE
        assert storage._collections_table == DEFAULT_COLLECTIONS_TABLE
        assert storage._default_vector_dimension == 1536
        assert storage._initialized is False

    def test_hnsw_config_default(self):
        """Test default HNSW configuration."""
        storage = EnhancedPgVectorStorage()
        assert storage._hnsw_config.m == 16
        assert storage._hnsw_config.ef_construction == 64
        assert storage._hnsw_config.ef_search == 40

    def test_capabilities(self):
        """Test backend capabilities."""
        storage = EnhancedPgVectorStorage()
        caps = storage.get_capabilities()
        assert caps.supports_vector_search is True
        assert caps.supports_hybrid_search is True
        assert caps.supports_hnsw_index is True
        assert caps.supports_filtered_search is True


@pytestmark_pgvector
class TestCollectionInfo:
    """Tests for CollectionInfo dataclass."""

    def test_collection_info_creation(self):
        """Test CollectionInfo creation."""
        info = CollectionInfo(
            name="test_collection",
            vector_dimension=384,
            description="Test collection",
            embedding_model_provider="openai",
            embedding_model_name="text-embedding-ada-002",
            document_count=100,
        )
        assert info.name == "test_collection"
        assert info.vector_dimension == 384
        assert info.document_count == 100

    def test_collection_info_to_dict(self):
        """Test CollectionInfo serialization."""
        info = CollectionInfo(
            name="test",
            vector_dimension=384,
            metadata={"key": "value"},
        )
        d = info.to_dict()
        assert d["name"] == "test"
        assert d["vector_dimension"] == 384
        assert d["metadata"] == {"key": "value"}


@pytestmark_pgvector
class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""

    def test_hybrid_result_creation(self):
        """Test HybridSearchResult creation."""
        doc = ContextDocument(id="doc_1", content="test")
        result = HybridSearchResult(
            document=doc,
            vector_score=0.8,
            text_score=0.6,
            combined_score=0.72,
            rank=1,
        )
        assert result.document.id == "doc_1"
        assert result.vector_score == 0.8
        assert result.text_score == 0.6
        assert result.combined_score == 0.72
        assert result.rank == 1


@pytestmark_pgvector
class TestStorageContextIntegration:
    """Tests for StorageContext usage with storage."""

    def test_user_context_collection_naming(self, user_context):
        """Test collection naming with user context."""
        name = user_context.get_collection_name("documents")
        assert name == "test_documents"

    def test_system_context_no_namespace(self, system_context):
        """Test system context has no namespace."""
        name = system_context.get_collection_name("documents")
        assert name == "documents"

    def test_context_write_validation(self, user_context):
        """Test context write validation passes for user context."""
        user_context.validate_for_write()  # Should not raise

    def test_context_write_validation_readonly(self):
        """Test context write validation fails for readonly."""
        ctx = StorageContext(user_id="user", read_only=True)
        with pytest.raises(PermissionError):
            ctx.validate_for_write()


# =============================================================================
# MOCK-BASED TESTS (Simulate database behavior)
# =============================================================================

@pytestmark_pgvector
@pytest.mark.asyncio
class TestInitializationMocked:
    """Tests for initialization with mocked dependencies."""

    async def test_missing_psycopg_raises(self):
        """Test missing psycopg raises ConfigError."""
        storage = EnhancedPgVectorStorage()

        # Use the module path where EnhancedPgVectorStorage was imported from
        module_path = EnhancedPgVectorStorage.__module__

        with patch.dict('sys.modules', {'psycopg': None}):
            # Force psycopg_available to False
            with patch(f'{module_path}.psycopg_available', False):
                with pytest.raises(Exception):  # ConfigError
                    await storage.initialize({"db_url": "postgresql://localhost/test"})

    async def test_missing_db_url_raises(self):
        """Test missing db_url raises ConfigError."""
        storage = EnhancedPgVectorStorage()

        # Use the module path where EnhancedPgVectorStorage was imported from
        module_path = EnhancedPgVectorStorage.__module__

        # Mock psycopg as available
        with patch(f'{module_path}.psycopg_available', True):
            with patch(f'{module_path}.pgvector_available', True):
                with pytest.raises(Exception):  # ConfigError
                    await storage.initialize({})  # No db_url


@pytestmark_pgvector
@pytest.mark.asyncio
class TestAddDocumentsMocked:
    """Tests for document operations with mocked pool."""

    async def test_empty_documents_returns_empty(self):
        """Test adding empty list returns empty."""
        storage = EnhancedPgVectorStorage()
        storage._pool = MagicMock()
        storage._initialized = True

        result = await storage.add_documents([])
        assert result == []

    async def test_document_validation(self, sample_documents):
        """Test document validation during add."""
        storage = EnhancedPgVectorStorage()
        storage._initialized = True

        # Document without embedding should fail
        invalid_doc = ContextDocument(id="no_embed", content="test")

        with pytest.raises(Exception):  # VectorStorageError
            await storage.add_documents([invalid_doc])


@pytestmark_pgvector
@pytest.mark.asyncio
class TestBatchUpsertMocked:
    """Tests for batch upsert with mocked operations."""

    async def test_empty_batch_returns_empty_result(self):
        """Test empty batch returns empty result."""
        storage = EnhancedPgVectorStorage()
        storage._pool = MagicMock()
        storage._initialized = True

        result = await storage.batch_upsert_documents([])
        assert isinstance(result, BatchResult)
        assert result.total == 0
        assert result.success_count == 0

    async def test_batch_config_default(self, sample_documents):
        """Test batch uses default config."""
        storage = EnhancedPgVectorStorage()
        storage._initialized = True
        storage.add_documents = AsyncMock(return_value=[d.id for d in sample_documents])

        result = await storage.batch_upsert_documents(sample_documents)

        assert result.total == 5
        assert result.success_count == 5


@pytestmark_pgvector
class TestSearchConfigMocked:
    """Tests for search configuration."""

    def test_vector_search_config_distance_ops(self):
        """Test distance operator mapping."""
        config = VectorSearchConfig(distance_metric="cosine")
        assert config.get_distance_operator() == "<=>"

        config = VectorSearchConfig(distance_metric="euclidean")
        assert config.get_distance_operator() == "<->"

        config = VectorSearchConfig(distance_metric="inner_product")
        assert config.get_distance_operator() == "<#>"


# =============================================================================
# POSTGRESQL INTEGRATION TESTS (Require database)
# =============================================================================

# These tests require a PostgreSQL instance with pgvector extension
# Skip if LLMCORE_SKIP_PG_TESTS=1 or database unavailable

import os

def _get_pg_config() -> Dict[str, Any]:
    """Get PostgreSQL configuration from environment variables."""
    return {
        "host": os.environ.get("LLMCORE_TEST_PG_HOST", "localhost"),
        "port": int(os.environ.get("LLMCORE_TEST_PG_PORT", "5432")),
        "user": os.environ.get("LLMCORE_TEST_PG_USER", "postgres"),
        "password": os.environ.get("LLMCORE_TEST_PG_PASSWORD", "postgres"),
        "database": os.environ.get("LLMCORE_TEST_PG_DATABASE", "llmcore_test"),
    }

def _get_pg_url() -> str:
    """Build PostgreSQL connection URL from environment variables."""
    config = _get_pg_config()
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )

def _should_skip_pg_tests() -> bool:
    """Check if PostgreSQL tests should be skipped."""
    skip = os.environ.get("LLMCORE_SKIP_PG_TESTS", "").lower()
    return skip in ("1", "true", "yes", "on")

# Try to import from conftest, fallback to local functions
try:
    from conftest import requires_postgres, requires_pgvector, get_pg_url, should_skip_pg_tests
except ImportError:
    # Define fallbacks if conftest not loaded
    requires_postgres = pytest.mark.skipif(_should_skip_pg_tests(), reason="PostgreSQL tests disabled")
    requires_pgvector = pytest.mark.skipif(_should_skip_pg_tests(), reason="pgvector tests disabled")
    get_pg_url = _get_pg_url
    should_skip_pg_tests = _should_skip_pg_tests


@pytestmark_pgvector
@pytest.mark.requires_postgres
@pytest.mark.requires_pgvector
@pytest.mark.asyncio
class TestPgVectorIntegration:
    """
    Integration tests requiring PostgreSQL with pgvector.

    These tests create real database tables and perform actual operations.
    Ensure test database is clean before running.
    """

    @pytest.fixture
    async def storage(self):
        """Create and initialize storage for testing."""
        storage = EnhancedPgVectorStorage()
        config = {
            "db_url": get_pg_url(),
            "vectors_table_name": "test_vectors",
            "collections_table_name": "test_vector_collections",
            "default_collection": "test_default",
            "default_vector_dimension": 384,
            "hnsw_m": 8,
            "hnsw_ef_construction": 32,
            "hnsw_ef_search": 20,
            "min_pool_size": 1,
            "max_pool_size": 3,
        }

        try:
            await storage.initialize(config)
            yield storage
        finally:
            # Cleanup tables
            if storage._pool:
                async with storage._pool.connection() as conn:
                    async with conn.transaction():
                        await conn.execute("DROP TABLE IF EXISTS test_vectors CASCADE")
                        await conn.execute("DROP TABLE IF EXISTS test_vector_collections CASCADE")
            await storage.close()

    async def test_initialization(self, storage):
        """Test storage initializes correctly."""
        assert storage._initialized is True
        assert storage._pool is not None

    async def test_health_check(self, storage):
        """Test health check passes."""
        is_healthy = await storage.health_check()
        assert is_healthy is True

    async def test_create_collection(self, storage):
        """Test creating a collection."""
        info = await storage.create_collection(
            name="test_new_collection",
            dimension=384,
            description="Test collection",
            embedding_model_provider="test",
            embedding_model_name="test-model",
        )

        assert info.name == "test_new_collection"
        assert info.vector_dimension == 384
        assert info.description == "Test collection"

    async def test_list_collections(self, storage):
        """Test listing collections."""
        # Create a collection first
        await storage.create_collection(name="list_test_col", dimension=384)

        collections = await storage.list_collection_names()
        assert "list_test_col" in collections

    async def test_get_collection_metadata(self, storage):
        """Test getting collection metadata."""
        await storage.create_collection(
            name="meta_test_col",
            dimension=384,
            description="Metadata test",
        )

        metadata = await storage.get_collection_metadata("meta_test_col")
        assert metadata is not None
        assert metadata["name"] == "meta_test_col"
        assert metadata["embedding_dimension"] == 384
        assert metadata["description"] == "Metadata test"

    async def test_add_documents(self, storage, sample_documents, user_context):
        """Test adding documents."""
        # Create collection
        await storage.create_collection(name="add_test", dimension=384)

        # Add documents
        ids = await storage.add_documents(
            sample_documents,
            collection_name="add_test",
            context=user_context,
        )

        assert len(ids) == len(sample_documents)
        assert all(d.id in ids for d in sample_documents)

    async def test_get_document(self, storage, sample_documents, user_context):
        """Test getting a single document."""
        await storage.create_collection(name="get_test", dimension=384)
        await storage.add_documents(sample_documents[:1], collection_name="get_test", context=user_context)

        doc = await storage.get_document(
            sample_documents[0].id,
            collection_name="get_test",
            context=user_context,
        )

        assert doc is not None
        assert doc.id == sample_documents[0].id
        assert doc.content == sample_documents[0].content

    async def test_delete_documents(self, storage, sample_documents, user_context):
        """Test deleting documents."""
        await storage.create_collection(name="delete_test", dimension=384)
        await storage.add_documents(sample_documents, collection_name="delete_test", context=user_context)

        # Delete first two documents
        deleted = await storage.delete_documents(
            [sample_documents[0].id, sample_documents[1].id],
            collection_name="delete_test",
            context=user_context,
        )

        assert deleted is True

        # Verify deletion
        doc = await storage.get_document(sample_documents[0].id, collection_name="delete_test")
        assert doc is None

    async def test_similarity_search(self, storage, sample_documents, user_context):
        """Test similarity search."""
        await storage.create_collection(name="search_test", dimension=384)
        await storage.add_documents(sample_documents, collection_name="search_test", context=user_context)

        # Search using first document's embedding
        results = await storage.similarity_search(
            query_embedding=sample_documents[0].embedding,
            k=3,
            collection_name="search_test",
            context=user_context,
        )

        assert len(results) == 3
        # First result should be the same document (highest similarity)
        assert results[0].id == sample_documents[0].id
        # All results should have scores
        assert all(r.score is not None for r in results)

    async def test_similarity_search_with_filter(self, storage, sample_documents, user_context):
        """Test similarity search with metadata filter."""
        await storage.create_collection(name="filter_test", dimension=384)
        await storage.add_documents(sample_documents, collection_name="filter_test", context=user_context)

        # Search with category filter
        config = VectorSearchConfig(
            k=10,
            filter_metadata={"category": "cat_0"},
        )

        results = await storage.similarity_search(
            query_embedding=sample_documents[0].embedding,
            k=10,
            collection_name="filter_test",
            context=user_context,
            search_config=config,
        )

        # Only documents with category=cat_0 should be returned
        assert all(r.metadata.get("category") == "cat_0" for r in results)

    async def test_batch_upsert(self, storage, sample_documents, user_context):
        """Test batch upsert operation."""
        await storage.create_collection(name="batch_test", dimension=384)

        config = BatchConfig(chunk_size=2, on_error="collect")
        result = await storage.batch_upsert_documents(
            sample_documents,
            collection_name="batch_test",
            context=user_context,
            batch_config=config,
        )

        assert result.total == 5
        assert result.success_count == 5
        assert result.failure_count == 0
        assert result.duration_ms > 0

    async def test_user_isolation(self, storage, sample_documents):
        """Test user isolation in queries."""
        await storage.create_collection(name="isolation_test", dimension=384)

        # Add documents as user_1
        ctx1 = StorageContext(user_id="user_1")
        await storage.add_documents(sample_documents[:2], collection_name="isolation_test", context=ctx1)

        # Add documents as user_2
        ctx2 = StorageContext(user_id="user_2")
        await storage.add_documents(sample_documents[2:], collection_name="isolation_test", context=ctx2)

        # Search as user_1 - should only see their documents
        results = await storage.similarity_search(
            query_embedding=sample_documents[0].embedding,
            k=10,
            collection_name="isolation_test",
            context=ctx1,
        )

        # User 1 should see their own docs plus any without user_id
        result_ids = {r.id for r in results}
        assert sample_documents[0].id in result_ids
        assert sample_documents[1].id in result_ids

    async def test_hybrid_search(self, storage, sample_documents, user_context):
        """Test hybrid search combining vector and text."""
        await storage.create_collection(name="hybrid_test", dimension=384)
        await storage.add_documents(sample_documents, collection_name="hybrid_test", context=user_context)

        results = await storage.hybrid_search(
            query_text="test document topic0",
            query_embedding=sample_documents[0].embedding,
            k=3,
            collection_name="hybrid_test",
            context=user_context,
            vector_weight=0.7,
        )

        assert len(results) <= 3
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert all(r.combined_score > 0 for r in results)

    async def test_collection_info_with_count(self, storage, sample_documents, user_context):
        """Test collection info includes document count."""
        await storage.create_collection(name="info_test", dimension=384)
        await storage.add_documents(sample_documents, collection_name="info_test", context=user_context)

        info = await storage.get_collection_info("info_test", context=user_context)

        assert info is not None
        assert info.document_count == len(sample_documents)

    async def test_delete_collection(self, storage):
        """Test deleting a collection."""
        await storage.create_collection(name="delete_col_test", dimension=384)

        # Verify exists
        collections = await storage.list_collection_names()
        assert "delete_col_test" in collections

        # Delete
        deleted = await storage.delete_collection("delete_col_test", force=True)
        assert deleted is True

        # Verify deleted
        collections = await storage.list_collection_names()
        assert "delete_col_test" not in collections

    async def test_upsert_updates_existing(self, storage, sample_documents, user_context):
        """Test upsert updates existing documents."""
        await storage.create_collection(name="upsert_test", dimension=384)

        # Add original document
        await storage.add_documents(sample_documents[:1], collection_name="upsert_test", context=user_context)

        # Update with same ID but different content
        updated_doc = ContextDocument(
            id=sample_documents[0].id,
            content="Updated content",
            metadata={"updated": True},
            embedding=sample_documents[0].embedding,
        )
        await storage.add_documents([updated_doc], collection_name="upsert_test", context=user_context)

        # Verify update
        doc = await storage.get_document(sample_documents[0].id, collection_name="upsert_test")
        assert doc is not None
        assert doc.content == "Updated content"
        assert doc.metadata.get("updated") is True


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytestmark_pgvector
@pytest.mark.requires_postgres
@pytest.mark.requires_pgvector
@pytest.mark.asyncio
class TestPgVectorPerformance:
    """Performance-focused tests for pgvector operations."""

    @pytest.fixture
    async def storage(self):
        """Create storage for performance testing."""
        storage = EnhancedPgVectorStorage()
        config = {
            "db_url": get_pg_url(),
            "vectors_table_name": "perf_vectors",
            "collections_table_name": "perf_collections",
            "default_collection": "perf_test",
            "default_vector_dimension": 384,
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "min_pool_size": 2,
            "max_pool_size": 5,
        }

        try:
            await storage.initialize(config)
            yield storage
        finally:
            if storage._pool:
                async with storage._pool.connection() as conn:
                    async with conn.transaction():
                        await conn.execute("DROP TABLE IF EXISTS perf_vectors CASCADE")
                        await conn.execute("DROP TABLE IF EXISTS perf_collections CASCADE")
            await storage.close()

    async def test_batch_insert_performance(self, storage):
        """Test batch insert handles many documents."""
        random.seed(42)

        # Generate 100 documents
        docs = [
            ContextDocument(
                id=f"perf_doc_{i}",
                content=f"Performance test document {i} with content",
                metadata={"batch": i // 10},
                embedding=[random.random() for _ in range(384)],
            )
            for i in range(100)
        ]

        await storage.create_collection(name="perf_batch", dimension=384)

        result = await storage.batch_upsert_documents(
            docs,
            collection_name="perf_batch",
            batch_config=BatchConfig(chunk_size=20),
        )

        assert result.success_count == 100
        assert result.duration_ms < 30000  # Should complete in 30s


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytestmark_pgvector
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_hnsw_config_boundary_values(self):
        """Test HNSW config at boundary values."""
        # Minimum valid values
        config = HNSWConfig(m=2, ef_construction=4, ef_search=1)
        assert config.m == 2

        # Maximum valid values
        config = HNSWConfig(m=100, ef_construction=1000, ef_search=1000)
        assert config.m == 100

    def test_vector_search_config_edge_values(self):
        """Test VectorSearchConfig edge values."""
        # Minimum k
        config = VectorSearchConfig(k=1)
        assert config.k == 1

        # Score boundaries
        config = VectorSearchConfig(min_score=0.0)
        assert config.min_score == 0.0

        config = VectorSearchConfig(min_score=1.0)
        assert config.min_score == 1.0

    def test_storage_context_empty_namespace(self):
        """Test context with empty string namespace."""
        ctx = StorageContext(namespace="")
        # Empty string should be treated as no namespace
        name = ctx.get_collection_name("docs")
        assert name == "docs"

    def test_batch_result_all_failed(self):
        """Test BatchResult with all failures."""
        result = BatchResult(
            successful=[],
            failed=[("doc1", "err1"), ("doc2", "err2")],
            total=2,
            duration_ms=100,
        )
        assert result.success_count == 0
        assert result.failure_count == 2
        assert result.success_rate == 0.0


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
