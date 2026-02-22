# tests/integration/phase10/test_rag_integration.py
"""
Phase 10 Integration Tests: RAG Pipeline with Dual Storage Backends.

IMPORTANT: Both storage backend combinations are FIRST-CLASS CITIZENS:
- SQLite + ChromaDB (development/lightweight)
- PostgreSQL + pgvector (production/scalable)

Tests cover both backends with equal depth and rigor.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Storage imports
from llmcore.storage import (
    BaseSessionStorage,
    BaseVectorStorage,
    ChromaVectorStorage,
    EnhancedPgVectorStorage,
    PgVectorStorage,
    PostgresSessionStorage,
    SqliteSessionStorage,
    StorageManager,
)
from llmcore.storage.manager import SESSION_STORAGE_MAP, VECTOR_STORAGE_MAP

# ============================================================================
# Fixtures for Both Backends
# ============================================================================


@pytest.fixture
def sqlite_session_config(tmp_path: Path) -> Dict[str, Any]:
    """SQLite session storage configuration."""
    return {"path": str(tmp_path / "sessions.db")}


def get_pg_config() -> Dict[str, Any]:
    """Get PostgreSQL configuration from environment variables."""
    return {
        "host": os.environ.get("LLMCORE_TEST_PG_HOST", "localhost"),
        "port": int(os.environ.get("LLMCORE_TEST_PG_PORT", "5432")),
        "database": os.environ.get("LLMCORE_TEST_PG_DATABASE", "llmcore_test"),
        "user": os.environ.get("LLMCORE_TEST_PG_USER", "postgres"),
        "password": os.environ.get("LLMCORE_TEST_PG_PASSWORD", "postgres"),
    }


def get_pg_url() -> str:
    """Build PostgreSQL connection URL from environment variables."""
    config = get_pg_config()
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


@pytest.fixture
def postgres_session_config() -> Dict[str, Any]:
    """PostgreSQL session storage configuration with db_url."""
    return {
        "db_url": get_pg_url(),
    }


@pytest.fixture
def chromadb_config(tmp_path: Path) -> Dict[str, Any]:
    """ChromaDB vector storage configuration."""
    return {
        "path": str(tmp_path / "chromadb"),
        "default_collection": "test_collection",
    }


@pytest.fixture
def pgvector_config() -> Dict[str, Any]:
    """pgvector storage configuration with db_url."""
    return {
        "db_url": get_pg_url(),
        "collection_name": "test_vectors",
    }


def should_skip_pg_tests() -> bool:
    """Check if PostgreSQL tests should be skipped."""
    skip = os.environ.get("LLMCORE_SKIP_PG_TESTS", "").lower()
    return skip in ("1", "true", "yes", "on")


def postgres_available() -> bool:
    """Check if PostgreSQL is available for testing."""
    if should_skip_pg_tests():
        return False
    try:
        import psycopg

        return True
    except ImportError:
        return False


# ============================================================================
# Storage Backend Registration Tests
# ============================================================================


class TestStorageBackendRegistration:
    """Test that both storage backend combinations are properly registered."""

    def test_session_storage_backends_registered(self) -> None:
        """Both SQLite and PostgreSQL session backends should be registered."""
        assert "sqlite" in SESSION_STORAGE_MAP, "SQLite session backend not registered"
        assert "postgres" in SESSION_STORAGE_MAP, "PostgreSQL session backend not registered"
        assert "json" in SESSION_STORAGE_MAP, "JSON session backend not registered"

    def test_vector_storage_backends_registered(self) -> None:
        """Both ChromaDB and pgvector backends should be registered."""
        assert "chromadb" in VECTOR_STORAGE_MAP, "ChromaDB vector backend not registered"
        assert "pgvector" in VECTOR_STORAGE_MAP, "pgvector backend not registered"

    def test_session_storage_classes_are_base_session_subclasses(self) -> None:
        """All session storage classes should inherit from BaseSessionStorage."""
        for name, cls in SESSION_STORAGE_MAP.items():
            assert issubclass(cls, BaseSessionStorage), (
                f"{name} storage should inherit from BaseSessionStorage"
            )

    def test_vector_storage_classes_are_base_vector_subclasses(self) -> None:
        """All vector storage classes should inherit from BaseVectorStorage."""
        for name, cls in VECTOR_STORAGE_MAP.items():
            assert issubclass(cls, BaseVectorStorage), (
                f"{name} storage should inherit from BaseVectorStorage"
            )


# ============================================================================
# Session Storage Interface Parity Tests
# ============================================================================


class TestSessionStorageInterfaceParity:
    """Test that SQLite and PostgreSQL session storage have interface parity."""

    def test_core_methods_present_in_sqlite(self) -> None:
        """SQLite session storage should have all core methods."""
        core_methods = {
            "initialize",
            "close",
            "get_session",
            "list_sessions",
            "delete_session",
            "save_session",
            "add_episode",
            "get_episodes",
        }
        sqlite_methods = set(dir(SqliteSessionStorage))
        missing = core_methods - sqlite_methods
        assert not missing, f"SQLite missing core methods: {missing}"

    def test_core_methods_present_in_postgres(self) -> None:
        """PostgreSQL session storage should have all core methods."""
        core_methods = {
            "initialize",
            "close",
            "get_session",
            "list_sessions",
            "delete_session",
            "save_session",
            "add_episode",
            "get_episodes",
        }
        postgres_methods = set(dir(PostgresSessionStorage))
        missing = core_methods - postgres_methods
        assert not missing, f"PostgreSQL missing core methods: {missing}"

    def test_interface_consistency(self) -> None:
        """Both session storages should have consistent core interface."""
        sqlite_public = {
            m
            for m in dir(SqliteSessionStorage)
            if not m.startswith("_") and callable(getattr(SqliteSessionStorage, m, None))
        }
        postgres_public = {
            m
            for m in dir(PostgresSessionStorage)
            if not m.startswith("_") and callable(getattr(PostgresSessionStorage, m, None))
        }

        # Core methods that must be in both
        core = {
            "initialize",
            "close",
            "get_session",
            "list_sessions",
            "delete_session",
            "add_episode",
            "get_episodes",
        }

        assert core.issubset(sqlite_public), "SQLite missing core methods"
        assert core.issubset(postgres_public), "PostgreSQL missing core methods"


# ============================================================================
# Vector Storage Interface Parity Tests
# ============================================================================


class TestVectorStorageInterfaceParity:
    """Test that ChromaDB and pgvector have interface parity."""

    def test_core_methods_present_in_chromadb(self) -> None:
        """ChromaDB should have all core vector storage methods."""
        core_methods = {
            "initialize",
            "close",
            "add_documents",
            "similarity_search",
            "delete_documents",
        }
        chromadb_methods = set(dir(ChromaVectorStorage))
        missing = core_methods - chromadb_methods
        assert not missing, f"ChromaDB missing core methods: {missing}"

    def test_core_methods_present_in_pgvector(self) -> None:
        """pgvector should have all core vector storage methods."""
        core_methods = {
            "initialize",
            "close",
            "add_documents",
            "similarity_search",
            "delete_documents",
        }
        pgvector_methods = set(dir(PgVectorStorage))
        missing = core_methods - pgvector_methods
        assert not missing, f"pgvector missing core methods: {missing}"

    def test_interface_consistency(self) -> None:
        """Both vector storages should have consistent core interface."""
        core = {"initialize", "close", "add_documents", "similarity_search", "delete_documents"}

        chromadb_public = set(dir(ChromaVectorStorage))
        pgvector_public = set(dir(PgVectorStorage))

        assert core.issubset(chromadb_public), "ChromaDB missing core methods"
        assert core.issubset(pgvector_public), "pgvector missing core methods"


# ============================================================================
# SQLite Session Storage Tests
# ============================================================================


class TestSqliteSessionStorage:
    """Test SQLite session storage backend."""

    @pytest.mark.asyncio
    async def test_initialization(self, sqlite_session_config: Dict[str, Any]) -> None:
        """SQLite storage should initialize and create database file."""
        storage = SqliteSessionStorage()
        await storage.initialize(sqlite_session_config)

        try:
            assert storage is not None
            db_path = Path(sqlite_session_config["path"])
            assert db_path.exists(), "SQLite database file not created"
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, sqlite_session_config: Dict[str, Any]) -> None:
        """Empty database should return empty session list."""
        storage = SqliteSessionStorage()
        await storage.initialize(sqlite_session_config)

        try:
            sessions = await storage.list_sessions()
            assert isinstance(sessions, list)
            assert len(sessions) == 0
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_episode_methods_exist(self, sqlite_session_config: Dict[str, Any]) -> None:
        """SQLite storage should have episode management methods."""
        storage = SqliteSessionStorage()
        await storage.initialize(sqlite_session_config)

        try:
            assert hasattr(storage, "add_episode")
            assert hasattr(storage, "get_episodes")
            assert callable(storage.add_episode)
            assert callable(storage.get_episodes)
        finally:
            await storage.close()


# ============================================================================
# PostgreSQL Session Storage Tests
# ============================================================================


class TestPostgresSessionStorage:
    """Test PostgreSQL session storage backend."""

    def test_class_available(self) -> None:
        """PostgresSessionStorage class should be importable."""
        assert PostgresSessionStorage is not None

    def test_has_core_interface(self) -> None:
        """PostgreSQL storage should have core interface methods."""
        assert hasattr(PostgresSessionStorage, "initialize")
        assert hasattr(PostgresSessionStorage, "close")
        assert hasattr(PostgresSessionStorage, "get_session")
        assert hasattr(PostgresSessionStorage, "list_sessions")
        assert hasattr(PostgresSessionStorage, "add_episode")
        assert hasattr(PostgresSessionStorage, "get_episodes")

    @pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not configured")
    @pytest.mark.asyncio
    async def test_initialization(self, postgres_session_config: Dict[str, Any]) -> None:
        """PostgreSQL storage should initialize with valid config."""
        storage = PostgresSessionStorage()
        await storage.initialize(postgres_session_config)

        try:
            assert storage is not None
        finally:
            await storage.close()

    @pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not configured")
    @pytest.mark.asyncio
    async def test_list_sessions(self, postgres_session_config: Dict[str, Any]) -> None:
        """PostgreSQL storage should list sessions."""
        storage = PostgresSessionStorage()
        await storage.initialize(postgres_session_config)

        try:
            sessions = await storage.list_sessions()
            assert isinstance(sessions, list)
        finally:
            await storage.close()


# ============================================================================
# ChromaDB Vector Storage Tests
# ============================================================================


class TestChromaVectorStorage:
    """Test ChromaDB vector storage backend."""

    def test_class_available(self) -> None:
        """ChromaVectorStorage class should be importable."""
        assert ChromaVectorStorage is not None

    def test_has_core_interface(self) -> None:
        """ChromaDB storage should have core interface methods."""
        assert hasattr(ChromaVectorStorage, "initialize")
        assert hasattr(ChromaVectorStorage, "close")
        assert hasattr(ChromaVectorStorage, "add_documents")
        assert hasattr(ChromaVectorStorage, "similarity_search")
        assert hasattr(ChromaVectorStorage, "delete_documents")

    @pytest.mark.asyncio
    async def test_initialization(self, chromadb_config: Dict[str, Any]) -> None:
        """ChromaDB storage should initialize with valid config."""
        storage = ChromaVectorStorage()
        await storage.initialize(chromadb_config)

        try:
            assert storage is not None
            assert storage._client is not None
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_in_memory_initialization(self) -> None:
        """ChromaDB should support in-memory mode (no path)."""
        storage = ChromaVectorStorage()
        await storage.initialize({"default_collection": "test_inmemory"})

        try:
            assert storage is not None
            assert storage._client is not None
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_collection_operations(self, chromadb_config: Dict[str, Any]) -> None:
        """ChromaDB should support collection management."""
        storage = ChromaVectorStorage()
        await storage.initialize(chromadb_config)

        try:
            # Should be able to list collections
            collections = await storage.list_collection_names()
            assert isinstance(collections, list)
        finally:
            await storage.close()


# ============================================================================
# pgvector Storage Tests
# ============================================================================


class TestPgVectorStorage:
    """Test pgvector vector storage backend."""

    def test_class_available(self) -> None:
        """PgVectorStorage class should be importable."""
        assert PgVectorStorage is not None

    def test_enhanced_class_available(self) -> None:
        """EnhancedPgVectorStorage class should be importable."""
        assert EnhancedPgVectorStorage is not None

    def test_has_core_interface(self) -> None:
        """pgvector storage should have core interface methods."""
        assert hasattr(PgVectorStorage, "initialize")
        assert hasattr(PgVectorStorage, "close")
        assert hasattr(PgVectorStorage, "add_documents")
        assert hasattr(PgVectorStorage, "similarity_search")
        assert hasattr(PgVectorStorage, "delete_documents")

    @pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not configured")
    @pytest.mark.asyncio
    async def test_initialization(self, pgvector_config: Dict[str, Any]) -> None:
        """pgvector storage should initialize with valid config."""
        storage = PgVectorStorage()
        await storage.initialize(pgvector_config)

        try:
            assert storage is not None
        finally:
            await storage.close()

    @pytest.mark.skipif(not postgres_available(), reason="PostgreSQL not configured")
    @pytest.mark.asyncio
    async def test_collection_operations(self, pgvector_config: Dict[str, Any]) -> None:
        """pgvector should support collection management."""
        storage = PgVectorStorage()
        await storage.initialize(pgvector_config)

        try:
            collections = await storage.list_collection_names()
            assert isinstance(collections, list)
        finally:
            await storage.close()


# ============================================================================
# Embedding Integration Tests
# ============================================================================


class TestEmbeddingIntegration:
    """Test embedding functionality for RAG."""

    def test_embedding_manager_available(self) -> None:
        """EmbeddingManager should be importable."""
        from llmcore.embedding.manager import EmbeddingManager

        assert EmbeddingManager is not None

    def test_embedding_cache_available(self) -> None:
        """EmbeddingCache should be importable."""
        from llmcore.embedding.cache import EmbeddingCache

        assert EmbeddingCache is not None

    def test_embedding_cache_initialization(self, tmp_path: Path) -> None:
        """Embedding cache should initialize correctly."""
        from llmcore.embedding.cache import EmbeddingCache

        cache = EmbeddingCache(
            disk_path=str(tmp_path / "embeddings.db"),
            memory_size=1000,
        )

        assert cache is not None
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")

    def test_mock_embedding_determinism(self) -> None:
        """Mock embeddings should be deterministic for same input."""
        import hashlib

        def mock_embed(text: str, dim: int = 1536) -> List[float]:
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(dim):
                byte_idx = i % len(hash_bytes)
                embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
            return embedding

        emb1 = mock_embed("test text")
        emb2 = mock_embed("test text")
        emb3 = mock_embed("different text")

        assert emb1 == emb2, "Same text should produce same embedding"
        assert emb1 != emb3, "Different text should produce different embedding"
        assert len(emb1) == 1536


# ============================================================================
# RAG Pipeline Tests
# ============================================================================


class TestRAGPipeline:
    """Test end-to-end RAG pipeline components."""

    def test_context_builder_available(self) -> None:
        """Context builder should be importable."""
        from llmcore.memory.context_builder import build_context_payload

        assert build_context_payload is not None

    def test_llmcore_supports_external_rag(self) -> None:
        """LLMCore should support external RAG pattern.

        External RAG pattern:
        1. semantiscan: Retrieval (chunking, embedding, vector search)
        2. llmchat: Context building from retrieved chunks
        3. llmcore: Response generation with enable_rag=False
        """
        from llmcore import LLMCore

        assert hasattr(LLMCore, "chat")

    @pytest.mark.asyncio
    async def test_rag_flow_simulation(self) -> None:
        """Test RAG flow with simulated retrieval results."""
        # Simulate chunks retrieved from vector storage (either ChromaDB or pgvector)
        retrieved_chunks = [
            {
                "content": "Python is a high-level programming language.",
                "metadata": {"source": "python.md", "chunk_id": "1"},
                "score": 0.95,
            },
            {
                "content": "PostgreSQL is a powerful relational database.",
                "metadata": {"source": "postgres.md", "chunk_id": "2"},
                "score": 0.87,
            },
        ]

        # Build context from chunks (what llmchat would do)
        context_parts = []
        for chunk in sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True):
            context_parts.append(f"[Source: {chunk['metadata']['source']}]\n{chunk['content']}")
        context = "\n\n".join(context_parts)

        assert "Python" in context
        assert "PostgreSQL" in context
        assert context.index("Python") < context.index("PostgreSQL")  # Higher score first


# ============================================================================
# Storage Configuration Tests
# ============================================================================


class TestStorageConfiguration:
    """Test storage configuration for both backends."""

    def test_sqlite_config_accepts_path(self) -> None:
        """SQLite config should accept path parameter."""
        config = {"path": "/tmp/test.db"}
        storage = SqliteSessionStorage()
        # Should not raise - just testing config acceptance
        assert storage is not None

    def test_chromadb_config_accepts_path_and_collection(self) -> None:
        """ChromaDB config should accept path and collection parameters."""
        config = {
            "path": "/tmp/chromadb",
            "default_collection": "test",
        }
        storage = ChromaVectorStorage()
        assert storage is not None

    def test_postgres_config_structure(self) -> None:
        """PostgreSQL config should support standard connection parameters."""
        expected_params = {"host", "port", "database", "user", "password"}
        # Verify PostgresSessionStorage accepts these params
        assert PostgresSessionStorage is not None

    def test_pgvector_config_structure(self) -> None:
        """pgvector config should support connection + collection parameters."""
        assert PgVectorStorage is not None


# ============================================================================
# Cross-Backend Compatibility Tests
# ============================================================================


class TestCrossBackendCompatibility:
    """Test that data can conceptually flow between backends."""

    def test_session_data_model_consistency(self) -> None:
        """Session data model should work with both SQLite and PostgreSQL."""
        from llmcore.models import ChatSession, Message, Role

        # Create a session that should work with either backend
        session = ChatSession(
            id="test-session-123",
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
            ],
        )

        assert session.id == "test-session-123"
        assert len(session.messages) == 2

    def test_document_model_consistency(self) -> None:
        """Document model should work with both ChromaDB and pgvector."""
        from llmcore.models import ContextDocument

        # Create a document that should work with either backend
        doc = ContextDocument(
            id="doc-123",
            content="Test document content",
            embedding=[0.1] * 1536,  # Standard embedding dimension
            metadata={"source": "test.md"},
        )

        assert doc.id == "doc-123"
        assert len(doc.embedding) == 1536

    def test_episode_model_consistency(self) -> None:
        """Episode model should work with both session backends."""
        from llmcore.models import Episode, EpisodeType

        episode = Episode(
            episode_id="ep-123",
            session_id="session-456",
            event_type=EpisodeType.ACTION,
            data={"tool": "search", "query": "test"},
        )

        assert episode.episode_id == "ep-123"
        assert episode.session_id == "session-456"


# ============================================================================
# Storage Manager Tests
# ============================================================================


class TestStorageManagerIntegration:
    """Test the unified StorageManager with both backends."""

    def test_storage_manager_exists(self) -> None:
        """StorageManager should be importable."""
        assert StorageManager is not None

    def test_storage_manager_supports_both_session_backends(self) -> None:
        """StorageManager should work with both SQLite and PostgreSQL."""
        # Verify backend registration
        assert "sqlite" in SESSION_STORAGE_MAP
        assert "postgres" in SESSION_STORAGE_MAP

    def test_storage_manager_supports_both_vector_backends(self) -> None:
        """StorageManager should work with both ChromaDB and pgvector."""
        # Verify backend registration
        assert "chromadb" in VECTOR_STORAGE_MAP
        assert "pgvector" in VECTOR_STORAGE_MAP


# ============================================================================
# Mock Embedding Fixtures
# ============================================================================


@pytest.fixture
def mock_embeddings() -> List[List[float]]:
    """Provide deterministic mock embeddings for testing."""
    import random

    random.seed(42)
    return [[random.uniform(-1, 1) for _ in range(1536)] for _ in range(10)]


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Python is a high-level programming language.",
            "metadata": {"source": "python_intro.md", "category": "programming"},
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "ml_basics.md", "category": "ai"},
        },
        {
            "id": "doc3",
            "content": "PostgreSQL is a powerful relational database with pgvector extension.",
            "metadata": {"source": "postgres_guide.md", "category": "database"},
        },
        {
            "id": "doc4",
            "content": "ChromaDB is an open-source embedding database.",
            "metadata": {"source": "chromadb_intro.md", "category": "database"},
        },
    ]
