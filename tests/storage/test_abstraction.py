# tests/storage/test_abstraction.py
"""
Tests for Storage Abstraction Layer (Phase 2 - NEXUS).

Tests cover:
- StorageContext: User isolation context management
- QueryBuilder: Safe SQL query construction
- HNSWConfig: HNSW index configuration
- VectorSearchConfig: Search configuration
- BatchConfig/BatchResult: Batch operation handling
- PoolConfig: Connection pool configuration
- Utility functions
"""

import sys
from pathlib import Path

import pytest

# Try to import from installed llmcore package first, then fallback to direct path
try:
    from llmcore.storage.abstraction import (
        CHROMADB_CAPABILITIES,
        POSTGRES_PGVECTOR_CAPABILITIES,
        SQLITE_CAPABILITIES,
        BackendCapabilities,
        BatchConfig,
        BatchResult,
        HNSWConfig,
        IsolationLevel,
        PoolConfig,
        QueryBuilder,
        QueryCondition,
        QueryOperator,
        StorageContext,
        VectorSearchConfig,
        chunk_list,
        execute_with_retry,
    )
except ImportError:
    # Fallback: add storage module to path for direct imports
    _storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
    if str(_storage_path) not in sys.path:
        sys.path.insert(0, str(_storage_path))

    from abstraction import (
        CHROMADB_CAPABILITIES,
        POSTGRES_PGVECTOR_CAPABILITIES,
        SQLITE_CAPABILITIES,
        BatchConfig,
        BatchResult,
        HNSWConfig,
        IsolationLevel,
        PoolConfig,
        QueryBuilder,
        QueryCondition,
        QueryOperator,
        StorageContext,
        VectorSearchConfig,
        chunk_list,
        execute_with_retry,
    )


# =============================================================================
# STORAGE CONTEXT TESTS
# =============================================================================


class TestStorageContext:
    """Tests for StorageContext."""

    def test_default_context(self):
        """Test default context creation."""
        ctx = StorageContext()
        assert ctx.user_id is None
        assert ctx.namespace is None
        assert ctx.metadata == {}
        assert ctx.read_only is False
        assert ctx.require_user is True

    def test_context_with_user(self):
        """Test context with user ID."""
        ctx = StorageContext(user_id="user_123")
        assert ctx.user_id == "user_123"

    def test_context_with_namespace(self):
        """Test context with namespace."""
        ctx = StorageContext(namespace="project_alpha")
        assert ctx.namespace == "project_alpha"

    def test_namespace_sanitization(self):
        """Test that namespace is sanitized."""
        ctx = StorageContext(namespace="my-project@123!")
        # Special chars replaced with underscore
        assert "_" in ctx.namespace or ctx.namespace == "my_project_123_"
        assert "@" not in ctx.namespace
        assert "!" not in ctx.namespace

    def test_namespace_length_limit(self):
        """Test namespace is truncated to 64 chars."""
        long_ns = "a" * 100
        ctx = StorageContext(namespace=long_ns)
        assert len(ctx.namespace) <= 64

    def test_namespace_starting_with_number(self):
        """Test namespace starting with number gets prefix."""
        ctx = StorageContext(namespace="123project")
        assert ctx.namespace.startswith("ns_")

    def test_system_context(self):
        """Test system-level context creation."""
        ctx = StorageContext.system_context()
        assert ctx.user_id is None
        assert ctx.require_user is False

    def test_system_context_with_metadata(self):
        """Test system context with metadata."""
        ctx = StorageContext.system_context(metadata={"request_id": "req_123"})
        assert ctx.metadata["request_id"] == "req_123"

    def test_with_namespace(self):
        """Test creating context with different namespace."""
        ctx = StorageContext(user_id="user_1", namespace="ns1")
        ctx2 = ctx.with_namespace("ns2")

        assert ctx.namespace == "ns1"
        assert ctx2.namespace == "ns2"
        assert ctx2.user_id == "user_1"

    def test_get_collection_name_with_namespace(self):
        """Test collection name prefixing."""
        ctx = StorageContext(namespace="myproject")
        name = ctx.get_collection_name("documents")
        assert name == "myproject_documents"

    def test_get_collection_name_without_namespace(self):
        """Test collection name without namespace."""
        ctx = StorageContext()
        name = ctx.get_collection_name("documents")
        assert name == "documents"

    def test_validate_for_write_readonly(self):
        """Test validation fails for read-only context."""
        ctx = StorageContext(read_only=True)
        with pytest.raises(PermissionError, match="read-only"):
            ctx.validate_for_write()

    def test_validate_for_write_missing_user(self):
        """Test validation fails without required user."""
        ctx = StorageContext(require_user=True)  # No user_id
        with pytest.raises(PermissionError, match="User ID required"):
            ctx.validate_for_write()

    def test_validate_for_write_system_context(self):
        """Test system context can write without user."""
        ctx = StorageContext.system_context()
        ctx.validate_for_write()  # Should not raise


# =============================================================================
# ISOLATION LEVEL TESTS
# =============================================================================


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_values(self):
        """Test isolation level values."""
        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.USER.value == "user"
        assert IsolationLevel.NAMESPACE.value == "namespace"
        assert IsolationLevel.FULL.value == "full"


# =============================================================================
# BACKEND CAPABILITIES TESTS
# =============================================================================


class TestBackendCapabilities:
    """Tests for BackendCapabilities."""

    def test_postgres_capabilities(self):
        """Test PostgreSQL + pgvector capabilities."""
        caps = POSTGRES_PGVECTOR_CAPABILITIES
        assert caps.supports_vector_search is True
        assert caps.supports_hybrid_search is True
        assert caps.supports_hnsw_index is True
        assert caps.supports_full_text_search is True
        assert caps.supports_transactions is True
        assert caps.max_vector_dimension == 16000

    def test_sqlite_capabilities(self):
        """Test SQLite capabilities."""
        caps = SQLITE_CAPABILITIES
        assert caps.supports_vector_search is False
        assert caps.supports_hybrid_search is False
        assert caps.supports_hnsw_index is False
        assert caps.supports_full_text_search is True  # FTS5
        assert caps.supports_transactions is True

    def test_chromadb_capabilities(self):
        """Test ChromaDB capabilities."""
        caps = CHROMADB_CAPABILITIES
        assert caps.supports_vector_search is True
        assert caps.supports_hybrid_search is False
        assert caps.supports_filtered_search is True
        assert caps.supports_transactions is False

    def test_capabilities_str(self):
        """Test capabilities string representation."""
        caps = POSTGRES_PGVECTOR_CAPABILITIES
        s = str(caps)
        assert "vector" in s
        assert "hybrid" in s
        assert "tx" in s


# =============================================================================
# QUERY BUILDER TESTS
# =============================================================================


class TestQueryBuilder:
    """Tests for QueryBuilder."""

    def test_simple_select(self):
        """Test simple SELECT query."""
        query = QueryBuilder("sessions").select("id", "name")
        sql, params = query.build_postgres()
        assert "SELECT id, name FROM sessions" in sql
        assert params == []

    def test_select_all(self):
        """Test SELECT * query."""
        query = QueryBuilder("sessions")
        sql, _ = query.build_postgres()
        assert "SELECT * FROM sessions" in sql

    def test_where_eq(self):
        """Test WHERE equality condition."""
        query = QueryBuilder("sessions").where_eq("user_id", "user_123")
        sql, params = query.build_postgres()
        assert "WHERE user_id = $1" in sql
        assert params == ["user_123"]

    def test_where_multiple_conditions(self):
        """Test multiple WHERE conditions."""
        query = QueryBuilder("sessions").where_eq("user_id", "user_123").where_eq("name", "test")
        sql, params = query.build_postgres()
        assert "WHERE user_id = $1 AND name = $2" in sql
        assert params == ["user_123", "test"]

    def test_where_in(self):
        """Test WHERE IN condition."""
        query = QueryBuilder("sessions").where_in("id", ["a", "b", "c"])
        sql, params = query.build_postgres()
        assert "IN" in sql
        assert "a" in params
        assert "b" in params
        assert "c" in params

    def test_where_null(self):
        """Test WHERE IS NULL condition."""
        query = QueryBuilder("sessions").where_null("deleted_at")
        sql, params = query.build_postgres()
        assert "WHERE deleted_at IS NULL" in sql
        assert params == []

    def test_where_not_null(self):
        """Test WHERE IS NOT NULL condition."""
        query = QueryBuilder("sessions").where_not_null("name")
        sql, params = query.build_postgres()
        assert "WHERE name IS NOT NULL" in sql

    def test_order_by(self):
        """Test ORDER BY clause."""
        query = QueryBuilder("sessions").order_by("created_at", descending=True)
        sql, _ = query.build_postgres()
        assert "ORDER BY created_at DESC" in sql

    def test_order_by_multiple(self):
        """Test multiple ORDER BY columns."""
        query = QueryBuilder("sessions").order_by("user_id").order_by("created_at", descending=True)
        sql, _ = query.build_postgres()
        assert "ORDER BY user_id ASC, created_at DESC" in sql

    def test_limit(self):
        """Test LIMIT clause."""
        query = QueryBuilder("sessions").limit(10)
        sql, _ = query.build_postgres()
        assert "LIMIT 10" in sql

    def test_offset(self):
        """Test OFFSET clause."""
        query = QueryBuilder("sessions").limit(10).offset(20)
        sql, _ = query.build_postgres()
        assert "LIMIT 10" in sql
        assert "OFFSET 20" in sql

    def test_for_update(self):
        """Test FOR UPDATE clause."""
        query = QueryBuilder("sessions").for_update()
        sql, _ = query.build_postgres()
        assert "FOR UPDATE" in sql

    def test_invalid_identifier(self):
        """Test rejection of invalid identifiers."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            QueryBuilder("users; DROP TABLE")

    def test_sqlite_query(self):
        """Test SQLite query generation."""
        query = (
            QueryBuilder("sessions").select("id", "name").where_eq("user_id", "user_123").limit(5)
        )
        sql, params = query.build_sqlite()
        assert "SELECT id, name FROM sessions" in sql
        assert "WHERE user_id = ?" in sql
        assert "LIMIT 5" in sql
        assert params == ["user_123"]

    def test_named_query(self):
        """Test named parameter query generation."""
        query = QueryBuilder("sessions").where_eq("user_id", "user_123").where_eq("name", "test")
        sql, params = query.build_named()
        assert ":user_id" in sql or ":user_id_0" in sql
        assert "user_123" in params.values()

    def test_where_user_with_context(self):
        """Test where_user with StorageContext."""
        ctx = StorageContext(user_id="user_abc")
        query = QueryBuilder("sessions").where_user(ctx)
        sql, params = query.build_postgres()
        assert "user_id = $1" in sql
        assert params == ["user_abc"]

    def test_where_user_without_required_user(self):
        """Test where_user fails without required user."""
        ctx = StorageContext(require_user=True)  # No user_id
        with pytest.raises(ValueError, match="requires user_id"):
            QueryBuilder("sessions").where_user(ctx)

    def test_where_user_system_context(self):
        """Test where_user with system context adds no condition."""
        ctx = StorageContext.system_context()
        query = QueryBuilder("sessions").where_user(ctx)
        sql, params = query.build_postgres()
        assert "user_id" not in sql
        assert params == []

    def test_limit_negative_raises(self):
        """Test negative limit raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            QueryBuilder("sessions").limit(-1)

    def test_offset_negative_raises(self):
        """Test negative offset raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            QueryBuilder("sessions").offset(-5)


class TestQueryCondition:
    """Tests for QueryCondition."""

    def test_eq_condition_postgres(self):
        """Test equality condition for PostgreSQL."""
        cond = QueryCondition(column="user_id", operator=QueryOperator.EQ, value="user_123")
        sql, value = cond.to_sql_postgres(1)
        assert sql == "user_id = $1"
        assert value == "user_123"

    def test_is_null_condition(self):
        """Test IS NULL condition."""
        cond = QueryCondition(column="deleted_at", operator=QueryOperator.IS_NULL, value=None)
        sql, value = cond.to_sql_postgres(1)
        assert sql == "deleted_at IS NULL"
        assert value is None


# =============================================================================
# HNSW CONFIG TESTS
# =============================================================================


class TestHNSWConfig:
    """Tests for HNSWConfig."""

    def test_default_config(self):
        """Test default HNSW configuration."""
        config = HNSWConfig()
        assert config.m == 16
        assert config.ef_construction == 64
        assert config.ef_search == 40

    def test_custom_config(self):
        """Test custom HNSW configuration."""
        config = HNSWConfig(m=32, ef_construction=200, ef_search=100)
        assert config.m == 32
        assert config.ef_construction == 200
        assert config.ef_search == 100

    def test_m_validation_low(self):
        """Test m parameter lower bound."""
        with pytest.raises(ValueError, match="m must be between"):
            HNSWConfig(m=1)

    def test_m_validation_high(self):
        """Test m parameter upper bound."""
        with pytest.raises(ValueError, match="m must be between"):
            HNSWConfig(m=101)

    def test_ef_construction_validation(self):
        """Test ef_construction parameter bounds."""
        with pytest.raises(ValueError, match="ef_construction must be between"):
            HNSWConfig(ef_construction=3)
        with pytest.raises(ValueError, match="ef_construction must be between"):
            HNSWConfig(ef_construction=1001)

    def test_ef_search_validation(self):
        """Test ef_search parameter bounds."""
        with pytest.raises(ValueError, match="ef_search must be between"):
            HNSWConfig(ef_search=0)

    def test_to_index_options(self):
        """Test HNSW index options string."""
        config = HNSWConfig(m=16, ef_construction=64)
        options = config.to_index_options()
        assert "m = 16" in options
        assert "ef_construction = 64" in options

    def test_to_search_options(self):
        """Test HNSW search options string."""
        config = HNSWConfig(ef_search=100)
        options = config.to_search_options()
        assert "SET hnsw.ef_search = 100" in options


# =============================================================================
# VECTOR SEARCH CONFIG TESTS
# =============================================================================


class TestVectorSearchConfig:
    """Tests for VectorSearchConfig."""

    def test_default_config(self):
        """Test default search configuration."""
        config = VectorSearchConfig()
        assert config.k == 10
        assert config.distance_metric == "cosine"
        assert config.filter_metadata is None
        assert config.use_hybrid is False

    def test_custom_config(self):
        """Test custom search configuration."""
        config = VectorSearchConfig(
            k=20, distance_metric="euclidean", filter_metadata={"type": "article"}, min_score=0.5
        )
        assert config.k == 20
        assert config.distance_metric == "euclidean"
        assert config.filter_metadata == {"type": "article"}
        assert config.min_score == 0.5

    def test_k_validation(self):
        """Test k parameter validation."""
        with pytest.raises(ValueError, match="k must be at least"):
            VectorSearchConfig(k=0)

    def test_invalid_distance_metric(self):
        """Test invalid distance metric."""
        with pytest.raises(ValueError, match="Unknown distance metric"):
            VectorSearchConfig(distance_metric="manhattan")

    def test_min_score_validation(self):
        """Test min_score bounds."""
        with pytest.raises(ValueError, match="min_score must be between"):
            VectorSearchConfig(min_score=1.5)

    def test_hybrid_weight_validation(self):
        """Test hybrid_weight bounds."""
        with pytest.raises(ValueError, match="hybrid_weight must be between"):
            VectorSearchConfig(hybrid_weight=-0.1)

    def test_get_distance_operator(self):
        """Test distance operator mapping."""
        assert VectorSearchConfig(distance_metric="cosine").get_distance_operator() == "<=>"
        assert VectorSearchConfig(distance_metric="euclidean").get_distance_operator() == "<->"
        assert VectorSearchConfig(distance_metric="inner_product").get_distance_operator() == "<#>"


# =============================================================================
# BATCH CONFIG TESTS
# =============================================================================


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_config(self):
        """Test default batch configuration."""
        config = BatchConfig()
        assert config.chunk_size == 100
        assert config.max_concurrent == 4
        assert config.retry_failed is True
        assert config.max_retries == 3
        assert config.on_error == "raise"

    def test_custom_config(self):
        """Test custom batch configuration."""
        config = BatchConfig(chunk_size=50, max_concurrent=8, retry_failed=False, on_error="skip")
        assert config.chunk_size == 50
        assert config.max_concurrent == 8
        assert config.retry_failed is False
        assert config.on_error == "skip"

    def test_chunk_size_validation(self):
        """Test chunk_size validation."""
        with pytest.raises(ValueError, match="chunk_size must be at least"):
            BatchConfig(chunk_size=0)

    def test_max_concurrent_validation(self):
        """Test max_concurrent validation."""
        with pytest.raises(ValueError, match="max_concurrent must be at least"):
            BatchConfig(max_concurrent=0)

    def test_invalid_on_error(self):
        """Test invalid on_error strategy."""
        with pytest.raises(ValueError, match="Unknown on_error strategy"):
            BatchConfig(on_error="ignore")


class TestBatchResult:
    """Tests for BatchResult."""

    def test_batch_result(self):
        """Test BatchResult properties."""
        result = BatchResult(
            successful=["a", "b", "c"],
            failed=[("d", "error1"), ("e", "error2")],
            total=5,
            duration_ms=100.0,
        )
        assert result.success_count == 3
        assert result.failure_count == 2
        assert result.success_rate == 60.0

    def test_empty_batch_result(self):
        """Test empty BatchResult."""
        result = BatchResult(successful=[], failed=[], total=0, duration_ms=0)
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.success_rate == 0.0  # 0/0 = 0


# =============================================================================
# POOL CONFIG TESTS
# =============================================================================


class TestPoolConfig:
    """Tests for PoolConfig."""

    def test_default_config(self):
        """Test default pool configuration."""
        config = PoolConfig()
        assert config.min_size == 2
        assert config.max_size == 10
        assert config.max_idle_seconds == 300.0
        assert config.statement_cache_size == 100

    def test_custom_config(self):
        """Test custom pool configuration."""
        config = PoolConfig(min_size=5, max_size=20, statement_cache_size=200)
        assert config.min_size == 5
        assert config.max_size == 20
        assert config.statement_cache_size == 200

    def test_min_size_validation(self):
        """Test min_size validation."""
        with pytest.raises(ValueError, match="min_size must be non-negative"):
            PoolConfig(min_size=-1)

    def test_max_size_validation(self):
        """Test max_size must be >= min_size."""
        with pytest.raises(ValueError, match="max_size must be >= min_size"):
            PoolConfig(min_size=10, max_size=5)

    def test_for_workload_light(self):
        """Test light workload configuration."""
        config = PoolConfig.for_workload("light")
        assert config.min_size == 1
        assert config.max_size == 5

    def test_for_workload_heavy(self):
        """Test heavy workload configuration."""
        config = PoolConfig.for_workload("heavy")
        assert config.min_size == 5
        assert config.max_size == 20

    def test_for_workload_invalid(self):
        """Test invalid workload type."""
        with pytest.raises(ValueError, match="Unknown workload type"):
            PoolConfig.for_workload("extreme")


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestChunkList:
    """Tests for chunk_list utility."""

    def test_exact_chunks(self):
        """Test chunking with exact division."""
        items = list(range(10))
        chunks = chunk_list(items, 5)
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2, 3, 4]
        assert chunks[1] == [5, 6, 7, 8, 9]

    def test_partial_last_chunk(self):
        """Test chunking with remainder."""
        items = list(range(7))
        chunks = chunk_list(items, 3)
        assert len(chunks) == 3
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6]

    def test_single_chunk(self):
        """Test when all items fit in one chunk."""
        items = [1, 2, 3]
        chunks = chunk_list(items, 10)
        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3]

    def test_empty_list(self):
        """Test chunking empty list."""
        chunks = chunk_list([], 5)
        assert chunks == []


@pytest.mark.asyncio
class TestExecuteWithRetry:
    """Tests for execute_with_retry utility."""

    async def test_success_first_try(self):
        """Test successful operation on first try."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await execute_with_retry(operation, max_retries=3)
        assert result == "success"
        assert call_count == 1

    async def test_success_after_retry(self):
        """Test successful operation after retry."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await execute_with_retry(
            operation, max_retries=3, base_delay=0.01, retryable_exceptions=(ValueError,)
        )
        assert result == "success"
        assert call_count == 3

    async def test_all_retries_failed(self):
        """Test failure after all retries exhausted."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            await execute_with_retry(
                operation, max_retries=2, base_delay=0.01, retryable_exceptions=(ValueError,)
            )
        assert call_count == 3  # Initial + 2 retries

    async def test_non_retryable_exception(self):
        """Test non-retryable exception is not retried."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            await execute_with_retry(
                operation,
                max_retries=3,
                retryable_exceptions=(ValueError,),  # TypeError not included
            )
        assert call_count == 1  # No retries


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAbstractionIntegration:
    """Integration tests combining multiple abstraction components."""

    def test_context_with_query_builder(self):
        """Test StorageContext with QueryBuilder."""
        ctx = StorageContext(user_id="user_123", namespace="project_a")

        query = (
            QueryBuilder("vectors")
            .select("id", "content", "embedding")
            .where_eq("collection_name", ctx.get_collection_name("documents"))
            .where_user(ctx)
            .order_by("created_at", descending=True)
            .limit(10)
        )

        sql, params = query.build_postgres()

        assert "collection_name = $1" in sql
        assert "user_id = $2" in sql
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT 10" in sql
        assert "project_a_documents" in params
        assert "user_123" in params

    def test_search_config_with_hnsw(self):
        """Test VectorSearchConfig with HNSWConfig."""
        hnsw = HNSWConfig(m=32, ef_construction=200, ef_search=100)
        search = VectorSearchConfig(
            k=20, distance_metric="cosine", hnsw_config=hnsw, use_hybrid=True, hybrid_weight=0.6
        )

        assert search.hnsw_config.m == 32
        assert search.get_distance_operator() == "<=>"
        assert search.use_hybrid is True
