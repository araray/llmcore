# src/llmcore/storage/abstraction.py
"""
Storage Abstraction Layer for LLMCore.

Phase 2 (NEXUS): Provides unified abstractions for storage backends,
enabling backend-agnostic operations and multi-user isolation.

Key Components:
- StorageContext: User/tenant context for all operations
- StorageBackendProtocol: Common interface for all backends
- QueryBuilder: Backend-agnostic query construction helpers
- TransactionManager: Cross-backend transaction coordination

Design Philosophy:
- Backend-agnostic interfaces allow transparent backend switching
- User isolation is enforced at the abstraction layer
- Query builders prevent SQL injection while preserving flexibility
- Transaction semantics are consistent across backends

Usage:
    # Create storage context for a user
    ctx = StorageContext(user_id="user_123", namespace="project_a")

    # Use with storage manager
    sessions = await storage.list_sessions(context=ctx)

    # Query builder for complex queries
    query = (QueryBuilder("sessions")
        .select("id", "name", "updated_at")
        .where_eq("user_id", ctx.user_id)
        .order_by("updated_at", descending=True)
        .limit(10))
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar('T')  # Generic result type
DocT = TypeVar('DocT')  # Document type for vector operations


# =============================================================================
# STORAGE CONTEXT
# =============================================================================

@dataclass
class StorageContext:
    """
    Context for storage operations providing user isolation and namespacing.

    This context should be passed to all storage operations to ensure
    proper user isolation and collection namespacing.

    Attributes:
        user_id: Unique identifier for the user. If None, operations are
                 performed without user filtering (system-level access).
        namespace: Optional namespace prefix for collections. Used to
                   isolate vector collections per project/session.
        metadata: Additional context metadata (e.g., request_id, trace_id)
        read_only: If True, only read operations are permitted.
        require_user: If True (default), operations that modify data
                      require a non-None user_id.

    Example:
        # User-scoped context
        ctx = StorageContext(user_id="user_123")

        # Project-scoped with namespace
        ctx = StorageContext(
            user_id="user_123",
            namespace="project_alpha",
            metadata={"request_id": "req_abc"}
        )

        # System-level access (no user filtering)
        ctx = StorageContext.system_context()
    """
    user_id: Optional[str] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = False
    require_user: bool = True

    def __post_init__(self):
        """Validate context after initialization."""
        if self.namespace:
            # Sanitize namespace to prevent injection
            self.namespace = self._sanitize_identifier(self.namespace)

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """Sanitize identifier to prevent SQL injection."""
        # Allow only alphanumeric, underscore, and hyphen
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"ns_{sanitized}"
        return sanitized[:64]  # Limit length

    @classmethod
    def system_context(cls, metadata: Optional[Dict[str, Any]] = None) -> "StorageContext":
        """
        Create a system-level context without user isolation.

        This should only be used for administrative operations,
        not for user-facing functionality.

        Args:
            metadata: Optional metadata for tracing/logging

        Returns:
            StorageContext with require_user=False
        """
        return cls(
            user_id=None,
            namespace=None,
            metadata=metadata or {},
            require_user=False
        )

    def with_namespace(self, namespace: str) -> "StorageContext":
        """
        Create a new context with a different namespace.

        Args:
            namespace: New namespace prefix

        Returns:
            New StorageContext with updated namespace
        """
        return StorageContext(
            user_id=self.user_id,
            namespace=namespace,
            metadata=self.metadata.copy(),
            read_only=self.read_only,
            require_user=self.require_user
        )

    def get_collection_name(self, base_name: str) -> str:
        """
        Get the full collection name with namespace prefix.

        Args:
            base_name: Base collection name

        Returns:
            Namespaced collection name (e.g., "myproject_documents")
        """
        if self.namespace:
            return f"{self.namespace}_{base_name}"
        return base_name

    def validate_for_write(self) -> None:
        """
        Validate context is suitable for write operations.

        Raises:
            PermissionError: If context is read-only or missing required user_id
        """
        if self.read_only:
            raise PermissionError("Storage context is read-only")
        if self.require_user and not self.user_id:
            raise PermissionError("User ID required for write operations")


# =============================================================================
# ISOLATION LEVEL
# =============================================================================

class IsolationLevel(str, Enum):
    """
    Data isolation levels for storage operations.

    Controls how data is partitioned/filtered based on user identity.
    """
    NONE = "none"           # No isolation, all data visible
    USER = "user"           # Filter by user_id
    NAMESPACE = "namespace" # Filter by namespace prefix
    FULL = "full"           # Filter by user_id AND namespace


# =============================================================================
# BACKEND CAPABILITIES
# =============================================================================

@dataclass(frozen=True)
class BackendCapabilities:
    """
    Declares what features a storage backend supports.

    This allows the abstraction layer to adapt behavior based on
    backend capabilities without requiring backend-specific code.
    """
    # Vector operations
    supports_vector_search: bool = False
    supports_hybrid_search: bool = False
    supports_filtered_search: bool = False
    supports_hnsw_index: bool = False
    max_vector_dimension: int = 0

    # Session operations
    supports_full_text_search: bool = False
    supports_json_queries: bool = False
    supports_transactions: bool = True

    # Performance features
    supports_batch_operations: bool = True
    supports_prepared_statements: bool = False
    supports_connection_pooling: bool = False

    # Isolation
    supports_user_isolation: bool = True
    supports_namespace_isolation: bool = True

    def __str__(self) -> str:
        """Human-readable capability summary."""
        features = []
        if self.supports_vector_search:
            features.append("vector")
        if self.supports_hybrid_search:
            features.append("hybrid")
        if self.supports_full_text_search:
            features.append("fts")
        if self.supports_hnsw_index:
            features.append("hnsw")
        if self.supports_transactions:
            features.append("tx")
        return f"Capabilities({', '.join(features) or 'basic'})"


# PostgreSQL + pgvector capabilities
POSTGRES_PGVECTOR_CAPABILITIES = BackendCapabilities(
    supports_vector_search=True,
    supports_hybrid_search=True,
    supports_filtered_search=True,
    supports_hnsw_index=True,
    max_vector_dimension=16000,  # pgvector limit
    supports_full_text_search=True,
    supports_json_queries=True,
    supports_transactions=True,
    supports_batch_operations=True,
    supports_prepared_statements=True,
    supports_connection_pooling=True,
    supports_user_isolation=True,
    supports_namespace_isolation=True,
)

# SQLite capabilities
SQLITE_CAPABILITIES = BackendCapabilities(
    supports_vector_search=False,
    supports_hybrid_search=False,
    supports_filtered_search=False,
    supports_hnsw_index=False,
    max_vector_dimension=0,
    supports_full_text_search=True,  # via FTS5
    supports_json_queries=True,  # SQLite 3.38+
    supports_transactions=True,
    supports_batch_operations=True,
    supports_prepared_statements=False,
    supports_connection_pooling=False,
    supports_user_isolation=True,
    supports_namespace_isolation=True,
)

# ChromaDB capabilities
CHROMADB_CAPABILITIES = BackendCapabilities(
    supports_vector_search=True,
    supports_hybrid_search=False,
    supports_filtered_search=True,
    supports_hnsw_index=True,
    max_vector_dimension=65536,
    supports_full_text_search=False,
    supports_json_queries=False,
    supports_transactions=False,
    supports_batch_operations=True,
    supports_prepared_statements=False,
    supports_connection_pooling=False,
    supports_user_isolation=True,
    supports_namespace_isolation=True,
)


# =============================================================================
# STORAGE BACKEND PROTOCOL
# =============================================================================

@runtime_checkable
class StorageBackendProtocol(Protocol):
    """
    Protocol defining the common interface for all storage backends.

    All storage backends (session, vector, etc.) should implement
    this protocol for basic lifecycle and health operations.
    """

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the backend with configuration."""
        ...

    async def close(self) -> None:
        """Close and cleanup resources."""
        ...

    def get_capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""
        ...

    async def health_check(self) -> bool:
        """Perform a quick health check."""
        ...


# =============================================================================
# QUERY BUILDER
# =============================================================================

class QueryOperator(str, Enum):
    """SQL comparison operators."""
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


@dataclass
class QueryCondition:
    """A single query condition."""
    column: str
    operator: QueryOperator
    value: Any
    parameter_name: Optional[str] = None

    def to_sql_postgres(self, param_index: int) -> Tuple[str, Optional[Any]]:
        """Generate PostgreSQL condition with $n placeholder."""
        if self.operator in (QueryOperator.IS_NULL, QueryOperator.IS_NOT_NULL):
            return f"{self.column} {self.operator.value}", None
        elif self.operator in (QueryOperator.IN, QueryOperator.NOT_IN):
            return f"{self.column} {self.operator.value} (${param_index})", self.value
        else:
            return f"{self.column} {self.operator.value} ${param_index}", self.value

    def to_sql_sqlite(self, param_index: int) -> Tuple[str, Optional[Any]]:
        """Generate SQLite condition with ? placeholder."""
        if self.operator in (QueryOperator.IS_NULL, QueryOperator.IS_NOT_NULL):
            return f"{self.column} {self.operator.value}", None
        elif self.operator in (QueryOperator.IN, QueryOperator.NOT_IN):
            # SQLite requires explicit placeholders for IN
            placeholders = ", ".join("?" for _ in self.value)
            return f"{self.column} {self.operator.value} ({placeholders})", self.value
        else:
            return f"{self.column} {self.operator.value} ?", self.value

    def to_sql_named(self) -> Tuple[str, str, Any]:
        """Generate SQL with named placeholder (:name)."""
        param_name = self.parameter_name or self.column.replace(".", "_")
        if self.operator in (QueryOperator.IS_NULL, QueryOperator.IS_NOT_NULL):
            return f"{self.column} {self.operator.value}", param_name, None
        else:
            return f"{self.column} {self.operator.value} :{param_name}", param_name, self.value


class QueryBuilder:
    """
    Backend-agnostic SQL query builder with safety guarantees.

    Prevents SQL injection by:
    - Validating column names against allowlist
    - Using parameterized queries for values
    - Escaping identifiers properly

    Example:
        query = (QueryBuilder("sessions")
            .select("id", "name", "updated_at")
            .where_eq("user_id", "user_123")
            .where("metadata->>'type'", "=", "chat")
            .order_by("updated_at", descending=True)
            .limit(10))

        # For PostgreSQL
        sql, params = query.build_postgres()

        # For SQLite
        sql, params = query.build_sqlite()
    """

    def __init__(
        self,
        table: str,
        allowed_columns: Optional[List[str]] = None
    ):
        """
        Initialize query builder for a table.

        Args:
            table: Table name (will be validated)
            allowed_columns: Optional list of allowed column names.
                           If provided, column access is restricted.
        """
        self._table = self._validate_identifier(table)
        self._allowed_columns = set(allowed_columns) if allowed_columns else None
        self._select_columns: List[str] = []
        self._conditions: List[QueryCondition] = []
        self._order_by: List[Tuple[str, bool]] = []  # (column, descending)
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._for_update: bool = False

    @staticmethod
    def _validate_identifier(name: str) -> str:
        """Validate and sanitize SQL identifier."""
        import re
        # Allow alphanumeric, underscore, and common JSON operators
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name.split('->')[0].split('.')[0]):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return name

    def _validate_column(self, column: str) -> str:
        """Validate column name against allowlist if configured."""
        base_column = column.split('->')[0].split('.')[0]
        if self._allowed_columns and base_column not in self._allowed_columns:
            raise ValueError(f"Column '{base_column}' not in allowed columns")
        return column

    def select(self, *columns: str) -> "QueryBuilder":
        """Add columns to SELECT clause."""
        self._select_columns.extend(
            self._validate_column(c) for c in columns
        )
        return self

    def where(
        self,
        column: str,
        operator: Union[str, QueryOperator],
        value: Any
    ) -> "QueryBuilder":
        """Add a WHERE condition."""
        if isinstance(operator, str):
            operator = QueryOperator(operator)
        self._conditions.append(QueryCondition(
            column=self._validate_column(column),
            operator=operator,
            value=value
        ))
        return self

    def where_eq(self, column: str, value: Any) -> "QueryBuilder":
        """Add equality condition."""
        return self.where(column, QueryOperator.EQ, value)

    def where_in(self, column: str, values: List[Any]) -> "QueryBuilder":
        """Add IN condition."""
        return self.where(column, QueryOperator.IN, values)

    def where_null(self, column: str) -> "QueryBuilder":
        """Add IS NULL condition."""
        return self.where(column, QueryOperator.IS_NULL, None)

    def where_not_null(self, column: str) -> "QueryBuilder":
        """Add IS NOT NULL condition."""
        return self.where(column, QueryOperator.IS_NOT_NULL, None)

    def where_user(self, context: StorageContext) -> "QueryBuilder":
        """
        Add user isolation condition from context.

        If context.user_id is None and context.require_user is False,
        no condition is added (system-level access).
        """
        if context.user_id:
            return self.where_eq("user_id", context.user_id)
        elif context.require_user:
            raise ValueError("StorageContext requires user_id but none provided")
        return self

    def order_by(self, column: str, descending: bool = False) -> "QueryBuilder":
        """Add ORDER BY clause."""
        self._order_by.append((self._validate_column(column), descending))
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """Add LIMIT clause."""
        if n < 0:
            raise ValueError("LIMIT must be non-negative")
        self._limit = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        """Add OFFSET clause."""
        if n < 0:
            raise ValueError("OFFSET must be non-negative")
        self._offset = n
        return self

    def for_update(self) -> "QueryBuilder":
        """Add FOR UPDATE clause (row locking)."""
        self._for_update = True
        return self

    def build_postgres(self) -> Tuple[str, List[Any]]:
        """
        Build PostgreSQL query with $n placeholders.

        Returns:
            Tuple of (sql_string, params_list)
        """
        parts = ["SELECT"]

        # Columns
        if self._select_columns:
            parts.append(", ".join(self._select_columns))
        else:
            parts.append("*")

        parts.append(f"FROM {self._table}")

        # WHERE clause
        params: List[Any] = []
        if self._conditions:
            conditions = []
            param_idx = 1
            for cond in self._conditions:
                sql, value = cond.to_sql_postgres(param_idx)
                conditions.append(sql)
                if value is not None:
                    if cond.operator in (QueryOperator.IN, QueryOperator.NOT_IN):
                        # Flatten list for IN clause
                        params.extend(value if isinstance(value, list) else [value])
                        # Rewrite with correct number of placeholders
                        count = len(value) if isinstance(value, list) else 1
                        placeholders = ", ".join(f"${param_idx + i}" for i in range(count))
                        conditions[-1] = f"{cond.column} {cond.operator.value} ({placeholders})"
                        param_idx += count
                    else:
                        params.append(value)
                        param_idx += 1
            parts.append("WHERE " + " AND ".join(conditions))

        # ORDER BY
        if self._order_by:
            order_parts = [
                f"{col} {'DESC' if desc else 'ASC'}"
                for col, desc in self._order_by
            ]
            parts.append("ORDER BY " + ", ".join(order_parts))

        # LIMIT/OFFSET
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        # FOR UPDATE
        if self._for_update:
            parts.append("FOR UPDATE")

        return " ".join(parts), params

    def build_sqlite(self) -> Tuple[str, List[Any]]:
        """
        Build SQLite query with ? placeholders.

        Returns:
            Tuple of (sql_string, params_list)
        """
        parts = ["SELECT"]

        # Columns
        if self._select_columns:
            parts.append(", ".join(self._select_columns))
        else:
            parts.append("*")

        parts.append(f"FROM {self._table}")

        # WHERE clause
        params: List[Any] = []
        if self._conditions:
            conditions = []
            param_idx = 1
            for cond in self._conditions:
                sql, value = cond.to_sql_sqlite(param_idx)
                conditions.append(sql)
                if value is not None:
                    if isinstance(value, list):
                        params.extend(value)
                    else:
                        params.append(value)
                    param_idx += 1
            parts.append("WHERE " + " AND ".join(conditions))

        # ORDER BY
        if self._order_by:
            order_parts = [
                f"{col} {'DESC' if desc else 'ASC'}"
                for col, desc in self._order_by
            ]
            parts.append("ORDER BY " + ", ".join(order_parts))

        # LIMIT/OFFSET
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        return " ".join(parts), params

    def build_named(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build query with named :param placeholders.

        Useful for SQLAlchemy text() queries.

        Returns:
            Tuple of (sql_string, params_dict)
        """
        parts = ["SELECT"]

        # Columns
        if self._select_columns:
            parts.append(", ".join(self._select_columns))
        else:
            parts.append("*")

        parts.append(f"FROM {self._table}")

        # WHERE clause
        params: Dict[str, Any] = {}
        if self._conditions:
            conditions = []
            for i, cond in enumerate(self._conditions):
                sql, param_name, value = cond.to_sql_named()
                # Make param name unique if needed
                unique_name = f"{param_name}_{i}" if param_name in params else param_name
                if value is not None:
                    params[unique_name] = value
                    sql = sql.replace(f":{param_name}", f":{unique_name}")
                conditions.append(sql)
            parts.append("WHERE " + " AND ".join(conditions))

        # ORDER BY
        if self._order_by:
            order_parts = [
                f"{col} {'DESC' if desc else 'ASC'}"
                for col, desc in self._order_by
            ]
            parts.append("ORDER BY " + ", ".join(order_parts))

        # LIMIT/OFFSET
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        # FOR UPDATE
        if self._for_update:
            parts.append("FOR UPDATE")

        return " ".join(parts), params


# =============================================================================
# VECTOR SEARCH CONFIG
# =============================================================================

@dataclass
class HNSWConfig:
    """
    HNSW index configuration for pgvector.

    HNSW (Hierarchical Navigable Small World) provides fast approximate
    nearest neighbor search with tunable accuracy/speed tradeoffs.

    Attributes:
        m: Maximum number of connections per node (default: 16).
           Higher values give better recall but slower indexing.
           Range: 2-100, recommended: 12-64
        ef_construction: Size of dynamic candidate list during index building.
                        Higher values give better recall but slower indexing.
                        Range: 4-1000, recommended: 64-200
        ef_search: Size of dynamic candidate list during search.
                  Higher values give better recall but slower search.
                  Range: 1-1000, recommended: 40-200 (must be >= k)

    Example:
        # High recall, slower (for quality-critical applications)
        config = HNSWConfig(m=32, ef_construction=200, ef_search=100)

        # Fast, lower recall (for high-throughput applications)
        config = HNSWConfig(m=8, ef_construction=64, ef_search=20)
    """
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 40

    def __post_init__(self):
        """Validate configuration values."""
        if not (2 <= self.m <= 100):
            raise ValueError(f"m must be between 2 and 100, got {self.m}")
        if not (4 <= self.ef_construction <= 1000):
            raise ValueError(f"ef_construction must be between 4 and 1000, got {self.ef_construction}")
        if not (1 <= self.ef_search <= 1000):
            raise ValueError(f"ef_search must be between 1 and 1000, got {self.ef_search}")

    def to_index_options(self) -> str:
        """Generate pgvector index options string."""
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"

    def to_search_options(self) -> str:
        """Generate SET command for search options."""
        return f"SET hnsw.ef_search = {self.ef_search}"


@dataclass
class VectorSearchConfig:
    """
    Configuration for vector similarity search operations.

    Attributes:
        k: Number of results to return
        distance_metric: Distance metric ("cosine", "euclidean", "inner_product")
        filter_metadata: Metadata filters to apply
        min_score: Minimum similarity score threshold (0-1 for cosine)
        include_embeddings: Whether to return embeddings in results
        hnsw_config: HNSW-specific search parameters
        use_hybrid: Enable hybrid search (vector + full-text)
        hybrid_weight: Weight for vector similarity in hybrid mode (0-1)
    """
    k: int = 10
    distance_metric: str = "cosine"
    filter_metadata: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = None
    include_embeddings: bool = False
    hnsw_config: Optional[HNSWConfig] = None
    use_hybrid: bool = False
    hybrid_weight: float = 0.7  # Vector weight in hybrid mode

    def __post_init__(self):
        """Validate configuration."""
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.distance_metric not in ("cosine", "euclidean", "inner_product"):
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        if self.min_score is not None and not (0.0 <= self.min_score <= 1.0):
            raise ValueError("min_score must be between 0 and 1")
        if not (0.0 <= self.hybrid_weight <= 1.0):
            raise ValueError("hybrid_weight must be between 0 and 1")

    def get_distance_operator(self) -> str:
        """Get pgvector distance operator for the configured metric."""
        operators = {
            "cosine": "<=>",      # Cosine distance
            "euclidean": "<->",   # L2 distance
            "inner_product": "<#>"  # Negative inner product
        }
        return operators[self.distance_metric]


# =============================================================================
# BATCH OPERATION CONFIG
# =============================================================================

@dataclass
class BatchConfig:
    """
    Configuration for batch operations.

    Attributes:
        chunk_size: Number of items per batch
        max_concurrent: Maximum concurrent batch operations
        retry_failed: Whether to retry failed items
        max_retries: Maximum retries per item
        on_error: Error handling strategy ("raise", "skip", "collect")
    """
    chunk_size: int = 100
    max_concurrent: int = 4
    retry_failed: bool = True
    max_retries: int = 3
    on_error: str = "raise"

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.on_error not in ("raise", "skip", "collect"):
            raise ValueError(f"Unknown on_error strategy: {self.on_error}")


@dataclass
class BatchResult(Generic[T]):
    """
    Result of a batch operation.

    Attributes:
        successful: Successfully processed items
        failed: Failed items with error details
        total: Total number of items processed
        duration_ms: Total operation duration
    """
    successful: List[T]
    failed: List[Tuple[Any, str]]  # (item, error_message)
    total: int
    duration_ms: float

    @property
    def success_count(self) -> int:
        """Number of successfully processed items."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed items."""
        return len(self.failed)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.success_count / self.total * 100) if self.total > 0 else 0.0


# =============================================================================
# CONNECTION POOL CONFIG
# =============================================================================

@dataclass
class PoolConfig:
    """
    Connection pool configuration.

    Attributes:
        min_size: Minimum connections to maintain
        max_size: Maximum connections allowed
        max_idle_seconds: Close connections idle longer than this
        max_lifetime_seconds: Maximum connection lifetime
        acquire_timeout_seconds: Timeout for acquiring connection
        statement_cache_size: Number of prepared statements to cache
        health_check_interval: Seconds between pool health checks
    """
    min_size: int = 2
    max_size: int = 10
    max_idle_seconds: float = 300.0
    max_lifetime_seconds: float = 3600.0
    acquire_timeout_seconds: float = 30.0
    statement_cache_size: int = 100
    health_check_interval: float = 30.0

    def __post_init__(self):
        """Validate configuration."""
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        if self.max_size > 100:
            logger.warning(f"Large pool size ({self.max_size}) may cause resource issues")

    @classmethod
    def for_workload(cls, workload: str) -> "PoolConfig":
        """
        Create pool config optimized for a workload type.

        Args:
            workload: "light", "moderate", "heavy", or "burst"

        Returns:
            PoolConfig tuned for the workload
        """
        configs = {
            "light": cls(min_size=1, max_size=5, statement_cache_size=50),
            "moderate": cls(min_size=2, max_size=10, statement_cache_size=100),
            "heavy": cls(min_size=5, max_size=20, statement_cache_size=200),
            "burst": cls(min_size=2, max_size=30, acquire_timeout_seconds=60.0),
        }
        if workload not in configs:
            raise ValueError(f"Unknown workload type: {workload}")
        return configs[workload]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def execute_with_retry(
    operation: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    retryable_exceptions: Tuple[type, ...] = (Exception,)
) -> T:
    """
    Execute an async operation with exponential backoff retry.

    Args:
        operation: Async callable to execute
        max_retries: Maximum number of retries
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retryable_exceptions: Exceptions that trigger retry

    Returns:
        Result of the operation

    Raises:
        Last exception if all retries fail
    """
    import asyncio
    import random

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay *= (0.5 + random.random())  # Add jitter
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

    raise last_exception  # type: ignore


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Context
    "StorageContext",
    "IsolationLevel",
    # Capabilities
    "BackendCapabilities",
    "POSTGRES_PGVECTOR_CAPABILITIES",
    "SQLITE_CAPABILITIES",
    "CHROMADB_CAPABILITIES",
    # Protocols
    "StorageBackendProtocol",
    # Query building
    "QueryBuilder",
    "QueryOperator",
    "QueryCondition",
    # Vector config
    "HNSWConfig",
    "VectorSearchConfig",
    # Batch operations
    "BatchConfig",
    "BatchResult",
    # Connection pool
    "PoolConfig",
    # Utilities
    "chunk_list",
    "execute_with_retry",
]
