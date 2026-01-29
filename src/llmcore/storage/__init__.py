# src/llmcore/storage/__init__.py
"""
Storage management module for the LLMCore library.

This package handles the persistence and retrieval of chat sessions,
context presets, episodic memory, and vector embeddings for RAG.
It provides both session storage (for conversations and episodes) and
vector storage (for semantic memory) backends.

STORAGE SYSTEM V2:
- Phase 1 (PRIMORDIUM): Schema versioning, health monitoring, config validation
- Phase 2 (NEXUS): Multi-backend abstraction, pgvector optimization, user isolation
- Phase 3 (SYMBIOSIS): SemantiScan vector delegation layer
- Phase 4 (PANOPTICON): Observability, metrics, event logging, diagnostics
"""

# Import key storage components for easier access
from .manager import StorageManager
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

# Import concrete implementations
from .json_session import JsonSessionStorage
from .sqlite_session import SqliteSessionStorage
from .postgres_session_storage import PostgresSessionStorage
from .pgvector_storage import PgVectorStorage
from .chromadb_vector import ChromaVectorStorage

# Phase 1: Schema management
from .schema_manager import (
    BaseSchemaManager,
    PostgresSchemaManager,
    SqliteSchemaManager,
    SchemaBackend,
    SchemaVersion,
    SchemaMigration,
    create_schema_manager,
    CURRENT_SCHEMA_VERSION,
)

# Phase 1: Health monitoring
from .health import (
    StorageHealthManager,
    StorageHealthMonitor,
    HealthConfig,
    HealthStatus,
    HealthCheckResult,
    StorageHealthReport,
    CircuitBreaker,
    CircuitState,
)

# Phase 1: Config validation
from .config_validator import (
    StorageConfigValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_storage_config,
)

# Phase 1: CLI commands
from .cli import (
    StorageCommands,
    cmd_validate,
    cmd_health,
    cmd_schema,
    cmd_info,
    # Phase 4 (PANOPTICON) CLI commands
    cmd_stats,
    cmd_inspect,
    cmd_diagnose,
    cmd_cleanup,
    main as cli_main,
)

# Phase 2: Storage Abstraction Layer
from .abstraction import (
    StorageContext,
    IsolationLevel,
    BackendCapabilities,
    POSTGRES_PGVECTOR_CAPABILITIES,
    SQLITE_CAPABILITIES,
    CHROMADB_CAPABILITIES,
    StorageBackendProtocol,
    QueryBuilder,
    QueryOperator,
    QueryCondition,
    HNSWConfig,
    VectorSearchConfig,
    BatchConfig,
    BatchResult,
    PoolConfig,
    chunk_list,
    execute_with_retry,
)

# Phase 2: Enhanced PgVector Storage
from .pgvector_enhanced import (
    EnhancedPgVectorStorage,
    CollectionInfo,
    HybridSearchResult,
)

# Phase 4 (PANOPTICON): Observability & Control Plane
from .instrumentation import (
    StorageInstrumentation,
    InstrumentationConfig,
    InstrumentationContext,
    OperationRecord,
    instrumented,
    MetricsBackend as InstrumentationMetricsBackend,
    TracingBackend,
)

from .metrics import (
    MetricsCollector,
    MetricsConfig,
    MetricsBackendType,
    InMemoryMetricsBackend,
    PrometheusMetricsBackend,
    NullMetricsBackend,
)

from .events import (
    EventLogger,
    EventLoggerConfig,
    EventType,
    StorageEvent,
)

from .observability import (
    ObservabilityConfig,
    DEFAULT_OBSERVABILITY_CONFIG,
    OBSERVABILITY_TOML_EXAMPLE,
)

# Phase 3: Volatile Memory Tier
from .tiers.volatile import (
    VolatileItem,
    VolatileMemoryTier,
    VolatileMemoryConfig,
    create_volatile_tier,
)

# Phase 3: Feedback System
from .feedback import (
    FeedbackRecord,
    AggregatedFeedback,
    FeedbackConfig,
    FeedbackManager,
    create_feedback_manager,
)

__all__ = [
    # Core components
    "StorageManager",
    "BaseSessionStorage",
    "BaseVectorStorage",
    # Session storage implementations
    "JsonSessionStorage",
    "SqliteSessionStorage",
    "PostgresSessionStorage",
    # Vector storage implementations
    "PgVectorStorage",
    "ChromaVectorStorage",
    "EnhancedPgVectorStorage",  # Phase 2
    # Schema management (Phase 1)
    "BaseSchemaManager",
    "PostgresSchemaManager",
    "SqliteSchemaManager",
    "SchemaBackend",
    "SchemaVersion",
    "SchemaMigration",
    "create_schema_manager",
    "CURRENT_SCHEMA_VERSION",
    # Health monitoring (Phase 1)
    "StorageHealthManager",
    "StorageHealthMonitor",
    "HealthConfig",
    "HealthStatus",
    "HealthCheckResult",
    "StorageHealthReport",
    "CircuitBreaker",
    "CircuitState",
    # Config validation (Phase 1)
    "StorageConfigValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_storage_config",
    # CLI (Phase 1)
    "StorageCommands",
    "cmd_validate",
    "cmd_health",
    "cmd_schema",
    "cmd_info",
    # CLI (Phase 4 - PANOPTICON)
    "cmd_stats",
    "cmd_inspect",
    "cmd_diagnose",
    "cmd_cleanup",
    "cli_main",
    # Abstraction Layer (Phase 2)
    "StorageContext",
    "IsolationLevel",
    "BackendCapabilities",
    "POSTGRES_PGVECTOR_CAPABILITIES",
    "SQLITE_CAPABILITIES",
    "CHROMADB_CAPABILITIES",
    "StorageBackendProtocol",
    "QueryBuilder",
    "QueryOperator",
    "QueryCondition",
    "HNSWConfig",
    "VectorSearchConfig",
    "BatchConfig",
    "BatchResult",
    "PoolConfig",
    "chunk_list",
    "execute_with_retry",
    # Enhanced PgVector (Phase 2)
    "CollectionInfo",
    "HybridSearchResult",
    # Instrumentation (Phase 4 - PANOPTICON)
    "StorageInstrumentation",
    "InstrumentationConfig",
    "InstrumentationContext",
    "OperationRecord",
    "instrumented",
    "InstrumentationMetricsBackend",
    "TracingBackend",
    # Metrics (Phase 4 - PANOPTICON)
    "MetricsCollector",
    "MetricsConfig",
    "MetricsBackendType",
    "InMemoryMetricsBackend",
    "PrometheusMetricsBackend",
    "NullMetricsBackend",
    # Events (Phase 4 - PANOPTICON)
    "EventLogger",
    "EventLoggerConfig",
    "EventType",
    "StorageEvent",
    # Observability Config (Phase 4 - PANOPTICON)
    "ObservabilityConfig",
    "DEFAULT_OBSERVABILITY_CONFIG",
    "OBSERVABILITY_TOML_EXAMPLE",
    # Volatile Memory Tier (Phase 3)
    "VolatileItem",
    "VolatileMemoryTier",
    "VolatileMemoryConfig",
    "create_volatile_tier",
    # Feedback System (Phase 3)
    "FeedbackRecord",
    "AggregatedFeedback",
    "FeedbackConfig",
    "FeedbackManager",
    "create_feedback_manager",
]
