# src/llmcore/storage/manager.py
"""
Storage Manager for LLMCore.

Handles the initialization and management of session and vector storage backends
based on the application's configuration. Now includes episodic memory management
through the session storage backends.

STORAGE SYSTEM V2 (Phase 1 - PRIMORDIUM):
- Configuration validation at startup
- Idempotent schema management with version tracking
- Health monitoring with circuit breakers
- Connection pool management with health checks
- Unified health API for all backends

STORAGE SYSTEM V2 (Phase 4 - PANOPTICON):
- Integrated observability layer
- Instrumentation for all storage operations
- Metrics collection (Prometheus, in-memory)
- Event logging for audit trails
- Slow query detection and logging

REFACTORED FOR LIBRARY MODE (Step 1.3): This manager now directly instantiates
and holds storage backend instances rather than acting as a factory. All multi-tenant
db_session parameters have been removed for single-tenant library usage.
"""

import logging
from typing import Any

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = dict[str, Any]  # type: ignore

from ..exceptions import ConfigError, StorageError
from ..models import Episode
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

# Import concrete storage implementations
from .chromadb_vector import ChromaVectorStorage

# Phase 1 imports: Config validation, health monitoring, schema management
from .config_validator import StorageConfigValidator, ValidationResult
from .events import EventLogger, EventLoggerConfig
from .health import (
    HealthConfig,
    HealthStatus,
    StorageHealthManager,
    StorageHealthMonitor,
    create_chromadb_health_check,
    create_postgres_health_check,
    create_sqlite_health_check,
)
from .instrumentation import InstrumentationConfig, StorageInstrumentation
from .json_session import JsonSessionStorage
from .metrics import MetricsBackendType, MetricsCollector, MetricsConfig

# Phase 4 imports: Observability (PANOPTICON)
from .observability import DEFAULT_OBSERVABILITY_CONFIG, ObservabilityConfig
from .pgvector_enhanced import EnhancedPgVectorStorage
from .pgvector_storage import PgVectorStorage
from .postgres_session_storage import PostgresSessionStorage
from .sqlite_session import SqliteSessionStorage

logger = logging.getLogger(__name__)

# --- Mappings from config type string to class ---
SESSION_STORAGE_MAP: dict[str, type[BaseSessionStorage]] = {
    "json": JsonSessionStorage,
    "sqlite": SqliteSessionStorage,
    "postgres": PostgresSessionStorage,
}

VECTOR_STORAGE_MAP: dict[str, type[BaseVectorStorage]] = {
    "chromadb": ChromaVectorStorage,
    "pgvector": EnhancedPgVectorStorage,  # Use enhanced version with full collection management
    "pgvector_legacy": PgVectorStorage,  # Legacy version for backwards compatibility
}
# --- End Mappings ---


class StorageManager:
    """
    Manages the initialization and access to storage backends.

    STORAGE SYSTEM V2 (Phase 1 - PRIMORDIUM):
    This manager now provides:
    - Configuration validation at startup with clear error messages
    - Health monitoring with circuit breakers for reliability
    - Unified health API for observability
    - Schema management (via storage backends)

    STORAGE SYSTEM V2 (Phase 4 - PANOPTICON):
    Integrated observability layer providing:
    - Instrumentation for all storage operations with timing
    - Metrics collection (counters, histograms, gauges)
    - Event logging for persistent audit trails
    - Slow query detection and logging

    REFACTORED (Step 1.3): Now operates in library mode. Storage backend instances
    are created once during initialization and held internally, rather than being
    created on-demand per-request. This provides a simple, single-tenant API suitable
    for library usage.
    """

    _config: ConfyConfig
    _session_storage_type: str | None = None
    _vector_storage_type: str | None = None
    _session_storage_config: dict[str, Any] = {}
    _vector_storage_config: dict[str, Any] = {}

    # Storage instances
    _session_storage_instance: BaseSessionStorage | None = None
    _vector_storage_instance: BaseVectorStorage | None = None

    # Phase 1: Health monitoring
    _health_manager: StorageHealthManager | None = None
    _health_config: HealthConfig
    _validation_result: ValidationResult | None = None

    # Phase 4: Observability (PANOPTICON)
    _observability_config: ObservabilityConfig
    _instrumentation: StorageInstrumentation | None = None
    _metrics_collector: MetricsCollector | None = None
    _event_logger: EventLogger | None = None

    def __init__(
        self,
        config: ConfyConfig,
        health_config: HealthConfig | None = None,
        observability_config: ObservabilityConfig | None = None,
        validate_config: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initializes the StorageManager.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
            health_config: Optional health monitoring configuration.
            observability_config: Optional Phase 4 observability configuration.
            validate_config: Whether to validate configuration at startup (default: True).
            strict_validation: Whether to treat warnings as errors (default: False).

        Raises:
            ConfigError: If validate_config=True and configuration has errors.
        """
        self._config = config
        self._health_config = health_config or HealthConfig()
        self._health_manager = StorageHealthManager()
        self._initialized = False

        # Phase 4: Initialize observability configuration
        # Priority: explicit parameter > config file > defaults
        if observability_config is not None:
            self._observability_config = observability_config
        else:
            self._observability_config = self._load_observability_config()

        # Validate configuration if requested
        if validate_config:
            self._validate_configuration(strict_validation)

        logger.debug("StorageManager initialized for library mode (single-tenant).")

    def _validate_configuration(self, strict: bool = False) -> None:
        """
        Validate storage configuration at startup.

        Args:
            strict: If True, treat warnings as errors.

        Raises:
            ConfigError: If validation fails with errors.
        """
        # Convert ConfyConfig to dict for validation
        config_dict = {}
        try:
            # Try to get the raw config dict
            if hasattr(self._config, "to_dict"):
                config_dict = self._config.to_dict()
            elif hasattr(self._config, "_data"):
                config_dict = dict(self._config._data)
            else:
                # Fallback: reconstruct from known keys
                config_dict = {
                    "storage": {
                        "session": self._config.get("storage.session", {}),
                        "vector": self._config.get("storage.vector", {}),
                    }
                }
        except Exception as e:
            logger.warning(f"Could not extract config dict for validation: {e}")
            config_dict = {"storage": {}}

        validator = StorageConfigValidator(strict=strict)
        self._validation_result = validator.validate(config_dict)

        if not self._validation_result.valid:
            error_msg = self._validation_result.format_report()
            logger.error(f"Storage configuration validation failed:\n{error_msg}")
            raise ConfigError(
                f"Invalid storage configuration. {len(self._validation_result.errors)} error(s) found."
            )

        # Log warnings even if validation passed
        if self._validation_result.warnings:
            logger.warning(
                f"Storage configuration has {len(self._validation_result.warnings)} warning(s):\n"
                + "\n".join(f"  - {w}" for w in self._validation_result.warnings)
            )

        logger.debug("Storage configuration validation passed.")

    def _load_observability_config(self) -> ObservabilityConfig:
        """
        Load observability configuration from the config object.

        Attempts to load from storage.observability section of config,
        falls back to defaults if not present.

        Returns:
            ObservabilityConfig instance.
        """
        try:
            obs_config_dict = self._config.get("storage.observability", {})
            if obs_config_dict:
                return ObservabilityConfig.from_dict(obs_config_dict)
        except Exception as e:
            logger.warning(f"Could not load observability config from file: {e}")

        # Try environment variables as fallback
        try:
            env_config = ObservabilityConfig.from_environment()
            if env_config != DEFAULT_OBSERVABILITY_CONFIG:
                logger.debug("Using observability config from environment variables")
                return env_config
        except Exception as e:
            logger.debug(f"No valid observability config from environment: {e}")

        # Use defaults
        logger.debug("Using default observability configuration")
        return DEFAULT_OBSERVABILITY_CONFIG

    @property
    def validation_result(self) -> ValidationResult | None:
        """Get the configuration validation result."""
        return self._validation_result

    @property
    def observability_config(self) -> ObservabilityConfig:
        """Get the observability configuration."""
        return self._observability_config

    @property
    def instrumentation(self) -> StorageInstrumentation | None:
        """Get the storage instrumentation instance (if initialized)."""
        return self._instrumentation

    @property
    def metrics_collector(self) -> MetricsCollector | None:
        """Get the metrics collector instance (if initialized)."""
        return self._metrics_collector

    @property
    def event_logger(self) -> EventLogger | None:
        """Get the event logger instance (if initialized)."""
        return self._event_logger

    @property
    def health_manager(self) -> StorageHealthManager:
        """Get the health manager for monitoring."""
        return self._health_manager

    async def initialize_storages(
        self,
        enable_health_monitoring: bool = True,
        run_initial_health_check: bool = True,
        enable_observability: bool = True,
    ) -> None:
        """
        Parses storage configuration and creates storage backend instances.

        STORAGE SYSTEM V2 (Phase 1): Now includes:
        - Configuration parsing with environment variable resolution
        - Backend instantiation with error handling
        - Health monitor registration
        - Optional initial health check

        STORAGE SYSTEM V2 (Phase 4 - PANOPTICON): Now also includes:
        - Instrumentation layer for operation timing
        - Metrics collector initialization
        - Event logger initialization

        Args:
            enable_health_monitoring: Whether to start background health monitoring.
            run_initial_health_check: Whether to run health check immediately after init.
            enable_observability: Whether to initialize observability components.

        Raises:
            ConfigError: If storage configuration is invalid.
            StorageError: If storage backend instantiation fails.
        """
        if self._initialized:
            logger.warning("StorageManager already initialized; skipping re-initialization.")
            return

        # Parse configuration
        await self._parse_session_storage_config()
        await self._parse_vector_storage_config()

        # Phase 4: Initialize observability components (before backend instantiation)
        if enable_observability and self._observability_config.enabled:
            await self._initialize_observability()

        # Instantiate storage backends
        await self._instantiate_session_storage()
        await self._instantiate_vector_storage()

        # Start health monitoring if enabled
        if enable_health_monitoring and self._health_config.enabled:
            await self._health_manager.start_all()
            logger.debug("Storage health monitoring started.")

        # Run initial health check if requested
        if run_initial_health_check:
            await self._run_initial_health_checks()

        self._initialized = True
        logger.debug("Storage backends initialized successfully.")

    async def _parse_session_storage_config(self) -> None:
        """
        Parses session storage configuration from the config object.

        Sets _session_storage_type and _session_storage_config based on
        the [storage.session] section of the configuration.
        """
        session_storage_config = self._config.get("storage.session", {})
        session_storage_type = session_storage_config.get("type")

        if not session_storage_type:
            logger.warning(
                "No session storage type configured ('storage.session.type'). Session persistence disabled."
            )
            self._session_storage_type = None
            return

        if session_storage_type.lower() not in SESSION_STORAGE_MAP:
            raise ConfigError(
                f"Unsupported session storage type configured: '{session_storage_type}'. "
                f"Available types: {list(SESSION_STORAGE_MAP.keys())}"
            )

        self._session_storage_type = session_storage_type.lower()
        self._session_storage_config = session_storage_config
        logger.debug(f"Session storage type '{session_storage_type}' configured.")

    async def _parse_vector_storage_config(self) -> None:
        """
        Parses vector storage configuration from the config object.

        Sets _vector_storage_type and _vector_storage_config based on
        the [storage.vector] section of the configuration.
        """
        vector_storage_config = self._config.get("storage.vector", {})
        vector_storage_type = vector_storage_config.get("type")

        if not vector_storage_type:
            logger.warning(
                "No vector storage type configured ('storage.vector.type'). RAG functionality will be unavailable."
            )
            self._vector_storage_type = None
            return

        if vector_storage_type.lower() not in VECTOR_STORAGE_MAP:
            raise ConfigError(
                f"Unsupported vector storage type configured: '{vector_storage_type}'. "
                f"Available types: {list(VECTOR_STORAGE_MAP.keys())}."
            )

        self._vector_storage_type = vector_storage_type.lower()
        self._vector_storage_config = vector_storage_config
        logger.debug(f"Vector storage type '{vector_storage_type}' configured.")

    async def _initialize_observability(self) -> None:
        """
        Initialize Phase 4 observability components.

        Creates and configures:
        - StorageInstrumentation: Operation timing and slow query detection
        - MetricsCollector: Counter, histogram, and gauge metrics
        - EventLogger: Persistent event logging to database

        These components are initialized before backend instantiation so that
        backends can optionally receive instrumentation callbacks.
        """
        obs_config = self._observability_config

        if not obs_config.enabled:
            logger.debug("Observability disabled; skipping initialization")
            return

        try:
            # Initialize metrics collector first (needed by instrumentation)
            if obs_config.metrics_enabled:
                metrics_cfg = obs_config.get_metrics_config()
                metrics_config = MetricsConfig(
                    enabled=metrics_cfg.get("enabled", True),
                    backend=MetricsBackendType(metrics_cfg.get("backend", "memory")),
                    prefix=metrics_cfg.get("prefix", "llmcore_storage"),
                    default_labels=metrics_cfg.get("default_labels", {}),
                    prometheus_port=metrics_cfg.get("prometheus_port", 9090),
                    histogram_buckets=tuple(metrics_cfg.get("histogram_buckets", ())),
                )
                self._metrics_collector = MetricsCollector(config=metrics_config)
                logger.debug(
                    f"Metrics collector initialized (backend: {obs_config.metrics_backend})"
                )

            # Initialize instrumentation with metrics collector
            inst_config = InstrumentationConfig(**obs_config.get_instrumentation_config())
            self._instrumentation = StorageInstrumentation(
                config=inst_config,
                metrics_collector=self._metrics_collector,  # May be None if metrics disabled
            )
            logger.debug("Storage instrumentation initialized")

            # Initialize event logger (requires database pool)
            # Note: Event logger is initialized lazily when a pool becomes available
            if obs_config.event_logging_enabled:
                event_config = EventLoggerConfig(**obs_config.get_event_logger_config())
                self._event_logger = EventLogger(config=event_config)
                logger.debug("Event logger initialized (database pool will be connected later)")

            logger.debug("Observability components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize observability: {e}", exc_info=True)
            # Non-fatal: observability failure shouldn't prevent storage from working
            logger.warning("Continuing without observability features")

    async def _connect_event_logger_to_pool(self) -> None:
        """
        Connect the event logger to the session storage database pool.

        Called after session storage is initialized to enable persistent event logging.
        """
        if self._event_logger is None:
            return

        # Try to get pool from session storage
        if (
            self._session_storage_instance is not None
            and hasattr(self._session_storage_instance, "_pool")
            and self._session_storage_instance._pool is not None
        ):
            try:
                await self._event_logger.set_pool(self._session_storage_instance._pool)
                logger.debug("Event logger connected to session storage pool")
            except Exception as e:
                logger.warning(f"Could not connect event logger to pool: {e}")

    async def _instantiate_session_storage(self) -> None:
        """
        Creates and initializes the session storage backend instance.

        STORAGE SYSTEM V2 (Phase 1): Now includes:
        - Health monitor registration for the backend
        - Structured logging with backend details

        STORAGE SYSTEM V2 (Phase 4): Now also includes:
        - Connection of event logger to database pool

        Raises:
            StorageError: If instantiation or initialization fails.
        """
        if self._session_storage_type is None:
            logger.info("Session storage not configured; skipping instantiation.")
            self._session_storage_instance = None
            return

        session_storage_cls = SESSION_STORAGE_MAP[self._session_storage_type]

        try:
            self._session_storage_instance = session_storage_cls()
            await self._session_storage_instance.initialize(self._session_storage_config)

            # Register health monitor for session storage
            await self._register_session_health_monitor()

            # Phase 4: Connect event logger to database pool
            await self._connect_event_logger_to_pool()

            logger.debug(
                f"Session storage backend '{self._session_storage_type}' instantiated and initialized.",
                extra={
                    "backend_type": self._session_storage_type,
                    "backend_class": session_storage_cls.__name__,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to instantiate session storage backend '{self._session_storage_type}': {e}",
                exc_info=True,
            )
            raise StorageError(f"Session storage instantiation failed: {e}")

    async def _register_session_health_monitor(self) -> None:
        """Register health monitor for the session storage backend."""
        if self._session_storage_instance is None:
            return

        backend_name = f"session_{self._session_storage_type}"

        # Create appropriate health check function based on backend type
        try:
            if self._session_storage_type == "postgres":
                # Get the connection pool from the storage instance
                if (
                    hasattr(self._session_storage_instance, "_pool")
                    and self._session_storage_instance._pool
                ):
                    check_fn = await create_postgres_health_check(
                        self._session_storage_instance._pool
                    )
                else:
                    logger.warning(
                        "Postgres session storage has no pool; health monitoring unavailable"
                    )
                    return

            elif self._session_storage_type == "sqlite":
                # Get the connection from the storage instance
                if (
                    hasattr(self._session_storage_instance, "_conn")
                    and self._session_storage_instance._conn
                ):
                    check_fn = await create_sqlite_health_check(
                        self._session_storage_instance._conn
                    )
                else:
                    logger.warning(
                        "SQLite session storage has no connection; health monitoring unavailable"
                    )
                    return

            elif self._session_storage_type == "json":
                # JSON storage doesn't need health monitoring (filesystem)
                logger.debug("JSON session storage does not require health monitoring")
                return

            else:
                logger.warning(
                    f"No health check available for session storage type: {self._session_storage_type}"
                )
                return

            monitor = StorageHealthMonitor(
                backend_name=backend_name,
                backend_type="session",
                check_fn=check_fn,
                config=self._health_config,
            )
            self._health_manager.register_monitor(monitor)
            logger.debug(f"Health monitor registered for {backend_name}")

        except Exception as e:
            logger.warning(f"Could not register health monitor for session storage: {e}")

    async def _instantiate_vector_storage(self) -> None:
        """
        Creates and initializes the vector storage backend instance.

        STORAGE SYSTEM V2 (Phase 1): Now includes:
        - Health monitor registration for the backend
        - Structured logging with backend details

        Raises:
            StorageError: If instantiation or initialization fails.
        """
        if self._vector_storage_type is None:
            logger.info("Vector storage not configured; skipping instantiation.")
            self._vector_storage_instance = None
            return

        vector_storage_cls = VECTOR_STORAGE_MAP[self._vector_storage_type]

        try:
            self._vector_storage_instance = vector_storage_cls()
            await self._vector_storage_instance.initialize(self._vector_storage_config)

            # Register health monitor for vector storage
            await self._register_vector_health_monitor()

            logger.debug(
                f"Vector storage backend '{self._vector_storage_type}' instantiated and initialized.",
                extra={
                    "backend_type": self._vector_storage_type,
                    "backend_class": vector_storage_cls.__name__,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to instantiate vector storage backend '{self._vector_storage_type}': {e}",
                exc_info=True,
            )
            raise StorageError(f"Vector storage instantiation failed: {e}")

    async def _register_vector_health_monitor(self) -> None:
        """Register health monitor for the vector storage backend."""
        if self._vector_storage_instance is None:
            return

        backend_name = f"vector_{self._vector_storage_type}"

        try:
            if self._vector_storage_type == "pgvector":
                # Get the connection pool from the storage instance
                if (
                    hasattr(self._vector_storage_instance, "_pool")
                    and self._vector_storage_instance._pool
                ):
                    check_fn = await create_postgres_health_check(
                        self._vector_storage_instance._pool
                    )
                else:
                    logger.warning("PgVector storage has no pool; health monitoring unavailable")
                    return

            elif self._vector_storage_type == "chromadb":
                # Get the ChromaDB client
                if (
                    hasattr(self._vector_storage_instance, "_client")
                    and self._vector_storage_instance._client
                ):
                    check_fn = await create_chromadb_health_check(
                        self._vector_storage_instance._client
                    )
                else:
                    logger.warning("ChromaDB storage has no client; health monitoring unavailable")
                    return

            else:
                logger.warning(
                    f"No health check available for vector storage type: {self._vector_storage_type}"
                )
                return

            monitor = StorageHealthMonitor(
                backend_name=backend_name,
                backend_type="vector",
                check_fn=check_fn,
                config=self._health_config,
            )
            self._health_manager.register_monitor(monitor)
            logger.debug(f"Health monitor registered for {backend_name}")

        except Exception as e:
            logger.warning(f"Could not register health monitor for vector storage: {e}")

    async def _run_initial_health_checks(self) -> None:
        """Run initial health checks on all registered backends."""
        logger.debug("Running initial storage health checks...")

        for backend_name in list(self._health_manager._monitors.keys()):
            try:
                result = await self._health_manager.run_health_check(backend_name)
                if result:
                    if result.status == HealthStatus.HEALTHY:
                        logger.debug(
                            f"Initial health check passed for {backend_name}: "
                            f"latency={result.latency_ms:.1f}ms"
                        )
                    else:
                        logger.warning(
                            f"Initial health check failed for {backend_name}: "
                            f"status={result.status.value}, error={result.error_message}"
                        )
            except Exception as e:
                logger.warning(f"Initial health check error for {backend_name}: {e}")

    @property
    def session_storage(self) -> BaseSessionStorage:
        """
        Returns the initialized session storage backend instance.

        REFACTORED (Step 1.3): Changed from get_session_storage(db_session) method
        to a simple property that returns the internally-held instance.

        Returns:
            BaseSessionStorage: The initialized storage backend instance.

        Raises:
            StorageError: If session storage is not configured or not initialized.
        """
        if self._session_storage_instance is None:
            if self._session_storage_type is None:
                raise StorageError(
                    "Session storage is not configured ('storage.session.type' missing)."
                )
            else:
                raise StorageError(
                    "Session storage not initialized. Call initialize_storages() first."
                )

        return self._session_storage_instance

    @property
    def vector_storage(self) -> BaseVectorStorage:
        """
        Returns the initialized vector storage backend instance.

        REFACTORED (Step 1.3): Changed from get_vector_storage(db_session) method
        to a simple property that returns the internally-held instance.

        Returns:
            BaseVectorStorage: The initialized storage backend instance.

        Raises:
            StorageError: If vector storage is not configured or not initialized.
        """
        if self._vector_storage_instance is None:
            if self._vector_storage_type is None:
                raise StorageError(
                    "Vector storage is not configured ('storage.vector.type' missing). RAG is unavailable."
                )
            else:
                raise StorageError(
                    "Vector storage not initialized. Call initialize_storages() first."
                )

        return self._vector_storage_instance

    # --- Convenience methods for episodic memory (delegate to session storage) ---

    async def add_episode(self, episode: Episode) -> None:
        """
        Adds a new episode to the episodic memory log through the session storage backend.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the session_storage property.

        Args:
            episode: The Episode object to add.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        await self.session_storage.add_episode(episode)
        logger.debug(
            f"Episode '{episode.episode_id}' added to session '{episode.session_id}' via StorageManager."
        )

    async def get_episodes(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]:
        """
        Retrieves episodes for a given session through the session storage backend.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the session_storage property.

        Args:
            session_id: The ID of the session to retrieve episodes for.
            limit: The maximum number of episodes to return.
            offset: The number of episodes to skip (for pagination).

        Returns:
            A list of Episode objects.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        episodes = await self.session_storage.get_episodes(session_id, limit, offset)
        logger.debug(
            f"Retrieved {len(episodes)} episodes for session '{session_id}' via StorageManager."
        )
        return episodes

    async def get_episode_count(self, session_id: str) -> int:
        """
        Gets the total count of episodes for a session.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the session_storage property.

        Args:
            session_id: The ID of the session to count episodes for.

        Returns:
            The total number of episodes for the session.

        Raises:
            StorageError: If session storage is not configured or if the operation fails.
        """
        # Get episodes in batches to count total without loading all into memory
        total_count = 0
        batch_size = 1000
        offset = 0

        while True:
            batch = await self.session_storage.get_episodes(
                session_id, limit=batch_size, offset=offset
            )
            batch_count = len(batch)
            total_count += batch_count

            if batch_count < batch_size:
                # We've reached the end
                break
            offset += batch_size

        logger.debug(f"Total episode count for session '{session_id}': {total_count}")
        return total_count

    async def list_vector_collection_names(self) -> list[str]:
        """
        Lists the names of all available collections in the configured vector store.

        REFACTORED (Step 1.3): Removed db_session parameter. Now uses internally-held
        storage instance via the vector_storage property.

        Returns:
            A list of collection name strings.

        Raises:
            StorageError: If vector storage is not configured, failed to initialize,
                          or if the backend fails to list collections.
        """
        try:
            return await self.vector_storage.list_collection_names()
        except NotImplementedError:
            logger.error(
                f"Vector storage backend {type(self.vector_storage).__name__} does not implement list_collection_names."
            )
            raise StorageError(
                f"Listing collections not supported by {type(self.vector_storage).__name__}."
            )
        except Exception as e:
            logger.error(
                f"Error listing vector collections via {type(self.vector_storage).__name__}: {e}",
                exc_info=True,
            )
            raise StorageError(f"Failed to list vector collections: {e}")

    async def close_storages(self) -> None:
        """
        Closes connections for all initialized storage backends.

        STORAGE SYSTEM V2 (Phase 1): Now also stops health monitoring.
        STORAGE SYSTEM V2 (Phase 4): Now also shuts down observability components.
        """
        # Stop health monitoring first
        if self._health_manager:
            try:
                await self._health_manager.stop_all()
                logger.debug("Health monitoring stopped.")
            except Exception as e:
                logger.warning(f"Error stopping health monitoring: {e}")

        # Phase 4: Shutdown observability components
        await self._shutdown_observability()

        if self._session_storage_instance:
            try:
                await self._session_storage_instance.close()
                logger.info("Session storage backend closed.")
            except Exception as e:
                logger.warning(f"Error closing session storage: {e}")

        if self._vector_storage_instance:
            try:
                await self._vector_storage_instance.close()
                logger.info("Vector storage backend closed.")
            except Exception as e:
                logger.warning(f"Error closing vector storage: {e}")

        self._initialized = False
        logger.info("Storage manager cleanup complete.")

    async def _shutdown_observability(self) -> None:
        """
        Shutdown Phase 4 observability components.

        Flushes pending events and cleanly shuts down collectors.
        """
        # Shutdown event logger first (flush pending events)
        if self._event_logger:
            try:
                await self._event_logger.shutdown()
                logger.debug("Event logger shut down.")
            except Exception as e:
                logger.warning(f"Error shutting down event logger: {e}")
            self._event_logger = None

        # Shutdown metrics collector
        if self._metrics_collector:
            try:
                if hasattr(self._metrics_collector, "shutdown"):
                    await self._metrics_collector.shutdown()
                logger.debug("Metrics collector shut down.")
            except Exception as e:
                logger.warning(f"Error shutting down metrics collector: {e}")
            self._metrics_collector = None

        # Clear instrumentation
        if self._instrumentation:
            self._instrumentation = None
            logger.debug("Instrumentation cleared.")

    async def close(self) -> None:
        """
        Alias for close_storages() to maintain API compatibility.

        LLMCore.close() calls StorageManager.close(), so this method
        delegates to close_storages() which does the actual cleanup.
        """
        await self.close_storages()

    # =========================================================================
    # HEALTH API (Phase 1 - PRIMORDIUM)
    # =========================================================================

    def is_healthy(self, backend_name: str | None = None) -> bool:
        """
        Check if storage backend(s) are healthy.

        This is a quick check suitable for use in request paths or
        load balancer health endpoints.

        Args:
            backend_name: Specific backend to check (e.g., "session_postgres").
                         If None, checks all backends.

        Returns:
            True if the specified backend(s) are healthy.
        """
        return self._health_manager.is_healthy(backend_name)

    def get_health_report(self, backend_name: str | None = None) -> dict[str, Any]:
        """
        Get detailed health report for storage backend(s).

        Provides comprehensive health metrics including:
        - Current status and circuit breaker state
        - Latency statistics
        - Failure counts and uptime percentage
        - Last check timestamps

        Args:
            backend_name: Specific backend to report on.
                         If None, reports on all backends.

        Returns:
            Health report dictionary.
        """
        return self._health_manager.get_report(backend_name)

    async def run_health_check(self, backend_name: str) -> dict[str, Any] | None:
        """
        Manually trigger a health check for a specific backend.

        Useful for diagnostics or forced health updates.

        Args:
            backend_name: Backend to check (e.g., "session_postgres", "vector_chromadb").

        Returns:
            HealthCheckResult as dict, or None if backend not found.
        """
        result = await self._health_manager.run_health_check(backend_name)
        return result.to_dict() if result else None

    @property
    def health_status(self) -> dict[str, Any]:
        """
        Quick access to overall health status.

        Returns:
            Dictionary with overall_healthy flag and per-backend status.
        """
        return {
            "overall_healthy": self.is_healthy(),
            "session_storage": {
                "configured": self._session_storage_type is not None,
                "type": self._session_storage_type,
                "healthy": self.is_healthy(f"session_{self._session_storage_type}")
                if self._session_storage_type
                else None,
            },
            "vector_storage": {
                "configured": self._vector_storage_type is not None,
                "type": self._vector_storage_type,
                "healthy": self.is_healthy(f"vector_{self._vector_storage_type}")
                if self._vector_storage_type
                else None,
            },
        }

    # =========================================================================
    # DIAGNOSTICS (Phase 1 - PRIMORDIUM / Phase 4 - PANOPTICON)
    # =========================================================================

    def get_storage_info(self) -> dict[str, Any]:
        """
        Get information about configured storage backends.

        Useful for diagnostics and CLI tools.

        Returns:
            Dictionary with storage configuration and status.
        """
        # Gather instrumentation statistics if available
        instrumentation_stats = {}
        if self._instrumentation:
            try:
                instrumentation_stats = self._instrumentation.get_statistics()
            except Exception:
                instrumentation_stats = {"error": "Could not retrieve statistics"}

        return {
            "initialized": self._initialized,
            "session_storage": {
                "type": self._session_storage_type,
                "config_keys": list(self._session_storage_config.keys())
                if self._session_storage_config
                else [],
                "instance_class": type(self._session_storage_instance).__name__
                if self._session_storage_instance
                else None,
            },
            "vector_storage": {
                "type": self._vector_storage_type,
                "config_keys": list(self._vector_storage_config.keys())
                if self._vector_storage_config
                else [],
                "instance_class": type(self._vector_storage_instance).__name__
                if self._vector_storage_instance
                else None,
            },
            "health_monitoring": {
                "enabled": self._health_config.enabled,
                "check_interval_seconds": self._health_config.check_interval_seconds,
                "registered_monitors": list(self._health_manager._monitors.keys())
                if self._health_manager
                else [],
            },
            "validation": {
                "performed": self._validation_result is not None,
                "valid": self._validation_result.valid if self._validation_result else None,
                "error_count": len(self._validation_result.errors)
                if self._validation_result
                else 0,
                "warning_count": len(self._validation_result.warnings)
                if self._validation_result
                else 0,
            },
            # Phase 4: Observability status
            "observability": {
                "enabled": self._observability_config.enabled,
                "instrumentation": {
                    "active": self._instrumentation is not None,
                    "slow_query_threshold_seconds": self._observability_config.slow_query_threshold_seconds,
                    "log_queries": self._observability_config.log_queries,
                    "statistics": instrumentation_stats,
                },
                "metrics": {
                    "enabled": self._observability_config.metrics_enabled,
                    "backend": self._observability_config.metrics_backend,
                    "collector_active": self._metrics_collector is not None,
                },
                "event_logging": {
                    "enabled": self._observability_config.event_logging_enabled,
                    "table_name": self._observability_config.event_table_name,
                    "retention_days": self._observability_config.event_retention_days,
                    "logger_active": self._event_logger is not None,
                    "pool_connected": (
                        self._event_logger.pool is not None
                        if self._event_logger and hasattr(self._event_logger, "pool")
                        else False
                    ),
                },
                "tracing": {
                    "enabled": self._observability_config.tracing_enabled,
                    "backend": self._observability_config.tracing_backend,
                },
            },
        }

    def get_observability_statistics(self) -> dict[str, Any]:
        """
        Get detailed observability statistics.

        Returns statistics from instrumentation, metrics collector, and event logger.
        Useful for dashboards and monitoring.

        Returns:
            Dictionary with detailed observability statistics.
        """
        stats: dict[str, Any] = {
            "enabled": self._observability_config.enabled,
            "instrumentation": None,
            "metrics": None,
            "event_logger": None,
        }

        if self._instrumentation:
            stats["instrumentation"] = self._instrumentation.get_statistics()

        if self._metrics_collector:
            try:
                stats["metrics"] = self._metrics_collector.get_statistics()
            except Exception:
                stats["metrics"] = {"error": "Could not retrieve metrics statistics"}

        if self._event_logger:
            try:
                stats["event_logger"] = {
                    "pool_connected": hasattr(self._event_logger, "pool")
                    and self._event_logger.pool is not None,
                    "queue_size": getattr(self._event_logger, "_queue_size", 0)
                    if hasattr(self._event_logger, "_queue_size")
                    else 0,
                }
            except Exception:
                stats["event_logger"] = {"error": "Could not retrieve event logger statistics"}

        return stats
