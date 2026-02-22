# src/llmcore/config/models.py
# llmcore/src/llmcore/config/models.py
"""
Pydantic models for llmcore configuration validation.

This module provides optional Pydantic models for validating configuration
sections, particularly the [semantiscan] section which mirrors the schema
from semantiscan.config.models.AppConfig.

These models are optional - llmcore primarily uses confy's dictionary-based
configuration. However, these models can be used for:
1. Type validation of configuration values
2. IDE autocomplete support
3. Documentation generation
4. Runtime validation when instantiating semantiscan components

Note: The models here should be kept in sync with semantiscan/config/models.py
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ==============================================================================
# Semantiscan Configuration Models
# ==============================================================================


class SemantiscanDatabaseConfig(BaseModel):
    """
    Vector database configuration for semantiscan.

    Controls where semantiscan stores embedded code chunks for RAG retrieval.
    Should match llmcore's [storage.vector.path] for unified storage.
    """

    type: str = Field("chromadb", description="Type of vector database (only chromadb supported)")
    path: str = Field(
        "~/.llmcore/chroma_db", description="Path to the directory where ChromaDB stores its data"
    )
    collection_name: str = Field(
        "default_semantiscan", description="Name of the collection within ChromaDB"
    )


class SemantiscanMetadataStoreConfig(BaseModel):
    """
    External metadata store configuration for semantiscan.

    Optional SQLite database for tracking ingestion history, Git metadata, etc.
    This stores rich metadata that doesn't fit well in the vector store.
    """

    enable: bool = Field(False, description="Enable the external metadata store")
    type: str = Field("sqlite", description="Type of external store (sqlite or postgresql)")
    path: str | None = Field(
        "~/.local/share/semantiscan/metadata.db",
        description="Path to the SQLite database file (required if type is sqlite)",
    )
    connection_string: str | None = Field(
        "", description="Connection string for PostgreSQL (required if type is postgresql)"
    )
    table_name: str = Field(
        "chunk_metadata", description="Name of the table to store rich chunk metadata"
    )
    ingestion_log_table_name: str = Field(
        "ingestion_log", description="Name of the table to track ingestion state per repo/branch"
    )
    file_history_table_name: str = Field(
        "file_history", description="Name of the table to track file changes per commit"
    )

    @model_validator(mode="after")
    def check_path_or_conn_string(self) -> "SemantiscanMetadataStoreConfig":
        """Validate that appropriate connection info is provided when enabled."""
        if self.enable:
            if self.type == "sqlite" and not self.path:
                raise ValueError(
                    "Metadata store 'path' is required when type is 'sqlite' and store is enabled."
                )
            if self.type == "postgresql" and not self.connection_string:
                raise ValueError(
                    "Metadata store 'connection_string' is required when type is "
                    "'postgresql' and store is enabled."
                )
            if self.type not in ["sqlite", "postgresql"]:
                raise ValueError(f"Unsupported metadata store type: {self.type}")
        return self


class SemantiscanEmbeddingModelConfig(BaseModel):
    """Configuration for a single embedding model used by semantiscan."""

    provider: str = Field(
        ..., description="Embedding provider (sentence-transformers, openai, ollama)"
    )
    model_name: str = Field(..., description="Name of the specific model to use")
    device: str = Field("cpu", description="Device for local models (cpu, cuda, mps)")
    api_key_env: str | None = Field("", description="Environment variable name holding the API key")
    max_request_tokens: int = Field(8000, description="Max tokens per API request batch")
    base_url: str | None = Field("", description="Base URL for API (Ollama, custom endpoints)")
    tokenizer_name: str | None = Field(
        "", description="Specific tokenizer name for token estimation"
    )
    uses_doc_query_prefixes: bool = Field(
        False, description="Whether model requires different prefixes for documents vs queries"
    )
    query_prefix: str | None = Field(
        "", description="Prefix for queries if uses_doc_query_prefixes is true"
    )
    document_prefix: str | None = Field(
        "", description="Prefix for documents if uses_doc_query_prefixes is true"
    )


class SemantiscanEmbeddingsConfig(BaseModel):
    """
    Embedding models configuration for semantiscan.

    Defines which embedding models are available for chunking ingestion.
    Semantiscan will use llmcore's EmbeddingManager for actual embedding generation.
    """

    default_model: str = Field(
        "sentence_transformer_local",
        description="Key of the default embedding model configuration to use",
    )
    models: dict[str, SemantiscanEmbeddingModelConfig] = Field(
        default_factory=dict, description="Dictionary of available embedding model configurations"
    )

    @field_validator("models")
    @classmethod
    def check_default_model_exists(cls, v, info):
        """Ensure the default model is defined in the models dictionary."""
        values = info.data
        default_key = values.get("default_model")
        if default_key and default_key not in v:
            raise ValueError(
                f"Default embedding model '{default_key}' is not defined "
                f"in the 'models' dictionary."
            )
        return v


class SemantiscanChunkingStrategyConfig(BaseModel):
    """Configuration for a single chunking strategy."""

    extensions: list[str] = Field(
        default_factory=list, description="File extensions this strategy applies to"
    )
    grammar: str | None = Field("", description="Name of the ANTLR grammar (for ANTLR strategy)")
    entry_points: list[str] = Field(
        default_factory=list, description="ANTLR rule names to treat as chunk boundaries"
    )
    parser: str | None = Field(
        "", description="Name of the format-specific parser (for format strategy)"
    )
    method: str | None = Field(
        "", description="Name of the agnostic chunking method (for agnostic strategy)"
    )
    strategy_sequence: list[dict[str, Any]] = Field(
        default_factory=list, description="Sequence of strategies to try"
    )
    hybrid_content: bool = Field(False, description="Combine related elements in ANTLR chunks")


class SemantiscanRecursiveSplitterParams(BaseModel):
    """Parameters for the RecursiveSplitter chunking method."""

    chunk_size: int = Field(1000, gt=0, description="Target chunk size in characters")
    chunk_overlap: int = Field(150, ge=0, description="Overlap between chunks")


class SemantiscanLineSplitterParams(BaseModel):
    """Parameters for the LineSplitter chunking method."""

    lines_per_chunk: int = Field(50, gt=0, description="Number of lines per chunk")


class SemantiscanSubChunkerParams(BaseModel):
    """Parameters for the SubChunker (handles oversized chunks)."""

    chunk_size: int = Field(500, gt=0, description="Chunk size for sub-chunking oversized chunks")
    chunk_overlap: int = Field(50, ge=0, description="Overlap for sub-chunking")


class SemantiscanChunkingParameters(BaseModel):
    """Collection of parameters for different chunking methods."""

    RecursiveSplitter: SemantiscanRecursiveSplitterParams = Field(
        default_factory=SemantiscanRecursiveSplitterParams
    )
    LineSplitter: SemantiscanLineSplitterParams = Field(
        default_factory=SemantiscanLineSplitterParams
    )
    SubChunker: SemantiscanSubChunkerParams = Field(default_factory=SemantiscanSubChunkerParams)


class SemantiscanChunkingConfig(BaseModel):
    """
    Chunking configuration for semantiscan.

    Defines how different file types are parsed and chunked during ingestion.
    """

    default_strategy: str = Field(
        "RecursiveSplitter", description="Default agnostic method if no specific rule matches"
    )
    strategies: dict[str, SemantiscanChunkingStrategyConfig] = Field(
        default_factory=dict, description="Mapping of strategy names to their configurations"
    )
    parameters: SemantiscanChunkingParameters = Field(
        default_factory=SemantiscanChunkingParameters,
        description="Parameters for different chunking methods",
    )


class SemantiscanIngestionGitConfig(BaseModel):
    """Git-specific ingestion configuration."""

    enabled: bool = Field(False, description="Enable Git-aware ingestion tracking")
    default_ref: str = Field("main", description="Default branch to ingest")
    ingestion_mode: Literal["snapshot", "historical", "historical_delta", "incremental"] = Field(
        "snapshot",
        description="Ingestion mode: snapshot, historical, historical_delta, or incremental",
    )
    historical_start_ref: str | None = Field(
        "", description="Starting Git reference for historical modes"
    )
    enable_commit_analysis: bool = Field(
        False, description="Extract commit messages, authors, etc."
    )
    enable_commit_llm_analysis: bool = Field(
        False, description="Use LLM to analyze/summarize commit messages"
    )
    commit_llm_provider_key: str | None = Field(
        "", description="LLM provider key for commit analysis"
    )
    commit_llm_prompt_template: str | None = Field(
        "", description="Custom prompt template for commit analysis"
    )
    commit_message_filter_regex: list[str] = Field(
        default_factory=list, description="Regex patterns to filter out certain commit messages"
    )

    @model_validator(mode="after")
    def check_llm_analysis_config(self) -> "SemantiscanIngestionGitConfig":
        """Validate that LLM provider is set if LLM analysis is enabled."""
        if self.enable_commit_llm_analysis and not self.commit_llm_provider_key:
            raise ValueError(
                "If 'enable_commit_llm_analysis' is true, 'commit_llm_provider_key' must be set."
            )
        return self


class SemantiscanIngestionConfig(BaseModel):
    """Ingestion pipeline configuration."""

    embedding_workers: int = Field(
        4, gt=0, description="Number of parallel workers for embedding generation"
    )
    batch_size: int = Field(
        100, gt=0, description="Number of chunks to process in parallel batches"
    )
    git: SemantiscanIngestionGitConfig = Field(
        default_factory=SemantiscanIngestionGitConfig,
        description="Git-aware ingestion configuration",
    )


class SemantiscanLLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider used by semantiscan."""

    provider: str = Field(..., description="LLM provider (ollama, openai, anthropic, etc.)")
    model_name: str = Field(..., description="Name of the specific LLM model")
    base_url: str | None = Field("", description="Base URL for API")
    api_key_env: str | None = Field("", description="Environment variable name holding the API key")
    tokenizer_name: str | None = Field(
        "", description="Specific tokenizer name for context estimation"
    )
    context_buffer: int = Field(
        200, ge=0, description="Safety buffer (in tokens) below the LLM's context window limit"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed directly to the LLM API"
    )


class SemantiscanLLMConfig(BaseModel):
    """
    LLM configuration for semantiscan query answering.

    In Phase 2, semantiscan uses llmcore's LLM providers directly.
    This section is for query answering and advanced features.
    """

    default_provider: str = Field("ollama_llama3", description="Key referencing llmcore's provider")
    prompt_template_path: str | None = Field(
        "", description="Optional: Custom prompt template file path"
    )
    enable_query_rewriting: bool = Field(False, description="Enable LLM-based query expansion")
    query_rewrite_provider_key: str | None = Field(
        "", description="LLM provider for query rewriting"
    )
    show_sources_in_text: bool = Field(
        True, description="Include source references in text responses"
    )
    tokenizer_name: str | None = Field(
        "", description="Optional: Specific tokenizer for LLM context estimation"
    )
    context_buffer: int = Field(200, ge=0, description="Reserved tokens for prompt formatting")
    providers: dict[str, SemantiscanLLMProviderConfig] = Field(
        default_factory=dict, description="Dictionary of available LLM provider configurations"
    )


class SemantiscanRetrievalConfig(BaseModel):
    """
    Retrieval configuration for semantiscan.

    Controls how semantiscan retrieves relevant chunks during RAG queries.
    """

    top_k: int = Field(10, gt=0, description="Number of most relevant chunks to retrieve")
    enable_hybrid_search: bool = Field(
        False, description="Use hybrid search (semantic + keyword BM25)"
    )
    bm25_k1: float = Field(1.5, ge=0, description="BM25 parameter: term frequency saturation")
    bm25_b: float = Field(0.75, ge=0, le=1, description="BM25 parameter: length normalization")
    enrich_with_external_metadata: bool = Field(
        False, description="Include file metadata from external store in retrieval"
    )


class SemantiscanDiscoveryConfig(BaseModel):
    """
    File discovery configuration for semantiscan.

    Controls which files are included/excluded during ingestion.
    """

    use_gitignore: bool = Field(True, description="Respect .gitignore rules during file scanning")
    excluded_dirs: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "node_modules",
            ".git",
            "venv",
            ".venv",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            "htmlcov",
            ".tox",
            ".eggs",
            "*.egg-info",
            "target",
        ],
        description="Directories to always skip (even if not in .gitignore)",
    )
    excluded_files: list[str] = Field(
        default_factory=lambda: [".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.pyd"],
        description="Files to always skip",
    )


class SemantiscanLoggingConfig(BaseModel):
    """
    Logging configuration for semantiscan.

    Semantiscan-specific logging (separate from llmcore's logging).
    """

    log_level_console: str = Field(
        "INFO", description="Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_file_enabled: bool = Field(False, description="Enable file logging for semantiscan")
    log_directory: str = Field(
        "~/.local/share/semantiscan/logs", description="Directory for log files"
    )
    log_filename_template: str = Field(
        "semantiscan_{timestamp:%Y%m%d_%H%M%S}.log", description="Log filename template"
    )
    log_level_file: str = Field("DEBUG", description="File log level (more verbose than console)")
    log_format: str = Field(
        "%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)",
        description="Python logging format string for file logs",
    )

    @field_validator("log_level_console", "log_level_file")
    @classmethod
    def check_log_level(cls, v):
        """Validate log level values."""
        allowed_levels = ["DEBUG", "INFO", "VERBOSE", "WARNING", "ERROR", "CRITICAL"]
        level_upper = v.upper()
        if level_upper not in allowed_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {allowed_levels}")
        return level_upper


class SemantiscanConfig(BaseModel):
    """
    Root configuration model for semantiscan.

    This mirrors the structure defined in semantiscan.config.models.AppConfig
    and represents the complete [semantiscan] section in llmcore's config.

    Usage:
        # Validate config loaded from confy
        config_dict = llmcore_instance.config.get('semantiscan', {})
        semantiscan_config = SemantiscanConfig(**config_dict)

        # Or convert to dict for passing to semantiscan
        semantiscan_config_dict = semantiscan_config.model_dump()
    """

    enabled: bool = Field(True, description="Top-level semantiscan enablement flag")
    database: SemantiscanDatabaseConfig = Field(default_factory=SemantiscanDatabaseConfig)
    metadata_store: SemantiscanMetadataStoreConfig = Field(
        default_factory=SemantiscanMetadataStoreConfig
    )
    embeddings: SemantiscanEmbeddingsConfig = Field(default_factory=SemantiscanEmbeddingsConfig)
    chunking: SemantiscanChunkingConfig = Field(default_factory=SemantiscanChunkingConfig)
    ingestion: SemantiscanIngestionConfig = Field(default_factory=SemantiscanIngestionConfig)
    llm: SemantiscanLLMConfig = Field(default_factory=SemantiscanLLMConfig)
    retrieval: SemantiscanRetrievalConfig = Field(default_factory=SemantiscanRetrievalConfig)
    discovery: SemantiscanDiscoveryConfig = Field(default_factory=SemantiscanDiscoveryConfig)
    logging: SemantiscanLoggingConfig = Field(default_factory=SemantiscanLoggingConfig)

    @model_validator(mode="after")
    def check_enrichment_requires_store(self) -> "SemantiscanConfig":
        """Validate that metadata store is enabled if enrichment is requested."""
        if self.retrieval.enrich_with_external_metadata and not self.metadata_store.enable:
            raise ValueError(
                "Cannot enable 'enrich_with_external_metadata' in [retrieval] "
                "if [metadata_store] is not enabled."
            )
        return self

    @model_validator(mode="after")
    def check_historical_modes_require_metadata_store(self) -> "SemantiscanConfig":
        """Validate that metadata store is enabled for certain ingestion modes."""
        if self.ingestion.git.ingestion_mode in ["historical", "historical_delta", "incremental"]:
            if not self.metadata_store.enable:
                raise ValueError(
                    f"Ingestion mode '{self.ingestion.git.ingestion_mode}' "
                    f"requires [metadata_store] to be enabled."
                )
        return self


class ModelCardsConfig(BaseModel):
    """
    Configuration for the Model Card Library.

    Controls where model cards are loaded from and how the registry behaves.
    User cards override built-in cards with the same model_id.
    """

    user_cards_path: str = Field(
        "~/.config/llmcore/model_cards",
        description="Path to user-defined model cards directory. "
        "Cards here override built-in cards with the same model_id.",
    )
    auto_load: bool = Field(
        True,
        description="Automatically load model cards on first access. "
        "Set to False if you want to control loading manually.",
    )
    strict_validation: bool = Field(
        False,
        description="If True, raise errors for invalid model card files. "
        "If False, log warnings and skip invalid files.",
    )

    @field_validator("user_cards_path")
    @classmethod
    def expand_user_path(cls, v: str) -> str:
        """Expand ~ in path but keep as string for serialization."""
        # Note: Actual expansion happens at runtime in registry
        return v


# ==============================================================================
# Utility Functions
# ==============================================================================


def validate_semantiscan_config(config_dict: dict[str, Any]) -> SemantiscanConfig:
    """
    Validate a semantiscan configuration dictionary.

    Args:
        config_dict: Dictionary containing semantiscan configuration

    Returns:
        Validated SemantiscanConfig instance

    Raises:
        ValidationError: If configuration is invalid

    Example:
        ```python
        config_dict = llmcore_instance.config.get('semantiscan', {})
        validated_config = validate_semantiscan_config(config_dict)
        ```
    """
    return SemantiscanConfig(**config_dict)


def semantiscan_config_to_dict(config: SemantiscanConfig) -> dict[str, Any]:
    """
    Convert a SemantiscanConfig instance to a dictionary.

    Args:
        config: SemantiscanConfig instance

    Returns:
        Dictionary representation of the configuration

    Example:
        ```python
        config_dict = semantiscan_config_to_dict(semantiscan_config)
        ```
    """
    return config.model_dump()
