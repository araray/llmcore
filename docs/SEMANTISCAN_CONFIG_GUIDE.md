# Semantiscan Configuration in LLMCore

## Overview

As of Phase 2, Step 2.1 of the resurrection plan, `llmcore` serves as the **single source of truth** for `semantiscan` configuration. The `[semantiscan]` section in `llmcore`'s `default_config.toml` contains all configuration parameters that `semantiscan` needs for operation.

This document explains the structure and purpose of each configuration section.

## Configuration Architecture

### Before Phase 2
- `semantiscan` maintained its own `config/` directory
- Configuration was loaded from `semantiscan/config.toml.example`
- Schema defined in `semantiscan/config/models.py`

### After Phase 2 (Current State)
- `semantiscan`'s `config/` directory has been removed
- All configuration is now in `llmcore/src/llmcore/config/default_config.toml` under `[semantiscan]`
- `llmcore` uses `confy` to load and expose configuration
- `llmchat` passes configuration to `semantiscan` pipelines via `ctx.llmcore_instance.config.semantiscan`

## Configuration Sections

### Top-Level Settings

```toml
[semantiscan]
enabled = true  # Global enable/disable flag for semantiscan features
```

---

### Database Configuration

Controls the vector store where embedded code chunks are stored.

```toml
[semantiscan.database]
type = "chromadb"  # Vector database type (currently only ChromaDB supported)
path = "~/.llmcore/chroma_db"  # Storage path (should match llmcore's vector storage)
collection_name = "default_semantiscan"  # Default collection for ingested code
```

**Key Points:**
- The `path` should be synchronized with llmcore's `[storage.vector.path]` for unified storage
- ChromaDB is currently the only supported vector store
- Collection names can be overridden per ingestion operation

---

### Metadata Store Configuration

Optional external database for rich metadata (Git info, commit history, etc.)

```toml
[semantiscan.metadata_store]
enable = false  # Enable external metadata tracking
type = "sqlite"  # Store type: "sqlite" or "postgresql" (future)
path = "~/.local/share/semantiscan/metadata.db"  # SQLite database path
connection_string = ""  # PostgreSQL connection string (future use)
table_name = "chunk_metadata"  # Table for rich chunk metadata
ingestion_log_table_name = "ingestion_log"  # Table for ingestion state tracking
file_history_table_name = "file_history"  # Table for file change tracking
```

**Key Points:**
- Required for Git-aware ingestion modes (`historical`, `historical_delta`, `incremental`)
- Stores metadata too rich/large for the vector store (commit messages, authors, etc.)
- Can be disabled for simple snapshot ingestion

**Validation Rules:**
- If `enable = true` and `type = "sqlite"`, then `path` is required
- If `enable = true` and `type = "postgresql"`, then `connection_string` is required

---

### Embedding Configuration

Defines available embedding models for code chunk vectorization.

```toml
[semantiscan.embeddings]
default_model = "sentence_transformer_local"  # Default embedding model key

[semantiscan.embeddings.models.sentence_transformer_local]
provider = "sentence-transformers"
model_name = "all-MiniLM-L6-v2"
device = "cpu"  # "cpu" | "cuda" | "mps"
max_request_tokens = 8000
base_url = ""
tokenizer_name = ""
uses_doc_query_prefixes = false
query_prefix = ""
document_prefix = ""
```

**Available Providers:**
- `sentence-transformers`: Local models (HuggingFace)
- `openai`: OpenAI embedding API
- `ollama`: Local Ollama embedding models

**Model-Specific Fields:**
- `device`: Only relevant for local models (sentence-transformers)
- `api_key_env`: Environment variable name for API key (OpenAI)
- `base_url`: API endpoint (Ollama, custom OpenAI endpoints)
- `uses_doc_query_prefixes`: For models that require different prefixes for documents vs queries (e.g., E5, BGE)

**Validation Rules:**
- The `default_model` must exist as a key in `embeddings.models`
- Each model must specify `provider` and `model_name`

---

### Chunking Configuration

Controls how different file types are parsed and split into chunks.

```toml
[semantiscan.chunking]
default_strategy = "RecursiveSplitter"  # Fallback for unknown file types

# Strategy Types:
# 1. ANTLR strategies: Syntax-aware parsing for code files
[semantiscan.chunking.strategies.python_code]
extensions = [".py", ".pyw"]
grammar = "Python3"
entry_points = ["funcdef", "async_funcdef", "classdef", "decorated"]
hybrid_content = false

# 2. Format strategies: Structure-aware parsing for data files
[semantiscan.chunking.strategies.yaml_format]
extensions = [".yaml", ".yml"]
parser = "PyYAML"

# 3. Agnostic strategies: Text-based splitting for documents
[semantiscan.chunking.strategies.markdown_agnostic]
extensions = [".md", ".markdown"]
method = "RecursiveSplitter"

# Parameters for chunking methods
[semantiscan.chunking.parameters.RecursiveSplitter]
chunk_size = 1000  # Target size in characters
chunk_overlap = 150  # Overlap between chunks

[semantiscan.chunking.parameters.SubChunker]
chunk_size = 500  # For splitting oversized chunks
chunk_overlap = 50
```

**Strategy Types:**

1. **ANTLR Strategy** (Syntax-aware)
   - Requires: `grammar` and `entry_points`
   - Best for: Source code files
   - Example languages: Python, Java, C++, Go

2. **Format Strategy** (Structure-aware)
   - Requires: `parser`
   - Best for: Structured data files
   - Supported formats: YAML, JSON, TOML, XML

3. **Agnostic Strategy** (Text-based)
   - Requires: `method`
   - Best for: Plain text, markdown, logs
   - Methods: `RecursiveSplitter`, `LineSplitter`, `WholeFile`

**Adding Custom Strategies:**
See the semantiscan documentation for details on adding new parsers and strategies.

---

### Ingestion Configuration

Controls the ingestion pipeline behavior and parallelism.

```toml
[semantiscan.ingestion]
embedding_workers = 4  # Parallel workers for embedding generation
batch_size = 100  # Chunks per batch

[semantiscan.ingestion.git]
enabled = false  # Enable Git-aware features
default_ref = "main"  # Default branch/ref to ingest
ingestion_mode = "snapshot"  # "snapshot" | "historical" | "historical_delta" | "incremental"
historical_start_ref = ""  # Starting point for historical modes
enable_commit_analysis = false  # Extract commit metadata
enable_commit_llm_analysis = false  # Use LLM to analyze commits
commit_llm_provider_key = ""  # LLM provider for commit analysis
commit_llm_prompt_template = ""  # Custom prompt template
commit_message_filter_regex = []  # Filter out certain commit messages
```

**Ingestion Modes:**

1. **snapshot** (default): Ingests current state only
   - Fastest, simplest
   - No metadata store required

2. **historical**: Ingests full commit history
   - Processes complete snapshot at each commit
   - Requires metadata store
   - Can be slow for large repos

3. **historical_delta**: Ingests commit history efficiently
   - Only processes changed files per commit
   - Tracks all file changes (A, M, D, R)
   - Requires metadata store

4. **incremental**: Ingests only new commits
   - Resumes from last ingestion point
   - Requires metadata store
   - Ideal for continuous ingestion

**Performance Tuning:**
- `embedding_workers`: Set based on CPU cores and embedding provider limits
  - Local models (sentence-transformers): CPU cores - 1
  - API models (OpenAI): Higher values may hit rate limits
- `batch_size`: Larger batches = more memory but faster processing

**Validation Rules:**
- If `ingestion_mode` is `historical`, `historical_delta`, or `incremental`, then `metadata_store.enable` must be `true`
- If `enable_commit_llm_analysis = true`, then `commit_llm_provider_key` must be set

---

### LLM Configuration

Defines LLM providers for query answering and advanced features.

```toml
[semantiscan.llm]
default_provider = "ollama_llama3"
prompt_template_path = ""  # Custom prompt template file
enable_query_rewriting = false  # LLM-based query expansion
query_rewrite_provider_key = ""  # Provider for query rewriting
show_sources_in_text = true  # Include source citations
tokenizer_name = ""  # For context estimation
context_buffer = 200  # Reserved tokens for formatting

[semantiscan.llm.providers.ollama_llama3]
provider = "ollama"
model_name = "llama3:8b"
base_url = "http://localhost:11434"
tokenizer_name = ""
context_buffer = 250

[semantiscan.llm.providers.ollama_llama3.parameters]
temperature = 0.5
num_ctx = 4096
top_p = 0.9
```

**Key Points:**
- In Phase 2, semantiscan uses llmcore's LLM providers directly
- Multiple providers can be configured for different purposes
- Query rewriting uses a separate provider (can be different from answer generation)

---

### Retrieval Configuration

Controls how relevant chunks are retrieved during RAG queries.

```toml
[semantiscan.retrieval]
top_k = 10  # Number of chunks to retrieve
enable_hybrid_search = false  # Combine semantic + keyword search
bm25_k1 = 1.5  # BM25 term frequency saturation
bm25_b = 0.75  # BM25 length normalization
enrich_with_external_metadata = false  # Include rich metadata in results
```

**Retrieval Modes:**

1. **Pure Vector Search** (default)
   - Uses only semantic similarity
   - Fast, works well for conceptual queries

2. **Hybrid Search** (`enable_hybrid_search = true`)
   - Combines semantic + BM25 keyword search
   - Uses Reciprocal Rank Fusion (RRF) to merge results
   - Better for queries with specific technical terms

**BM25 Parameters:**
- `bm25_k1`: Controls term frequency saturation (typical: 1.2-2.0)
- `bm25_b`: Controls length normalization (typical: 0.5-0.8)

**Validation Rules:**
- If `enrich_with_external_metadata = true`, then `metadata_store.enable` must be `true`

---

### Discovery Configuration

Controls file scanning and filtering during ingestion.

```toml
[semantiscan.discovery]
use_gitignore = true  # Respect .gitignore rules
excluded_dirs = [  # Always skip these directories
    "__pycache__",
    "node_modules",
    ".git",
    "venv",
    ".venv",
    "build",
    "dist"
]
excluded_files = [".DS_Store", "Thumbs.db", "*.pyc"]  # Always skip these files
```

**Key Points:**
- `.gitignore` rules are applied first (if `use_gitignore = true`)
- `excluded_dirs` and `excluded_files` are applied additionally
- Glob patterns are supported (e.g., `"*.pyc"`, `"build*"`)

---

### Logging Configuration

Semantiscan-specific logging (separate from llmcore's logging).

```toml
[semantiscan.logging]
log_level_console = "INFO"  # Console: DEBUG | INFO | WARNING | ERROR | CRITICAL
log_file_enabled = false  # Enable file logging
log_directory = "~/.local/share/semantiscan/logs"
log_filename_template = "semantiscan_{timestamp:%Y%m%d_%H%M%S}.log"
log_level_file = "DEBUG"  # File log level (typically more verbose)
log_format = "%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)"
```

**Key Points:**
- Console and file logging can have different levels
- File logs are typically more verbose for debugging
- Timestamp template uses Python `strftime` format codes

---

## Accessing Configuration from llmchat

In llmchat, semantiscan configuration is accessed via:

```python
# Get the entire semantiscan config as a dictionary
semantiscan_config_dict = ctx.llmcore_instance.config.get('semantiscan', {})

# Or access specific fields
db_path = ctx.llmcore_instance.config.get('semantiscan.database.path')
top_k = ctx.llmcore_instance.config.get('semantiscan.retrieval.top_k')
```

The configuration is then passed to semantiscan pipelines:

```python
from semantiscan.pipelines.ingest import IngestionPipeline
from semantiscan.config.models import AppConfig

# Build AppConfig from llmcore's config
semantiscan_config_dict = ctx.llmcore_instance.config.get('semantiscan', {})

# Optionally override storage paths for unification
semantiscan_config_dict['database']['path'] = ctx.llmcore_instance.config.get('storage.vector.path')

# Create AppConfig and run pipeline
app_config = AppConfig(**semantiscan_config_dict)
pipeline = IngestionPipeline(app_config)
pipeline.run(...)
```

---

## Configuration Validation

### Using Pydantic Models (Optional)

While llmcore uses confy (dictionary-based config), optional Pydantic models are provided in `llmcore/src/llmcore/config/models.py` for validation:

```python
from llmcore.config.models import (
    SemantiscanConfig,
    validate_semantiscan_config
)

# Validate config loaded from confy
config_dict = llmcore_instance.config.get('semantiscan', {})
validated_config = validate_semantiscan_config(config_dict)

# Access validated fields with type safety
print(validated_config.database.path)
print(validated_config.retrieval.top_k)
```

### Validation Rules

The Pydantic models enforce several validation rules:

1. **Default model existence**: `embeddings.default_model` must exist in `embeddings.models`

2. **Metadata store requirements**:
   - SQLite type requires `path`
   - PostgreSQL type requires `connection_string`

3. **Cross-section dependencies**:
   - `retrieval.enrich_with_external_metadata = true` requires `metadata_store.enable = true`
   - Historical ingestion modes require `metadata_store.enable = true`
   - `ingestion.git.enable_commit_llm_analysis = true` requires `commit_llm_provider_key`

---

## Environment Variables

All `[semantiscan]` settings can be overridden via environment variables:

```bash
# Prefix: LLMCORE_SEMANTISCAN__
# Use double underscores (__) to represent dots (.)

# Examples:
export LLMCORE_SEMANTISCAN__DATABASE__PATH="/custom/path/to/db"
export LLMCORE_SEMANTISCAN__RETRIEVAL__TOP_K="15"
export LLMCORE_SEMANTISCAN__EMBEDDINGS__DEFAULT_MODEL="openai_large"
export LLMCORE_SEMANTISCAN__INGESTION__EMBEDDING_WORKERS="8"
```

---

## Configuration Precedence

Settings are loaded in the following order (highest precedence last):

1. **Packaged Defaults**: `llmcore/src/llmcore/config/default_config.toml`
2. **User Config File**: `~/.config/llmcore/config.toml`
3. **Custom Config File**: Specified via `LLMCore.create(config_file_path=...)`
4. **Environment Variables**: Prefixed with `LLMCORE_`
5. **Direct Overrides**: Passed via `LLMCore.create(config_overrides=...)`

---

## Migration from Old Semantiscan Config

If you have an existing `semantiscan/config.toml` file, you can migrate it to llmcore:

1. **Copy the relevant sections** from your `config.toml` to `~/.config/llmcore/config.toml`
2. **Add the `[semantiscan]` prefix** to all section names:
   ```toml
   # Old: semantiscan/config.toml
   [database]
   path = "/my/custom/path"

   # New: ~/.config/llmcore/config.toml
   [semantiscan.database]
   path = "/my/custom/path"
   ```
3. **Update field names** if they changed:
   - `metadata_store.enable` (was `enable`)
   - `metadata_store.table_name` (was just `table_name`)
   - `metadata_store.ingestion_log_table_name` (new)
   - `metadata_store.file_history_table_name` (new)
4. **Remove old config files** from semantiscan (they're no longer used)

---

## Testing Configuration

Run the test suite to verify your configuration:

```bash
# Run all semantiscan config tests
pytest tests/config/test_semantiscan_config.py -v

# Run specific test classes
pytest tests/config/test_semantiscan_config.py::TestSemantiscanConfigInDefaultTOML -v
pytest tests/config/test_semantiscan_config.py::TestSemantiscanConfigValidation -v
```

---

## References

- **Resurrection Plan**: Phase 2, Step 2.1
- **semantiscan Config Models**: `semantiscan/config/models.py`
- **llmcore Config Models**: `llmcore/src/llmcore/config/models.py`
- **confy Documentation**: https://github.com/araray/confy

---

## Troubleshooting

### Issue: "Default model not found in embeddings.models"
**Solution**: Ensure `semantiscan.embeddings.default_model` matches a key in `semantiscan.embeddings.models`

### Issue: "Metadata store path required"
**Solution**: When `metadata_store.enable = true` and `type = "sqlite"`, you must provide `metadata_store.path`

### Issue: "Historical mode requires metadata store"
**Solution**: Set `metadata_store.enable = true` when using `historical`, `historical_delta`, or `incremental` ingestion modes

### Issue: "Config not loading from environment variables"
**Solution**: Ensure you're using the correct prefix (`LLMCORE_SEMANTISCAN__`) and double underscores (`__`) for nesting

---

## Future Enhancements

Planned improvements to the configuration system:

1. **PostgreSQL metadata store support** (currently SQLite only)
2. **Additional vector stores** (Pinecone, Weaviate, Qdrant)
3. **More embedding providers** (Cohere, Voyage AI, etc.)
4. **Advanced retrieval strategies** (MMR, contextual compression)
5. **Config validation UI** in llmchat

---

*Last Updated: 2025-11-16 (Phase 2, Step 2.1)*
