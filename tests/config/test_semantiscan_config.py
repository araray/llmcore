# llmcore/tests/config/test_semantiscan_config.py
"""
Tests for semantiscan configuration in llmcore.

These tests verify that:
1. The [semantiscan] section is properly populated in default_config.toml
2. Config can be loaded via confy
3. All required fields are present
4. Types are correct
5. Validation rules work as expected
"""

import pytest
import importlib.resources
from typing import Dict, Any
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = None

from llmcore.config.models import (
    SemantiscanConfig,
    validate_semantiscan_config,
    semantiscan_config_to_dict
)


class TestSemantiscanConfigInDefaultTOML:
    """Test that [semantiscan] section exists and is properly populated in default_config.toml."""

    @pytest.fixture
    def default_config_dict(self) -> Dict[str, Any]:
        """Load the default_config.toml file."""
        if hasattr(importlib.resources, 'files'):
            default_config_path = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
            with default_config_path.open('rb') as f:
                return tomllib.load(f)
        else:
            default_config_content = importlib.resources.read_text(
                'llmcore.config',
                'default_config.toml',
                encoding='utf-8'
            )
            return tomllib.loads(default_config_content)

    def test_semantiscan_section_exists(self, default_config_dict):
        """Test that [semantiscan] section exists in default_config.toml."""
        assert 'semantiscan' in default_config_dict, \
            "The [semantiscan] section is missing from default_config.toml"

    def test_semantiscan_enabled_field(self, default_config_dict):
        """Test that semantiscan.enabled field exists and is boolean."""
        assert 'enabled' in default_config_dict['semantiscan'], \
            "The semantiscan.enabled field is missing"
        assert isinstance(default_config_dict['semantiscan']['enabled'], bool), \
            "The semantiscan.enabled field must be a boolean"

    def test_database_section_exists(self, default_config_dict):
        """Test that [semantiscan.database] section exists and has required fields."""
        assert 'database' in default_config_dict['semantiscan'], \
            "The [semantiscan.database] section is missing"

        db_config = default_config_dict['semantiscan']['database']
        assert 'type' in db_config, "database.type is missing"
        assert 'path' in db_config, "database.path is missing"
        assert 'collection_name' in db_config, "database.collection_name is missing"

        assert db_config['type'] == 'chromadb', \
            "database.type should be 'chromadb'"

    def test_metadata_store_section_exists(self, default_config_dict):
        """Test that [semantiscan.metadata_store] section exists with correct fields."""
        assert 'metadata_store' in default_config_dict['semantiscan'], \
            "The [semantiscan.metadata_store] section is missing"

        ms_config = default_config_dict['semantiscan']['metadata_store']
        required_fields = [
            'enable', 'type', 'path', 'table_name',
            'ingestion_log_table_name', 'file_history_table_name'
        ]

        for field in required_fields:
            assert field in ms_config, \
                f"metadata_store.{field} is missing"

    def test_embeddings_section_exists(self, default_config_dict):
        """Test that [semantiscan.embeddings] section exists with models."""
        assert 'embeddings' in default_config_dict['semantiscan'], \
            "The [semantiscan.embeddings] section is missing"

        emb_config = default_config_dict['semantiscan']['embeddings']
        assert 'default_model' in emb_config, "embeddings.default_model is missing"
        assert 'models' in emb_config, "embeddings.models is missing"
        assert isinstance(emb_config['models'], dict), \
            "embeddings.models must be a dictionary"

        # Check that default model exists in models dict
        default_model = emb_config['default_model']
        assert default_model in emb_config['models'], \
            f"Default model '{default_model}' not found in embeddings.models"

    def test_chunking_section_exists(self, default_config_dict):
        """Test that [semantiscan.chunking] section exists with strategies."""
        assert 'chunking' in default_config_dict['semantiscan'], \
            "The [semantiscan.chunking] section is missing"

        chunk_config = default_config_dict['semantiscan']['chunking']
        assert 'default_strategy' in chunk_config, "chunking.default_strategy is missing"
        assert 'strategies' in chunk_config, "chunking.strategies is missing"
        assert 'parameters' in chunk_config, "chunking.parameters is missing"

    def test_ingestion_section_exists(self, default_config_dict):
        """Test that [semantiscan.ingestion] section exists with git config."""
        assert 'ingestion' in default_config_dict['semantiscan'], \
            "The [semantiscan.ingestion] section is missing"

        ing_config = default_config_dict['semantiscan']['ingestion']
        assert 'embedding_workers' in ing_config, "ingestion.embedding_workers is missing"
        assert 'git' in ing_config, "ingestion.git is missing"

        git_config = ing_config['git']
        required_git_fields = [
            'enabled', 'default_ref', 'ingestion_mode', 'enable_commit_analysis'
        ]
        for field in required_git_fields:
            assert field in git_config, f"ingestion.git.{field} is missing"

    def test_llm_section_exists(self, default_config_dict):
        """Test that [semantiscan.llm] section exists with providers."""
        assert 'llm' in default_config_dict['semantiscan'], \
            "The [semantiscan.llm] section is missing"

        llm_config = default_config_dict['semantiscan']['llm']
        assert 'default_provider' in llm_config, "llm.default_provider is missing"
        assert 'providers' in llm_config, "llm.providers is missing"

    def test_retrieval_section_exists(self, default_config_dict):
        """Test that [semantiscan.retrieval] section exists."""
        assert 'retrieval' in default_config_dict['semantiscan'], \
            "The [semantiscan.retrieval] section is missing"

        ret_config = default_config_dict['semantiscan']['retrieval']
        required_fields = [
            'top_k', 'enable_hybrid_search', 'bm25_k1', 'bm25_b',
            'enrich_with_external_metadata'
        ]
        for field in required_fields:
            assert field in ret_config, f"retrieval.{field} is missing"

    def test_discovery_section_exists(self, default_config_dict):
        """Test that [semantiscan.discovery] section exists."""
        assert 'discovery' in default_config_dict['semantiscan'], \
            "The [semantiscan.discovery] section is missing"

        disc_config = default_config_dict['semantiscan']['discovery']
        assert 'use_gitignore' in disc_config, "discovery.use_gitignore is missing"
        assert 'excluded_dirs' in disc_config, "discovery.excluded_dirs is missing"
        assert 'excluded_files' in disc_config, "discovery.excluded_files is missing"

    def test_logging_section_exists(self, default_config_dict):
        """Test that [semantiscan.logging] section exists."""
        assert 'logging' in default_config_dict['semantiscan'], \
            "The [semantiscan.logging] section is missing"

        log_config = default_config_dict['semantiscan']['logging']
        required_fields = [
            'log_level_console', 'log_file_enabled', 'log_directory',
            'log_filename_template', 'log_level_file', 'log_format'
        ]
        for field in required_fields:
            assert field in log_config, f"logging.{field} is missing"


class TestSemantiscanConfigValidation:
    """Test Pydantic model validation for semantiscan configuration."""

    @pytest.fixture
    def minimal_valid_config(self) -> Dict[str, Any]:
        """Provide a minimal valid semantiscan configuration."""
        return {
            'enabled': True,
            'database': {
                'type': 'chromadb',
                'path': '~/.llmcore/chroma_db',
                'collection_name': 'test_collection'
            },
            'metadata_store': {
                'enable': False,
                'type': 'sqlite',
                'path': '~/.local/share/semantiscan/metadata.db',
                'connection_string': '',
                'table_name': 'chunk_metadata',
                'ingestion_log_table_name': 'ingestion_log',
                'file_history_table_name': 'file_history'
            },
            'embeddings': {
                'default_model': 'test_model',
                'models': {
                    'test_model': {
                        'provider': 'sentence-transformers',
                        'model_name': 'all-MiniLM-L6-v2',
                        'device': 'cpu',
                        'api_key_env': '',
                        'max_request_tokens': 8000,
                        'base_url': '',
                        'tokenizer_name': '',
                        'uses_doc_query_prefixes': False,
                        'query_prefix': '',
                        'document_prefix': ''
                    }
                }
            },
            'chunking': {
                'default_strategy': 'RecursiveSplitter',
                'strategies': {},
                'parameters': {
                    'RecursiveSplitter': {
                        'chunk_size': 1000,
                        'chunk_overlap': 150
                    },
                    'LineSplitter': {
                        'lines_per_chunk': 50
                    },
                    'SubChunker': {
                        'chunk_size': 500,
                        'chunk_overlap': 50
                    }
                }
            },
            'ingestion': {
                'embedding_workers': 4,
                'batch_size': 100,
                'git': {
                    'enabled': False,
                    'default_ref': 'main',
                    'ingestion_mode': 'snapshot',
                    'historical_start_ref': '',
                    'enable_commit_analysis': False,
                    'enable_commit_llm_analysis': False,
                    'commit_llm_provider_key': '',
                    'commit_llm_prompt_template': '',
                    'commit_message_filter_regex': []
                }
            },
            'llm': {
                'default_provider': 'test_provider',
                'prompt_template_path': '',
                'enable_query_rewriting': False,
                'query_rewrite_provider_key': '',
                'show_sources_in_text': True,
                'tokenizer_name': '',
                'context_buffer': 200,
                'providers': {
                    'test_provider': {
                        'provider': 'ollama',
                        'model_name': 'gemma3:4b',
                        'base_url': 'http://localhost:11434',
                        'api_key_env': '',
                        'tokenizer_name': '',
                        'context_buffer': 250,
                        'parameters': {
                            'temperature': 0.5
                        }
                    }
                }
            },
            'retrieval': {
                'top_k': 10,
                'enable_hybrid_search': False,
                'bm25_k1': 1.5,
                'bm25_b': 0.75,
                'enrich_with_external_metadata': False
            },
            'discovery': {
                'use_gitignore': True,
                'excluded_dirs': ['__pycache__', '.git'],
                'excluded_files': ['.DS_Store']
            },
            'logging': {
                'log_level_console': 'INFO',
                'log_file_enabled': False,
                'log_directory': '~/.local/share/semantiscan/logs',
                'log_filename_template': 'semantiscan_{timestamp:%Y%m%d_%H%M%S}.log',
                'log_level_file': 'DEBUG',
                'log_format': '%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s'
            }
        }

    def test_minimal_config_validates(self, minimal_valid_config):
        """Test that minimal valid configuration validates successfully."""
        config = SemantiscanConfig(**minimal_valid_config)
        assert config.enabled is True
        assert config.database.type == 'chromadb'

    def test_validate_semantiscan_config_function(self, minimal_valid_config):
        """Test the validate_semantiscan_config utility function."""
        config = validate_semantiscan_config(minimal_valid_config)
        assert isinstance(config, SemantiscanConfig)

    def test_config_to_dict_function(self, minimal_valid_config):
        """Test the semantiscan_config_to_dict utility function."""
        config = SemantiscanConfig(**minimal_valid_config)
        config_dict = semantiscan_config_to_dict(config)
        assert isinstance(config_dict, dict)
        assert 'enabled' in config_dict
        assert 'database' in config_dict

    def test_metadata_store_validation_sqlite_requires_path(self):
        """Test that sqlite metadata store requires path when enabled."""
        config_dict = {
            'enable': True,
            'type': 'sqlite',
            'path': None,  # Missing path
            'connection_string': '',
            'table_name': 'chunk_metadata',
            'ingestion_log_table_name': 'ingestion_log',
            'file_history_table_name': 'file_history'
        }

        with pytest.raises(ValueError, match="path.*required"):
            from llmcore.config.models import SemantiscanMetadataStoreConfig
            SemantiscanMetadataStoreConfig(**config_dict)

    def test_embeddings_default_model_must_exist(self):
        """Test that default embedding model must exist in models dict."""
        config_dict = {
            'default_model': 'nonexistent_model',
            'models': {
                'existing_model': {
                    'provider': 'sentence-transformers',
                    'model_name': 'test',
                    'device': 'cpu',
                    'api_key_env': '',
                    'max_request_tokens': 8000,
                    'base_url': '',
                    'tokenizer_name': '',
                    'uses_doc_query_prefixes': False,
                    'query_prefix': '',
                    'document_prefix': ''
                }
            }
        }

        with pytest.raises(ValueError, match="not defined in the 'models' dictionary"):
            from llmcore.config.models import SemantiscanEmbeddingsConfig
            SemantiscanEmbeddingsConfig(**config_dict)

    def test_enrichment_requires_metadata_store(self, minimal_valid_config):
        """Test that enrichment requires metadata store to be enabled."""
        minimal_valid_config['retrieval']['enrich_with_external_metadata'] = True
        minimal_valid_config['metadata_store']['enable'] = False

        with pytest.raises(ValueError, match="metadata_store.*not enabled"):
            SemantiscanConfig(**minimal_valid_config)

    def test_llm_analysis_requires_provider_key(self, minimal_valid_config):
        """Test that LLM commit analysis requires provider key."""
        minimal_valid_config['ingestion']['git']['enable_commit_llm_analysis'] = True
        minimal_valid_config['ingestion']['git']['commit_llm_provider_key'] = ''

        with pytest.raises(ValueError, match="commit_llm_provider_key.*must be set"):
            SemantiscanConfig(**minimal_valid_config)

    def test_historical_modes_require_metadata_store(self, minimal_valid_config):
        """Test that historical ingestion modes require metadata store."""
        minimal_valid_config['ingestion']['git']['ingestion_mode'] = 'historical'
        minimal_valid_config['metadata_store']['enable'] = False

        with pytest.raises(ValueError, match="requires.*metadata_store.*enabled"):
            SemantiscanConfig(**minimal_valid_config)


@pytest.mark.skipif(ConfyConfig is None, reason="confy not installed")
class TestSemantiscanConfigWithConfy:
    """Test semantiscan configuration loading via confy."""

    def test_config_loads_via_confy(self):
        """Test that semantiscan config can be loaded via confy."""
        # Load default config
        if hasattr(importlib.resources, 'files'):
            default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
            with default_config_path_obj.open('rb') as f:
                default_config_dict = tomllib.load(f)
        else:
            default_config_content = importlib.resources.read_text(
                'llmcore.config',
                'default_config.toml',
                encoding='utf-8'
            )
            default_config_dict = tomllib.loads(default_config_content)

        # Create confy config
        config = ConfyConfig(defaults=default_config_dict)

        # Test accessing semantiscan config
        semantiscan_config = config.get('semantiscan')
        assert semantiscan_config is not None
        assert isinstance(semantiscan_config, dict)
        assert 'enabled' in semantiscan_config
        assert 'database' in semantiscan_config

    def test_semantiscan_config_dot_notation(self):
        """Test that semantiscan config can be accessed via dot notation."""
        if hasattr(importlib.resources, 'files'):
            default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
            with default_config_path_obj.open('rb') as f:
                default_config_dict = tomllib.load(f)
        else:
            default_config_content = importlib.resources.read_text(
                'llmcore.config',
                'default_config.toml',
                encoding='utf-8'
            )
            default_config_dict = tomllib.loads(default_config_content)

        config = ConfyConfig(defaults=default_config_dict)

        # Test dot notation access
        assert config.get('semantiscan.enabled') is not None
        assert config.get('semantiscan.database.type') == 'chromadb'
        assert config.get('semantiscan.retrieval.top_k') is not None

    def test_semantiscan_config_validates_with_pydantic(self):
        """Test that semantiscan config from confy validates with Pydantic model."""
        if hasattr(importlib.resources, 'files'):
            default_config_path_obj = importlib.resources.files('llmcore.config').joinpath('default_config.toml')
            with default_config_path_obj.open('rb') as f:
                default_config_dict = tomllib.load(f)
        else:
            default_config_content = importlib.resources.read_text(
                'llmcore.config',
                'default_config.toml',
                encoding='utf-8'
            )
            default_config_dict = tomllib.loads(default_config_content)

        config = ConfyConfig(defaults=default_config_dict)
        semantiscan_config_dict = config.get('semantiscan')

        # Validate with Pydantic model
        validated_config = validate_semantiscan_config(semantiscan_config_dict)
        assert isinstance(validated_config, SemantiscanConfig)
        assert validated_config.database.type == 'chromadb'


class TestSemantiscanConfigTypes:
    """Test that configuration field types are correct."""

    def test_enabled_is_bool(self):
        """Test that enabled field is boolean."""
        config = SemantiscanConfig(enabled=True)
        assert isinstance(config.enabled, bool)

    def test_embedding_workers_is_positive_int(self):
        """Test that embedding_workers must be positive integer."""
        with pytest.raises(ValueError):
            from llmcore.config.models import SemantiscanIngestionConfig
            SemantiscanIngestionConfig(embedding_workers=0, batch_size=100, git={})

    def test_top_k_is_positive_int(self):
        """Test that top_k must be positive integer."""
        with pytest.raises(ValueError):
            from llmcore.config.models import SemantiscanRetrievalConfig
            SemantiscanRetrievalConfig(top_k=0)

    def test_bm25_params_are_floats(self):
        """Test that BM25 parameters are floats with correct ranges."""
        from llmcore.config.models import SemantiscanRetrievalConfig

        # Valid config
        config = SemantiscanRetrievalConfig(bm25_k1=1.5, bm25_b=0.75)
        assert isinstance(config.bm25_k1, float)
        assert isinstance(config.bm25_b, float)

        # Invalid b value (must be 0-1)
        with pytest.raises(ValueError):
            SemantiscanRetrievalConfig(bm25_b=1.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
