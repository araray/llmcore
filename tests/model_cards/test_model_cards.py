# tests/model_cards/test_model_cards.py
# tests/test_model_cards.py
"""
Comprehensive test suite for the Model Card Library.

Tests cover:
- Schema validation and serialization
- Registry singleton behavior
- Card loading from directories
- Lookup operations (direct, alias, case-insensitive)
- Filtering and listing
- User card override behavior
- Cost calculation
- Edge cases and error handling

Run with: pytest tests/test_model_cards.py -v
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest

# Import the model_cards package from llmcore
from llmcore.model_cards import (
    ModelCard,
    # Registry
    ModelCardRegistry,
    ModelCardSummary,
    ModelContext,
    ModelPricing,
    ModelType,
    # Provider extensions
    OllamaExtension,
    OpenAIExtension,
    Provider,
    TokenPricing,
    clear_model_card_cache,
    get_model_card,
    get_model_card_registry,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_card_data() -> Dict:
    """Sample model card data for testing."""
    return {
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "provider": "openai",
        "model_type": "chat",
        "architecture": {
            "family": "gpt",
            "architecture_type": "transformer",
        },
        "context": {
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "default_output_tokens": 4096,
        },
        "capabilities": {
            "streaming": True,
            "function_calling": True,
            "tool_use": True,
            "json_mode": True,
            "structured_output": True,
            "vision": True,
            "audio_input": True,
            "audio_output": True,
            "reasoning": False,
        },
        "pricing": {
            "currency": "USD",
            "per_million_tokens": {
                "input": 2.50,
                "output": 10.00,
                "cached_input": 1.25,
            },
            "batch_discount_percent": 50,
        },
        "lifecycle": {
            "release_date": "2024-05-13",
            "knowledge_cutoff": "2023-10",
            "status": "active",
        },
        "aliases": ["gpt-4o-2024-05-13"],
        "tags": ["flagship", "multimodal", "vision", "audio"],
        "provider_openai": {
            "owned_by": "openai",
            "supports_predicted_outputs": True,
            "fine_tuning_available": True,
        },
    }


@pytest.fixture
def sample_embedding_card_data() -> Dict:
    """Sample embedding model card data."""
    return {
        "model_id": "text-embedding-3-large",
        "display_name": "OpenAI Embedding 3 Large",
        "provider": "openai",
        "model_type": "embedding",
        "context": {
            "max_input_tokens": 8191,
        },
        "capabilities": {
            "streaming": False,
        },
        "pricing": {
            "currency": "USD",
            "per_million_tokens": {
                "input": 0.13,
                "output": 0,
            },
        },
        "lifecycle": {
            "status": "active",
        },
        "tags": ["embedding", "retrieval"],
        "embedding_config": {
            "dimensions_default": 3072,
            "dimensions_configurable": [256, 1024, 3072],
            "supports_matryoshka": True,
            "similarity_metrics": ["cosine", "dot_product", "euclidean"],
            "normalization": "L2",
        },
    }


@pytest.fixture
def temp_cards_dir(sample_card_data, sample_embedding_card_data) -> Path:
    """Create a temporary directory with test card files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create provider directories
        openai_dir = base / "openai"
        anthropic_dir = base / "anthropic"
        openai_dir.mkdir()
        anthropic_dir.mkdir()

        # Write OpenAI cards
        with open(openai_dir / "gpt-4o.json", "w") as f:
            json.dump(sample_card_data, f)

        with open(openai_dir / "text-embedding-3-large.json", "w") as f:
            json.dump(sample_embedding_card_data, f)

        # Write Anthropic card
        claude_card = {
            "model_id": "claude-sonnet-4-20250514",
            "display_name": "Claude Sonnet 4",
            "provider": "anthropic",
            "model_type": "chat",
            "context": {
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
            },
            "capabilities": {
                "streaming": True,
                "tool_use": True,
                "vision": True,
                "reasoning": True,
            },
            "pricing": {
                "currency": "USD",
                "per_million_tokens": {
                    "input": 3.00,
                    "output": 15.00,
                    "cached_input": 0.30,
                },
            },
            "lifecycle": {
                "status": "active",
            },
            "aliases": ["claude-sonnet-4", "sonnet-4"],
            "tags": ["flagship", "multimodal", "reasoning"],
        }
        with open(anthropic_dir / "claude-sonnet-4.json", "w") as f:
            json.dump(claude_card, f)

        yield base


@pytest.fixture
def fresh_registry(temp_cards_dir):
    """Provide a fresh registry loaded with test data."""
    # Reset the singleton
    ModelCardRegistry.reset_instance()
    clear_model_card_cache()

    registry = ModelCardRegistry.get_instance()
    registry.load(
        builtin_path=temp_cards_dir,
        user_path=Path("/nonexistent"),  # No user cards
        force_reload=True,
    )

    yield registry

    # Cleanup
    ModelCardRegistry.reset_instance()
    clear_model_card_cache()


# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestModelCardSchema:
    """Tests for ModelCard Pydantic model."""

    def test_create_minimal_card(self):
        """Should create card with minimal required fields."""
        card = ModelCard(
            model_id="test-model",
            provider="openai",
            model_type="chat",
            context=ModelContext(max_input_tokens=4096),
        )

        assert card.model_id == "test-model"
        assert card.provider == "openai"
        assert card.model_type == "chat"
        assert card.context.max_input_tokens == 4096
        assert card.source == "builtin"  # default

    def test_create_full_card(self, sample_card_data):
        """Should create card with all fields populated."""
        card = ModelCard.model_validate(sample_card_data)

        assert card.model_id == "gpt-4o"
        assert card.display_name == "GPT-4o"
        assert card.context.max_input_tokens == 128000
        assert card.capabilities.vision is True
        assert card.pricing.per_million_tokens.input == 2.50
        assert "gpt-4o-2024-05-13" in card.aliases

    def test_provider_enum_serialization(self):
        """Should handle Provider enum correctly."""
        card = ModelCard(
            model_id="test",
            provider=Provider.ANTHROPIC,
            model_type=ModelType.CHAT,
            context=ModelContext(max_input_tokens=100000),
        )

        # Serialize to dict
        data = card.model_dump()
        assert data["provider"] == "anthropic"
        assert data["model_type"] == "chat"

    def test_json_round_trip(self, sample_card_data):
        """Should serialize to JSON and back without data loss."""
        original = ModelCard.model_validate(sample_card_data)

        # To JSON and back
        json_str = original.model_dump_json()
        restored = ModelCard.model_validate_json(json_str)

        assert restored.model_id == original.model_id
        assert restored.context.max_input_tokens == original.context.max_input_tokens
        assert (
            restored.pricing.per_million_tokens.input == original.pricing.per_million_tokens.input
        )

    def test_helper_methods(self, sample_card_data):
        """Should have working helper methods."""
        card = ModelCard.model_validate(sample_card_data)

        # get_context_length
        assert card.get_context_length() == 128000

        # get_max_output
        assert card.get_max_output() == 16384

        # supports_capability
        assert card.supports_capability("vision") is True
        assert card.supports_capability("reasoning") is False

        # is_local
        assert card.is_local() is False

        # is_deprecated
        assert card.is_deprecated() is False


class TestModelPricing:
    """Tests for pricing calculations."""

    def test_basic_cost_calculation(self):
        """Should calculate cost correctly."""
        pricing = ModelPricing(
            per_million_tokens=TokenPricing(
                input=2.50,
                output=10.00,
            )
        )

        # 1000 input + 500 output tokens
        cost = pricing.get_cost(input_tokens=1000, output_tokens=500)

        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001

    def test_cached_token_cost(self):
        """Should handle cached tokens correctly."""
        pricing = ModelPricing(
            per_million_tokens=TokenPricing(
                input=2.50,
                output=10.00,
                cached_input=1.25,
            )
        )

        # 1000 input, 500 cached
        cost = pricing.get_cost(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
        )

        # 500 non-cached @ 2.50, 500 cached @ 1.25, 500 output @ 10.00
        expected = (500 / 1_000_000) * 2.50 + (500 / 1_000_000) * 1.25 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_from_card(self, sample_card_data):
        """Should estimate cost via ModelCard."""
        card = ModelCard.model_validate(sample_card_data)

        cost = card.estimate_cost(
            input_tokens=10000,
            output_tokens=1000,
        )

        assert cost is not None
        expected = (10000 / 1_000_000) * 2.50 + (1000 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001


class TestModelCardSummary:
    """Tests for ModelCardSummary creation."""

    def test_from_card(self, sample_card_data):
        """Should create summary from full card."""
        card = ModelCard.model_validate(sample_card_data)
        summary = ModelCardSummary.from_card(card)

        assert summary.model_id == "gpt-4o"
        assert summary.provider == "openai"
        assert summary.model_type == "chat"
        assert summary.context_length == 128000
        assert summary.status == "active"
        assert summary.has_pricing is True


class TestProviderExtensions:
    """Tests for provider-specific extensions."""

    def test_ollama_extension(self):
        """Should handle Ollama-specific fields."""
        card = ModelCard(
            model_id="llama3.2:70b",
            provider="ollama",
            model_type="chat",
            context=ModelContext(max_input_tokens=131072),
            provider_ollama=OllamaExtension(
                format="gguf",
                quantization_level="Q4_K_M",
                file_size_bytes=40_000_000_000,
            ),
        )

        ext = card.get_provider_extension()
        assert isinstance(ext, OllamaExtension)
        assert ext.quantization_level == "Q4_K_M"

    def test_openai_extension(self, sample_card_data):
        """Should handle OpenAI-specific fields."""
        card = ModelCard.model_validate(sample_card_data)

        ext = card.get_provider_extension()
        assert isinstance(ext, OpenAIExtension)
        assert ext.owned_by == "openai"
        assert ext.supports_predicted_outputs is True


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestRegistrySingleton:
    """Tests for ModelCardRegistry singleton behavior."""

    def test_singleton_same_instance(self):
        """Should return same instance on multiple calls."""
        ModelCardRegistry.reset_instance()

        r1 = ModelCardRegistry.get_instance()
        r2 = ModelCardRegistry.get_instance()
        r3 = ModelCardRegistry()

        assert r1 is r2
        assert r2 is r3

        ModelCardRegistry.reset_instance()

    def test_reset_instance(self):
        """Should create new instance after reset."""
        r1 = ModelCardRegistry.get_instance()
        r1._test_marker = "test"

        ModelCardRegistry.reset_instance()

        r2 = ModelCardRegistry.get_instance()
        assert not hasattr(r2, "_test_marker")

        ModelCardRegistry.reset_instance()


class TestRegistryLoading:
    """Tests for registry card loading."""

    def test_load_from_directory(self, fresh_registry):
        """Should load cards from directory."""
        stats = fresh_registry.stats()

        assert stats["total_cards"] == 3
        assert stats["providers"] == 2
        assert "openai" in stats["cards_by_provider"]
        assert "anthropic" in stats["cards_by_provider"]

    def test_lazy_loading(self, temp_cards_dir):
        """Should load on first access if not loaded."""
        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry.get_instance()

        # Not loaded yet
        assert registry._loaded is False

        # Load explicitly with test paths
        registry.load(
            builtin_path=temp_cards_dir,
            user_path=Path("/nonexistent"),
        )

        assert registry._loaded is True

        # Now should find the card
        card = registry.get("openai", "gpt-4o")
        assert card is not None

        ModelCardRegistry.reset_instance()


class TestRegistryLookup:
    """Tests for registry lookup operations."""

    def test_direct_lookup(self, fresh_registry):
        """Should find card by provider and model_id."""
        card = fresh_registry.get("openai", "gpt-4o")

        assert card is not None
        assert card.model_id == "gpt-4o"
        assert card.display_name == "GPT-4o"

    def test_lookup_nonexistent(self, fresh_registry):
        """Should return None for unknown model."""
        card = fresh_registry.get("openai", "nonexistent-model")
        assert card is None

    def test_alias_lookup(self, fresh_registry):
        """Should find card by alias."""
        card = fresh_registry.get("anthropic", "claude-sonnet-4")

        assert card is not None
        assert card.model_id == "claude-sonnet-4-20250514"

    def test_case_insensitive_lookup(self, fresh_registry):
        """Should match case-insensitively by default."""
        card = fresh_registry.get("openai", "GPT-4O")

        assert card is not None
        assert card.model_id == "gpt-4o"

    def test_case_sensitive_lookup(self, fresh_registry):
        """Should fail with wrong case when case_sensitive=True."""
        card = fresh_registry.get("openai", "GPT-4O", case_sensitive=True)
        assert card is None

        card = fresh_registry.get("openai", "gpt-4o", case_sensitive=True)
        assert card is not None

    def test_get_by_alias(self, fresh_registry):
        """Should find card by alias without provider."""
        card = fresh_registry.get_by_alias("sonnet-4")

        assert card is not None
        assert card.model_id == "claude-sonnet-4-20250514"


class TestRegistryFiltering:
    """Tests for registry filtering and listing."""

    def test_list_all_cards(self, fresh_registry):
        """Should list all cards."""
        cards = fresh_registry.list_cards()
        assert len(cards) == 3

    def test_filter_by_provider(self, fresh_registry):
        """Should filter by provider."""
        cards = fresh_registry.list_cards(provider="openai")

        assert len(cards) == 2
        assert all(c.provider == "openai" for c in cards)

    def test_filter_by_type(self, fresh_registry):
        """Should filter by model type."""
        cards = fresh_registry.list_cards(model_type="embedding")

        assert len(cards) == 1
        assert cards[0].model_type == "embedding"

    def test_filter_by_tags(self, fresh_registry):
        """Should filter by tags (any match)."""
        cards = fresh_registry.list_cards(tags=["reasoning"])

        assert len(cards) == 1
        assert "reasoning" in cards[0].tags

    def test_combined_filters(self, fresh_registry):
        """Should combine multiple filters."""
        cards = fresh_registry.list_cards(
            provider="openai",
            model_type="chat",
        )

        assert len(cards) == 1
        assert cards[0].model_id == "gpt-4o"

    def test_get_providers(self, fresh_registry):
        """Should list providers."""
        providers = fresh_registry.get_providers()

        assert "openai" in providers
        assert "anthropic" in providers

    def test_get_models_for_provider(self, fresh_registry):
        """Should list models for a provider."""
        models = fresh_registry.get_models_for_provider("openai")

        assert "gpt-4o" in models
        assert "text-embedding-3-large" in models


class TestRegistryPricing:
    """Tests for registry pricing queries."""

    def test_get_pricing(self, fresh_registry):
        """Should return pricing data."""
        pricing = fresh_registry.get_pricing("openai", "gpt-4o")

        assert pricing is not None
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00
        assert pricing["currency"] == "USD"

    def test_get_pricing_nonexistent(self, fresh_registry):
        """Should return None for unknown model."""
        pricing = fresh_registry.get_pricing("openai", "nonexistent")
        assert pricing is None

    def test_get_context_length(self, fresh_registry):
        """Should return context length."""
        length = fresh_registry.get_context_length("openai", "gpt-4o")
        assert length == 128000

    def test_get_context_length_default(self, fresh_registry):
        """Should return default for unknown model."""
        length = fresh_registry.get_context_length("unknown", "model", default=8192)
        assert length == 8192


class TestRegistrySaveCard:
    """Tests for saving user cards."""

    def test_save_card(self, fresh_registry, tmp_path):
        """Should save card to user directory."""
        fresh_registry._user_path = tmp_path

        card = ModelCard(
            model_id="my-custom-model",
            provider="ollama",
            model_type="chat",
            context=ModelContext(max_input_tokens=8192),
        )

        path = fresh_registry.save_card(card)

        assert path.exists()
        assert path.suffix == ".json"
        assert "my-custom-model" in path.name

        # Should be in registry now
        loaded = fresh_registry.get("ollama", "my-custom-model")
        assert loaded is not None
        assert loaded.source == "user"


class TestUserOverride:
    """Tests for user card override behavior."""

    def test_user_overrides_builtin(self, temp_cards_dir, tmp_path):
        """User cards should override built-in with same ID."""
        ModelCardRegistry.reset_instance()

        # Create user card with different context length
        user_dir = tmp_path / "openai"
        user_dir.mkdir(parents=True)

        user_card = {
            "model_id": "gpt-4o",
            "provider": "openai",
            "model_type": "chat",
            "context": {"max_input_tokens": 256000},  # Different!
        }
        with open(user_dir / "gpt-4o.json", "w") as f:
            json.dump(user_card, f)

        # Load with both paths
        registry = ModelCardRegistry.get_instance()
        registry.load(
            builtin_path=temp_cards_dir,
            user_path=tmp_path,
            force_reload=True,
        )

        # Should get user version
        card = registry.get("openai", "gpt-4o")
        assert card is not None
        assert card.context.max_input_tokens == 256000
        assert card.source == "user"

        ModelCardRegistry.reset_instance()


# =============================================================================
# MODULE FUNCTION TESTS
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_model_card_registry(self, fresh_registry):
        """Should return the singleton."""
        registry = get_model_card_registry()
        assert registry is fresh_registry

    def test_get_model_card_cached(self, fresh_registry):
        """Should use cache for repeated lookups."""
        clear_model_card_cache()

        card1 = get_model_card("openai", "gpt-4o")
        card2 = get_model_card("openai", "gpt-4o")

        assert card1 is card2  # Same object from cache

    def test_clear_cache(self, fresh_registry):
        """Should clear the lookup cache."""
        card1 = get_model_card("openai", "gpt-4o")
        clear_model_card_cache()

        # Info about cache
        info = get_model_card.cache_info()
        assert info.currsize == 0


class TestRegistryConfig:
    """Tests for registry configuration integration."""

    def test_default_user_path(self):
        """Should return default user path."""
        path = ModelCardRegistry.get_default_user_path()
        assert path.name == "model_cards"
        assert "llmcore" in str(path)
        assert ".config" in str(path)

    def test_configure_from_config_dict(self, tmp_path):
        """Should accept dict-style config."""
        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry.get_instance()

        custom_path = tmp_path / "custom_cards"
        custom_path.mkdir()

        mock_config = {"model_cards": {"user_cards_path": str(custom_path)}}
        registry.configure_from_config(mock_config)

        # Should use configured path
        configured = registry._get_configured_user_path()
        assert configured == custom_path

    def test_configure_from_config_with_tilde(self, tmp_path):
        """Should expand ~ in configured path."""
        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry.get_instance()

        mock_config = {"model_cards": {"user_cards_path": "~/custom_model_cards"}}
        registry.configure_from_config(mock_config)

        configured = registry._get_configured_user_path()
        assert configured is not None
        assert "~" not in str(configured)
        assert "custom_model_cards" in str(configured)

    def test_load_uses_configured_path(self, tmp_path):
        """load() should use configured user_cards_path."""
        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry.get_instance()

        custom_path = tmp_path / "configured_cards"
        custom_path.mkdir()

        mock_config = {"model_cards": {"user_cards_path": str(custom_path)}}
        registry.configure_from_config(mock_config)

        # Create a test card in custom path
        provider_dir = custom_path / "test_provider"
        provider_dir.mkdir()
        card_data = {
            "model_id": "config-test-model",
            "provider": "test_provider",
            "model_type": "chat",
            "context": {"max_input_tokens": 4096},
        }
        with open(provider_dir / "config-test-model.json", "w") as f:
            json.dump(card_data, f)

        registry.load(force_reload=True)

        # Should have loaded from configured path
        assert registry._user_path == custom_path
        card = registry.get("test_provider", "config-test-model")
        assert card is not None
        assert card.model_id == "config-test-model"

    def test_explicit_path_overrides_config(self, tmp_path):
        """Explicit user_path argument should override config."""
        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry.get_instance()

        config_path = tmp_path / "config_path"
        explicit_path = tmp_path / "explicit_path"
        config_path.mkdir()
        explicit_path.mkdir()

        mock_config = {"model_cards": {"user_cards_path": str(config_path)}}
        registry.configure_from_config(mock_config)

        # Load with explicit path
        registry.load(user_path=explicit_path, force_reload=True)

        # Should use explicit path, not config
        assert registry._user_path == explicit_path

    def test_no_config_uses_default(self, tmp_path):
        """Without config, should use default path."""
        ModelCardRegistry.reset_instance()
        registry = ModelCardRegistry.get_instance()

        # Don't configure - should use default
        default = ModelCardRegistry.get_default_user_path()

        # _get_configured_user_path should return None
        configured = registry._get_configured_user_path()
        assert configured is None


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
