# src/llmcore/model_cards/registry.py
# llmcore/model_cards/registry.py
"""
Model Card Registry - Singleton for managing model metadata.

This module provides the ModelCardRegistry class which:
1. Loads model cards from built-in and user directories
2. Provides lookup by provider/model_id or alias
3. Supports filtering by provider, type, and tags
4. Allows saving user-defined model cards

The registry uses a singleton pattern to ensure consistent state
across the application.

Version: 1.0.0
Phase: Foundation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from .schema import (
    ModelCard,
    ModelCardSummary,
    ModelStatus,
    ModelType,
)

logger = logging.getLogger(__name__)


class ModelCardRegistry:
    """
    Singleton registry for model cards.

    Loads cards from:
    1. Built-in defaults (llmcore package's default_cards/)
    2. User overrides (~/.config/llmcore/model_cards/)

    User cards override built-in cards with the same model_id.

    Example:
        >>> registry = ModelCardRegistry.get_instance()
        >>> card = registry.get("openai", "gpt-4o")
        >>> if card:
        ...     print(f"Context: {card.get_context_length()}")
        ...     print(f"Supports vision: {card.capabilities.vision}")
    """

    _instance: ModelCardRegistry | None = None
    _lock: Lock = Lock()

    def __new__(cls) -> ModelCardRegistry:
        """Thread-safe singleton creation."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only once for singleton)."""
        if self._initialized:
            return

        # Provider -> model_id -> ModelCard
        self._cards: dict[str, dict[str, ModelCard]] = {}

        # Alias -> (provider, model_id)
        self._aliases: dict[str, tuple[str, str]] = {}

        # Lowercase model_id -> (provider, original_model_id) for case-insensitive lookup
        self._lowercase_index: dict[str, list[tuple[str, str]]] = {}

        self._loaded = False
        self._builtin_path: Path | None = None
        self._user_path: Path | None = None
        self._config: Any = None  # Optional llmcore config object

        self._initialized = True
        logger.debug("ModelCardRegistry instance created")

    @classmethod
    def get_instance(cls) -> ModelCardRegistry:
        """
        Get the singleton instance of ModelCardRegistry.

        Returns:
            The singleton ModelCardRegistry instance.
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).

        This clears the cached instance, allowing a fresh registry
        to be created on next access.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._cards.clear()
                cls._instance._aliases.clear()
                cls._instance._lowercase_index.clear()
                cls._instance._loaded = False
            cls._instance = None

    def configure_from_config(self, config: Any) -> None:
        """
        Configure the registry from an llmcore config object.

        This method should be called by LLMCore during initialization
        to pass the config object to the registry.

        Args:
            config: The confy Config object from LLMCore

        Example:
            >>> # Called internally by LLMCore
            >>> registry = ModelCardRegistry.get_instance()
            >>> registry.configure_from_config(llmcore_instance.config)
        """
        self._config = config
        logger.debug("ModelCardRegistry configured with llmcore config")

    def _get_configured_user_path(self) -> Path | None:
        """
        Get user_cards_path from stored config.

        Returns:
            Path from config if available and configured, None otherwise.
        """
        if not hasattr(self, "_config") or self._config is None:
            return None

        try:
            # Try config.get() for confy Config
            user_path_str = self._config.get("model_cards.user_cards_path")
            if user_path_str:
                return Path(user_path_str).expanduser()
        except (AttributeError, KeyError, TypeError):
            pass

        try:
            # Try dict-style access
            model_cards_config = self._config.get("model_cards", {})
            if isinstance(model_cards_config, dict):
                user_path_str = model_cards_config.get("user_cards_path")
                if user_path_str:
                    return Path(user_path_str).expanduser()
        except (AttributeError, KeyError, TypeError):
            pass

        return None

    def load(
        self,
        builtin_path: Path | None = None,
        user_path: Path | None = None,
        force_reload: bool = False,
    ) -> None:
        """
        Load model cards from directories.

        Args:
            builtin_path: Path to built-in cards (default: package/default_cards)
            user_path: Path to user cards. Resolution order:
                1. Explicit user_path argument
                2. Config value (model_cards.user_cards_path)
                3. Default: ~/.config/llmcore/model_cards
            force_reload: Reload even if already loaded

        Example:
            >>> registry = ModelCardRegistry.get_instance()
            >>> registry.load()  # Uses defaults or config
            >>> registry.load(force_reload=True)  # Force refresh
            >>> registry.load(user_path=Path("/custom/path"))  # Custom path
        """
        if self._loaded and not force_reload:
            logger.debug("Registry already loaded, skipping (use force_reload=True to reload)")
            return

        # Clear existing data
        self._cards.clear()
        self._aliases.clear()
        self._lowercase_index.clear()

        # Determine builtin path
        if builtin_path is None:
            builtin_path = Path(__file__).parent / "default_cards"
        self._builtin_path = builtin_path

        # Determine user path with resolution order:
        # 1. Explicit argument
        # 2. Config value
        # 3. Default
        if user_path is None:
            # Try to get from config
            user_path = self._get_configured_user_path()
        if user_path is None:
            # Fall back to default
            user_path = self.get_default_user_path()
        self._user_path = user_path

        # Load built-in cards first (lower priority)
        builtin_count = 0
        if builtin_path.exists():
            builtin_count = self._load_directory(builtin_path, source="builtin")
            logger.info(f"Loaded {builtin_count} built-in model cards from {builtin_path}")
        else:
            logger.debug(f"Built-in cards directory not found: {builtin_path}")

        # Load user cards (override built-in)
        user_count = 0
        if user_path.exists():
            user_count = self._load_directory(user_path, source="user")
            logger.info(f"Loaded {user_count} user model cards from {user_path}")
        else:
            logger.debug(f"User cards directory not found: {user_path}")

        self._loaded = True
        total_cards = sum(len(cards) for cards in self._cards.values())
        logger.info(
            f"ModelCardRegistry loaded: {total_cards} cards, "
            f"{len(self._aliases)} aliases across {len(self._cards)} providers"
        )

    def _load_directory(self, path: Path, source: str) -> int:
        """
        Load all cards from a directory tree.

        Args:
            path: Root directory to scan
            source: Source label ("builtin" or "user")

        Returns:
            Number of cards loaded
        """
        count = 0
        for json_file in path.rglob("*.json"):
            # Skip index files and non-card files
            if json_file.name in ("index.json", "manifest.json"):
                continue

            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Handle single card or array of cards
                cards_data = data if isinstance(data, list) else [data]

                for card_data in cards_data:
                    # Inject source and update timestamp
                    card_data["source"] = source
                    if "last_updated" not in card_data:
                        card_data["last_updated"] = datetime.now().isoformat()

                    try:
                        card = ModelCard.model_validate(card_data)
                        self._register_card(card)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to validate model card in {json_file}: {e}")

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {json_file}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load model card from {json_file}: {e}")

        return count

    def _register_card(self, card: ModelCard) -> None:
        """
        Register a card in the registry.

        This updates the main index, aliases, and case-insensitive lookup.

        Args:
            card: ModelCard to register
        """
        # Normalize provider to string
        provider = card.provider if isinstance(card.provider, str) else card.provider.value
        provider_lower = provider.lower()

        # Initialize provider dict if needed
        if provider_lower not in self._cards:
            self._cards[provider_lower] = {}

        # Store card (overwrites if exists - user cards override builtin)
        self._cards[provider_lower][card.model_id] = card

        # Update case-insensitive index
        model_id_lower = card.model_id.lower()
        if model_id_lower not in self._lowercase_index:
            self._lowercase_index[model_id_lower] = []

        # Add to index (avoid duplicates)
        entry = (provider_lower, card.model_id)
        if entry not in self._lowercase_index[model_id_lower]:
            self._lowercase_index[model_id_lower].append(entry)

        # Register aliases
        for alias in card.aliases:
            alias_lower = alias.lower()
            self._aliases[alias_lower] = (provider_lower, card.model_id)

            # Also add alias to case-insensitive index
            if alias_lower not in self._lowercase_index:
                self._lowercase_index[alias_lower] = []
            if entry not in self._lowercase_index[alias_lower]:
                self._lowercase_index[alias_lower].append(entry)

        logger.debug(
            f"Registered model card: {provider_lower}/{card.model_id} "
            f"(source={card.source}, aliases={len(card.aliases)})"
        )

    @staticmethod
    def get_default_user_path() -> Path:
        """
        Get the default user cards path.

        Returns:
            Default path: ~/.config/llmcore/model_cards

        This is the fallback when config is not available or doesn't
        specify a custom path.
        """
        return Path.home() / ".config" / "llmcore" / "model_cards"

    def _ensure_loaded(self) -> None:
        """
        Ensure cards are loaded before any query operation.

        Uses configured user_cards_path if available, otherwise
        uses the default path (~/.config/llmcore/model_cards).
        """
        if not self._loaded:
            self.load()

    def get(
        self,
        provider: str,
        model_id: str,
        *,
        case_sensitive: bool = False,
    ) -> ModelCard | None:
        """
        Get a model card by provider and model ID.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model_id: Model identifier or alias
            case_sensitive: If True, require exact case match

        Returns:
            ModelCard if found, None otherwise

        Example:
            >>> card = registry.get("openai", "gpt-4o")
            >>> card = registry.get("anthropic", "claude-sonnet-4")  # alias
            >>> card = registry.get("openai", "GPT-4O", case_sensitive=False)
        """
        self._ensure_loaded()

        provider_lower = provider.lower()

        # Try direct lookup first
        if provider_lower in self._cards:
            if model_id in self._cards[provider_lower]:
                return self._cards[provider_lower][model_id]

        # Try alias lookup
        alias_key = model_id.lower()
        if alias_key in self._aliases:
            alias_provider, alias_model = self._aliases[alias_key]
            if alias_provider == provider_lower:
                return self._cards[alias_provider].get(alias_model)

        # Try case-insensitive lookup if allowed
        if not case_sensitive:
            model_id_lower = model_id.lower()
            if model_id_lower in self._lowercase_index:
                for idx_provider, idx_model_id in self._lowercase_index[model_id_lower]:
                    if idx_provider == provider_lower:
                        return self._cards[idx_provider][idx_model_id]

        return None

    def get_by_alias(self, alias: str) -> ModelCard | None:
        """
        Get a model card by alias (any provider).

        Args:
            alias: Model alias to look up

        Returns:
            ModelCard if found, None otherwise
        """
        self._ensure_loaded()

        alias_lower = alias.lower()
        if alias_lower in self._aliases:
            provider, model_id = self._aliases[alias_lower]
            return self._cards[provider].get(model_id)
        return None

    def list_cards(
        self,
        provider: str | None = None,
        model_type: ModelType | str | None = None,
        tags: list[str] | None = None,
        status: ModelStatus | str | None = None,
        include_deprecated: bool = True,
    ) -> list[ModelCardSummary]:
        """
        List model cards with optional filtering.

        Args:
            provider: Filter by provider name
            model_type: Filter by model type (chat, embedding, etc.)
            tags: Filter by tags (any match)
            status: Filter by lifecycle status
            include_deprecated: Include deprecated/legacy models

        Returns:
            List of ModelCardSummary objects

        Example:
            >>> # List all chat models
            >>> cards = registry.list_cards(model_type="chat")
            >>>
            >>> # List OpenAI models with vision
            >>> cards = registry.list_cards(provider="openai", tags=["vision"])
        """
        self._ensure_loaded()

        summaries: list[ModelCardSummary] = []

        # Normalize filters
        providers_to_check = [provider.lower()] if provider else list(self._cards.keys())

        target_type: str | None = None
        if model_type:
            target_type = model_type if isinstance(model_type, str) else model_type.value

        target_status: str | None = None
        if status:
            target_status = status if isinstance(status, str) else status.value

        for prov in providers_to_check:
            if prov not in self._cards:
                continue

            for card in self._cards[prov].values():
                # Apply type filter
                card_type = (
                    card.model_type if isinstance(card.model_type, str) else card.model_type.value
                )
                if target_type and card_type != target_type:
                    continue

                # Apply status filter
                card_status = card.lifecycle.status
                if isinstance(card_status, ModelStatus):
                    card_status = card_status.value

                if target_status and card_status != target_status:
                    continue

                # Apply deprecated filter
                if not include_deprecated and card_status in ("deprecated", "legacy", "retired"):
                    continue

                # Apply tags filter (any match)
                if tags and not any(t in card.tags for t in tags):
                    continue

                summaries.append(ModelCardSummary.from_card(card))

        return summaries

    def get_context_length(
        self,
        provider: str,
        model_id: str,
        default: int = 4096,
    ) -> int:
        """
        Get context length for a model.

        Args:
            provider: Provider name
            model_id: Model identifier
            default: Default value if model not found

        Returns:
            Context length in tokens
        """
        card = self.get(provider, model_id)
        if card:
            return card.get_context_length()
        return default

    def get_pricing(
        self,
        provider: str,
        model_id: str,
    ) -> dict[str, Any] | None:
        """
        Get pricing information for a model.

        Args:
            provider: Provider name
            model_id: Model identifier

        Returns:
            Dict with pricing info or None if not available
        """
        card = self.get(provider, model_id)
        if card and card.pricing:
            return {
                "input": card.pricing.per_million_tokens.input,
                "output": card.pricing.per_million_tokens.output,
                "cached_input": card.pricing.per_million_tokens.cached_input,
                "reasoning_output": card.pricing.per_million_tokens.reasoning_output,
                "currency": card.pricing.currency,
                "batch_discount_percent": card.pricing.batch_discount_percent,
            }
        return None

    def get_providers(self) -> list[str]:
        """
        Get list of providers that have cards registered.

        Returns:
            List of provider names
        """
        self._ensure_loaded()
        return list(self._cards.keys())

    def get_models_for_provider(self, provider: str) -> list[str]:
        """
        Get list of model IDs for a provider.

        Args:
            provider: Provider name

        Returns:
            List of model IDs
        """
        self._ensure_loaded()
        provider_lower = provider.lower()
        if provider_lower in self._cards:
            return list(self._cards[provider_lower].keys())
        return []

    def save_card(
        self,
        card: ModelCard,
        *,
        user_override: bool = True,
    ) -> Path:
        """
        Save a model card to disk.

        Args:
            card: ModelCard to save
            user_override: If True, save to user directory; else to package

        Returns:
            Path where the card was saved

        Example:
            >>> card = ModelCard(
            ...     model_id="my-custom-model",
            ...     provider="ollama",
            ...     model_type="chat",
            ...     context=ModelContext(max_input_tokens=8192),
            ... )
            >>> path = registry.save_card(card)
        """
        # Determine base path
        if user_override:
            base_path = self._user_path or (Path.home() / ".config" / "llmcore" / "model_cards")
        else:
            base_path = self._builtin_path or (Path(__file__).parent / "default_cards")

        # Get provider string
        provider = card.provider if isinstance(card.provider, str) else card.provider.value
        provider_lower = provider.lower()

        # Create provider directory
        provider_dir = base_path / provider_lower
        provider_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from model_id
        safe_id = card.model_id.replace("/", "_").replace(":", "_").replace("\\", "_")
        file_path = provider_dir / f"{safe_id}.json"

        # Update source and timestamp
        card_data = card.model_dump(exclude_none=True, mode="json")
        card_data["source"] = "user" if user_override else "builtin"
        card_data["last_updated"] = datetime.now().isoformat()

        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(card_data, f, indent=2, default=str)

        # Update in-memory registry
        card.source = card_data["source"]
        self._register_card(card)

        logger.info(f"Saved model card to {file_path}")
        return file_path

    def remove_card(
        self,
        provider: str,
        model_id: str,
        *,
        delete_file: bool = False,
    ) -> bool:
        """
        Remove a model card from the registry.

        Args:
            provider: Provider name
            model_id: Model identifier
            delete_file: If True, also delete the source file

        Returns:
            True if card was removed
        """
        self._ensure_loaded()

        provider_lower = provider.lower()
        if provider_lower not in self._cards:
            return False

        if model_id not in self._cards[provider_lower]:
            return False

        card = self._cards[provider_lower][model_id]

        # Remove aliases
        for alias in card.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._aliases:
                del self._aliases[alias_lower]

        # Remove from main index
        del self._cards[provider_lower][model_id]

        # Remove from case-insensitive index
        model_id_lower = model_id.lower()
        if model_id_lower in self._lowercase_index:
            self._lowercase_index[model_id_lower] = [
                entry
                for entry in self._lowercase_index[model_id_lower]
                if entry != (provider_lower, model_id)
            ]
            if not self._lowercase_index[model_id_lower]:
                del self._lowercase_index[model_id_lower]

        logger.info(f"Removed model card: {provider_lower}/{model_id}")
        return True

    def stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with statistics about the registry
        """
        self._ensure_loaded()

        total_cards = sum(len(cards) for cards in self._cards.values())
        cards_by_provider = {provider: len(cards) for provider, cards in self._cards.items()}

        return {
            "total_cards": total_cards,
            "providers": len(self._cards),
            "aliases": len(self._aliases),
            "cards_by_provider": cards_by_provider,
            "loaded": self._loaded,
            "builtin_path": str(self._builtin_path) if self._builtin_path else None,
            "user_path": str(self._user_path) if self._user_path else None,
        }


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


def get_model_card_registry() -> ModelCardRegistry:
    """
    Get the model card registry singleton.

    This is the recommended way to access the registry.

    Returns:
        The singleton ModelCardRegistry instance

    Example:
        >>> from llmcore.model_cards import get_model_card_registry
        >>> registry = get_model_card_registry()
        >>> card = registry.get("openai", "gpt-4o")
    """
    return ModelCardRegistry.get_instance()


@lru_cache(maxsize=128)
def get_model_card(provider: str, model_id: str) -> ModelCard | None:
    """
    Cached lookup for a model card.

    This function caches results for performance in hot paths.
    Use registry.get() directly if you need fresh data.

    Args:
        provider: Provider name
        model_id: Model identifier

    Returns:
        ModelCard if found, None otherwise
    """
    registry = get_model_card_registry()
    return registry.get(provider, model_id)


def clear_model_card_cache() -> None:
    """
    Clear the model card lookup cache.

    Call this after modifying cards to ensure fresh data.
    """
    get_model_card.cache_clear()
