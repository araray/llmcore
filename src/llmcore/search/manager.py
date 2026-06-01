# src/llmcore/search/manager.py
"""Manager for LLMCore web/data **search** providers.

The search-side analogue of :class:`llmcore.providers.manager.ProviderManager`.
It reads the ``[search_providers]`` configuration section, instantiates each
configured provider, and exposes uniform accessors.

Key difference from :class:`ProviderManager`
---------------------------------------------
Search is an **optional** capability.  Unlike the LLM ``ProviderManager`` —
which *requires* a working default LLM provider and raises if none is
configured — :class:`SearchProviderManager` is happy to load **zero** providers.
A consuming application that never configures ``[search_providers]`` keeps
working exactly as before; calling a search method then raises a clear,
actionable error rather than failing at construction time.

Configuration shape
--------------------
.. code-block:: toml

    [llmcore]
    default_search_provider = "brightdata"   # optional

    [search_providers.brightdata]
    # type = "brightdata"                    # inferred from the section name
    api_key_env_var = "BRIGHTDATA_API_TOKEN"
    serp_zone = "my_serp_zone"
    unlocker_zone = "my_unlocker_zone"

Multiple instances of the same provider type are supported by using distinct
section names plus an explicit ``type`` key (mirroring ``[providers.*]``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

try:
    from confy.loader import Config as ConfyConfig
except ImportError:  # pragma: no cover
    ConfyConfig = dict[str, Any]  # type: ignore

from ..exceptions import ConfigError
from .base import BaseSearchProvider
from .providers.brightdata_provider import BrightDataSearchProvider
from .providers.semanticscholar_provider import SemanticScholarSearchProvider
from .providers.serpapi_provider import SerpApiSearchProvider
from .providers.serper_provider import SerperSearchProvider

logger = logging.getLogger(__name__)

# --- Mapping from config provider type string to class ---
SEARCH_PROVIDER_MAP: dict[str, type[BaseSearchProvider]] = {
    "brightdata": BrightDataSearchProvider,
    # Alias: hyphen/underscore spellings resolve to the same class.
    "bright_data": BrightDataSearchProvider,
    "bright-data": BrightDataSearchProvider,
    "serper": SerperSearchProvider,
    "serper_dev": SerperSearchProvider,
    "serperdev": SerperSearchProvider,
    "serpapi": SerpApiSearchProvider,
    "serp_api": SerpApiSearchProvider,
    "serpapi_search": SerpApiSearchProvider,
    "semanticscholar": SemanticScholarSearchProvider,
    "semantic_scholar": SemanticScholarSearchProvider,
    "semantic-scholar": SemanticScholarSearchProvider,
    "s2": SemanticScholarSearchProvider,
}

# Conventional environment variables for known search-provider types.
# Mirrors ``_OPENAI_COMPATIBLE_DEFAULTS`` in the LLM ProviderManager.
_SEARCH_PROVIDER_ENV_DEFAULTS: dict[str, str] = {
    "brightdata": "BRIGHTDATA_API_TOKEN",
    "bright_data": "BRIGHTDATA_API_TOKEN",
    "bright-data": "BRIGHTDATA_API_TOKEN",
    "serper": "SERPER_API_KEY",
    "serper_dev": "SERPER_API_KEY",
    "serperdev": "SERPER_API_KEY",
    "serpapi": "SERPAPI_API_KEY",
    "serp_api": "SERPAPI_API_KEY",
    "serpapi_search": "SERPAPI_API_KEY",
    # Semantic Scholar's key is OPTIONAL; if SEMANTIC_SCHOLAR_API_KEY is set it is
    # used, otherwise the provider operates against the public (rate-limited) pool.
    "semanticscholar": "SEMANTIC_SCHOLAR_API_KEY",
    "semantic_scholar": "SEMANTIC_SCHOLAR_API_KEY",
    "semantic-scholar": "SEMANTIC_SCHOLAR_API_KEY",
    "s2": "SEMANTIC_SCHOLAR_API_KEY",
}


class SearchProviderManager:
    """Loads and provides access to configured search providers.

    Args:
        config: The unified LLMCore configuration object.
        log_raw_payloads: When ``True``, enables raw payload logging for all
            loaded search providers; otherwise the ``llmcore.log_raw_payloads``
            config value is used.

    Attributes:
        (private) ``_providers``: Mapping of instance name → provider instance.
    """

    _providers: dict[str, BaseSearchProvider]
    _config: Any
    _default_provider_name: str | None
    _log_raw_payloads_override: bool | None
    _initialized: bool

    def __init__(self, config: Any, log_raw_payloads: bool = False) -> None:
        self._config = config
        self._providers = {}
        self._log_raw_payloads_override = log_raw_payloads
        self._initialized = False

        default = self._config.get("llmcore.default_search_provider", None)
        self._default_provider_name = (
            default.lower() if isinstance(default, str) and default else None
        )

        self._load_configured_providers()

        # Unlike the LLM ProviderManager, a missing/unknown default is NOT fatal
        # for search (which is optional).  We only warn so misconfiguration is
        # visible without breaking applications that do not use search.
        if self._default_provider_name and self._default_provider_name not in self._providers:
            logger.warning(
                "Default search provider '%s' is not configured or failed to load. "
                "Loaded search provider instances: %s",
                self._default_provider_name,
                list(self._providers.keys()),
            )
            self._default_provider_name = None

        # If exactly one provider is loaded and no default was set, adopt it.
        if self._default_provider_name is None and len(self._providers) == 1:
            self._default_provider_name = next(iter(self._providers))

        self._initialized = True

    async def initialize(self) -> None:
        """Async initialization hook (idempotent; no-op for current providers)."""
        if self._initialized:
            logger.debug("SearchProviderManager already initialized; skipping initialize().")
            return
        logger.debug("SearchProviderManager async initialize() complete (no-op).")

    def _load_configured_providers(self) -> None:
        """Instantiate every provider defined under ``[search_providers]``."""
        providers_config = self._config.get("search_providers", {})
        if not isinstance(providers_config, dict):
            logger.debug(
                "'[search_providers]' section is absent or not a dict. No search providers loaded."
            )
            return

        log_raw_payloads_global = (
            self._log_raw_payloads_override
            if self._log_raw_payloads_override
            else self._config.get("llmcore.log_raw_payloads", False)
        )

        for section_name, provider_specific_config in providers_config.items():
            section_lower = section_name.lower()
            if not isinstance(provider_specific_config, dict):
                logger.warning(
                    "Config for search provider '%s' is not a dict. Skipping.", section_lower
                )
                continue

            # Work on a shallow copy so we never mutate the shared config object.
            cfg = dict(provider_specific_config)
            provider_type_key = str(cfg.get("type", section_lower)).lower()
            provider_cls = SEARCH_PROVIDER_MAP.get(provider_type_key)
            if not provider_cls:
                logger.warning(
                    "Search provider type '%s' for section '%s' is not supported. Skipping. "
                    "Supported types: %s",
                    provider_type_key,
                    section_lower,
                    sorted(SEARCH_PROVIDER_MAP),
                )
                continue

            # --- Flexible API key resolution (mirrors ProviderManager) ---
            if "api_key" not in cfg and "api_key_env_var" in cfg:
                env_var_name = cfg["api_key_env_var"]
                api_key = os.environ.get(env_var_name)
                if api_key:
                    cfg["api_key"] = api_key
                    logger.info(
                        "Loaded API key for search provider '%s' from env var '%s'.",
                        section_lower,
                        env_var_name,
                    )
                else:
                    logger.warning(
                        "Env var '%s' specified for search provider '%s' but it is not set.",
                        env_var_name,
                        section_lower,
                    )

            if "api_key" not in cfg and "api_key_env_var" not in cfg:
                conventional_env = _SEARCH_PROVIDER_ENV_DEFAULTS.get(
                    provider_type_key, f"{section_lower.upper()}_API_KEY"
                )
                api_key = os.environ.get(conventional_env)
                if api_key:
                    cfg["api_key"] = api_key
                    logger.debug(
                        "Loaded API key for '%s' from conventional env var '%s'.",
                        section_lower,
                        conventional_env,
                    )

            try:
                cfg["_instance_name"] = section_lower
                self._providers[section_lower] = provider_cls(
                    cfg, log_raw_payloads=log_raw_payloads_global
                )
                logger.debug(
                    "Search provider instance '%s' (type: '%s') initialized.",
                    section_lower,
                    provider_type_key,
                )
            except ImportError as e:
                logger.warning(
                    "Failed to init search provider '%s': missing library. "
                    "Install with 'pip install llmcore[%s]'. Error: %s",
                    section_lower,
                    provider_type_key,
                    e,
                )
            except ConfigError as e:
                # Expected for missing API keys — log without a traceback.
                logger.warning("Search provider '%s' not configured: %s", section_lower, e)
            except Exception as e:
                logger.error(
                    "Failed to initialize search provider '%s': %s",
                    section_lower,
                    e,
                    exc_info=True,
                )

        if not self._providers:
            logger.debug("No search provider instances were loaded (search is optional).")

    def get_search_provider(self, name: str | None = None) -> BaseSearchProvider:
        """Return a provider instance by name, or the default.

        Args:
            name: The configuration section name.  If ``None``, returns the
                default search provider.

        Returns:
            The requested :class:`BaseSearchProvider`.

        Raises:
            ConfigError: If the requested/default provider is not available.
        """
        target = name or self._default_provider_name
        if target is None:
            raise ConfigError(
                "No search provider requested and no default configured. "
                "Set llmcore.default_search_provider or configure a "
                "[search_providers.<name>] section."
            )
        target = target.lower()
        provider = self._providers.get(target)
        if provider is None:
            raise ConfigError(
                f"Search provider instance '{target}' not configured or failed to load. "
                f"Available instances: {list(self._providers.keys())}"
            )
        return provider

    def get_default_search_provider(self) -> BaseSearchProvider:
        """Return the configured default search provider.

        Raises:
            ConfigError: If no default is configured/available.
        """
        return self.get_search_provider(self._default_provider_name)

    def get_available_search_providers(self) -> list[str]:
        """Return the names of all successfully loaded provider instances."""
        return list(self._providers.keys())

    def has_search_providers(self) -> bool:
        """Return ``True`` if at least one search provider is configured."""
        return bool(self._providers)

    @property
    def default_search_provider_name(self) -> str | None:
        """The resolved default search-provider instance name (or ``None``)."""
        return self._default_provider_name

    def update_log_raw_payloads_setting(self, enable: bool) -> None:
        """Update raw-payload logging on all loaded providers.

        Args:
            enable: New value for ``log_raw_payloads_enabled``.
        """
        self._log_raw_payloads_override = enable
        for name, provider in self._providers.items():
            try:
                provider.log_raw_payloads_enabled = enable
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Error updating raw_payload_logging for '%s': %s", name, e)

    async def close_all(self) -> None:
        """Close all loaded search providers, ignoring individual failures."""
        tasks = [self._close_single(name, p) for name, p in self._providers.items()]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Error during search provider closure: %s", result)

    async def _close_single(self, name: str, provider: BaseSearchProvider) -> None:
        """Close a single provider, logging any error."""
        try:
            await provider.close()
            logger.debug("Search provider instance '%s' closed.", name)
        except Exception as e:
            logger.error("Error closing search provider instance '%s': %s", name, e, exc_info=True)
