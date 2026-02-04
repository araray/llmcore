# src/llmcore/providers/manager.py
"""
Provider Manager for LLMCore.

Handles the dynamic loading and management of LLM provider instances based on configuration.

UPDATED: Added log_raw_payloads parameter to __init__ for explicit control from LLMCore.
UPDATED: Added async initialize() method for future async provider initialization needs.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Type

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = dict[str, Any]  # type: ignore


from ..exceptions import ConfigError
from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .gemini_provider import GeminiProvider

# Import concrete implementations
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# --- Mapping from config provider name string to class ---
PROVIDER_MAP: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
# --- End Mapping ---


class ProviderManager:
    """
    Manages the initialization and access to LLM providers.

    Reads configuration, instantiates configured providers, and provides
    access to them. It handles flexible API key management and passes the
    global raw payload logging setting to each provider instance.

    UPDATED: Now accepts an optional log_raw_payloads parameter for explicit
    control from LLMCore.create(), while maintaining backward compatibility.
    """

    _providers: dict[str, BaseProvider]
    _config: ConfyConfig
    _default_provider_name: str
    _log_raw_payloads_override: bool | None
    _initialized: bool

    def __init__(self, config: ConfyConfig, log_raw_payloads: bool = False):
        """
        Initializes the ProviderManager and loads configured providers.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
            log_raw_payloads: Optional flag to enable raw payload logging for all
                             providers. This parameter takes precedence over the
                             config value 'llmcore.log_raw_payloads'. Default is False.

        Raises:
            ConfigError: If the default provider is not configured or supported.
            ProviderError: If a configured provider fails to initialize.
        """
        self._config = config
        self._providers = {}
        self._log_raw_payloads_override = log_raw_payloads
        self._initialized = False
        self._default_provider_name = self._config.get("llmcore.default_provider", "ollama").lower()
        logger.info(
            f"ProviderManager initialized. Default provider set to '{self._default_provider_name}'."
        )

        self._load_configured_providers()

        if self._default_provider_name not in self._providers:
            raise ConfigError(
                f"Default provider '{self._default_provider_name}' is not configured or failed to load. "
                f"Loaded provider instances: {list(self._providers.keys())}"
            )

        self._initialized = True

    async def initialize(self) -> None:
        """
        Asynchronous initialization hook for providers that need async setup.

        Currently, providers are initialized synchronously in __init__.
        This method is provided for:
        1. API compatibility with LLMCore's async initialization pattern
        2. Future support for providers that require async initialization
        3. Post-construction async setup tasks

        This method is idempotent - calling it multiple times is safe.
        """
        if self._initialized:
            logger.debug("ProviderManager already initialized, skipping async initialize()")
            return

        # Future: Add any async provider initialization here
        # For example, some providers might need to fetch model lists asynchronously

        logger.debug("ProviderManager async initialize() complete (no-op for current providers)")

    def _load_configured_providers(self) -> None:
        """
        Loads and initializes all providers defined in the [providers] configuration section.

        It determines the provider class based on a 'type' field, handles API key indirection
        via 'api_key_env_var', and passes the global `log_raw_payloads` setting.

        The log_raw_payloads value is determined by:
        1. The explicitly passed parameter to __init__ (if True)
        2. Otherwise, the config value 'llmcore.log_raw_payloads'
        """
        providers_config = self._config.get("providers", {})
        if not isinstance(providers_config, dict):
            logger.warning("'[providers]' section is not a valid dictionary. No providers loaded.")
            return

        # Prefer explicitly passed log_raw_payloads, fall back to config
        log_raw_payloads_global = (
            self._log_raw_payloads_override
            if self._log_raw_payloads_override
            else self._config.get("llmcore.log_raw_payloads", False)
        )
        logger.debug(f"Global 'log_raw_payloads' setting for providers: {log_raw_payloads_global}")

        for section_name, provider_specific_config in providers_config.items():
            current_section_name_lower = section_name.lower()
            if not isinstance(provider_specific_config, dict):
                logger.warning(
                    f"Config for provider '{current_section_name_lower}' is not a dict. Skipping."
                )
                continue

            provider_type_key = provider_specific_config.get(
                "type", current_section_name_lower
            ).lower()
            provider_cls = PROVIDER_MAP.get(provider_type_key)

            if not provider_cls:
                logger.warning(
                    f"Provider type '{provider_type_key}' for section '{current_section_name_lower}' is not supported. Skipping."
                )
                continue

            # Handle flexible API key loading
            if (
                "api_key" not in provider_specific_config
                and "api_key_env_var" in provider_specific_config
            ):
                env_var_name = provider_specific_config["api_key_env_var"]
                api_key = os.environ.get(env_var_name)
                if api_key:
                    provider_specific_config["api_key"] = api_key
                    logger.info(
                        f"Loaded API key for provider '{current_section_name_lower}' from environment variable '{env_var_name}'."
                    )
                else:
                    logger.warning(
                        f"Environment variable '{env_var_name}' specified for provider '{current_section_name_lower}' but it is not set."
                    )

            try:
                self._providers[current_section_name_lower] = provider_cls(
                    provider_specific_config, log_raw_payloads=log_raw_payloads_global
                )
                logger.info(
                    f"Provider instance '{current_section_name_lower}' (type: '{provider_type_key}') initialized."
                )
            except ImportError as e:
                logger.error(
                    f"Failed to init '{current_section_name_lower}': Missing library. Install with 'pip install llmcore[{provider_type_key}]'. Error: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize provider '{current_section_name_lower}': {e}",
                    exc_info=True,
                )

        if not self._providers:
            logger.warning("No provider instances were successfully loaded.")

    def get_provider(self, name: str | None = None) -> BaseProvider:
        """
        Gets a provider instance by its configured name, or the default provider.

        Args:
            name: The configuration section name of the provider instance.
                  If None, returns the default provider instance.

        Returns:
            The initialized BaseProvider instance.

        Raises:
            ConfigError: If the requested provider is not configured or loaded.
        """
        target_name_lower = name.lower() if name else self._default_provider_name
        provider_instance = self._providers.get(target_name_lower)

        if provider_instance is None:
            raise ConfigError(
                f"Provider instance '{target_name_lower}' not configured or failed to load. "
                f"Available instances: {list(self._providers.keys())}"
            )
        return provider_instance

    def get_default_provider(self) -> BaseProvider:
        """Gets the instance of the configured default provider."""
        return self.get_provider(self._default_provider_name)

    def get_available_providers(self) -> list[str]:
        """Lists the names of all successfully loaded provider instances."""
        return list(self._providers.keys())

    def update_log_raw_payloads_setting(self, enable: bool) -> None:
        """
        Updates the `log_raw_payloads_enabled` setting for all managed provider instances.

        Args:
            enable: True to enable raw payload logging, False to disable.
        """
        logger.info(f"ProviderManager updating raw payload logging for all providers to: {enable}")
        self._log_raw_payloads_override = enable
        for provider_name, provider_instance in self._providers.items():
            try:
                provider_instance.log_raw_payloads_enabled = enable
            except Exception as e_update:
                logger.error(
                    f"Error updating raw_payload_logging for provider '{provider_name}': {e_update}"
                )

    async def close_providers(self) -> None:
        """Closes connections or cleans up resources for all loaded providers."""
        logger.info("Closing provider connections...")
        close_tasks = [
            self._close_single_provider(name, provider)
            for name, provider in self._providers.items()
        ]
        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error during provider closure: {result}", exc_info=result)
        logger.info("Provider connections closure attempt complete.")

    async def close_all(self) -> None:
        """
        Alias for close_providers() to maintain API compatibility.

        LLMCore.close() calls ProviderManager.close_all(), so this method
        delegates to close_providers() which does the actual cleanup.
        """
        await self.close_providers()

    async def _close_single_provider(self, name: str, provider: BaseProvider):
        """Helper coroutine to close a single provider and log errors."""
        try:
            await provider.close()
            logger.info(f"Provider instance '{name}' closed.")
        except Exception as e:
            logger.error(f"Error closing provider instance '{name}': {e}", exc_info=True)
