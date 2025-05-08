# src/llmcore/providers/manager.py
"""
Provider Manager for LLMCore.

Handles the dynamic loading and management of LLM provider instances based on configuration.
"""

import logging
from typing import Dict, Any, Type, Optional, List

from confy.loader import Config as ConfyConfig

from ..exceptions import ConfigError, ProviderError
from .base import BaseProvider

# Import concrete implementations (add more as they are created)
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider # Example for Phase 2
from .anthropic_provider import AnthropicProvider # Example for Phase 2
# from .gemini_provider import GeminiProvider # Example for Phase 2

logger = logging.getLogger(__name__)

# --- Mapping from config provider name string to class ---
PROVIDER_MAP: Dict[str, Type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    # "gemini": GeminiProvider,
}
# --- End Mapping ---


class ProviderManager:
    """
    Manages the initialization and access to LLM providers.

    Reads configuration, instantiates configured providers, and provides
    access to them.
    """
    _providers: Dict[str, BaseProvider]
    _config: ConfyConfig
    _default_provider_name: str

    def __init__(self, config: ConfyConfig):
        """
        Initializes the ProviderManager and loads configured providers.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).

        Raises:
            ConfigError: If the default provider is not configured or supported.
            ProviderError: If a configured provider fails to initialize.
        """
        self._config = config
        self._providers = {}
        self._default_provider_name = self._config.get('llmcore.default_provider', 'ollama') # Default to ollama if not specified
        logger.info(f"ProviderManager initialized. Default provider set to '{self._default_provider_name}'.")

        self._load_configured_providers()

        # Ensure the default provider was loaded successfully
        if self._default_provider_name not in self._providers:
            raise ConfigError(f"Default provider '{self._default_provider_name}' is configured but failed to load or is not supported. "
                              f"Available types: {list(PROVIDER_MAP.keys())}")

    def _load_configured_providers(self) -> None:
        """Loads and initializes all providers defined in the configuration."""
        providers_config = self._config.get('providers', {})
        if not providers_config:
            logger.warning("No providers configured in the '[providers]' section.")
            # Attempt to load at least the default provider if its config exists implicitly
            if self._default_provider_name in PROVIDER_MAP:
                 provider_config = self._config.get(f'providers.{self._default_provider_name}', {})
                 if provider_config:
                      logger.info(f"Attempting to load default provider '{self._default_provider_name}' based on implicit config.")
                      providers_config[self._default_provider_name] = provider_config
                 else:
                      logger.error("Default provider specified but no configuration found under [providers] or [providers.<default_provider>].")
                      return # Cannot proceed without any provider config
            else:
                 return # No providers configured at all

        loaded_provider_names = []
        for name, config in providers_config.items():
            provider_name_lower = name.lower()
            if provider_name_lower in PROVIDER_MAP:
                provider_cls = PROVIDER_MAP[provider_name_lower]
                try:
                    # Ensure config is a dictionary, even if empty
                    provider_config_dict = config if isinstance(config, dict) else {}
                    self._providers[provider_name_lower] = provider_cls(provider_config_dict)
                    logger.info(f"Provider '{provider_name_lower}' initialized successfully.")
                    loaded_provider_names.append(provider_name_lower)
                except Exception as e:
                    logger.error(f"Failed to initialize provider '{provider_name_lower}': {e}", exc_info=True)
                    # Optionally raise ProviderError here or just log and skip
                    # raise ProviderError(provider_name_lower, f"Initialization failed: {e}")
            else:
                logger.warning(f"Configured provider '{name}' is not supported or mapped. Skipping.")

        logger.debug(f"Successfully loaded providers: {loaded_provider_names}")


    def get_provider(self, name: Optional[str] = None) -> BaseProvider:
        """
        Gets a specific provider instance by name, or the default provider if name is None.

        Args:
            name: The name of the provider to retrieve (e.g., "openai", "ollama").
                  If None, returns the default provider.

        Returns:
            An instance of the requested BaseProvider.

        Raises:
            ConfigError: If the requested provider name is not configured or loaded.
        """
        target_name = name.lower() if name else self._default_provider_name
        provider_instance = self._providers.get(target_name)

        if provider_instance is None:
            # Check if it was configured but failed to load vs not configured at all
            if target_name in self._config.get('providers', {}):
                 raise ProviderError(target_name, "Provider was configured but failed to initialize.")
            else:
                 raise ConfigError(f"Provider '{target_name}' is not configured or supported. "
                                   f"Available loaded providers: {list(self._providers.keys())}")

        return provider_instance

    def get_default_provider(self) -> BaseProvider:
        """
        Gets the instance of the configured default provider.

        Returns:
            An instance of the default BaseProvider.

        Raises:
            ConfigError: If the default provider is not configured or loaded.
        """
        return self.get_provider(self._default_provider_name)

    def get_available_providers(self) -> List[str]:
        """
        Lists the names of all successfully loaded providers.

        Returns:
            A list of provider name strings.
        """
        return list(self._providers.keys())

    async def close_providers(self) -> None:
        """Closes connections or cleans up resources for all loaded providers."""
        logger.info("Closing provider connections...")
        for name, provider in self._providers.items():
            if hasattr(provider, 'close') and callable(provider.close):
                try:
                    await provider.close() # type: ignore # Assuming close is async
                    logger.info(f"Provider '{name}' closed.")
                except Exception as e:
                    logger.error(f"Error closing provider '{name}': {e}", exc_info=True)
        logger.info("Provider connections closure attempt complete.")
