# src/llmcore/providers/manager.py
"""
Provider Manager for LLMCore.

Handles the dynamic loading and management of LLM provider instances based on configuration.
"""

import logging
import asyncio
from typing import Dict, Any, Type, Optional, List

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore


from ..exceptions import ConfigError, ProviderError
from .base import BaseProvider

# Import concrete implementations
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider # Import the new Gemini provider

logger = logging.getLogger(__name__)

# --- Mapping from config provider name string to class ---
# Add the new Gemini provider to the map
PROVIDER_MAP: Dict[str, Type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider, # Added Gemini
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
        # Read default provider, fallback to 'ollama' if not set
        self._default_provider_name = self._config.get('llmcore.default_provider', 'ollama')
        logger.info(f"ProviderManager initialized. Default provider set to '{self._default_provider_name}'.")

        self._load_configured_providers()

        # Ensure the default provider was loaded successfully
        if self._default_provider_name not in self._providers:
            # Check if it was configured but failed vs. never configured/supported
            providers_config = self._config.get('providers', {})
            if self._default_provider_name in providers_config:
                 # It was configured but failed to load
                 raise ProviderError(self._default_provider_name, "Default provider was configured but failed to initialize.")
            else:
                 # It was not configured or is not supported
                 raise ConfigError(f"Default provider '{self._default_provider_name}' is not configured or is not supported. "
                                   f"Available types: {list(PROVIDER_MAP.keys())}")


    def _load_configured_providers(self) -> None:
        """Loads and initializes all providers defined in the configuration."""
        providers_config = self._config.get('providers', {})
        if not providers_config:
            logger.warning("No providers configured in the '[providers]' section.")
            # Attempt to load at least the default provider if its config exists implicitly
            # and it's a known type
            if self._default_provider_name in PROVIDER_MAP:
                 provider_config = self._config.get(f'providers.{self._default_provider_name}', {})
                 # Check if there's *any* config for the default, even if empty {}
                 # This allows providers like Ollama (with defaults) or Gemini/OpenAI (using env vars)
                 # to load even without explicit [providers.<name>] sections, as long as
                 # llmcore.default_provider is set correctly.
                 # We still need *some* indication it's intended to be used.
                 # Let's refine this: only attempt implicit load if default_provider is set.
                 if self._config.get('llmcore.default_provider'):
                      logger.info(f"Attempting to load default provider '{self._default_provider_name}' based on implicit config/env vars.")
                      providers_config[self._default_provider_name] = provider_config
                 else:
                      logger.error("No providers section found and no default provider explicitly set in llmcore.default_provider.")
                      return # Cannot proceed without any provider config indication
            else:
                 logger.error(f"Default provider '{self._default_provider_name}' is not a supported type.")
                 return # No providers configured or default is invalid type

        loaded_provider_names = []
        for name, config_data in providers_config.items():
            provider_name_lower = name.lower()
            if provider_name_lower in PROVIDER_MAP:
                provider_cls = PROVIDER_MAP[provider_name_lower]
                try:
                    # Ensure config is a dictionary, even if empty
                    provider_config_dict = config_data if isinstance(config_data, dict) else {}
                    # Instantiate the provider class with its specific config
                    self._providers[provider_name_lower] = provider_cls(provider_config_dict)
                    logger.info(f"Provider '{provider_name_lower}' initialized successfully.")
                    loaded_provider_names.append(provider_name_lower)
                except ImportError as e:
                     # Catch missing optional dependencies (e.g., `pip install llmcore[openai]`)
                     logger.error(f"Failed to initialize provider '{provider_name_lower}': Missing required library. "
                                  f"Install it using 'pip install llmcore[{provider_name_lower}]'. Error: {e}")
                     # Optionally raise ProviderError here or just log and skip
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
            ProviderError: If the requested provider was configured but failed to initialize.
        """
        target_name = name.lower() if name else self._default_provider_name
        provider_instance = self._providers.get(target_name)

        if provider_instance is None:
            # Check if it was configured but failed to load vs not configured at all
            providers_config = self._config.get('providers', {})
            if target_name in providers_config or (target_name == self._default_provider_name and self._config.get('llmcore.default_provider')):
                 # It was configured (or was the default) but failed to load
                 raise ProviderError(target_name, "Provider was configured but failed to initialize (check logs for details, ensure dependencies are installed).")
            else:
                 # It was not configured at all
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
            ProviderError: If the default provider was configured but failed to initialize.
        """
        return self.get_provider(self._default_provider_name) # Use get_provider for consistent error handling

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
        # Use asyncio.gather to close providers concurrently
        close_tasks = []
        for name, provider in self._providers.items():
            # Check if the provider has an async close method
            if hasattr(provider, 'close') and asyncio.iscoroutinefunction(provider.close):
                close_tasks.append(self._close_single_provider(name, provider))

        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    # Log errors that occurred during closing, but don't stop others
                    logger.error(f"Error during provider closure: {result}", exc_info=result)

        logger.info("Provider connections closure attempt complete.")

    async def _close_single_provider(self, name: str, provider: BaseProvider):
        """Helper coroutine to close a single provider and log errors."""
        try:
            await provider.close() # type: ignore # Assume close is async if it exists and iscoroutinefunction
            logger.info(f"Provider '{name}' closed.")
        except Exception as e:
            logger.error(f"Error closing provider '{name}': {e}", exc_info=True)
            # Raise the exception so asyncio.gather can report it
            raise
