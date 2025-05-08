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
from .gemini_provider import GeminiProvider
from .github_mcp_provider import GithubMCPProvider # Added GithubMCPProvider

logger = logging.getLogger(__name__)

# --- Mapping from config provider name string to class ---
PROVIDER_MAP: Dict[str, Type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "github_mcp": GithubMCPProvider, # Added GithubMCPProvider mapping
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
        self._default_provider_name = self._config.get('llmcore.default_provider', 'ollama')
        logger.info(f"ProviderManager initialized. Default provider set to '{self._default_provider_name}'.")

        self._load_configured_providers()

        if self._default_provider_name not in self._providers:
            providers_config = self._config.get('providers', {})
            if self._default_provider_name in providers_config:
                 raise ProviderError(self._default_provider_name, "Default provider was configured but failed to initialize.")
            else:
                 raise ConfigError(f"Default provider '{self._default_provider_name}' is not configured or is not supported. "
                                   f"Available types: {list(PROVIDER_MAP.keys())}")


    def _load_configured_providers(self) -> None:
        """Loads and initializes all providers defined in the configuration."""
        providers_config = self._config.get('providers', {})
        if not providers_config:
            logger.warning("No providers configured in the '[providers]' section.")
            # Attempt implicit load of default provider if possible (logic remains the same)
            if self._default_provider_name in PROVIDER_MAP:
                 provider_config = self._config.get(f'providers.{self._default_provider_name}', {})
                 if self._config.get('llmcore.default_provider'):
                      logger.info(f"Attempting implicit load of default provider '{self._default_provider_name}'.")
                      providers_config[self._default_provider_name] = provider_config
                 else: logger.error("No providers section and no default provider explicitly set."); return
            else: logger.error(f"Default provider '{self._default_provider_name}' is not a supported type."); return

        loaded_provider_names = []
        for name, config_data in providers_config.items():
            provider_name_lower = name.lower()
            if provider_name_lower in PROVIDER_MAP:
                provider_cls = PROVIDER_MAP[provider_name_lower]
                try:
                    provider_config_dict = config_data if isinstance(config_data, dict) else {}
                    self._providers[provider_name_lower] = provider_cls(provider_config_dict)
                    logger.info(f"Provider '{provider_name_lower}' initialized successfully.")
                    loaded_provider_names.append(provider_name_lower)
                except ImportError as e:
                     logger.error(f"Failed to initialize provider '{provider_name_lower}': Missing required library. "
                                  f"Install dependencies (e.g., 'pip install llmcore[{provider_name_lower}]'). Error: {e}")
                except Exception as e:
                    logger.error(f"Failed to initialize provider '{provider_name_lower}': {e}", exc_info=True)
            else:
                logger.warning(f"Configured provider '{name}' is not supported or mapped. Skipping.")
        logger.debug(f"Successfully loaded providers: {loaded_provider_names}")


    def get_provider(self, name: Optional[str] = None) -> BaseProvider:
        """
        Gets a specific provider instance by name, or the default provider if name is None.
        """
        # (Logic remains the same)
        target_name = name.lower() if name else self._default_provider_name
        provider_instance = self._providers.get(target_name)
        if provider_instance is None:
            providers_config = self._config.get('providers', {})
            if target_name in providers_config or (target_name == self._default_provider_name and self._config.get('llmcore.default_provider')):
                 raise ProviderError(target_name, "Provider configured but failed to initialize (check logs/dependencies).")
            else:
                 raise ConfigError(f"Provider '{target_name}' not configured/supported. Loaded: {list(self._providers.keys())}")
        return provider_instance

    def get_default_provider(self) -> BaseProvider:
        """Gets the instance of the configured default provider."""
        # (Remains the same)
        return self.get_provider(self._default_provider_name)

    def get_available_providers(self) -> List[str]:
        """Lists the names of all successfully loaded providers."""
        # (Remains the same)
        return list(self._providers.keys())

    async def close_providers(self) -> None:
        """Closes connections or cleans up resources for all loaded providers."""
        # (Logic remains the same)
        logger.info("Closing provider connections...")
        close_tasks = [self._close_single_provider(name, provider)
                       for name, provider in self._providers.items()
                       if hasattr(provider, 'close') and asyncio.iscoroutinefunction(provider.close)]
        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception): logger.error(f"Error during provider closure: {result}", exc_info=result)
        logger.info("Provider connections closure attempt complete.")

    async def _close_single_provider(self, name: str, provider: BaseProvider):
        """Helper coroutine to close a single provider and log errors."""
        # (Remains the same)
        try: await provider.close(); logger.info(f"Provider '{name}' closed.") # type: ignore
        except Exception as e: logger.error(f"Error closing provider '{name}': {e}", exc_info=True); raise
