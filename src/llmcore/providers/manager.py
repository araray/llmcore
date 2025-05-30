# src/llmcore/providers/manager.py
"""
Provider Manager for LLMCore.

Handles the dynamic loading and management of LLM provider instances based on configuration.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

# Assume ConfyConfig type for hinting
try:
    from confy.loader import Config as ConfyConfig
except ImportError:
    ConfyConfig = Dict[str, Any] # type: ignore


from ..exceptions import ConfigError, ProviderError
from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .gemini_provider import GeminiProvider
# Import concrete implementations
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# --- Mapping from config provider name string to class ---
PROVIDER_MAP: Dict[str, Type[BaseProvider]] = {
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
    access to them. It passes the global raw payload logging setting
    to each provider instance.
    """
    _providers: Dict[str, BaseProvider]
    _config: ConfyConfig
    _default_provider_name: str

    def __init__(self, config: ConfyConfig):
        """
        Initializes the ProviderManager and loads configured providers.

        Args:
            config: The main LLMCore configuration object (ConfyConfig instance).
                    This config is expected to contain the 'llmcore.log_raw_payloads'
                    setting which will be passed to individual providers.

        Raises:
            ConfigError: If the default provider is not configured or supported,
                         or if its configuration is invalid.
            ProviderError: If a configured provider fails to initialize.
        """
        self._config = config
        self._providers = {}
        # Ensure default provider name is stored and processed in lowercase
        self._default_provider_name = self._config.get('llmcore.default_provider', 'ollama').lower()
        logger.info(f"ProviderManager initialized. Default provider set to '{self._default_provider_name}'.")

        self._load_configured_providers()

        # Check if the (now lowercased) default provider was successfully loaded
        if self._default_provider_name not in self._providers:
            providers_config = self._config.get('providers', {})
            if not isinstance(providers_config, dict): # Should be a dict
                providers_config = {}

            default_provider_section_exists = any(
                key.lower() == self._default_provider_name for key in providers_config.keys()
            )

            if default_provider_section_exists:
                 # A section for the default provider was in the config, but it failed to load.
                 raise ProviderError(self._default_provider_name, "Default provider was configured but failed to initialize (check logs/dependencies).")
            else:
                 # No section for the default provider was found.
                 # Check if the default provider *type* is known.
                 is_known_type = self._default_provider_name in PROVIDER_MAP

                 if not is_known_type:
                    # Additionally, check if any configured provider *declares* its type as the default provider name
                    # This handles cases like: default_provider = "my_custom_openai" and a section [providers.some_other_name] type = "my_custom_openai"
                    # However, this check is complex as 'my_custom_openai' itself needs to map to a class.
                    # The primary check should be if self._default_provider_name is a key in PROVIDER_MAP.
                    pass # Fall through to the more general error

                 if not is_known_type:
                    raise ConfigError(f"Default provider type '{self._default_provider_name}' is not a recognized provider type in PROVIDER_MAP.")
                 else: # It was a known type, but no section was defined for it, and it wasn't loaded implicitly.
                    raise ConfigError(f"Default provider '{self._default_provider_name}' is not configured in the '[providers]' section or failed to load. "
                                   f"Available base types: {list(PROVIDER_MAP.keys())}. Loaded provider instances: {list(self._providers.keys())}")


    def _load_configured_providers(self) -> None:
        """
        Loads and initializes all providers defined in the [providers] configuration section.
        It determines the provider class based on a 'type' field within each provider's
        configuration block, falling back to the section name if 'type' is not specified.
        It also passes the global `log_raw_payloads` setting to each provider.
        """
        providers_config = self._config.get('providers', {})
        if not isinstance(providers_config, dict):
            logger.warning("'[providers]' section in config is not a valid dictionary. No providers will be loaded.")
            providers_config = {}

        # Attempt to implicitly load the default provider if 'providers' is empty
        # but 'llmcore.default_provider' is set and is a known type.
        if not providers_config and self._default_provider_name in PROVIDER_MAP:
            logger.info(f"No '[providers]' section found. Attempting to load default provider '{self._default_provider_name}' implicitly.")
            # Try to get config for the default provider if it was defined directly under [providers.default_provider_name]
            # This handles cases where the user might have e.g. [providers.ollama] but not a general [providers] section.
            # More robustly, it should look for a section named self._default_provider_name
            # or a section that has type = self._default_provider_name.
            # For implicit loading, we assume the section name matches the default provider name.
            default_provider_specific_config = self._config.get(f'providers.{self._default_provider_name}', {})
            if isinstance(default_provider_specific_config, dict):
                 providers_config[self._default_provider_name] = default_provider_specific_config
            else: # Not a dict, treat as empty config for the default provider type
                 providers_config[self._default_provider_name] = {}


        if not providers_config:
            logger.warning("No providers configured in the '[providers]' section and no default provider could be implicitly loaded.")
            return

        loaded_provider_names = []
        # Get the global setting for raw payload logging from the main LLMCore config
        log_raw_payloads_global = self._config.get('llmcore.log_raw_payloads', False)
        logger.debug(f"Global 'log_raw_payloads' setting to be passed to providers: {log_raw_payloads_global}")

        for section_name, provider_specific_config in providers_config.items():
            current_section_name_lower = section_name.lower()
            if not isinstance(provider_specific_config, dict):
                logger.warning(f"Configuration for provider '{current_section_name_lower}' is not a valid dictionary. Skipping.")
                continue

            # Determine the provider type: use 'type' field in config, fallback to section_name itself as type key.
            # Example: [providers.my_openai_clone] type = "openai" -> provider_type_key = "openai"
            # Example: [providers.ollama] (no type field) -> provider_type_key = "ollama"
            provider_type_key = provider_specific_config.get("type", current_section_name_lower).lower()
            provider_cls = PROVIDER_MAP.get(provider_type_key)

            if not provider_cls:
                logger.warning(f"Provider base type '{provider_type_key}' (for section '{current_section_name_lower}') is not supported or mapped in PROVIDER_MAP. Skipping.")
                continue

            try:
                # Instantiate the provider class with its specific configuration block
                # AND the global raw payload logging flag.
                self._providers[current_section_name_lower] = provider_cls(
                    provider_specific_config,
                    log_raw_payloads=log_raw_payloads_global # Pass the flag here
                )
                logger.info(f"Provider instance '{current_section_name_lower}' (base type: '{provider_type_key}') initialized successfully. Raw payload logging: {log_raw_payloads_global}.")
                loaded_provider_names.append(current_section_name_lower)
            except ImportError as e:
                logger.error(f"Failed to initialize provider instance '{current_section_name_lower}' (base type: '{provider_type_key}'): Missing required library. "
                             f"Install dependencies (e.g., 'pip install llmcore[{provider_type_key}]'). Error: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize provider instance '{current_section_name_lower}' (base type: '{provider_type_key}'): {e}", exc_info=True)

        logger.debug(f"Successfully attempted to load provider instances: {loaded_provider_names}")
        if not self._providers:
            logger.warning("No provider instances were successfully loaded after processing configuration.")


    def get_provider(self, name: Optional[str] = None) -> BaseProvider:
        """
        Gets a specific provider instance by its configured section name,
        or the default provider if name is None.

        Args:
            name: The configuration section name of the provider instance (e.g., "my_openai_clone").
                  If None, returns the default provider instance.

        Returns:
            The initialized BaseProvider instance.

        Raises:
            ConfigError: If the requested provider name or default provider is not configured
                         or not a recognized type.
            ProviderError: If the provider was configured but failed to initialize.
        """
        target_name_lower = name.lower() if name else self._default_provider_name
        # self._default_provider_name is already lowercased in __init__

        provider_instance = self._providers.get(target_name_lower)

        if provider_instance is None:
            # Check if a section for this name existed in the original config
            # to distinguish between "not configured" and "configured but failed to load".
            original_providers_config = self._config.get('providers', {})
            if not isinstance(original_providers_config, dict): original_providers_config = {}

            section_existed = any(key.lower() == target_name_lower for key in original_providers_config.keys())

            if section_existed:
                 # It was in the config, but not in self._providers, meaning it failed to load.
                 raise ProviderError(target_name_lower, "Provider was configured but failed to initialize (check logs/dependencies).")
            else:
                 # It was not in the config under this name.
                 # Could it be a type name that wasn't instantiated?
                 if target_name_lower in PROVIDER_MAP:
                      raise ConfigError(f"Provider type '{target_name_lower}' is known, but no instance named '{target_name_lower}' "
                                        f"is configured or loaded. Loaded instances: {list(self._providers.keys())}")
                 else:
                      raise ConfigError(f"Provider instance or type '{target_name_lower}' not configured/supported. "
                                        f"Loaded instances: {list(self._providers.keys())}")
        return provider_instance

    def get_default_provider(self) -> BaseProvider:
        """Gets the instance of the configured default provider."""
        return self.get_provider(self._default_provider_name)

    def get_available_providers(self) -> List[str]:
        """Lists the names (config section names) of all successfully loaded provider instances."""
        return list(self._providers.keys())

    def update_log_raw_payloads_setting(self, enable: bool) -> None:
        """
        Updates the `log_raw_payloads_enabled` setting for all managed provider instances.

        Args:
            enable: True to enable raw payload logging, False to disable.
        """
        logger.info(f"ProviderManager updating raw payload logging for all providers to: {enable}")
        for provider_name, provider_instance in self._providers.items():
            try:
                provider_instance.log_raw_payloads_enabled = enable
                logger.debug(f"Updated raw_payload_logging for provider '{provider_name}' to {enable}.")
            except Exception as e_update:
                logger.error(f"Error updating raw_payload_logging for provider '{provider_name}': {e_update}")


    async def close_providers(self) -> None:
        """Closes connections or cleans up resources for all loaded providers."""
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
        try:
            await provider.close() # type: ignore
            logger.info(f"Provider instance '{name}' closed.")
        except Exception as e:
            logger.error(f"Error closing provider instance '{name}': {e}", exc_info=True)
            # Optionally re-raise or collect errors if critical
            # For now, just log, as it's a cleanup phase.
