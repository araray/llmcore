# src/llmcore/config/__init__.py
"""
Configuration module for the LLMCore library.

This package handles the loading and management of configuration
settings for the library, leveraging the `confy` library and
a default TOML configuration file.

Configuration files:
    - default_config.toml: Packaged defaults
    - User config: ~/.config/llmcore/config.toml
    - Custom config: Specified via LLMCore.create(config_file_path=...)

Environment variables:
    - Prefix: LLMCORE_
    - Nested keys use double underscores: LLMCORE_AGENTS__SANDBOX__MODE
"""

# Configuration-related exports can be added here as needed
# Example:
# from .loader import ConfigLoader
# __all__ = ["ConfigLoader"]
