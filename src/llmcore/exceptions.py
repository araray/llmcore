# src/llmcore/exceptions.py
"""
Custom exceptions for the LLMCore library.

This module defines a hierarchy of custom exception classes to provide
more specific error information and allow for targeted error handling
by applications using LLMCore.
"""

class LLMCoreError(Exception):
    """Base class for all LLMCore specific errors."""
    def __init__(self, message: str = "An unspecified error occurred in LLMCore."):
        super().__init__(message)

class ConfigError(LLMCoreError):
    """Raised for errors related to configuration loading or validation."""
    def __init__(self, message: str = "Configuration error."):
        super().__init__(message)

class ProviderError(LLMCoreError):
    """Raised for errors originating from an LLM provider (e.g., API errors, connection issues)."""
    def __init__(self, provider_name: str = "Unknown", message: str = "Provider error."):
        self.provider_name = provider_name
        super().__init__(f"Error with provider '{provider_name}': {message}")

class StorageError(LLMCoreError):
    """Base class for errors related to storage operations."""
    def __init__(self, message: str = "Storage error."):
        super().__init__(message)

class SessionStorageError(StorageError):
    """Raised for errors specific to session storage operations."""
    def __init__(self, message: str = "Session storage error."):
        super().__init__(message)

class VectorStorageError(StorageError):
    """Raised for errors specific to vector storage operations."""
    def __init__(self, message: str = "Vector storage error."):
        super().__init__(message)

class SessionNotFoundError(StorageError):
    """
    Raised when a specified session ID is not found in storage.
    Inherits from StorageError as it's a storage-related lookup failure.
    """
    def __init__(self, session_id: str, message: str = "Session not found."):
        self.session_id = session_id
        super().__init__(f"{message} Session ID: '{session_id}'")

class ContextError(LLMCoreError):
    """Base class for errors related to context management."""
    def __init__(self, message: str = "Context management error."):
        super().__init__(message)

class ContextLengthError(ContextError):
    """Raised when the context length exceeds the model's maximum limit, even after truncation attempts."""
    def __init__(self, model_name: str = "Unknown", limit: int = 0, actual: int = 0, message: str = "Context length exceeded."):
        self.model_name = model_name
        self.limit = limit
        self.actual = actual
        super().__init__(f"{message} Model: '{model_name}', Limit: {limit} tokens, Actual: {actual} tokens.")

class EmbeddingError(LLMCoreError):
    """Raised for errors related to embedding generation."""
    def __init__(self, model_name: str = "Unknown", message: str = "Embedding generation error."):
        self.model_name = model_name
        super().__init__(f"Error with embedding model '{model_name}': {message}")

class MCPError(LLMCoreError):
    """Raised for errors related to Model Context Protocol (MCP) formatting or handling."""
    def __init__(self, message: str = "MCP processing error."):
        super().__init__(message)

# Further specific exceptions can be added as needed, for example:
# class AuthenticationError(ProviderError):
#     """Raised for authentication failures with an LLM provider."""
#     def __init__(self, provider_name: str, message: str = "Authentication failed."):
#         super().__init__(provider_name, message)

# class RateLimitError(ProviderError):
#     """Raised when an LLM provider's rate limit is exceeded."""
#     def __init__(self, provider_name: str, message: str = "Rate limit exceeded."):
#         super().__init__(provider_name, message)
