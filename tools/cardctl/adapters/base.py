# tools/cardctl/adapters/base.py
"""Base adapter interface and normalized model representation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedModel:
    """Provider-agnostic intermediate representation of a discovered model.

    Adapters populate this from their respective APIs.  The
    :class:`~tools.cardctl.core.builder.CardBuilder` then converts it
    into a ``ModelCard``-compatible dict.
    """

    model_id: str
    provider: str

    # Display / description
    display_name: str | None = None
    description: str | None = None

    # Type & lifecycle
    model_type: str = "chat"
    is_deprecated: bool = False
    deprecation_date: str | None = None
    created_timestamp: int | None = None
    status: str = "active"

    # Context
    context_length: int | None = None
    max_output_tokens: int | None = None
    default_output_tokens: int | None = None

    # Capabilities
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_audio_input: bool = False
    supports_audio_output: bool = False
    supports_json_mode: bool = False
    supports_structured_output: bool = False
    supports_reasoning: bool = False
    supports_fim: bool = False

    # Architecture (if the API exposes it)
    architecture_family: str | None = None
    architecture_type: str | None = None  # "transformer", "dense", "moe", etc.
    parameter_count: str | None = None
    active_parameters: str | None = None

    # Metadata
    owned_by: str | None = None
    aliases: list[str] = field(default_factory=list)
    open_weights: bool = False
    license: str | None = None
    tags: list[str] = field(default_factory=list)

    # Provider-specific raw data (for extension fields)
    raw_api_data: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """Abstract base for provider model-discovery adapters.

    Subclasses implement :meth:`fetch_models` which calls the provider API
    and returns a list of :class:`NormalizedModel` instances.
    """

    # --- Class-level metadata (override in subclasses) ---

    provider_name: str = ""
    """Canonical provider name (e.g. ``"openai"``, ``"anthropic"``)."""

    requires_api_key: bool = True
    """Whether this adapter needs an API key (False for local Ollama)."""

    api_key_env_var: str = ""
    """Environment variable name for the API key."""

    base_url: str = ""
    """Base URL for the provider API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._api_key = api_key
        if base_url:
            self.base_url = base_url

    def get_api_key(self) -> str | None:
        """Resolve API key from init arg, env var, or None."""
        if self._api_key:
            return self._api_key
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var)
        return None

    def check_api_key(self) -> None:
        """Raise if an API key is required but not available."""
        if self.requires_api_key and not self.get_api_key():
            raise RuntimeError(
                f"API key required for {self.provider_name}. "
                f"Set ${self.api_key_env_var} or pass --api-key."
            )

    @abstractmethod
    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch all models from the provider API.

        Returns:
            List of normalized model representations.
        """
        ...
