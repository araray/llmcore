# src/llmcore/search/__init__.py
"""Web/data **search** providers for LLMCore.

This package adds a second family of pluggable backends alongside the LLM
:mod:`llmcore.providers`: *search providers* that give consuming applications a
uniform, config-driven way to run web searches, scrape URLs, perform AI-ranked
discovery, and collect structured datasets — used "just like" LLM providers.

Public surface::

    from llmcore.search import (
        BaseSearchProvider,
        SearchCapability,
        SearchProviderManager,
        BrightDataSearchProvider,
        WebSearchResult,
        ScrapeResult,
        DiscoverResult,
        DatasetSnapshot,
        DatasetInfo,
        DatasetMetadata,
    )

See :class:`llmcore.search.base.BaseSearchProvider` for the provider contract
and :class:`llmcore.search.manager.SearchProviderManager` for discovery/loading.
"""

from .base import BaseSearchProvider, SearchCapability
from .manager import SEARCH_PROVIDER_MAP, SearchProviderManager
from .models import (
    DatasetField,
    DatasetInfo,
    DatasetMetadata,
    DatasetSnapshot,
    DiscoverItem,
    DiscoverResult,
    ScrapeResult,
    SearchItem,
    SearchResultBase,
    WebSearchResult,
)
from .providers.brightdata_provider import BrightDataSearchProvider
from .providers.serper_provider import SerperSearchProvider

__all__ = [
    # Base / capabilities
    "BaseSearchProvider",
    "SearchCapability",
    # Manager
    "SearchProviderManager",
    "SEARCH_PROVIDER_MAP",
    # Providers
    "BrightDataSearchProvider",
    "SerperSearchProvider",
    # Result models
    "SearchResultBase",
    "SearchItem",
    "WebSearchResult",
    "ScrapeResult",
    "DiscoverItem",
    "DiscoverResult",
    "DatasetInfo",
    "DatasetField",
    "DatasetMetadata",
    "DatasetSnapshot",
]
