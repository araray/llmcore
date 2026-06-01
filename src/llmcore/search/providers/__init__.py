# src/llmcore/search/providers/__init__.py
"""Concrete web/data search provider implementations.

Ships the Bright Data, Serper.dev, SerpApi and Semantic Scholar providers.
Additional providers can be added here and registered in
:data:`llmcore.search.manager.SEARCH_PROVIDER_MAP`.
"""

from .brightdata_provider import BrightDataSearchProvider
from .semanticscholar_provider import SemanticScholarSearchProvider
from .serpapi_provider import SerpApiSearchProvider
from .serper_provider import SerperSearchProvider

__all__ = [
    "BrightDataSearchProvider",
    "SemanticScholarSearchProvider",
    "SerpApiSearchProvider",
    "SerperSearchProvider",
]
