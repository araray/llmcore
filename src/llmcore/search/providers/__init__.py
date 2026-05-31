# src/llmcore/search/providers/__init__.py
"""Concrete web/data search provider implementations.

Currently ships the Bright Data provider.  Additional providers can be added
here and registered in
:data:`llmcore.search.manager.SEARCH_PROVIDER_MAP`.
"""

from .brightdata_provider import BrightDataSearchProvider

__all__ = ["BrightDataSearchProvider"]
