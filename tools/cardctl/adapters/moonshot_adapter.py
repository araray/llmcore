# tools/cardctl/adapters/moonshot_adapter.py
"""Backward-compatibility shim.

The Moonshot (Kimi) adapter was renamed to the canonical ``KimiAdapter`` in
``kimi_adapter.py`` (provider key ``"kimi"``, matching ``Provider.KIMI`` and the
``default_cards/kimi/`` directory).  This module re-exports the new class under
the old name so existing imports keep working.
"""

from __future__ import annotations

from .kimi_adapter import KimiAdapter as MoonshotAdapter

__all__ = ["MoonshotAdapter"]
