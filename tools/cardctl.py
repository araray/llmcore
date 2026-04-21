#!/usr/bin/env python3
# tools/cardctl.py
"""Convenience entry point for cardctl.

Usage::

    python tools/cardctl.py generate openai
    python tools/cardctl.py validate
    python tools/cardctl.py stats
"""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``from tools.cardctl...`` works.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.cardctl.cli import main

sys.exit(main())
