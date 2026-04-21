# tools/cardctl/__main__.py
"""Entry point for ``python -m tools.cardctl``."""
import sys
from .cli import main
sys.exit(main())
