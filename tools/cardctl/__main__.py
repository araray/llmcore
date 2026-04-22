# tools/cardctl/__main__.py
"""Entry point for cardctl.

Supports all invocation styles:
    python -m tools.cardctl ...     (package module)
    python tools/cardctl ...        (directory with __main__)
    python tools/cardctl.py ...     (convenience script)
"""

import sys
from pathlib import Path


def _bootstrap() -> None:
    """Ensure the repo root is on sys.path so absolute imports work."""
    # __file__ is tools/cardctl/__main__.py → repo root is 2 levels up
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# When run as `python tools/cardctl`, Python executes this file as a
# standalone script (__name__=="__main__", __package__==None), so
# relative imports like `from .cli import main` fail.  We detect that
# and fall back to absolute imports after bootstrapping sys.path.
if __package__ is None or __package__ == "":
    _bootstrap()
    from tools.cardctl.cli import main
else:
    from .cli import main

sys.exit(main())
