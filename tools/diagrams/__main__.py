"""Regenerate ``docs/diagrams/*.svg``.

    python -m tools.diagrams              # write the SVGs
    python -m tools.diagrams --check      # fail if what is committed differs
    python -m tools.diagrams --krita      # font-substituted fragments to paste
                                          # into krita_add_svg_shapes
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .diagrams import build_all
from .dsl import MONO, SANS

#: Where the committed assets live, relative to the repository root.
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs" / "diagrams"

#: Faces Krita has. The shipped files name a portable *stack* so the browser
#: can resolve it; Krita needs one real family to render a preview with.
KRITA_SANS, KRITA_MONO = "DejaVu Sans", "DejaVu Sans Mono"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.diagrams", description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed SVGs match this source; write nothing")
    parser.add_argument("--krita", action="store_true",
                        help="print font-substituted fragments for visual checking")
    parser.add_argument("--out", default=None, help="write somewhere other than docs/diagrams")
    args = parser.parse_args(argv)

    diagrams = build_all()
    out_dir = pathlib.Path(args.out) if args.out else OUT_DIR

    if args.krita:
        for slug, diagram in diagrams.items():
            fragment = diagram.fragment().replace(SANS, KRITA_SANS).replace(MONO, KRITA_MONO)
            print(f"\n# ---- {slug}  ({diagram.w}x{diagram.h}) ----\n{fragment}")
        return 0

    if args.check:
        stale, missing = [], []
        for slug, diagram in diagrams.items():
            path = out_dir / f"{slug}.svg"
            if not path.exists():
                missing.append(slug)
            elif path.read_text() != diagram.standalone():
                stale.append(slug)
        for slug in missing:
            print(f"missing: {slug}.svg", file=sys.stderr)
        for slug in stale:
            print(f"stale:   {slug}.svg", file=sys.stderr)
        if missing or stale:
            print("\nRun `python -m tools.diagrams` to regenerate.", file=sys.stderr)
            return 1
        print(f"{len(diagrams)} diagram(s) match their source.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, diagram in diagrams.items():
        path = out_dir / f"{slug}.svg"
        svg = diagram.standalone()
        changed = not path.exists() or path.read_text() != svg
        path.write_text(svg)
        print(f"  {'wrote  ' if changed else 'current'} {slug}.svg  ({diagram.w}x{diagram.h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
