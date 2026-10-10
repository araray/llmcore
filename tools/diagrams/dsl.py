"""A small layout DSL for llmcore's documentation diagrams.

Authoring happens here rather than by hand-placing coordinates, so every
diagram shares one design system and a box can be moved without recomputing
the arrows that touch it.

``python -m tools.diagrams`` regenerates ``docs/diagrams/*.svg``, and
``tests/docs/test_diagrams.py`` fails if what is committed no longer matches
what this produces — so the pictures cannot drift away from their source.

Design decisions, once, for everything:

* **An explicit light card background.** The diagrams sit in a README that
  GitHub renders on white *or* near-black, and an SVG with a transparent
  background and dark strokes vanishes on the dark one. A self-contained
  light card is legible on both; it is what the ASCII blocks it replaced
  effectively were.
* **Text stays live ``<text>``**, so the *browser* lays it out and a reader
  can select and scale it. Labels therefore name a portable font stack and
  are centred with generous padding, because a viewer's font may be wider
  than the one the layout was checked against.
* **Monospace for anything that is an identifier** — module paths, config
  keys, class names — and sans for prose, the same split the prose uses.

**Verifying a change.** Geometry bugs are invisible in markup and obvious on
screen: the first pass of these diagrams had a title overlapping a label by
5px, a box 26px below the bottom of its canvas, a label off the left edge,
and arrowheads longer than their own shafts. ``--krita`` emits each diagram
with the font substituted for one Krita has, to paste into
``krita_add_svg_shapes`` and look at via the av-krita-mcp server. Rendering
with ``rsvg-convert`` is the cheaper check and catches overflow under a
different font.
"""
from __future__ import annotations

import html
import re

# --- design system -------------------------------------------------------

INK = "#0f172a"        # primary text
INK_SOFT = "#475569"   # secondary text, strokes
LINE = "#94a3b8"       # connectors
CARD = "#f8fafc"       # page background of the diagram
PANEL = "#ffffff"      # box fill
BAND = "#eef2f7"       # grouping band / header fill

ACCENT = {
    "api":     ("#2563eb", "#eff6ff"),   # entry points, facade
    "data":    ("#059669", "#ecfdf5"),   # storage, cards, memory
    "agent":   ("#7c3aed", "#f5f3ff"),   # agents, cognition
    "risk":    ("#d97706", "#fffbeb"),   # sandbox, isolation, spend
    "neutral": (INK_SOFT, PANEL),
}

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


class Diagram:
    """Collects shapes, then emits a fragment for Krita and a file for docs."""

    def __init__(self, width: int, height: int, title: str, desc: str = "") -> None:
        self.w, self.h = width, height
        self.title, self.desc = title, desc
        self.parts: list[str] = []

    # -- primitives -------------------------------------------------------

    def bg(self) -> "Diagram":
        self.parts.append(
            f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="10" '
            f'fill="{CARD}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        return self

    def box(self, x, y, w, h, *, tone="neutral", dashed=False, radius=7, fill=None):
        stroke, default_fill = ACCENT[tone]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill or default_fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
        )
        return self

    def band(self, x, y, w, h, *, fill=BAND, radius=7):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="none"/>'
        )
        return self

    def text(self, x, y, s, *, size=13, mono=False, bold=False, fill=INK,
             anchor="middle", opacity=None):
        if not str(s).strip():
            # An empty <text> is invisible but still ships in the asset; a
            # caller passing "" means "no label", so honour that.
            return self
        family = MONO if mono else SANS
        weight = ' font-weight="600"' if bold else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}px" '
            f'fill="{fill}" text-anchor="{anchor}"{weight}{op}>{esc(s)}</text>'
        )
        return self

    def vtext(self, x, y, s, *, size=11, fill=INK_SOFT, up=True):
        """Text rotated to run along a vertical lane.

        A horizontal label beside a narrow return lane either collides with
        the lane or runs off the canvas; rotating it is the only placement
        that fits in ~30px of gutter.
        """
        family = SANS
        angle = -90 if up else 90
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}px" '
            f'fill="{fill}" text-anchor="middle" '
            f'transform="rotate({angle} {x} {y})">{esc(s)}</text>'
        )
        return self

    def line(self, x1, y1, x2, y2, *, stroke=LINE, width=1.6, dashed=False):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}"{dash}/>'
        )
        return self

    def arrow(self, x1, y1, x2, y2, *, stroke=LINE, width=1.6, dashed=False, head=6.5):
        """A connector with an explicit polygon head.

        Drawn rather than using ``marker-end``: Krita's SVG round-trip does not
        carry markers through, and a arrowless diagram reads as a diagram about
        nothing in particular.
        """
        import math

        self.line(x1, y1, x2, y2, stroke=stroke, width=width, dashed=dashed)
        angle = math.atan2(y2 - y1, x2 - x1)
        back, spread = head * 1.9, head * 0.95
        bx, by = x2 - back * math.cos(angle), y2 - back * math.sin(angle)
        px, py = -math.sin(angle) * spread, math.cos(angle) * spread
        self.parts.append(
            f'<path d="M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} '
            f'L {bx - px:.1f} {by - py:.1f} Z" fill="{stroke}" stroke="none"/>'
        )
        return self

    # -- composites -------------------------------------------------------

    def labelled_box(self, x, y, w, h, title, *, subtitle=None, tone="neutral",
                     mono_title=False, dashed=False, size=13):
        self.box(x, y, w, h, tone=tone, dashed=dashed)
        if subtitle:
            self.text(x + w / 2, y + h / 2 - 3, title, size=size, bold=True,
                      mono=mono_title)
            self.text(x + w / 2, y + h / 2 + 14, subtitle, size=size - 2,
                      fill=INK_SOFT, mono=True)
        else:
            self.text(x + w / 2, y + h / 2 + 4.5, title, size=size, bold=True,
                      mono=mono_title)
        return self

    # -- output -----------------------------------------------------------

    @staticmethod
    def _round(markup: str) -> str:
        """Trim coordinate noise.

        Dividing a width by three yields ``210.66666666666666``, which is
        meaningless precision in a committed asset and makes diffs unreadable
        when a box moves by a pixel.
        """
        def fix(m: "re.Match[str]") -> str:
            value = float(m.group(0))
            return f"{value:.2f}".rstrip("0").rstrip(".")

        return re.sub(r"\d+\.\d{3,}", fix, markup)

    def fragment(self) -> str:
        return self._round("".join(self.parts))

    def standalone(self) -> str:
        """Wrap the authored fragment as a complete, docs-ready SVG.

        The authored fragment is what ships, not Krita's re-export, for one
        reason: Krita rewrites ``font-family`` to the single face it rendered
        with (DejaVu), and a reader without that font installed would get a
        fallback chosen by chance. The portable stack has to survive into the
        file. Krita's job here was to *render and prove the layout*, which it
        did -- every one of these was corrected after looking at it.
        """
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" role="img" '
            f'aria-labelledby="title desc">\n'
            f"  <title id=\"title\">{esc(self.title)}</title>\n"
            f"  <desc id=\"desc\">{esc(self.desc or self.title)}</desc>\n"
            f"  {self.fragment()}\n"
            f"</svg>\n"
        )
