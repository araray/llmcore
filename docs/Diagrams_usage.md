# Documentation diagrams — usage guide

The diagrams in the README and the guides are **generated from code** in
`tools/diagrams/`, not drawn by hand. The `.svg` files under `docs/diagrams/`
are a build product that happens to be committed — committed because GitHub
has to serve them, not because anyone maintains them directly.

**Edit the generator, never the SVG.** A test fails if they disagree.

---

## Contents

- [Regenerating](#regenerating)
- [Adding a diagram](#adding-a-diagram)
- [The layout DSL](#the-layout-dsl)
- [The design system](#the-design-system)
- [Verifying a change](#verifying-a-change)
- [Why these choices](#why-these-choices)
- [What is deliberately not a diagram](#what-is-deliberately-not-a-diagram)

---

## Regenerating

```bash
python -m tools.diagrams            # write docs/diagrams/*.svg
python -m tools.diagrams --check    # fail if what is committed has drifted
python -m tools.diagrams --out /tmp/preview   # write somewhere else
```

`--check` is what `tests/docs/test_diagrams.py` enforces, so a forgotten
regeneration fails the suite rather than shipping a picture that no longer
matches its source.

---

## Adding a diagram

Three steps, and the tests check all three.

**1. Write a function in `tools/diagrams/diagrams.py`.** It builds a
`Diagram` and registers it under the slug that becomes its filename:

```python
def failover_path():
    W, H = 760, 300
    d = Diagram(
        W, H,
        "Pool failover",
        "A request tries the first pool member; on a retryable failure it "
        "moves to the next, and the failed target cools down.",
    )
    d.bg()
    d.text(W / 2, 42, "Pool failover", size=17, bold=True)
    d.labelled_box(60, 90, 200, 60, "openai:gpt-5.4", tone="api", mono_title=True)
    d.arrow(268, 120, 330, 120)
    d.labelled_box(338, 90, 200, 60, "anthropic:opus", tone="api", mono_title=True)
    return reg("failover-path", d)
```

The `title` and `desc` are not decoration: they become the SVG's `<title>`
and `<desc>`, which is what a screen reader reads. A test requires the
description to be longer than 40 characters, because "Pool failover" twice
helps nobody.

**2. Add it to `build_all()`**, in the order you want it generated.

**3. Reference it from a doc** with descriptive alt text:

```markdown
![A request tries the first pool member, then moves to the next on a
retryable failure while the failed target cools down](diagrams/failover-path.svg)
```

Paths are relative to the referring file: `docs/diagrams/x.svg` from the
README, `diagrams/x.svg` from inside `docs/`.

Then run `python -m tools.diagrams` and commit both the generator change and
the regenerated SVG.

---

## The layout DSL

`tools/diagrams/dsl.py`. Everything takes plain pixel coordinates in a
top-left origin.

| call | draws |
|---|---|
| `bg()` | the light card background — call it first |
| `box(x, y, w, h, tone=…, dashed=…, fill=…)` | a rounded rectangle |
| `band(x, y, w, h)` | a flat fill with no stroke, for header strips |
| `text(x, y, s, size=…, mono=…, bold=…, anchor=…)` | a label |
| `vtext(x, y, s)` | a label rotated to run up a vertical lane |
| `line(x1, y1, x2, y2, dashed=…)` | a plain connector |
| `arrow(x1, y1, x2, y2)` | a connector with a drawn head |
| `labelled_box(x, y, w, h, title, subtitle=…)` | box plus centred caption |

Two things worth knowing:

- **`arrow()` draws its own head as a polygon** rather than using
  `marker-end`, because markers do not survive a round trip through Krita and
  a diagram of headless lines says nothing about direction.
- **Arrows need about 14px of run.** Below that the head overshoots its own
  shaft and the connector reads as a stray caret. A test enforces this.

`labelled_box()` centres a single title. If you want more than two lines in a
box, call `box()` and place the text yourself — the convenience version
collided with extra lines the first time these were drawn.

---

## The design system

Colours come from `dsl.py`, so a diagram should not invent its own. Each tone
is a (stroke, fill) pair carrying meaning rather than decoration:

| tone | meaning | example |
|---|---|---|
| `api` | entry points, the facade | `LLMCore`, `AgentManager` |
| `data` | storage, cards, memory, results | user cards, `RESULT` |
| `agent` | agents and cognition | the cognitive phases |
| `risk` | sandboxing, isolation, spend | the sandbox, Docker/VM |
| `neutral` | everything else | supporting registries |

Monospace (`mono=True`) is for identifiers — module paths, config keys, class
names — and sans for prose. The same split the written docs use.

---

## Verifying a change

**Geometry bugs are invisible in markup and obvious on screen.** The first
pass of these eight shipped with a title overlapping a label by 5px, a box
26px below the bottom of its own canvas, a loop label running off the left
edge, and arrowheads longer than their 4px shafts. None of that is visible in
the SVG source; all of it is visible in a render.

So look at it. Cheapest first:

```bash
python -m tools.diagrams --out /tmp/d && rsvg-convert -w 900 /tmp/d/your-diagram.svg -o /tmp/d/preview.png
```

`rsvg-convert` is the more useful of the two checks, because it resolves
fonts differently from the machine the layout was designed on — which is
exactly the failure mode to catch, since the final text is laid out by the
*reader's* browser, not by anything here.

For a richer look, the **av-krita-mcp** server renders through real Krita:

```bash
python -m tools.diagrams --krita     # fragments with the font substituted
```

Paste a fragment into `krita_add_svg_shapes` on a document of the diagram's
size, then `krita_see_document`. The font is substituted because Krita needs
one real installed family while the shipped file names a portable stack — so
Krita proves the **layout**, not the exact glyphs a reader will see. That is
also why boxes are padded generously.

Then run the tests, which catch the mechanical problems for free:

```bash
python -m pytest tests/docs/test_diagrams.py -q
```

---

## Why these choices

**A light card background, not transparency.** GitHub renders a README on
white *or* near-black depending on the reader's theme. An SVG with a
transparent background and dark strokes disappears on the dark one. A
self-contained light card is legible against either — which is what the ASCII
blocks these replaced effectively were.

**Live `<text>`, not outlined paths.** The reader can select it, it scales
cleanly, and it stays searchable. The cost is that the browser lays it out
with its own fonts, which is why labels are centred with padding rather than
packed to measured widths.

**A portable font stack, not the one it was drawn with.** Krita rewrites
`font-family` to the single face it rendered with, so the generator's output
ships rather than Krita's re-export — otherwise a reader without that face
would get a fallback chosen by chance.

**Coordinates rounded to 2dp.** Dividing a width by three yields
`210.66666666666666`, which makes a one-pixel move unreadable in a diff.

---

## What is deliberately not a diagram

File trees and the exception hierarchy stay as fenced text: the README
project structure, the model-card directory listings, and the `SandboxError`
hierarchy.

An image of a directory listing is strictly worse than the listing. It cannot
be copied, searched, or diffed, and it carries no information a monospace
block does not. Converting them would be change for its own sake.
