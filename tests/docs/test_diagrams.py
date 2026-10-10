# tests/docs/test_diagrams.py
"""The committed diagrams must still match the code that draws them.

Generated assets rot the moment the generator becomes optional: someone edits
the SVG directly, the next regeneration silently reverts it, and nobody can
tell which was intended. This makes ``tools/diagrams`` the source of truth and
the ``.svg`` files a build product that happens to be committed -- committed
because GitHub has to serve them, not because they are hand-maintained.

It also covers the properties that make these usable *as docs*, each of which
was a real defect at some point while drawing them.
"""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

from tools.diagrams import build_all
from tools.diagrams.__main__ import OUT_DIR

SVG = "{http://www.w3.org/2000/svg}"


def prose_of(path: pathlib.Path) -> str:
    """A doc's text with fenced code blocks removed.

    ``docs/Diagrams_usage.md`` shows an example reference to a diagram that
    does not exist, inside a fence. Scanning raw text treats that sample as a
    broken link -- which this test did, on its first run, correctly by its own
    rule and wrongly in substance.
    """
    out, fenced = [], False
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


@pytest.fixture(scope="module")
def diagrams():
    return build_all()


class TestTheCommittedFilesMatchTheirSource:
    def test_every_diagram_has_a_committed_file(self, diagrams):
        missing = [s for s in diagrams if not (OUT_DIR / f"{s}.svg").exists()]
        assert not missing, (
            f"no committed SVG for {missing}. Run `python -m tools.diagrams`."
        )

    def test_no_committed_file_has_drifted(self, diagrams):
        stale = [
            slug
            for slug, built in diagrams.items()
            if (OUT_DIR / f"{slug}.svg").read_text() != built.standalone()
        ]
        assert not stale, (
            f"{stale} differ from what tools/diagrams produces. Edit the "
            f"generator rather than the SVG, then run `python -m tools.diagrams`."
        )

    def test_no_orphan_svgs(self, diagrams):
        """A diagram removed from the code should not leave its picture
        behind, still referenced and quietly out of date."""
        on_disk = {p.stem for p in OUT_DIR.glob("*.svg")}
        assert on_disk - set(diagrams) == set()


class TestTheyWorkAsDocumentation:
    def test_each_is_valid_xml(self, diagrams):
        for slug in diagrams:
            ET.parse(OUT_DIR / f"{slug}.svg")  # raises if malformed

    def test_each_carries_a_title_and_description(self, diagrams):
        """A screen reader should get the explanation, not a filename."""
        for slug in diagrams:
            root = ET.parse(OUT_DIR / f"{slug}.svg").getroot()
            title = root.find(f"{SVG}title")
            desc = root.find(f"{SVG}desc")
            assert title is not None and (title.text or "").strip(), slug
            assert desc is not None and len(desc.text or "") > 40, (
                f"{slug}: the description should explain the diagram, not repeat its title"
            )
            assert root.get("role") == "img", slug

    def test_each_scales(self, diagrams):
        """A viewBox is what lets the image scale; width/height alone pins it."""
        for slug in diagrams:
            assert ET.parse(OUT_DIR / f"{slug}.svg").getroot().get("viewBox"), slug

    def test_each_has_an_opaque_background(self, diagrams):
        """GitHub renders a README on white *or* near-black. Dark strokes on a
        transparent background vanish on the dark one."""
        for slug in diagrams:
            head = (OUT_DIR / f"{slug}.svg").read_text()
            assert 'fill="#f8fafc"' in head, f"{slug} has no card background"

    def test_text_stays_selectable(self, diagrams):
        """Live <text>, not paths: the reader can select it and it reflows
        with their font rather than being a picture of words."""
        for slug in diagrams:
            root = ET.parse(OUT_DIR / f"{slug}.svg").getroot()
            assert list(root.iter(f"{SVG}text")), slug

    def test_fonts_are_a_portable_stack(self, diagrams):
        """A single family would resolve to whatever the reader happens to
        have. Krita renders previews with one real face; the shipped file must
        not inherit that."""
        for slug in diagrams:
            body = (OUT_DIR / f"{slug}.svg").read_text()
            assert "DejaVu Sans\"" not in body, f"{slug} names Krita's preview font"
            assert "sans-serif" in body, slug


class TestGeometry:
    """The bugs that reading the markup does not reveal."""

    def _shapes(self, diagram):
        return diagram.fragment()

    def test_nothing_overflows_its_canvas(self, diagrams):
        offenders = []
        for slug, diagram in diagrams.items():
            for m in re.finditer(
                r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"',
                self._shapes(diagram),
            ):
                x, y, w, h = map(float, m.groups())
                if x < 1 and y < 1:
                    continue  # the background card is the canvas
                if x + w > diagram.w or y + h > diagram.h or x < 0 or y < 0:
                    offenders.append(f"{slug}: rect at {x},{y} {w}x{h}")
        assert not offenders, offenders

    def test_no_arrow_is_shorter_than_its_own_head(self, diagrams):
        """A 10px gap cannot hold an arrow: the head overshoots the shaft and
        the connector reads as a stray caret."""
        offenders = []
        for slug, diagram in diagrams.items():
            for x1, y1, x2, y2 in re.findall(
                r'd="M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)"', self._shapes(diagram)
            ):
                span = max(abs(float(x2) - float(x1)), abs(float(y2) - float(y1)))
                if 0 < span < 14:
                    offenders.append(f"{slug}: {span:.0f}px shaft")
        assert not offenders, offenders

    def test_no_empty_text_elements(self, diagrams):
        for slug, diagram in diagrams.items():
            assert "></text>" not in self._shapes(diagram), slug

    def test_coordinates_are_not_float_noise(self, diagrams):
        """Dividing a width by three yields 210.66666666666666, which makes a
        one-pixel move unreadable in a diff."""
        for slug, diagram in diagrams.items():
            assert not re.search(r"\d\.\d{3,}", self._shapes(diagram)), slug


class TestTheDocsReferenceThem:
    def test_every_diagram_is_referenced_by_some_doc(self, diagrams):
        """An unreferenced picture is one nobody will notice going stale."""
        root = OUT_DIR.parents[1]
        prose = "\n".join(
            prose_of(p) for p in [root / "README.md", *(root / "docs").glob("*.md")]
        )
        unused = [s for s in diagrams if f"{s}.svg" not in prose]
        assert not unused, f"not referenced from any doc: {unused}"

    def test_every_reference_resolves(self):
        root = OUT_DIR.parents[1]
        broken = []
        for doc in [root / "README.md", *(root / "docs").glob("*.md")]:
            for ref in re.findall(r"!\[[^\]]*\]\(([^)]*diagrams/[^)]+)\)", prose_of(doc)):
                target = (root if ref.startswith("docs/") else root / "docs") / ref
                if not target.exists():
                    broken.append(f"{doc.name} -> {ref}")
        assert not broken, broken

    def test_each_reference_has_descriptive_alt_text(self):
        """Alt text that repeats the filename tells a screen reader nothing."""
        root = OUT_DIR.parents[1]
        thin = []
        for doc in [root / "README.md", *(root / "docs").glob("*.md")]:
            for alt in re.findall(
                r"!\[([^\]]*)\]\([^)]*diagrams/[^)]+\)", prose_of(doc)
            ):
                if len(alt) < 25 or alt.endswith(".svg"):
                    thin.append(f"{doc.name}: {alt!r}")
        assert not thin, thin
