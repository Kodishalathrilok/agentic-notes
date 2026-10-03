"""Render a fixture's pages.txt into the source.pdf the eval reads.

pages.txt is the editable source: pages are separated by lines reading
"=== PAGE ===", and the first line of each page is set as its heading. Each
page becomes exactly one PDF page; the build fails if any page would overflow
onto a second one, because that would silently change the page numbers the
labels and citations are checked against.

    python evals/fixtures/build_pdf.py evals/fixtures/cell_biology

The PDF is committed, so the eval never depends on reportlab's output being
byte-identical across versions; rebuild only after editing pages.txt.
"""

import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Frame, PageTemplate, BaseDocTemplate, Paragraph, PageBreak
from xml.sax.saxutils import escape

PAGE_MARKER = "=== PAGE ==="


def read_pages(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    return [p.strip() for p in raw.split(PAGE_MARKER) if p.strip()]


def build(fixture_dir):
    pages = read_pages(os.path.join(fixture_dir, "pages.txt"))
    out = os.path.join(fixture_dir, "source.pdf")

    heading = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12, leading=15,
                             spaceAfter=6)
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=12.2,
                          spaceAfter=5)

    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=16 * mm, bottomMargin=16 * mm,
                          title=os.path.basename(fixture_dir), author="agentic-notes eval",
                          invariant=1)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    rendered = {"pages": 0}

    def _count(canvas, _doc):
        rendered["pages"] += 1

    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_count)])

    story = []
    for i, page in enumerate(pages):
        lines = page.split("\n")
        story.append(Paragraph(escape(lines[0].strip()), heading))
        for para in "\n".join(lines[1:]).strip().split("\n\n"):
            para = " ".join(para.split())
            if para:
                story.append(Paragraph(escape(para), body))
        if i < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)

    if rendered["pages"] != len(pages):
        os.remove(out)
        raise SystemExit(f"{rendered['pages']} PDF pages for {len(pages)} source pages: "
                         f"a page overflowed. Shorten it or split it with {PAGE_MARKER!r}.")
    print(f"wrote {out}: {len(pages)} pages")


if __name__ == "__main__":
    build(sys.argv[1])
