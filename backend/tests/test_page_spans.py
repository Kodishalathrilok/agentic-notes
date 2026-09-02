"""
Page provenance: mapping a chunk back to the PDF page it came from.

The whole mechanism rests on one invariant — page spans and chunk offsets
must be measured against the SAME string. Chunks index the whitespace-
normalized document; a page span computed on the raw extraction would drift
by every run of whitespace the normalizer collapsed, and the failure is
silent: citations point at plausible but wrong pages.
"""

from retriever import chunk_document, normalize, page_for_offset, page_spans


def _joined(pages):
    """Exactly how /api/extract-pdf assembles pages into one document."""
    return "\n\n".join(p.strip() for p in pages if p.strip()).strip()


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def test_spans_line_up_with_the_normalized_document():
    pages = ["First page text.", "Second page text.", "Third page text."]
    norm = normalize(_joined(pages))
    for span in page_spans(pages):
        assert norm[span["start"]:span["end"]] == normalize(pages[span["page"] - 1])


def test_spans_survive_messy_whitespace():
    """Real PDF extraction is full of ragged newlines and runs of spaces."""
    pages = [
        "Line one\n\n\n   Line   two\t\tend of page one",
        "\n\n  Page two starts here \n  and wraps  \n",
        "Third\n\n\n\n\n\npage",
    ]
    norm = normalize(_joined(pages))
    for span in page_spans(pages):
        assert norm[span["start"]:span["end"]] == normalize(pages[span["page"] - 1])


def test_blank_pages_do_not_shift_later_pages():
    """A scanned/blank page has no text layer; the extractor drops it."""
    pages = ["Alpha content", "   \n\n  ", "Beta content", "", "Gamma content"]
    norm = normalize(_joined(pages))
    spans = page_spans(pages)

    assert spans[1]["start"] == spans[1]["end"]  # blank -> zero width
    assert spans[3]["start"] == spans[3]["end"]
    for span in spans:
        body = normalize(pages[span["page"] - 1])
        if body:
            assert norm[span["start"]:span["end"]] == body


# ---------------------------------------------------------------------------
# Offset -> page
# ---------------------------------------------------------------------------

def test_offset_resolves_to_the_right_page():
    pages = ["Alpha " * 40, "Beta " * 40, "Gamma " * 40]
    spans = page_spans(pages)
    for span in spans:
        if span["end"] > span["start"]:
            assert page_for_offset(spans, span["start"]) == span["page"]
            assert page_for_offset(spans, span["end"] - 1) == span["page"]


def test_offset_in_the_join_between_pages_resolves_to_the_earlier_page():
    pages = ["Alpha content here", "Beta content here"]
    spans = page_spans(pages)
    assert page_for_offset(spans, spans[0]["end"]) == 1


def test_no_spans_means_no_page():
    """Pasted text, URLs and transcripts have no pages — cite without one."""
    assert page_for_offset([], 0) is None
    assert page_for_offset(None, 123) is None


# ---------------------------------------------------------------------------
# End to end: a real chunk lands on the page its text actually came from
# ---------------------------------------------------------------------------

def test_chunks_map_back_to_the_page_containing_their_text():
    pages = [
        "Mitochondria are the powerhouse of the cell. " * 20,
        "Photosynthesis converts light into chemical energy. " * 20,
        "Osmosis is the movement of water across a membrane. " * 20,
    ]
    document = _joined(pages)
    spans = page_spans(pages)

    markers = {1: "Mitochondria", 2: "Photosynthesis", 3: "Osmosis"}
    checked = 0
    for chunk in chunk_document(document):
        page = page_for_offset(spans, chunk["start_offset"])
        # A chunk can straddle a boundary; only assert on ones wholly inside
        # a single page, which is where a citation must not be wrong.
        span = next(s for s in spans if s["page"] == page)
        if chunk["end_offset"] <= span["end"]:
            assert markers[page] in chunk["text"]
            checked += 1
    assert checked > 0, "no single-page chunks were produced to verify"


# ---------------------------------------------------------------------------
# Page-bounded chunking: the fix for citations landing 1-3 pages early
# ---------------------------------------------------------------------------

def _slide_deck(n=30, words=40):
    """Short pages, like a slide deck — the shape that exposed the bug. A 700
    char chunk used to cover ~3 of these and got labelled with the first."""
    return [f"Slide {i} heading. " + " ".join(f"w{i}x{j}" for j in range(words))
            for i in range(1, n + 1)]


def test_no_chunk_ever_straddles_a_page_boundary():
    pages = _slide_deck()
    spans = page_spans(pages)
    for chunk in chunk_document(_joined(pages), spans=spans):
        first = page_for_offset(spans, chunk["start_offset"])
        last = page_for_offset(spans, chunk["end_offset"] - 1)
        assert first == last == chunk["page"]


def test_every_chunk_page_is_the_page_its_text_sits_in():
    """The regression that mattered: a chunk labelled with a page it only
    partly overlaps. Each chunk's text must appear inside its own page."""
    pages = _slide_deck()
    spans = page_spans(pages)
    for chunk in chunk_document(_joined(pages), spans=spans):
        page_text = normalize(pages[chunk["page"] - 1])
        assert chunk["text"] in page_text


def test_all_pages_are_represented():
    pages = _slide_deck()
    got = {c["page"] for c in chunk_document(_joined(pages), spans=page_spans(pages))}
    assert got == set(range(1, len(pages) + 1))


def test_long_page_splits_into_several_chunks_on_the_same_page():
    pages = ["Alpha " * 600, "Beta short page"]
    chunks = chunk_document(_joined(pages), spans=page_spans(pages))
    page_one = [c for c in chunks if c["page"] == 1]
    assert len(page_one) > 1, "a page longer than target_chars should split"
    assert all(c["page"] == 1 for c in page_one)


def test_blank_pages_are_skipped_without_shifting_pages():
    pages = ["Alpha content here", "   ", "Beta content here", "", "Gamma content here"]
    chunks = chunk_document(_joined(pages), spans=page_spans(pages))
    by_page = {c["page"]: c["text"] for c in chunks}
    assert "Alpha" in by_page[1] and "Beta" in by_page[3] and "Gamma" in by_page[5]
    assert 2 not in by_page and 4 not in by_page


def test_without_spans_behaviour_is_unchanged():
    """Pasted text / URLs / transcripts have no pages and must not regress."""
    doc = _joined(_slide_deck())
    plain = chunk_document(doc)
    assert all("page" not in c for c in plain)
    assert [c["text"] for c in plain] == [c["text"] for c in chunk_document(doc, spans=None)]


def test_stale_spans_are_ignored_rather_than_mislabelling():
    """If the user edits the extracted text, the spans no longer describe it.
    Citing pages from stale spans would be confidently wrong, so the chunker
    falls back to unpaged chunks and the UI cites passage numbers instead."""
    pages = _slide_deck(n=5)
    spans = page_spans(pages)
    edited = _joined(pages) + " an extra sentence the user typed in later"
    chunks = chunk_document(edited, spans=spans)
    assert all("page" not in c for c in chunks)

