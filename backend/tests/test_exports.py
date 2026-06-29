"""Export rendering tests (Markdown / PDF / DOCX) — all offline."""

from pdf_export import notes_to_markdown, notes_to_pdf, notes_to_docx

NOTES = "**Intro:**\n• point one\n• point two\n1. numbered item"


def test_markdown_joins_sections():
    md = notes_to_markdown(NOTES, "quiz text", "cards text")
    assert "AI Generated Notes" in md
    assert "---" in md  # section dividers
    assert "quiz text" in md


def test_pdf_returns_pdf_bytes():
    data = notes_to_pdf(NOTES, "Q1) ...", "CARD 1")
    assert isinstance(data, bytes)
    assert data[:4] == b"%PDF"  # valid PDF header


def test_docx_returns_zip_bytes():
    data = notes_to_docx(NOTES, "", "")
    assert isinstance(data, bytes)
    assert data[:2] == b"PK"  # .docx is a zip container
