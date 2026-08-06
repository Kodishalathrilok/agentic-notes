"""Pure-function tests: JSON recovery, flashcard parsing, prompt helpers."""

import main
from models import safe_json
from pdf_export import parse_flashcards, flashcards_to_csv
from agent import _max_tokens, _format_instructions, _instr_block


# ---------------------------------------------------------------------------
# YouTube transcript errors
#
# youtube-transcript-api >= 1.0 dropped the .get_transcript/.list_transcripts
# classmethods. The old fallback still called them, so every failure surfaced
# as `AttributeError: no attribute 'list_transcripts'` instead of the real
# reason (captions disabled, video unavailable, IP blocked).
# ---------------------------------------------------------------------------

def test_legacy_classmethods_are_gone():
    """Pins the assumption behind the rewrite — if this fails, revisit it."""
    from youtube_transcript_api import YouTubeTranscriptApi

    assert not hasattr(YouTubeTranscriptApi, "list_transcripts")
    assert hasattr(YouTubeTranscriptApi(), "fetch")


def test_failure_reason_is_specific_per_error():
    class TranscriptsDisabled(Exception):
        pass

    class VideoUnavailable(Exception):
        pass

    assert "captions turned off" in main._youtube_failure_reason(TranscriptsDisabled())
    assert "isn't available" in main._youtube_failure_reason(VideoUnavailable())


def test_failure_reason_falls_back_for_unknown_errors():
    reason = main._youtube_failure_reason(RuntimeError("boom"))
    assert "Couldn't read" in reason
    # Never echo the raw exception — the library's messages run to hundreds
    # of characters of GitHub-issue boilerplate.
    assert "boom" not in reason


def test_youtube_id_parsing():
    parse = main._youtube_id
    assert parse("https://www.youtube.com/watch?v=lG-tM_4iKOg") == "lG-tM_4iKOg"
    assert parse("https://youtu.be/lG-tM_4iKOg") == "lG-tM_4iKOg"
    assert parse("https://www.youtube.com/shorts/lG-tM_4iKOg") == "lG-tM_4iKOg"
    assert parse("https://example.com/watch?v=lG-tM_4iKOg") is None


def test_safe_json_plain():
    assert safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_code_fenced():
    assert safe_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_safe_json_embedded_in_prose():
    assert safe_json('Here you go: {"a": 1} thanks') == {"a": 1}


def test_safe_json_trailing_comma():
    assert safe_json('{"a": 1,}') == {"a": 1}


def test_safe_json_garbage_returns_empty():
    assert safe_json("not json at all") == {}
    assert safe_json("") == {}


def test_parse_flashcards_basic():
    raw = "CARD 1\nFront: Q1\nBack: A1\n\nCARD 2\nFront: Q2\nBack: A2"
    cards = parse_flashcards(raw)
    assert len(cards) == 2
    assert cards[0] == {"front": "Q1", "back": "A1"}
    assert cards[1]["back"] == "A2"


def test_parse_flashcards_multiline_back():
    raw = "CARD 1\nFront: Q\nBack: line one\nline two"
    cards = parse_flashcards(raw)
    assert cards[0]["back"] == "line one line two"


def test_flashcards_to_csv():
    csv_text = flashcards_to_csv("CARD 1\nFront: Q1\nBack: A1")
    assert "Q1" in csv_text and "A1" in csv_text


def test_max_tokens_scales_with_length():
    assert _max_tokens("short") < _max_tokens("medium") < _max_tokens("long")
    assert _max_tokens("unknown") == _max_tokens("medium")  # safe default


def test_format_instructions_varies_by_format():
    assert "bullet" in _format_instructions("bullet").lower()
    assert "1." in _format_instructions("numbered")
    assert "paragraph" in _format_instructions("paragraph").lower()


def test_instr_block():
    assert _instr_block("") == ""
    assert "ADDITIONAL" in _instr_block("focus on formulas")
