"""
Tests for structured (JSON-mode) quiz and flashcard generation.

The model now returns JSON, Python validates it, and the plain-text wire
format is rendered deterministically — so the frontend parser can never
receive a malformed quiz. A legacy plain-text prompt remains as fallback.
"""

import json
import re

import agent


GOOD_QUIZ_JSON = json.dumps({
    "questions": [
        {
            "question": "What does chlorophyll absorb?",
            "options": {"A": "Light", "B": "Water", "C": "Soil", "D": "Oxygen"},
            "answer": "A",
            "explanation": "Chlorophyll absorbs light energy.",
        },
        {
            "question": "Where does the Calvin cycle occur?",
            "options": {"a": "Nucleus", "b": "Stroma", "c": "Membrane", "d": "Cytosol"},
            "answer": "b",
            "explanation": "It occurs in the stroma.",
        },
    ]
})

GOOD_CARDS_JSON = json.dumps({
    "cards": [
        {"front": "Chlorophyll", "back": "Pigment that absorbs light"},
        {"front": "Stroma", "back": "Fluid where the Calvin cycle runs"},
    ]
})


def _frontend_parse_quiz(raw):
    """Mirror of QuizPanel.jsx's parser — proves our output is parseable."""
    questions, current = [], None
    for raw_line in raw.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        q = re.match(r"^Q?\s*\d+[).:]\s*(.+)$", line, re.I)
        opt = re.match(r"^([A-D])[).:]\s*(.+)$", line)
        ans = re.match(r"^Answer\s*:?\s*([A-D])", line, re.I)
        exp = re.match(r"^Explanation\s*:?\s*(.+)$", line, re.I)
        if q and not opt:
            if current:
                questions.append(current)
            current = {"question": q.group(1), "options": {}, "answer": "", "explanation": ""}
        elif opt and current:
            current["options"][opt.group(1)] = opt.group(2)
        elif ans and current:
            current["answer"] = ans.group(1)
        elif exp and current:
            current["explanation"] = exp.group(1)
    if current:
        questions.append(current)
    return questions


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

def test_quiz_json_rendered_to_frontend_format(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: GOOD_QUIZ_JSON)
    out = agent.generate_quiz("notes", n=5)
    parsed = _frontend_parse_quiz(out)
    assert len(parsed) == 2
    assert parsed[0]["answer"] == "A"
    assert parsed[1]["answer"] == "B"  # lowercase letters normalized
    assert set(parsed[0]["options"]) == {"A", "B", "C", "D"}


def test_quiz_falls_back_to_text_on_bad_json(monkeypatch):
    calls = []

    def fake(prompt, **kwargs):
        calls.append(kwargs.get("json_mode", False))
        if kwargs.get("json_mode"):
            return "utter nonsense, not json"
        return "Q1) fallback?\nA) a\nB) b\nC) c\nD) d\nAnswer: A"

    monkeypatch.setattr(agent, "call_model", fake)
    out = agent.generate_quiz("notes")
    assert "fallback?" in out
    assert calls == [True, False]  # tried JSON first, then legacy prompt


def test_quiz_rejects_incomplete_questions(monkeypatch):
    bad = json.dumps({"questions": [
        {"question": "Missing option D", "options": {"A": "x", "B": "y", "C": "z"}, "answer": "A"},
        {"question": "Bad answer letter", "options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": "E"},
    ]})
    seen = []

    def fake(prompt, **kwargs):
        seen.append(1)
        return bad if kwargs.get("json_mode") else "Q1) legacy\nA) a\nB) b\nC) c\nD) d\nAnswer: B"

    monkeypatch.setattr(agent, "call_model", fake)
    out = agent.generate_quiz("notes")
    assert "legacy" in out  # every JSON question invalid -> fallback


def test_quiz_newlines_inside_fields_flattened(monkeypatch):
    tricky = json.dumps({"questions": [{
        "question": "Multi\nline\nquestion",
        "options": {"A": "opt\nA", "B": "b", "C": "c", "D": "d"},
        "answer": "A",
        "explanation": "exp\nlanation",
    }]})
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: tricky)
    out = agent.generate_quiz("notes")
    parsed = _frontend_parse_quiz(out)
    assert len(parsed) == 1
    assert parsed[0]["question"] == "Multi line question"


# ---------------------------------------------------------------------------
# Quiz verification (per-question verdicts; evidence-checked corrections)
# ---------------------------------------------------------------------------

QUIZ_TEXT = (
    "Q1) What absorbs light?\nA) Chlorophyll\nB) Water\nC) Soil\nD) Roots\n"
    "Answer: B\nExplanation: wrong on purpose\n\n"
    "Q2) Where is the stroma?\nA) Nucleus\nB) Chloroplast\nC) Wall\nD) Vacuole\n"
    "Answer: B\nExplanation: correct"
)


VERIFY_NOTES = "- Chlorophyll absorbs light energy for photosynthesis [1].\n- The stroma is inside the chloroplast [2].\n"


def test_verify_quiz_applies_corrections(monkeypatch):
    corr = json.dumps({"verdicts": [
        {"q": 1, "correct": False, "answer": "A",
         "evidence": "Chlorophyll absorbs light energy for photosynthesis",
         "explanation": "Chlorophyll absorbs light."},
        {"q": 2, "correct": True, "answer": "B", "evidence": ""},
    ]})
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: corr)
    out = agent.verify_quiz(VERIFY_NOTES, QUIZ_TEXT)
    parsed = _frontend_parse_quiz(out)
    assert parsed[0]["answer"] == "A"          # corrected
    assert parsed[1]["answer"] == "B"          # untouched
    assert "Chlorophyll absorbs light." in out


def test_verify_quiz_no_corrections_returns_original(monkeypatch):
    ok = json.dumps({"verdicts": [{"q": 1, "correct": True}, {"q": 2, "correct": True}]})
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: ok)
    assert agent.verify_quiz(VERIFY_NOTES, QUIZ_TEXT) == QUIZ_TEXT


def test_verify_quiz_garbage_returns_original(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: "not json at all")
    assert agent.verify_quiz(VERIFY_NOTES, QUIZ_TEXT) == QUIZ_TEXT


def test_verify_quiz_ignores_invalid_corrections(monkeypatch):
    ev = "Chlorophyll absorbs light energy for photosynthesis"
    corr = json.dumps({"verdicts": [
        {"q": 99, "correct": False, "answer": "A", "evidence": ev},   # out of range
        {"q": "x", "correct": False, "answer": "A", "evidence": ev},  # non-numeric
        {"q": 2, "correct": False, "answer": "Z", "evidence": ev},    # bad letter
    ]})
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: corr)
    out = agent.verify_quiz(VERIFY_NOTES, QUIZ_TEXT)
    parsed = _frontend_parse_quiz(out)
    assert parsed[0]["answer"] == "B" and parsed[1]["answer"] == "B"


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

def test_flashcards_json_rendered(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: GOOD_CARDS_JSON)
    out = agent.generate_flashcards("notes")
    assert "CARD 1\nFront: Chlorophyll\nBack: Pigment that absorbs light" in out
    assert "CARD 2" in out


def test_flashcards_fallback_on_bad_json(monkeypatch):
    def fake(prompt, **kwargs):
        if kwargs.get("json_mode"):
            return "[]"  # valid JSON but wrong shape
        return "CARD 1\nFront: legacy\nBack: fallback"

    monkeypatch.setattr(agent, "call_model", fake)
    out = agent.generate_flashcards("notes")
    assert "legacy" in out


def test_flashcards_skip_incomplete_cards(monkeypatch):
    data = json.dumps({"cards": [
        {"front": "ok", "back": "fine"},
        {"front": "", "back": "no front"},
        {"front": "no back"},
    ]})
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: data)
    out = agent.generate_flashcards("notes")
    assert "CARD 1" in out and "CARD 2" not in out