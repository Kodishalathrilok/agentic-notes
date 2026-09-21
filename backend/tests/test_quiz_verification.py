"""The quiz users actually get must have its answer key checked — honestly.

Same bug class as the partial-view fixes elsewhere (critique coverage, judge
evidence, edit/rewrite): a partial view treated as complete.

- verify_quiz only ran inside run_agent(include_quiz=True), which the UI never
  sets; the on-demand /api/quiz path shipped generate_quiz's key unchecked.
- the verifier saw a 9000-char excerpt of notes the generator wrote from at
  12000, so it could "correct" a right answer using notes it could not see.
- a correction was applied on the verifier's word alone, and "no corrections"
  looked the same as "the verifier failed".

The fix: one shared excerpt, a verdict per question with a quote from the
notes that Python checks deterministically, and a report saying what was
actually checked. Tests fake agent.call_model the way test_structured_quiz does.
"""
import json
import re

import pytest
from fastapi.testclient import TestClient

import agent
import auth
import main


NOTES = (
    "# Cell biology\n\n"
    "- **Mitochondria** produce ATP through cellular respiration [3].\n"
    "- The nucleus stores the cell's genetic material as DNA [1].\n"
    "- Ribosomes assemble proteins from amino acids in the cytoplasm [2].\n"
    "- The cell membrane controls what enters and leaves the cell [4].\n"
    "- Chloroplasts carry out photosynthesis in plant cells [5].\n"
)

QUESTIONS = [
    ("What produces ATP?", ["Nucleus", "Mitochondria", "Ribosome", "Membrane"], "B"),
    ("Where is DNA stored?", ["Nucleus", "Ribosome", "Membrane", "Chloroplast"], "C"),  # WRONG: should be A
    ("What assembles proteins?", ["Membrane", "Nucleus", "Ribosomes", "Chloroplast"], "C"),
    ("What controls entry to the cell?", ["Cell membrane", "DNA", "Ribosome", "Stroma"], "A"),
    ("Where does photosynthesis happen?", ["Nucleus", "Membrane", "Ribosome", "Chloroplasts"], "D"),
]


def _gen_json(questions=QUESTIONS):
    return json.dumps({"questions": [
        {"question": q, "options": dict(zip("ABCD", opts)), "answer": a,
         "explanation": "because"}
        for q, opts, a in questions
    ]})


def _verdicts(items):
    return json.dumps({"verdicts": items})


def _all_ok(n=5, skip=()):
    return [{"q": i, "correct": True, "answer": QUESTIONS[i - 1][2],
             "evidence": ""} for i in range(1, n + 1) if i not in skip]


GOOD_FIX = {"q": 2, "correct": False, "answer": "A",
            "evidence": "The nucleus stores the cell's genetic material as DNA"}


def _fake(monkeypatch, verifier_reply, prompts=None):
    """Generator gets the quiz JSON; the verifier gets `verifier_reply`
    (a string, or an Exception to raise)."""
    def call_model(prompt, *a, **k):
        if prompts is not None:
            prompts.append(prompt)
        if "QUIZ VERIFIER" in prompt:
            if isinstance(verifier_reply, Exception):
                raise verifier_reply
            return verifier_reply
        return _gen_json()
    monkeypatch.setattr(agent, "call_model", call_model)


@pytest.fixture(autouse=True)
def _fresh_rate_limits(monkeypatch):
    monkeypatch.setattr(auth, "_hits", {})


def _answers(quiz):
    return re.findall(r"^Answer:\s*([A-D])", quiz, re.M)


def _post_quiz(notes=NOTES):
    return TestClient(main.app).post("/api/quiz", json={"notes": notes, "model": ""})


# (a) a correction backed by a real quote from the notes is applied -----------

def test_api_quiz_fixes_wrong_key_with_supported_correction(monkeypatch):
    _fake(monkeypatch, _verdicts(_all_ok(skip=(2,)) + [GOOD_FIX]))
    r = _post_quiz()
    assert r.status_code == 200
    body = r.json()
    assert _answers(body["quiz"]) == ["B", "A", "C", "A", "D"], \
        "the on-demand quiz must ship a checked answer key"
    rep = body["verification"]
    assert rep["checked"] is True
    assert rep["corrected"] == 1 and rep["rejected"] == 0
    assert rep["questions"] == 5 and rep["judged"] == 5
    assert rep["disputed"] == []


# (b) a correction whose quote is not in the notes is rejected ---------------

def test_fabricated_evidence_rejects_correction(monkeypatch):
    bad = dict(GOOD_FIX, evidence="DNA is stored inside the ribosome of every cell")
    _fake(monkeypatch, _verdicts(_all_ok(skip=(2,)) + [bad]))
    quiz = agent.generate_quiz(NOTES)
    out, rep = agent.verify_quiz_detailed(NOTES, quiz)
    assert _answers(out) == ["B", "C", "C", "A", "D"]  # original kept
    assert out == quiz
    assert rep["rejected"] == 1 and rep["corrected"] == 0
    assert rep["disputed"] == [2]
    assert rep["checked"] is True


# (c) normalisation: markdown bold + citation markers; short quotes fail -----

def test_evidence_matches_through_markdown_and_citations(monkeypatch):
    q_wrong = list(QUESTIONS)
    q_wrong[0] = (QUESTIONS[0][0], QUESTIONS[0][1], "A")  # wrong: should be B
    fix = {"q": 1, "correct": False, "answer": "B",
           "evidence": "Mitochondria produce ATP through cellular respiration."}
    monkeypatch.setattr(agent, "call_model", lambda p, *a, **k: _verdicts([fix]))
    quiz = agent._render_quiz(agent._valid_questions(json.loads(_gen_json(q_wrong)), 5))
    out, rep = agent.verify_quiz_detailed(NOTES, quiz)
    assert _answers(out)[0] == "B"
    assert rep["corrected"] == 1 and rep["rejected"] == 0


def test_too_short_evidence_is_rejected(monkeypatch):
    short = dict(GOOD_FIX, evidence="the nucleus")  # present in notes, but trivial
    _fake(monkeypatch, _verdicts([short]))
    quiz = agent.generate_quiz(NOTES)
    out, rep = agent.verify_quiz_detailed(NOTES, quiz)
    assert _answers(out)[1] == "C"
    assert rep["rejected"] == 1 and rep["disputed"] == [2]


# (d) verifier failure: quiz unchanged, checked=False, request still OK -------

@pytest.mark.parametrize("reply", [RuntimeError("provider down"), "not json",
                                   '{"verdicts": "nope"}', '{"verdicts": []}'])
def test_verifier_failure_is_reported_not_hidden(monkeypatch, reply):
    _fake(monkeypatch, reply)
    quiz = agent.generate_quiz(NOTES)
    out, rep = agent.verify_quiz_detailed(NOTES, quiz)
    assert out == quiz
    assert rep["checked"] is False and rep["judged"] == 0
    assert agent.verify_quiz(NOTES, quiz) == quiz


def test_api_quiz_fails_open_when_verifier_raises(monkeypatch):
    _fake(monkeypatch, RuntimeError("provider down"))
    r = _post_quiz()
    assert r.status_code == 200
    body = r.json()
    assert _answers(body["quiz"]) == ["B", "C", "C", "A", "D"]
    assert body["verification"]["checked"] is False


# (e) partial verdicts are reported as partial ------------------------------

def test_partial_verdicts_reported(monkeypatch):
    _fake(monkeypatch, _verdicts(_all_ok(n=3)))
    quiz = agent.generate_quiz(NOTES)
    out, rep = agent.verify_quiz_detailed(NOTES, quiz)
    assert out == quiz
    assert rep["checked"] is True
    assert rep["questions"] == 5 and rep["judged"] == 3


# (f) generator and verifier see exactly the same notes ---------------------

def _notes_block(prompt):
    m = re.search(r'NOTES:\n"""(.*?)"""', prompt, re.S)
    assert m, "prompt has no NOTES block"
    return m.group(1)


def test_generator_and_verifier_see_same_excerpt(monkeypatch):
    long_notes = "".join(f"- Filler fact number {i} about cells [1].\n" for i in range(900))
    long_notes += "- TAILMARKER the final section is here.\n"
    assert len(long_notes) > agent.QUIZ_NOTES_CHARS
    prompts = []
    _fake(monkeypatch, _verdicts(_all_ok()), prompts)
    r = _post_quiz(long_notes)
    assert r.status_code == 200
    gen = [p for p in prompts if "QUIZ VERIFIER" not in p]
    ver = [p for p in prompts if "QUIZ VERIFIER" in p]
    assert gen and ver
    assert "TAILMARKER" in _notes_block(gen[0])
    assert "TAILMARKER" in _notes_block(ver[0])
    assert _notes_block(ver[0]) == _notes_block(gen[0])


# (g) response shape ---------------------------------------------------------

def test_api_quiz_response_shape(monkeypatch):
    _fake(monkeypatch, _verdicts(_all_ok()))
    body = _post_quiz().json()
    assert set(body) >= {"quiz", "verification"}
    assert set(body["verification"]) >= {"checked", "questions", "judged",
                                         "corrected", "rejected", "disputed"}
    assert body["verification"]["corrected"] == 0
