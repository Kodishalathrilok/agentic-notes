"""
Pipeline orchestration test with a MOCKED model — no API key or network.

Verifies run_agent emits the right event sequence and that streamed note
deltas reassemble into the final notes.
"""

import threading

import agent


def _fake_call_model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
                     on_serve=None):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    if "PLANNING agent" in prompt:
        return '{"outline":["A","B"],"checklist":["x"],"difficulty":"easy","suggested_format":"bullet"}'
    if "CRITIQUE agent" in prompt:
        # high score, no revision needed -> deterministic (no loop)
        return (
            '{"score":9,"needs_revision":false,"unsupported_claims":[],'
            '"missing_topics":[],"issues":[],"strengths":["clear"]}'
        )
    if "QUIZ VERIFIER" in prompt or "QUIZ agent" in prompt:
        return "Q1) Test?\nA) a\nB) b\nC) c\nD) d\nAnswer: A\nExplanation: because reasons."
    if "FLASHCARD agent" in prompt:
        return "CARD 1\nFront: f\nBack: b"
    if "title" in prompt.lower():
        return "Test Title"
    return "stub"


def _fake_stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    yield "Some "
    yield "notes."


def test_pipeline_event_sequence(monkeypatch):
    monkeypatch.setattr(agent, "call_model", _fake_call_model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(agent.run_agent("source text", "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]

    for expected in ("plan_done", "notes_done", "critique_done", "quiz_done", "flashcards_done"):
        assert expected in types, f"missing {expected}"
    assert types[-1] == "done"
    assert "error" not in types

    deltas = [e["content"] for e in events if e["type"] == "notes_delta"]
    assert "".join(deltas) == "Some notes."


def test_long_document_uses_sectioned_writing(monkeypatch):
    """Docs over SECTION_DOC_THRESHOLD are written section-by-section, each with
    its own retrieval — so a large source's later parts aren't left out."""

    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "PLANNING agent" in prompt:
            return (
                '{"outline":["Alpha","Beta","Gamma"],"checklist":["x"],'
                '"difficulty":"easy","suggested_format":"bullet"}'
            )
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    streams = {"n": 0}
    lock = threading.Lock()

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        with lock:  # sections are now written concurrently
            streams["n"] += 1
            n = streams["n"]
        yield f"body-{n} "

    monkeypatch.setattr(agent, "call_model", model)
    monkeypatch.setattr(agent, "call_model_stream", stream)

    # long, multi-topic source (> SECTION_DOC_THRESHOLD chars)
    long_text = (
        "Photosynthesis converts light into chemical energy in chloroplasts. "
        "Binary search halves a sorted interval, running in O(log n). "
        "The French Revolution began in 1789 and reshaped Europe. "
    ) * 300
    assert len(long_text) > agent.SECTION_DOC_THRESHOLD

    events = list(agent.run_agent(long_text, "exam", "academic", "long", "bullet"))
    types = [e["type"] for e in events]

    assert "error" not in types
    assert types.count("sources") == 1  # one unified sources event
    notes = next(e["content"] for e in events if e["type"] == "notes_done")

    # Sections are now DOCUMENT WINDOWS, not outline topics: the writer covers
    # the whole source by construction instead of by retrieval luck.
    sources = next(e["data"] for e in events if e["type"] == "sources")
    windows = [s for s in (e["content"] for e in events if e["type"] == "status")
               if s.startswith("Writing section ")]
    assert windows, "expected per-window progress messages"
    assert streams["n"] == len(windows), "one write stream per window"
    # Every window's body reaches the notes. (The mechanical page-range
    # heading was dropped: the writer supplies its own topical headings,
    # so "**Pages 1-3:**" above "**Pipeline Overview:**" was pure noise.)
    for n in range(1, streams["n"] + 1):
        assert f"body-{n}" in notes, f"window {n} missing from the notes"

    # Coverage is the point: every chunk of the document reaches the writer.
    assert len(sources) >= len(windows)
    statuses = [e["content"] for e in events if e["type"] == "status"]
    assert any("Covering the whole document" in s for s in statuses)


def test_digest_document_scans_every_segment(monkeypatch):
    """The digest must read 100% of the document, one segment at a time."""
    seen = []

    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        seen.append(prompt)
        return "- topic A\n- topic B"

    monkeypatch.setattr(agent, "call_model", model)

    text = "q" * (agent.DIGEST_SEGMENT_CHARS * 3 + 100)  # 4 segments
    inventory = digest = agent.digest_document(text)

    assert len(seen) == 4
    # every character of the doc was included in some prompt
    total_scanned = sum(p.count("q") for p in seen)
    assert total_scanned == len(text)
    assert "topic A" in inventory and digest.count("topic B") == 4


def test_big_docs_plan_from_full_coverage_inventory(monkeypatch):
    """Docs over DIGEST_DOC_THRESHOLD: helper scans the whole doc and the
    planner prompt must contain the resulting TOPIC INVENTORY. Small docs
    must NOT trigger the scan."""
    captured = {"plan_prompt": "", "digest_calls": 0}

    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "topic inventory" in prompt and "scanning part" in prompt:
            captured["digest_calls"] += 1
            return "- rare topic on page 93"
        if "PLANNING agent" in prompt:
            captured["plan_prompt"] = prompt
            return (
                '{"outline":["A","B","C"],"checklist":["x"],'
                '"difficulty":"easy","suggested_format":"bullet"}'
            )
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    monkeypatch.setattr(agent, "call_model", model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    big = "Physics history biology economics content. " * 2000  # > 60k chars
    assert len(big) > agent.DIGEST_DOC_THRESHOLD
    events = list(agent.run_agent(big, "exam", "academic", "medium", "bullet"))
    assert "error" not in [e["type"] for e in events]
    assert captured["digest_calls"] >= 2  # multiple segments scanned
    assert "TOPIC INVENTORY" in captured["plan_prompt"]
    assert "rare topic on page 93" in captured["plan_prompt"]

    # a small doc must skip the scan entirely
    captured["digest_calls"] = 0
    captured["plan_prompt"] = ""
    list(agent.run_agent("short source text", "exam", "academic", "medium", "bullet"))
    assert captured["digest_calls"] == 0
    assert "TOPIC INVENTORY" not in captured["plan_prompt"]


def test_pipeline_blocks_non_academic(monkeypatch):
    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "gatekeeper" in prompt.lower():
            return '{"academic": false, "subject": "n/a", "reason": "celebrity topic"}'
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    monkeypatch.setattr(agent, "call_model", model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(agent.run_agent("Tom Cruise", "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]
    assert "blocked" in types
    assert "plan_done" not in types  # stopped before generating


def test_pipeline_triggers_revision_when_needed(monkeypatch):
    calls = {"critique": 0}

    def model_low_then_high(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "CRITIQUE agent" in prompt:
            calls["critique"] += 1
            needs = "true" if calls["critique"] == 1 else "false"
            score = 5 if calls["critique"] == 1 else 9
            return (
                f'{{"score":{score},"needs_revision":{needs},'
                f'"unsupported_claims":[],"missing_topics":[],"issues":[],"strengths":[]}}'
            )
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    monkeypatch.setattr(agent, "call_model", model_low_then_high)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(agent.run_agent("source text", "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]

    # A low first critique should trigger a revise pass and a re-critique.
    assert "revise_start" in types
    assert types.count("critique_done") >= 2
    assert "error" not in types


# ---------------------------------------------------------------------------
# New behaviors: citation enforcement, corrective re-retrieval, advisory
# missing_topics
# ---------------------------------------------------------------------------


def test_enforce_citations_strips_invented_ids():
    notes = "Point one [1]. Point two [2][9]. Link [text](http://x) stays. [77]\nEnd [3]"
    cleaned = agent.enforce_citations(notes, valid_ids={1, 3})
    assert "[1]" in cleaned and "[3]" in cleaned
    assert "[2]" not in cleaned and "[9]" not in cleaned and "[77]" not in cleaned
    assert "[text](http://x)" in cleaned  # markdown links untouched


def test_final_notes_have_invalid_citations_stripped(monkeypatch):
    """A hallucinated [99] in the written notes must not survive the pipeline."""

    def stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None):
        yield "A real point [1]. "
        yield "A fabricated citation [99]."

    monkeypatch.setattr(agent, "call_model", _fake_call_model)
    monkeypatch.setattr(agent, "call_model_stream", stream)

    events = list(agent.run_agent("source text " * 50, "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]
    assert "error" not in types

    # The cleanup is surfaced to the client as a notes_revised event.
    final = [e["content"] for e in events if e["type"] in ("notes_done", "notes_revised")][-1]
    assert "[99]" not in final
    assert "[1]" in final


def test_missing_topics_alone_do_not_trigger_revision(monkeypatch):
    """missing_topics is advisory: with a high score, no unsupported claims and
    needs_revision=false, the pipeline must NOT revise."""

    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "CRITIQUE agent" in prompt:
            return (
                '{"score":9,"needs_revision":false,"unsupported_claims":[],'
                '"missing_topics":["some minor aside"],"issues":[],"strengths":[]}'
            )
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    monkeypatch.setattr(agent, "call_model", model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(agent.run_agent("source text", "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]
    assert "revise_start" not in types
    assert "error" not in types


def test_corrective_retrieval_adds_context_for_missing_topics(monkeypatch):
    """When a revision IS needed and missing_topics is non-empty, the pipeline
    must query the retriever per topic and re-emit the merged sources."""
    calls = {"critique": 0}
    retrieval_queries = []

    def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
        if "CRITIQUE agent" in prompt:
            calls["critique"] += 1
            if calls["critique"] == 1:
                return (
                    '{"score":5,"needs_revision":true,"unsupported_claims":[],'
                    '"missing_topics":["photosynthesis details"],"issues":[],"strengths":[]}'
                )
            return (
                '{"score":9,"needs_revision":false,"unsupported_claims":[],'
                '"missing_topics":[],"issues":[],"strengths":[]}'
            )
        return _fake_call_model(prompt, max_tokens, model, temperature, json_mode)

    real_retrieve = agent.Retriever.retrieve

    def spy_retrieve(self, query, k=None):
        retrieval_queries.append(query)
        return real_retrieve(self, query, k=k)

    monkeypatch.setattr(agent, "call_model", model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)
    monkeypatch.setattr(agent.Retriever, "retrieve", spy_retrieve)

    # enough distinct text that retrieval produces multiple chunks
    src = (
        "Photosynthesis converts light into chemical energy in chloroplasts. "
        "Binary search halves a sorted interval each step. "
        "The French Revolution began in 1789. "
    ) * 40

    events = list(agent.run_agent(src, "exam", "academic", "medium", "bullet"))
    types = [e["type"] for e in events]

    assert "error" not in types
    assert "revise_start" in types
    # the missing topic itself was used as a retrieval query
    assert any("photosynthesis details" in q for q in retrieval_queries)

def test_quiz_and_flashcards_skipped_when_flags_off(monkeypatch):
    """On-demand generation: include_quiz/include_flashcards=False must skip
    those stages entirely (no quiz_done / flashcards_done events) while the
    rest of the pipeline runs to completion."""
    monkeypatch.setattr(agent, "call_model", _fake_call_model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    events = list(
        agent.run_agent(
            "source text", "exam", "academic", "medium", "bullet",
            include_quiz=False, include_flashcards=False,
        )
    )
    types = [e["type"] for e in events]

    assert "quiz_done" not in types
    assert "flashcards_done" not in types
    for expected in ("plan_done", "notes_done", "critique_done"):
        assert expected in types, f"missing {expected}"
    assert types[-1] == "done"


def test_single_pass_long_retrieves_more_context(monkeypatch):
    """A small doc taking the single-pass path must retrieve a larger slice for
    length='long' than for 'short', so the writer has enough source material to
    reach the long word target."""
    monkeypatch.setattr(agent, "call_model", _fake_call_model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)

    captured = {}

    class SpyRetriever:
        def __init__(self, text, spans=None):
            self.texts = ["chunk"] * 30
            self.chunks_meta = []
        def sample(self, n=16000):
            return "sample text"
        def retrieve(self, query, k=None):
            captured["k"] = k
            return [
                {"chunk_id": i, "text": f"c{i}", "start_offset": 0, "end_offset": 1}
                for i in range(k or 8)
            ]

    monkeypatch.setattr(agent, "Retriever", SpyRetriever)

    # Short source (below SECTION_DOC_THRESHOLD) -> single-pass path
    short_doc = "Immunology basics. " * 20

    list(agent.run_agent(short_doc, "exam", "academic", "short", "bullet",
                         include_quiz=False, include_flashcards=False))
    k_short = captured["k"]

    list(agent.run_agent(short_doc, "exam", "academic", "long", "bullet",
                         include_quiz=False, include_flashcards=False))
    k_long = captured["k"]

    assert k_long > k_short, f"long ({k_long}) should retrieve more than short ({k_short})"

# ---------------------------------------------------------------------------
# Document intent: a task document must not be written up as if it taught
# ---------------------------------------------------------------------------

def test_task_doc_types_map_to_the_task_writing_rule():
    assert agent.is_task_document("question_bank")
    assert agent.is_task_document("assignment")
    assert agent.is_task_document("exam")
    assert agent.is_task_document("worksheet")
    assert not agent.is_task_document("explanatory")
    assert not agent.is_task_document("mixed")


def test_explanatory_prompt_is_unchanged_by_the_doc_type_feature():
    """Ordinary lecture/textbook PDFs must behave exactly as before."""
    plan = {"outline": ["A"], "checklist": ["x"]}
    base = agent._write_prompt("[1] ctx", "exam", "academic", "medium", "bullet", plan)
    typed = agent._write_prompt("[1] ctx", "exam", "academic", "medium", "bullet", plan,
                                doc_type="explanatory")
    assert base == typed
    assert "list of TASKS" not in base


def test_task_prompt_forbids_turning_tasks_into_facts():
    plan = {"outline": ["Unit 1"], "checklist": []}
    prompt = agent._write_prompt("[1] Write a program to implement single inheritance.",
                                 "exam", "academic", "medium", "bullet", plan,
                                 doc_type="question_bank")
    assert "list of TASKS" in prompt
    assert "Topic covered" in prompt
    assert "did not say it" in prompt


def test_mixed_prompt_keeps_explanation_and_tasks_apart():
    plan = {"outline": ["A"], "checklist": []}
    prompt = agent._write_prompt("[1] ctx", "exam", "academic", "medium", "bullet", plan,
                                 doc_type="mixed")
    assert "MIXES explanation" in prompt


def test_unknown_doc_type_falls_back_to_explanatory(monkeypatch):
    """A classifier glitch must not silently downgrade a normal PDF."""
    monkeypatch.setattr(agent, "call_model",
                        lambda *a, **k: '{"academic": true, "doc_type": "nonsense-value"}')
    assert agent.classify_academic("text")["doc_type"] == "explanatory"


def test_classifier_parse_failure_defaults_to_explanatory(monkeypatch):
    monkeypatch.setattr(agent, "call_model", lambda *a, **k: "not json at all")
    gate = agent.classify_academic("text")
    assert gate["academic"] is True and gate["doc_type"] == "explanatory"


def test_done_event_reports_the_serving_provider(monkeypatch):
    """The done event must name the provider that actually generated the notes."""
    monkeypatch.setattr(agent, "call_model", _fake_call_model)
    monkeypatch.setattr(agent, "call_model_stream", _fake_stream)
    done = [e for e in agent.run_agent("Photosynthesis converts light. " * 40,
                                       "exam", "academic", "short", "bullet",
                                       include_quiz=False, include_flashcards=False)
            if e["type"] == "done"]
    assert done, "pipeline must emit a done event"
    data = done[0]["data"] or {}
    assert data.get("provider") == "nvidia"
    assert data.get("fallback_used") is False
    assert "doc_type" in data
    # No secret-bearing fields.
    assert not any("key" in k.lower() or "token" in k.lower() for k in data)
