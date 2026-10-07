"""Latency work: parallel grounding / digest, fast NVIDIA 503 failover,
topic-check / index-build overlap, and server-side stage timings.

Every concurrency test compares against the sequential path (limit 1) with
deliberately shuffled completion order, because "same output as before" is
the contract. No network: providers are faked at requests.post or _dispatch.
"""

import json
import logging
import random
import threading
import time

import pytest
import requests

import agent
import models
import retrieval.semantic as sem
from tests.test_cancellation_capacity import FakeProviders, _long_doc, MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class InFlight:
    """Counts concurrent calls and remembers the peak."""

    def __init__(self):
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc):
        with self._lock:
            self.now -= 1


CHUNKS = {7: {"text": "Lists support append and pop.", "page": 7},
          8: {"text": "Tuples are immutable sequences.", "page": 8}}


def _claims_notes(n):
    lines = ["## Section"]
    for i in range(1, n + 1):
        cite = " [7]" if i % 2 else ""
        lines.append(f"- claim number {i} states a specific fact about item{i}{cite}")
    return "\n".join(lines)


def _verdict_for(text):
    """Deterministic verdict from the claim text alone (order-independent)."""
    i = int(text.split("claim number ")[1].split()[0])
    if i % 5 == 0:
        return ("unsupported", "")
    if i % 7 == 0:
        return ("partial", f"- claim number {i} states a fact [7]")
    return ("supported", "")


def _fake_verify(flight, fail_on=None, jitter=True):
    def verify(items, chunk_map, model=None, uncited_evidence=None):
        with flight:
            if jitter:
                time.sleep(random.uniform(0, 0.03))  # shuffle completion order
            if fail_on is not None and any(f"claim number {fail_on} " in t for _, t, _ in items):
                raise RuntimeError("provider exploded")
            return {n: _verdict_for(text) for n, (_, text, _) in enumerate(items, 1)}
    return verify


# ---------------------------------------------------------------------------
# 1. Grounding batches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fail_on", [None, 9])
def test_parallel_grounding_matches_sequential(monkeypatch, fail_on):
    notes = _claims_notes(40)  # 40 claims -> 7 batches of 6

    def run(limit):
        monkeypatch.setattr(agent, "GROUNDING_CONCURRENCY", limit)
        flight = InFlight()
        monkeypatch.setattr(agent, "_verify_batch", _fake_verify(flight, fail_on))
        return agent.verify_claim_support(notes, CHUNKS), flight.peak

    (seq_notes, seq_stats), seq_peak = run(1)
    (par_notes, par_stats), par_peak = run(4)

    assert par_notes == seq_notes
    assert par_stats == seq_stats
    assert seq_peak == 1
    assert 1 < par_peak <= 4, "batches must overlap but never exceed the limit"
    if fail_on is not None:
        # The failed batch is kept as written (fail-open), in both modes.
        assert "claim number 9 " in par_notes and "claim number 10 " in par_notes
        assert par_stats["unjudged"] == 6


def test_grounding_in_flight_is_bounded(monkeypatch):
    monkeypatch.setattr(agent, "GROUNDING_CONCURRENCY", 3)
    flight = InFlight()

    def slow(items, chunk_map, model=None, uncited_evidence=None):
        with flight:
            time.sleep(0.05)
            return {}
    monkeypatch.setattr(agent, "_verify_batch", slow)
    agent.verify_claim_support(_claims_notes(60), CHUNKS)  # 10 batches
    assert flight.peak == 3


# ---------------------------------------------------------------------------
# 2. Document digest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failing_part", [None, 3])
def test_parallel_digest_matches_sequential(monkeypatch, failing_part):
    text = "".join(chr(ord("a") + i) * agent.DIGEST_SEGMENT_CHARS for i in range(6))

    def run(limit):
        monkeypatch.setattr(agent, "DIGEST_CONCURRENCY", limit)
        flight = InFlight()

        def model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False):
            part = int(prompt.split("scanning part ")[1].split()[0])
            with flight:
                time.sleep(random.uniform(0, 0.03))
                if part == failing_part:
                    raise RuntimeError("segment failed")
                return f"- topic of part {part}"
        monkeypatch.setattr(agent, "call_model", model)
        return agent.digest_document(text), flight.peak

    seq, seq_peak = run(1)
    par, par_peak = run(3)
    assert par == seq
    expected = [p for p in range(1, 7) if p != failing_part]
    assert par == "\n".join(f"- topic of part {p}" for p in expected)
    assert seq_peak == 1 and 1 < par_peak <= 3


# ---------------------------------------------------------------------------
# 3. NVIDIA 503: one retry, fast failover, cooldown
# ---------------------------------------------------------------------------

class JsonResponse:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return {"choices": [{"message": {"content": "nvidia answer"}, "finish_reason": "stop"}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def close(self):
        pass


@pytest.fixture
def router(monkeypatch):
    """NVIDIA through the real _nvidia_post/_post_retrying; Gemini faked."""
    state = {"statuses": [], "headers": {}, "posts": 0, "sleeps": [], "gemini": 0}

    def post(**kwargs):
        state["posts"] += 1
        status = state["statuses"].pop(0) if state["statuses"] else 200
        return JsonResponse(status, state["headers"] if status == 503 else None)

    def gemini(*a, **k):
        state["gemini"] += 1
        return "gemini answer"

    monkeypatch.setattr(models.requests, "post", post)
    monkeypatch.setattr(models.time, "sleep", lambda s: state["sleeps"].append(s))
    monkeypatch.setattr(models, "_call_gemini", gemini)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p in ("nvidia", "gemini"))
    monkeypatch.setattr(models, "_nvidia_cooldown", {})
    monkeypatch.setattr(models, "RATE_LIMIT_RETRIES", 3)
    monkeypatch.setattr(models, "NVIDIA_503_RETRIES", 1)
    monkeypatch.setattr(models, "NVIDIA_503_RETRY_DELAY", 1.0)
    monkeypatch.setattr(models, "NVIDIA_503_COOLDOWN_S", 60.0)
    return state


def test_first_503_is_retried_once(router):
    router["statuses"] = [503, 200]
    assert models.call_model("p", model=MODEL) == "nvidia answer"
    assert router["posts"] == 2
    assert router["sleeps"] == [1.0]
    assert router["gemini"] == 0
    assert not models._cooling_down(MODEL)


def test_second_503_fails_over_immediately(router):
    router["statuses"] = [503, 503, 503, 503]
    assert models.call_model("p", model=MODEL) == "gemini answer"
    assert router["posts"] == 2, "no third NVIDIA attempt after the second 503"
    assert router["sleeps"] == [1.0]
    assert router["gemini"] == 1
    assert models._cooling_down(MODEL)


def test_503_respects_retry_after(router):
    router["statuses"] = [503, 200]
    router["headers"] = {"Retry-After": "2"}
    assert models.call_model("p", model=MODEL) == "nvidia answer"
    assert router["sleeps"] == [2.5]  # existing _retry_wait: header + 0.5


def test_cooldown_skips_nvidia_then_expires(router, monkeypatch):
    router["statuses"] = [503, 503]
    models.call_model("p", model=MODEL)
    posts = router["posts"]

    # Within the cooldown: straight to Gemini, zero NVIDIA requests.
    assert models.call_model("p", model=MODEL) == "gemini answer"
    assert router["posts"] == posts

    # Streams honour it too.
    seen = []

    def stream(prov, *a, **k):
        seen.append(prov)
        yield "x"
    monkeypatch.setattr(models, "_dispatch_stream", stream)
    assert "".join(models.call_model_stream("p", model=MODEL)) == "x"
    assert seen == ["gemini"]

    # A different NVIDIA model is not affected (cooldown is per model id).
    other = "nvidia/other-model"
    assert models.call_model("p", model=other) == "nvidia answer"

    # After it expires NVIDIA is tried again.
    real = time.monotonic
    monkeypatch.setattr(models.time, "monotonic", lambda: real() + 61)
    assert models.call_model("p", model=MODEL) == "nvidia answer"


def test_cooldown_never_strands_an_nvidia_only_deployment(router, monkeypatch):
    models._start_cooldown(MODEL)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    assert models.call_model("p", model=MODEL) == "nvidia answer"


def test_429_handling_is_unchanged(router):
    router["statuses"] = [429, 429, 429, 429]
    assert models.call_model("p", model=MODEL) == "gemini answer"
    assert router["posts"] == 4  # RATE_LIMIT_RETRIES + 1, as before
    assert router["sleeps"] == [2.0, 4.0, 8.0]
    assert not models._cooling_down(MODEL), "a 429 must not start the 503 cooldown"


def test_gemini_503_keeps_full_retries(monkeypatch):
    posts, sleeps = [], []
    monkeypatch.setattr(models.requests, "post",
                        lambda **k: posts.append(1) or JsonResponse(503))
    monkeypatch.setattr(models.time, "sleep", sleeps.append)
    monkeypatch.setattr(models, "RATE_LIMIT_RETRIES", 3)
    assert models._post_retrying("gemini x", url="http://x").status_code == 503
    assert len(posts) == 4 and sleeps == [2.0, 4.0, 8.0]


# ---------------------------------------------------------------------------
# 4-5. Pipeline: overlap, cancellation, timing logs
# ---------------------------------------------------------------------------

@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(models, "_nvidia_available", lambda: True)
    monkeypatch.setenv("NVIDIA_MODELS", MODEL)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    fp = FakeProviders()
    monkeypatch.setattr(models, "_dispatch", fp.dispatch)
    monkeypatch.setattr(models, "_dispatch_stream", fp.dispatch_stream)
    yield fp
    fp.release_all()


SMALL = ("Photosynthesis converts light energy into chemical energy in plants. "
         "Chlorophyll absorbs mostly blue and red light. ") * 20


def _run(text=SMALL, **kw):
    return list(agent.run_agent(text, "exam", "academic", "medium", "bullet", model=MODEL,
                                include_quiz=False, include_flashcards=False, **kw))


def test_index_build_overlaps_the_topic_check(fake, monkeypatch):
    started = threading.Event()
    real = agent.Retriever

    def retriever(*a, **k):
        started.set()
        return real(*a, **k)

    def gate(text, model=None):
        # Only returns once the index build has begun: proves they overlap.
        assert started.wait(5), "index build did not start during the topic check"
        return {"academic": True, "doc_type": "explanatory"}

    monkeypatch.setattr(agent, "Retriever", retriever)
    monkeypatch.setattr(agent, "classify_academic", gate)
    events = _run()
    assert events[-1]["type"] == "done"


def test_blocked_document_does_not_wait_for_the_index(fake, monkeypatch):
    release = threading.Event()

    def slow_retriever(*a, **k):
        release.wait(5)
        raise AssertionError("never used")

    monkeypatch.setattr(agent, "Retriever", slow_retriever)
    monkeypatch.setattr(agent, "classify_academic",
                        lambda text, model=None: {"academic": False, "reason": "no"})
    t0 = time.monotonic()
    events = _run()
    release.set()
    assert time.monotonic() - t0 < 2
    assert [e["type"] for e in events] == ["status", "blocked"]


def test_index_build_error_surfaces_where_it_did_before(fake, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(agent, "Retriever", broken)
    events = _run()
    types = [e["type"] for e in events]
    # Gate status, then the generic error - no plan, no notes.
    assert types == ["status", "error"]
    assert "plan" not in fake.calls


def test_cancel_during_grounding_starts_no_new_batches(monkeypatch):
    monkeypatch.setattr(agent, "GROUNDING_CONCURRENCY", 3)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    entered, release = threading.Semaphore(0), threading.Event()
    calls = []

    def dispatch(prov, prompt, *a, **k):
        calls.append(1)
        entered.release()
        release.wait(5)
        return '{"verdicts": []}'

    monkeypatch.setattr(models, "_dispatch", dispatch)
    cancel = threading.Event()
    outcome = {}

    def work():
        with models.cancel_scope(cancel):
            try:
                agent.verify_claim_support(_claims_notes(60), CHUNKS)  # 10 batches
                outcome["result"] = "finished"
            except models.PipelineCancelled:
                outcome["result"] = "cancelled"

    t = threading.Thread(target=work)
    t.start()
    for _ in range(3):
        assert entered.acquire(timeout=5)
    cancel.set()
    release.set()
    t.join(5)
    assert not t.is_alive()
    assert outcome["result"] == "cancelled"
    assert len(calls) == 3, "only the batches already in flight ever called a provider"
    assert not [th for th in threading.enumerate()
                if th.name.startswith("grounding") and th.is_alive()]


def test_cancel_during_digest_starts_no_new_segments(monkeypatch):
    monkeypatch.setattr(agent, "DIGEST_CONCURRENCY", 2)
    monkeypatch.setattr(models, "_provider_ready", lambda p: p == "nvidia")
    entered, release = threading.Semaphore(0), threading.Event()
    calls = []

    def dispatch(prov, prompt, *a, **k):
        calls.append(1)
        entered.release()
        release.wait(5)
        return "- topic"

    monkeypatch.setattr(models, "_dispatch", dispatch)
    cancel = threading.Event()
    result = {}

    def work():
        with models.cancel_scope(cancel):
            try:
                agent.digest_document("z" * (agent.DIGEST_SEGMENT_CHARS * 6))
                result["r"] = "finished"
            except models.PipelineCancelled:
                result["r"] = "cancelled"

    t = threading.Thread(target=work)
    t.start()
    for _ in range(2):
        assert entered.acquire(timeout=5)
    cancel.set()
    release.set()
    t.join(5)
    assert result["r"] == "cancelled" and len(calls) == 2


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def timing_log():
    logger = logging.getLogger("agentic")
    handler = _Capture()
    old = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old)


def _timings(handler):
    return [json.loads(line.split("[timing] ", 1)[1])
            for line in handler.lines if line.startswith("[timing] ")]


def test_timing_logs_cover_every_stage_without_content(fake, timing_log):
    events = _run(generation_id="gen-test-1")
    assert events[-1]["type"] == "done"
    rows = _timings(timing_log)
    stages = [r["stage"] for r in rows]
    for name in ("index_build", "topic_check", "plan", "write", "critique",
                 "grounding", "title", "total"):
        assert name in stages, f"missing timing for {name}"
    assert {r["gen_id"] for r in rows} == {"gen-test-1"}
    for r in rows:
        assert r["duration_ms"] >= 0 and r["end_ms"] >= r["start_ms"]
    by = {r["stage"]: r for r in rows}
    assert by["topic_check"]["model_calls"] == 1
    assert by["plan"]["model_calls"] == 1
    assert by["plan"]["providers"] == [f"nvidia:{MODEL}"]
    assert by["total"]["outcome"] == "done"
    assert by["total"]["model_calls"] == len(fake.calls)
    assert sum(r["model_calls"] for r in rows if r["stage"] != "total") == len(fake.calls)
    # No document text, prompts or notes in the log.
    blob = "\n".join(timing_log.lines)
    assert "Photosynthesis" not in blob and "Chlorophyll" not in blob
    assert "First words" not in blob


def test_timing_logs_sectioned_write_counts_every_writer(fake, timing_log, monkeypatch):
    monkeypatch.setattr(agent, "SECTION_CONCURRENCY", 2)
    text, spans = _long_doc()
    events = _run(text, page_spans=spans)
    assert events[-1]["type"] == "done"
    write = next(r for r in _timings(timing_log) if r["stage"] == "write")
    assert write["mode"] == "sectioned" and write["concurrency"] == 2
    assert write["model_calls"] == fake.calls.count("write") == write["windows"]


# ---------------------------------------------------------------------------
# 6. Long documents: nothing waits for the plan
# ---------------------------------------------------------------------------
# Measured on a 55-page document: planning took 28-78 s on the main model and
# the first word of notes could not appear until it was done - yet a long
# document's windows are cut by position and never read the plan. Only the
# critique does.

def _pump(gen):
    """Drain a pipeline on a thread; `text` is set at the first notes_delta."""
    events, text = [], threading.Event()

    def run():
        for ev in gen:
            events.append(ev)
            if ev["type"] == "notes_delta":
                text.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return events, text, thread


def _long_gen(**kw):
    doc, spans = _long_doc()
    return agent.run_agent(doc, "exam", "academic", "medium", "bullet", model=MODEL,
                           include_quiz=False, include_flashcards=False, page_spans=spans, **kw)


def test_long_document_windows_are_written_while_the_plan_is_made(fake):
    gate = fake.gate("plan")
    events, text, thread = _pump(_long_gen())
    assert gate.entered.wait(5), "the planner was never called"
    wrote = text.wait(3)
    gate.release.set()
    thread.join(10)
    assert wrote, "no window wrote a word until the plan was finished"
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert types.index("plan_done") < types.index("critique_done"), (
        "the critic is the plan's reader: it must have it")


def test_short_document_is_still_planned_before_it_is_written(fake):
    """Single-pass notes follow the outline, so there the plan comes first."""
    gate = fake.gate("plan")
    events, text, thread = _pump(agent.run_agent(
        SMALL, "exam", "academic", "medium", "bullet", model=MODEL,
        include_quiz=False, include_flashcards=False))
    assert gate.entered.wait(5)
    assert not text.wait(0.5), "a short document was written before its plan existed"
    gate.release.set()
    thread.join(10)
    types = [e["type"] for e in events]
    assert types[-1] == "done" and types.index("plan_done") < types.index("notes_delta")


def _planner_models(fake, monkeypatch):
    """Which model each planning call was sent to."""
    used, real = [], fake.dispatch

    def dispatch(prov, prompt, max_tokens, model, temperature, json_mode, **kw):
        if "PLANNING agent" in prompt:
            used.append(model)
        return real(prov, prompt, max_tokens, model, temperature, json_mode, **kw)

    monkeypatch.setattr(models, "_dispatch", dispatch)
    monkeypatch.setattr(models, "HELPER_NVIDIA_MODEL", "nvidia/helper-model")
    return used


def test_long_document_plan_runs_on_the_helper_model(fake, monkeypatch):
    used = _planner_models(fake, monkeypatch)
    assert list(_long_gen())[-1]["type"] == "done"
    assert used == ["nvidia/helper-model"]


def test_short_document_plan_stays_on_the_chosen_model(fake, monkeypatch):
    used = _planner_models(fake, monkeypatch)
    assert _run()[-1]["type"] == "done"
    assert used == [MODEL]


def test_a_failed_plan_does_not_cost_a_long_document_its_notes(fake, monkeypatch):
    real = fake.dispatch

    def dispatch(prov, prompt, max_tokens, model, temperature, json_mode, **kw):
        if "PLANNING agent" in prompt:
            raise RuntimeError("503 from https://provider.example/v1?api_key=SECRET_VALUE")
        return real(prov, prompt, max_tokens, model, temperature, json_mode, **kw)

    monkeypatch.setattr(models, "_dispatch", dispatch)
    events = list(_long_gen())
    types = [e["type"] for e in events]
    assert "error" not in types and types[-1] == "done"
    plan = next(e["data"] for e in events if e["type"] == "plan_done")
    assert plan["outline"] and plan["checklist"], "the critic still needs a checklist"
    assert "SECRET_VALUE" not in json.dumps(events)


def test_long_document_timing_log_still_adds_up(fake, timing_log, monkeypatch):
    monkeypatch.setattr(agent, "SECTION_CONCURRENCY", 2)
    assert list(_long_gen(generation_id="gen-long-1"))[-1]["type"] == "done"
    rows = _timings(timing_log)
    by = {r["stage"]: r for r in rows}
    assert by["plan"]["model_calls"] == 1 and by["plan"]["providers"] == [f"nvidia:{MODEL}"]
    assert by["write"]["model_calls"] == by["write"]["windows"], (
        "the planner's call was counted as a window's")
    assert by["total"]["model_calls"] == len(fake.calls)
    assert sum(r["model_calls"] for r in rows if r["stage"] != "total") == len(fake.calls)


# ---------------------------------------------------------------------------
# 7. Long documents: critique alongside the claim check, and no whole rewrite
# ---------------------------------------------------------------------------
# Measured on a 55-page document: the critique took 36-62 s and the claim
# check 47 s, one after the other, though neither reads the other's result.
# And one critique reply that asked for a revision cost a 267-second rewrite
# of all 3,000 words, which came back 268 words and 15 citations shorter.

LOW_CRITIQUE = ('{"score":3,"needs_revision":true,"unsupported_claims":[],'
                '"missing_topics":[],"issues":["weak"],"strengths":[]}')


def _statuses(events):
    return [e["content"] for e in events if e["type"] == "status"]


def test_long_document_claims_are_checked_while_the_critique_runs(fake):
    gate = fake.gate("critique")
    events, _text, thread = _pump(_long_gen())
    assert gate.entered.wait(5), "the critic was never called"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not any(
            "Checking every claim" in s for s in _statuses(events)):
        time.sleep(0.02)
    started = any("Checking every claim" in s for s in _statuses(events))
    gate.release.set()
    thread.join(10)
    assert started, "the claim check waited for the critique to finish"
    types = [e["type"] for e in events]
    assert types[-1] == "done" and types.count("critique_done") == 1


def _critic_says(fake, monkeypatch, reply):
    real = fake.dispatch

    def dispatch(prov, prompt, max_tokens, model, temperature, json_mode, **kw):
        if "CRITIQUE agent" in prompt:
            fake.calls.append("critique")
            return reply
        return real(prov, prompt, max_tokens, model, temperature, json_mode, **kw)

    monkeypatch.setattr(models, "_dispatch", dispatch)


def test_long_document_is_never_rewritten_whole(fake, monkeypatch):
    _critic_says(fake, monkeypatch, LOW_CRITIQUE)
    events = list(_long_gen())
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert "revise_start" not in types and fake.calls.count("revise") == 0
    verdict = next(e for e in events if e["type"] == "critique_done")
    assert verdict["data"]["score"] == 3 and verdict["data"]["needs_revision"] is True, (
        "the critic's verdict is still reported as it was given")
    assert "revising" not in verdict["content"].lower(), "it says a rewrite is under way"
    assert any(e["step"] == "revise" and "no revision" in e["content"].lower()
               for e in events if e["type"] == "status"), "the revise step is left hanging"


def test_short_document_is_still_revised_on_a_low_score(fake, monkeypatch):
    _critic_says(fake, monkeypatch, LOW_CRITIQUE)
    types = [e["type"] for e in _run()]
    assert "revise_start" in types and types[-1] == "done"


def test_the_critic_is_asked_for_a_reply_that_fits_its_budget(monkeypatch):
    """On a real 23,000-character draft the reply listed so many items that it
    was cut at the token cap, and half a JSON object is no verdict at all."""
    seen = {}

    def model(prompt, **kw):
        seen["prompt"] = prompt
        return '{"score":9,"needs_revision":false}'

    monkeypatch.setattr(agent, "call_model", model)
    agent.critique_notes("A claim [1].", {"checklist": []}, "exam", source="[1] some text")
    assert "at most 5" in seen["prompt"]


def test_cancelled_run_logs_a_cancelled_total(fake, timing_log):
    gate = fake.gate("plan")
    cancel = threading.Event()
    gen = agent.run_agent(SMALL, "exam", "academic", "medium", "bullet", model=MODEL,
                          include_quiz=False, include_flashcards=False, cancel=cancel)
    events = []
    t = threading.Thread(target=lambda: events.extend(gen), daemon=True)
    t.start()
    assert gate.entered.wait(5)
    cancel.set()
    gate.release.set()
    t.join(5)
    assert not t.is_alive()
    total = next(r for r in _timings(timing_log) if r["stage"] == "total")
    assert total["outcome"] == "cancelled"
    assert "done" not in [e["type"] for e in events]
