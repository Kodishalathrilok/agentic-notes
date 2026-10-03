"""The claim-level eval, its fixtures, its judge rules and its gate.

None of this needs a model: the judge and the pipeline are faked where a
model would answer. What is checked is the harness itself - that labels match
their sources, that a verdict only counts when its quote is really there,
that the judge can never be the writer's family, and that the gate fails on
a regression and refuses to compare things that are not comparable.
"""

import copy
import json
import logging
import os
import shutil

import pytest

import agent
import models
import retrieval.semantic as sem
from eval import claim_judge, gate
from eval.families import SameFamilyError, model_family, require_different_families
from eval.instrument import GroundingCapture, TimingCapture, UsageMeter, estimate_cost
from eval.judge_call import JudgeUnavailableError, call_judge
from eval.long_fixtures import (FIXTURES_DIR, REPO_DIR, fixture_fingerprint, fixture_names,
                                label_problems, load_fixture, normalize)


@pytest.fixture(scope="module")
def fixtures():
    return {n: load_fixture(n) for n in fixture_names()}


# ---------------------------------------------------------------------------
# fixtures and labels
# ---------------------------------------------------------------------------

def test_there_are_three_to_five_long_fixtures_and_one_is_a_paged_pdf(fixtures):
    assert 3 <= len(fixtures) <= 5
    for fx in fixtures.values():
        assert len(fx["text"]) > 60000, fx["id"]
        # Over the digest threshold, so the eval exercises the digest scan and
        # the windowed writer - the paths the old fixtures never reached.
        assert len(fx["text"]) > agent.DIGEST_DOC_THRESHOLD
    pdfs = [fx for fx in fixtures.values() if fx["labels"]["source"]["type"] == "pdf"]
    assert pdfs, "one fixture must be a paged PDF"
    for fx in pdfs:
        assert fx["pages"] >= 30
        assert len(fx["page_spans"]) == fx["pages"]


def test_every_label_matches_its_source(fixtures):
    for fx in fixtures.values():
        assert label_problems(fx) == [], fx["id"]


def test_every_fixture_plants_all_three_traps(fixtures):
    for fx in fixtures.values():
        traps = fx["labels"]["traps"]
        assert traps["late_fact"]["fact_ids"]
        assert traps["contradictions"]
        assert traps["question_bank"]["questions"]
        assert len(fx["labels"]["facts"]) >= 25


def test_label_check_is_not_vacuous(fixtures):
    fx = copy.deepcopy(fixtures["thermodynamics"])
    fx["labels"]["facts"][0]["quote"] = "a sentence this document never contains"
    late = fx["labels"]["traps"]["late_fact"]
    late["fact_ids"] = ["F02"]  # an early fact, labelled as late
    c = fx["labels"]["traps"]["contradictions"][0]
    c["value_b"] = c["value_a"]
    problems = label_problems(fx)
    assert any("quote not in source" in p for p in problems)
    assert any("sits at" in p for p in problems)
    assert any("same" in p for p in problems)


def test_pdf_fixture_is_read_through_the_real_extraction_endpoint(fixtures):
    fx = fixtures["cell_biology"]
    # The labels were written against pages.txt; they must still match what a
    # user's upload of the PDF produces, and pages must line up with spans.
    with open(os.path.join(fx["dir"], "pages.txt"), encoding="utf-8") as f:
        pages = [p for p in f.read().split("=== PAGE ===") if p.strip()]
    assert fx["pages"] == len(pages)
    norm = normalize(fx["text"])
    last = fx["page_spans"][-1]
    assert abs(last["end"] - len(norm)) <= 2


def test_fingerprint_changes_when_labels_change(tmp_path):
    root = tmp_path / "fx"
    shutil.copytree(FIXTURES_DIR, root)
    names = fixture_names(str(root))
    before = fixture_fingerprint(names, str(root))
    p = root / "thermodynamics" / "labels.json"
    data = json.loads(p.read_text())
    data["facts"][0]["fact"] += " (edited)"
    p.write_text(json.dumps(data))
    assert fixture_fingerprint(names, str(root)) != before


# ---------------------------------------------------------------------------
# the judge is never the writer's family, and never fails over
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mid,family", [
    ("gemini-3.5-flash-lite", "google"),
    ("gemini-3.6-flash", "google"),
    ("nvidia/llama-3.1-nemotron-70b-instruct", "meta-llama"),
    ("meta/llama-3.3-70b-instruct", "meta-llama"),
    ("nvidia/nemotron-4-340b-instruct", "nvidia-nemotron"),
    ("mistralai/mixtral-8x22b-instruct", "mistral"),
    ("qwen2.5:3b", "qwen"),
])
def test_model_family(mid, family):
    assert model_family(mid) == family


def test_same_family_judge_is_refused():
    with pytest.raises(SameFamilyError):
        require_different_families("gemini-3.6-flash", "gemini-3.5-flash-lite")
    with pytest.raises(SameFamilyError):
        require_different_families("gemini-3.6-flash", "")
    require_different_families("gemini-3.6-flash", "nvidia/llama-3.1-nemotron-70b-instruct")


def test_legacy_eval_requires_a_judge_model(monkeypatch, capsys):
    from eval import run_eval
    monkeypatch.setattr("sys.argv", ["run_eval", "--variant", "full"])
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2
    assert "--judge-model" in capsys.readouterr().err


def test_legacy_eval_refuses_a_same_family_judge(monkeypatch, capsys):
    from eval import run_eval
    monkeypatch.setattr("sys.argv", ["run_eval", "--model", "gemini-3.6-flash",
                                     "--judge-model", "gemini-3.5-flash-lite"])
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2
    assert "family" in capsys.readouterr().err


def test_judge_call_never_fails_over_to_another_provider(monkeypatch):
    calls = []
    monkeypatch.setattr(models, "_provider_ready", lambda prov: True)

    def dispatch(prov, prompt, max_tokens, model, temperature, json_mode, **kw):
        calls.append(prov)
        raise RuntimeError("503")

    monkeypatch.setattr(models, "_dispatch", dispatch)
    with pytest.raises(JudgeUnavailableError):
        call_judge("p", "nvidia/llama-3.1-nemotron-70b-instruct")
    assert calls == ["nvidia"]


def test_judge_call_refuses_an_unconfigured_provider(monkeypatch):
    monkeypatch.setattr(models, "_provider_ready", lambda prov: False)
    monkeypatch.setattr(models, "_dispatch", lambda *a, **k: pytest.fail("called"))
    with pytest.raises(JudgeUnavailableError):
        call_judge("p", "nvidia/llama-3.1-nemotron-70b-instruct")


# ---------------------------------------------------------------------------
# claim extraction and classification
# ---------------------------------------------------------------------------

SOURCE = ("The Carnot efficiency equals one minus the ratio of the cold reservoir "
          "temperature to the hot reservoir temperature. Water has an unusually high "
          "specific heat capacity of about 4186 J/(kg·K).")
CHUNKS = {
    3: {"id": 3, "text": "The Carnot efficiency equals one minus the ratio of the cold "
                         "reservoir temperature to the hot reservoir temperature."},
    4: {"id": 4, "text": "Water has an unusually high specific heat capacity of about "
                         "4186 J/(kg·K)."},
}
CARNOT_Q = ("The Carnot efficiency equals one minus the ratio of the cold reservoir "
            "temperature to the hot reservoir temperature")


def test_extract_claims_skips_structure_and_keeps_every_citation():
    notes = ("**Heat engines:**\n## Entropy\n---\n• Short label\n"
             "• The Carnot efficiency depends only on the two temperatures [3][1234]\n"
             "```\ncode line that is long enough to count\n```\n"
             "- Water has a high specific heat capacity [4]")
    claims = claim_judge.extract_claims(notes)
    assert [c["ids"] for c in claims] == [[3, 1234], [4]]
    assert claims[0]["text"] == "The Carnot efficiency depends only on the two temperatures"


def _claim(ids):
    return {"n": 1, "line": 0, "text": "claim", "ids": ids}


def _classify(claim, verdict):
    return claim_judge.classify_claim(claim, verdict, claim_judge._plain(SOURCE), CHUNKS)


def test_supported_needs_both_quotes_to_be_real():
    v = {"in_source": True, "source_quote": CARNOT_Q, "cited_backs": True,
         "cited_quote": CARNOT_Q}
    assert _classify(_claim([3]), v) == "supported"
    assert _classify(_claim([3]), {**v, "cited_quote": "a sentence that is not there at all"}
                     ) == "unverifiable"
    assert _classify(_claim([3]), {**v, "source_quote": "invented words that are nowhere"}
                     ) == "unverifiable"


def test_true_claim_with_a_citation_that_does_not_back_it_is_miscited():
    v = {"in_source": True, "source_quote": CARNOT_Q, "cited_backs": False,
         "cited_quote": "Water has an unusually high specific heat capacity"}
    assert _classify(_claim([4]), v) == "miscited"


def test_a_citation_that_resolves_to_nothing_is_miscited():
    v = {"in_source": True, "source_quote": CARNOT_Q, "cited_backs": True,
         "cited_quote": CARNOT_Q}
    assert _classify(_claim([3, 999]), v) == "miscited"


def test_unsupported_needs_a_real_closest_quote_too():
    v = {"in_source": False, "source_quote": CARNOT_Q, "cited_backs": None,
         "cited_quote": ""}
    assert _classify(_claim([]), v) == "unsupported"
    assert _classify(_claim([]), {**v, "source_quote": "nothing like this"}) == "unverifiable"
    assert _classify(_claim([]), None) == "unjudged"


def test_claim_metrics():
    results = [
        {"ids": [3], "class": "supported"}, {"ids": [], "class": "supported"},
        {"ids": [4], "class": "miscited"}, {"ids": [], "class": "unsupported"},
        {"ids": [], "class": "unverifiable"},
    ]
    m = claim_judge.claim_metrics(results)
    assert m["claim_precision"] == pytest.approx(2 / 4)
    assert m["unsupported_rate"] == pytest.approx(1 / 4)
    assert m["miscitation_rate"] == pytest.approx(1 / 2)
    assert m["unverifiable_rate"] == pytest.approx(1 / 5)


def test_judge_claims_batches_and_maps_verdicts(monkeypatch):
    seen = []

    def judge(prompt, model, max_tokens=1024, temperature=0.0, json_mode=True):
        seen.append(prompt)
        n = prompt.count("CLAIM ")
        return json.dumps({"verdicts": [
            {"n": k, "in_source": True, "source_quote": CARNOT_Q, "cited_backs": True,
             "cited_quote": CARNOT_Q} for k in range(1, n + 1)]})

    monkeypatch.setattr(claim_judge, "call_judge", judge)
    notes = "\n".join(f"• The Carnot efficiency claim number {i} here [3]" for i in range(5))
    out = claim_judge.judge_claims(notes, SOURCE, CHUNKS, "nvidia/x", batch_size=2)
    assert len(seen) == 3
    assert [r["class"] for r in out] == ["supported"] * 5


def test_a_failed_judge_call_raises_instead_of_scoring(monkeypatch):
    def judge(*a, **k):
        raise JudgeUnavailableError("down")
    monkeypatch.setattr(claim_judge, "call_judge", judge)
    with pytest.raises(JudgeUnavailableError):
        claim_judge.judge_claims("• The Carnot efficiency depends on temperatures [3]",
                                 SOURCE, CHUNKS, "nvidia/x")


# ---------------------------------------------------------------------------
# facts, contradictions, question bank
# ---------------------------------------------------------------------------

LABELS = {
    "facts": [{"id": "F1", "fact": "Water's specific heat is about 4186 J/(kg·K)."},
              {"id": "F2", "fact": "Carnot efficiency formula."},
              {"id": "F3", "fact": "Late fact."}],
    "traps": {
        "late_fact": {"fact_ids": ["F3"]},
        "contradictions": [{"id": "C1", "about": "calorimeter heat capacity",
                            "value_a": "85 J/K", "value_b": "120 J/K"}],
        "question_bank": {"questions": [{"id": "Q1", "quote": "Derive the Otto cycle",
                                         "topic": "Otto"},
                                        {"id": "Q2", "quote": "Describe Stirling",
                                         "topic": "Stirling"}]},
    },
}
NOTES = ("• Water has an unusually high specific heat capacity of about 4186 J/(kg·K) [4]\n"
         "• The calorimeter heat capacity is given as 85 J/K in one place and 120 J/K in "
         "another; the reader is inconsistent.\n"
         "• The Otto cycle efficiency is one minus the compression ratio to a power.\n"
         "• The reader asks students to describe the Stirling cycle.")


def _fake_judge(payload):
    return lambda *a, **k: json.dumps(payload)


def test_fact_recall_only_counts_quotes_found_in_the_notes(monkeypatch):
    monkeypatch.setattr(claim_judge, "call_judge", _fake_judge({"facts": [
        {"id": "F1", "covered": True,
         "notes_quote": "Water has an unusually high specific heat capacity of about 4186"},
        {"id": "F2", "covered": True, "notes_quote": "a quote the notes do not contain"},
        {"id": "F3", "covered": False, "notes_quote": ""}]}))
    r = claim_judge.judge_facts(NOTES, LABELS, "nvidia/x")
    assert r["fact_recall"] == pytest.approx(1 / 3)
    assert r["late_fact_recall"] == 0
    assert r["fact_claims_unverified"] == 1


def test_contradiction_is_caught_only_when_both_values_are_in_the_notes(monkeypatch):
    quote = "The calorimeter heat capacity is given as 85 J/K in one place"
    monkeypatch.setattr(claim_judge, "call_judge", _fake_judge({"contradictions": [
        {"id": "C1", "status": "flagged", "notes_quote": quote}]}))
    assert claim_judge.judge_contradictions(NOTES, LABELS, "x")["contradictions_caught"] == 1
    one_value = NOTES.replace("and 120 J/K in another", "elsewhere")
    assert claim_judge.judge_contradictions(one_value, LABELS, "x")["contradictions_caught"] == 0


def test_question_bank_answers_are_violations(monkeypatch):
    monkeypatch.setattr(claim_judge, "call_judge", _fake_judge({"questions": [
        {"id": "Q1", "status": "answered",
         "notes_quote": "The Otto cycle efficiency is one minus the compression ratio"},
        {"id": "Q2", "status": "reported_as_task",
         "notes_quote": "The reader asks students to describe the Stirling cycle"}]}))
    r = claim_judge.judge_question_bank(NOTES, LABELS, "x")
    assert r["qb_violations"] == 1 and r["qb_violation_rate"] == 0.5
    assert r["statuses"] == {"Q1": "answered", "Q2": "reported_as_task"}


# ---------------------------------------------------------------------------
# aggregation and the gate
# ---------------------------------------------------------------------------

def test_summary_reports_mean_and_spread_over_valid_runs_only():
    from eval.claim_eval import summarize
    rows = [{"fixture": "a", "valid": True, "metrics": {"claim_precision": 0.8}},
            {"fixture": "a", "valid": True, "metrics": {"claim_precision": 0.9}},
            {"fixture": "a", "valid": False},
            {"fixture": "b", "valid": True, "metrics": {"claim_precision": 1.0}}]
    s = summarize(rows)
    overall = s["overall"]["claim_precision"]
    assert overall["n"] == 3 and overall["mean"] == pytest.approx(0.9)
    assert overall["min"] == 0.8 and overall["max"] == 1.0 and overall["std"] > 0
    assert s["per_fixture"]["a"]["claim_precision"]["n"] == 2


def _report(precision, tokens=1000.0, fingerprint="fp", version=1):
    return {"eval_version": version, "fixture_fingerprint": fingerprint,
            "summary": {"overall": {"claim_precision": {"mean": precision},
                                    "est_pipeline_tokens": {"mean": tokens},
                                    "pipeline_cost_usd": None}}}


def _baseline(precision, tokens=1000.0):
    r = _report(precision, tokens)
    return {"claim_eval": {"status": "ok", **r}}


CONFIG = {"metrics": {
    "claim_precision": {"better": "higher", "margin": 0.05},
    "est_pipeline_tokens": {"better": "lower", "margin": 0.25, "relative": True},
    "pipeline_cost_usd": {"better": "lower", "margin": 0.25, "relative": True}},
    "informational": ["grounding_removed"]}


def test_gate_passes_within_margin_and_fails_beyond_it():
    ok, rows = gate.compare(_report(0.86), _baseline(0.90), CONFIG)
    assert ok
    ok, rows = gate.compare(_report(0.84), _baseline(0.90), CONFIG)
    assert not ok
    assert next(r for r in rows if r["metric"] == "claim_precision")["status"] == "fail"


def test_gate_relative_margin_and_unmeasured_cost():
    ok, rows = gate.compare(_report(0.9, tokens=1300), _baseline(0.9, tokens=1000), CONFIG)
    assert not ok
    cost = next(r for r in rows if r["metric"] == "pipeline_cost_usd")
    assert cost["status"] == "not_compared"
    ok, _ = gate.compare(_report(0.9, tokens=1200), _baseline(0.9, tokens=1000), CONFIG)
    assert ok


def test_gate_refuses_to_compare_incomparable_runs():
    with pytest.raises(gate.GateError):
        gate.compare(_report(0.9), {"claim_eval": {"status": "not_run"}}, CONFIG)
    with pytest.raises(gate.GateError):
        gate.compare(_report(0.9, fingerprint="other"), _baseline(0.9), CONFIG)
    with pytest.raises(gate.GateError):
        gate.compare(_report(0.9, version=2), _baseline(0.9), CONFIG)


def test_the_configured_gate_fails_until_a_real_baseline_is_recorded():
    # The 329a031 baseline holds no measured claim_eval (it was recorded
    # without provider keys). The gate must fail on it, not pass vacuously.
    passed, rows, msg = gate.run_gate(_report(1.0), os.path.join(REPO_DIR, "evals",
                                                                 "gate.json"), REPO_DIR)
    assert not passed and "no measured claim_eval" in msg


# ---------------------------------------------------------------------------
# run_full keeps what the run reported about itself
# ---------------------------------------------------------------------------

def _pipeline_model(prompt, max_tokens=1024, model=None, temperature=0.4, json_mode=False,
                    on_serve=None, **kw):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    if "PLANNING agent" in prompt:
        return ('{"outline":["A","B"],"checklist":["x"],'
                '"difficulty":"easy","suggested_format":"bullet"}')
    if "CRITIQUE agent" in prompt:
        return '{"score":9,"needs_revision":false,"unsupported_claims":[],"missing_topics":[]}'
    if "GROUNDING agent" in prompt:
        return ('{"verdicts":[{"n":1,"status":"unsupported","evidence_quote":"x"}]}')
    if "gatekeeper" in prompt:
        return '{"academic":true,"subject":"science","doc_type":"explanatory"}'
    if "QUIZ" in prompt:
        return '{"questions":[]}'
    return "Title"


def _pipeline_stream(prompt, max_tokens=1400, model=None, temperature=0.4, on_serve=None,
                     **kw):
    if on_serve:
        on_serve("nvidia", "test-model", False, "")
    yield "• Heat flows from hot bodies to cold bodies in every case [1]\n"


def test_run_full_keeps_coverage_grounding_and_timings(monkeypatch):
    from eval.run_eval import run_full
    monkeypatch.setattr(agent, "call_model", _pipeline_model)
    monkeypatch.setattr(agent, "call_model_stream", _pipeline_stream)
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    text = " ".join(f"Paragraph {i} about heat flow and temperature." for i in range(400))
    out = run_full(text)
    assert out["coverage"]["complete"] is True
    assert out["grounding"]["calls"], "grounding stats were not kept"
    assert out["grounding"]["totals"]["checked"] >= 1
    stages = {s["stage"] for s in out["timings"]}
    assert {"plan", "write", "critique", "grounding", "total"} <= stages
    assert all(s["gen_id"] == out["gen_id"] for s in out["timings"])
    assert out["sources"] and "text" in out["sources"][0]
    assert out["elapsed_s"] >= 0
    # The capture hooks are removed afterwards.
    assert agent.verify_claim_support.__name__ == "verify_claim_support"
    assert getattr(agent.verify_claim_support, "__module__", "") == "agent"


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------

def test_usage_meter_counts_by_role_and_model_and_restores(monkeypatch):
    monkeypatch.setattr(models, "_dispatch", lambda prov, p, *a, **k: "x" * 40)

    def stream(prov, p, *a, **k):
        yield "ab"
        yield "cd"

    monkeypatch.setattr(models, "_dispatch_stream", stream)
    orig = (models._dispatch, models._dispatch_stream)
    with UsageMeter(models) as meter:
        models._dispatch("gemini", "p" * 400, 10, "gemini-3.5-flash-lite", 0.0, True)
        meter.role = "judge"
        list(models._dispatch_stream("gemini", "q" * 8, 10, "gemini-3.5-flash-lite", 0.0))
    assert (models._dispatch, models._dispatch_stream) == orig
    pipe, judge = meter.snapshot("pipeline"), meter.snapshot("judge")
    assert pipe["prompt_chars"] == 400 and pipe["output_chars"] == 40
    assert pipe["est_input_tokens"] == 100 and pipe["est_output_tokens"] == 10
    assert judge["output_chars"] == 4 and judge["calls"] == 1
    assert list(pipe["by_model"]) == ["gemini-3.5-flash-lite"]


def test_cost_is_null_when_any_model_is_unpriced():
    snap = {"by_model": {"a": {"prompt_chars": 4_000_000, "output_chars": 400_000},
                         "b": {"prompt_chars": 4, "output_chars": 4}}}
    prices = {"a": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
    assert estimate_cost(snap, prices) == (None, ["b"])
    prices["b"] = {"input_per_mtok": 0.0, "output_per_mtok": 0.0}
    cost, unpriced = estimate_cost(snap, prices)
    assert unpriced == [] and cost == pytest.approx(1.0 + 0.2)


def test_timing_capture_keeps_only_its_own_run():
    log = logging.getLogger("agentic")
    with TimingCapture("mine") as cap:
        log.info("[timing] %s", json.dumps({"gen_id": "mine", "stage": "plan"}))
        log.info("[timing] %s", json.dumps({"gen_id": "other", "stage": "plan"}))
        log.info("not a timing line")
    assert [s["stage"] for s in cap.stages] == ["plan"]


def test_grounding_capture_restores_the_original():
    orig = agent.verify_claim_support
    with GroundingCapture(agent) as cap:
        assert agent.verify_claim_support is not orig
        agent.verify_claim_support("## Heading only", {})
    assert agent.verify_claim_support is orig
    assert cap.calls and "checked" in cap.calls[0]


# ---------------------------------------------------------------------------
# retrieval benchmark dataset
# ---------------------------------------------------------------------------

def test_retrieval_dataset_is_big_enough_that_ranking_matters():
    from eval import benchmark_retrieval as br
    queries = br.load_queries()
    assert len(queries) >= 30
    retrievers = br.build_retrievers("bm25", queries)
    for name, r in retrievers.items():
        assert len(r.chunks_meta) >= 100, name
    for q in queries:
        r = retrievers[q["fixture"]]
        assert br.expected_ids(r.chunks_meta, q["must_contain"]), q["query"]
        # top-k is a small slice of the document, so recall@5 is earned.
        assert len(r.retrieve(q["query"], k=br.TOP_K)) < len(r.chunks_meta) / 5


def test_bm25_recall_at_5_is_not_trivially_perfect():
    from eval import benchmark_retrieval as br
    queries = br.load_queries()
    summary = br.summarize(br.evaluate("bm25", queries))
    assert summary["recall@5"] < 1.0


# ---------------------------------------------------------------------------
# end to end, with a faked pipeline and judge
# ---------------------------------------------------------------------------

def test_claim_eval_end_to_end_writes_a_report_baseline_and_gates(monkeypatch, tmp_path):
    from eval import claim_eval

    monkeypatch.setattr(agent, "call_model", _pipeline_model)
    monkeypatch.setattr(agent, "call_model_stream", _pipeline_stream)
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    fx = load_fixture("thermodynamics")
    late = fx["labels"]["facts"][-1]["quote"]

    def judge(prompt, model, max_tokens=1024, temperature=0.0, json_mode=True):
        if "CLAIM 1" in prompt:
            n = prompt.count("\nCLAIM ") + prompt.startswith("CLAIM ")
            return json.dumps({"verdicts": [
                {"n": k, "in_source": False, "source_quote": late, "cited_backs": None,
                 "cited_quote": ""} for k in range(1, n + 1)]})
        if '"facts"' in prompt:
            return json.dumps({"facts": []})
        if '"contradictions"' in prompt:
            return json.dumps({"contradictions": [{"id": "C1", "status": "omitted",
                                                   "notes_quote": ""}]})
        return json.dumps({"questions": []})

    monkeypatch.setattr(claim_judge, "call_judge", judge)
    out, base = tmp_path / "report.json", tmp_path / "baseline.json"
    args = ["--writer-model", "gemini-3.5-flash-lite",
            "--judge-model", "nvidia/llama-3.1-nemotron-70b-instruct",
            "--fixtures", "thermodynamics", "--runs", "2", "--out", str(out),
            "--write-baseline", str(base), "--min-valid-runs", "2"]
    assert claim_eval.main(args) == 0
    report = json.loads(out.read_text())
    assert report["valid_runs"] == 2 and report["runs_per_fixture"] == 2
    s = report["summary"]["overall"]
    assert s["claim_precision"]["mean"] == 0.0   # the fake judge rejects every claim
    assert s["unsupported_rate"]["mean"] == 1.0
    assert s["fact_recall"]["mean"] == 0.0
    assert s["seconds"]["n"] == 2
    assert report["models"]["judge_family"] != report["models"]["writer_family"]
    stored = json.loads(base.read_text())
    assert stored["claim_eval"]["status"] == "ok"

    cfg = tmp_path / "gate.json"
    rel = os.path.relpath(base, REPO_DIR)
    cfg.write_text(json.dumps({"baseline": rel, "metrics": {
        "claim_precision": {"better": "higher", "margin": 0.05}}}))
    passed, rows, msg = gate.run_gate(report, str(cfg), REPO_DIR)
    assert passed, msg


def test_claim_eval_refuses_a_same_family_judge(capsys):
    from eval import claim_eval
    with pytest.raises(SystemExit):
        claim_eval.main(["--writer-model", "gemini-3.6-flash",
                         "--judge-model", "gemini-3.5-flash-lite"])
