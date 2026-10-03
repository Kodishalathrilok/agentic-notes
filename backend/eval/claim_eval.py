"""Claim-level eval over long documents, with a baseline gate.

For every fixture in evals/fixtures/ (each over 60,000 characters, one a
paged PDF) the full pipeline runs --runs times. Each run's final notes are
judged by a model from a DIFFERENT family than the writer, claim by claim and
fact by fact, with every verdict backed by a quote Python finds (see
claim_judge). Per run it records:

  claim_precision      supported / (supported + unsupported + miscited)
  unsupported_rate     unsupported / the same denominator
  miscitation_rate     cited claims whose citations do not back them
  unverifiable_rate    verdicts whose quote could not be found
  fact_recall          labelled facts the notes state
  late_fact_recall     the same, for facts stated only at the end
  contradictions_caught_rate   planted contradictions the notes flag
  qb_violation_rate    question-bank items the notes answer anyway
  grounding_removed    claims the pipeline's grounding step deleted
  est_*_tokens         ESTIMATED tokens (characters / 4) - see instrument.py
  *_cost_usd           from evals/prices.json; null if any model is unpriced
  seconds              wall-clock time of the pipeline run

and reports mean, standard deviation, min and max per fixture and overall.

    python -m eval.claim_eval --writer-model gemini-3.5-flash-lite \\
        --judge-model nvidia/nemotron-3-super-120b-a12b --runs 3 \\
        --gate ../evals/gate.json

--pipeline-dir runs the same harness against another checkout's backend
(e.g. a worktree of an older commit), which is how a before/after pair is
measured with one eval. --write-baseline records the summary into
evals/baselines/<commit>.json.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from eval.families import SameFamilyError, model_family, require_different_families  # noqa: E402

# Bump when a metric's definition, a prompt or the claim extraction changes:
# the gate refuses to compare reports from different versions.
EVAL_VERSION = 1

# Gated metrics plus the informational ones, in report order.
METRICS = (
    "claim_precision", "unsupported_rate", "miscitation_rate", "unverifiable_rate",
    "fact_recall", "late_fact_recall", "contradictions_caught_rate", "qb_violation_rate",
    "grounding_removed", "grounding_rewritten", "grounding_unjudged", "claims_total",
    "est_pipeline_tokens", "est_judge_tokens", "pipeline_cost_usd", "judge_cost_usd",
    "seconds",
)


def spread(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


def summarize(rows):
    """Mean and spread per metric, overall and per fixture (valid runs only)."""
    valid = [r for r in rows if r.get("valid")]
    out = {"overall": {}, "per_fixture": {}}
    for m in METRICS:
        out["overall"][m] = spread([r["metrics"].get(m) for r in valid])
    for fx in sorted({r["fixture"] for r in rows}):
        fr = [r for r in valid if r["fixture"] == fx]
        out["per_fixture"][fx] = {m: spread([r["metrics"].get(m) for r in fr])
                                  for m in METRICS}
    return out


def _git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _bootstrap(pipeline_dir):
    """Load the pipeline modules from `pipeline_dir` BEFORE anything else
    imports them, so this harness measures that checkout's code."""
    pipeline_dir = os.path.abspath(pipeline_dir)
    sys.path.insert(0, pipeline_dir)
    import models  # noqa: F401
    import agent  # noqa: F401
    import main  # noqa: F401
    for name in ("models", "agent", "main"):
        loaded = os.path.dirname(os.path.abspath(sys.modules[name].__file__))
        if loaded != pipeline_dir:
            raise SystemExit(f"{name} was loaded from {loaded}, not {pipeline_dir}")
    return pipeline_dir


def judge_run(fixture, gen, judge_model, claim_batch):
    """Score one finished run. Raises JudgeUnavailableError if unmeasured."""
    from eval import claim_judge
    from eval.run_eval import judgeable

    notes = judgeable(gen["notes"])
    chunk_map = {int(s["id"]): s for s in gen.get("sources") or [] if "id" in s}
    claims = claim_judge.judge_claims(notes, fixture["text"], chunk_map, judge_model,
                                      batch_size=claim_batch)
    cm = claim_judge.claim_metrics(claims)
    facts = claim_judge.judge_facts(notes, fixture["labels"], judge_model)
    contra = claim_judge.judge_contradictions(notes, fixture["labels"], judge_model)
    qb = claim_judge.judge_question_bank(notes, fixture["labels"], judge_model)
    return {
        "claims": cm, "facts": facts, "contradictions": contra, "question_bank": qb,
        "claim_details": [{k: c[k] for k in ("line", "text", "ids", "class")}
                          for c in claims],
    }


def run_metrics(scored, gen, pipeline_usage, judge_usage, prices):
    from eval.instrument import estimate_cost

    g = (gen.get("grounding") or {}).get("totals") or {}
    p_cost, p_unpriced = estimate_cost(pipeline_usage, prices)
    j_cost, j_unpriced = estimate_cost(judge_usage, prices)
    return {
        "claim_precision": scored["claims"]["claim_precision"],
        "unsupported_rate": scored["claims"]["unsupported_rate"],
        "miscitation_rate": scored["claims"]["miscitation_rate"],
        "unverifiable_rate": scored["claims"]["unverifiable_rate"],
        "claims_total": scored["claims"]["claims_total"],
        "fact_recall": scored["facts"]["fact_recall"],
        "late_fact_recall": scored["facts"]["late_fact_recall"],
        "contradictions_caught_rate": scored["contradictions"]["contradictions_caught_rate"],
        "qb_violation_rate": scored["question_bank"]["qb_violation_rate"],
        "grounding_removed": g.get("removed", 0),
        "grounding_rewritten": g.get("rewritten", 0),
        "grounding_unjudged": g.get("unjudged", 0),
        "est_pipeline_tokens": pipeline_usage["est_input_tokens"]
        + pipeline_usage["est_output_tokens"],
        "est_judge_tokens": judge_usage["est_input_tokens"] + judge_usage["est_output_tokens"],
        "pipeline_cost_usd": p_cost,
        "judge_cost_usd": j_cost,
        "seconds": gen.get("elapsed_s"),
    }, sorted(set(p_unpriced) | set(j_unpriced))


def _stage_seconds(timings):
    out = {}
    for s in timings or []:
        out[s["stage"]] = round(out.get(s["stage"], 0) + s.get("duration_ms", 0) / 1000, 2)
    return out


def write_baseline(path, report):
    """Record the report's summary as the claim_eval part of a baseline file."""
    data = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        have = (data.get("commit") or "")
        if have and not (have.startswith(report["commit"]) or report["commit"].startswith(have)):
            raise SystemExit(f"{path} is the baseline for commit {have}, but this run "
                             f"measured {report['commit']}; refusing to mix them")
    data.setdefault("commit", report["commit"])
    data.setdefault("date", report["timestamp"][:10])
    data.setdefault("models", {"writer": report["models"]["writer"],
                               "judge": report["models"]["judge"]})
    data["claim_eval"] = {
        "status": "ok",
        "eval_version": report["eval_version"],
        "fixture_fingerprint": report["fixture_fingerprint"],
        "fixtures": report["fixtures"],
        "runs_per_fixture": report["runs_per_fixture"],
        "models": report["models"],
        "dirty_worktree": report["dirty_worktree"],
        "generated": report["timestamp"],
        "token_estimate": report["token_estimate"],
        "valid_runs": report["valid_runs"],
        "attempted_runs": report["attempted_runs"],
        "summary": report["summary"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _parse(argv):
    from eval.long_fixtures import EVALS_DIR

    ap = argparse.ArgumentParser(description="Claim-level eval over long documents.")
    ap.add_argument("--writer-model", default="",
                    help="pipeline model (default: the server's default model)")
    ap.add_argument("--judge-model", required=True,
                    help="judge model; required, from a different family than the writer")
    ap.add_argument("--runs", type=int, default=3, help="runs per fixture (default 3)")
    ap.add_argument("--fixtures", nargs="*", default=None, help="subset of fixture names")
    ap.add_argument("--pipeline-dir", default=_BACKEND,
                    help="backend directory whose pipeline code is measured")
    ap.add_argument("--claim-batch", type=int, default=20, help="claims per judge call")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between runs")
    ap.add_argument("--prices", default=os.path.join(EVALS_DIR, "prices.json"))
    ap.add_argument("--out", default=os.path.join(EVALS_DIR, "reports", "claim_eval.json"))
    ap.add_argument("--min-valid-runs", type=int, default=2,
                    help="gate fails if any fixture has fewer valid runs than this")
    ap.add_argument("--gate", default="", help="gate config (e.g. ../evals/gate.json)")
    ap.add_argument("--write-baseline", default="", help="baseline file to record into")
    return ap, ap.parse_args(argv)


def main(argv=None):
    ap, args = _parse(argv)
    pipeline_dir = _bootstrap(args.pipeline_dir)

    import models
    from eval.instrument import UsageMeter, CHARS_PER_TOKEN
    from eval.judge_call import JudgeUnavailableError
    from eval.long_fixtures import (REPO_DIR, fixture_fingerprint, fixture_names,
                                    label_problems, load_fixture)
    from eval.run_eval import run_full

    writer = models.resolve_model(args.writer_model or None)
    try:
        require_different_families(writer, args.judge_model)
    except SameFamilyError as exc:
        ap.error(str(exc))
    judge_family = model_family(args.judge_model)

    names = args.fixtures or fixture_names()
    fixtures = [load_fixture(n) for n in names]
    for fx in fixtures:
        problems = label_problems(fx)
        if problems:
            raise SystemExit(f"{fx['id']}: labels do not match the source: {problems}")

    prices = {}
    if os.path.isfile(args.prices):
        with open(args.prices, encoding="utf-8") as f:
            prices = (json.load(f) or {}).get("models") or {}

    print(f"\nclaim eval: writer={writer} judge={args.judge_model} runs={args.runs} "
          f"fixtures={names}\n  pipeline={pipeline_dir}\n")
    rows = []
    started = time.time()
    with UsageMeter(models) as meter:
        for fx in fixtures:
            for run_i in range(1, args.runs + 1):
                if args.delay:
                    time.sleep(args.delay)
                label = f"{fx['id']} run {run_i}/{args.runs}"
                print(f"  running {label} ...", flush=True)
                row = {"fixture": fx["id"], "run": run_i, "valid": False}
                meter.reset()
                meter.role = "pipeline"
                try:
                    gen = run_full(fx["text"], model=args.writer_model or None,
                                   page_spans=fx["page_spans"])
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"pipeline: {exc}"
                    print(f"    ! pipeline failed: {exc}")
                    rows.append(row)
                    continue
                pipe_usage = meter.snapshot("pipeline")
                served = sorted(pipe_usage["by_model"])
                clash = [m for m in served if model_family(m) == judge_family]
                row.update({"coverage": gen.get("coverage"), "done": gen.get("done"),
                            "grounding": gen.get("grounding"),
                            "stage_seconds": _stage_seconds(gen.get("timings")),
                            "served_pipeline_models": served})
                if clash:
                    row["error"] = (f"judge family {judge_family!r} also served the pipeline "
                                    f"({clash}); not a valid measurement")
                    print(f"    ! {row['error']}")
                    rows.append(row)
                    continue
                meter.role = "judge"
                try:
                    scored = judge_run(fx, gen, args.judge_model, args.claim_batch)
                except JudgeUnavailableError as exc:
                    row["error"] = f"judge: {exc}"
                    print(f"    ! judge failed - run not scored: {exc}")
                    rows.append(row)
                    continue
                finally:
                    meter.role = "pipeline"
                judge_usage = meter.snapshot("judge")
                metrics, unpriced = run_metrics(scored, gen, pipe_usage, judge_usage, prices)
                row.update({"valid": True, "metrics": metrics, "unpriced_models": unpriced,
                            "usage": {"pipeline": pipe_usage, "judge": judge_usage},
                            "scored": scored})
                rows.append(row)
                print("    precision={claim_precision} recall={fact_recall} "
                      "late={late_fact_recall} miscite={miscitation_rate} "
                      "contradictions={contradictions_caught_rate} qb={qb_violation_rate} "
                      "removed={grounding_removed} tokens~{est_pipeline_tokens} "
                      "{seconds}s".format(**metrics))

    summary = summarize(rows)
    commit = _git(["rev-parse", "HEAD"], pipeline_dir)
    report = {
        "eval": "claim_eval",
        "eval_version": EVAL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty_worktree": bool(_git(["status", "--porcelain", "--", "."], pipeline_dir)),
        "pipeline_dir": pipeline_dir,
        "models": {"writer": writer, "judge": args.judge_model,
                   "writer_family": model_family(writer), "judge_family": judge_family},
        "fixtures": names,
        "fixture_fingerprint": fixture_fingerprint(names),
        "runs_per_fixture": args.runs,
        "attempted_runs": len(rows),
        "valid_runs": sum(1 for r in rows if r.get("valid")),
        "token_estimate": f"characters / {CHARS_PER_TOKEN}; providers' own counts are not "
                          f"available outside models.py",
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "runs": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwrote {args.out}: {report['valid_runs']}/{report['attempted_runs']} runs valid")
    for m in METRICS:
        s = summary["overall"].get(m)
        print(f"  {m:<28}" + ("n/a" if not s else
                              f"{s['mean']:.4g} ±{s['std']:.3g} [{s['min']:.4g}, {s['max']:.4g}]"))

    exit_code = 0
    short = [fx for fx in names
             if sum(1 for r in rows if r["fixture"] == fx and r.get("valid")) < args.min_valid_runs]
    if args.write_baseline:
        if short:
            print(f"\nnot writing a baseline: too few valid runs for {short}")
            exit_code = 1
        else:
            write_baseline(args.write_baseline, report)
            print(f"recorded baseline in {args.write_baseline}")
    if args.gate:
        from eval.gate import format_rows, run_gate
        if short:
            print(f"\nGATE FAIL: fewer than {args.min_valid_runs} valid runs for {short}")
            return 1
        passed, gate_rows, msg = run_gate(report, args.gate, REPO_DIR)
        if gate_rows:
            print("\n" + format_rows(gate_rows))
        print(f"\nGATE {msg}")
        if not passed:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
