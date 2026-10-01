"""
Eval harness for the agentic notes pipeline.

Runs the pipeline over fixed test inputs and uses an independent LLM-judge to
score faithfulness / coverage / clarity and quiz answer-key accuracy. Compares
two variants so the Tier-1 improvements are measurable:

  baseline  — plan -> write only (no critique/revise, no quiz verification)
  full      — the complete run_agent pipeline (grounded critique + revise loop)

Usage (from the backend/ directory):
  python -m eval.run_eval                 # both variants, 1 run each
  python -m eval.run_eval --runs 2        # average over 2 runs per fixture
  python -m eval.run_eval --variant full  # only the full pipeline
  python -m eval.run_eval --model llama-3.1-8b-instant

Writes eval/report.md and eval/report.json.
"""

import os
import re
import sys
import json
import time
import argparse
import statistics
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_agent, plan_outline, write_notes  # noqa: E402
from eval.fixtures import FIXTURES  # noqa: E402
from eval.judge import judge_notes, judge_quiz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

# The pipeline prepends this when a window or a draft could not be completed.
# It is a statement ABOUT THE RUN, not a claim about the subject, so the judge
# must not see it: judged as content it is by definition unsupported by the
# source, and the pipeline gets marked down precisely for being honest that a
# provider cut it off. Observed: it cost one fixture 5 faithfulness points and
# produced the run's only "hallucination".
_BANNER_RE = re.compile(r"^>\s*\*\*Incomplete coverage\*\*.*?(?:\n\n|\Z)", re.DOTALL)


def judgeable(notes: str) -> str:
    """The notes as the judge should see them: content only, no run banners."""
    return _BANNER_RE.sub("", notes or "", count=1).lstrip()


def run_baseline(text, model=None):
    """Pre-Tier-1 behaviour: plan + single write, no critique/revise/verify."""
    plan = plan_outline(text, "exam", "academic", "medium", model=model)
    notes = write_notes(text, "exam", "academic", "medium", "bullet", plan, model=model)
    return {"notes": notes, "quiz": "", "revisions": 0, "self_unsupported": 0}


def run_full(text, model=None):
    """The complete pipeline (consume the SSE-style event generator)."""
    notes = ""
    quiz = ""
    revisions = 0
    self_unsupported = 0
    for ev in run_agent(text, "exam", "academic", "medium", "bullet", model=model):
        t = ev["type"]
        if t == "notes_done":
            notes = ev["content"]
        elif t == "notes_revised":
            notes = ev["content"]
        elif t == "status" and ev["step"] == "revise" and "Revising" in (ev["content"] or ""):
            revisions += 1
        elif t == "critique_done" and ev.get("data"):
            self_unsupported += len(ev["data"].get("unsupported_claims", []) or [])
        elif t == "quiz_done":
            quiz = ev["content"]
        elif t == "error":
            raise RuntimeError(ev["content"])
    return {"notes": notes, "quiz": quiz, "revisions": revisions, "self_unsupported": self_unsupported}


VARIANTS = {"baseline": run_baseline, "full": run_full}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate(rows):
    """rows: list of per-run dicts -> summary dict with mean/std per metric."""
    out = {}
    for key in ("faithfulness", "coverage", "clarity"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = {"mean": round(mean(vals), 2), "std": round(std(vals), 2)}
    halluc = [r["hallucinations"] for r in rows]
    out["avg_hallucinations"] = round(mean(halluc), 2)
    quiz_rows = [r for r in rows if r.get("quiz_total")]
    if quiz_rows:
        accs = [r["quiz_correct"] / r["quiz_total"] for r in quiz_rows]
        out["quiz_accuracy"] = {"mean": round(mean(accs), 2), "std": round(std(accs), 2)}
    else:
        out["quiz_accuracy"] = None
    out["avg_revisions"] = round(mean([r.get("revisions", 0) for r in rows]), 2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate the notes pipeline.")
    ap.add_argument("--runs", type=int, default=1, help="runs per fixture (averaged)")
    ap.add_argument("--variant", choices=["baseline", "full", "both"], default="both")
    ap.add_argument("--model", default="", help="model id override (e.g. gemini-flash-lite-latest)")
    ap.add_argument("--judge-model", default="", help="model for the judge (defaults to --model)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to pause between runs (avoids rate limits)")
    ap.add_argument(
        "--check-faithfulness",
        type=float,
        default=0.0,
        help="exit non-zero if the full pipeline's mean faithfulness is below this (CI gate)",
    )
    args = ap.parse_args()

    model = args.model or None
    judge_model = args.judge_model or model
    variants = ["baseline", "full"] if args.variant == "both" else [args.variant]

    print(f"\nEval: variants={variants} runs={args.runs} model={args.model or 'default'}\n")

    results = {v: [] for v in variants}
    failures = {v: {"count": 0, "last": ""} for v in variants}
    started = time.time()

    for variant in variants:
        runner = VARIANTS[variant]
        for fx in FIXTURES:
            for run_i in range(args.runs):
                if args.delay:
                    time.sleep(args.delay)
                label = f"[{variant}] {fx['id']} run {run_i + 1}/{args.runs}"
                print(f"  running {label} …", flush=True)
                try:
                    gen = runner(fx["source"], model=model)
                    graded = judgeable(gen["notes"])
                    jn = judge_notes(fx["source"], graded, model=judge_model)
                    jq = judge_quiz(fx["source"], graded, gen["quiz"], model=judge_model)
                except Exception as exc:  # noqa: BLE001
                    failures[variant]["count"] += 1
                    failures[variant]["last"] = str(exc)
                    print(f"    ! FAILED: {exc}")
                    continue

                # A judge that returned no usable score has measured NOTHING.
                # Recording it as a number - any number - lets provider
                # flakiness masquerade as a quality result, which is exactly
                # how a clean 10.0 baseline was once reported as 7.5.
                if None in (jn["faithfulness"], jn["coverage"], jn["clarity"]):
                    failures[variant]["count"] += 1
                    failures[variant]["last"] = "judge returned no usable scores"
                    print("    ! JUDGE RETURNED NO SCORES — sample dropped (not scored 0)")
                    continue

                results[variant].append(
                    {
                        "fixture": fx["id"],
                        "faithfulness": jn["faithfulness"],
                        "coverage": jn["coverage"],
                        "clarity": jn["clarity"],
                        "hallucinations": len(jn["hallucinations"]),
                        "hallucination_list": jn["hallucinations"],
                        "quiz_total": jq["total"],
                        "quiz_correct": jq["correct"],
                        "revisions": gen.get("revisions", 0),
                    }
                )
                print(
                    f"    faith={jn['faithfulness']} cov={jn['coverage']} "
                    f"clarity={jn['clarity']} halluc={len(jn['hallucinations'])} "
                    f"quiz={jq['correct']}/{jq['total']}"
                )

    elapsed = round(time.time() - started, 1)
    summary = {v: aggregate(results[v]) for v in variants}

    # ---- Console report ----
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)

    # Sample counts first — makes failed runs obvious instead of silent zeros.
    total_per_variant = len(FIXTURES) * args.runs
    any_data = False
    for v in variants:
        ok = len(results[v])
        if ok:
            any_data = True
        print(f"  {v:<10} samples: {ok}/{total_per_variant} succeeded, {failures[v]['count']} failed")
    if not any_data:
        print("\n  ⚠ ALL RUNS FAILED — scores below are meaningless (0 samples).")
        last = next((failures[v]["last"] for v in variants if failures[v]["last"]), "")
        if last:
            print(f"    Last error: {last}")
        print("    Likely a provider rate limit. Try: --model gemini-flash-lite-latest --delay 2")
    print()

    header = f"{'metric':<22}" + "".join(f"{v:>18}" for v in variants)
    print(header)
    print("-" * len(header))

    def cell(v, metric):
        s = summary[v]
        if metric == "quiz_accuracy":
            return f"{s['quiz_accuracy']['mean']*100:.0f}%" if s["quiz_accuracy"] else "n/a"
        if metric == "avg_hallucinations":
            return f"{s['avg_hallucinations']:.2f}"
        if metric == "avg_revisions":
            return f"{s['avg_revisions']:.2f}"
        return f"{s[metric]['mean']:.2f} ±{s[metric]['std']:.2f}"

    for metric in ("faithfulness", "coverage", "clarity", "quiz_accuracy", "avg_hallucinations", "avg_revisions"):
        print(f"{metric:<22}" + "".join(f"{cell(v, metric):>18}" for v in variants))

    if "baseline" in variants and "full" in variants:
        bf = summary["baseline"]["faithfulness"]["mean"]
        ff = summary["full"]["faithfulness"]["mean"]
        bh = summary["baseline"]["avg_hallucinations"]
        fh = summary["full"]["avg_hallucinations"]
        print("-" * len(header))
        print(f"Faithfulness lift (full - baseline): {ff - bf:+.2f}")
        print(f"Hallucination reduction:             {bh - fh:+.2f} per run")

    print(f"\nElapsed: {elapsed}s\n")

    # ---- Persist ----
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model or "default",
        "runs": args.runs,
        "summary": summary,
        "rows": results,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(HERE, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_markdown(os.path.join(HERE, "report.md"), report, variants)
    print(f"Wrote {os.path.join('eval', 'report.json')} and eval/report.md")

    # CI gate: fail the build if faithfulness regressed below the threshold.
    if args.check_faithfulness:
        if "full" not in summary or not any_data:
            print(f"\nFAIL: no eval data to check against threshold {args.check_faithfulness}.")
            sys.exit(1)
        score = summary["full"]["faithfulness"]["mean"]
        if score < args.check_faithfulness:
            print(f"\nFAIL: full faithfulness {score:.2f} < threshold {args.check_faithfulness}")
            sys.exit(1)
        print(f"\nPASS: full faithfulness {score:.2f} >= threshold {args.check_faithfulness}")


def _write_markdown(path, report, variants):
    s = report["summary"]
    lines = [
        "# Pipeline Eval Report",
        "",
        f"- Generated: {report['timestamp']}",
        f"- Model: `{report['model']}`  ·  Runs/fixture: {report['runs']}  ·  Elapsed: {report['elapsed_s']}s",
        "",
        "## Scores (higher is better; hallucinations lower is better)",
        "",
        "| Metric | " + " | ".join(variants) + " |",
        "|" + "---|" * (len(variants) + 1),
    ]

    def cell(v, metric):
        d = s[v]
        if metric == "quiz_accuracy":
            return f"{d['quiz_accuracy']['mean']*100:.0f}%" if d["quiz_accuracy"] else "n/a"
        if metric in ("avg_hallucinations", "avg_revisions"):
            return f"{d[metric]:.2f}"
        return f"{d[metric]['mean']:.2f} ±{d[metric]['std']:.2f}"

    for metric in ("faithfulness", "coverage", "clarity", "quiz_accuracy", "avg_hallucinations", "avg_revisions"):
        lines.append(f"| {metric} | " + " | ".join(cell(v, metric) for v in variants) + " |")

    if "baseline" in variants and "full" in variants:
        ff = s["full"]["faithfulness"]["mean"] - s["baseline"]["faithfulness"]["mean"]
        hh = s["baseline"]["avg_hallucinations"] - s["full"]["avg_hallucinations"]
        lines += [
            "",
            "## Tier-1 impact",
            "",
            f"- **Faithfulness lift:** {ff:+.2f} points",
            f"- **Hallucination reduction:** {hh:+.2f} per run",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
