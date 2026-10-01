"""Record a legacy eval report (eval/report.json) into a baseline file.

    python eval/baselines.py record-legacy --baseline evals/baselines/abc1234.json \\
        --report backend/eval/report.json --commit <sha> --writer W --judge J

A report with no successful samples is recorded as status "failed" with no
summary: run_eval prints 0.00 for zero samples, and a zero copied into a
baseline would read as a measured catastrophe.
"""

import argparse
import json
import os
from datetime import date


def record_legacy(baseline_path, report_path, commit, writer, judge):
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    samples = sum(len(v) for v in (report.get("rows") or {}).values())
    data = {}
    if os.path.isfile(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            data = json.load(f)
        have = data.get("commit") or ""
        if have and not (have.startswith(commit) or commit.startswith(have)):
            raise SystemExit(f"{baseline_path} belongs to commit {have}, not {commit}")
    data.setdefault("commit", commit)
    data.setdefault("date", date.today().isoformat())
    data.setdefault("models", {"writer": writer, "judge": judge})
    data["legacy_eval"] = {
        "status": "ok" if samples else "failed",
        "models": {"writer": writer, "judge": judge},
        "samples_succeeded": samples,
        "report_timestamp": report.get("timestamp"),
        "runs": report.get("runs"),
        "summary": report.get("summary") if samples else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rl = sub.add_parser("record-legacy")
    rl.add_argument("--baseline", required=True)
    rl.add_argument("--report", required=True)
    rl.add_argument("--commit", required=True)
    rl.add_argument("--writer", required=True)
    rl.add_argument("--judge", required=True)
    args = ap.parse_args(argv)
    data = record_legacy(args.baseline, args.report, args.commit, args.writer, args.judge)
    print(f"{args.baseline}: legacy_eval {data['legacy_eval']['status']} "
          f"({data['legacy_eval']['samples_succeeded']} samples)")


if __name__ == "__main__":
    main()
