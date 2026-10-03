"""Compare a claim-eval report with a stored baseline; fail on regressions.

The gate config (evals/gate.json) names the baseline file and, per metric,
which direction is better and how much worse the current run may be:

  "claim_precision": {"better": "higher", "margin": 0.05}
  "est_pipeline_tokens": {"better": "lower", "margin": 0.25, "relative": true}

An absolute margin is in the metric's own units; a relative one is a
fraction of the baseline value. Comparison is on the overall mean.

The gate refuses to pass rather than compare things that are not
comparable: no baseline, a baseline that was never measured, a different
eval version, or different fixtures. Metrics listed under "informational"
are reported but never fail the gate.
"""

import json
import os


class GateError(RuntimeError):
    """The comparison cannot be made; the gate fails."""


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(summary, metric):
    m = ((summary or {}).get("overall") or {}).get(metric)
    return None if not m else m.get("mean")


def check_comparable(report, baseline):
    claim = (baseline or {}).get("claim_eval") or {}
    if claim.get("status") != "ok" or not claim.get("summary"):
        raise GateError(
            "the baseline has no measured claim_eval (status "
            f"{claim.get('status')!r}); record one with evals/run_before_after.sh")
    if claim.get("eval_version") != report.get("eval_version"):
        raise GateError(f"eval version differs: baseline {claim.get('eval_version')}, "
                        f"current {report.get('eval_version')}")
    if claim.get("fixture_fingerprint") != report.get("fixture_fingerprint"):
        raise GateError("the fixtures or labels differ from the baseline's; re-record the "
                        "baseline before gating on it")
    return claim


def compare(report, baseline, config):
    """Return (passed, rows). Raises GateError if not comparable."""
    claim = check_comparable(report, baseline)
    cur_s, base_s = report.get("summary"), claim.get("summary")
    rows, passed = [], True
    for metric, rule in (config.get("metrics") or {}).items():
        base, cur = _mean(base_s, metric), _mean(cur_s, metric)
        row = {"metric": metric, "baseline": base, "current": cur,
               "better": rule["better"], "margin": rule["margin"],
               "relative": bool(rule.get("relative"))}
        if base is None or cur is None:
            row["status"] = "not_compared"
            row["reason"] = "no value in " + ("baseline" if base is None else "current run")
            rows.append(row)
            continue
        allowed = rule["margin"] * abs(base) if rule.get("relative") else rule["margin"]
        worse_by = (base - cur) if rule["better"] == "higher" else (cur - base)
        row["worse_by"] = round(worse_by, 6)
        row["allowed"] = round(allowed, 6)
        row["status"] = "fail" if worse_by > allowed + 1e-12 else "pass"
        passed = passed and row["status"] == "pass"
        rows.append(row)
    for metric in config.get("informational") or []:
        rows.append({"metric": metric, "baseline": _mean(base_s, metric),
                     "current": _mean(cur_s, metric), "status": "informational"})
    return passed, rows


def run_gate(report, config_path, repo_dir):
    """(passed, rows, message) for the report against the configured baseline."""
    config = load_json(config_path)
    path = os.path.join(repo_dir, config["baseline"])
    if not os.path.isfile(path):
        return False, [], f"baseline file {config['baseline']} does not exist"
    try:
        passed, rows = compare(report, load_json(path), config)
    except GateError as exc:
        return False, [], f"cannot compare with {config['baseline']}: {exc}"
    failed = [r["metric"] for r in rows if r.get("status") == "fail"]
    msg = (f"PASS against {config['baseline']}" if passed
           else f"FAIL against {config['baseline']}: {', '.join(failed)}")
    return passed, rows, msg


def format_rows(rows):
    lines = [f"{'metric':<28}{'baseline':>12}{'current':>12}{'allowed':>10}  status"]
    for r in rows:
        def f(v):
            return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"{r['metric']:<28}{f(r.get('baseline')):>12}{f(r.get('current')):>12}"
                     f"{f(r.get('allowed')):>10}  {r['status']}"
                     + (f" ({r['reason']})" if r.get("reason") else ""))
    return "\n".join(lines)
