#!/usr/bin/env bash
# Measure a base commit and the current checkout with the SAME eval, and
# record both baselines. Needs provider keys for the writer and the judge
# (e.g. GEMINI_API_KEY for a Gemini writer, NVIDIA_API_KEY for an NVIDIA judge).
#
#   JUDGE_MODEL=nvidia/llama-3.1-nemotron-70b-instruct evals/run_before_after.sh 329a031
#
# For each commit it records, in evals/baselines/<short-sha>.json:
#   legacy_eval - that commit's own eval.run_eval (the short-fixture eval)
#   claim_eval  - THIS checkout's claim eval run against that commit's pipeline
set -euo pipefail

BASE="${1:-329a031}"
WRITER="${WRITER_MODEL:-gemini-3.5-flash-lite}"
JUDGE="${JUDGE_MODEL:?set JUDGE_MODEL to a model from a different family than $WRITER}"
RUNS="${RUNS:-3}"
REPO="$(git rev-parse --show-toplevel)"
PY="${PYTHON:-python}"
WT="$(mktemp -d)/base"

git -C "$REPO" worktree add --detach "$WT" "$BASE" >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT

record_legacy() {  # <backend dir> <baseline file> <judge flag...>
  local dir="$1" out="$2"; shift 2
  (cd "$dir" && "$PY" -m eval.run_eval --variant full --model "$WRITER" --delay 2 "$@") \
    || echo "legacy eval exited non-zero; its report is recorded as-is"
  "$PY" "$REPO/backend/eval/baselines.py" record-legacy --baseline "$out" \
    --report "$dir/eval/report.json" --commit "$(git -C "$dir" rev-parse HEAD)" \
    --writer "$WRITER" --judge "${JUDGE_USED:-$WRITER}"
}

BASE_SHA="$(git -C "$WT" rev-parse --short=7 HEAD)"
HEAD_SHA="$(git -C "$REPO" rev-parse --short=7 HEAD)"
BASE_OUT="$REPO/evals/baselines/$BASE_SHA.json"
HEAD_OUT="$REPO/evals/baselines/$HEAD_SHA.json"

# 1. The base commit's own eval, exactly as it shipped (its judge defaults to
#    the writer; --judge-model did not exist yet).
JUDGE_USED="$WRITER" record_legacy "$WT/backend" "$BASE_OUT"

# 2. The claim eval against the base pipeline, then against this checkout.
cd "$REPO/backend"
"$PY" -m eval.claim_eval --pipeline-dir "$WT/backend" --writer-model "$WRITER" \
  --judge-model "$JUDGE" --runs "$RUNS" --delay 2 \
  --out "$REPO/evals/reports/claim_eval_$BASE_SHA.json" --write-baseline "$BASE_OUT"
"$PY" -m eval.claim_eval --writer-model "$WRITER" --judge-model "$JUDGE" \
  --runs "$RUNS" --delay 2 \
  --out "$REPO/evals/reports/claim_eval_$HEAD_SHA.json" --write-baseline "$HEAD_OUT"

# 3. This checkout's legacy eval (now with a required, different-family judge).
JUDGE_USED="$JUDGE" record_legacy "$REPO/backend" "$HEAD_OUT" --judge-model "$JUDGE"

echo "recorded $BASE_OUT and $HEAD_OUT"
