# Evals

Two evals, both run by `.github/workflows/eval.yml` (weekly and on demand):

| Eval | What it runs | What it measures |
| --- | --- | --- |
| `backend/eval/run_eval.py` | 4 short fixtures (under 1,000 characters) | holistic 1-10 faithfulness, coverage and clarity; gate: mean faithfulness >= 7.0 |
| `backend/eval/claim_eval.py` | the long fixtures below, 3 runs each | claim-level grounding, fact recall, trap handling, cost; gate: no metric worse than the stored baseline by more than its margin |

Both require `--judge-model`, from a different model family than the writer
(`backend/eval/families.py`). Judge calls go to the judge's own provider and
never fail over, so a judge outage cannot be answered by the writer's family.

## Fixtures (`fixtures/`)

| Fixture | Source | Size | Facts |
| --- | --- | --- | --- |
| `thermodynamics` | plain text, course reader | 61,394 characters | 33 |
| `cell_biology` | paged PDF, 44 pages (built from `pages.txt` by `fixtures/build_pdf.py`) | 61,781 characters as extracted | 34 |
| `computing_history` | plain text, background reading | 61,331 characters | 36 |

Every source is original text written for this eval. Each `labels.json` holds
hand-written atomic facts, each with a `quote` that appears exactly once in
the source, and three planted traps:

- **late fact**: facts stated only in the last 10% of the document;
- **contradiction**: the document states one value early and a different one
  much later (the calorimeter's heat capacity, the centrifuge setting, the
  year the university got its first computer);
- **question bank**: review questions the document asks but never answers
  (Otto cycle, CRISPR, Plankalkül...). Notes may say the document asks them;
  answering them is a violation.

The test suite checks every quote against the source (for the PDF, against
what `/api/extract-pdf` returns), that late facts really are late, and that
the contradiction values really differ, so a source edit that breaks a label
fails CI before an eval spends tokens on it.

## Claim eval metrics

Every judge verdict carries a word-for-word quote that Python must find (in
the source, the cited passages, or the notes). A verdict whose quote is not
found is **unverifiable**: counted, never treated as a pass or a fail.

| Metric | Definition |
| --- | --- |
| `claim_precision` | supported / (supported + unsupported + miscited) |
| `unsupported_rate` | unsupported / the same denominator |
| `miscitation_rate` | cited claims whose citations do not back them (including citations to passages that do not exist) / cited claims decided |
| `unverifiable_rate` | claims whose verdict quote was not found / all claims |
| `fact_recall`, `late_fact_recall` | labelled facts the notes state (quote found in the notes) |
| `contradictions_caught_rate` | planted contradictions the notes flag, with both values present |
| `qb_violation_rate` | question-bank items the notes answer anyway |
| `grounding_removed` | claims the pipeline's grounding step deleted (informational) |
| `est_pipeline_tokens`, `est_judge_tokens` | **estimated** tokens, characters / 4 |
| `pipeline_cost_usd` | from `prices.json`; null while any model used has no price |
| `seconds` | wall-clock time of the pipeline run |

Each is reported per fixture and overall as mean, standard deviation, min and
max over the valid runs. A run is invalid (and excluded, with its reason in
the report) if the pipeline errored, if any judge call failed, or if the
judge's family also served part of the pipeline.

## Baselines and the gate

`baselines/<commit>.json` records, per commit, the legacy eval (`legacy_eval`)
and the claim eval (`claim_eval`). `gate.json` names the baseline to compare
against and the margin per metric. The gate fails, rather than passing
vacuously, when the baseline has no measured claim eval, or was measured on
different fixtures or a different eval version.

`baselines/329a031.json` has no model scores: it was recorded in a session
without provider keys, and says so. To record real ones for a base commit and
the current checkout with the same eval:

```bash
export GEMINI_API_KEY=... NVIDIA_API_KEY=...
JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b evals/run_before_after.sh 329a031
```

then point `gate.json` at the newer baseline.

## Retrieval benchmark

`python -m eval.benchmark_retrieval` (from `backend/`) runs `retrieval/queries.json`
against the same three fixtures. See `backend/RETRIEVAL.md`.
