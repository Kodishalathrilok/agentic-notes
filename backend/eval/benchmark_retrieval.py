"""
Retrieval benchmark: semantic vs BM25 vs hybrid, over the long eval fixtures.

The queries (evals/retrieval/queries.json) run against the three long
fixtures in evals/fixtures/, each chunked exactly as the pipeline chunks it
(page-bounded for the PDF). Every document is about 60,000 characters - well
over a hundred chunks - so a query's top-k is a small fraction of the
document and recall@5 measures ranking. (The previous dataset was one
1,415-character document in 3 chunks: every query returned every chunk, and
every metric was 1.0 by construction.)

Reports, overall and per query category:
  - Recall@5, Recall@10, MRR (mean reciprocal rank of the first relevant chunk)
  - Average retrieval latency (ms)

Semantic/hybrid need GEMINI_API_KEY; without it they degrade to BM25 and the
output says so.

Run from backend/:  python -m eval.benchmark_retrieval [--json out.json]
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import Retriever  # noqa: E402
from retrieval.semantic import semantic_available  # noqa: E402
from eval.long_fixtures import EVALS_DIR, load_fixture, normalize  # noqa: E402

TOP_K = 10
QUERIES_PATH = os.path.join(EVALS_DIR, "retrieval", "queries.json")


def load_queries(path=QUERIES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["queries"]


def expected_ids(chunks, must_contain):
    wanted = [normalize(m).lower() for m in must_contain]
    return {c["chunk_id"] for c in chunks
            if any(w in normalize(c["text"]).lower() for w in wanted)}


def build_retrievers(mode, queries):
    out = {}
    for name in sorted({q["fixture"] for q in queries}):
        fx = load_fixture(name)
        out[name] = Retriever(fx["text"], mode=mode, spans=fx["page_spans"])
    return out


def evaluate(mode, queries, retrievers=None):
    retrievers = retrievers or build_retrievers(mode, queries)
    rows = []
    for q in queries:
        r = retrievers[q["fixture"]]
        expected = expected_ids(r.chunks_meta, q["must_contain"])
        t = time.perf_counter()
        results = r.retrieve(q["query"], k=TOP_K)
        latency_ms = round((time.perf_counter() - t) * 1000, 2)
        ids = [x["id"] for x in results]

        def recall_at(k):
            if not expected:
                return None
            return len(expected & set(ids[:k])) / len(expected)

        rr = 0.0
        for rank, cid in enumerate(ids, 1):
            if cid in expected:
                rr = 1.0 / rank
                break
        rows.append({"fixture": q["fixture"], "category": q["category"], "query": q["query"],
                     "expected": sorted(expected), "recall5": recall_at(5),
                     "recall10": recall_at(10), "rr": rr, "latency_ms": latency_ms})
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 3) if vals else 0.0


def summarize(rows):
    return {
        "recall@5": _mean([r["recall5"] for r in rows]),
        "recall@10": _mean([r["recall10"] for r in rows]),
        "mrr": _mean([r["rr"] for r in rows]),
        "avg_latency_ms": _mean([r["latency_ms"] for r in rows]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrieval benchmark over the long fixtures.")
    ap.add_argument("--json", default="", help="also write the results to this file")
    args = ap.parse_args(argv)
    queries = load_queries()
    have_gemini = semantic_available()
    print("\n=== Retrieval Benchmark ===")
    print(f"Gemini embeddings: {'available' if have_gemini else 'NOT set (semantic/hybrid degrade to BM25)'}")

    modes = ["semantic", "bm25", "hybrid"]
    results, docs = {}, {}
    for m in modes:
        retrievers = build_retrievers(m, queries)
        # Set after loading: importing main (for PDF extraction) configures
        # logging. Per-query INFO lines would bury the table.
        logging.getLogger("retrieval").setLevel(logging.WARNING)
        docs = {n: len(r.chunks_meta) for n, r in retrievers.items()}
        results[m] = evaluate(m, queries, retrievers)
    print(f"Queries: {len(queries)}  |  Top-K: {TOP_K}  |  chunks per document: {docs}\n")

    header = f"{'metric':<16}" + "".join(f"{m:>12}" for m in modes)
    print(header)
    print("-" * len(header))
    summaries = {m: summarize(results[m]) for m in modes}
    for metric in ("recall@5", "recall@10", "mrr", "avg_latency_ms"):
        print(f"{metric:<16}" + "".join(f"{summaries[m][metric]:>12}" for m in modes))

    cats = sorted({q["category"] for q in queries})
    by_cat = {m: {c: summarize([r for r in results[m] if r["category"] == c]) for c in cats}
              for m in modes}
    print("\nRecall@5 by category:")
    print(f"{'category':<18}" + "".join(f"{m:>12}" for m in modes))
    for cat in cats:
        print(f"{cat:<18}" + "".join(f"{by_cat[m][cat]['recall@5']:>12}" for m in modes))

    lift = round(summaries["hybrid"]["recall@5"] - summaries["semantic"]["recall@5"], 3)
    print(f"\nHybrid vs semantic-only  Recall@5 lift: {lift:+.3f}\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"embeddings_available": have_gemini, "top_k": TOP_K,
                       "queries": len(queries), "chunks_per_document": docs,
                       "summary": summaries, "by_category": by_cat, "rows": results},
                      f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
