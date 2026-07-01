"""
Retrieval benchmark: current semantic-only vs BM25 vs Hybrid.

Reports, overall and per query category:
  - Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - Average retrieval latency (ms)

Uses the reusable dataset in eval/retrieval_dataset.py. Semantic/hybrid require a
GEMINI_API_KEY; without one they degrade to BM25 (noted in the output).

Run from backend/:  python -m eval.benchmark_retrieval
"""

import os
import sys
import time
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import Retriever, chunk_document  # noqa: E402
from retrieval.semantic import semantic_available  # noqa: E402
from eval.retrieval_dataset import DOCUMENT, QUERIES  # noqa: E402

TOP_K = 10


def _expected_ids(chunks, must_contain):
    return {
        c["chunk_id"]
        for c in chunks
        if any(m.lower() in c["text"].lower() for m in must_contain)
    }


def evaluate(mode):
    chunks = chunk_document(DOCUMENT)
    retriever = Retriever(DOCUMENT, mode=mode)
    rows = []
    for q in QUERIES:
        expected = _expected_ids(chunks, q["must_contain"])
        t = time.perf_counter()
        results = retriever.retrieve(q["query"], k=TOP_K)
        latency_ms = round((time.perf_counter() - t) * 1000, 2)
        ids = [r["id"] for r in results]

        def recall_at(k):
            if not expected:
                return None
            hit = expected & set(ids[:k])
            return len(hit) / len(expected)

        rr = 0.0
        for rank, cid in enumerate(ids, 1):
            if cid in expected:
                rr = 1.0 / rank
                break

        rows.append(
            {
                "category": q["category"],
                "recall5": recall_at(5),
                "recall10": recall_at(10),
                "rr": rr,
                "latency_ms": latency_ms,
            }
        )
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


def main():
    have_gemini = semantic_available()
    print("\n=== Retrieval Benchmark ===")
    print(f"Gemini embeddings: {'available' if have_gemini else 'NOT set (semantic/hybrid degrade to BM25)'}")
    print(f"Queries: {len(QUERIES)}  |  Top-K: {TOP_K}\n")

    modes = ["semantic", "bm25", "hybrid"]
    results = {m: evaluate(m) for m in modes}

    # Overall table
    header = f"{'metric':<16}" + "".join(f"{m:>12}" for m in modes)
    print(header)
    print("-" * len(header))
    summaries = {m: summarize(results[m]) for m in modes}
    for metric in ("recall@5", "recall@10", "mrr", "avg_latency_ms"):
        print(f"{metric:<16}" + "".join(f"{summaries[m][metric]:>12}" for m in modes))

    # Per-category (recall@10)
    print("\nRecall@10 by category:")
    cats = sorted({q["category"] for q in QUERIES})
    print(f"{'category':<18}" + "".join(f"{m:>12}" for m in modes))
    for cat in cats:
        line = f"{cat:<18}"
        for m in modes:
            rows = [r for r in results[m] if r["category"] == cat]
            line += f"{_mean([r['recall10'] for r in rows]):>12}"
        print(line)

    if "hybrid" in summaries and "semantic" in summaries:
        lift = round(summaries["hybrid"]["recall@10"] - summaries["semantic"]["recall@10"], 3)
        print(f"\nHybrid vs semantic-only  Recall@10 lift: {lift:+.3f}")
    print()


if __name__ == "__main__":
    main()
