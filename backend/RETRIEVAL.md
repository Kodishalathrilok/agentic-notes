# Hybrid Retrieval

The retrieval layer combines **semantic search** (Gemini embeddings) and
**keyword search** (BM25) and fuses them with **Reciprocal Rank Fusion (RRF)**.
It replaces the previous semantic-only (with TF-IDF fallback) design.

## Why hybrid?

- **Semantic** matches *meaning* — great for paraphrased queries, but it can miss
  exact tokens.
- **BM25** matches *exact terms* — great for acronyms, symbols, and formulas
  (`ATP`, `O(log n)`, `F = ma`) that embeddings often blur.
- **Fusing both** gives the best of each: broad recall from embeddings, precise
  keyword hits from BM25.

## Pipeline

```
Document
  → chunk_document()                     # ~700-char chunks, stable positional IDs + offsets
  → HybridRetriever
       ├─ SemanticRetriever  (Gemini embeddings + cosine) → ranked list
       └─ BM25Retriever      (rank-bm25 Okapi)            → ranked list
  → Reciprocal Rank Fusion (RRF)         # rank-based, no score normalization needed
  → candidate pool (INITIAL_RETRIEVAL_K = 30, relevance order)
  → candidate processing stage           # identity today; the reranker plugs in here
  → final context (FINAL_CONTEXT_K = 8) → LLM
```

**Results stay in relevance order** (most relevant first). Citation IDs are chunk
*identity* (position + 1), independent of ordering, so citations are unaffected.

## Reciprocal Rank Fusion

Each ranked list contributes `1 / (RRF_K + rank)` to every document; contributions
sum across lists, then documents are re-ranked by the fused score. Because it uses
**ranks, not raw scores**, it needs no normalization between the two incomparable
scales (cosine vs BM25). Default `RRF_K = 60`.

## Configuration (env only)

| Variable | Default | Meaning |
| --- | --- | --- |
| `RETRIEVAL_MODE` | `hybrid` | `semantic` \| `bm25` \| `hybrid` |
| `INITIAL_RETRIEVAL_K` | `30` | candidates pulled per method and fused (the pool) |
| `FINAL_CONTEXT_K` | `8` | chunks handed to the LLM |
| `RRF_K` | `60` | RRF constant |

Switching modes is a config change only — no code edits.

## Failure handling

If Gemini embeddings are unavailable, retrieval does **not** fail silently — it
logs clearly and continues on BM25:

```
Gemini embeddings unavailable. Running BM25 retrieval only.
```

## Metrics (internal)

Every retrieval records structured metrics on the `Retriever` instance
(`.last_metrics`) — not exposed via the API, ready for a future dashboard:

```json
{
  "mode": "hybrid",
  "semantic_candidates": 30, "bm25_candidates": 30, "fusion_candidates": 30,
  "returned_context": 8,
  "semantic_latency_ms": 1037, "bm25_latency_ms": 0.3,
  "fusion_latency_ms": 0.1, "total_latency_ms": 960,
  "average_semantic_score": 0.71, "average_bm25_score": 6.2, "average_rrf_score": 0.03
}
```

Per-chunk metadata is preserved in `.last_context`: `chunk_id`, `text`,
`start_offset`, `end_offset`, `semantic_score`, `bm25_score`, `rrf_score`, and a
selection `source` trace (`semantic` / `bm25` / `both`).

## Structure

```
backend/
├── retriever.py              # public facade (coordinator) — unchanged interface
└── retrieval/
    ├── config.py             # RetrievalConfig
    ├── tokenizer.py          # lightweight technical-term tokenizer
    ├── semantic.py           # SemanticRetriever (Gemini)
    ├── bm25.py               # BM25Retriever (rank-bm25)
    ├── fusion.py             # reciprocal_rank_fusion
    └── hybrid.py             # HybridRetriever + metrics + failure logging
```

## Testing & benchmark

```bash
cd backend
python -m pytest tests/test_retriever.py      # unit + citation-integrity regression
python -m eval.benchmark_retrieval            # semantic vs bm25 vs hybrid
```

The benchmark uses the reusable dataset in `eval/retrieval_dataset.py` (queries
categorized as keyword / semantic / mixed / technical_acronym / formula) and
reports Recall@5, Recall@10, MRR, and average latency.

## Next

The isolated **candidate-processing stage** in `retriever.py` is where a
Cross-Encoder reranker will plug in: it will reorder the 30-candidate pool before
the final top-8 cut, without changing any caller.
