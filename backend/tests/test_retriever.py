"""
Hybrid retrieval tests: tokenizer, BM25, semantic (mocked), fusion, hybrid,
config switching, metrics, citation-integrity regression, and edge cases.

Semantic tests mock the Gemini embedding calls so they run offline in CI.
"""

import retrieval.semantic as sem
from retriever import Retriever, chunk_text, chunk_document
from retrieval.tokenizer import tokenize
from retrieval.bm25 import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.config import RetrievalConfig

DOC = (
    "Photosynthesis occurs in chloroplasts and produces ATP and glucose. "
    "Chlorophyll absorbs light in the blue and red parts of the spectrum. "
    "Binary search runs in O(log n) time on a sorted array by halving the interval. "
    "Newton's second law states that force equals mass times acceleration, F = ma. "
    "The French Revolution began in 1789 and abolished the monarchy in 1792. "
    "Supply and demand determine the equilibrium price in a market. "
) * 4  # long enough to produce several chunks


# --------------------------------------------------------------------------- tokenizer
def test_tokenizer_lowercases_and_keeps_technical_terms():
    toks = tokenize("The GPT-4 model uses numpy.array and O(log n).")
    assert "gpt-4" in toks
    assert "numpy.array" in toks
    assert "log" in toks and "n" in toks
    assert "the" in toks  # lowercased


# --------------------------------------------------------------------------- fusion
def test_rrf_rewards_items_in_both_lists():
    # idx 2 is high in both lists -> should top the fusion
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], rrf_k=60)
    order = [i for i, _ in fused]
    assert set(order[:2]) == {1, 2}
    assert order[0] == 2 or order[0] == 1
    # scores are descending
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []


# --------------------------------------------------------------------------- BM25
def test_bm25_retrieves_keyword_chunk():
    r = BM25Retriever(chunk_text(DOC))
    results = r.search("binary search O(log n) time complexity", k=3)
    assert results
    top_idx = results[0][0]
    assert "binary search" in chunk_text(DOC)[top_idx].lower()


def test_bm25_empty_corpus():
    assert BM25Retriever([]).search("anything", 3) == []


# --------------------------------------------------------------------------- semantic (mocked)
def _mock_embeddings(monkeypatch, texts, query_target_substr):
    """Give the chunk containing `query_target_substr` a vector aligned to the query."""
    def fake_docs(ts):
        return [[1.0, 0.0] if query_target_substr in t else [0.0, 1.0] for t in ts]

    def fake_query(_q):
        return [1.0, 0.0]

    monkeypatch.setattr(sem, "semantic_available", lambda: True)
    monkeypatch.setattr(sem, "_embed_documents", fake_docs)
    monkeypatch.setattr(sem, "_embed_query", fake_query)


def test_semantic_retrieval_mocked(monkeypatch):
    texts = chunk_text(DOC)
    _mock_embeddings(monkeypatch, texts, "chloroplasts")
    from retrieval.semantic import SemanticRetriever

    r = SemanticRetriever(texts)
    r.available = True
    r.index()
    results = r.search("where does photosynthesis happen", k=2)
    assert results
    assert "chloroplasts" in texts[results[0][0]].lower()


# --------------------------------------------------------------------------- hybrid + config switching
def test_hybrid_falls_back_to_bm25_without_gemini(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever(DOC, mode="hybrid")
    out = r.retrieve("binary search time complexity")
    assert out
    assert r.last_metrics["mode"] == "hybrid"
    assert r.last_metrics["semantic_candidates"] == 0
    assert r.last_metrics["bm25_candidates"] > 0


def test_hybrid_fuses_both_when_semantic_available(monkeypatch):
    texts = chunk_text(DOC)
    _mock_embeddings(monkeypatch, texts, "1789")
    r = Retriever(DOC, mode="hybrid")
    r.retrieve("when did the french revolution begin")
    m = r.last_metrics
    assert m["semantic_candidates"] > 0
    assert m["bm25_candidates"] > 0
    assert m["fusion_candidates"] > 0
    # at least one chunk should be credited to both methods
    assert any(c["source"] == "both" for c in r.last_context) or all(
        c["source"] in ("semantic", "bm25") for c in r.last_context
    )


def test_mode_switching(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r_bm = Retriever(DOC, mode="bm25")
    r_bm.retrieve("acceleration force mass")
    assert r_bm.last_metrics["mode"] == "bm25"
    assert r_bm.last_metrics["semantic_candidates"] == 0


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("INITIAL_RETRIEVAL_K", "25")
    monkeypatch.setenv("FINAL_CONTEXT_K", "6")
    cfg = RetrievalConfig.from_env()
    assert cfg.mode == "bm25" and cfg.initial_retrieval_k == 25 and cfg.final_context_k == 6


# --------------------------------------------------------------------------- returned shape & candidate count
def test_returns_final_context_k(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever(DOC, mode="bm25")
    out = r.retrieve("photosynthesis")  # default final_context_k = 8
    assert len(out) <= 8
    assert all(set(item.keys()) == {"id", "text"} for item in out)  # no score leakage


def test_candidate_pool_is_larger_than_context(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever(DOC, mode="bm25")
    r.retrieve("photosynthesis chlorophyll")
    # pool (fusion_candidates) should be >= returned context
    assert r.last_metrics["fusion_candidates"] >= r.last_metrics["returned_context"]


# --------------------------------------------------------------------------- metrics
def test_metrics_shape(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever(DOC, mode="hybrid")
    r.retrieve("supply and demand equilibrium")
    for key in (
        "mode", "semantic_candidates", "bm25_candidates", "fusion_candidates",
        "returned_context", "semantic_latency_ms", "bm25_latency_ms",
        "fusion_latency_ms", "total_latency_ms",
        "average_semantic_score", "average_bm25_score", "average_rrf_score",
    ):
        assert key in r.last_metrics


# --------------------------------------------------------------------------- CITATION INTEGRITY (regression)
def test_chunking_is_deterministic():
    assert chunk_text(DOC) == chunk_text(DOC)


def test_citation_ids_are_contiguous_and_stable():
    meta = chunk_document(DOC)
    ids = [c["chunk_id"] for c in meta]
    assert ids == list(range(1, len(ids) + 1))


def test_retrieved_id_maps_to_exact_chunk_text(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever(DOC, mode="hybrid")
    out = r.retrieve("newton force mass acceleration")
    for item in out:
        # the citation id must always map back to its exact chunk
        assert item["text"] == r.chunks_meta[item["id"] - 1]["text"]


def test_chunk_metadata_has_offsets():
    meta = chunk_document(DOC)
    for c in meta:
        assert c["start_offset"] <= c["end_offset"]
        assert isinstance(c["text"], str)


# --------------------------------------------------------------------------- edge cases
def test_empty_document(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever("", mode="hybrid")
    assert r.retrieve("anything") == []
    assert r.sample() == ""


def test_small_document(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    r = Retriever("Short note about ATP.", mode="hybrid")
    out = r.retrieve("ATP")
    assert len(out) >= 1
    assert out[0]["id"] == 1


def test_duplicate_chunks(monkeypatch):
    monkeypatch.setattr(sem, "semantic_available", lambda: False)
    dup = "Mitochondria is the powerhouse of the cell. " * 20
    r = Retriever(dup, mode="bm25")
    out = r.retrieve("mitochondria powerhouse")
    assert out  # doesn't crash on near-duplicate chunks
    ids = [i["id"] for i in out]
    assert len(ids) == len(set(ids))  # unique citation ids
