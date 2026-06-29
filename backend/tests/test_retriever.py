"""Tests for the TF-IDF retriever (chunking + relevance + edge cases)."""

from retriever import Retriever, chunk_text

SOURCE = (
    "Photosynthesis converts light energy into chemical energy in chloroplasts. "
    "Binary search halves a sorted array each step, giving O(log n) time complexity. "
    "The French Revolution began in 1789 and abolished the monarchy. "
) * 6


def test_chunking_produces_multiple_chunks():
    chunks = chunk_text(SOURCE, target_chars=120)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_retrieve_returns_relevant_chunk():
    r = Retriever(SOURCE, backend="tfidf")
    results = r.retrieve("binary search time complexity", k=3)
    assert len(results) >= 1
    joined = " ".join(c["text"].lower() for c in results)
    assert "binary search" in joined
    assert all("id" in c and "text" in c for c in results)


def test_retrieve_ids_are_in_document_order():
    r = Retriever(SOURCE, backend="tfidf")
    results = r.retrieve("photosynthesis", k=4)
    ids = [c["id"] for c in results]
    assert ids == sorted(ids)


def test_empty_source_is_safe():
    r = Retriever("", backend="tfidf")
    assert r.retrieve("anything") == []
    assert r.sample() == ""


def test_sample_returns_text():
    r = Retriever(SOURCE, backend="tfidf")
    assert len(r.sample(200)) > 0
