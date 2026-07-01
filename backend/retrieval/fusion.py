"""Reciprocal Rank Fusion (RRF).

Combines several ranked lists using only ranks (not raw scores), so the two
methods' incomparable score scales don't need normalization. Each list gives a
document `1 / (rrf_k + rank)`; contributions sum across lists.
"""


def reciprocal_rank_fusion(rankings, rrf_k: int = 60):
    """
    rankings: list of ranked lists of chunk indices (best-first).
    Returns [(index, rrf_score)] sorted by fused score, descending.
    """
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
