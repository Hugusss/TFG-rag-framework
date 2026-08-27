"""Retrieval-quality metrics, judged at document level.

Pure functions over ranked document ids and a relevance set — no
retrieval, no storage, no I/O — so the same code scores every experiment
and each function is exhaustively unit-testable. Relevance judgements in
``evaluation/queries.jsonl`` name documents, so a chunk-level result list
must be collapsed with :func:`document_ranking` first.
"""

from __future__ import annotations

from rag_framework.models import SearchResult


def document_ranking(results: list[SearchResult]) -> list[str]:
    """Unique document ids in best-first order.

    A document retrieved through several chunks ranks at its best
    (first) chunk; later occurrences are ignored.
    """
    seen: set[str] = set()
    ranking: list[str] = []
    for result in results:
        if result.document_id not in seen:
            seen.add(result.document_id)
            ranking.append(result.document_id)
    return ranking


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents found in the top ``k``.

    Undefined for an empty relevance set — callers must not score
    unanswerable queries with recall (raise instead of guessing).
    """
    if not relevant:
        raise ValueError("recall is undefined for an empty relevance set")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` positions holding a relevant document.

    Conventional fixed-denominator form: fewer than ``k`` retrieved
    documents count as misses, so shallow result lists are penalized,
    not hidden.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    return len(set(ranking[:k]) & relevant) / k


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document, 0.0 if none appears."""
    for rank, document_id in enumerate(ranking, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def summarize(values: list[float]) -> dict:
    """The summary block used by every benchmark: min/median/p95/mean/std."""
    if not values:
        raise ValueError("cannot summarize an empty list")
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    variance = sum((x - mean) ** 2 for x in ordered) / n
    return {
        "n": n,
        "min": ordered[0],
        "median": ordered[n // 2],
        "p95": ordered[min(n - 1, round(0.95 * (n - 1)))],
        "max": ordered[-1],
        "mean": mean,
        "std": variance**0.5,
    }


# --- ranking comparisons: candidate vs reference ---


def overlap_at_k(ranking: list[str], reference: list[str], k: int) -> float:
    """Fraction of the reference top-``k`` ids present in the candidate
    top-``k``, with ``k`` as the denominator (shallow lists count)."""
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ranking[:k]) & set(reference[:k])) / k


def positions_preserved(ranking: list[str], reference: list[str], k: int) -> float:
    """Fraction of the ``k`` positions holding the same id in both."""
    if k < 1:
        raise ValueError("k must be positive")
    return sum(a == b for a, b in zip(ranking[:k], reference[:k])) / k


def discordant_pairs(ranking: list[str], reference: list[str]) -> list[tuple[str, str]]:
    """Pairs of ids present in both lists whose relative order differs
    (``(x, y)`` with x before y in ``ranking``, after it in
    ``reference``) — zero means identical order on the common ids."""
    position = {cid: i for i, cid in enumerate(reference)}
    common = [cid for cid in ranking if cid in position]
    pairs = []
    for i, x in enumerate(common):
        for y in common[i + 1 :]:
            if position[x] > position[y]:
                pairs.append((x, y))
    return pairs


def max_score_difference(
    scores: dict[str, float], reference_scores: dict[str, float]
) -> float | None:
    """Largest absolute score gap over the ids both rankings hold;
    ``None`` when they share nothing."""
    shared = scores.keys() & reference_scores.keys()
    if not shared:
        return None
    return max(abs(scores[cid] - reference_scores[cid]) for cid in shared)


def classify_differences(
    candidate: list[SearchResult],
    baseline: list[SearchResult],
    exact: list[SearchResult],
    *,
    tolerance: float = 1e-6,
) -> dict:
    """Attribute every candidate-vs-baseline difference using the
    exact reference, so a difference is explained rather than assumed.

    Counts returned:

    - ``tie_reorders``: discordant pairs whose two scores are equal
      (within ``tolerance``) — an ordering convention, not an error;
    - ``score_reorders``: discordant pairs with different scores —
      should be zero for exact-scoring backends;
    - ``baseline_misses``: ids the exact reference ranks in its top-k
      that the baseline lacks but the candidate has;
    - ``candidate_misses``: the converse;
    - ``both_miss``: exact top-k ids neither index returned;
    - ``outside_exact``: ids either index returned that are not in the
      exact top-k (the flip side of a miss).
    """
    cand_ids = [r.chunk_id for r in candidate]
    base_ids = [r.chunk_id for r in baseline]
    exact_ids = {r.chunk_id for r in exact}
    cand_scores = {r.chunk_id: r.score for r in candidate}
    base_scores = {r.chunk_id: r.score for r in baseline}

    ties = score_reorders = 0
    for x, y in discordant_pairs(cand_ids, base_ids):
        if abs(cand_scores[x] - cand_scores[y]) <= tolerance:
            ties += 1
        else:
            score_reorders += 1
    cand_set, base_set = set(cand_ids), set(base_ids)
    return {
        "tie_reorders": ties,
        "score_reorders": score_reorders,
        "baseline_misses": len((exact_ids & cand_set) - base_set),
        "candidate_misses": len((exact_ids & base_set) - cand_set),
        "both_miss": len(exact_ids - cand_set - base_set),
        "outside_exact": len((cand_set | base_set) - exact_ids),
    }


def imbalance(values: list[float]) -> dict:
    """Spread of a per-partition quantity (sizes, worker times) relative
    to its mean: how unevenly the work was divided."""
    if not values:
        raise ValueError("imbalance is undefined for an empty list")
    mean = sum(values) / len(values)
    if mean == 0:
        return {"max_over_mean": None, "min_over_mean": None, "mean": 0.0}
    return {
        "max_over_mean": max(values) / mean,
        "min_over_mean": min(values) / mean,
        "mean": mean,
    }
