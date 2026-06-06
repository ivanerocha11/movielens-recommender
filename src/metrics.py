import math

def safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Precision@k for a single user's recommendations.

    Of the top-k recommended items, what fraction were actually relevant.

    Args:
        recommended: Movie IDs in ranked order (best first).
        relevant: Movie IDs the user actually liked (rating >= 4).
        k: Number of top recommendations to evaluate.

    Returns:
        Fraction of the top-k recommendations that were relevant, in [0, 1].
    """
    top_k = recommended[:k]
    hits = len([movie for movie in top_k if movie in relevant])
    return hits / k

def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float | None:
    """Compute Recall@k for a single user's recommendations.

    Of all the items the user actually liked, what fraction appeared
    in the top-k recommendations.

    Args:
        recommended: Movie IDs in ranked order (best first).
        relevant: Movie IDs the user actually liked (rating >= 4).
        k: Number of top recommendations to evaluate.

    Returns:
        Fraction of relevant items captured in the top-k, in [0, 1],
        or None if the user has no relevant items (recall is undefined).
    """
    if not relevant:
        return None
    top_k = recommended[:k]
    hits = len([movie for movie in top_k if movie in relevant])
    return hits / len(relevant)

def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float | None:
    """Compute NDCG@k for a single user's recommendations.

    Rewards placing relevant items higher in the ranking via a logarithmic
    position discount, then normalizes by the ideal ranking so scores are
    comparable across users.

    Args:
        recommended: Movie IDs in ranked order (best first).
        relevant: Movie IDs the user actually liked (rating >= 4).
        k: Number of top recommendations to evaluate.

    Returns:
        NDCG in [0, 1], where 1.0 is a perfect ranking, or None if the
        user has no relevant items (NDCG is undefined).
    """
    if not relevant:
        return None

    top_k = recommended[:k]

    # DCG: walk the ranked list, each relevant hit contributes a
    # discounted gain based on its position (1-indexed).
    dcg = 0.0
    for i, movie in enumerate(top_k):
        position = i + 1  # enumerate starts at 0; positions start at 1
        if movie in relevant:
            dcg += 1 / math.log2(position + 1)

    # IDCG: the best possible DCG — all relevant items packed at the top.
    # The number of ideal "hits" is limited by both how many relevant
    # items exist and how many slots (k) we have.
    ideal_hits = min(len(relevant), k)
    idcg = 0.0
    for i in range(ideal_hits):
        position = i + 1
        idcg += 1 / math.log2(position + 1)

    return dcg / idcg

def evaluate(
    recommendations: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Average Precision@k, Recall@k, and NDCG@k across all users.

    Args:
        recommendations: Maps user_id to their ranked list of recommended movie IDs.
        relevant_by_user: Maps user_id to the set of movie IDs they found relevant.
        k: Number of top recommendations to evaluate.

    Returns:
        Dict with mean 'precision', 'recall', and 'ndcg' across users.
        Users with no relevant items are skipped for recall and ndcg.
    """
    precisions = []
    recalls = []
    ndcgs = []

    for user_id, recommended in recommendations.items():
        relevant = relevant_by_user.get(user_id, set())

        precisions.append(precision_at_k(recommended, relevant, k))
        if relevant:
            recalls.append(recall_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

    return {
        "precision": safe_mean(precisions),
        "recall": safe_mean(recalls),
        "ndcg": safe_mean(ndcgs),
    }