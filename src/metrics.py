import math
import numpy as np
import torch

def safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else None

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

def evaluate_loo(model, user_idx, true_idx, seen, n_movies, k=10, n_neg=100):
    """Leave-one-out score for a single user.

    Ranks the held-out true item against n_neg random unseen items.

    Args:
        model: trained recommender (callable on user/movie tensors).
        user_idx: dense index of the user.
        true_idx: dense index of the held-out true item.
        seen: set of movie indices this user saw in training.
        n_movies: total number of movies.
        k: cutoff for hit/ndcg.
        n_neg: number of random distractors.

    Returns:
        (hit, ndcg) for this user.
    """
    negatives = []
    while len(negatives) < n_neg:
        cand = np.random.randint(n_movies)
        if cand not in seen and cand != true_idx:
            negatives.append(cand)

    candidates = [true_idx] + negatives
    with torch.no_grad():
        users = torch.full((len(candidates),), user_idx, dtype=torch.long)
        movies = torch.tensor(candidates, dtype=torch.long)
        scores = model(users, movies)

    rank = torch.argsort(scores, descending=True).tolist().index(0)
    hit = 1.0 if rank < k else 0.0
    ndcg = 1.0 / np.log2(rank + 2) if rank < k else 0.0
    return hit, ndcg


def score_loo(model, held_out, user_to_idx, movie_to_idx, seen_by_user,
              n_movies, k=10, n_neg=100):
    """Average leave-one-out Hit@k and NDCG@k over all held-out users.

    Args:
        model: trained recommender.
        held_out: DataFrame indexed by user_id, with a 'movie_id' column
                  giving each user's single held-out true item.
        user_to_idx, movie_to_idx: original-id -> dense-index maps.
        seen_by_user: dict user_idx -> set of seen movie idx.
        n_movies: total number of movies.
        k, n_neg: eval settings.

    Returns:
        (mean_hit, mean_ndcg).
    """
    model.eval()
    hits, ndcgs = [], []
    for user_id in held_out.index:
        true_movie = held_out.loc[user_id, "movie_id"]
        if true_movie not in movie_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        true_idx = movie_to_idx[true_movie]
        seen = seen_by_user[user_idx]
        hit, ndcg = evaluate_loo(model, user_idx, true_idx, seen, n_movies, k, n_neg)
        hits.append(hit)
        ndcgs.append(ndcg)
    return np.mean(hits), np.mean(ndcgs)