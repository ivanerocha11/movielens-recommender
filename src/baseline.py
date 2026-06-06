import pandas as pd

def popular_movies(train: pd.DataFrame) -> list[int]:
    """Return all movie IDs ranked by popularity (number of ratings), most first.

    Args:
        train: Training ratings with a 'movie_id' column.

    Returns:
        Movie IDs sorted by rating count, descending.
    """
    return list(train["movie_id"].value_counts().index)

def recommend_popular(
    train: pd.DataFrame,
    user_ids: list[int],
    k: int,
) -> dict[int, list[int]]:
    """Recommend the most popular unseen movies to each user."""
    ranking = popular_movies(train)
    seen_by_user = train.groupby("user_id")["movie_id"].apply(set)
    recommendations = {}
    for user_id in user_ids:
        seen = seen_by_user.get(user_id, set())
        recs = []
        for movie in ranking:
            if movie not in seen:
                recs.append(movie)
                if len(recs) == k:
                    break
        recommendations[user_id] = recs
    return recommendations