import pandas as pd


def load_ratings(path: str = "../data/ml-1m/ratings.dat") -> pd.DataFrame:
    """Load the MovieLens-1M ratings file into a DataFrame."""
    return pd.read_csv(
        path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )


def time_split(ratings: pd.DataFrame, frac: float = 0.8):
    """Sort by time and split into train/test, returning train and warm test."""
    ratings_sorted = ratings.sort_values("timestamp").reset_index(drop=True)
    cutoff = int(len(ratings_sorted) * frac)
    train = ratings_sorted[:cutoff]
    test = ratings_sorted[cutoff:]
    train_users = set(train["user_id"])
    test_warm = test[test["user_id"].isin(train_users)]
    return train, test_warm


def build_id_maps(train: pd.DataFrame):
    """Build contiguous user/movie index maps from the training set."""
    user_to_idx = {uid: i for i, uid in enumerate(train["user_id"].unique())}
    movie_to_idx = {mid: i for i, mid in enumerate(train["movie_id"].unique())}
    return user_to_idx, movie_to_idx