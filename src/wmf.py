import numpy as np
import torch

from src.model import MatrixFactorization


def sample_negative(user_idx, seen_by_user, n_movies):
    """Draw a uniform random movie this user hasn't interacted with."""
    seen = seen_by_user[user_idx]
    while True:
        movie = np.random.randint(n_movies)
        if movie not in seen:
            return movie


def make_batch(batch_indices, train_u, train_m, train_r,
               seen_by_user, n_movies, alpha=40):
    """Build a batch of positives + sampled negatives with confidence weights.

    Returns (users, movies, labels, confidence) as tensors. The batch has
    2 * len(batch_indices) rows: positives (label 1) then negatives (label 0).
    """
    users = train_u[batch_indices]
    pos_movies = train_m[batch_indices]
    ratings = train_r[batch_indices]

    neg_movies = np.array([sample_negative(u, seen_by_user, n_movies) for u in users])

    batch_users = np.concatenate([users, users])
    batch_movies = np.concatenate([pos_movies, neg_movies])
    batch_labels = np.concatenate([np.ones(len(users)), np.zeros(len(users))])
    batch_confidence = np.concatenate([1 + alpha * ratings, np.ones(len(users))])

    return (
        torch.tensor(batch_users, dtype=torch.long),
        torch.tensor(batch_movies, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.float),
        torch.tensor(batch_confidence, dtype=torch.float),
    )


def wmf_loss(preds, labels, confidence):
    """Confidence-weighted squared error (WMF objective)."""
    squared_error = (labels - preds) ** 2
    weighted = confidence * squared_error
    return weighted.mean()


def train_wmf(train_u, train_m, train_r, seen_by_user, n_users, n_movies,
              k=50, lr=0.01, weight_decay=1e-4, n_epochs=80,
              batch_size=1024, verbose=False):
    """Train a weighted matrix factorization model via negative sampling.

    Args:
        train_u, train_m, train_r: aligned arrays of user idx, movie idx, rating.
        seen_by_user: dict mapping user idx -> set of movie idx seen in training.
        n_users, n_movies: number of distinct users / movies.
        k: latent dimension.
        lr, weight_decay: optimizer settings.
        n_epochs, batch_size: training loop settings.
        verbose: print per-epoch loss.

    Returns:
        The trained MatrixFactorization model.
    """
    model = MatrixFactorization(n_users, n_movies, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = len(train_u)
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            batch_indices = perm[i:i + batch_size]
            users, movies, labels, confidence = make_batch(
                batch_indices, train_u, train_m, train_r,
                seen_by_user, n_movies,
            )
            preds = model(users, movies)
            loss = wmf_loss(preds, labels, confidence)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"  epoch {epoch+1}: loss={loss.item():.2f}")
    return model