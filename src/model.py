import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, k=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, k)
        self.movie_factors = nn.Embedding(n_movies, k)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user, movie):
        p_u = self.user_factors(user)
        q_m = self.movie_factors(movie)
        b_u = self.user_bias(user).squeeze()
        b_m = self.movie_bias(movie).squeeze()
        dot = (p_u * q_m).sum(dim=1)
        return self.global_bias + b_u + b_m + dot