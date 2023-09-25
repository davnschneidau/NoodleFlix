import torch
import torch.nn as nn
import numpy as np
from recommendations.utils import load_data

# import Prisma models
from my_recommendation_package.prisma_models import User, Account, Movie

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

    def forward(self, user, movie):
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        dot_product = torch.sum(user_embed * movie_embed, dim=1)
        return dot_product

def train_recommendation_model(user_id):
    # load user data from Prisma
    user = User.find_first(where={"id": user_id})
    if not user:
        raise Exception(f"User with ID {user_id} not found")

    # fetch and preprocess user-movie interaction data from Prisma models
    accounts = Account.find_many(where={"userId": user_id})
    movie_ids = [account.type for account in accounts]
    movies = Movie.find_many(where={"id": {"in": movie_ids}})

    num_users = 1  # for one user at a time
    num_movies = len(movies)
    embedding_dim = 50  # adjust as needed

    model = MatrixFactorization(num_users, num_movies, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        for user_index, movie_index in zip([0] * len(movie_ids), range(len(movie_ids))):
            user = torch.tensor(user_index, dtype=torch.long)
            movie = torch.tensor(movie_index, dtype=torch.long)
            target = torch.tensor(1, dtype=torch.float32)  # Assuming all interactions are positive

            optimizer.zero_grad()
            prediction = model(user, movie)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

    return model

def get_recommendations(user_id, model):
    # load user data from Prisma
    user = User.find_first(where={"id": user_id})
    if not user:
        raise Exception(f"User with ID {user_id} not found")

    # fetch user-movie interaction data from Prisma models
    accounts = Account.find_many(where={"userId": user_id})
    movie_ids = [account.type for account in accounts]
    movies = Movie.find_many(where={"id": {"in": movie_ids}})

    user_index = user_id
    user = torch.tensor(user_index, dtype=torch.long)
    user_embedding = model.user_embedding(user)
    movie_embeddings = model.movie_embedding.weight

    movie_scores = torch.matmul(user_embedding, movie_embeddings.t())
    recommended_movie_indices = torch.argsort(movie_scores, descending=True)[:10]

    recommended_movies = [movies[i] for i in recommended_movie_indices.tolist()]

    return recommended_movies
