import os
import pymongo
import numpy as np
from sklearn.model_selection import train_test_split
from recommendations.model import MatrixFactorization
from recommendations.utils import load_data

def train_recommendation_model(user_id):
    db_url = os.environ.get("DATABASE_URL")
    client = pymongo.MongoClient(db_url)
    db = client.get_default_database()

    # fetch data from collections using Prisma models
    users = User.find_many()
    accounts = Account.find_many()
    movies = Movie.find_many()

    # create a user-movie interaction matrix
    user_ids = [str(user.id) for user in users]
    movie_ids = [str(movie.id) for movie in movies]

    user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    interaction_matrix = np.zeros((num_users, num_movies))

    for account in accounts:
        user_index = user_id_to_index[str(account.userId)]
        movie_index = movie_id_to_index[str(account.type)]
        interaction_matrix[user_index, movie_index] = 1

    # training and testing sets
    train_matrix, test_matrix = train_test_split(interaction_matrix, test_size=0.2, random_state=42)

    embedding_dim = 50
    model = MatrixFactorization(num_users, num_movies, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        for user_index, movie_index in zip(*np.where(train_matrix == 1)):
            user = torch.tensor(user_index, dtype=torch.long)
            movie = torch.tensor(movie_index, dtype=torch.long)
            target = torch.tensor(train_matrix[user_index, movie_index], dtype=torch.float32)

            optimizer.zero_grad()
            prediction = model(user, movie)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

    return model
