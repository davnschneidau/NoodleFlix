import os
import pymongo
import numpy as np

def load_data():
    db_url = os.environ.get("DATABASE_URL")
    client = pymongo.MongoClient(db_url)
    db = client.get_default_database()

    users = list(db.User.find())
    accounts = list(db.Account.find())
    movies = list(db.Movie.find())

    return users, accounts, movies

