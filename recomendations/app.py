import os
from flask import Flask, request, jsonify
from my_recommendation_package.model import train_recommendation_model, get_recommendations

app = Flask(__name__)

@app.route('/recommendations', methods=['POST'])
def recommend_movies():
    user_data = request.json
    user_id = user_data.get("user_id")
    recommendations = get_recommendations(user_id)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run()
