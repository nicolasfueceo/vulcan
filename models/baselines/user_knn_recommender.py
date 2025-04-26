"""
User-based K-Nearest Neighbors Recommender for FUEGO Benchmark System

This module implements a user-based collaborative filtering recommender using
the k-nearest neighbors algorithm.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Import base recommender
from models.baselines.base_recommender import CollaborativeFilteringRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UserKNNRecommender(CollaborativeFilteringRecommender):
    """
    User-based K-Nearest Neighbors collaborative filtering recommender.

    This recommender finds similar users and recommends items that similar users have liked.
    It works by representing each user as a vector of their ratings, computing similarity
    between users, and then generating recommendations based on what similar users liked.
    """

    def __init__(
        self,
        name: str = "UserKNN",
        description: str = "User-based K-Nearest Neighbors collaborative filtering",
        k: int = 20,
    ):
        """
        Initialize the UserKNN recommender.

        Args:
            name (str, optional): Name of the recommender. Defaults to "UserKNN".
            description (str, optional): Description of the recommender.
                                        Defaults to "User-based K-Nearest Neighbors collaborative filtering".
            k (int, optional): Number of neighbors to consider. Defaults to 20.
        """
        super().__init__(name, description)
        self.k = k  # Number of neighbors to consider
        self.user_item_matrix = None  # Sparse matrix of user-item ratings
        self.user_similarity = None  # Matrix of user-user similarities

        # Mappings between original IDs and matrix indices
        self.users = []  # List of user IDs
        self.items = []  # List of item IDs
        self.user_map = {}  # Map from user ID to matrix index
        self.item_map = {}  # Map from item ID to matrix index
        self.reverse_user_map = {}  # Map from matrix index to user ID
        self.reverse_item_map = {}  # Map from matrix index to item ID

        # Item metadata
        self.items_data = {}  # Dictionary storing item metadata

    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the recommender model by calculating user similarity.

        Args:
            training_data (Dict[str, Any]): Training data containing:
                - ratings (pd.DataFrame): User-item ratings with columns 'userId', 'itemId', 'rating'
                - items (pd.DataFrame, optional): Item metadata
        """
        logger.info("Training UserKNN recommender")

        ratings_df = training_data.get("ratings")
        items_df = training_data.get("items")

        if ratings_df is None:
            raise ValueError("Ratings data is required for training")

        # Create user and item mappings
        self.users = sorted(ratings_df["userId"].unique())
        self.items = sorted(ratings_df["itemId"].unique())

        self.user_map = {user: i for i, user in enumerate(self.users)}
        self.item_map = {item: i for i, item in enumerate(self.items)}

        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # Create user-item matrix
        rows = []
        cols = []
        data = []

        for _, row in ratings_df.iterrows():
            user_idx = self.user_map[row["userId"]]
            item_idx = self.item_map[row["itemId"]]
            rating = row["rating"]

            rows.append(user_idx)
            cols.append(item_idx)
            data.append(rating)

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(self.users), len(self.items))
        )

        # Calculate user similarity using cosine similarity
        # This is a computationally expensive operation for large datasets
        logger.info("Calculating user similarity matrix")
        self.user_similarity = cosine_similarity(self.user_item_matrix)

        # Store item data if available for including metadata in recommendations
        if items_df is not None:
            self.items_data = items_df.set_index("itemId").to_dict("index")

        self._is_trained = True
        logger.info(
            f"UserKNN recommender trained successfully with {len(self.users)} users and {len(self.items)} items"
        )

    def get_similar_users(
        self, user_id: Union[int, str], n: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Get similar users for a given user.

        Args:
            user_id (Union[int, str]): ID of the user.
            n (int, optional): Number of similar users to return. Defaults to 10.

        Returns:
            List[Tuple[Union[int, str], float]]: List of similar users with similarity scores.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        if user_id not in self.user_map:
            logger.warning(f"User {user_id} not found in training data")
            return []

        user_idx = self.user_map[user_id]

        # Get similarity scores for this user
        similarity_scores = self.user_similarity[user_idx]

        # Sort users by similarity (excluding the user itself)
        similar_users_indices = np.argsort(similarity_scores)[::-1][1 : n + 1]

        # Map indices back to user IDs
        similar_users = [
            (self.reverse_user_map[idx], float(similarity_scores[idx]))
            for idx in similar_users_indices
        ]

        return similar_users

    def get_similar_items(
        self, item_id: Union[int, str], n: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Get similar items for a given item based on user ratings.

        Args:
            item_id (Union[int, str]): ID of the item.
            n (int, optional): Number of similar items to return. Defaults to 10.

        Returns:
            List[Tuple[Union[int, str], float]]: List of similar items with similarity scores.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        if item_id not in self.item_map:
            logger.warning(f"Item {item_id} not found in training data")
            return []

        item_idx = self.item_map[item_id]

        # Get item vector (all user ratings for this item)
        item_vector = self.user_item_matrix[:, item_idx].toarray().flatten()

        # Calculate similarity with all other items
        item_similarities = []

        for other_idx in range(len(self.items)):
            if other_idx == item_idx:
                continue

            other_vector = self.user_item_matrix[:, other_idx].toarray().flatten()

            # Calculate cosine similarity
            dot_product = np.dot(item_vector, other_vector)
            norm_product = np.linalg.norm(item_vector) * np.linalg.norm(other_vector)

            similarity = dot_product / norm_product if norm_product != 0 else 0

            item_similarities.append((other_idx, similarity))

        # Sort by similarity
        item_similarities.sort(key=lambda x: x[1], reverse=True)

        # Map indices back to item IDs
        similar_items = [
            (self.reverse_item_map[idx], float(sim))
            for idx, sim in item_similarities[:n]
        ]

        return similar_items

    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_already_liked: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user based on similar users' preferences.

        Args:
            user_id (Union[int, str]): ID of the user to generate recommendations for.
            n (int, optional): Number of recommendations to generate. Defaults to 10.
            filter_already_liked (bool, optional): Whether to filter out items the user has already liked.
                                                  Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: List of recommended items with metadata.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        if user_id not in self.user_map:
            # Return popular items for new users
            logger.warning(
                f"User {user_id} not found in training data, using popularity-based recommendations"
            )
            # Calculate item popularity
            item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
            top_items_indices = np.argsort(item_popularity)[::-1][:n]

            recommendations = []
            for idx in top_items_indices:
                item_id = self.reverse_item_map[idx]
                popularity = float(item_popularity[idx])
                item_data = self.items_data.get(item_id, {})

                recommendations.append(
                    {"item_id": item_id, "score": popularity, "metadata": item_data}
                )

            return recommendations

        user_idx = self.user_map[user_id]

        # Get similar users
        similar_users = self.get_similar_users(user_id, self.k)

        if not similar_users:
            logger.warning(f"No similar users found for user {user_id}")
            return []

        # Get items the user has already interacted with
        user_items = set(self.user_item_matrix[user_idx].nonzero()[1])
        user_items = {self.reverse_item_map[idx] for idx in user_items}

        # Calculate item scores based on similar users' ratings
        item_scores = {}

        for similar_user_id, similarity in similar_users:
            similar_user_idx = self.user_map[similar_user_id]

            # Get items rated by the similar user
            similar_user_items = self.user_item_matrix[similar_user_idx].nonzero()[1]

            for item_idx in similar_user_items:
                item_id = self.reverse_item_map[item_idx]

                # Skip if the user has already interacted with this item and we're filtering
                if filter_already_liked and item_id in user_items:
                    continue

                # Get the similar user's rating for this item
                rating = self.user_item_matrix[similar_user_idx, item_idx]

                # Weight the rating by similarity
                if item_id not in item_scores:
                    item_scores[item_id] = 0

                item_scores[item_id] += similarity * rating

        # Sort items by score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-n items
        top_items = sorted_items[:n]

        # Prepare recommendations with metadata
        recommendations = []
        for item_id, score in top_items:
            item_data = self.items_data.get(item_id, {})
            recommendations.append(
                {"item_id": item_id, "score": float(score), "metadata": item_data}
            )

        return recommendations

    def predict(self, user_id: Union[int, str], item_id: Union[int, str]) -> float:
        """
        Predict the rating for a user-item pair based on similar users' ratings.

        Args:
            user_id (Union[int, str]): ID of the user.
            item_id (Union[int, str]): ID of the item.

        Returns:
            float: Predicted rating.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        if user_id not in self.user_map or item_id not in self.item_map:
            logger.warning(
                f"User {user_id} or item {item_id} not found in training data"
            )
            return 0.0

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        # Check if the user has already rated this item
        if self.user_item_matrix[user_idx, item_idx] != 0:
            return float(self.user_item_matrix[user_idx, item_idx])

        # Get similar users
        similar_users = self.get_similar_users(user_id, self.k)

        if not similar_users:
            logger.warning(f"No similar users found for user {user_id}")
            return 0.0

        # Calculate predicted rating based on similar users' ratings
        numerator = 0.0
        denominator = 0.0

        for similar_user_id, similarity in similar_users:
            similar_user_idx = self.user_map[similar_user_id]
            rating = self.user_item_matrix[similar_user_idx, item_idx]

            # Only consider users who have rated this item
            if rating != 0:
                numerator += similarity * rating
                denominator += similarity

        # Return predicted rating
        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path (str): Path to save the model to.
        """
        model_data = {
            "k": self.k,
            "users": self.users,
            "items": self.items,
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "items_data": self.items_data,
            "user_item_matrix": self.user_item_matrix,
            "user_similarity": self.user_similarity,
            "is_trained": self._is_trained,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path (str): Path to load the model from.
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.k = model_data["k"]
        self.users = model_data["users"]
        self.items = model_data["items"]
        self.user_map = model_data["user_map"]
        self.item_map = model_data["item_map"]
        self.reverse_user_map = model_data["reverse_user_map"]
        self.reverse_item_map = model_data["reverse_item_map"]
        self.items_data = model_data["items_data"]
        self.user_item_matrix = model_data["user_item_matrix"]
        self.user_similarity = model_data["user_similarity"]
        self._is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {path}")
