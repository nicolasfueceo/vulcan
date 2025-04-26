"""
Matrix Factorization Recommender for FUEGO Benchmark System

This module implements a collaborative filtering recommender using matrix factorization
with Singular Value Decomposition (SVD).
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Import base recommender
from models.baselines.base_recommender import CollaborativeFilteringRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MatrixFactorizationRecommender(CollaborativeFilteringRecommender):
    """
    Matrix Factorization based Collaborative Filtering recommender.

    This recommender uses Singular Value Decomposition (SVD) to decompose the user-item
    matrix into user and item latent factor matrices, which can then be used to predict
    ratings and generate recommendations.
    """

    def __init__(
        self,
        name: str = "MatrixFactorization",
        description: str = "Matrix Factorization based Collaborative Filtering",
        n_factors: int = 50,
        regularization: float = 0.1,
        iterations: int = 20,
    ):
        """
        Initialize the Matrix Factorization recommender.

        Args:
            name (str, optional): Name of the recommender. Defaults to "MatrixFactorization".
            description (str, optional): Description of the recommender.
                                        Defaults to "Matrix Factorization based Collaborative Filtering".
            n_factors (int, optional): Number of latent factors. Defaults to 50.
            regularization (float, optional): Regularization parameter. Defaults to 0.1.
            iterations (int, optional): Number of iterations for SVD. Defaults to 20.
        """
        super().__init__(name, description)
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        # Model parameters
        self.user_factors = None  # User latent factors
        self.item_factors = None  # Item latent factors
        self.user_biases = None  # User biases
        self.item_biases = None  # Item biases
        self.global_bias = None  # Global bias (mean rating)

        # Mappings between original IDs and matrix indices
        self.users = []  # List of user IDs
        self.items = []  # List of item IDs
        self.user_map = {}  # Map from user ID to matrix index
        self.item_map = {}  # Map from item ID to matrix index
        self.reverse_user_map = {}  # Map from matrix index to user ID
        self.reverse_item_map = {}  # Map from matrix index to item ID

        # User-item interactions for filtering seen items
        self.user_items = defaultdict(
            set
        )  # Dictionary mapping user_id to set of items they've interacted with

        # Item metadata
        self.items_data = {}  # Dictionary storing item metadata

    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the recommender model using matrix factorization.

        Args:
            training_data (Dict[str, Any]): Training data containing:
                - ratings (pd.DataFrame): User-item ratings with columns 'userId', 'itemId', 'rating'
                - items (pd.DataFrame, optional): Item metadata
        """
        logger.info("Training Matrix Factorization recommender")

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

        # Store user-item interactions for filtering seen items during recommendation
        for _, row in ratings_df.iterrows():
            user_id = row["userId"]
            item_id = row["itemId"]
            self.user_items[user_id].add(item_id)

        # Store item data if available for including metadata in recommendations
        if items_df is not None:
            self.items_data = items_df.set_index("itemId").to_dict("index")

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

        user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(self.users), len(self.items))
        )

        # Calculate global bias (mean rating)
        self.global_bias = np.mean(data)

        # Calculate user and item biases
        # This is a simplified implementation; in practice, you might want to use
        # more sophisticated methods to calculate biases
        self.user_biases = np.zeros(len(self.users))
        self.item_biases = np.zeros(len(self.items))

        # Perform matrix factorization using SVD
        logger.info(f"Performing SVD with {self.n_factors} factors")

        # Center the ratings by subtracting the global bias
        centered_matrix = user_item_matrix.copy()
        centered_matrix.data = centered_matrix.data - self.global_bias

        # Perform SVD
        U, sigma, Vt = svds(centered_matrix, k=self.n_factors)

        # Convert to the correct orientation
        self.user_factors = U
        self.item_factors = Vt.T

        self._is_trained = True
        logger.info(
            f"Matrix Factorization recommender trained successfully with {len(self.users)} users and {len(self.items)} items"
        )

    def get_similar_users(
        self, user_id: Union[int, str], n: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Get similar users for a given user based on latent factors.

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
        user_vector = self.user_factors[user_idx]

        # Calculate similarity with all other users
        similarities = []
        for other_idx in range(len(self.users)):
            if other_idx == user_idx:
                continue

            other_vector = self.user_factors[other_idx]

            # Calculate cosine similarity
            dot_product = np.dot(user_vector, other_vector)
            norm_product = np.linalg.norm(user_vector) * np.linalg.norm(other_vector)

            similarity = dot_product / norm_product if norm_product != 0 else 0

            similarities.append((other_idx, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Map indices back to user IDs
        similar_users = [
            (self.reverse_user_map[idx], float(sim)) for idx, sim in similarities[:n]
        ]

        return similar_users

    def get_similar_items(
        self, item_id: Union[int, str], n: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Get similar items for a given item based on latent factors.

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
        item_vector = self.item_factors[item_idx]

        # Calculate similarity with all other items
        similarities = []
        for other_idx in range(len(self.items)):
            if other_idx == item_idx:
                continue

            other_vector = self.item_factors[other_idx]

            # Calculate cosine similarity
            dot_product = np.dot(item_vector, other_vector)
            norm_product = np.linalg.norm(item_vector) * np.linalg.norm(other_vector)

            similarity = dot_product / norm_product if norm_product != 0 else 0

            similarities.append((other_idx, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Map indices back to item IDs
        similar_items = [
            (self.reverse_item_map[idx], float(sim)) for idx, sim in similarities[:n]
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
        Generate recommendations for a user using matrix factorization.

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
            # Calculate item popularity as the sum of ratings
            item_scores = np.sum(self.item_factors, axis=1)
            top_items_indices = np.argsort(item_scores)[::-1][:n]

            recommendations = []
            for idx in top_items_indices:
                item_id = self.reverse_item_map[idx]
                score = float(item_scores[idx])
                item_data = self.items_data.get(item_id, {})

                recommendations.append(
                    {"item_id": item_id, "score": score, "metadata": item_data}
                )

            return recommendations

        user_idx = self.user_map[user_id]
        user_vector = self.user_factors[user_idx]

        # Get items the user has already interacted with
        user_items = self.user_items.get(user_id, set())

        # Calculate predicted ratings for all items
        item_scores = []
        for item_idx in range(len(self.items)):
            item_id = self.reverse_item_map[item_idx]

            # Skip if the user has already interacted with this item and we're filtering
            if filter_already_liked and item_id in user_items:
                continue

            # Calculate predicted rating
            item_vector = self.item_factors[item_idx]
            score = self.global_bias + np.dot(user_vector, item_vector)

            item_scores.append((item_id, score))

        # Sort items by predicted rating
        sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)

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
        Predict the rating for a user-item pair using matrix factorization.

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
            return self.global_bias

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        # Calculate predicted rating
        user_vector = self.user_factors[user_idx]
        item_vector = self.item_factors[item_idx]

        prediction = self.global_bias + np.dot(user_vector, item_vector)

        # Clip prediction to valid rating range (e.g., 1-5)
        prediction = max(1.0, min(5.0, prediction))

        return float(prediction)

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path (str): Path to save the model to.
        """
        model_data = {
            "n_factors": self.n_factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "users": self.users,
            "items": self.items,
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "user_items": dict(self.user_items),
            "items_data": self.items_data,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "global_bias": self.global_bias,
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

        self.n_factors = model_data["n_factors"]
        self.regularization = model_data["regularization"]
        self.iterations = model_data["iterations"]
        self.users = model_data["users"]
        self.items = model_data["items"]
        self.user_map = model_data["user_map"]
        self.item_map = model_data["item_map"]
        self.reverse_user_map = model_data["reverse_user_map"]
        self.reverse_item_map = model_data["reverse_item_map"]
        self.user_items = defaultdict(set, model_data["user_items"])
        self.items_data = model_data["items_data"]
        self.user_factors = model_data["user_factors"]
        self.item_factors = model_data["item_factors"]
        self.user_biases = model_data["user_biases"]
        self.item_biases = model_data["item_biases"]
        self.global_bias = model_data["global_bias"]
        self._is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {path}")
