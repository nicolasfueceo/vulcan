"""
Sparse Vector Recommender for FUEGO Benchmark System

This module implements a recommender system that uses Qdrant's sparse vector capabilities
for efficient collaborative filtering, following the approach described in Qdrant documentation.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict
import pickle

# Import base recommender and Qdrant manager
from models.baselines.base_recommender import QdrantRecommender
from utils.qdrant_manager import QdrantManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QdrantSparseVectorRecommender(QdrantRecommender):
    """
    Sparse Vector based Collaborative Filtering recommender using Qdrant.

    This recommender uses sparse vectors to represent user ratings and leverages
    Qdrant's sparse vector search capabilities for efficient recommendation generation.
    This approach is based on the Qdrant documentation for recommendation systems.
    """

    def __init__(
        self,
        name: str = "QdrantSparseVector",
        description: str = "Sparse Vector Collaborative Filtering with Qdrant",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        use_local: bool = True,
        normalize_ratings: bool = True,
    ):
        """
        Initialize the QdrantSparseVector recommender.

        Args:
            name (str, optional): Name of the recommender. Defaults to "QdrantSparseVector".
            description (str, optional): Description of the recommender.
                                        Defaults to "Sparse Vector Collaborative Filtering with Qdrant".
            qdrant_host (str, optional): Qdrant server host. Defaults to "localhost".
            qdrant_port (int, optional): Qdrant server port. Defaults to 6333.
            use_local (bool, optional): Whether to use local Qdrant instance. Defaults to True.
            normalize_ratings (bool, optional): Whether to normalize ratings. Defaults to True.
        """
        super().__init__(name, description)

        # Initialize Qdrant manager
        self.qdrant = QdrantManager(
            host=qdrant_host, port=qdrant_port
            )

        self.normalize_ratings = normalize_ratings

        # Initialize data
        self.user_items = defaultdict(
            set
        )  # Dictionary mapping user_id to set of items they've interacted with
        self.items_data = {}  # Dictionary storing item metadata
        self.rating_mean = 0.0  # Mean rating (for normalization)
        self.rating_std = 1.0  # Rating standard deviation (for normalization)

        # Collection name
        self.collection_name = f"{name}_collection"

    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the recommender model by creating sparse vectors and storing them in Qdrant.

        Args:
            training_data (Dict[str, Any]): Training data containing:
                - ratings (pd.DataFrame): User-item ratings with columns 'userId', 'itemId', 'rating'
                - items (pd.DataFrame, optional): Item metadata
        """
        logger.info(f"Training {self.name} recommender")

        ratings_df = training_data.get("ratings")
        items_df = training_data.get("items")

        if ratings_df is None:
            raise ValueError("Ratings data is required for training")

        # Store user-item interactions for filtering seen items
        for _, row in ratings_df.iterrows():
            user_id = row["userId"]
            item_id = row["itemId"]
            self.user_items[user_id].add(item_id)

        # Store item data if available
        if items_df is not None:
            self.items_data = items_df.set_index("itemId").to_dict("index")

        # Normalize ratings if requested
        if self.normalize_ratings:
            self.rating_mean = ratings_df["rating"].mean()
            self.rating_std = ratings_df["rating"].std()
            ratings_df["normalized_rating"] = (
                ratings_df["rating"] - self.rating_mean
            ) / self.rating_std
        else:
            ratings_df["normalized_rating"] = ratings_df["rating"]

        # Create sparse vectors for each user
        user_sparse_vectors = defaultdict(lambda: {"values": [], "indices": []})

        for row in ratings_df.itertuples():
            user_id = row.userId
            item_id = row.itemId
            rating = row.normalized_rating

            user_sparse_vectors[user_id]["values"].append(float(rating))
            user_sparse_vectors[user_id]["indices"].append(int(item_id))

        # Create Qdrant collection for sparse vectors
        logger.info("Creating Qdrant collection for sparse vectors")

        # Delete collection if it already exists
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)

        # Create collection with sparse vector configuration
        self.qdrant.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "sparse": {}  # No need to specify dimension for sparse vectors
            },
        )

        # Upload sparse vectors to Qdrant
        logger.info("Uploading sparse vectors to Qdrant")

        points = []
        for user_id, sparse_vector in user_sparse_vectors.items():
            points.append(
                {
                    "id": user_id,
                    "vector": {
                        "sparse": {
                            "indices": sparse_vector["indices"],
                            "values": sparse_vector["values"],
                        }
                    },
                    "payload": {"user_id": user_id},
                }
            )

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant.add_points(
                collection_name=self.collection_name,
                points=batch,
                batch_size=batch_size,
            )

        self._is_trained = True
        logger.info(
            f"{self.name} recommender trained successfully with {len(user_sparse_vectors)} users"
        )

    def get_user_vector(self, user_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get the sparse vector representation of a user.

        Args:
            user_id (Union[int, str]): ID of the user.

        Returns:
            Dict[str, Any]: Sparse vector representation.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        try:
            # Get point from Qdrant
            point = self.qdrant.get_point(
                collection_name=self.collection_name, point_id=user_id
            )

            if point is None:
                logger.warning(f"User {user_id} not found in Qdrant")
                return None

            return point.get("vector", {}).get("sparse", {})
        except Exception as e:
            logger.error(f"Error getting user vector: {e}")
            return None

    def get_item_vector(self, item_id: Union[int, str]) -> List[float]:
        """
        Get the vector representation of an item.

        Note: In this sparse vector approach, items don't have their own vectors.
        This method is implemented to satisfy the interface but returns None.

        Args:
            item_id (Union[int, str]): ID of the item.

        Returns:
            List[float]: Vector representation (None in this case).
        """
        logger.warning(
            "Items don't have their own vectors in the sparse vector approach"
        )
        return None

    def update_vectors(
        self, user_id: Union[int, str], item_id: Union[int, str], rating: float
    ) -> None:
        """
        Update vector representations based on a new rating.

        Args:
            user_id (Union[int, str]): ID of the user.
            item_id (Union[int, str]): ID of the item.
            rating (float): Rating given by the user.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        try:
            # Get current user vector
            user_vector = self.get_user_vector(user_id)

            if user_vector is None:
                logger.warning(
                    f"User {user_id} not found in Qdrant, creating new vector"
                )
                user_vector = {"indices": [], "values": []}

            # Normalize rating if needed
            if self.normalize_ratings:
                normalized_rating = (rating - self.rating_mean) / self.rating_std
            else:
                normalized_rating = rating

            # Check if item is already in the vector
            if "indices" in user_vector and "values" in user_vector:
                try:
                    item_index = user_vector["indices"].index(item_id)
                    # Update existing rating
                    user_vector["values"][item_index] = normalized_rating
                except ValueError:
                    # Add new rating
                    user_vector["indices"].append(item_id)
                    user_vector["values"].append(normalized_rating)

                # Update vector in Qdrant
                self.qdrant.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        {
                            "id": user_id,
                            "vector": {
                                "sparse": {
                                    "indices": user_vector["indices"],
                                    "values": user_vector["values"],
                                }
                            },
                            "payload": {"user_id": user_id},
                        }
                    ],
                )

                # Update user_items
                self.user_items[user_id].add(item_id)

                logger.info(f"Updated vector for user {user_id}")
            else:
                logger.error(f"Invalid user vector format for user {user_id}")
        except Exception as e:
            logger.error(f"Error updating vector: {e}")

    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_already_liked: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user using Qdrant sparse vector similarity search.

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

        # Get user vector
        user_vector = self.get_user_vector(user_id)

        if user_vector is None:
            logger.warning(
                f"User {user_id} not found, using popularity-based recommendations"
            )
            # Fall back to popularity-based recommendations
            item_counts = {
                item_id: len(users) for item_id, users in self.user_items.items()
            }
            sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

            # Filter out items the user has already interacted with if requested
            if filter_already_liked:
                user_items = self.user_items.get(user_id, set())
                sorted_items = [
                    (item_id, count)
                    for item_id, count in sorted_items
                    if item_id not in user_items
                ]

            # Get top-n items
            top_items = sorted_items[:n]

            # Prepare recommendations with metadata
            recommendations = []
            for item_id, count in top_items:
                item_data = self.items_data.get(item_id, {})
                recommendations.append(
                    {
                        "item_id": item_id,
                        "score": count / max(item_counts.values())
                        if item_counts
                        else 0,
                        "metadata": item_data,
                    }
                )

            return recommendations

        # Get items the user has already interacted with
        user_items = self.user_items.get(user_id, set())

        # Search for similar users in Qdrant
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector={"sparse": user_vector},
            limit=20,  # Get more similar users to ensure we have enough recommendations
        )

        # Aggregate item scores from similar users
        item_scores = {}

        for result in search_results:
            similar_user_id = result.get("payload", {}).get("user_id")
            similarity = result.get("score", 0)

            # Skip the user itself
            if similar_user_id == user_id:
                continue

            # Get similar user's vector
            similar_user_vector = result.get("vector", {}).get("sparse", {})

            if (
                not similar_user_vector
                or "indices" not in similar_user_vector
                or "values" not in similar_user_vector
            ):
                continue

            # Add items from similar user to recommendations
            for item_idx, item_id in enumerate(similar_user_vector["indices"]):
                # Skip if the user has already interacted with this item and we're filtering
                if filter_already_liked and item_id in user_items:
                    continue

                # Get the similar user's rating for this item
                rating = similar_user_vector["values"][item_idx]

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
        Predict the rating for a user-item pair using similar users' ratings.

        Args:
            user_id (Union[int, str]): ID of the user.
            item_id (Union[int, str]): ID of the item.

        Returns:
            float: Predicted rating.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        # Get user vector
        user_vector = self.get_user_vector(user_id)

        if user_vector is None:
            logger.warning(f"User {user_id} not found in Qdrant")
            return self.rating_mean

        # Search for similar users in Qdrant
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector={"sparse": user_vector},
            limit=20,  # Get more similar users for better prediction
        )

        # Calculate predicted rating based on similar users' ratings
        numerator = 0.0
        denominator = 0.0

        for result in search_results:
            similar_user_id = result.get("payload", {}).get("user_id")
            similarity = result.get("score", 0)

            # Skip the user itself
            if similar_user_id == user_id:
                continue

            # Get similar user's vector
            similar_user_vector = result.get("vector", {}).get("sparse", {})

            if (
                not similar_user_vector
                or "indices" not in similar_user_vector
                or "values" not in similar_user_vector
            ):
                continue

            # Check if the similar user has rated this item
            try:
                item_index = similar_user_vector["indices"].index(item_id)
                rating = similar_user_vector["values"][item_index]

                numerator += similarity * rating
                denominator += similarity
            except ValueError:
                # Similar user hasn't rated this item
                continue

        # Return predicted rating
        if denominator == 0:
            return self.rating_mean

        # Denormalize the prediction if needed
        prediction = numerator / denominator
        if self.normalize_ratings:
            prediction = prediction * self.rating_std + self.rating_mean

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
            "normalize_ratings": self.normalize_ratings,
            "rating_mean": self.rating_mean,
            "rating_std": self.rating_std,
            "user_items": dict(self.user_items),
            "items_data": self.items_data,
            "collection_name": self.collection_name,
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

        self.normalize_ratings = model_data["normalize_ratings"]
        self.rating_mean = model_data["rating_mean"]
        self.rating_std = model_data["rating_std"]
        self.user_items = defaultdict(set, model_data["user_items"])
        self.items_data = model_data["items_data"]
        self.collection_name = model_data["collection_name"]
        self._is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {path}")
