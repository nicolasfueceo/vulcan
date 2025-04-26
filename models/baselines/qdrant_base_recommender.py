"""
Qdrant-based Recommender for FUEGO Benchmark System

This module implements a recommender system that uses Qdrant vector database
for efficient similarity search based on the approach described in Qdrant documentation.
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

# Import base recommender and Qdrant manager
from models.baselines.base_recommender import QdrantRecommender
from utils.qdrant_manager import QdrantManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QdrantMatrixFactorizationRecommender(QdrantRecommender):
    """
    Matrix Factorization based Collaborative Filtering recommender using Qdrant.

    This recommender uses Singular Value Decomposition (SVD) to generate user and item
    embeddings, which are then stored in Qdrant for efficient similarity search.
    """

    def __init__(
        self,
        name: str = "QdrantMF",
        description: str = "Matrix Factorization with Qdrant",
        n_factors: int = 50,
        regularization: float = 0.1,
        iterations: int = 20,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        use_local: bool = True,
    ):
        """
        Initialize the QdrantMF recommender.

        Args:
            name (str, optional): Name of the recommender. Defaults to "QdrantMF".
            description (str, optional): Description of the recommender.
                                        Defaults to "Matrix Factorization with Qdrant".
            n_factors (int, optional): Number of latent factors. Defaults to 50.
            regularization (float, optional): Regularization parameter. Defaults to 0.1.
            iterations (int, optional): Number of iterations for SVD. Defaults to 20.
            qdrant_host (str, optional): Qdrant server host. Defaults to "localhost".
            qdrant_port (int, optional): Qdrant server port. Defaults to 6333.
            use_local (bool, optional): Whether to use local Qdrant instance. Defaults to True.
        """
        super().__init__(name, description)
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        # Initialize Qdrant manager
        self.qdrant = QdrantManager(
            host=qdrant_host,
            port=qdrant_port,
            use_local=use_local,
            vector_size=n_factors,
        )

        # Initialize model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None

        # Initialize mappings
        self.users = []
        self.items = []
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

        # Initialize data
        self.user_items = defaultdict(set)
        self.items_data = {}

        # Collection names
        self.user_collection = f"{name}_users"
        self.item_collection = f"{name}_items"

    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the recommender model using matrix factorization and store embeddings in Qdrant.

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

        # Create user and item mappings
        self.users = sorted(ratings_df["userId"].unique())
        self.items = sorted(ratings_df["itemId"].unique())

        self.user_map = {user: i for i, user in enumerate(self.users)}
        self.item_map = {item: i for i, item in enumerate(self.items)}

        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # Store user-item interactions for filtering seen items
        for _, row in ratings_df.iterrows():
            user_id = row["userId"]
            item_id = row["itemId"]
            self.user_items[user_id].add(item_id)

        # Store item data if available
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

        # Create Qdrant collections
        logger.info("Creating Qdrant collections")

        # Create user collection
        self.qdrant.create_collection(
            collection_name=self.user_collection, vector_size=self.n_factors
        )

        # Create item collection
        self.qdrant.create_collection(
            collection_name=self.item_collection, vector_size=self.n_factors
        )

        # Upload user vectors to Qdrant
        logger.info("Uploading user vectors to Qdrant")
        user_vectors = []
        user_ids = []
        user_payloads = []

        for user_idx, user_id in self.reverse_user_map.items():
            user_vector = self.user_factors[user_idx].tolist()
            user_vectors.append(user_vector)
            user_ids.append(user_id)

            # Create payload
            payload = {"user_id": user_id, "metadata": {}}
            user_payloads.append(payload)

        self.qdrant.upload_vectors(
            collection_name=self.user_collection,
            vectors=user_vectors,
            ids=user_ids,
            payloads=user_payloads,
        )

        # Upload item vectors to Qdrant
        logger.info("Uploading item vectors to Qdrant")
        item_vectors = []
        item_ids = []
        item_payloads = []

        for item_idx, item_id in self.reverse_item_map.items():
            item_vector = self.item_factors[item_idx].tolist()
            item_vectors.append(item_vector)
            item_ids.append(item_id)

            # Create payload
            payload = {"item_id": item_id, "metadata": self.items_data.get(item_id, {})}
            item_payloads.append(payload)

        self.qdrant.upload_vectors(
            collection_name=self.item_collection,
            vectors=item_vectors,
            ids=item_ids,
            payloads=item_payloads,
        )

        self._is_trained = True
        logger.info(f"{self.name} recommender trained successfully")

    def get_user_vector(self, user_id: Union[int, str]) -> List[float]:
        """
        Get the vector representation of a user.

        Args:
            user_id (Union[int, str]): ID of the user.

        Returns:
            List[float]: Vector representation.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        try:
            # Get point from Qdrant
            point = self.qdrant.get_point(
                collection_name=self.user_collection, point_id=user_id
            )

            if point is None:
                logger.warning(f"User {user_id} not found in Qdrant")
                return None

            return point.get("vector")
        except Exception as e:
            logger.error(f"Error getting user vector: {e}")
            return None

    def get_item_vector(self, item_id: Union[int, str]) -> List[float]:
        """
        Get the vector representation of an item.

        Args:
            item_id (Union[int, str]): ID of the item.

        Returns:
            List[float]: Vector representation.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        try:
            # Get point from Qdrant
            point = self.qdrant.get_point(
                collection_name=self.item_collection, point_id=item_id
            )

            if point is None:
                logger.warning(f"Item {item_id} not found in Qdrant")
                return None

            return point.get("vector")
        except Exception as e:
            logger.error(f"Error getting item vector: {e}")
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
        # This is a simplified implementation that doesn't actually update the vectors
        # In a real system, you would use online learning to update the vectors
        logger.warning("Vector update not implemented for QdrantMF recommender")

    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_already_liked: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user using Qdrant similarity search.

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

        # Search for similar items in Qdrant
        search_limit = n
        if filter_already_liked:
            # Request more items to account for filtering
            search_limit = n * 2

        search_results = self.qdrant.search(
            collection_name=self.item_collection,
            query_vector=user_vector,
            limit=search_limit,
        )

        # Filter and format results
        recommendations = []

        for result in search_results:
            item_id = result.get("payload", {}).get("item_id")
            score = result.get("score", 0)

            # Skip if the user has already interacted with this item and we're filtering
            if filter_already_liked and item_id in user_items:
                continue

            item_data = self.items_data.get(item_id, {})

            recommendations.append(
                {"item_id": item_id, "score": score, "metadata": item_data}
            )

            if len(recommendations) >= n:
                break

        return recommendations

    def predict(self, user_id: Union[int, str], item_id: Union[int, str]) -> float:
        """
        Predict the rating for a user-item pair using vector dot product.

        Args:
            user_id (Union[int, str]): ID of the user.
            item_id (Union[int, str]): ID of the item.

        Returns:
            float: Predicted rating.
        """
        if not self.is_trained:
            raise RuntimeError("Recommender is not trained yet")

        # Get user and item vectors
        user_vector = self.get_user_vector(user_id)
        item_vector = self.get_item_vector(item_id)

        if user_vector is None or item_vector is None:
            logger.warning(f"User {user_id} or item {item_id} not found in Qdrant")
            return self.global_bias

        # Calculate predicted rating
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
            "global_bias": self.global_bias,
            "user_collection": self.user_collection,
            "item_collection": self.item_collection,
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
        self.global_bias = model_data["global_bias"]
        self.user_collection = model_data["user_collection"]
        self.item_collection = model_data["item_collection"]
        self._is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {path}")
