"""
Base Recommender Interface for FUEGO Benchmark System

This module defines the base interface for all recommender systems in the FUEGO benchmark.
"""

import abc
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseRecommender(abc.ABC):
    """Abstract base class for all recommender systems in the FUEGO benchmark."""

    def __init__(self, name: str, description: str):
        """
        Initialize the recommender.

        Args:
            name (str): Name of the recommender.
            description (str): Description of the recommender.
        """
        self.name = name
        self.description = description
        self._is_trained = False

    @abc.abstractmethod
    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the recommender model.

        Args:
            training_data (Dict[str, Any]): Training data for the model.
        """
        pass

    @abc.abstractmethod
    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_already_liked: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user.

        Args:
            user_id (Union[int, str]): ID of the user to generate recommendations for.
            n (int, optional): Number of recommendations to generate. Defaults to 10.
            filter_already_liked (bool, optional): Whether to filter out items the user has already liked. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: List of recommended items with metadata.
        """
        pass

    @abc.abstractmethod
    def predict(self, user_id: Union[int, str], item_id: Union[int, str]) -> float:
        """
        Predict the rating or score for a user-item pair.

        Args:
            user_id (Union[int, str]): ID of the user.
            item_id (Union[int, str]): ID of the item.

        Returns:
            float: Predicted rating or score.
        """
        pass

    @property
    def is_trained(self) -> bool:
        """
        Check if the recommender is trained.

        Returns:
            bool: True if the recommender is trained, False otherwise.
        """
        return self._is_trained

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path (str): Path to save the model to.
        """
        raise NotImplementedError("This recommender does not support saving models.")

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path (str): Path to load the model from.
        """
        raise NotImplementedError("This recommender does not support loading models.")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dict[str, Any]: Model information.
        """
        return {
            "name": self.name,
            "description": self.description,
            "is_trained": self.is_trained,
        }


class CollaborativeFilteringRecommender(BaseRecommender):
    """Interface for collaborative filtering recommenders."""

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def get_similar_items(
        self, item_id: Union[int, str], n: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Get similar items for a given item.

        Args:
            item_id (Union[int, str]): ID of the item.
            n (int, optional): Number of similar items to return. Defaults to 10.

        Returns:
            List[Tuple[Union[int, str], float]]: List of similar items with similarity scores.
        """
        pass


class ContentBasedRecommender(BaseRecommender):
    """Interface for content-based recommenders."""

    @abc.abstractmethod
    def get_item_features(self, item_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get features for a given item.

        Args:
            item_id (Union[int, str]): ID of the item.

        Returns:
            Dict[str, Any]: Item features.
        """
        pass

    @abc.abstractmethod
    def get_user_profile(self, user_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get profile for a given user.

        Args:
            user_id (Union[int, str]): ID of the user.

        Returns:
            Dict[str, Any]: User profile.
        """
        pass

    @abc.abstractmethod
    def update_user_profile(
        self, user_id: Union[int, str], item_id: Union[int, str], rating: float
    ) -> None:
        """
        Update a user's profile based on a new rating.

        Args:
            user_id (Union[int, str]): ID of the user.
            item_id (Union[int, str]): ID of the item.
            rating (float): Rating given by the user.
        """
        pass


class QdrantRecommender(BaseRecommender):
    """Interface for recommenders that use Qdrant vector database."""

    @abc.abstractmethod
    def get_user_vector(self, user_id: Union[int, str]) -> List[float]:
        """
        Get the vector representation of a user.

        Args:
            user_id (Union[int, str]): ID of the user.

        Returns:
            List[float]: Vector representation.
        """
        pass

    @abc.abstractmethod
    def get_item_vector(self, item_id: Union[int, str]) -> List[float]:
        """
        Get the vector representation of an item.

        Args:
            item_id (Union[int, str]): ID of the item.

        Returns:
            List[float]: Vector representation.
        """
        pass

    @abc.abstractmethod
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
        pass
