"""
Abstract base class for baseline recommendation models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pandas as pd


class BaseBaseline(ABC):
    """Abstract base class for baseline recommendation models."""

    def __init__(self, name: str):
        """
        Initialize the baseline model.

        Args:
            name: Name of the baseline model
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the baseline model on training data.

        Args:
            train_data: Training data with columns [user_id, book_id, rating]
        """
        pass

    @abstractmethod
    def predict(
        self, user_ids: List[int], k: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for users.

        Args:
            user_ids: List of user IDs to generate recommendations for
            k: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (book_id, score) tuples
        """
        pass

    @abstractmethod
    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Predict rating for a specific user-book pair.

        Args:
            user_id: User ID
            book_id: Book ID

        Returns:
            Predicted rating
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {"name": self.name, "is_fitted": self.is_fitted, "type": "baseline"}
