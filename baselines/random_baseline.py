"""
Random baseline recommendation model.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base_baseline import BaseBaseline


class RandomBaseline(BaseBaseline):
    """
    Random baseline that recommends random books.

    This baseline provides a lower bound for recommendation performance
    by making completely random recommendations.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize random baseline.

        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__("Random")
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.all_books = None
        self.global_mean_rating = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the random baseline on training data.

        Args:
            train_data: Training data with columns [user_id, book_id, rating]
        """
        # Store all available books for random sampling
        self.all_books = train_data["book_id"].unique()
        self.global_mean_rating = train_data["rating"].mean()
        self.is_fitted = True

        print(f"Random baseline fitted on {len(train_data)} ratings")
        print(f"Found {len(self.all_books)} unique books")
        print(f"Global mean rating: {self.global_mean_rating:.3f}")

    def predict(
        self, user_ids: List[int], k: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate random recommendations for users.

        Args:
            user_ids: List of user IDs to generate recommendations for
            k: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (book_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        recommendations = {}

        for user_id in user_ids:
            # Randomly sample k books
            if len(self.all_books) >= k:
                selected_books = self.rng.choice(self.all_books, size=k, replace=False)
            else:
                selected_books = self.all_books

            # Assign random scores
            random_scores = self.rng.uniform(0, 1, size=len(selected_books))

            user_recs = [
                (int(book_id), float(score))
                for book_id, score in zip(selected_books, random_scores)
            ]

            # Sort by score (descending)
            user_recs.sort(key=lambda x: x[1], reverse=True)

            recommendations[user_id] = user_recs

        return recommendations

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Predict a random rating for a specific user-book pair.

        Args:
            user_id: User ID
            book_id: Book ID

        Returns:
            Random rating between 1 and 5
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Return a random rating between 1 and 5
        return float(self.rng.uniform(1, 5))

    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information."""
        info = super().get_model_info()

        info.update({"random_state": self.random_state, "algorithm": "Random sampling"})

        if self.is_fitted:
            info.update(
                {
                    "num_books": len(self.all_books),
                    "global_mean_rating": self.global_mean_rating,
                }
            )

        return info
