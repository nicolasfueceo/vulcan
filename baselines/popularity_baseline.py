"""
Popularity-based baseline recommendation model.
"""

from typing import Dict, List, Tuple

import pandas as pd

from .base_baseline import BaseBaseline


class PopularityBaseline(BaseBaseline):
    """
    Popularity-based baseline that recommends the most popular books.

    This baseline recommends books based on their overall popularity
    (number of ratings and average rating).
    """

    def __init__(self):
        super().__init__("Popularity")
        self.book_popularity = None
        self.global_mean_rating = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the popularity baseline on training data.

        Args:
            train_data: Training data with columns [user_id, book_id, rating]
        """
        # Calculate book popularity metrics
        book_stats = (
            train_data.groupby("book_id")
            .agg({"rating": ["count", "mean", "std"]})
            .round(4)
        )

        book_stats.columns = ["rating_count", "rating_mean", "rating_std"]
        book_stats = book_stats.reset_index()

        # Fill NaN std with 0 (for books with only one rating)
        book_stats["rating_std"] = book_stats["rating_std"].fillna(0)

        # Calculate popularity score (weighted by count and rating)
        # Use a weighted combination of rating and popularity
        min_ratings = 5  # Minimum ratings to be considered

        # Bayesian average to handle books with few ratings
        C = book_stats["rating_count"].mean()  # Average number of ratings
        m = train_data["rating"].mean()  # Global mean rating

        book_stats["popularity_score"] = (
            book_stats["rating_count"] * book_stats["rating_mean"] + C * m
        ) / (book_stats["rating_count"] + C)

        # Sort by popularity score
        self.book_popularity = book_stats.sort_values(
            "popularity_score", ascending=False
        ).reset_index(drop=True)

        self.global_mean_rating = train_data["rating"].mean()
        self.is_fitted = True

        print(f"Popularity baseline fitted on {len(train_data)} ratings")
        print(f"Found {len(self.book_popularity)} unique books")
        print(f"Global mean rating: {self.global_mean_rating:.3f}")

    def predict(
        self, user_ids: List[int], k: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for users based on popularity.

        Args:
            user_ids: List of user IDs to generate recommendations for
            k: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (book_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        recommendations = {}

        # Get top-k most popular books
        top_books = self.book_popularity.head(k)

        for user_id in user_ids:
            # For popularity baseline, all users get the same recommendations
            user_recs = [
                (int(row["book_id"]), float(row["popularity_score"]))
                for _, row in top_books.iterrows()
            ]
            recommendations[user_id] = user_recs

        return recommendations

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Predict rating for a specific user-book pair.

        Args:
            user_id: User ID
            book_id: Book ID

        Returns:
            Predicted rating (book's average rating or global mean)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Find book in popularity data
        book_data = self.book_popularity[self.book_popularity["book_id"] == book_id]

        if len(book_data) > 0:
            return float(book_data.iloc[0]["rating_mean"])
        else:
            # Return global mean for unknown books
            return self.global_mean_rating

    def get_top_books(self, k: int = 20) -> pd.DataFrame:
        """
        Get the top-k most popular books.

        Args:
            k: Number of top books to return

        Returns:
            DataFrame with top books and their statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting top books")

        return self.book_popularity.head(k)

    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information."""
        info = super().get_model_info()

        if self.is_fitted:
            info.update(
                {
                    "num_books": len(self.book_popularity),
                    "global_mean_rating": self.global_mean_rating,
                    "top_book_score": float(
                        self.book_popularity.iloc[0]["popularity_score"]
                    ),
                    "algorithm": "Bayesian average popularity",
                }
            )

        return info
