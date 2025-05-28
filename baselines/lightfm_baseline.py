"""
LightFM-based baseline recommendation model.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import coo_matrix

from .base_baseline import BaseBaseline


class LightFMBaseline(BaseBaseline):
    """
    LightFM-based collaborative filtering baseline.

    Uses the LightFM library for matrix factorization with WARP loss
    for implicit feedback recommendation.
    """

    def __init__(
        self,
        no_components: int = 30,
        learning_rate: float = 0.05,
        loss: str = "warp",
        max_sampled: int = 10,
        epochs: int = 20,
        random_state: int = 42,
    ):
        """
        Initialize LightFM baseline.

        Args:
            no_components: Number of latent factors
            learning_rate: Learning rate for optimization
            loss: Loss function ('warp', 'logistic', 'bpr')
            max_sampled: Maximum number of negative samples for WARP
            epochs: Number of training epochs
            random_state: Random seed
        """
        super().__init__("LightFM")

        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_sampled = max_sampled
        self.epochs = epochs
        self.random_state = random_state

        self.model = None
        self.user_mapping = None
        self.book_mapping = None
        self.reverse_user_mapping = None
        self.reverse_book_mapping = None
        self.interaction_matrix = None
        self.global_mean_rating = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the LightFM model on training data.

        Args:
            train_data: Training data with columns [user_id, book_id, rating]
        """
        print(f"Fitting LightFM model with {len(train_data)} interactions...")

        # Create user and item mappings
        unique_users = train_data["user_id"].unique()
        unique_books = train_data["book_id"].unique()

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.book_mapping = {book_id: idx for idx, book_id in enumerate(unique_books)}

        self.reverse_user_mapping = {
            idx: user_id for user_id, idx in self.user_mapping.items()
        }
        self.reverse_book_mapping = {
            idx: book_id for book_id, idx in self.book_mapping.items()
        }

        # Map IDs to indices
        train_data_mapped = train_data.copy()
        train_data_mapped["user_idx"] = train_data_mapped["user_id"].map(
            self.user_mapping
        )
        train_data_mapped["book_idx"] = train_data_mapped["book_id"].map(
            self.book_mapping
        )

        # Create interaction matrix
        # For LightFM, we'll use binary interactions (rating >= 4 is positive)
        train_data_mapped["interaction"] = (train_data_mapped["rating"] >= 4).astype(
            int
        )

        self.interaction_matrix = coo_matrix(
            (
                train_data_mapped["interaction"],
                (train_data_mapped["user_idx"], train_data_mapped["book_idx"]),
            ),
            shape=(len(unique_users), len(unique_books)),
        ).tocsr()

        # Initialize and train model
        self.model = LightFM(
            no_components=self.no_components,
            learning_rate=self.learning_rate,
            loss=self.loss,
            max_sampled=self.max_sampled,
            random_state=self.random_state,
        )

        # Fit the model
        self.model.fit(self.interaction_matrix, epochs=self.epochs, verbose=True)

        self.global_mean_rating = train_data["rating"].mean()
        self.is_fitted = True

        print("LightFM model fitted successfully!")
        print(f"Users: {len(unique_users)}, Books: {len(unique_books)}")
        print(f"Interactions: {self.interaction_matrix.nnz}")
        print(
            f"Sparsity: {1 - self.interaction_matrix.nnz / (len(unique_users) * len(unique_books)):.4f}"
        )

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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        recommendations = {}

        for user_id in user_ids:
            if user_id not in self.user_mapping:
                # Cold start user - return popular items or empty list
                recommendations[user_id] = []
                continue

            user_idx = self.user_mapping[user_id]

            # Get all book indices
            all_book_indices = np.arange(len(self.book_mapping))

            # Predict scores for all books
            scores = self.model.predict(user_idx, all_book_indices)

            # Get top-k recommendations
            top_indices = np.argsort(scores)[::-1][:k]

            user_recs = [
                (self.reverse_book_mapping[idx], float(scores[idx]))
                for idx in top_indices
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
            Predicted rating (scaled from LightFM score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if user_id not in self.user_mapping or book_id not in self.book_mapping:
            return self.global_mean_rating

        user_idx = self.user_mapping[user_id]
        book_idx = self.book_mapping[book_id]

        # Get LightFM score and scale to rating range
        score = self.model.predict(user_idx, book_idx)

        # Scale score to rating range (1-5)
        # LightFM scores are typically in range [-inf, inf], we'll use sigmoid scaling
        scaled_score = 1 / (1 + np.exp(-score))  # Sigmoid to [0, 1]
        rating = 1 + 4 * scaled_score  # Scale to [1, 5]

        return float(rating)

    def evaluate_on_test(
        self, test_data: pd.DataFrame, k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data with columns [user_id, book_id, rating]
            k: Number of recommendations for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Map test data
        test_data_mapped = test_data.copy()
        test_data_mapped = test_data_mapped[
            test_data_mapped["user_id"].isin(self.user_mapping.keys())
            & test_data_mapped["book_id"].isin(self.book_mapping.keys())
        ]

        if len(test_data_mapped) == 0:
            return {"precision@k": 0.0, "recall@k": 0.0}

        test_data_mapped["user_idx"] = test_data_mapped["user_id"].map(
            self.user_mapping
        )
        test_data_mapped["book_idx"] = test_data_mapped["book_id"].map(
            self.book_mapping
        )
        test_data_mapped["interaction"] = (test_data_mapped["rating"] >= 4).astype(int)

        # Create test interaction matrix
        test_interactions = coo_matrix(
            (
                test_data_mapped["interaction"],
                (test_data_mapped["user_idx"], test_data_mapped["book_idx"]),
            ),
            shape=self.interaction_matrix.shape,
        ).tocsr()

        # Calculate metrics
        precision = precision_at_k(self.model, test_interactions, k=k).mean()
        recall = recall_at_k(self.model, test_interactions, k=k).mean()

        return {"precision@k": float(precision), "recall@k": float(recall)}

    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information."""
        info = super().get_model_info()

        info.update(
            {
                "no_components": self.no_components,
                "learning_rate": self.learning_rate,
                "loss": self.loss,
                "epochs": self.epochs,
                "algorithm": "Matrix Factorization (LightFM)",
            }
        )

        if self.is_fitted:
            info.update(
                {
                    "num_users": len(self.user_mapping),
                    "num_books": len(self.book_mapping),
                    "num_interactions": self.interaction_matrix.nnz,
                    "sparsity": 1
                    - self.interaction_matrix.nnz
                    / (len(self.user_mapping) * len(self.book_mapping)),
                    "global_mean_rating": self.global_mean_rating,
                }
            )

        return info
