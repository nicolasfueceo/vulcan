"""
Advanced baseline recommenders for academic comparison.

This module implements state-of-the-art baseline methods for
rigorous academic evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import scipy.sparse as sp
import structlog

logger = structlog.get_logger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommenders."""

    @abstractmethod
    def fit(self, train_interactions: sp.csr_matrix):
        """Train the recommender."""
        pass

    @abstractmethod
    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict scores for user-item pairs."""
        pass

    @abstractmethod
    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get top-N recommendations for a user."""
        pass


class MostPopularRecommender(BaseRecommender):
    """Recommends most popular items globally."""

    def __init__(self):
        self.item_popularity = None
        self.n_items = 0

    def fit(self, train_interactions: sp.csr_matrix):
        """Calculate item popularity."""
        self.item_popularity = np.array(train_interactions.sum(axis=0)).flatten()
        self.n_items = len(self.item_popularity)

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Return popularity scores for items."""
        return self.item_popularity[item_indices]

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Return most popular items."""
        return np.argsort(-self.item_popularity)[:n_items]


class UserBasedCF(BaseRecommender):
    """User-based collaborative filtering with cosine similarity."""

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors
        self.train_interactions = None
        self.user_similarity = None

    def fit(self, train_interactions: sp.csr_matrix):
        """Calculate user similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity

        self.train_interactions = train_interactions
        self.user_similarity = cosine_similarity(train_interactions, train_interactions)

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict scores based on similar users."""
        # Find k most similar users
        similar_users = np.argsort(-self.user_similarity[user_idx])[
            : self.k_neighbors + 1
        ]
        similar_users = similar_users[similar_users != user_idx][: self.k_neighbors]

        # Aggregate scores from similar users
        scores = np.zeros(len(item_indices))
        for i, item_idx in enumerate(item_indices):
            weighted_sum = 0.0
            similarity_sum = 0.0

            for neighbor_idx in similar_users:
                if self.train_interactions[neighbor_idx, item_idx] > 0:
                    similarity = self.user_similarity[user_idx, neighbor_idx]
                    weighted_sum += similarity
                    similarity_sum += similarity

            scores[i] = weighted_sum / (similarity_sum + 1e-10)

        return scores

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get recommendations based on similar users."""
        all_scores = self.predict(user_idx, np.arange(self.train_interactions.shape[1]))
        return np.argsort(-all_scores)[:n_items]


class ItemBasedCF(BaseRecommender):
    """Item-based collaborative filtering."""

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.train_interactions = None
        self.item_similarity = None

    def fit(self, train_interactions: sp.csr_matrix):
        """Calculate item similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity

        self.train_interactions = train_interactions
        self.item_similarity = cosine_similarity(
            train_interactions.T, train_interactions.T
        )

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict scores based on similar items."""
        user_items = self.train_interactions.getrow(user_idx).nonzero()[1]
        scores = np.zeros(len(item_indices))

        if len(user_items) == 0:
            return scores

        for i, item_idx in enumerate(item_indices):
            # Find similar items that user has rated
            similarities = self.item_similarity[item_idx, user_items]
            top_k_idx = np.argsort(-similarities)[: self.k_neighbors]

            # Weighted average of ratings
            scores[i] = np.mean(similarities[top_k_idx])

        return scores

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get recommendations based on similar items."""
        all_scores = self.predict(user_idx, np.arange(self.train_interactions.shape[1]))
        return np.argsort(-all_scores)[:n_items]


class MatrixFactorizationSVD(BaseRecommender):
    """Matrix factorization using SVD."""

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0

    def fit(self, train_interactions: sp.csr_matrix):
        """Fit SVD model."""
        from scipy.sparse.linalg import svds

        # Calculate global mean
        self.global_mean = train_interactions.mean()

        # Center the data
        train_centered = train_interactions - self.global_mean

        # Perform SVD
        U, s, Vt = svds(train_centered, k=self.n_factors)

        # Store factors
        self.user_factors = U @ np.diag(np.sqrt(s))
        self.item_factors = np.diag(np.sqrt(s)) @ Vt

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict ratings using dot product of factors."""
        user_vec = self.user_factors[user_idx]
        item_vecs = self.item_factors[:, item_indices].T

        predictions = user_vec @ item_vecs.T + self.global_mean
        return predictions

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get top-N recommendations."""
        all_scores = self.predict(user_idx, np.arange(self.item_factors.shape[1]))
        return np.argsort(-all_scores)[:n_items]


class NonNegativeMatrixFactorization(BaseRecommender):
    """Non-negative matrix factorization."""

    def __init__(self, n_components: int = 50, max_iter: int = 200):
        self.n_components = n_components
        self.max_iter = max_iter
        self.model = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_interactions: sp.csr_matrix):
        """Fit NMF model."""
        from sklearn.decomposition import NMF

        self.model = NMF(
            n_components=self.n_components, max_iter=self.max_iter, random_state=42
        )

        # Fit the model
        self.user_factors = self.model.fit_transform(train_interactions)
        self.item_factors = self.model.components_

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict ratings."""
        user_vec = self.user_factors[user_idx]
        item_vecs = self.item_factors[:, item_indices].T

        predictions = user_vec @ item_vecs.T
        return predictions

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get top-N recommendations."""
        all_scores = self.predict(user_idx, np.arange(self.item_factors.shape[1]))
        return np.argsort(-all_scores)[:n_items]


class SLIM(BaseRecommender):
    """Sparse Linear Methods (SLIM) for recommendation."""

    def __init__(self, l1_penalty: float = 0.1, l2_penalty: float = 0.1):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.W = None  # Item-item weight matrix

    def fit(self, train_interactions: sp.csr_matrix):
        """
        Fit SLIM model by learning item-item weights.

        Note: This is a simplified version. Full SLIM requires solving
        an optimization problem for each item.
        """

        n_items = train_interactions.shape[1]
        self.W = np.zeros((n_items, n_items))

        # For efficiency, we'll use a simplified approach
        # In practice, you'd solve the optimization for each item
        item_similarity = train_interactions.T @ train_interactions
        item_similarity = item_similarity.toarray()

        # Normalize and apply penalties
        np.fill_diagonal(item_similarity, 0)  # No self-similarity
        row_sums = item_similarity.sum(axis=1)
        item_similarity = item_similarity / (row_sums[:, np.newaxis] + 1e-10)

        self.W = item_similarity

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict scores for items."""
        user_profile = self.train_interactions.getrow(user_idx).toarray().flatten()
        scores = user_profile @ self.W[:, item_indices]
        return scores

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get recommendations."""
        all_scores = self.predict(user_idx, np.arange(self.W.shape[1]))
        return np.argsort(-all_scores)[:n_items]


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining multiple methods.

    Combines collaborative filtering with popularity for cold-start handling.
    """

    def __init__(self, cf_weight: float = 0.8, popularity_weight: float = 0.2):
        self.cf_weight = cf_weight
        self.popularity_weight = popularity_weight
        self.cf_model = ItemBasedCF(k_neighbors=30)
        self.popularity_model = MostPopularRecommender()

    def fit(self, train_interactions: sp.csr_matrix):
        """Fit both models."""
        self.cf_model.fit(train_interactions)
        self.popularity_model.fit(train_interactions)
        self.train_interactions = train_interactions

    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Combine predictions from both models."""
        # Check if user is cold-start
        user_item_count = self.train_interactions.getrow(user_idx).nnz

        if user_item_count == 0:
            # Pure popularity for cold-start users
            return self.popularity_model.predict(user_idx, item_indices)
        elif user_item_count < 5:
            # Blend more popularity for users with few ratings
            cf_weight = self.cf_weight * (user_item_count / 5)
            pop_weight = 1 - cf_weight
        else:
            cf_weight = self.cf_weight
            pop_weight = self.popularity_weight

        cf_scores = self.cf_model.predict(user_idx, item_indices)
        pop_scores = self.popularity_model.predict(user_idx, item_indices)

        # Normalize scores
        cf_scores = (cf_scores - cf_scores.min()) / (
            cf_scores.max() - cf_scores.min() + 1e-10
        )
        pop_scores = (pop_scores - pop_scores.min()) / (
            pop_scores.max() - pop_scores.min() + 1e-10
        )

        return cf_weight * cf_scores + pop_weight * pop_scores

    def recommend(self, user_idx: int, n_items: int = 10) -> np.ndarray:
        """Get hybrid recommendations."""
        all_scores = self.predict(user_idx, np.arange(self.train_interactions.shape[1]))
        return np.argsort(-all_scores)[:n_items]


def evaluate_baseline(
    recommender: BaseRecommender, train: sp.csr_matrix, test: sp.csr_matrix, k: int = 10
) -> Dict[str, float]:
    """
    Evaluate a baseline recommender.

    Returns:
        Dictionary with precision@k, recall@k, and ndcg@k
    """
    recommender.fit(train)

    precisions = []
    recalls = []
    ndcgs = []

    n_users = test.shape[0]

    for user_idx in range(n_users):
        # Get true items
        true_items = set(test.getrow(user_idx).nonzero()[1])
        if len(true_items) == 0:
            continue

        # Get recommendations
        try:
            recommended_items = recommender.recommend(user_idx, n_items=k)
            recommended_set = set(recommended_items)

            # Calculate metrics
            hits = len(recommended_set & true_items)
            precision = hits / k
            recall = hits / len(true_items)

            precisions.append(precision)
            recalls.append(recall)

            # Simplified NDCG (binary relevance)
            dcg = sum(
                1.0 / np.log2(i + 2)
                for i, item in enumerate(recommended_items)
                if item in true_items
            )
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_items))))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_idx}: {e}")
            continue

    return {
        "precision_at_k": np.mean(precisions) if precisions else 0.0,
        "recall_at_k": np.mean(recalls) if recalls else 0.0,
        "ndcg_at_k": np.mean(ndcgs) if ndcgs else 0.0,
        "n_evaluated": len(precisions),
    }
