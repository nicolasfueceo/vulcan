"""
Baseline evaluator for cold start recommendation performance.
"""

import logging
import sqlite3
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lightfm_baseline import LightFMBaseline
from .popularity_baseline import PopularityBaseline
from .random_baseline import RandomBaseline

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluator for baseline recommendation models."""

    def __init__(self, db_path: str, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = config
        self.baselines = {
            "random": RandomBaseline(),
            "popularity": PopularityBaseline(),
            "lightfm": LightFMBaseline(epochs=5),  # Reduced for speed
        }
        self.results = {}
        self.train_data = None
        self.test_data = None

    def load_data(self, sample_size: int = 5000) -> None:
        """Load and split data."""
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT user_id, book_id, rating
        FROM reviews
        WHERE rating IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """

        all_data = pd.read_sql_query(query, conn)
        conn.close()

        # Split 80/20
        split_idx = int(0.8 * len(all_data))
        self.train_data = all_data[:split_idx]
        self.test_data = all_data[split_idx:]

        logger.info(f"Loaded {len(all_data)} samples")

    def fit_baselines(self) -> None:
        """Fit all models."""
        for name, model in self.baselines.items():
            logger.info(f"Fitting {name}...")
            try:
                model.fit(self.train_data)
            except Exception as e:
                logger.error(f"Error fitting {name}: {e}")

    def evaluate_baselines(self, k: int = 10) -> Dict[str, float]:
        """Evaluate models and return precision@k scores."""
        results = {}
        test_users = self.test_data["user_id"].unique()[:50]  # Sample

        for name, model in self.baselines.items():
            if not model.is_fitted:
                continue

            try:
                recs = model.predict(test_users.tolist(), k=k)
                precision = self._calculate_precision(recs, self.test_data, k)
                results[name] = precision
                logger.info(f"{name}: {precision:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")

        self.results = results
        return results

    def _calculate_precision(self, recommendations, test_data, k):
        """Calculate precision@k."""
        precisions = []

        for user_id, recs in recommendations.items():
            user_test = test_data[test_data["user_id"] == user_id]
            liked_books = set(user_test[user_test["rating"] >= 4]["book_id"])

            if len(recs) == 0:
                continue

            rec_books = set([book_id for book_id, _ in recs[:k]])
            if len(rec_books) > 0:
                precision = len(rec_books.intersection(liked_books)) / len(rec_books)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def compare_with_vulcan(self, vulcan_score: float) -> Dict[str, Any]:
        """Compare VULCAN with baselines."""
        comparison = {
            "vulcan_score": vulcan_score,
            "baseline_scores": self.results.copy(),
            "improvements": {},
            "ranking": [],
        }

        # Calculate improvements
        for name, score in self.results.items():
            if score > 0:
                improvement = ((vulcan_score - score) / score) * 100
            else:
                improvement = float("inf") if vulcan_score > 0 else 0
            comparison["improvements"][name] = improvement

        # Create ranking
        all_scores = {"VULCAN": vulcan_score, **self.results}
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        comparison["ranking"] = sorted_scores

        return comparison

    def plot_results(self, vulcan_score: float, save_path: Optional[str] = None):
        """Plot comparison results."""
        all_scores = {"VULCAN": vulcan_score, **self.results}
        names = list(all_scores.keys())
        scores = list(all_scores.values())
        colors = ["red" if name == "VULCAN" else "blue" for name in names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, scores, color=colors, alpha=0.7)
        plt.title("Model Performance Comparison")
        plt.ylabel("Precision@10")
        plt.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def print_summary(self, vulcan_score: float):
        """Print evaluation summary."""
        print("=" * 50)
        print("BASELINE EVALUATION SUMMARY")
        print("=" * 50)

        comparison = self.compare_with_vulcan(vulcan_score)

        print(f"\nVULCAN Score: {vulcan_score:.4f}")
        print("\nBaseline Scores:")
        for name, score in self.results.items():
            print(f"  {name}: {score:.4f}")

        print("\nImprovements over baselines:")
        for name, improvement in comparison["improvements"].items():
            print(f"  vs {name}: {improvement:+.1f}%")

        print("\nRanking:")
        for i, (name, score) in enumerate(comparison["ranking"], 1):
            print(f"  {i}. {name}: {score:.4f}")

        print("=" * 50)
