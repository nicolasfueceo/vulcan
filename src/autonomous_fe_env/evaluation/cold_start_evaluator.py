"""
Cold start evaluator for VULCAN feature engineering system.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add baselines to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines"))

from baselines import BaselineEvaluator

from ..feature import FeatureDefinition
from .feature_evaluator import FeatureEvaluator

logger = logging.getLogger(__name__)


class ColdStartEvaluator:
    """
    Cold start evaluator for VULCAN feature engineering.

    Evaluates feature engineering performance specifically for cold start
    scenarios and compares against baseline models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cold start evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.baseline_evaluator = None
        self.feature_evaluator = None
        self.baseline_results = {}

        # Cold start specific configuration
        self.cold_start_config = config.get("cold_start", {})
        self.min_user_interactions = self.cold_start_config.get(
            "min_user_interactions", 5
        )
        self.test_ratio = self.cold_start_config.get("test_ratio", 0.2)

    def setup(self, dal, db_path: str) -> None:
        """
        Set up the evaluator with data access.

        Args:
            dal: Data access layer
            db_path: Path to database
        """
        self.dal = dal
        self.db_path = db_path

        # Initialize baseline evaluator
        self.baseline_evaluator = BaselineEvaluator(db_path, self.config)

        # Initialize feature evaluator
        self.feature_evaluator = FeatureEvaluator(self.config)
        self.feature_evaluator.setup(dal)

        logger.info("Cold start evaluator setup complete")

    def run_baseline_evaluation(self, sample_size: int = 5000) -> Dict[str, float]:
        """
        Run baseline model evaluation.

        Args:
            sample_size: Number of samples to use for evaluation

        Returns:
            Dictionary with baseline results
        """
        if not self.baseline_evaluator:
            raise ValueError("Evaluator not set up. Call setup() first.")

        logger.info("Running baseline evaluation...")

        # Load data and fit baselines
        self.baseline_evaluator.load_data(sample_size=sample_size)
        self.baseline_evaluator.fit_baselines()

        # Evaluate baselines
        self.baseline_results = self.baseline_evaluator.evaluate_baselines(k=10)

        logger.info("Baseline evaluation complete")
        return self.baseline_results

    def evaluate_feature_set_for_cold_start(
        self,
        features: List[FeatureDefinition],
        cold_start_users: Optional[List[int]] = None,
    ) -> float:
        """
        Evaluate a feature set specifically for cold start performance.

        Args:
            features: List of feature definitions to evaluate
            cold_start_users: Optional list of cold start user IDs

        Returns:
            Cold start performance score
        """
        if not self.feature_evaluator:
            raise ValueError("Evaluator not set up. Call setup() first.")

        try:
            # Get cold start data
            if cold_start_users is None:
                cold_start_data = self._get_cold_start_data()
            else:
                cold_start_data = self.dal.get_data_for_users(cold_start_users)

            if cold_start_data.empty:
                logger.warning("No cold start data available")
                return -float("inf")

            # Apply features to cold start data
            feature_data, errors = self._apply_features_to_data(
                features, cold_start_data
            )

            if errors:
                logger.warning(f"Feature application errors: {len(errors)}")

            # Evaluate clustering performance on cold start users
            score = self._evaluate_cold_start_clustering(feature_data, features)

            logger.info(f"Cold start evaluation score: {score:.4f}")
            return score

        except Exception as e:
            logger.error(f"Error in cold start evaluation: {e}")
            return -float("inf")

    def _get_cold_start_data(self) -> pd.DataFrame:
        """Get data for cold start evaluation."""
        # Get users with limited interaction history
        query = f"""
        SELECT user_id, COUNT(*) as interaction_count
        FROM reviews
        GROUP BY user_id
        HAVING interaction_count <= {self.min_user_interactions}
        ORDER BY RANDOM()
        LIMIT 1000
        """

        cold_start_users_df = self.dal.execute_query(query)

        if cold_start_users_df.empty:
            logger.warning("No cold start users found")
            return pd.DataFrame()

        cold_start_user_ids = cold_start_users_df["user_id"].tolist()

        # Get their interaction data
        cold_start_data = self.dal.get_data_for_users(cold_start_user_ids)

        logger.info(
            f"Retrieved {len(cold_start_data)} interactions for {len(cold_start_user_ids)} cold start users"
        )
        return cold_start_data

    def _apply_features_to_data(
        self, features: List[FeatureDefinition], data: pd.DataFrame
    ) -> tuple:
        """Apply features to data (simplified version)."""
        # Use the feature evaluator's method
        return self.feature_evaluator._apply_features_to_data(features, data)

    def _evaluate_cold_start_clustering(
        self, feature_data: pd.DataFrame, features: List[FeatureDefinition]
    ) -> float:
        """
        Evaluate clustering performance for cold start scenarios.

        Args:
            feature_data: Data with applied features
            features: List of features used

        Returns:
            Clustering performance score
        """
        try:
            # Extract feature columns
            feature_columns = [f.output_column_name for f in features]
            available_columns = [
                col for col in feature_columns if col in feature_data.columns
            ]

            if not available_columns:
                # Use baseline columns if no features available
                baseline_columns = ["rating", "user_id", "book_id"]
                available_columns = [
                    col for col in baseline_columns if col in feature_data.columns
                ]

            if not available_columns:
                logger.warning("No columns available for clustering")
                return -1.0

            clustering_data = feature_data[available_columns]

            # Handle missing values
            clustering_data = clustering_data.fillna(clustering_data.mean())
            clustering_data = clustering_data.replace([np.inf, -np.inf], np.nan)
            clustering_data = clustering_data.fillna(0)

            # Check if we have enough data
            n_clusters = min(5, len(clustering_data) // 10)  # Adaptive cluster count
            if n_clusters < 2:
                logger.warning("Not enough data for clustering")
                return -1.0

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(clustering_data)

            # Check if we have multiple clusters
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                logger.warning("Clustering produced only one cluster")
                return -1.0

            # Calculate silhouette score
            score = silhouette_score(clustering_data, cluster_labels)

            # Adjust score for cold start context (penalize if too few features help)
            if len(available_columns) == len(feature_columns):
                # All features were successfully applied
                score *= 1.1  # Bonus for successful feature application
            elif len(available_columns) < len(feature_columns):
                # Some features failed
                penalty = 0.9 ** (len(feature_columns) - len(available_columns))
                score *= penalty

            return float(score)

        except Exception as e:
            logger.error(f"Error in cold start clustering evaluation: {e}")
            return -float("inf")

    def compare_with_baselines(self, vulcan_score: float) -> Dict[str, Any]:
        """
        Compare VULCAN performance with baseline models.

        Args:
            vulcan_score: VULCAN feature engineering score

        Returns:
            Comparison results
        """
        if not self.baseline_results:
            logger.warning(
                "No baseline results available. Running baseline evaluation..."
            )
            self.run_baseline_evaluation()

        if not self.baseline_evaluator:
            raise ValueError("Baseline evaluator not initialized")

        # Use baseline evaluator's comparison method
        comparison = self.baseline_evaluator.compare_with_vulcan(vulcan_score)

        # Add cold start specific analysis
        comparison["cold_start_analysis"] = {
            "evaluation_type": "cold_start_clustering",
            "metric": "silhouette_score",
            "description": "Clustering performance on users with limited interaction history",
        }

        return comparison

    def generate_visualization(
        self, vulcan_score: float, save_path: Optional[str] = None
    ) -> None:
        """
        Generate visualization comparing VULCAN with baselines.

        Args:
            vulcan_score: VULCAN score
            save_path: Optional path to save the plot
        """
        if not self.baseline_evaluator:
            raise ValueError("Baseline evaluator not initialized")

        self.baseline_evaluator.plot_results(vulcan_score, save_path)

    def print_detailed_report(self, vulcan_score: float) -> None:
        """
        Print a detailed evaluation report.

        Args:
            vulcan_score: VULCAN feature engineering score
        """
        print("\n" + "=" * 60)
        print("VULCAN COLD START EVALUATION REPORT")
        print("=" * 60)

        # Print baseline summary
        if self.baseline_evaluator:
            self.baseline_evaluator.print_summary(vulcan_score)

        # Print comparison analysis
        if self.baseline_results:
            comparison = self.compare_with_baselines(vulcan_score)

            print("\nCOLD START ANALYSIS:")
            print(
                f"Evaluation Type: {comparison['cold_start_analysis']['evaluation_type']}"
            )
            print(f"Metric: {comparison['cold_start_analysis']['metric']}")
            print(f"Description: {comparison['cold_start_analysis']['description']}")

            print("\nVULCAN Performance:")
            print(f"  Score: {vulcan_score:.4f}")

            best_baseline = max(self.baseline_results.items(), key=lambda x: x[1])
            print(f"  Best Baseline: {best_baseline[0]} ({best_baseline[1]:.4f})")

            if best_baseline[1] > 0:
                improvement = (
                    (vulcan_score - best_baseline[1]) / best_baseline[1]
                ) * 100
                print(f"  Improvement: {improvement:+.1f}%")

        print("=" * 60)
