"""
Feature evaluator for assessing the performance of feature sets.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..data import BaseDAL
from ..feature import FeatureDefinition
from ..sandbox import CodeSandbox

logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """
    Evaluates feature sets for their effectiveness in improving clustering
    for recommendation systems.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_config = config.get("evaluation", {})

        # Evaluation parameters
        self.n_clusters = self.evaluation_config.get("n_clusters", 5)
        self.random_state = self.evaluation_config.get("random_state", 42)
        self.sample_size = self.evaluation_config.get("sample_size", 1000)
        self.metric = self.evaluation_config.get("metric", "silhouette")

        # Components
        self.dal: Optional[BaseDAL] = None
        self.sandbox = CodeSandbox(config.get("sandbox", {}))

        # Caching
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.evaluation_cache: Dict[str, float] = {}

    def setup(self, dal: BaseDAL) -> None:
        """
        Set up the evaluator with a data access layer.

        Args:
            dal: Data access layer instance
        """
        self.dal = dal

    def evaluate_feature_set(self, features: List[FeatureDefinition]) -> float:
        """
        Evaluate a set of features for their clustering performance.

        Args:
            features: List of feature definitions to evaluate

        Returns:
            Evaluation score (higher is better)
        """
        if not self.dal:
            raise ValueError("Data access layer not set up. Call setup() first.")

        try:
            # Create cache key for this feature set
            cache_key = self._create_cache_key(features)

            if cache_key in self.evaluation_cache:
                logger.info(f"Using cached evaluation for feature set: {cache_key}")
                return self.evaluation_cache[cache_key]

            # Get evaluation data
            eval_data = self.dal.get_data_for_evaluation("train")

            if eval_data.empty:
                logger.warning("No evaluation data available")
                return -float("inf")

            # Sample data if it's too large
            if len(eval_data) > self.sample_size:
                eval_data = eval_data.sample(
                    n=self.sample_size, random_state=self.random_state
                )

            # Apply features to the data
            feature_data, errors = self._apply_features_to_data(features, eval_data)

            if errors:
                logger.warning(
                    f"Errors during feature application: {len(errors)} errors"
                )
                for error in errors[:5]:  # Log first 5 errors
                    logger.warning(f"Feature error: {error}")

            # Extract feature columns for clustering
            feature_columns = [f.output_column_name for f in features]

            if not feature_columns:
                # No features - use baseline columns
                baseline_columns = self._get_baseline_columns(feature_data)
                clustering_data = feature_data[baseline_columns]
            else:
                # Check if all feature columns exist
                available_columns = [
                    col for col in feature_columns if col in feature_data.columns
                ]
                if not available_columns:
                    logger.warning("No feature columns available for clustering")
                    return -float("inf")

                clustering_data = feature_data[available_columns]

            # Evaluate clustering performance
            score = self._evaluate_clustering(clustering_data)

            # Cache the result
            self.evaluation_cache[cache_key] = score

            logger.info(f"Evaluated feature set with score: {score:.4f}")
            return score

        except Exception as e:
            logger.error(f"Error evaluating feature set: {str(e)}")
            logger.error(traceback.format_exc())
            return -float("inf")

    def _apply_features_to_data(
        self, features: List[FeatureDefinition], data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply feature definitions to data.

        Args:
            features: List of feature definitions
            data: Input data

        Returns:
            Tuple of (modified data, list of errors)
        """
        data_copy = data.copy()
        all_errors = []

        for feature in features:
            try:
                # Check if required input columns exist
                missing_cols = [
                    col
                    for col in feature.required_input_columns
                    if col not in data_copy.columns
                ]

                if missing_cols:
                    error_msg = f"Missing required columns for feature '{feature.name}': {missing_cols}"
                    all_errors.append(error_msg)
                    continue

                # Apply feature to each row
                feature_values = []
                feature_errors = []

                for idx, row in data_copy.iterrows():
                    try:
                        # Convert row to dictionary
                        instance_data = row.to_dict()

                        # Execute feature code
                        result = self.sandbox.execute_code(
                            feature.code,
                            {"df": instance_data, "current_review_data": instance_data},
                            feature.get_function_name(),
                        )

                        if result["success"]:
                            feature_values.append(result["result"])
                        else:
                            feature_values.append(feature.default_value)
                            feature_errors.append(f"Row {idx}: {result['error']}")

                    except Exception as e:
                        feature_values.append(feature.default_value)
                        feature_errors.append(f"Row {idx}: {str(e)}")

                # Add feature column to data
                data_copy[feature.output_column_name] = feature_values

                if feature_errors:
                    all_errors.extend(feature_errors[:5])  # Limit error reporting

                logger.info(
                    f"Applied feature '{feature.name}' with {len(feature_errors)} errors"
                )

            except Exception as e:
                error_msg = f"Failed to apply feature '{feature.name}': {str(e)}"
                all_errors.append(error_msg)
                logger.error(error_msg)

        return data_copy, all_errors

    def _evaluate_clustering(self, data: pd.DataFrame) -> float:
        """
        Evaluate clustering performance on the given data.

        Args:
            data: Data to cluster

        Returns:
            Clustering score (higher is better)
        """
        try:
            # Handle missing values
            data_clean = data.fillna(data.mean())

            # Handle infinite values
            data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.fillna(0)

            # Check if we have enough data points
            if len(data_clean) < self.n_clusters:
                logger.warning(
                    f"Not enough data points ({len(data_clean)}) for {self.n_clusters} clusters"
                )
                return -1.0

            # Check if we have any features
            if data_clean.shape[1] == 0:
                logger.warning("No features available for clustering")
                return -1.0

            # Standardize features
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_clean)

            # Check for constant features (zero variance)
            if np.any(np.var(data_scaled, axis=0) == 0):
                logger.warning("Some features have zero variance")
                # Remove constant features
                non_constant_mask = np.var(data_scaled, axis=0) > 1e-10
                if not np.any(non_constant_mask):
                    logger.warning("All features are constant")
                    return -1.0
                data_scaled = data_scaled[:, non_constant_mask]

            # Perform clustering
            kmeans = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
            cluster_labels = kmeans.fit_predict(data_scaled)

            # Check if we have multiple clusters
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                logger.warning("Clustering produced only one cluster")
                return -1.0

            # Calculate clustering metric
            if self.metric == "silhouette":
                score = silhouette_score(data_scaled, cluster_labels)
            elif self.metric == "inertia":
                score = -kmeans.inertia_  # Negative because lower inertia is better
            else:
                logger.warning(f"Unknown metric: {self.metric}, using silhouette")
                score = silhouette_score(data_scaled, cluster_labels)

            return float(score)

        except Exception as e:
            logger.error(f"Error in clustering evaluation: {str(e)}")
            return -float("inf")

    def _get_baseline_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get baseline columns for clustering when no features are available.

        Args:
            data: Input data

        Returns:
            List of column names to use for baseline clustering
        """
        # Common numeric columns that might be useful for clustering
        potential_columns = ["rating", "user_id", "book_id"]

        available_columns = [
            col
            for col in potential_columns
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
        ]

        if not available_columns:
            # Fall back to any numeric columns
            available_columns = [
                col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])
            ]

        return available_columns[:5]  # Limit to first 5 columns

    def _create_cache_key(self, features: List[FeatureDefinition]) -> str:
        """
        Create a cache key for a feature set.

        Args:
            features: List of feature definitions

        Returns:
            String cache key
        """
        if not features:
            return "baseline"

        # Sort feature names for consistent caching
        feature_names = sorted([f.name for f in features])
        return "|".join(feature_names)

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        self.feature_cache.clear()
        logger.info("Evaluation cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "evaluation_cache_size": len(self.evaluation_cache),
            "feature_cache_size": len(self.feature_cache),
        }
