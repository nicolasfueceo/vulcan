"""
Base evaluator class for VULCAN feature evaluation.

This module provides the common functionality for all evaluators,
following DRY (Don't Repeat Yourself) principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from vulcan.types import (
    DataContext,
    FeatureEvaluation,
    FeatureMetrics,
    FeatureSet,
    FeatureValue,
    VulcanConfig,
)
from vulcan.utils import get_vulcan_logger

logger = structlog.get_logger(__name__)


class BaseFeatureEvaluator(ABC):
    """Abstract base class for feature evaluators with common functionality."""

    def __init__(self, config: VulcanConfig):
        """
        Initialize base evaluator.

        Args:
            config: VULCAN configuration
        """
        self.config = config
        self.logger = get_vulcan_logger(self.__class__.__name__)
        self.scaler = StandardScaler()

    async def initialize(self) -> bool:
        """Initialize the evaluator."""
        try:
            self.logger.info(f"{self.__class__.__name__} initialized")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to initialize {self.__class__.__name__}", error=str(e)
            )
            return False

    @abstractmethod
    async def evaluate_feature_set(
        self,
        feature_set: FeatureSet,
        feature_results: Dict[str, List[FeatureValue]],
        data_context: DataContext,
        iteration: int,
    ) -> FeatureEvaluation:
        """
        Evaluate a feature set. Must be implemented by subclasses.

        Args:
            feature_set: Feature set to evaluate
            feature_results: Computed feature values
            data_context: Data context
            iteration: Current iteration number

        Returns:
            Feature evaluation result
        """
        pass

    def _build_feature_matrix(
        self, feature_results: Dict[str, List[FeatureValue]]
    ) -> pd.DataFrame:
        """
        Build feature matrix from feature results.

        This is a common method used by all evaluators.

        Args:
            feature_results: Dictionary of feature results

        Returns:
            Feature matrix DataFrame with users as rows and features as columns
        """
        if not feature_results:
            return pd.DataFrame()

        # Collect all user IDs
        all_user_ids = set()
        for feature_values in feature_results.values():
            for fv in feature_values:
                all_user_ids.add(fv.user_id)

        if not all_user_ids:
            return pd.DataFrame()

        # Build feature matrix
        feature_data = {}

        for feature_name, feature_values in feature_results.items():
            # Create user_id to value mapping
            value_map = {
                fv.user_id: fv.value for fv in feature_values if fv.value is not None
            }

            # Create feature column
            feature_column = []
            for user_id in sorted(all_user_ids):
                value = value_map.get(user_id, 0.0)  # Default to 0.0 for missing values

                # Convert to numeric if possible
                if isinstance(value, (int, float)):
                    feature_column.append(float(value))
                elif isinstance(value, str):
                    try:
                        feature_column.append(float(value))
                    except ValueError:
                        # Use hash of string as numeric value
                        feature_column.append(float(hash(value) % 1000))
                else:
                    feature_column.append(0.0)

            feature_data[feature_name] = feature_column

        if not feature_data:
            return pd.DataFrame()

        df = pd.DataFrame(feature_data, index=sorted(all_user_ids))

        # Remove columns with all NaN
        df = df.dropna(axis=1, how="all")

        # Handle constant columns
        if len(df.columns) > 0:
            variances = df.var()
            constant_cols = variances[variances == 0]

            if len(constant_cols) > 0:
                self.logger.debug(
                    "Found constant columns",
                    constant_columns=list(constant_cols.index),
                    total_columns=len(df.columns),
                )

                # Only remove if not all columns are constant
                if len(constant_cols) < len(df.columns):
                    df = df.drop(columns=constant_cols.index)
                    self.logger.debug(
                        "Removed constant columns",
                        removed=len(constant_cols),
                        remaining=len(df.columns),
                    )

        return df

    def _compute_clustering_metrics(
        self,
        feature_df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        return_labels: bool = False,
    ) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Compute standard clustering metrics.

        Args:
            feature_df: Feature matrix DataFrame
            n_clusters: Number of clusters (if None, will be determined)
            return_labels: Whether to return cluster labels

        Returns:
            Tuple of (metrics dict, cluster labels if requested)
        """
        X_scaled = self.scaler.fit_transform(feature_df)
        n_samples = X_scaled.shape[0]

        # Handle edge cases
        if n_samples < 3:
            metrics = {
                "silhouette_score": 0.0,
                "calinski_harabasz": 0.0,
                "davies_bouldin": 1.0,
            }
            labels = np.zeros(n_samples) if return_labels else None
            return metrics, labels

        # Determine number of clusters
        if n_clusters is None:
            if self.config.evaluation.clustering_config.n_clusters is not None:
                n_clusters = self.config.evaluation.clustering_config.n_clusters
            else:
                cluster_range = self.config.evaluation.clustering_config.cluster_range
                n_clusters = min(max(cluster_range), n_samples - 1)
        n_clusters = max(min(n_clusters, n_samples - 1), 2)

        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.evaluation.random_state,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Compute metrics
        metrics = {
            "silhouette_score": float(silhouette_score(X_scaled, cluster_labels)),
            "calinski_harabasz": float(
                calinski_harabasz_score(X_scaled, cluster_labels)
            ),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, cluster_labels)),
        }

        if return_labels:
            return metrics, cluster_labels
        return metrics, None

    def _create_default_evaluation(
        self,
        feature_set: FeatureSet,
        data_context: DataContext,
        iteration: int,
        evaluation_time: float,
        **kwargs,
    ) -> FeatureEvaluation:
        """
        Create default evaluation for failed cases.

        Args:
            feature_set: Feature set
            data_context: Data context
            iteration: Iteration number
            evaluation_time: Evaluation time
            **kwargs: Additional metrics to include

        Returns:
            Default feature evaluation with zero/worst scores
        """
        # Base metrics
        base_metrics = {
            "silhouette_score": 0.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": 1.0,
            "extraction_time": evaluation_time,
            "missing_rate": 1.0,
            "unique_rate": 0.0,
        }

        # Add any additional metrics from kwargs
        base_metrics.update(kwargs)

        return FeatureEvaluation(
            feature_set=feature_set,
            metrics=FeatureMetrics(**base_metrics),
            overall_score=0.0,
            fold_id=data_context.fold_id,
            iteration=iteration,
            evaluation_time=evaluation_time,
        )

    def _normalize_clustering_metrics(
        self, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize clustering metrics to 0-1 scale (higher is better).

        Args:
            metrics: Raw clustering metrics

        Returns:
            Normalized metrics
        """
        normalized = {}

        # Silhouette score: -1 to 1 -> 0 to 1
        normalized["silhouette"] = (metrics.get("silhouette_score", 0) + 1) / 2

        # Calinski-Harabasz: 0 to inf -> 0 to 1 (cap at 1000)
        normalized["calinski"] = min(metrics.get("calinski_harabasz", 0) / 1000, 1.0)

        # Davies-Bouldin: 0 to inf -> 1 to 0 (lower is better, cap at 5)
        davies = metrics.get("davies_bouldin", 1.0)
        normalized["davies"] = max(0, 1 - min(davies / 5, 1.0))

        return normalized

    def _compute_data_quality_metrics(
        self, feature_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute data quality metrics for the feature matrix.

        Args:
            feature_df: Feature matrix DataFrame

        Returns:
            Dictionary with data quality metrics
        """
        total_cells = feature_df.shape[0] * feature_df.shape[1]

        metrics = {
            "missing_rate": feature_df.isnull().sum().sum() / total_cells
            if total_cells > 0
            else 0.0,
            "unique_rate": feature_df.nunique().mean() / len(feature_df)
            if len(feature_df) > 0
            else 0.0,
            "n_features": feature_df.shape[1],
            "n_samples": feature_df.shape[0],
        }

        return metrics
