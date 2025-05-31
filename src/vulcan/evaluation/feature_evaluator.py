"""Feature evaluation engine for VULCAN system."""

import time
from typing import Dict, List

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


class FeatureEvaluator:
    """Evaluates feature sets using clustering metrics."""

    def __init__(self, config: VulcanConfig) -> None:
        """Initialize feature evaluator.

        Args:
            config: VULCAN configuration.
        """
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.scaler = StandardScaler()

    async def initialize(self) -> bool:
        """Initialize the feature evaluator.

        Returns:
            True if initialization successful.
        """
        try:
            self.logger.info("Feature evaluator initialized")
            return True

        except Exception as e:
            self.logger.error("Failed to initialize feature evaluator", error=str(e))
            return False

    async def evaluate_feature_set(
        self,
        feature_set: FeatureSet,
        feature_results: Dict[str, List[FeatureValue]],
        data_context: DataContext,
        iteration: int,
    ) -> FeatureEvaluation:
        """Evaluate a feature set using clustering metrics.

        Args:
            feature_set: Feature set to evaluate.
            feature_results: Computed feature values.
            data_context: Data context.
            iteration: Current iteration number.

        Returns:
            Feature evaluation result.
        """
        start_time = time.time()

        try:
            # Convert feature results to DataFrame
            feature_df = self._build_feature_matrix(feature_results)

            if feature_df.empty or len(feature_df.columns) == 0:
                # No valid features to evaluate
                return self._create_default_evaluation(
                    feature_set, data_context, iteration, time.time() - start_time
                )

            # Compute clustering metrics
            metrics = await self._compute_clustering_metrics(feature_df)

            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)

            evaluation_time = time.time() - start_time

            evaluation = FeatureEvaluation(
                feature_set=feature_set,
                metrics=metrics,
                overall_score=overall_score,
                fold_id=data_context.fold_id,
                iteration=iteration,
                evaluation_time=evaluation_time,
            )

            self.logger.info(
                "Feature set evaluated",
                feature_count=len(feature_set.features),
                overall_score=overall_score,
                silhouette_score=metrics.silhouette_score,
                evaluation_time=evaluation_time,
            )

            return evaluation

        except Exception as e:
            evaluation_time = time.time() - start_time
            self.logger.error(
                "Feature evaluation failed",
                error=str(e),
                evaluation_time=evaluation_time,
            )

            # Return default evaluation with low score
            return self._create_default_evaluation(
                feature_set, data_context, iteration, evaluation_time
            )

    def _build_feature_matrix(
        self, feature_results: Dict[str, List[FeatureValue]]
    ) -> pd.DataFrame:
        """Build feature matrix from feature results.

        Args:
            feature_results: Dictionary of feature results.

        Returns:
            Feature matrix DataFrame.
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
                    # Try to convert string to float
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

        # Remove columns with all NaN or constant values
        df = df.dropna(axis=1, how="all")

        # Check for constant columns but be less aggressive about removing them
        variances = df.var()
        constant_cols = variances[variances <= 1e-6]

        # Log information about constant columns
        if len(constant_cols) > 0:
            self.logger.warning(
                "Found constant columns in feature matrix",
                constant_columns=list(constant_cols.index),
                variances=variances.to_dict(),
                feature_matrix_shape=df.shape,
            )

            # Only remove columns if ALL columns are constant (to avoid empty matrix)
            if len(constant_cols) == len(df.columns):
                self.logger.warning(
                    "All columns are constant - keeping one column to avoid empty matrix"
                )
                # Keep the first column to avoid empty matrix
                df = df.iloc[:, :1]
            else:
                # Remove only truly constant columns (variance exactly 0)
                truly_constant = variances[variances == 0.0]
                if len(truly_constant) > 0:
                    df = df.drop(columns=truly_constant.index)
                    self.logger.info(
                        "Removed truly constant columns",
                        removed_columns=list(truly_constant.index),
                        remaining_shape=df.shape,
                    )

        return df

    async def _compute_clustering_metrics(
        self, feature_df: pd.DataFrame
    ) -> FeatureMetrics:
        """Compute clustering metrics for feature matrix.

        Args:
            feature_df: Feature matrix DataFrame.

        Returns:
            Feature metrics.
        """
        start_time = time.time()

        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(feature_df)

            # Handle edge cases
            if X_scaled.shape[0] < 3:
                # Not enough samples for clustering
                return FeatureMetrics(
                    silhouette_score=0.0,
                    calinski_harabasz=0.0,
                    davies_bouldin=1.0,
                    extraction_time=time.time() - start_time,
                    missing_rate=0.0,
                    unique_rate=1.0,
                )

            # Determine number of clusters
            n_samples = X_scaled.shape[0]
            if self.config.evaluation.clustering_config.n_clusters is not None:
                n_clusters = self.config.evaluation.clustering_config.n_clusters
            else:
                cluster_range = self.config.evaluation.clustering_config.cluster_range
                n_clusters = min(max(cluster_range), n_samples - 1)
            n_clusters = max(n_clusters, 2)  # At least 2 clusters

            # Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.evaluation.random_state,
                n_init=10,
            )
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Compute metrics
            silhouette = silhouette_score(X_scaled, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

            # Compute data quality metrics
            missing_rate = feature_df.isnull().sum().sum() / (
                feature_df.shape[0] * feature_df.shape[1]
            )
            unique_rate = feature_df.nunique().mean() / len(feature_df)

            return FeatureMetrics(
                silhouette_score=float(silhouette),
                calinski_harabasz=float(calinski_harabasz),
                davies_bouldin=float(davies_bouldin),
                extraction_time=time.time() - start_time,
                missing_rate=float(missing_rate),
                unique_rate=float(unique_rate),
            )

        except Exception as e:
            self.logger.warning("Clustering metrics computation failed", error=str(e))
            return FeatureMetrics(
                silhouette_score=0.0,
                calinski_harabasz=0.0,
                davies_bouldin=1.0,
                extraction_time=time.time() - start_time,
                missing_rate=1.0,
                unique_rate=0.0,
            )

    def _calculate_overall_score(self, metrics: FeatureMetrics) -> float:
        """Calculate overall score from individual metrics.

        Args:
            metrics: Feature metrics.

        Returns:
            Overall score (0-1, higher is better).
        """
        # Normalize metrics to 0-1 scale (higher is better)
        silhouette_norm = max(
            0.0, (metrics.silhouette_score + 1) / 2
        )  # -1 to 1 -> 0 to 1
        calinski_norm = min(1.0, metrics.calinski_harabasz / 1000.0)  # Cap at 1000
        davies_norm = max(
            0.0, 1.0 - min(1.0, metrics.davies_bouldin / 3.0)
        )  # Lower is better, cap at 3

        # Data quality penalties
        missing_penalty = 1.0 - metrics.missing_rate
        unique_bonus = min(1.0, metrics.unique_rate)

        # Weighted combination
        overall_score = (
            0.4 * silhouette_norm
            + 0.3 * calinski_norm
            + 0.2 * davies_norm
            + 0.05 * missing_penalty
            + 0.05 * unique_bonus
        )

        return max(0.0, min(1.0, overall_score))

    def _create_default_evaluation(
        self,
        feature_set: FeatureSet,
        data_context: DataContext,
        iteration: int,
        evaluation_time: float,
    ) -> FeatureEvaluation:
        """Create default evaluation for failed cases.

        Args:
            feature_set: Feature set.
            data_context: Data context.
            iteration: Iteration number.
            evaluation_time: Evaluation time.

        Returns:
            Default feature evaluation.
        """
        return FeatureEvaluation(
            feature_set=feature_set,
            metrics=FeatureMetrics(
                silhouette_score=0.0,
                calinski_harabasz=0.0,
                davies_bouldin=1.0,
                extraction_time=evaluation_time,
                missing_rate=1.0,
                unique_rate=0.0,
            ),
            overall_score=0.0,
            fold_id=data_context.fold_id,
            iteration=iteration,
            evaluation_time=evaluation_time,
        )
