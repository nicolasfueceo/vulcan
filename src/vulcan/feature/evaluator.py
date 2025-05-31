"""Feature evaluation engine for VULCAN."""

import time
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from vulcan.types import FeatureSet, VulcanConfig
from vulcan.utils import get_vulcan_logger

logger = get_vulcan_logger(__name__)


class FeatureEvaluator:
    """Evaluates feature sets using clustering metrics."""

    def __init__(self, config: VulcanConfig):
        """Initialize feature evaluator.

        Args:
            config: VULCAN configuration
        """
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.scaler = StandardScaler()

    async def evaluate_feature_set(
        self,
        feature_set: FeatureSet,
        feature_results: Dict[str, List[Any]],
        data_context: Any,
        iteration: int,
    ) -> Dict[str, Any]:
        """Evaluate a cumulative feature set using clustering metrics.

        Args:
            feature_set: FeatureSet object with cumulative features
            feature_results: Computed feature values
            data_context: Data context
            iteration: Current iteration number

        Returns:
            Evaluation metrics
        """
        start_time = time.time()

        try:
            # Convert results to matrix
            feature_matrix = []
            feature_names = [f.name for f in feature_set.features]

            for feature_name in feature_names:
                if feature_name not in feature_results:
                    self.logger.warning(f"Feature {feature_name} not in results")
                    continue

                values = [r["value"] for r in feature_results[feature_name]]
                feature_matrix.append(values)

            if not feature_matrix:
                self.logger.warning("No features found in results")
                return self._create_default_evaluation(start_time)

            # Convert to numpy array and transpose
            X = np.array(feature_matrix).T

            # Standardize
            X_scaled = self.scaler.fit_transform(X)

            # Determine number of clusters
            n_samples = X_scaled.shape[0]
            if self.config.evaluation.clustering_config.n_clusters:
                n_clusters = self.config.evaluation.clustering_config.n_clusters
            else:
                n_clusters = min(
                    max(self.config.evaluation.clustering_config.cluster_range),
                    n_samples - 1,
                )
            n_clusters = max(n_clusters, 2)

            # Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.evaluation.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(X_scaled)

            # Compute metrics
            metrics = {
                "silhouette_score": float(silhouette_score(X_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
            }

            # Calculate overall score (silhouette score normalized to 0-1)
            overall_score = (
                metrics["silhouette_score"] + 1
            ) / 2  # Convert from [-1, 1] to [0, 1]

            self.logger.info(
                "Evaluated feature set",
                n_features=len(feature_names),
                overall_score=overall_score,
                silhouette=metrics["silhouette_score"],
                action=feature_set.action_taken,
            )

            return {
                "feature_set": feature_set,
                "score": overall_score,
                "metrics": metrics,
                "execution_time": time.time() - start_time,
                "n_features": len(feature_names),
            }

        except Exception as e:
            self.logger.error(
                "Feature evaluation failed",
                error=str(e),
                n_features=len(feature_set.features),
            )
            return self._create_default_evaluation(start_time)

    def _create_default_evaluation(self, start_time: float) -> Dict[str, Any]:
        """Create default evaluation for failed cases."""
        return {
            "score": 0.0,
            "metrics": {
                "silhouette_score": 0.0,
                "calinski_harabasz": 0.0,
                "davies_bouldin": 1.0,
            },
            "execution_time": time.time() - start_time,
            "n_features": 0,
        }
