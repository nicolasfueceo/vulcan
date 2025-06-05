"""Basic feature evaluator for VULCAN system using clustering metrics."""

import time
from typing import Dict, List

import structlog

from vulcan.evaluation.base_evaluator import BaseFeatureEvaluator
from vulcan.schemas import (
    DataContext,
    FeatureEvaluation,
    FeatureMetrics,
    FeatureSet,
    FeatureValue,
    VulcanConfig,
)

logger = structlog.get_logger(__name__)


class FeatureEvaluator(BaseFeatureEvaluator):
    """Basic evaluator that uses clustering metrics only."""

    def __init__(self, config: VulcanConfig) -> None:
        """
        Initialize feature evaluator.

        Args:
            config: VULCAN configuration.
        """
        super().__init__(config)

    async def evaluate_feature_set(
        self,
        feature_set: FeatureSet,
        feature_results: Dict[str, List[FeatureValue]],
        data_context: DataContext,
        iteration: int,
    ) -> FeatureEvaluation:
        """
        Evaluate a feature set using clustering metrics.

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
            # Build feature matrix using base class method
            feature_df = self._build_feature_matrix(feature_results)

            if feature_df.empty or len(feature_df.columns) == 0:
                # No valid features to evaluate
                return self._create_default_evaluation(
                    feature_set, data_context, iteration, time.time() - start_time
                )

            # Compute clustering metrics using base class method
            metrics_dict, _ = self._compute_clustering_metrics(feature_df)

            # Get data quality metrics
            quality_metrics = self._compute_data_quality_metrics(feature_df)

            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics_dict, quality_metrics)

            evaluation_time = time.time() - start_time

            # Create metrics object
            metrics = FeatureMetrics(
                silhouette_score=metrics_dict["silhouette_score"],
                calinski_harabasz=metrics_dict["calinski_harabasz"],
                davies_bouldin=metrics_dict["davies_bouldin"],
                extraction_time=evaluation_time,
                missing_rate=quality_metrics["missing_rate"],
                unique_rate=quality_metrics["unique_rate"],
            )

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

    def _calculate_overall_score(
        self, metrics: Dict[str, float], quality_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate overall score from individual metrics.

        Args:
            metrics: Clustering metrics.
            quality_metrics: Data quality metrics.

        Returns:
            Overall score (0-1, higher is better).
        """
        # Normalize clustering metrics using base class method
        normalized = self._normalize_clustering_metrics(metrics)

        # Data quality factors
        missing_penalty = 1.0 - quality_metrics["missing_rate"]
        unique_bonus = min(1.0, quality_metrics["unique_rate"])

        # Weighted combination
        overall_score = (
            0.4 * normalized["silhouette"]
            + 0.3 * normalized["calinski"]
            + 0.2 * normalized["davies"]
            + 0.05 * missing_penalty
            + 0.05 * unique_bonus
        )

        return max(0.0, min(1.0, overall_score))
