"""Performance tracking system for VULCAN feature engineering."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from vulcan.schemas import (
    FeatureEvaluation,
)
from vulcan.utils import get_vulcan_logger

logger = get_vulcan_logger(__name__)


@dataclass
class FeaturePerformanceMetrics:
    """Comprehensive performance metrics for a feature."""

    feature_name: str
    # Performance scores
    avg_score: float
    max_score: float
    min_score: float
    score_std: float

    # Trend analysis
    score_trend: float  # Positive = improving, negative = degrading
    recent_performance: float  # Performance in last 5 evaluations

    # Usage statistics
    appearances: int  # How many times this feature appeared
    success_rate: float  # Percentage of evaluations where it improved performance

    # Timing metrics
    avg_execution_time: float
    total_execution_time: float

    # Cost metrics
    avg_computational_cost: float

    # Stability metrics
    performance_variance: float
    consistency_score: float  # Lower variance = higher consistency


class PerformanceTracker:
    """Tracks and analyzes feature performance over time."""

    def __init__(self, max_history: int = 1000):
        """Initialize performance tracker.

        Args:
            max_history: Maximum number of evaluations to keep in history
        """
        self.max_history = max_history
        self.evaluation_history: deque = deque(maxlen=max_history)
        self.feature_metrics: Dict[str, FeaturePerformanceMetrics] = {}
        self.feature_appearances: Dict[str, List[float]] = defaultdict(list)
        self.feature_execution_times: Dict[str, List[float]] = defaultdict(list)
        self.feature_costs: Dict[str, List[float]] = defaultdict(list)
        self.baseline_score: Optional[float] = None

        logger.info("Performance tracker initialized", max_history=max_history)

    def record_evaluation(self, evaluation: FeatureEvaluation) -> None:
        """Record a new feature evaluation.

        Args:
            evaluation: Feature evaluation to record
        """
        start_time = time.time()

        logger.debug(
            "Recording feature evaluation",
            evaluation_id=len(self.evaluation_history),
            overall_score=evaluation.overall_score,
            feature_count=len(evaluation.feature_set.features),
        )

        # Add to history
        self.evaluation_history.append(evaluation)

        # Set baseline if first evaluation
        if self.baseline_score is None:
            self.baseline_score = evaluation.overall_score
            logger.info("Baseline score established", baseline=self.baseline_score)

        # Update feature-specific metrics
        self._update_feature_metrics(evaluation)

        # Recompute performance metrics
        self._recompute_metrics()

        processing_time = time.time() - start_time
        logger.debug(
            "Evaluation recorded",
            processing_time=f"{processing_time:.3f}s",
            total_evaluations=len(self.evaluation_history),
        )

    def _update_feature_metrics(self, evaluation: FeatureEvaluation) -> None:
        """Update feature-specific tracking data."""
        for feature in evaluation.feature_set.features:
            # Record appearance
            self.feature_appearances[feature.name].append(evaluation.overall_score)

            # Record execution time
            if hasattr(evaluation.metrics, "extraction_time"):
                self.feature_execution_times[feature.name].append(
                    evaluation.metrics.extraction_time
                )

            # Record computational cost
            self.feature_costs[feature.name].append(feature.computational_cost)

    def _recompute_metrics(self) -> None:
        """Recompute comprehensive metrics for all features."""
        logger.debug("Recomputing feature performance metrics")

        for feature_name in self.feature_appearances:
            scores = self.feature_appearances[feature_name]
            execution_times = self.feature_execution_times.get(feature_name, [])
            costs = self.feature_costs.get(feature_name, [])

            if not scores:
                continue

            # Basic statistics
            avg_score = mean(scores)
            max_score = max(scores)
            min_score = min(scores)
            score_std = stdev(scores) if len(scores) > 1 else 0.0

            # Trend analysis (improvement over time)
            score_trend = self._calculate_trend(scores)
            recent_performance = mean(scores[-5:]) if len(scores) >= 5 else avg_score

            # Success rate (percentage of above-baseline performances)
            success_count = sum(
                1 for score in scores if score > (self.baseline_score or 0)
            )
            success_rate = success_count / len(scores) if scores else 0.0

            # Timing metrics
            avg_execution_time = mean(execution_times) if execution_times else 0.0
            total_execution_time = sum(execution_times) if execution_times else 0.0

            # Cost metrics
            avg_computational_cost = mean(costs) if costs else 0.0

            # Stability metrics
            performance_variance = score_std**2
            consistency_score = 1.0 / (
                1.0 + performance_variance
            )  # Higher = more consistent

            # Create metrics object
            self.feature_metrics[feature_name] = FeaturePerformanceMetrics(
                feature_name=feature_name,
                avg_score=avg_score,
                max_score=max_score,
                min_score=min_score,
                score_std=score_std,
                score_trend=score_trend,
                recent_performance=recent_performance,
                appearances=len(scores),
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                total_execution_time=total_execution_time,
                avg_computational_cost=avg_computational_cost,
                performance_variance=performance_variance,
                consistency_score=consistency_score,
            )

    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate performance trend using linear regression slope."""
        if len(scores) < 2:
            return 0.0

        n = len(scores)
        x_values = list(range(n))

        # Calculate linear regression slope
        sum_x = sum(x_values)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(x_values, scores))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def get_best_features(
        self, top_k: int = 5, criteria: str = "avg_score"
    ) -> List[FeaturePerformanceMetrics]:
        """Get the best performing features based on specified criteria.

        Args:
            top_k: Number of top features to return
            criteria: Criteria for ranking ('avg_score', 'recent_performance', 'consistency_score', etc.)

        Returns:
            List of top performing features
        """
        if not self.feature_metrics:
            return []

        # Sort by specified criteria
        sorted_features = sorted(
            self.feature_metrics.values(),
            key=lambda x: getattr(x, criteria),
            reverse=True,
        )

        return sorted_features[:top_k]

    def get_worst_features(
        self, bottom_k: int = 5, criteria: str = "avg_score"
    ) -> List[FeaturePerformanceMetrics]:
        """Get the worst performing features based on specified criteria."""
        if not self.feature_metrics:
            return []

        # Sort by specified criteria (ascending for worst)
        sorted_features = sorted(
            self.feature_metrics.values(),
            key=lambda x: getattr(x, criteria),
            reverse=False,
        )

        return sorted_features[:bottom_k]

    def get_feature_metrics(
        self, feature_name: str
    ) -> Optional[FeaturePerformanceMetrics]:
        """Get comprehensive metrics for a specific feature."""
        return self.feature_metrics.get(feature_name)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.evaluation_history:
            return {"status": "no_data"}

        # Overall statistics
        overall_scores = [eval.overall_score for eval in self.evaluation_history]
        recent_scores = (
            overall_scores[-10:] if len(overall_scores) >= 10 else overall_scores
        )

        # Feature statistics
        unique_features = set()
        for eval in self.evaluation_history:
            unique_features.update(f.name for f in eval.feature_set.features)

        return {
            "total_evaluations": len(self.evaluation_history),
            "unique_features_tested": len(unique_features),
            "baseline_score": self.baseline_score,
            "best_score": max(overall_scores),
            "current_score": overall_scores[-1],
            "recent_avg_score": mean(recent_scores),
            "overall_improvement": overall_scores[-1] - (self.baseline_score or 0),
            "recent_trend": self._calculate_trend(recent_scores),
            "active_features": len(self.feature_metrics),
            "best_feature": self.get_best_features(1)[0].feature_name
            if self.feature_metrics
            else None,
            "worst_feature": self.get_worst_features(1)[0].feature_name
            if self.feature_metrics
            else None,
        }

    def suggest_feature_actions(self) -> Dict[str, List[str]]:
        """Suggest actions based on feature performance analysis."""
        if not self.feature_metrics:
            return {"suggestions": ["No performance data available yet"]}

        suggestions = {
            "remove_features": [],
            "investigate_features": [],
            "promote_features": [],
            "general_suggestions": [],
        }

        # Find features to remove (consistently poor performance)
        for feature in self.get_worst_features(3, "avg_score"):
            if feature.success_rate < 0.3 and feature.appearances >= 3:
                suggestions["remove_features"].append(
                    f"{feature.feature_name} (success rate: {feature.success_rate:.2f})"
                )

        # Find features to investigate (high variance)
        for feature_name, metrics in self.feature_metrics.items():
            if metrics.performance_variance > 0.1 and metrics.appearances >= 3:
                suggestions["investigate_features"].append(
                    f"{feature_name} (high variance: {metrics.performance_variance:.3f})"
                )

        # Find features to promote (consistently good performance)
        for feature in self.get_best_features(3, "consistency_score"):
            if feature.success_rate > 0.7 and feature.appearances >= 3:
                suggestions["promote_features"].append(
                    f"{feature.feature_name} (success rate: {feature.success_rate:.2f})"
                )

        # General suggestions
        if len(self.evaluation_history) >= 10:
            recent_trend = self._calculate_trend(
                [e.overall_score for e in list(self.evaluation_history)[-10:]]
            )
            if recent_trend < -0.01:
                suggestions["general_suggestions"].append(
                    "Performance is declining - consider feature exploration"
                )
            elif recent_trend > 0.01:
                suggestions["general_suggestions"].append(
                    "Performance is improving - continue current strategy"
                )

        return suggestions

    def export_metrics(self) -> Dict[str, Any]:
        """Export all performance metrics for external analysis."""
        return {
            "evaluation_history": [
                {
                    "overall_score": eval.overall_score,
                    "features": [f.name for f in eval.feature_set.features],
                    "evaluation_time": eval.evaluation_time,
                    "fold_id": eval.fold_id,
                    "iteration": eval.iteration,
                }
                for eval in self.evaluation_history
            ],
            "feature_metrics": {
                name: {
                    "avg_score": metrics.avg_score,
                    "max_score": metrics.max_score,
                    "min_score": metrics.min_score,
                    "score_trend": metrics.score_trend,
                    "success_rate": metrics.success_rate,
                    "appearances": metrics.appearances,
                    "consistency_score": metrics.consistency_score,
                    "avg_execution_time": metrics.avg_execution_time,
                    "avg_computational_cost": metrics.avg_computational_cost,
                }
                for name, metrics in self.feature_metrics.items()
            },
            "summary": self.get_performance_summary(),
        }
