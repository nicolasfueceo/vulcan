"""Feature type definitions for VULCAN system."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    """Types of features that can be generated."""

    CODE_BASED = "code_based"  # Direct pandas/numpy operations
    LLM_BASED = "llm_based"  # LLM extraction from text data
    HYBRID = "hybrid"  # Combination of code + LLM


class MCTSAction(str, Enum):
    """Actions that can be taken at each MCTS node."""

    ADD = "add"  # Keep existing features + add new feature
    MUTATE = "mutate"  # Modify existing feature
    REPLACE = "replace"  # Replace worst feature with new one
    COMBINE = "combine"  # Merge two features into composite


class FeatureDefinition(BaseModel):
    """Complete feature definition with execution logic."""

    name: str = Field(..., description="Feature name")
    feature_type: FeatureType = Field(..., description="Type of feature")
    description: str = Field(..., description="Human-readable description")

    # Code-based features
    code: Optional[str] = Field(None, description="Python code for feature extraction")
    dependencies: List[str] = Field(
        default_factory=list, description="Required columns/features"
    )

    # LLM-based features
    llm_prompt: Optional[str] = Field(
        None, description="LLM prompt for text extraction"
    )
    text_columns: List[str] = Field(
        default_factory=list, description="Text columns to process"
    )

    # Hybrid features
    preprocessing_code: Optional[str] = Field(
        None, description="Code to run before LLM"
    )
    postprocessing_code: Optional[str] = Field(
        None, description="Code to run after LLM"
    )

    # Metadata
    computational_cost: float = Field(
        default=1.0, description="Relative computational cost"
    )
    expected_cardinality: Optional[int] = Field(
        None, description="Expected unique values"
    )

    def validate_definition(self) -> bool:
        """Validate that feature definition is complete."""
        if self.feature_type == FeatureType.CODE_BASED:
            return self.code is not None
        elif self.feature_type == FeatureType.LLM_BASED:
            return self.llm_prompt is not None and len(self.text_columns) > 0
        elif self.feature_type == FeatureType.HYBRID:
            return (
                self.llm_prompt is not None
                and len(self.text_columns) > 0
                and (
                    self.preprocessing_code is not None
                    or self.postprocessing_code is not None
                )
            )
        return False


class FeatureValue(BaseModel):
    """Computed feature value for a specific data point."""

    user_id: str = Field(..., description="User identifier")
    feature_name: str = Field(..., description="Feature name")
    value: Union[float, int, str, bool] = Field(..., description="Feature value")
    confidence: Optional[float] = Field(
        None, description="Confidence in LLM-based features"
    )


class FeatureSet(BaseModel):
    """Collection of features at a specific MCTS node."""

    features: List[FeatureDefinition] = Field(..., description="List of features")
    action_taken: MCTSAction = Field(
        default=MCTSAction.ADD, description="Action that created this set"
    )
    parent_features: Optional[List[str]] = Field(
        None, description="Parent feature names"
    )

    def get_feature_by_name(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature by name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def get_total_cost(self) -> float:
        """Calculate total computational cost."""
        return sum(f.computational_cost for f in self.features)


class FeatureMetrics(BaseModel):
    """Feature evaluation metrics."""

    # Clustering metrics
    silhouette_score: Optional[float] = Field(None, description="Silhouette score")
    calinski_harabasz: Optional[float] = Field(
        None, description="Calinski-Harabasz index"
    )
    davies_bouldin: Optional[float] = Field(None, description="Davies-Bouldin index")

    # Feature-specific metrics
    feature_importance: Optional[float] = Field(
        None, description="Feature importance score"
    )
    correlation_with_target: Optional[float] = Field(
        None, description="Correlation with target"
    )
    mutual_information: Optional[float] = Field(
        None, description="Mutual information score"
    )

    # Data quality metrics
    missing_rate: float = Field(default=0.0, description="Percentage of missing values")
    unique_rate: float = Field(default=0.0, description="Percentage of unique values")

    # Computational metrics
    extraction_time: float = Field(..., description="Time to extract feature")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")


class FeatureEvaluation(BaseModel):
    """Complete feature evaluation result."""

    feature_set: FeatureSet = Field(..., description="Evaluated feature set")
    metrics: FeatureMetrics = Field(..., description="Evaluation metrics")

    # Performance tracking
    overall_score: float = Field(..., description="Combined evaluation score")
    improvement_over_parent: Optional[float] = Field(
        None, description="Improvement vs parent"
    )

    # Context
    fold_id: str = Field(..., description="Data fold identifier")
    iteration: int = Field(..., description="MCTS iteration number")
    evaluation_time: float = Field(..., description="Total evaluation time")


class DataContext(BaseModel):
    """Data context passed to agents for feature extraction."""

    # Data splits
    train_data: Dict[str, Any] = Field(..., description="Training data")
    validation_data: Dict[str, Any] = Field(..., description="Validation data")
    test_data: Dict[str, Any] = Field(..., description="Test data")

    # Metadata
    fold_id: str = Field(..., description="Current fold identifier")
    data_schema: Dict[str, str] = Field(..., description="Column name to type mapping")
    text_columns: List[str] = Field(
        default_factory=list, description="Available text columns"
    )

    # Statistics
    n_users: int = Field(..., description="Number of users")
    n_items: int = Field(..., description="Number of items")
    sparsity: float = Field(..., description="Data sparsity ratio")

    def get_column_info(self, column: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific column."""
        if column not in self.data_schema:
            return None

        # Get sample values from train data
        if column in self.train_data:
            sample_values = self.train_data[column].head(5).tolist()
            return {
                "type": self.data_schema[column],
                "sample_values": sample_values,
                "is_text": column in self.text_columns,
            }
        return None


class ActionContext(BaseModel):
    """Context for deciding next MCTS action."""

    current_features: FeatureSet = Field(..., description="Current feature set")
    performance_history: List[FeatureEvaluation] = Field(
        ..., description="Performance history"
    )
    available_actions: List[MCTSAction] = Field(..., description="Available actions")

    # Constraints
    max_features: int = Field(default=10, description="Maximum number of features")
    max_cost: float = Field(default=100.0, description="Maximum computational cost")

    def can_add_feature(self) -> bool:
        """Check if we can add more features."""
        current_feature_count = len(self.current_features.features)
        return current_feature_count < self.max_features

    def can_increase_cost(self, additional_cost: float) -> bool:
        """Check if we can increase computational cost."""
        current_cost = self.current_features.get_total_cost()
        return (current_cost + additional_cost) <= self.max_cost

    def get_worst_performing_feature(self) -> Optional[str]:
        """Get the name of the worst performing feature based on performance history."""
        if not self.performance_history:
            return None

        if len(self.current_features.features) <= 1:
            return None

        # Calculate feature performance scores based on improvement trends
        feature_scores = self._calculate_feature_performance_scores()

        if not feature_scores:
            return None

        # Return the feature with the lowest performance score
        worst_feature = min(feature_scores.items(), key=lambda x: x[1])
        return worst_feature[0]

    def _calculate_feature_performance_scores(self) -> Dict[str, float]:
        """Calculate performance scores for each feature based on evaluation history."""
        feature_scores = {}

        # Get all feature names from current feature set
        current_feature_names = {f.name for f in self.current_features.features}

        # Initialize scores
        for feature_name in current_feature_names:
            feature_scores[feature_name] = 0.0

        # Analyze performance history to score features
        if len(self.performance_history) >= 2:
            # Look at recent evaluations to determine feature impact
            recent_evaluations = self.performance_history[-5:]  # Last 5 evaluations

            for i in range(1, len(recent_evaluations)):
                current_eval = recent_evaluations[i]
                previous_eval = recent_evaluations[i - 1]

                # Calculate improvement
                improvement = current_eval.overall_score - previous_eval.overall_score

                # Get features that were added/changed between evaluations
                current_features = {f.name for f in current_eval.feature_set.features}
                previous_features = {f.name for f in previous_eval.feature_set.features}

                # Assign improvement/degradation to features
                new_features = current_features - previous_features
                removed_features = previous_features - current_features

                # Reward new features for positive improvements
                for feature_name in new_features:
                    if feature_name in feature_scores:
                        feature_scores[feature_name] += improvement

                # Penalize removed features for negative improvements
                for feature_name in removed_features:
                    if feature_name in feature_scores:
                        feature_scores[feature_name] -= improvement

                # For existing features, distribute improvement proportionally
                common_features = current_features & previous_features
                if common_features and improvement != 0:
                    improvement_per_feature = improvement / len(common_features)
                    for feature_name in common_features:
                        if feature_name in feature_scores:
                            feature_scores[feature_name] += improvement_per_feature

        return feature_scores

    def get_best_performing_features(self, top_k: int = 3) -> List[str]:
        """Get the names of the best performing features."""
        feature_scores = self._calculate_feature_performance_scores()

        if not feature_scores:
            return [f.name for f in self.current_features.features[:top_k]]

        # Sort by score descending and return top k
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [name for name, score in sorted_features[:top_k]]

    def get_feature_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary for all features."""
        feature_scores = self._calculate_feature_performance_scores()

        summary = {
            "total_features": len(self.current_features.features),
            "total_evaluations": len(self.performance_history),
            "feature_scores": feature_scores,
            "best_feature": None,
            "worst_feature": None,
            "average_score": 0.0,
        }

        if feature_scores:
            best_feature = max(feature_scores.items(), key=lambda x: x[1])
            worst_feature = min(feature_scores.items(), key=lambda x: x[1])

            summary["best_feature"] = {
                "name": best_feature[0],
                "score": best_feature[1],
            }
            summary["worst_feature"] = {
                "name": worst_feature[0],
                "score": worst_feature[1],
            }
            summary["average_score"] = sum(feature_scores.values()) / len(
                feature_scores
            )

        return summary
