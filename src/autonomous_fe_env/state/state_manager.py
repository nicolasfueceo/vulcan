"""State manager for tracking feature engineering progress and state."""

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..feature import FeatureDefinition


@dataclass
class FeatureState:
    """Represents the state of a feature in the feature engineering process."""

    feature: FeatureDefinition
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature": self.feature.to_dict(),
            "score": self.score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureState":
        """Create from dictionary representation."""
        from ..feature import FeatureDefinition

        return cls(
            feature=FeatureDefinition.from_dict(data["feature"]),
            score=data["score"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.datetime.now().isoformat()),
        )


class StateManager:
    """Manages the state of the feature engineering process."""

    def __init__(self, state_dir: str = "state"):
        """
        Initialize the state manager.

        Args:
            state_dir: Directory to save state files to.
        """
        self.state_dir = state_dir
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Track the overall best feature set found so far
        self.current_score: float = 0.0
        self.current_features: List[FeatureDefinition] = []

        # Track the history of all tried feature sets
        self.feature_history: List[FeatureState] = []

        # Track baseline score
        self.baseline_score: float = 0.0

        # Track MCTS statistics
        self.mcts_stats: Dict[str, Any] = {
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "start_time": datetime.datetime.now().isoformat(),
        }

        # Create state directory if it doesn't exist
        os.makedirs(state_dir, exist_ok=True)

    def update_state(
        self,
        features: List[FeatureDefinition],
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update the current state with a new feature set and score.

        Args:
            features: List of features in the new state
            score: Evaluation score for the feature set
            metadata: Optional additional metadata

        Returns:
            True if this is a new best state, False otherwise
        """
        # Create feature state
        feature_state = FeatureState(
            feature=features[-1] if features else None,  # Latest feature added
            score=score,
            metadata=metadata or {},
        )

        # Add to history
        self.feature_history.append(feature_state)

        # Check if this is a new best state
        is_new_best = score > self.current_score

        if is_new_best:
            self.current_score = score
            self.current_features = (
                features.copy()
            )  # Make a copy to avoid reference issues

            # Save the new best state
            self._save_state()

        return is_new_best

    def get_current_score(self) -> float:
        """Get the current best score."""
        return self.current_score

    def get_current_features(self) -> List[FeatureDefinition]:
        """Get the current best feature set."""
        return self.current_features

    def get_feature_history(self) -> List[FeatureState]:
        """Get the history of all tried features."""
        return self.feature_history

    def set_baseline_score(self, score: float) -> None:
        """Set the baseline score (performance with no features)."""
        self.baseline_score = score

    def get_baseline_score(self) -> float:
        """Get the baseline score."""
        return self.baseline_score

    def update_mcts_stats(self, iteration_success: bool) -> None:
        """
        Update MCTS statistics.

        Args:
            iteration_success: Whether the iteration was successful
        """
        self.mcts_stats["total_iterations"] += 1

        if iteration_success:
            self.mcts_stats["successful_iterations"] += 1
        else:
            self.mcts_stats["failed_iterations"] += 1

        # Update end time
        self.mcts_stats["end_time"] = datetime.datetime.now().isoformat()

        # Calculate duration
        start_time = datetime.datetime.fromisoformat(self.mcts_stats["start_time"])
        end_time = datetime.datetime.fromisoformat(self.mcts_stats["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.mcts_stats["duration_seconds"] = duration

    def get_mcts_stats(self) -> Dict[str, Any]:
        """Get MCTS statistics."""
        return self.mcts_stats

    def get_successful_features(self) -> List[FeatureDefinition]:
        """Get all features that have been successfully evaluated."""
        return [
            state.feature for state in self.feature_history if state.feature is not None
        ]

    def get_summary(self) -> str:
        """Get a summary of the current state for use in prompts."""
        summary = [
            f"Current best score: {self.current_score:.4f}",
            f"Baseline score: {self.baseline_score:.4f}",
            f"Improvement over baseline: {self.current_score - self.baseline_score:.4f}",
            f"Number of features in best set: {len(self.current_features)}",
            f"Total features tried: {len(self.feature_history)}",
            f"Total MCTS iterations: {self.mcts_stats['total_iterations']}",
            f"Successful iterations: {self.mcts_stats['successful_iterations']}",
            f"Failed iterations: {self.mcts_stats['failed_iterations']}",
        ]

        if self.current_features:
            summary.append("\nCurrent best features:")
            for i, feature in enumerate(self.current_features):
                summary.append(f"- {i + 1}. {feature.name}: {feature.description}")

        return "\n".join(summary)

    def save_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save the current state to a file.

        Args:
            filename: Optional filename to save to (default: uses session ID)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"state_{self.session_id}.json"

        filepath = os.path.join(self.state_dir, filename)

        state_data = {
            "session_id": self.session_id,
            "current_score": self.current_score,
            "baseline_score": self.baseline_score,
            "current_features": [
                feature.to_dict() for feature in self.current_features
            ],
            "feature_history": [state.to_dict() for state in self.feature_history],
            "mcts_stats": self.mcts_stats,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

        return filepath

    def load_from_file(self, filepath: str) -> bool:
        """
        Load state from a file.

        Args:
            filepath: Path to the state file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.session_id = data.get("session_id", self.session_id)
            self.current_score = data.get("current_score", 0.0)
            self.baseline_score = data.get("baseline_score", 0.0)

            # Load current features
            from ..feature import FeatureDefinition

            self.current_features = [
                FeatureDefinition.from_dict(feature_data)
                for feature_data in data.get("current_features", [])
            ]

            # Load feature history
            self.feature_history = [
                FeatureState.from_dict(state_data)
                for state_data in data.get("feature_history", [])
            ]

            # Load MCTS stats
            self.mcts_stats = data.get(
                "mcts_stats",
                {
                    "total_iterations": 0,
                    "successful_iterations": 0,
                    "failed_iterations": 0,
                    "start_time": datetime.datetime.now().isoformat(),
                },
            )

            return True

        except Exception as e:
            print(f"Error loading state from {filepath}: {e}")
            return False

    def _save_state(self) -> None:
        """Save the current state to the default session file."""
        self.save_to_file(f"state_{self.session_id}.json")
