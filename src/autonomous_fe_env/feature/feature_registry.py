"""Feature registry to manage feature definitions."""

import json
import os
from typing import Any, Dict, List, Optional, Set

from .feature_representation import FeatureDefinition


class FeatureRegistry:
    """Registry for managing and persisting feature definitions."""

    def __init__(self, features_dir: str = "features"):
        """
        Initialize the feature registry.

        Args:
            features_dir: Directory to save feature definitions to.
        """
        self.features_dir = features_dir
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_scores: Dict[str, float] = {}
        self.feature_metadata: Dict[str, Dict[str, Any]] = {}

        # Create the features directory if it doesn't exist
        os.makedirs(features_dir, exist_ok=True)

        # Track dependency relationships between features
        self.feature_dependencies: Dict[str, Set[str]] = {}

    def add_feature(
        self, feature: FeatureDefinition, score: Optional[float] = None
    ) -> bool:
        """
        Add a feature to the registry.

        Args:
            feature: The feature definition to add
            score: Optional evaluation score for the feature

        Returns:
            True if the feature was added successfully, False if it already exists
        """
        if feature.name in self.features:
            print(f"Feature '{feature.name}' already exists in the registry.")
            return False

        self.features[feature.name] = feature
        if score is not None:
            self.feature_scores[feature.name] = score

        # Analyze dependencies
        self._update_dependencies(feature)

        # Save to disk
        self._save_feature(feature)

        print(f"Feature '{feature.name}' added to registry.")
        return True

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """
        Get a feature by name.

        Args:
            name: Name of the feature to retrieve

        Returns:
            The feature definition if found, None otherwise
        """
        return self.features.get(name)

    def get_all_features(self) -> List[FeatureDefinition]:
        """
        Get all features in the registry.

        Returns:
            List of all feature definitions
        """
        return list(self.features.values())

    def get_top_features(self, n: int = 10) -> List[FeatureDefinition]:
        """
        Get the top N features by score.

        Args:
            n: Number of features to return

        Returns:
            List of the top N features by score
        """
        if not self.feature_scores:
            return self.get_all_features()[:n]

        # Sort features by score (descending)
        sorted_features = sorted(
            self.feature_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Get the top N feature names
        top_feature_names = [name for name, _ in sorted_features[:n]]

        # Return the feature definitions
        return [
            self.features[name] for name in top_feature_names if name in self.features
        ]

    def update_feature_score(self, name: str, score: float) -> bool:
        """
        Update the score for a feature.

        Args:
            name: Name of the feature to update
            score: New score for the feature

        Returns:
            True if the feature was updated, False if it doesn't exist
        """
        if name not in self.features:
            print(f"Feature '{name}' not found in registry.")
            return False

        self.feature_scores[name] = score

        # Update the feature's metadata
        if name in self.feature_metadata:
            self.feature_metadata[name]["score"] = score
        else:
            self.feature_metadata[name] = {"score": score}

        # Update the feature file
        self._save_feature_metadata(name)

        return True

    def remove_feature(self, name: str) -> bool:
        """
        Remove a feature from the registry.

        Args:
            name: Name of the feature to remove

        Returns:
            True if the feature was removed, False if it doesn't exist
        """
        if name not in self.features:
            print(f"Feature '{name}' not found in registry.")
            return False

        # Remove from in-memory collections
        del self.features[name]
        if name in self.feature_scores:
            del self.feature_scores[name]
        if name in self.feature_metadata:
            del self.feature_metadata[name]
        if name in self.feature_dependencies:
            del self.feature_dependencies[name]

        # Remove from disk
        feature_path = os.path.join(self.features_dir, f"{name}.json")
        if os.path.exists(feature_path):
            try:
                os.remove(feature_path)
                print(f"Feature '{name}' removed from registry and disk.")
            except OSError as e:
                print(f"Error removing feature file: {e}")

        return True

    def load_features_from_disk(self) -> int:
        """
        Load all feature definitions from disk.

        Returns:
            Number of features loaded
        """
        # Create the features directory if it doesn't exist
        os.makedirs(self.features_dir, exist_ok=True)

        # Clear existing features
        self.features = {}
        self.feature_scores = {}
        self.feature_metadata = {}
        self.feature_dependencies = {}

        # Load all feature files
        count = 0
        for filename in os.listdir(self.features_dir):
            if filename.endswith(".json"):
                try:
                    feature_path = os.path.join(self.features_dir, filename)
                    with open(feature_path, "r") as f:
                        data = json.load(f)

                    # Extract feature definition and metadata
                    feature_data = data.get("feature", {})
                    metadata = data.get("metadata", {})

                    # Create feature definition
                    feature = FeatureDefinition.from_dict(feature_data)

                    # Add to registry
                    self.features[feature.name] = feature

                    # Add score if available
                    if "score" in metadata:
                        self.feature_scores[feature.name] = metadata["score"]

                    # Store metadata
                    self.feature_metadata[feature.name] = metadata

                    # Update dependencies
                    self._update_dependencies(feature)

                    count += 1

                except Exception as e:
                    print(f"Error loading feature from {filename}: {e}")

        print(f"Loaded {count} features from disk.")
        return count

    def _save_feature(self, feature: FeatureDefinition) -> bool:
        """
        Save a feature definition to disk.

        Args:
            feature: The feature definition to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Get metadata for the feature
            metadata = self.feature_metadata.get(feature.name, {})

            # Add score if available
            if feature.name in self.feature_scores:
                metadata["score"] = self.feature_scores[feature.name]

            # Create data structure for serialization
            data = {"feature": feature.to_dict(), "metadata": metadata}

            # Write to file
            feature_path = os.path.join(self.features_dir, f"{feature.name}.json")
            with open(feature_path, "w") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving feature '{feature.name}': {e}")
            return False

    def _save_feature_metadata(self, name: str) -> bool:
        """
        Save updated metadata for a feature.

        Args:
            name: Name of the feature to update

        Returns:
            True if saved successfully, False otherwise
        """
        if name not in self.features:
            return False

        return self._save_feature(self.features[name])

    def _update_dependencies(self, feature: FeatureDefinition) -> None:
        """
        Update the dependency tracking for a feature.

        Args:
            feature: The feature to analyze for dependencies
        """
        # Simple approach: look for feature output column names in the code
        dependencies = set()

        for other_feature in self.features.values():
            # Skip self-reference
            if other_feature.name == feature.name:
                continue

            # Check if the output column of the other feature is used as input for this feature
            if other_feature.output_column_name in feature.required_input_columns:
                dependencies.add(other_feature.name)

            # Also check for references in the code
            if other_feature.output_column_name in feature.code:
                dependencies.add(other_feature.name)

        self.feature_dependencies[feature.name] = dependencies

    def get_feature_dependencies(self, name: str) -> Set[str]:
        """
        Get the dependencies for a feature.

        Args:
            name: Name of the feature to get dependencies for

        Returns:
            Set of feature names that this feature depends on
        """
        return self.feature_dependencies.get(name, set())

    def get_dependent_features(self, name: str) -> Set[str]:
        """
        Get features that depend on the given feature.

        Args:
            name: Name of the feature to get dependents for

        Returns:
            Set of feature names that depend on this feature
        """
        dependents = set()

        for feat_name, dependencies in self.feature_dependencies.items():
            if name in dependencies:
                dependents.add(feat_name)

        return dependents
