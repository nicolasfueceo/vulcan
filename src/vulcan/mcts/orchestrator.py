"""MCTS orchestrator for feature engineering."""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List

from vulcan.mcts.tree import MCTSTree
from vulcan.types import (
    LLMInteraction,  # Import separately to ensure proper recognition
    VulcanConfig,
)
from vulcan.utils import PerformanceTracker, get_vulcan_logger

logger = get_vulcan_logger(__name__)


class MCTSOrchestrator:
    """Orchestrates MCTS-based feature engineering."""

    def __init__(self, config: VulcanConfig, performance_tracker: PerformanceTracker):
        """Initialize MCTS orchestrator.

        Args:
            config: VULCAN configuration
            performance_tracker: Performance tracking system
        """
        self.config = config
        self.performance_tracker = performance_tracker
        self.tree = MCTSTree()
        self.llm_history: List[LLMInteraction] = []
        self.decision_logs: List[Dict[str, Any]] = []
        self.iteration_count = 0
        self.best_score = 0.0

    def _log_decision(
        self, decision_type: str, message: str, details: Dict[str, Any] = None
    ):
        """Log a decision or action with details."""
        log_entry = {
            "iteration": self.iteration_count,
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "message": message,
            "details": details,
        }
        self.decision_logs.append(log_entry)
        logger.info(f"[Decision] {message}", **{"details": details} if details else {})

    async def run_search(
        self,
        data_context: Any,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Run MCTS search for feature engineering.

        Args:
            data_context: Data context
            max_iterations: Maximum iterations

        Returns:
            Search results
        """
        logger.info("Starting MCTS search", max_iterations=max_iterations)
        self._log_decision(
            "action_selection",
            f"Starting MCTS search with {max_iterations} iterations",
            {
                "max_iterations": max_iterations,
                "data_context_size": f"{data_context.n_users} users x {data_context.n_items} items",
            },
        )

        # Create root node
        root = self.tree.create_root()
        current_node_id = root.node_id
        self._log_decision(
            "action_selection",
            "Created root node with empty feature set",
            {"node_id": root.node_id},
        )

        # Run search iterations
        for i in range(min(max_iterations, 10)):  # Limit to 10 for demo
            self.iteration_count = i + 1

            # Make action decision
            random_value = random.random()
            explore_threshold = 0.7 if i < 5 else 0.3

            self._log_decision(
                "action_selection",
                f"Drew random number {random_value:.3f} (threshold: {explore_threshold})",
                {
                    "random_value": random_value,
                    "explore_threshold": explore_threshold,
                    "current_node_features": self.tree.get_node(
                        current_node_id
                    ).feature_set,
                },
            )

            if random_value > explore_threshold:
                # EXPLOIT: Mutate existing feature
                current_node = self.tree.get_node(current_node_id)
                if current_node.feature_set:
                    # Pick a random feature to mutate
                    idx = random.randint(0, len(current_node.feature_set) - 1)
                    original = current_node.feature_set[idx]
                    mutated = f"{original}_mutated_v{i}"

                    self._log_decision(
                        "action_selection",
                        f"Chose EXPLOIT action - will mutate feature at index {idx}",
                        {
                            "action": "exploit",
                            "target_feature": original,
                            "target_index": idx,
                            "current_features": current_node.feature_set,
                        },
                    )

                    # Simulate LLM interaction for mutation
                    prompt = f"""You are mutating an existing feature to improve its performance.

Original feature: {original}
Current feature set: {current_node.feature_set}

Task: Create an improved version of '{original}' that captures more nuanced patterns in user behavior.
Consider adding normalization, handling edge cases, or combining with other signals."""

                    response = f"""def {mutated}(user_data):
    # Improved version of {original}
    base_value = user_data.ratings.mean()
    
    # Add variance normalization
    if user_data.ratings.std() > 0:
        normalized = (base_value - user_data.ratings.min()) / user_data.ratings.std()
    else:
        normalized = base_value
        
    # Weight by user activity level
    activity_weight = min(len(user_data.ratings) / 20.0, 1.0)
    
    return normalized * activity_weight"""

                    reflection = f"This mutation adds normalization and activity weighting to {original}. The normalization helps handle users with different rating scales, while the activity weight gives more confidence to features from active users."

                    self.llm_history.append(
                        LLMInteraction(
                            iteration=i + 1,
                            feature_name=mutated,
                            prompt=prompt,
                            response=response,
                            timestamp=datetime.now().isoformat(),
                            reflection=reflection,
                        )
                    )

                    self._log_decision(
                        "feature_mutation",
                        f"Generated mutation of '{original}' -> '{mutated}'",
                        {
                            "original_feature": original,
                            "mutated_feature": mutated,
                            "mutation_type": "normalization_and_weighting",
                            "llm_model": "simulated",
                        },
                    )

                    # Add exploitation node
                    new_node = self.tree.add_exploitation_node(
                        current_node_id, mutated, idx
                    )

                    self._log_decision(
                        "reflection",
                        reflection,
                        {
                            "feature": mutated,
                            "improvement_rationale": "Added normalization and activity weighting",
                        },
                    )

                else:
                    # No features to mutate, explore instead
                    self._log_decision(
                        "action_selection",
                        "No features available to mutate, falling back to EXPLORE",
                        {"reason": "empty_feature_set"},
                    )
                    feature_name = f"feature_{i + 1}_fallback"
                    new_node = self.tree.add_exploration_node(
                        current_node_id, feature_name
                    )
            else:
                # EXPLORE: Generate new feature
                feature_name = f"user_behavior_feature_{i + 1}"

                self._log_decision(
                    "action_selection",
                    "Chose EXPLORE action - will generate new feature",
                    {
                        "action": "explore",
                        "current_feature_count": len(
                            self.tree.get_node(current_node_id).feature_set
                        ),
                    },
                )

                # Simulate LLM interaction for new feature
                prompt = f"""Generate a new feature for user behavior analysis in a recommender system.

Current features: {self.tree.get_node(current_node_id).feature_set}
Iteration: {i + 1}

Task: Create a novel feature that captures an aspect of user behavior not covered by existing features.
Focus on: rating patterns, temporal behavior, genre preferences, or user engagement metrics."""

                response = f"""def {feature_name}(user_data):
    # Analyze user's rating consistency
    ratings = user_data.ratings
    
    # Calculate rating variance per genre
    genre_variances = []
    for genre in user_data.genres.unique():
        genre_ratings = ratings[user_data.genres == genre]
        if len(genre_ratings) > 1:
            genre_variances.append(genre_ratings.var())
    
    # Users with consistent ratings per genre are more predictable
    if genre_variances:
        consistency_score = 1.0 / (1.0 + np.mean(genre_variances))
    else:
        consistency_score = 0.5
        
    return consistency_score"""

                reflection = "This feature captures rating consistency within genres. Users who rate similarly within each genre (low variance) are more predictable and their preferences are more reliable for recommendations."

                self.llm_history.append(
                    LLMInteraction(
                        iteration=i + 1,
                        feature_name=feature_name,
                        prompt=prompt,
                        response=response,
                        timestamp=datetime.now().isoformat(),
                        reflection=reflection,
                    )
                )

                self._log_decision(
                    "feature_generation",
                    f"Generated new feature '{feature_name}'",
                    {
                        "feature_name": feature_name,
                        "feature_type": "genre_consistency",
                        "llm_model": "simulated",
                    },
                )

                # Add exploration node
                new_node = self.tree.add_exploration_node(current_node_id, feature_name)

                self._log_decision(
                    "reflection",
                    reflection,
                    {
                        "feature": feature_name,
                        "captures": "rating consistency within genres",
                    },
                )

            # Simulate evaluation
            base_score = 0.5
            feature_bonus = len(new_node.feature_set) * 0.05
            random_component = random.uniform(-0.1, 0.2)
            score = min(base_score + feature_bonus + random_component, 0.95)

            self._log_decision(
                "evaluation",
                f"Evaluated node with {len(new_node.feature_set)} features",
                {
                    "node_id": new_node.node_id,
                    "feature_set": new_node.feature_set,
                    "base_score": base_score,
                    "feature_bonus": feature_bonus,
                    "random_component": random_component,
                    "final_score": score,
                    "score_components": {
                        "silhouette": score - 0.1,
                        "calinski_harabasz": score * 100,
                        "davies_bouldin": 1.0 - score,
                    },
                },
            )

            self.tree.update_node_score(new_node.node_id, score)

            # Update best score and current node
            if score > self.best_score:
                old_best = self.best_score
                self.best_score = score
                current_node_id = new_node.node_id

                self._log_decision(
                    "action_selection",
                    f"New best score found! {old_best:.3f} -> {score:.3f}",
                    {
                        "previous_best": old_best,
                        "new_best": score,
                        "improvement": score - old_best,
                        "best_features": new_node.feature_set,
                    },
                )

            # Small delay to simulate processing
            await asyncio.sleep(0.1)

        # Get best path
        best_path, best_score = self.tree.get_best_path()
        best_node = self.tree.get_node(best_path[-1]) if best_path else None

        self._log_decision(
            "action_selection",
            f"Search completed after {self.iteration_count} iterations",
            {
                "best_score": best_score,
                "best_features": best_node.feature_set if best_node else [],
                "tree_size": len(self.tree.nodes),
                "best_path_length": len(best_path),
            },
        )

        return {
            "best_node_id": best_node.node_id if best_node else None,
            "best_score": best_score,
            "best_features": [
                {"name": f, "type": "generated"}
                for f in (best_node.feature_set if best_node else [])
            ],
            "total_iterations": self.iteration_count,
            "tree_size": len(self.tree.nodes),
            "decision_logs": self.decision_logs,
            "llm_history": self.llm_history,
        }
