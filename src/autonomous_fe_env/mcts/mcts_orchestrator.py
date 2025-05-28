"""MCTS Orchestrator for managing the feature engineering process."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ..agents import BaseAgent
from ..data import BaseDAL
from ..evaluation import FeatureEvaluator
from ..feature import FeatureDefinition
from ..reflection import ReflectionEngine
from ..state import StateManager
from .mcts_node import MCTSNode

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCTSOrchestrator:
    """Orchestrates the Monte Carlo Tree Search for feature engineering."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MCTS orchestrator.

        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing MCTS Orchestrator...")
        self.config = config

        # MCTS parameters
        mcts_config = self.config.get("mcts", {})
        self.max_iterations = mcts_config.get("max_iterations", 50)
        self.exploration_factor = mcts_config.get("exploration_factor", 1.414)
        self.reward_discount = mcts_config.get("reward_discount", 1.0)
        self.max_depth = mcts_config.get("max_depth", 10)  # Maximum feature set size
        self.agent_failure_strategy = mcts_config.get(
            "agent_failure_strategy", "skip_node"
        )

        # Components
        self.dal: BaseDAL = None  # Will be initialized in setup()
        self.agent: BaseAgent = None  # Will be initialized in setup()
        self.evaluator: FeatureEvaluator = None  # Will be initialized in setup()
        self.reflection_engine: ReflectionEngine = (
            None  # Will be initialized in setup()
        )
        self.state_manager: StateManager = StateManager(
            state_dir=self.config.get("state_dir", "state")
        )

        # Visualization support
        self.visualizer = None  # Will be set if visualization is enabled

        # MCTS state
        self.root_node: Optional[MCTSNode] = None
        self.best_node: Optional[MCTSNode] = None

        # Tracking exploration
        self.node_cache: Dict[str, MCTSNode] = {}  # Maps node IDs to nodes
        self.feature_set_cache: Dict[
            str, Tuple[float, MCTSNode]
        ] = {}  # Maps feature set hash to (score, node)

        logger.info("MCTS Orchestrator initialized")

    def setup(
        self,
        dal: BaseDAL,
        agent: BaseAgent,
        evaluator: FeatureEvaluator,
        reflection_engine: Optional[ReflectionEngine] = None,
        visualizer=None,
    ) -> None:
        """
        Set up the orchestrator with required components.

        Args:
            dal: Data access layer
            agent: Agent for proposing features
            evaluator: Evaluator for scoring features
            reflection_engine: Optional engine for reflection on feature engineering
            visualizer: Optional visualizer for capturing tree snapshots
        """
        self.dal = dal
        self.agent = agent
        self.evaluator = evaluator
        self.reflection_engine = reflection_engine
        self.visualizer = visualizer

        logger.info("MCTS Orchestrator setup complete")

    def calculate_baseline(self) -> float:
        """
        Calculate the baseline score (performance with no features).

        Returns:
            Baseline score
        """
        logger.info("Calculating baseline score...")

        # Evaluate with empty feature set
        baseline_score = self.evaluator.evaluate_feature_set([])

        logger.info(f"Baseline Score: {baseline_score:.4f}")
        self.state_manager.set_baseline_score(baseline_score)

        # Initialize root node with baseline score
        self.root_node = MCTSNode(state_features=[], score=baseline_score)
        self.best_node = self.root_node

        # Cache the root node
        self.node_cache[self.root_node.node_id] = self.root_node
        self.feature_set_cache[self._hash_feature_set([])] = (
            baseline_score,
            self.root_node,
        )

        return baseline_score

    def run(self) -> MCTSNode:
        """
        Run the MCTS feature engineering process.

        Returns:
            The best node found during the search
        """
        logger.info("Starting MCTS Feature Engineering Process...")

        # Ensure data source is connected
        if not self.dal:
            raise ValueError(
                "Data access layer (DAL) is not initialized. Call setup() first."
            )

        self.dal.connect()

        # Calculate baseline if needed
        if self.root_node is None:
            self.calculate_baseline()

        # Main MCTS loop
        start_time = time.time()
        for i in range(self.max_iterations):
            logger.info(f"\n--- MCTS Iteration {i + 1}/{self.max_iterations} ---")

            # 1. Selection: Find a node to expand
            selected_node = self._select_node(self.root_node)
            logger.info(f"Selected Node: {selected_node}")

            # 2. Expansion: Generate a new feature and expand the selected node
            expanded_node = self._expand_node(selected_node)

            if expanded_node is None:
                logger.info("Expansion failed, skipping backpropagation")
                self.state_manager.update_mcts_stats(False)
                continue

            logger.info(f"Expanded Node: {expanded_node}")

            # 3. Simulation: The reward is the score of the expanded node
            # (Simulation is implicit in our case - we directly evaluate the expanded node)
            reward = expanded_node.score - self.state_manager.get_baseline_score()
            logger.info(f"Reward: {reward:.4f}")

            # 4. Backpropagation: Update values up the tree
            self._backpropagate(expanded_node, reward)

            # Update best node if this is better
            if expanded_node.score > self.best_node.score:
                self.best_node = expanded_node
                logger.info(f"New best node found: {self.best_node}")

            # Update state manager
            self.state_manager.update_state(
                expanded_node.state_features,
                expanded_node.score,
                metadata={"node_id": expanded_node.node_id},
            )

            # Capture tree snapshot for visualization (every few iterations)
            if self.visualizer and (i + 1) % 3 == 0:  # Every 3 iterations
                try:
                    self.visualizer.log_mcts_tree_snapshot(self.root_node, i + 1)
                    logger.info(f"Captured MCTS tree snapshot for iteration {i + 1}")
                except Exception as e:
                    logger.warning(f"Failed to capture tree snapshot: {e}")

            # Generate reflection if available
            if (
                self.reflection_engine
                and (i + 1) % self.config.get("reflection_interval", 5) == 0
            ):
                self._generate_reflection(i + 1)

            # Update MCTS stats
            self.state_manager.update_mcts_stats(True)

            logger.info(f"Current best score: {self.best_node.score:.4f}")

        # Search complete
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"\nMCTS search completed in {duration:.2f} seconds")
        logger.info(f"Best node: {self.best_node}")
        logger.info(f"Best score: {self.best_node.score:.4f}")

        # Save the final state
        self._save_search_results()

        # Clean up
        self.dal.disconnect()

        return self.best_node

    def _select_node(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node for expansion using UCT.

        Args:
            node: Starting node for selection

        Returns:
            Selected node for expansion
        """
        current = node

        # While not at a leaf node and not at max depth
        while current.children and current.depth < self.max_depth:
            # Check if the node is fully expanded (terminal)
            if current.is_fully_expanded() or current.is_terminal:
                return current

            # Select child using UCT
            current = current.select_child_uct(self.exploration_factor)

        return current

    def _expand_node(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expand a node by generating a new feature.

        Args:
            node: Node to expand

        Returns:
            Expanded child node, or None if expansion failed
        """
        # Check if we've reached max depth
        if node.depth >= self.max_depth:
            logger.info(
                f"Reached maximum depth ({self.max_depth}), marking node as terminal"
            )
            node.is_terminal = True
            return None

        # If node is terminal, can't expand
        if node.is_terminal:
            logger.info("Node is terminal, cannot expand")
            return None

        # Get existing child feature names
        existing_child_features = node.tried_feature_names

        # Ask agent for feature proposal
        logger.info("Asking agent for feature proposal...")
        new_feature = self.agent.generate_feature_proposal(
            self.state_manager,
            context={
                "mcts_node": node,
                "current_features": node.state_features,
                "existing_child_features": existing_child_features,
                "database_schema": self.dal.get_schema()
                if hasattr(self.dal, "get_schema")
                else {},
            },
        )

        if new_feature is None:
            logger.info("Agent failed to propose a feature")
            return None

        # Check if this feature has already been tried from this node
        if node.has_tried_feature(new_feature.name):
            logger.info(
                f"Agent proposed feature '{new_feature.name}' which has already been tried"
            )

            # Find the existing child node
            for child in node.children:
                if (
                    child.feature_that_led_here
                    and child.feature_that_led_here.name == new_feature.name
                ):
                    logger.info(
                        f"Using existing child node for feature '{new_feature.name}'"
                    )
                    return child

            # Should not reach here if the feature is properly tracked
            logger.warning(
                f"Could not find existing child for feature '{new_feature.name}'"
            )
            return None

        # Create new state features
        new_state_features = node.state_features + [new_feature]

        # Check if this exact feature set has been evaluated before
        feature_set_hash = self._hash_feature_set(new_state_features)
        if feature_set_hash in self.feature_set_cache:
            score, existing_node = self.feature_set_cache[feature_set_hash]
            logger.info(f"Found cached evaluation for feature set: {score:.4f}")

            # Create a new child node linked to this parent
            child_node = MCTSNode(
                state_features=new_state_features,
                parent=node,
                feature_that_led_here=new_feature,
                score=score,
            )
            node.add_child(child_node)

            # Cache the new node
            self.node_cache[child_node.node_id] = child_node

            return child_node

        # Evaluate the new feature set
        logger.info(f"Evaluating feature set with new feature: {new_feature.name}")
        try:
            evaluation_score = self.evaluator.evaluate_feature_set(new_state_features)
        except Exception as e:
            logger.error(f"Error evaluating feature set: {str(e)}")
            return None

        if evaluation_score == -float("inf"):
            logger.info(f"Evaluation failed for feature '{new_feature.name}'")
            return None

        # Create the new child node
        child_node = MCTSNode(
            state_features=new_state_features,
            parent=node,
            feature_that_led_here=new_feature,
            score=evaluation_score,
        )

        # Add child to parent
        node.add_child(child_node)

        # Cache the new node and feature set
        self.node_cache[child_node.node_id] = child_node
        self.feature_set_cache[feature_set_hash] = (evaluation_score, child_node)

        return child_node

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate reward up the tree.

        Args:
            node: Starting node for backpropagation
            reward: Reward to propagate
        """
        current = node
        current_reward = reward

        while current:
            current.update(current_reward)
            current = current.parent
            current_reward *= self.reward_discount  # Apply discount factor if desired

    def _hash_feature_set(self, features: List[FeatureDefinition]) -> str:
        """
        Create a hash of a feature set for caching purposes.

        Args:
            features: List of features to hash

        Returns:
            String hash representing the feature set
        """
        # Sort feature names for consistent hashing
        feature_names = sorted([f.name for f in features])
        return "|".join(feature_names)

    async def _generate_reflection(self, iteration: int) -> None:
        """
        Generate reflections on the feature engineering process.

        Args:
            iteration: Current iteration number
        """
        if not self.reflection_engine:
            return

        logger.info("Generating reflection...")

        # Convert feature history to format needed by reflection engine
        feature_history = []
        for state in self.state_manager.get_feature_history():
            if state.feature:
                feature_history.append(
                    {
                        "name": state.feature.name,
                        "description": state.feature.description,
                        "score": state.score,
                    }
                )

        # Get database schema
        schema = self.dal.get_schema() if hasattr(self.dal, "get_schema") else {}

        # Generate reflection
        try:
            reflection = await self.reflection_engine.generate_feature_reflection(
                self.state_manager, feature_history, schema
            )
            logger.info(f"Reflection at iteration {iteration}:\n{reflection}")
        except Exception as e:
            logger.error(f"Error generating reflection: {str(e)}")

    def _save_search_results(self) -> None:
        """Save the search results to disk."""
        results_dir = self.config.get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"mcts_results_{timestamp}.json")

        # Save state manager state
        self.state_manager.save_to_file()

        # Save MCTS tree (just the nodes, not the full structure)
        tree_nodes = [node.to_dict() for node in self.node_cache.values()]

        # Save results
        results = {
            "timestamp": timestamp,
            "config": self.config,
            "best_node_id": self.best_node.node_id if self.best_node else None,
            "best_score": self.best_node.score if self.best_node else None,
            "baseline_score": self.state_manager.get_baseline_score(),
            "improvement": self.best_node.score
            - self.state_manager.get_baseline_score()
            if self.best_node
            else 0,
            "total_iterations": self.state_manager.get_mcts_stats()["total_iterations"],
            "successful_iterations": self.state_manager.get_mcts_stats()[
                "successful_iterations"
            ],
            "failed_iterations": self.state_manager.get_mcts_stats()[
                "failed_iterations"
            ],
            "duration_seconds": self.state_manager.get_mcts_stats()["duration_seconds"],
            "best_features": [f.to_dict() for f in self.best_node.state_features]
            if self.best_node
            else [],
            "tree_nodes": tree_nodes,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Search results saved to {results_file}")

    def get_best_features(self) -> List[FeatureDefinition]:
        """
        Get the best feature set found during the search.

        Returns:
            List of features in the best feature set
        """
        if self.best_node:
            return self.best_node.state_features
        return []
