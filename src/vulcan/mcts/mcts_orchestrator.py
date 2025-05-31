"""MCTS Orchestrator for VULCAN feature engineering."""

import time
from typing import Any, Dict, List, Optional

import structlog

from vulcan.agents import FeatureAgent
from vulcan.evaluation import FeatureEvaluator
from vulcan.features import FeatureExecutor
from vulcan.types import (
    ActionContext,
    DataContext,
    ExperimentResult,
    FeatureEvaluation,
    FeatureMetrics,
    FeatureSet,
    MCTSAction,
    VulcanConfig,
)
from vulcan.utils import ExperimentTracker, PerformanceTracker, get_vulcan_logger

from .mcts_node import MCTSNode

logger = structlog.get_logger(__name__)


class MCTSOrchestrator:
    """Orchestrates MCTS-based feature engineering with intelligent actions."""

    def __init__(
        self,
        config: VulcanConfig,
        performance_tracker: Optional[PerformanceTracker] = None,
    ) -> None:
        """Initialize MCTS orchestrator.

        Args:
            config: VULCAN configuration.
            performance_tracker: Optional external performance tracker to use.
        """
        self.config = config
        self.logger = get_vulcan_logger(__name__)

        # Core components
        self.feature_agent = FeatureAgent(config)
        self.feature_executor = FeatureExecutor(config)
        self.feature_evaluator = FeatureEvaluator(config)
        self.experiment_tracker: Optional[ExperimentTracker] = None

        # Performance tracking
        self.performance_tracker = performance_tracker or PerformanceTracker(
            max_history=500
        )

        # MCTS state
        self.root_node: Optional[MCTSNode] = None
        self.current_node: Optional[MCTSNode] = None
        self.best_node: Optional[MCTSNode] = None
        self.best_score = 0.0

        # Tracking
        self.iteration_count = 0
        self.feature_evaluations: List[FeatureEvaluation] = []

    async def initialize(self) -> bool:
        """Initialize all components.

        Returns:
            True if initialization successful.
        """
        try:
            # Initialize components
            await self.feature_agent.initialize()
            await self.feature_executor.initialize()
            await self.feature_evaluator.initialize()

            # Initialize experiment tracking
            if self.config.experiment.wandb_enabled:
                self.experiment_tracker = ExperimentTracker(self.config.experiment)

            # Create root node
            self.root_node = MCTSNode()
            self.current_node = self.root_node

            self.logger.info("MCTS orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error("Failed to initialize MCTS orchestrator", error=str(e))
            return False

    async def run_search(
        self,
        data_context: DataContext,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run MCTS search for optimal feature set.

        Args:
            data_context: Data context with all splits.
            max_iterations: Maximum iterations (uses config if None).

        Returns:
            Search results with best feature set and performance.
        """
        # Ensure we're initialized
        if self.root_node is None:
            await self.initialize()

        max_iterations = max_iterations or self.config.mcts.max_iterations

        self.logger.info(
            "Starting MCTS search",
            max_iterations=max_iterations,
            fold_id=data_context.fold_id,
        )

        start_time = time.time()

        try:
            for iteration in range(max_iterations):
                self.iteration_count = iteration + 1

                # Run single MCTS iteration
                await self._run_iteration(data_context)

                # Log progress
                if iteration % 10 == 0:
                    self.logger.info(
                        "MCTS progress",
                        iteration=iteration + 1,
                        best_score=self.best_score,
                        current_depth=self.current_node.depth
                        if self.current_node
                        else 0,
                    )

                # Track with W&B
                if self.experiment_tracker:
                    self.experiment_tracker.log_mcts_iteration(
                        iteration=iteration + 1,
                        best_score=self.best_score,
                        nodes_explored=self._count_tree_nodes(),
                        current_depth=self.current_node.depth
                        if self.current_node
                        else 0,
                    )

            execution_time = time.time() - start_time

            # Generate final results
            results = await self._generate_results(data_context, execution_time)

            self.logger.info(
                "MCTS search completed",
                total_iterations=max_iterations,
                best_score=self.best_score,
                execution_time=execution_time,
            )

            return results

        except Exception as e:
            self.logger.error("MCTS search failed", error=str(e))
            raise

    async def _run_iteration(self, data_context: DataContext) -> None:
        """Run a single MCTS iteration.

        Args:
            data_context: Data context.
        """
        try:
            # 1. Selection: Navigate to leaf node
            selected_node = self._select_node()

            # 2. Expansion: Generate new feature and create child node
            child_node = await self._expand_node(selected_node, data_context)

            if child_node:
                # 3. Simulation: Evaluate the new feature set
                evaluation = await self._simulate_node(child_node, data_context)

                # 4. Backpropagation: Update tree with results
                self._backpropagate(child_node, evaluation.overall_score)

                # Track evaluation
                self.feature_evaluations.append(evaluation)

                # Update best node if needed
                if evaluation.overall_score > self.best_score:
                    self.best_score = evaluation.overall_score
                    self.best_node = child_node

                    self.logger.info(
                        "New best score found",
                        score=self.best_score,
                        node_id=child_node.node_id,
                        iteration=self.iteration_count,
                    )

        except Exception as e:
            self.logger.error(
                "MCTS iteration failed", error=str(e), iteration=self.iteration_count
            )

    def _select_node(self) -> MCTSNode:
        """Select a node for expansion using UCB.

        Returns:
            Selected node for expansion.
        """
        current = self.root_node

        # Navigate to a leaf node using UCB
        while current and not current.is_leaf and not current.is_terminal:
            next_node = current.select_best_child(self.config.mcts.exploration_factor)
            if next_node is None:
                break
            current = next_node

        return current or self.root_node

    async def _expand_node(
        self, node: MCTSNode, data_context: DataContext
    ) -> Optional[MCTSNode]:
        """Expand a node by generating a new feature.

        Args:
            node: Node to expand.
            data_context: Data context.

        Returns:
            New child node or None if expansion failed.
        """
        try:
            # Check if we should stop expanding (max depth)
            if node.depth >= self.config.mcts.max_depth:
                node.set_terminal("max_depth")
                return None

            # Build action context
            current_features = self._get_feature_set_for_node(node)
            action_context = ActionContext(
                current_features=current_features,
                performance_history=self._get_performance_history_for_path(node),
                available_actions=list(MCTSAction),
                max_features=10,  # Could be configurable
                max_cost=100.0,  # Could be configurable
            )

            # Generate new feature using agent
            agent_context = {
                "data_context": data_context,
                "action_context": action_context,
            }

            result = await self.feature_agent.execute(agent_context)

            if not result.get("success"):
                self.logger.warning(
                    "Feature generation failed", error=result.get("error")
                )
                return None

            feature = result["feature"]
            action = result["action"]

            # Create child node
            child_node = node.add_child(feature)

            # Store action information
            child_node.action_taken = action

            self.logger.debug(
                "Node expanded",
                parent_id=node.node_id,
                child_id=child_node.node_id,
                action=action,
                feature_name=feature.name,
            )

            return child_node

        except Exception as e:
            self.logger.error("Node expansion failed", error=str(e))
            return None

    async def _simulate_node(
        self, node: MCTSNode, data_context: DataContext
    ) -> FeatureEvaluation:
        """Simulate (evaluate) a node's feature set.

        Args:
            node: Node to evaluate.
            data_context: Data context.

        Returns:
            Feature evaluation result.
        """
        try:
            # Get complete feature set for this node
            feature_set = self._get_feature_set_for_node(node)

            # Execute all features
            feature_results = await self.feature_executor.execute_feature_set(
                features=feature_set.features,
                data_context=data_context,
                target_split="validation",  # Use validation for evaluation
            )

            # Evaluate feature set
            evaluation = await self.feature_evaluator.evaluate_feature_set(
                feature_set=feature_set,
                feature_results=feature_results,
                data_context=data_context,
                iteration=self.iteration_count,
            )

            # Store evaluation in node
            node.evaluation_score = evaluation.overall_score

            # Record evaluation in performance tracker
            self.performance_tracker.record_evaluation(evaluation)

            self.logger.debug(
                "Node simulated",
                node_id=node.node_id,
                score=evaluation.overall_score,
                feature_count=len(feature_set.features),
            )

            return evaluation

        except Exception as e:
            self.logger.error("Node simulation failed", error=str(e))
            # Return a default evaluation with low score
            return FeatureEvaluation(
                feature_set=self._get_feature_set_for_node(node),
                metrics=FeatureMetrics(extraction_time=0.0),
                overall_score=0.0,
                fold_id=data_context.fold_id,
                iteration=self.iteration_count,
                evaluation_time=0.0,
            )

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """Backpropagate evaluation score up the tree.

        Args:
            node: Starting node for backpropagation.
            score: Score to propagate.
        """
        node.backpropagate(score)

    def _get_feature_set_for_node(self, node: MCTSNode) -> FeatureSet:
        """Get the complete feature set for a node (including path from root).

        Args:
            node: Target node.

        Returns:
            Complete feature set.
        """
        features = node.get_feature_sequence()

        # Determine action taken (default to ADD for root)
        action_str = getattr(node, "action_taken", "add")
        action = MCTSAction(action_str.lower()) if action_str else MCTSAction.ADD

        # Get parent feature names
        parent_features = None
        if node.parent and node.parent.feature:
            parent_path = node.parent.get_feature_sequence()
            parent_features = [f.name for f in parent_path]

        return FeatureSet(
            features=features,
            action_taken=action,
            parent_features=parent_features,
        )

    def _get_performance_history_for_path(
        self, node: MCTSNode
    ) -> List[FeatureEvaluation]:
        """Get performance history for the path to a node.

        Args:
            node: Target node.

        Returns:
            Performance history for the path.
        """
        # For now, return recent evaluations
        # In a more sophisticated implementation, this would track path-specific history
        return (
            self.feature_evaluations[-5:]
            if len(self.feature_evaluations) >= 5
            else self.feature_evaluations
        )

    async def _generate_results(
        self, data_context: DataContext, execution_time: float
    ) -> Dict[str, Any]:
        """Generate final search results.

        Args:
            data_context: Data context.
            execution_time: Total execution time.

        Returns:
            Results dictionary.
        """
        results = {
            "best_score": self.best_score,
            "total_iterations": self.iteration_count,
            "execution_time": execution_time,
            "fold_id": data_context.fold_id,
        }

        if self.best_node:
            best_feature_set = self._get_feature_set_for_node(self.best_node)
            results.update(
                {
                    "best_node_id": self.best_node.node_id,
                    "best_features": [f.dict() for f in best_feature_set.features],
                    "best_feature_count": len(best_feature_set.features),
                    "best_node_depth": self.best_node.depth,
                }
            )

        # Add tree statistics
        results["tree_stats"] = {
            "total_nodes": self._count_tree_nodes(),
            "max_depth": self._get_max_tree_depth(),
            "avg_branching_factor": self._get_avg_branching_factor(),
        }

        return results

    def _count_tree_nodes(self) -> int:
        """Count total nodes in the tree."""
        if not self.root_node:
            return 0

        count = 0
        stack = [self.root_node]

        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)

        return count

    def _get_max_tree_depth(self) -> int:
        """Get maximum depth of the tree."""
        if not self.root_node:
            return 0

        max_depth = 0
        stack = [(self.root_node, 0)]

        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)

            for child in node.children:
                stack.append((child, depth + 1))

        return max_depth

    def _get_avg_branching_factor(self) -> float:
        """Get average branching factor of the tree."""
        if not self.root_node:
            return 0.0

        total_children = 0
        internal_nodes = 0
        stack = [self.root_node]

        while stack:
            node = stack.pop()
            if node.children:
                total_children += len(node.children)
                internal_nodes += 1
                stack.extend(node.children)

        return total_children / internal_nodes if internal_nodes > 0 else 0.0

    async def get_tree_visualization_data(self) -> Dict[str, Any]:
        """Get tree data for visualization.

        Returns:
            Tree visualization data.
        """
        if not self.root_node:
            return {"nodes": [], "edges": []}

        nodes = []
        edges = []
        stack = [self.root_node]

        while stack:
            node = stack.pop()

            # Add node data
            nodes.append(node.to_dict())

            # Add edges to children
            for child in node.children:
                edges.append(
                    {
                        "parent_id": node.node_id,
                        "child_id": child.node_id,
                        "score": child.best_score,
                    }
                )
                stack.append(child)

        return {
            "nodes": nodes,
            "edges": edges,
            "best_node_id": self.best_node.node_id if self.best_node else None,
        }

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        await self.feature_agent.cleanup()
        if self.experiment_tracker and self.experiment_tracker._experiment_id:
            # Create a dummy result for cleanup if needed
            dummy_result = ExperimentResult(
                experiment_id=self.experiment_tracker._experiment_id,
                experiment_name="cleanup",
                best_node_id=None,
                best_score=self.best_score,
                best_feature="",
                total_iterations=self.iteration_count,
                execution_time=0.0,
            )
            self.experiment_tracker.finish_experiment(dummy_result)

        self.logger.info("MCTS orchestrator cleanup completed")
