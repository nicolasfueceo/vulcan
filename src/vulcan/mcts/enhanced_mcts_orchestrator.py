"""Enhanced MCTS Orchestrator for VULCAN with improved exploration/exploitation and multi-feature nodes."""

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog

from vulcan.agents import FeatureAgent
from vulcan.evaluation import FeatureEvaluator
from vulcan.features import FeatureExecutor
from vulcan.types import (
    ActionContext,
    DataContext,
    FeatureEvaluation,
    FeatureMetrics,
    FeatureSet,
    MCTSAction,
    VulcanConfig,
)
from vulcan.utils import ExperimentTracker, PerformanceTracker, get_vulcan_logger

from .mcts_node import MCTSNode

logger = structlog.get_logger(__name__)


class SearchMode(Enum):
    """MCTS search mode for exploration vs exploitation."""

    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"


class EnhancedMCTSOrchestrator:
    """Enhanced MCTS orchestrator with improved exploration/exploitation and multi-feature nodes."""

    def __init__(
        self,
        config: VulcanConfig,
        performance_tracker: Optional[PerformanceTracker] = None,
        websocket_callback: Optional[callable] = None,
    ) -> None:
        """Initialize enhanced MCTS orchestrator.

        Args:
            config: VULCAN configuration.
            performance_tracker: Optional external performance tracker.
            websocket_callback: Optional callback for real-time tree updates.
        """
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.websocket_callback = websocket_callback

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

        # Enhanced features
        self.features_per_node = getattr(
            config.mcts, "features_per_node", 3
        )  # Generate 3 features per node
        self.exploration_threshold = getattr(
            config.mcts, "exploration_threshold", 0.3
        )  # When to explore vs exploit
        self.feature_diversity_weight = getattr(
            config.mcts, "diversity_weight", 0.5
        )  # How much to weight diversity

        # Feature tracking for diversity
        self.all_generated_features: Set[str] = set()  # Track all feature names/types
        self.feature_categories: Dict[str, int] = {}  # Track feature category usage

        # Tracking
        self.iteration_count = 0
        self.feature_evaluations: List[FeatureEvaluation] = []
        self.tree_updates = []  # Store tree updates for real-time visualization

    async def initialize(self) -> bool:
        """Initialize all components."""
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

            self.logger.info("Enhanced MCTS orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(
                "Failed to initialize enhanced MCTS orchestrator", error=str(e)
            )
            return False

    async def run_search(
        self,
        data_context: DataContext,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run enhanced MCTS search with improved exploration/exploitation."""
        if self.root_node is None:
            await self.initialize()

        max_iterations = max_iterations or self.config.mcts.max_iterations

        self.logger.info(
            "Starting enhanced MCTS search",
            max_iterations=max_iterations,
            features_per_node=self.features_per_node,
            exploration_threshold=self.exploration_threshold,
            fold_id=data_context.fold_id,
        )

        start_time = time.time()

        try:
            for iteration in range(max_iterations):
                self.iteration_count = iteration + 1

                # Determine search mode
                search_mode = self._determine_search_mode(iteration, max_iterations)

                # Run enhanced MCTS iteration
                await self._run_enhanced_iteration(data_context, search_mode)

                # Send real-time tree update
                await self._send_tree_update()

                # Log progress
                if iteration % 10 == 0:
                    self.logger.info(
                        "Enhanced MCTS progress",
                        iteration=iteration + 1,
                        best_score=self.best_score,
                        search_mode=search_mode.value,
                        tree_nodes=self._count_tree_nodes(),
                        feature_diversity=len(self.all_generated_features),
                    )

            execution_time = time.time() - start_time
            results = await self._generate_enhanced_results(
                data_context, execution_time
            )

            self.logger.info(
                "Enhanced MCTS search completed",
                total_iterations=max_iterations,
                best_score=self.best_score,
                execution_time=execution_time,
                unique_features_generated=len(self.all_generated_features),
            )

            return results

        except Exception as e:
            self.logger.error("Enhanced MCTS search failed", error=str(e))
            raise

    def _determine_search_mode(self, iteration: int, max_iterations: int) -> SearchMode:
        """Determine whether to explore or exploit based on search progress."""
        # Early iterations: more exploration
        if iteration < max_iterations * 0.3:
            return SearchMode.EXPLORATION

        # Later iterations: balance based on UCB scores and performance
        if self.best_score < 0.5:  # Low performance, need more exploration
            return SearchMode.EXPLORATION

        # Check if we're getting stuck (no improvement)
        if len(self.feature_evaluations) >= 10:
            recent_scores = [
                eval.overall_score for eval in self.feature_evaluations[-10:]
            ]
            if max(recent_scores) - min(recent_scores) < 0.1:  # Not much variation
                return SearchMode.EXPLORATION

        # Use random exploration with probability
        exploration_prob = max(
            self.exploration_threshold, 1.0 - (iteration / max_iterations)
        )
        return (
            SearchMode.EXPLORATION
            if random.random() < exploration_prob
            else SearchMode.EXPLOITATION
        )

    async def _run_enhanced_iteration(
        self, data_context: DataContext, search_mode: SearchMode
    ) -> None:
        """Run a single enhanced MCTS iteration with multi-feature generation."""
        try:
            # 1. Selection: Navigate to leaf node
            selected_node = self._select_node()

            # 2. Enhanced Expansion: Generate multiple features
            child_nodes = await self._expand_node_with_multiple_features(
                selected_node, data_context, search_mode
            )

            if child_nodes:
                # 3. Evaluation: Evaluate all generated features
                best_evaluation = None
                best_child = None

                for child_node in child_nodes:
                    evaluation = await self._simulate_node(child_node, data_context)

                    if (
                        best_evaluation is None
                        or evaluation.overall_score > best_evaluation.overall_score
                    ):
                        best_evaluation = evaluation
                        best_child = child_node

                    # Track all evaluations
                    self.feature_evaluations.append(evaluation)

                # 4. Backpropagation: Use best score from the batch
                if best_evaluation and best_child:
                    self._backpropagate(best_child, best_evaluation.overall_score)

                    # Update best node if needed
                    if best_evaluation.overall_score > self.best_score:
                        self.best_score = best_evaluation.overall_score
                        self.best_node = best_child

                        self.logger.info(
                            "New best score found",
                            score=self.best_score,
                            node_id=best_child.node_id,
                            iteration=self.iteration_count,
                            search_mode=search_mode.value,
                            features_in_batch=len(child_nodes),
                        )

        except Exception as e:
            self.logger.error(
                "Enhanced MCTS iteration failed",
                error=str(e),
                iteration=self.iteration_count,
                search_mode=search_mode.value,
            )

    def _select_node(self) -> MCTSNode:
        """Select a node for expansion using enhanced UCB."""
        current = self.root_node

        # Navigate to a leaf node using UCB with diversity bonus
        while current and not current.is_leaf and not current.is_terminal:
            next_node = self._select_best_child_with_diversity(current)
            if next_node is None:
                break
            current = next_node

        return current or self.root_node

    def _select_best_child_with_diversity(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select best child with diversity bonus."""
        if not node.children:
            return None

        best_child = None
        best_score = float("-inf")

        for child in node.children:
            # Standard UCB score
            ucb_score = child.calculate_ucb_score(self.config.mcts.exploration_factor)

            # Add diversity bonus
            diversity_bonus = self._calculate_diversity_bonus(child)
            total_score = ucb_score + self.feature_diversity_weight * diversity_bonus

            if total_score > best_score:
                best_score = total_score
                best_child = child

        return best_child

    def _calculate_diversity_bonus(self, node: MCTSNode) -> float:
        """Calculate diversity bonus for a node based on feature uniqueness."""
        if not node.feature:
            return 0.0

        # Bonus for rare feature types
        feature_type = getattr(node.feature, "feature_type", "unknown")
        type_count = self.feature_categories.get(feature_type, 0)
        type_bonus = 1.0 / (1.0 + type_count)

        # Bonus for unique feature names
        feature_name = node.feature.name
        name_bonus = 1.0 if feature_name not in self.all_generated_features else 0.0

        return type_bonus + name_bonus

    async def _expand_node_with_multiple_features(
        self, node: MCTSNode, data_context: DataContext, search_mode: SearchMode
    ) -> List[MCTSNode]:
        """Expand a node by generating multiple diverse features."""
        try:
            # Check if we should stop expanding
            if node.depth >= self.config.mcts.max_depth:
                node.set_terminal("max_depth")
                return []

            child_nodes = []
            current_features = self._get_feature_set_for_node(node)

            for i in range(self.features_per_node):
                # Build action context with search mode
                action_context = self._build_enhanced_action_context(
                    current_features, node, search_mode, attempt=i
                )

                # Generate feature using agent with mode-specific instructions
                agent_context = {
                    "data_context": data_context,
                    "action_context": action_context,
                    "search_mode": search_mode.value,
                    "diversity_requirement": i
                    > 0,  # Require diversity for subsequent features
                    "existing_features_in_batch": [
                        child.feature for child in child_nodes if child.feature
                    ],
                }

                result = await self.feature_agent.execute(agent_context)

                if result.get("success"):
                    feature = result["feature"]
                    action = result["action"]

                    # Create child node
                    child_node = node.add_child(feature)
                    child_node.action_taken = action
                    child_node.search_mode = search_mode

                    # Track feature diversity
                    self.all_generated_features.add(feature.name)
                    feature_type = getattr(feature, "feature_type", "unknown")
                    self.feature_categories[feature_type] = (
                        self.feature_categories.get(feature_type, 0) + 1
                    )

                    child_nodes.append(child_node)

                    self.logger.debug(
                        "Multi-feature expansion",
                        parent_id=node.node_id,
                        child_id=child_node.node_id,
                        feature_name=feature.name,
                        search_mode=search_mode.value,
                        batch_index=i,
                    )

            self.logger.info(
                "Node expanded with multiple features",
                parent_id=node.node_id,
                children_generated=len(child_nodes),
                search_mode=search_mode.value,
            )

            return child_nodes

        except Exception as e:
            self.logger.error("Multi-feature node expansion failed", error=str(e))
            return []

    def _build_enhanced_action_context(
        self,
        current_features: FeatureSet,
        node: MCTSNode,
        search_mode: SearchMode,
        attempt: int,
    ) -> ActionContext:
        """Build enhanced action context based on search mode."""

        if search_mode == SearchMode.EXPLORATION:
            # Exploration mode: encourage diversity, ignore or minimize past features
            performance_history = []  # Empty history for more randomness
            available_actions = list(MCTSAction)  # All actions available
            exploration_bonus = 1.0

            # Add diversity requirements
            context_notes = {
                "exploration_mode": True,
                "encourage_diversity": True,
                "ignore_past_performance": True,
                "feature_categories_to_avoid": list(self.feature_categories.keys())
                if attempt > 0
                else [],
                "generate_novel_features": True,
            }

        else:
            # Exploitation mode: build on successful features
            performance_history = self._get_performance_history_for_path(node)
            available_actions = [MCTSAction.ADD, MCTSAction.MUTATE]  # Focus on building
            exploration_bonus = 0.0

            context_notes = {
                "exploration_mode": False,
                "build_on_success": True,
                "use_past_performance": True,
                "successful_features": [
                    f.name
                    for f in current_features.features
                    if f.name in self.all_generated_features
                ],
                "incremental_improvement": True,
            }

        return ActionContext(
            current_features=current_features,
            performance_history=performance_history,
            available_actions=available_actions,
            max_features=15,  # Allow more features
            max_cost=150.0,
            exploration_bonus=exploration_bonus,
            context_notes=context_notes,
        )

    async def _simulate_node(
        self, node: MCTSNode, data_context: DataContext
    ) -> FeatureEvaluation:
        """Simulate (evaluate) a node's feature set with enhanced tracking."""
        try:
            # Get complete feature set for this node
            feature_set = self._get_feature_set_for_node(node)

            # Execute all features
            feature_results = await self.feature_executor.execute_feature_set(
                features=feature_set.features,
                data_context=data_context,
                target_split="validation",
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
                "Enhanced node simulation",
                node_id=node.node_id,
                score=evaluation.overall_score,
                feature_count=len(feature_set.features),
                search_mode=getattr(node, "search_mode", "unknown"),
            )

            return evaluation

        except Exception as e:
            self.logger.error("Enhanced node simulation failed", error=str(e))
            # Return a default evaluation
            return FeatureEvaluation(
                feature_set=self._get_feature_set_for_node(node),
                metrics=FeatureMetrics(extraction_time=0.0),
                overall_score=0.0,
                fold_id=data_context.fold_id,
                iteration=self.iteration_count,
                evaluation_time=0.0,
            )

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """Backpropagate evaluation score up the tree."""
        node.backpropagate(score)

    async def _send_tree_update(self) -> None:
        """Send real-time tree update via WebSocket."""
        if not self.websocket_callback:
            return

        try:
            tree_data = await self.get_tree_visualization_data()
            update = {
                "type": "tree_update",
                "iteration": self.iteration_count,
                "best_score": self.best_score,
                "tree_data": tree_data,
                "diversity_stats": {
                    "unique_features": len(self.all_generated_features),
                    "feature_categories": dict(self.feature_categories),
                },
                "timestamp": time.time(),
            }

            await self.websocket_callback(update)

        except Exception as e:
            self.logger.error("Failed to send tree update", error=str(e))

    async def get_tree_visualization_data(self) -> Dict[str, Any]:
        """Get enhanced tree data for visualization."""
        if not self.root_node:
            return {"nodes": [], "edges": [], "stats": {}}

        nodes = []
        edges = []

        # Traverse tree and collect node data
        stack = [self.root_node]
        while stack:
            node = stack.pop()

            # Node data with enhanced information
            node_data = {
                "id": node.node_id,
                "parent_id": node.parent.node_id if node.parent else None,
                "feature_name": node.feature.name if node.feature else "root",
                "depth": node.depth,
                "visits": node.visits,
                "value": node.average_reward,
                "best_score": node.best_score,
                "ucb_score": node.calculate_ucb_score(
                    self.config.mcts.exploration_factor
                ),
                "is_terminal": node.is_terminal,
                "is_leaf": node.is_leaf,
                "search_mode": getattr(node, "search_mode", None),
                "action_taken": getattr(node, "action_taken", None),
                "evaluation_score": node.evaluation_score,
                "feature_sequence": [f.name for f in node.get_feature_sequence()],
                "reflections": [node.reflection] if node.reflection else [],
            }

            nodes.append(node_data)

            # Add edges
            for child in node.children:
                edges.append(
                    {
                        "source": node.node_id,
                        "target": child.node_id,
                    }
                )
                stack.append(child)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "max_depth": self._get_max_tree_depth(),
                "avg_branching_factor": self._get_avg_branching_factor(),
                "best_score": self.best_score,
                "unique_features": len(self.all_generated_features),
                "feature_categories": dict(self.feature_categories),
            },
        }

    def _get_feature_set_for_node(self, node: MCTSNode) -> FeatureSet:
        """Get the complete feature set for a node."""
        features = node.get_feature_sequence()
        action_str = getattr(node, "action_taken", "add")
        action = MCTSAction(action_str.lower()) if action_str else MCTSAction.ADD

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
        """Get performance history for the path to a node."""
        return (
            self.feature_evaluations[-10:]
            if len(self.feature_evaluations) >= 10
            else self.feature_evaluations
        )

    async def _generate_enhanced_results(
        self, data_context: DataContext, execution_time: float
    ) -> Dict[str, Any]:
        """Generate enhanced final search results."""
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
                    "best_node_search_mode": getattr(
                        self.best_node, "search_mode", None
                    ),
                }
            )

        # Enhanced tree statistics
        results["tree_stats"] = {
            "total_nodes": self._count_tree_nodes(),
            "max_depth": self._get_max_tree_depth(),
            "avg_branching_factor": self._get_avg_branching_factor(),
            "features_per_node": self.features_per_node,
        }

        # Diversity statistics
        results["diversity_stats"] = {
            "unique_features_generated": len(self.all_generated_features),
            "feature_categories": dict(self.feature_categories),
            "diversity_score": len(self.all_generated_features)
            / max(1, self.iteration_count),
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

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.feature_agent.cleanup()
        await self.feature_executor.cleanup()
        # FeatureEvaluator doesn't have cleanup method, so we skip it
        if hasattr(self.feature_evaluator, "cleanup"):
            await self.feature_evaluator.cleanup()
        self.logger.info("Enhanced MCTS Orchestrator cleaned up")
