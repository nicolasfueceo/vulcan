"""MCTS Node implementation for VULCAN feature engineering."""

import math
import uuid
from typing import Any, Dict, List, Optional

import structlog

from vulcan.types import FeatureDefinition
from vulcan.utils import get_vulcan_logger

logger = structlog.get_logger(__name__)

# Constants
DEFAULT_UCB_CONSTANT = 1.414
MIN_VISITS_FOR_UCB = 1


class MCTSNode:
    """Monte Carlo Tree Search node for feature engineering exploration."""

    def __init__(
        self,
        feature: Optional[FeatureDefinition] = None,
        parent: Optional["MCTSNode"] = None,
        depth: int = 0,
    ) -> None:
        """Initialize MCTS node.

        Args:
            feature: Feature definition for this node.
            parent: Parent node in the tree.
            depth: Depth of this node in the tree.
        """
        self.node_id = str(uuid.uuid4())
        self.feature = feature
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.depth = depth

        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.best_score = 0.0

        # Feature engineering specific
        self.is_terminal = False
        self.evaluation_score: Optional[float] = None
        self.reflection: Optional[str] = None

        # Enhanced MCTS fields
        self.search_mode: Optional[str] = None  # 'exploration' or 'exploitation'
        self.action_taken: Optional[str] = None  # Action that led to this node

        # Metadata
        self.created_at = None
        self.last_visited = None

        self.logger = get_vulcan_logger(__name__).bind_node(self.node_id)

    @property
    def average_reward(self) -> float:
        """Calculate average reward for this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this node is the root (no parent)."""
        return self.parent is None

    def add_child(self, feature: FeatureDefinition) -> "MCTSNode":
        """Add a child node with the given feature.

        Args:
            feature: Feature definition for the child node.

        Returns:
            The newly created child node.
        """
        child = MCTSNode(
            feature=feature,
            parent=self,
            depth=self.depth + 1,
        )
        self.children.append(child)

        self.logger.debug(
            "Added child node",
            child_id=child.node_id,
            feature_name=feature.name,
            children_count=len(self.children),
        )

        return child

    def update_statistics(self, reward: float) -> None:
        """Update node statistics with new reward.

        Args:
            reward: Reward value to incorporate.
        """
        self.visits += 1
        self.total_reward += reward
        self.best_score = max(self.best_score, reward)

        self.logger.debug(
            "Updated node statistics",
            visits=self.visits,
            total_reward=self.total_reward,
            average_reward=self.average_reward,
            best_score=self.best_score,
        )

    def calculate_ucb_score(
        self, exploration_constant: float = DEFAULT_UCB_CONSTANT
    ) -> float:
        """Calculate Upper Confidence Bound (UCB) score for node selection.

        Args:
            exploration_constant: Exploration vs exploitation balance parameter.

        Returns:
            UCB score for this node.
        """
        if self.visits < MIN_VISITS_FOR_UCB:
            return float("inf")  # Prioritize unvisited nodes

        if self.parent is None or self.parent.visits == 0:
            return self.average_reward

        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        ucb_score = exploitation + exploration

        self.logger.debug(
            "Calculated UCB score",
            exploitation=exploitation,
            exploration=exploration,
            ucb_score=ucb_score,
        )

        return ucb_score

    def select_best_child(
        self, exploration_constant: float = DEFAULT_UCB_CONSTANT
    ) -> Optional["MCTSNode"]:
        """Select the best child node using UCB criterion.

        Args:
            exploration_constant: Exploration vs exploitation balance parameter.

        Returns:
            Best child node or None if no children.
        """
        if not self.children:
            return None

        best_child = max(
            self.children,
            key=lambda child: child.calculate_ucb_score(exploration_constant),
        )

        self.logger.debug(
            "Selected best child",
            best_child_id=best_child.node_id,
            ucb_score=best_child.calculate_ucb_score(exploration_constant),
        )

        return best_child

    def backpropagate(self, reward: float) -> None:
        """Backpropagate reward up the tree to the root.

        Args:
            reward: Reward value to propagate.
        """
        current_node = self
        while current_node is not None:
            current_node.update_statistics(reward)
            current_node = current_node.parent

        self.logger.debug("Backpropagated reward", reward=reward)

    def get_path_to_root(self) -> List["MCTSNode"]:
        """Get the path from this node to the root.

        Returns:
            List of nodes from this node to root.
        """
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        return path

    def get_feature_sequence(self) -> List[FeatureDefinition]:
        """Get the sequence of features from root to this node.

        Returns:
            List of feature definitions in the path.
        """
        path = self.get_path_to_root()
        path.reverse()  # Root to current node

        features = []
        for node in path:
            if node.feature is not None:
                features.append(node.feature)

        return features

    def set_terminal(self, reason: str = "max_depth") -> None:
        """Mark this node as terminal.

        Args:
            reason: Reason for marking as terminal.
        """
        self.is_terminal = True
        self.logger.info("Node marked as terminal", reason=reason)

    def add_reflection(self, reflection: str) -> None:
        """Add LLM reflection to this node.

        Args:
            reflection: Reflection text from LLM.
        """
        self.reflection = reflection
        self.logger.debug("Added reflection to node", reflection_length=len(reflection))

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "feature": self.feature.dict() if self.feature else None,
            "depth": self.depth,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "best_score": self.best_score,
            "is_terminal": self.is_terminal,
            "is_leaf": self.is_leaf,
            "children_count": len(self.children),
            "evaluation_score": self.evaluation_score,
            "reflection": self.reflection,
        }

    def __repr__(self) -> str:
        """String representation of the node."""
        feature_name = self.feature.name if self.feature else "root"
        return (
            f"MCTSNode(id={self.node_id[:8]}, "
            f"feature={feature_name}, "
            f"depth={self.depth}, "
            f"visits={self.visits}, "
            f"avg_reward={self.average_reward:.3f})"
        )
