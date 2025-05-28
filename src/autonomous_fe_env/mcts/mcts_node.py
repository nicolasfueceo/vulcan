"""MCTS Node implementation for feature engineering."""

import math
import random
import uuid
from typing import Any, Dict, List, Optional, Set

from ..feature import FeatureDefinition


class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search.

    Attributes:
        state_features: List of features defining the state this node represents.
        parent: The parent node (None for the root).
        children: List of child nodes.
        visits: Number of times this node has been visited during simulation.
        value: Total reward accumulated through this node (Q value).
        feature_that_led_here: The feature added to the parent state to reach this state.
        score: The actual evaluation score of the state_features (cached).
        node_id: Unique identifier for the node (useful for parallel processing).
        metadata: Additional metadata about this node.
    """

    def __init__(
        self,
        state_features: List[FeatureDefinition],
        parent: Optional["MCTSNode"] = None,
        feature_that_led_here: Optional[FeatureDefinition] = None,
        score: float = -float("inf"),
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an MCTS node.

        Args:
            state_features: List of features defining this state
            parent: Parent node (None for root)
            feature_that_led_here: Feature that was added to reach this state
            score: Evaluation score for this state
            node_id: Unique ID for this node (generated if not provided)
            metadata: Additional metadata for this node
        """
        self.state_features: List[FeatureDefinition] = state_features
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0  # Q value
        self.feature_that_led_here: Optional[FeatureDefinition] = feature_that_led_here
        self.score: float = score  # Cached evaluation score for this state
        self.node_id: str = node_id or str(uuid.uuid4())
        self.metadata: Dict[str, Any] = metadata or {}

        # Keep track of feature names already tried as children
        self.tried_feature_names: Set[str] = set()
        if self.feature_that_led_here:
            self.tried_feature_names.add(self.feature_that_led_here.name)

        # Additional properties for parallel exploration
        self.is_being_expanded: bool = False
        self.is_terminal: bool = False
        self.depth: int = 0 if parent is None else parent.depth + 1

    def is_fully_expanded(self) -> bool:
        """
        Check if this node is fully expanded.

        A node is considered fully expanded if it has been marked as terminal,
        or if all possible feature additions have been tried.

        Returns:
            True if fully expanded, False otherwise
        """
        if self.is_terminal:
            return True

        # In practice, this will always return False unless we have a finite set of features
        # and have tried them all. We would need to track the full set of possible features.
        return False

    def select_child_uct(self, exploration_factor: float) -> "MCTSNode":
        """
        Select a child node using the UCT formula.

        Args:
            exploration_factor: Factor for controlling exploration vs exploitation

        Returns:
            The selected child node
        """
        best_score = -float("inf")
        best_child = None

        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children
                uct_score = float("inf")
            else:
                exploit_term = child.value / child.visits
                explore_term = exploration_factor * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                uct_score = exploit_term + explore_term

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        if best_child is None:
            # This should ideally not happen if node has children, but as fallback:
            return random.choice(self.children) if self.children else self

        return best_child

    def add_child(self, child_node: "MCTSNode") -> None:
        """
        Add a child node.

        Args:
            child_node: The child node to add
        """
        # Track the feature name that led to this child
        if child_node.feature_that_led_here:
            self.tried_feature_names.add(child_node.feature_that_led_here.name)

        self.children.append(child_node)

    def update(self, reward: float) -> None:
        """
        Update the node's visit count and value.

        Args:
            reward: The reward to add to this node's value
        """
        self.visits += 1
        self.value += reward

    def has_tried_feature(self, feature_name: str) -> bool:
        """
        Check if a feature has already been tried from this node.

        Args:
            feature_name: Name of the feature to check

        Returns:
            True if the feature has been tried, False otherwise
        """
        return feature_name in self.tried_feature_names

    def get_state_feature_names(self) -> List[str]:
        """
        Get the names of all features in this node's state.

        Returns:
            List of feature names
        """
        return [feature.name for feature in self.state_features]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary for serialization.

        Returns:
            Dictionary representation of the node
        """
        return {
            "node_id": self.node_id,
            "visits": self.visits,
            "value": self.value,
            "score": self.score,
            "metadata": self.metadata,
            "tried_feature_names": list(self.tried_feature_names),
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "state_features": [feature.to_dict() for feature in self.state_features],
            "feature_that_led_here": self.feature_that_led_here.to_dict()
            if self.feature_that_led_here
            else None,
            "children": [child.node_id for child in self.children],
            "parent": self.parent.node_id if self.parent else None,
        }

    @staticmethod
    def from_dict(
        data: Dict[str, Any], nodes_dict: Optional[Dict[str, "MCTSNode"]] = None
    ) -> "MCTSNode":
        """
        Create a node from dictionary representation.

        Args:
            data: Dictionary representation of the node
            nodes_dict: Dictionary mapping node IDs to nodes (for reconstructing tree structure)

        Returns:
            Reconstructed MCTSNode
        """
        from ..feature import FeatureDefinition

        # Dictionary to track all nodes if not provided
        if nodes_dict is None:
            nodes_dict = {}

        # Create the node with basic properties
        feature_that_led_here = None
        if data.get("feature_that_led_here"):
            feature_that_led_here = FeatureDefinition.from_dict(
                data["feature_that_led_here"]
            )

        state_features = [
            FeatureDefinition.from_dict(feature_data)
            for feature_data in data.get("state_features", [])
        ]

        node = MCTSNode(
            state_features=state_features,
            feature_that_led_here=feature_that_led_here,
            score=data.get("score", -float("inf")),
            node_id=data.get("node_id"),
            metadata=data.get("metadata", {}),
        )

        # Set additional properties
        node.visits = data.get("visits", 0)
        node.value = data.get("value", 0.0)
        node.is_terminal = data.get("is_terminal", False)
        node.depth = data.get("depth", 0)
        node.tried_feature_names = set(data.get("tried_feature_names", []))

        # Add to nodes dictionary
        nodes_dict[node.node_id] = node

        return node

    @staticmethod
    def reconstruct_tree(nodes_data: List[Dict[str, Any]]) -> Optional["MCTSNode"]:
        """
        Reconstruct a tree from a list of node dictionaries.

        Args:
            nodes_data: List of dictionary representations of nodes

        Returns:
            The root node of the reconstructed tree, or None if reconstruction failed
        """
        # Create all nodes first
        nodes_dict = {}
        for node_data in nodes_data:
            MCTSNode.from_dict(node_data, nodes_dict)

        # Set parent and children relationships
        for node_data in nodes_data:
            node_id = node_data["node_id"]
            node = nodes_dict.get(node_id)

            if not node:
                continue

            # Set parent
            parent_id = node_data.get("parent")
            if parent_id and parent_id in nodes_dict:
                node.parent = nodes_dict[parent_id]

            # Set children
            for child_id in node_data.get("children", []):
                if child_id in nodes_dict:
                    child = nodes_dict[child_id]
                    if child not in node.children:  # Avoid duplicates
                        node.children.append(child)

        # Find the root node (node with no parent)
        root_nodes = [node for node in nodes_dict.values() if node.parent is None]

        if not root_nodes:
            return None

        return root_nodes[0]  # Return the first root node found

    def __repr__(self) -> str:
        """String representation of the node."""
        feature_names = [f.name for f in self.state_features]
        return f"Node(ID: {self.node_id[:8]}, Features: {feature_names}, Visits: {self.visits}, Value: {self.value:.4f}, Score: {self.score:.4f})"
