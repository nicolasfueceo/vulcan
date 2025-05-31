"""MCTS tree implementation for VULCAN."""

import uuid
from typing import Dict, List, Optional, Set, Tuple

from vulcan.types import MCTSNode


class MCTSTree:
    """Monte Carlo Tree Search tree structure for feature engineering."""

    def __init__(self):
        """Initialize empty tree."""
        self.nodes: Dict[str, MCTSNode] = {}
        self.root_id: Optional[str] = None

    def create_root(self) -> MCTSNode:
        """Create root node with empty feature set."""
        node_id = str(uuid.uuid4())
        root = MCTSNode(
            node_id=node_id,
            parent_id=None,
            children=[],
            new_feature=None,
            feature_set=[],  # Root has no features
            evaluation_score=0.0,
            visits=0,
            depth=0,
            is_terminal=False,
            is_expanded=False,
            action_type=None,
        )
        self.nodes[node_id] = root
        self.root_id = node_id
        return root

    def add_exploration_node(self, parent_id: str, new_feature: str) -> MCTSNode:
        """Add a new node by exploring (adding a new feature).

        Args:
            parent_id: ID of parent node
            new_feature: New feature to add

        Returns:
            New node with cumulative feature set
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")

        parent = self.nodes[parent_id]
        node_id = str(uuid.uuid4())

        # Cumulative feature set = parent's features + new feature
        cumulative_features = parent.feature_set + [new_feature]

        node = MCTSNode(
            node_id=node_id,
            parent_id=parent_id,
            children=[],
            new_feature=new_feature,
            feature_set=cumulative_features,
            evaluation_score=0.0,
            visits=0,
            depth=parent.depth + 1,
            is_terminal=False,
            is_expanded=False,
            action_type="explore",
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)

        return node

    def add_exploitation_node(
        self, parent_id: str, mutated_feature: str, mutated_index: int
    ) -> MCTSNode:
        """Add a new node by exploiting (mutating an existing feature).

        Args:
            parent_id: ID of parent node
            mutated_feature: The mutated version of the feature
            mutated_index: Index of the feature that was mutated in parent's feature set

        Returns:
            New node with mutated feature set
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")

        parent = self.nodes[parent_id]
        node_id = str(uuid.uuid4())

        # Create new feature set with the mutated feature
        cumulative_features = parent.feature_set.copy()
        if mutated_index >= len(cumulative_features):
            raise ValueError(f"Invalid mutation index {mutated_index}")

        original_feature = cumulative_features[mutated_index]
        cumulative_features[mutated_index] = mutated_feature

        node = MCTSNode(
            node_id=node_id,
            parent_id=parent_id,
            children=[],
            new_feature=mutated_feature,
            feature_set=cumulative_features,
            evaluation_score=0.0,
            visits=0,
            depth=parent.depth + 1,
            is_terminal=False,
            is_expanded=False,
            action_type="exploit",
            parent_feature_mutated=original_feature,
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)

        return node

    def get_cumulative_features(self, node_id: str) -> List[str]:
        """Get the cumulative feature set for a node.

        Args:
            node_id: Target node ID

        Returns:
            List of all features from root to this node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        return self.nodes[node_id].feature_set.copy()

    def get_node(self, node_id: str) -> MCTSNode:
        """Get node by ID."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        return self.nodes[node_id]

    def get_path_to_node(self, node_id: str) -> List[str]:
        """Get path from root to given node.

        Args:
            node_id: Target node ID

        Returns:
            List of node IDs from root to target
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        path = []
        current = self.nodes[node_id]

        while current:
            path.append(current.node_id)
            if current.parent_id:
                current = self.nodes[current.parent_id]
            else:
                break

        return list(reversed(path))

    def get_tree_state(self) -> Dict[str, Dict]:
        """Get current tree state for visualization.

        Returns:
            Dictionary of all nodes as dicts for JSON serialization
        """
        return {node_id: node.dict() for node_id, node in self.nodes.items()}

    def get_leaves(self) -> Set[str]:
        """Get all leaf node IDs.

        Returns:
            Set of leaf node IDs
        """
        return {node_id for node_id, node in self.nodes.items() if not node.children}

    def get_unexpanded_nodes(self) -> Set[str]:
        """Get all unexpanded node IDs.

        Returns:
            Set of unexpanded node IDs
        """
        return {node_id for node_id, node in self.nodes.items() if not node.is_expanded}

    def update_node_score(self, node_id: str, score: float):
        """Update node's evaluation score.

        Args:
            node_id: Node to update
            score: New evaluation score
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        node.evaluation_score = score
        node.visits += 1

    def get_best_path(self) -> Tuple[List[str], float]:
        """Get the path to the best scoring node.

        Returns:
            Tuple of (path node IDs, best score)
        """
        if not self.nodes:
            return [], 0.0

        best_node_id = max(
            self.nodes.keys(), key=lambda nid: self.nodes[nid].evaluation_score
        )

        path = self.get_path_to_node(best_node_id)
        score = self.nodes[best_node_id].evaluation_score

        return path, score

    def mark_expanded(self, node_id: str):
        """Mark a node as expanded."""
        if node_id in self.nodes:
            self.nodes[node_id].is_expanded = True
