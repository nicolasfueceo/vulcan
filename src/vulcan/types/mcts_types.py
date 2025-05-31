"""MCTS type definitions for VULCAN system."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MCTSNode(BaseModel):
    """MCTS node representing a cumulative feature set."""

    node_id: str = Field(..., description="Unique node identifier")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    children: List[str] = Field(default_factory=list, description="Child node IDs")

    # Feature information
    new_feature: Optional[str] = Field(
        None, description="New feature added at this node"
    )
    feature_set: List[str] = Field(
        default_factory=list, description="Cumulative features from root to this node"
    )

    # Evaluation metrics
    evaluation_score: float = Field(
        0.0, description="Evaluation score of cumulative feature set"
    )
    visits: int = Field(default=0, description="Number of visits")

    # Tree structure
    depth: int = Field(0, description="Node depth in tree")
    is_terminal: bool = Field(default=False, description="Whether node is terminal")
    is_expanded: bool = Field(
        default=False, description="Whether node has been expanded"
    )

    # Additional metadata
    action_type: Optional[str] = Field(None, description="'explore' or 'exploit'")
    parent_feature_mutated: Optional[str] = Field(
        None, description="Which feature was mutated (for exploit)"
    )


class MCTSEdge(BaseModel):
    """MCTS edge model."""

    parent_id: str = Field(..., description="Parent node ID")
    child_id: str = Field(..., description="Child node ID")
    score: float = Field(..., description="Edge score")


class MCTSTree(BaseModel):
    """MCTS tree model."""

    nodes: Dict[str, MCTSNode] = Field(..., description="Tree nodes")
    edges: List[MCTSEdge] = Field(..., description="Tree edges")
    total_nodes: int = Field(..., description="Total number of nodes")
    max_depth: int = Field(..., description="Maximum tree depth")


class TreeNodeData(MCTSNode):
    """Extended node data for visualization."""

    x: Optional[float] = Field(None, description="X coordinate")
    y: Optional[float] = Field(None, description="Y coordinate")
    children_nodes: Optional[List["TreeNodeData"]] = Field(
        None, description="Child nodes"
    )
    parent_node: Optional["TreeNodeData"] = Field(None, description="Parent node")


class TreeLayoutConfig(BaseModel):
    """Tree layout configuration."""

    width: int = Field(default=800, description="Layout width")
    height: int = Field(default=600, description="Layout height")
    node_radius: int = Field(default=20, description="Node radius")
    link_distance: int = Field(default=100, description="Link distance")
    charge_strength: int = Field(default=-300, description="Charge strength")


# Enable forward references
TreeNodeData.model_rebuild()
