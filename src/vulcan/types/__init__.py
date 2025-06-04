"""Type definitions for VULCAN system."""

# Additional types for exploration visualization
from dataclasses import dataclass
from typing import Dict, List, Optional

from .api_types import (
    ApiResponse,
    ErrorResponse,
    HealthResponse,
    StatusResponse,
)
from .config_types import (
    ApiConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    LLMConfig,
    LoggingConfig,
    MCTSConfig,
    VulcanConfig,
)
from .experiment_types import (
    ExperimentRequest,
    ExperimentResult,
    ExperimentStatus,
    HyperparameterSweepRequest,
    SweepResult,
)
from .feature_types import (
    ActionContext,
    DataContext,
    FeatureDefinition,
    FeatureEvaluation,
    FeatureMetrics,
    FeatureSet,
    FeatureType,
    FeatureValue,
    MCTSAction,
)
from .llm_schemas import (
    ExpectedCostEnum,
    FeatureGenerationResponse,
    FeatureTypeEnum,
    LLMFeatureExtractionResponse,
    MCTSActionEnum,
)
from .mcts_types import (
    MCTSEdge,
    MCTSNode,
    MCTSTree,
    TreeLayoutConfig,
    TreeNodeData,
)
from .websocket_types import (
    WebSocketMessage,
    WebSocketMessageType,
)


@dataclass
class LLMInteraction:
    """Record of an LLM interaction."""

    iteration: int
    feature_name: str
    prompt: str
    response: str
    timestamp: str
    reflection: Optional[str] = None  # Optional reflection on the generated feature


@dataclass
class ExplorationState:
    """Current state of the exploration process."""

    current_node_id: str
    tree: Dict[str, MCTSNode]
    llm_history: List[LLMInteraction]
    best_score: float
    total_iterations: int
    current_path: List[str]


__all__ = [
    # API Types
    "ApiResponse",
    "ErrorResponse",
    "HealthResponse",
    "StatusResponse",
    # Config Types
    "ApiConfig",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "LLMConfig",
    "LoggingConfig",
    "MCTSConfig",
    "VulcanConfig",
    # Experiment Types
    "ExperimentRequest",
    "ExperimentResult",
    "ExperimentStatus",
    "HyperparameterSweepRequest",
    "SweepResult",
    # Feature Types
    "ActionContext",
    "DataContext",
    "FeatureDefinition",
    "FeatureEvaluation",
    "FeatureMetrics",
    "FeatureSet",
    "FeatureType",
    "FeatureValue",
    "MCTSAction",
    # LLM Schema Types
    "ExpectedCostEnum",
    "FeatureGenerationResponse",
    "FeatureTypeEnum",
    "LLMFeatureExtractionResponse",
    "MCTSActionEnum",
    # MCTS Types
    "MCTSEdge",
    "MCTSNode",
    "MCTSTree",
    "TreeLayoutConfig",
    "TreeNodeData",
    # WebSocket Types
    "WebSocketMessage",
    "WebSocketMessageType",
    # Exploration Types
    "LLMInteraction",
    "ExplorationState",
]
