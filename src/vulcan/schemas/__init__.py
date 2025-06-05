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
    VulcanConfig,
)
from .feature_types import (
    FeatureEvaluation,
    FeatureMetrics,
)
from .evolution_types import (
    FeatureCandidate,
    GenerationStats,
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
    EvolutionAction,
    FeatureDefinition,
    FeatureSet,
    FeatureType,
    FeatureValue,
)
from .llm_schemas import (
    EvolutionActionEnum,
    ExpectedCostEnum,
    FeatureGenerationResponse,
    FeatureTypeEnum,
    LLMFeatureExtractionResponse,
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
    "VulcanConfig",
    # Data Types
    "DataContext",
    "FeatureDefinition",
    "FeatureSet",
    "FeatureType",
    # Evaluation Types
    "FeatureEvaluation",
    "FeatureMetrics",
    # Evolution Types
    "FeatureCandidate",
    "GenerationStats",
    # Experiment Types
    "ExperimentRequest",
    "ExperimentResult",
    "ExperimentStatus",
    "HyperparameterSweepRequest",
    "SweepResult",
    # Feature Types
    "ActionContext",
    "EvolutionAction",
    "FeatureValue",
    # LLM Schema Types
    "EvolutionActionEnum",
    "ExpectedCostEnum",
    "FeatureGenerationResponse",
    "FeatureTypeEnum",
    "LLMFeatureExtractionResponse",
    # WebSocket Types
    "WebSocketMessage",
    "WebSocketMessageType",
    # Exploration Types
    "LLMInteraction",
    "ExplorationState",
]

# This file makes the 'schemas' directory a Python package.

# This file intentionally left blank to mark this directory as a Python package.
