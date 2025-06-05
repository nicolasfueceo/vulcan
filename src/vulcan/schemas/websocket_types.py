"""WebSocket type definitions for VULCAN system."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""

    # General Notifications
    NOTIFICATION = "notification"
    STATUS_UPDATE = "status_update"
    LOG_MESSAGE = "log_message"
    ERROR_MESSAGE = "error_message"

    # Experiment Lifecycle
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    EXPERIMENT_STOPPED = "experiment_stopped"
    EXPERIMENT_FAILED = "experiment_failed"

    # For backward compatibility with old names
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"

    # Evolution / MCTS Updates
    EXPLORATION_UPDATE = "exploration_update"
    FEATURE_EVALUATION = "feature_evaluation"

    # Sweep Lifecycle
    SWEEP_STARTED = "sweep_started"
    SWEEP_COMPLETED = "sweep_completed"
    SWEEP_FAILED = "sweep_failed"
    EXPERIMENT_UPDATE = "experiment_update"


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: WebSocketMessageType = Field(..., description="Message type")
    timestamp: float = Field(..., description="Message timestamp")
    experiment_id: Optional[str] = Field(None, description="Experiment ID")
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")
    error: Optional[str] = Field(None, description="Error message")
    results: Optional[Dict[str, Any]] = Field(None, description="Results data")
