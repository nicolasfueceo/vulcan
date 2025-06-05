"""WebSocket type definitions for VULCAN system."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""

    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    EXPERIMENT_STOPPED = "experiment_stopped"
    EXPLORATION_UPDATE = "exploration_update"
    SWEEP_STARTED = "sweep_started"
    SWEEP_COMPLETED = "sweep_completed"
    SWEEP_FAILED = "sweep_failed"
    FEATURE_EVALUATION = "feature_evaluation"
    STATUS_UPDATE = "status_update"
    EXPERIMENT_UPDATE = "experiment_update"
    LOG_MESSAGE = "log_message"
    ERROR_MESSAGE = "error_message"


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: WebSocketMessageType = Field(..., description="Message type")
    timestamp: float = Field(..., description="Message timestamp")
    experiment_id: Optional[str] = Field(None, description="Experiment ID")
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")
    error: Optional[str] = Field(None, description="Error message")
    results: Optional[Dict[str, Any]] = Field(None, description="Results data")
