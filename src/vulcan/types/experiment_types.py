"""Experiment type definitions for VULCAN system."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperimentRequest(BaseModel):
    """Request model for starting experiments."""

    experiment_name: Optional[str] = Field(None, description="Experiment name")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Configuration overrides"
    )


class ExperimentResult(BaseModel):
    """Experiment result model."""

    experiment_id: str = Field(..., description="Experiment ID")
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    best_node_id: Optional[str] = Field(None, description="Best node ID")
    best_score: float = Field(..., description="Best score achieved")
    best_feature: str = Field(..., description="Best feature name")
    best_features: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of best features"
    )
    total_iterations: int = Field(..., description="Total iterations run")
    execution_time: float = Field(..., description="Execution time in seconds")
    state_file: Optional[str] = Field(None, description="State file path")


class ExperimentStatus(BaseModel):
    """Experiment status model."""

    is_running: bool = Field(..., description="Whether experiment is running")
    experiment_id: Optional[str] = Field(None, description="Current experiment ID")
    config_summary: Dict[str, Any] = Field(..., description="Configuration summary")
    components_initialized: Dict[str, bool] = Field(
        ..., description="Component initialization status"
    )
    current_experiment: Optional[ExperimentResult] = Field(
        None, description="Current experiment result"
    )
    experiment_history_count: int = Field(
        ..., description="Number of experiments in history"
    )


class HyperparameterSweepRequest(BaseModel):
    """Request model for hyperparameter sweeps."""

    param_grid: Dict[str, List[Any]] = Field(..., description="Parameter grid")
    n_trials: int = Field(default=10, ge=1, le=100, description="Number of trials")
    experiment_name: Optional[str] = Field(None, description="Experiment name")


class SweepResult(BaseModel):
    """Hyperparameter sweep result model."""

    trial_number: int = Field(..., description="Trial number")
    parameters: Dict[str, Any] = Field(..., description="Trial parameters")
    best_score: float = Field(..., description="Best score achieved")
    best_feature: str = Field(..., description="Best feature name")
    execution_time: float = Field(..., description="Execution time in seconds")
