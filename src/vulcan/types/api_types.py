"""API type definitions for VULCAN system."""

from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""

    status: str = Field(..., description="Response status")
    data: Optional[T] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if any")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")
    version: str = Field(..., description="API version")


class StatusResponse(BaseModel):
    """System status response model."""

    status: str = Field(..., description="System status")
    components: Dict[str, bool] = Field(..., description="Component status")
    config_loaded: bool = Field(..., description="Whether config is loaded")
    experiments_count: int = Field(..., description="Number of experiments run")
