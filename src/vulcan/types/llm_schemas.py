"""Pydantic schemas for structured LLM outputs in VULCAN."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class FeatureTypeEnum(str, Enum):
    """Feature type enumeration for LLM responses."""

    CODE_BASED = "code_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"


class ExpectedCostEnum(str, Enum):
    """Expected computational cost enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FeatureGenerationResponse(BaseModel):
    """Structured response for feature generation requests."""

    feature_name: str = Field(description="A descriptive name for the feature")
    feature_type: FeatureTypeEnum = Field(
        description="The type of feature: code_based, llm_based, or hybrid"
    )
    description: str = Field(
        description="Clear explanation of what the feature does and how it works"
    )
    reasoning: str = Field(
        description="Detailed reasoning for why this feature will help improve the recommender system"
    )
    implementation: str = Field(
        description="Code implementation (for code_based/hybrid) or LLM prompt (for llm_based/hybrid)"
    )
    expected_cost: ExpectedCostEnum = Field(
        description="Expected computational cost: low, medium, or high"
    )
    dependencies: Optional[List[str]] = Field(
        default=None, description="List of required data columns or features"
    )
    text_columns: Optional[List[str]] = Field(
        default=None, description="Text columns required for LLM-based features"
    )


class MCTSActionEnum(str, Enum):
    """MCTS action enumeration."""

    ADD = "add"
    MUTATE = "mutate"
    REPLACE = "replace"
    COMBINE = "combine"


class ReflectionResponse(BaseModel):
    """Structured response for performance reflection requests."""

    performance_analysis: str = Field(
        description="Analysis of which features are performing well and why"
    )
    performance_trends: str = Field(
        description="Patterns observed in the performance trends"
    )
    feature_interactions: str = Field(
        description="Analysis of feature interactions or redundancies"
    )
    missing_features: str = Field(
        description="Types of features that might be missing from the current set"
    )
    recommended_action: MCTSActionEnum = Field(
        description="Best next action to improve performance: add, mutate, replace, or combine"
    )
    action_reasoning: str = Field(
        description="Specific reasoning for the recommended action"
    )
    specific_suggestions: List[str] = Field(
        description="Specific suggestions for implementing the recommended action"
    )


class LLMFeatureExtractionResponse(BaseModel):
    """Structured response for LLM-based feature extraction."""

    feature_value: float = Field(description="Numerical value extracted from the text")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Brief explanation of how the value was determined"
    )
