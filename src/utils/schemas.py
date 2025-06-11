# src/utils/schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    id: str = Field(
        ..., description="A unique identifier for the hypothesis, e.g., 'H1'."
    )
    text: str = Field(..., description="The full text of the hypothesis.")
    priority: Optional[int] = Field(
        None, description="A priority score, to be filled in later."
    )
    notes: Optional[str] = Field(
        "", description="Initial notes or context about the hypothesis."
    )


class PrioritizedHypothesis(BaseModel):
    id: str = Field(..., description="The unique identifier for the hypothesis.")
    priority: int = Field(
        ..., ge=1, le=5, description="The priority score from 1 to 5."
    )
    feasibility: int = Field(
        ..., ge=1, le=5, description="The feasibility score from 1 to 5."
    )
    notes: str = Field(..., description="A brief justification for the scores.")


class CandidateFeature(BaseModel):
    name: str = Field(
        ..., description="A descriptive, camel-case name for the feature."
    )
    type: str = Field(
        ..., description="The type of feature: 'code', 'llm', or 'composition'."
    )
    spec: str = Field(
        ...,
        description="The feature specification (e.g., DSL for code, prompt for llm).",
    )
    depends_on: Optional[List[str]] = Field(
        [], description="A list of other features this feature depends on."
    )
    rationale: str = Field(
        ..., description="A brief explanation of why this feature is useful."
    )
    effort: int = Field(
        ...,
        ge=1,
        le=5,
        description="The estimated effort to implement this feature (1-5).",
    )
    impact: int = Field(
        ...,
        ge=1,
        le=5,
        description="The estimated impact of this feature on the model (1-5).",
    )
