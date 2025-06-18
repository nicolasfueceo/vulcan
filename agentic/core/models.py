from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional

class ParameterSpec(BaseModel):
    name: str
    lower: float
    upper: float
    type: str = "float"  # "float" or "int"
    init: Optional[float] = None
    description: Optional[str] = None

    @field_validator("name", "type")
    @classmethod
    def not_empty(cls, v):
        if not v or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("type")
    @classmethod
    def valid_type(cls, v):
        if v not in ("float", "int"):
            raise ValueError("type must be 'float' or 'int'")
        return v

    @field_validator("upper")
    @classmethod
    def upper_gt_lower(cls, v, values):
        lower = values.data.get("lower")
        if lower is not None and v <= lower:
            raise ValueError("upper must be greater than lower")
        return v

class Hypothesis(BaseModel):
    id: str
    summary: str
    rationale: str
    depends_on: List[str]

    @field_validator("id", "summary", "rationale")
    @classmethod
    def not_empty(cls, v):
        if not v or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("Field cannot be empty")
        return v

from typing import Literal

class CandidateFeature(BaseModel):
    name: str
    type: str
    spec: str
    feature_scope: Literal["user", "item"]  # Indicates if the feature is for user-user or item-item kNN
    depends_on: List[str] = Field(default_factory=list)
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    rationale: str

    @field_validator("name", "type", "spec", "rationale", "feature_scope")
    @classmethod
    def not_empty(cls, v):
        if not v or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("Field cannot be empty")
        return v

class VettedFeature(CandidateFeature):
    feature_scope: Literal["user", "item"]

class BOResult(BaseModel):
    feature_name: str
    best_params: Dict[str, Any]
    best_value: float
    study_name: str
    storage: str
    n_trials: int
    timestamp: str

    @field_validator("feature_name", "study_name", "storage", "timestamp")
    @classmethod
    def not_empty(cls, v):
        if not v or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("Field cannot be empty")
        return v

class RealizedFeature(CandidateFeature):
    feature_scope: Literal["user", "item"]
    best_params: Dict[str, Any]
    best_value: float
    bo_study_name: str
    bo_storage: str
    realization_timestamp: str

    @field_validator("bo_study_name", "bo_storage", "realization_timestamp")
    @classmethod
    def not_empty(cls, v):
        if not v or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("Field cannot be empty")
        return v
