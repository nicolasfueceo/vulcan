"""Dataclasses for progressive evolution components."""

from dataclasses import dataclass
from typing import Optional

from .feature_types import FeatureDefinition, FeatureEvaluation


@dataclass
class FeatureCandidate:
    """A feature candidate in the population."""

    feature: FeatureDefinition
    score: float
    generation: int
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    status: str = "pending"  # pending, evaluating, evaluated, failed
    error_message: Optional[str] = None
    repair_attempts: int = 0
    evaluation_result: Optional[FeatureEvaluation] = None


@dataclass
class GenerationStats:
    """Statistics for a generation."""

    generation: int
    total_features: int
    successful_features: int
    failed_features: int
    repaired_features: int
    avg_score: float
    best_score: float
    action_taken: str
    population_size: int
