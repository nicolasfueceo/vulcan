"""Agent modules for VULCAN system."""

from .base_agent import BaseAgent
from .feature_agent import FeatureAgent

# from .reflection_agent import ReflectionAgent  # TODO: Implement reflection agent

__all__ = [
    "BaseAgent",
    "FeatureAgent",
    # "ReflectionAgent",  # TODO: Add when implemented
]
