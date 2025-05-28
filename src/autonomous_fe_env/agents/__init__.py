"""
Agents module for VULCAN autonomous feature engineering.

This module contains the agent implementations for feature engineering tasks.
"""

from .base_agent import BaseAgent
from .feature_agent import FeatureAgent
from .llm_feature_agent import LLMFeatureAgent
from .reflection_agent import ReflectionAgent


def get_agent(agent_type: str, **kwargs) -> BaseAgent:
    """Factory function to get agent instances."""
    if agent_type == "feature":
        return FeatureAgent(**kwargs)
    elif agent_type == "llm_feature":
        return LLMFeatureAgent(**kwargs)
    elif agent_type == "reflection":
        return ReflectionAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


__all__ = [
    "BaseAgent",
    "FeatureAgent",
    "LLMFeatureAgent",
    "ReflectionAgent",
    "get_agent",
]
