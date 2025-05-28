# src/autonomous_fe_env/__init__.py
# This file marks the directory as a Python package.
# Optionally, expose key classes or functions here.

# Example:
# from .orchestrator import Orchestrator
# from .config_manager import ConfigManager

"""
VULCAN: Versatile User-Learning Conversational Agent for Nudging

A two-phase recommender system with:
1. LLM-driven autonomous feature engineering using MCTS
2. Conversational cold-start user assignment
"""

# Core components
from .agents import BaseAgent, get_agent
from .config import ConfigManager
from .data import BaseDAL, get_dal
from .evaluation import FeatureEvaluator
from .feature import DataRequirement, FeatureDefinition
from .mcts import MCTSNode, MCTSOrchestrator, ParallelMCTS
from .reflection import ReflectionEngine
from .sandbox import CodeSandbox
from .state import StateManager


# LLM components available through lazy loading to avoid circular imports
def get_llm_service():
    """Get LLM service with lazy loading to avoid circular imports."""
    from .llm import LLMService

    return LLMService
