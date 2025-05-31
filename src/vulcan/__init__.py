"""VULCAN: Autonomous Feature Engineering for Recommender Systems."""

from vulcan.core import ConfigManager, VulcanOrchestrator
from vulcan.utils import ExperimentTracker, setup_logging

__version__ = "2.0.0"
__author__ = "VULCAN Research Team"
__email__ = "vulcan@imperial.ac.uk"

__all__ = [
    "ConfigManager",
    "VulcanOrchestrator",
    "ExperimentTracker",
    "setup_logging",
]
