"""
Base agent class for VULCAN autonomous feature engineering.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the VULCAN system.

    Agents are responsible for specific tasks in the feature engineering pipeline,
    such as generating features, reflecting on performance, or evaluating results.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.

        Args:
            name: Name of the agent
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.

        Args:
            context: Context dictionary containing relevant information

        Returns:
            Dictionary containing the results of the agent's execution
        """
        pass

    @abstractmethod
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate that the context contains all required information.

        Args:
            context: Context dictionary to validate

        Returns:
            True if context is valid, False otherwise
        """
        pass

    def get_required_context_keys(self) -> List[str]:
        """
        Get the list of required context keys for this agent.

        Returns:
            List of required context keys
        """
        return []

    def preprocess_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the context before execution.

        Args:
            context: Raw context dictionary

        Returns:
            Preprocessed context dictionary
        """
        return context.copy()

    def postprocess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the results after execution.

        Args:
            results: Raw results dictionary

        Returns:
            Postprocessed results dictionary
        """
        return results

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete agent pipeline: preprocess, validate, execute, postprocess.

        Args:
            context: Context dictionary

        Returns:
            Final results dictionary
        """
        try:
            # Preprocess context
            processed_context = self.preprocess_context(context)

            # Validate context
            if not self.validate_context(processed_context):
                raise ValueError(f"Invalid context for agent {self.name}")

            # Execute main task
            results = self.execute(processed_context)

            # Postprocess results
            final_results = self.postprocess_results(results)

            self.logger.info(f"Agent {self.name} executed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Agent {self.name} execution failed: {str(e)}")
            raise

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
