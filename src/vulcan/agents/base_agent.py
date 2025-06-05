"""Base agent class for VULCAN system."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import structlog

from vulcan.schemas import VulcanConfig
from vulcan.utils import get_vulcan_logger

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all VULCAN agents."""

    def __init__(self, config: VulcanConfig, agent_name: str) -> None:
        """Initialize base agent.

        Args:
            config: VULCAN configuration.
            agent_name: Name of the agent.
        """
        self.config = config
        self.agent_name = agent_name
        self.logger = get_vulcan_logger(__name__).bind(agent_name=agent_name)
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent.

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main functionality.

        Args:
            context: Execution context with relevant data.

        Returns:
            Result dictionary with agent output.
        """
        pass

    @abstractmethod
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate the execution context.

        Args:
            context: Context to validate.

        Returns:
            True if context is valid, False otherwise.
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        self.logger.info("Agent cleanup completed")

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized

    def _set_initialized(self, status: bool = True) -> None:
        """Set initialization status."""
        self._initialized = status
        self.logger.info("Agent initialization status changed", initialized=status)
