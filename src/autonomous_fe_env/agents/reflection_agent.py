"""
Reflection agent for analyzing feature engineering decisions and performance.
"""

import logging
from typing import Any, Dict, List, Optional

from ..reflection import ReflectionEngine
from ..state import StateManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """
    Agent responsible for generating reflections on feature engineering decisions.

    This agent analyzes the performance of features and provides insights
    to guide future feature engineering decisions.
    """

    def __init__(
        self, name: str = "ReflectionAgent", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the reflection agent.

        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        super().__init__(name, config)

        self.reflection_engine = None
        self.reflection_types = self.config.get(
            "reflection_types", ["feature_proposal", "feature_evaluation", "strategy"]
        )

    def setup_reflection_engine(self, reflection_engine: ReflectionEngine) -> None:
        """
        Set up the reflection engine for this agent.

        Args:
            reflection_engine: The reflection engine to use
        """
        self.reflection_engine = reflection_engine

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reflection generation task.

        Args:
            context: Context containing state manager and other information

        Returns:
            Dictionary containing the generated reflections
        """
        if not self.reflection_engine:
            self.logger.warning("No reflection engine available")
            return {"reflections": []}

        reflections = []
        reflection_type = context.get("reflection_type", "strategy")

        if reflection_type == "feature_proposal":
            reflection = self._generate_feature_proposal_reflection(context)
        elif reflection_type == "feature_evaluation":
            reflection = self._generate_feature_evaluation_reflection(context)
        elif reflection_type == "strategy":
            reflection = self._generate_strategic_reflection(context)
        else:
            self.logger.warning(f"Unknown reflection type: {reflection_type}")
            reflection = None

        if reflection:
            reflections.append({"type": reflection_type, "content": reflection})

        return {"reflections": reflections}

    def validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate that the context contains required information.

        Args:
            context: Context dictionary to validate

        Returns:
            True if context is valid, False otherwise
        """
        required_keys = ["state_manager"]
        return all(key in context for key in required_keys)

    def get_required_context_keys(self) -> List[str]:
        """Get the list of required context keys."""
        return ["state_manager"]

    async def _generate_feature_proposal_reflection(
        self, context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a reflection on what features to propose next."""
        if not self.reflection_engine:
            return None

        state_manager = context["state_manager"]
        feature_history = self._format_feature_history(state_manager)
        database_schema = context.get("database_schema", {})

        try:
            reflection = await self.reflection_engine.generate_feature_reflection(
                state_manager, feature_history, database_schema
            )
            return reflection
        except Exception as e:
            self.logger.error(f"Error generating feature proposal reflection: {e}")
            return None

    async def _generate_feature_evaluation_reflection(
        self, context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a reflection on a feature's evaluation results."""
        if not self.reflection_engine:
            return None

        feature = context.get("feature")
        evaluation_results = context.get("evaluation_results", {})
        state_manager = context["state_manager"]
        feature_history = self._format_feature_history(state_manager)

        if not feature:
            self.logger.warning("No feature provided for evaluation reflection")
            return None

        try:
            reflection = await self.reflection_engine.evaluate_feature_reflection(
                feature, evaluation_results, feature_history
            )
            return reflection
        except Exception as e:
            self.logger.error(f"Error generating feature evaluation reflection: {e}")
            return None

    async def _generate_strategic_reflection(
        self, context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a strategic reflection on the overall approach."""
        if not self.reflection_engine:
            return None

        state_manager = context["state_manager"]
        feature_history = self._format_feature_history(state_manager)
        performance_history = self._format_performance_history(state_manager)

        try:
            reflection = await self.reflection_engine.strategic_reflection(
                state_manager, feature_history, performance_history
            )
            return reflection
        except Exception as e:
            self.logger.error(f"Error generating strategic reflection: {e}")
            return None

    def _format_feature_history(
        self, state_manager: StateManager
    ) -> List[Dict[str, Any]]:
        """Format feature history for reflection engine."""
        feature_history = []

        for feature_state in state_manager.get_feature_history():
            if feature_state.feature:
                feature_history.append(
                    {
                        "name": feature_state.feature.name,
                        "description": feature_state.feature.description,
                        "score": feature_state.score,
                        "timestamp": feature_state.timestamp,
                    }
                )

        return feature_history

    def _format_performance_history(
        self, state_manager: StateManager
    ) -> List[Dict[str, Any]]:
        """Format performance history for reflection engine."""
        performance_history = []

        for i, feature_state in enumerate(state_manager.get_feature_history()):
            performance_history.append(
                {
                    "iteration": i + 1,
                    "score": feature_state.score,
                    "features": [feature_state.feature.name]
                    if feature_state.feature
                    else [],
                    "timestamp": feature_state.timestamp,
                }
            )

        return performance_history

    def generate_reflection(
        self,
        reflection_type: str,
        state_manager: StateManager,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate a reflection of the specified type.

        Args:
            reflection_type: Type of reflection to generate
            state_manager: Current state manager
            context: Optional additional context

        Returns:
            Generated reflection text or None if generation fails
        """
        full_context = {
            "state_manager": state_manager,
            "reflection_type": reflection_type,
        }

        if context:
            full_context.update(context)

        result = self.run(full_context)
        reflections = result.get("reflections", [])

        if reflections:
            return reflections[0].get("content")

        return None
