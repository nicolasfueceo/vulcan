"""
Real LLM-based feature agent using OpenAI and LangChain.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ..feature import DataRequirement, FeatureDefinition
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Import LLMService with lazy loading to avoid circular imports
def _get_llm_service():
    from ..llm import LLMService

    return LLMService


class LLMFeatureAgent(BaseAgent):
    """
    Real LLM-based feature agent using OpenAI and LangChain.

    This agent generates features using actual LLM calls with comprehensive
    prompt engineering and response parsing.
    """

    def __init__(
        self, name: str = "LLMFeatureAgent", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLM feature agent.

        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        super().__init__(name, config)

        # Initialize LLM service
        llm_config = {
            "model_name": self.config.get("model_name", "gpt-4o-mini"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 2000),
            "max_retries": self.config.get("max_retries", 3),
            "rate_limit_delay": self.config.get("rate_limit_delay", 0.5),
        }

        LLMServiceClass = _get_llm_service()
        self.llm_service = LLMServiceClass(llm_config)

        logger.info(f"LLM Feature Agent initialized with {self.llm_service.model_name}")

    def set_prompt_logger(self, logger_func):
        """Set the prompt logging function."""
        self.llm_service.set_prompt_logger(logger_func)

    def parse_llm_response(self, response: str) -> Optional[FeatureDefinition]:
        """Parse LLM response into a FeatureDefinition."""
        try:
            # Extract feature name
            feature_name_match = re.search(
                r"Feature Name:\s*([^\n]+)", response, re.IGNORECASE
            )
            feature_name = (
                feature_name_match.group(1).strip() if feature_name_match else None
            )

            # Extract description
            desc_match = re.search(r"Description:\s*([^\n]+)", response, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else None

            # Extract implementation code
            impl_match = re.search(
                r"Implementation:\s*```python\s*(.*?)\s*```",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if not impl_match:
                impl_match = re.search(
                    r"Implementation:\s*(def\s+.*?)(?=Required Columns|Business Logic|$)",
                    response,
                    re.DOTALL | re.IGNORECASE,
                )

            implementation = impl_match.group(1).strip() if impl_match else None

            # Extract required columns
            cols_match = re.search(
                r"Required Columns:\s*\[([^\]]+)\]", response, re.IGNORECASE
            )
            if not cols_match:
                cols_match = re.search(
                    r"Required Columns:\s*([^\n]+)", response, re.IGNORECASE
                )

            required_columns = []
            if cols_match:
                cols_text = cols_match.group(1).strip()
                required_columns = [
                    col.strip().strip("'\"") for col in cols_text.split(",")
                ]

            # Validate required fields
            if not all([feature_name, description, implementation]):
                logger.error("Failed to parse required fields from LLM response")
                return None

            # Clean feature name (ensure snake_case)
            feature_name = re.sub(r"[^a-zA-Z0-9_]", "_", feature_name.lower())
            feature_name = re.sub(r"_+", "_", feature_name).strip("_")

            # Create data requirement
            data_requirement = DataRequirement(
                type="horizontal",
                entity_type="user",
                columns=required_columns or ["user_id"],
                lookup_id_field="user_id",
            )

            return FeatureDefinition(
                name=feature_name,
                description=description,
                code=implementation,
                required_input_columns=required_columns or ["user_id"],
                output_column_name=feature_name,
                data_requirements=data_requirement,
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            return None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the feature generation task using real LLM.

        Args:
            context: Context containing state manager and other information

        Returns:
            Dictionary containing the generated feature or None
        """
        try:
            logger.info("🤖 Starting real LLM-based feature generation")

            # Use LLM service to generate feature
            response = self.llm_service.generate_feature(context, self.name)

            logger.info("🤖 Received LLM response for feature generation")

            # Parse response into feature
            feature = self.parse_llm_response(response)

            if feature:
                logger.info(f"✅ Successfully generated feature: {feature.name}")
            else:
                logger.warning("❌ Failed to parse LLM response into valid feature")

            return {"feature": feature}

        except Exception as e:
            logger.error(f"❌ Error in LLM feature generation: {e}")
            return {"feature": None}

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

    def generate_feature_proposal(
        self, state_manager, context: Dict[str, Any]
    ) -> Optional[FeatureDefinition]:
        """
        Generate a feature proposal for MCTS orchestrator.

        Args:
            state_manager: State manager instance
            context: Context containing MCTS node and other information

        Returns:
            Generated feature definition or None
        """
        try:
            # Prepare context for execute method
            execute_context = {
                "state_manager": state_manager,
                "iteration": context.get("iteration", 1),
                "mcts_node": context.get("mcts_node"),
                "current_features": context.get("current_features", []),
                "existing_child_features": context.get(
                    "existing_child_features", set()
                ),
                "database_schema": context.get("database_schema", {}),
            }

            # Use the existing execute method
            result = self.execute(execute_context)
            return result.get("feature")

        except Exception as e:
            logger.error(f"Error in generate_feature_proposal: {e}")
            return None
