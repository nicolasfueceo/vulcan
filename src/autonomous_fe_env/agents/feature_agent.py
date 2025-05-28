"""
Feature agent for generating feature engineering proposals.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from ..feature import FeatureDefinition
from ..state import StateManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FeatureAgent(BaseAgent):
    """
    Agent responsible for generating feature engineering proposals.

    This agent can work in different modes:
    - Predefined: Uses a list of predefined features
    - LLM-based: Uses an LLM to generate features (requires LLM integration)
    - Hybrid: Combines both approaches
    """

    def __init__(
        self, name: str = "FeatureAgent", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the feature agent.

        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        super().__init__(name, config)

        self.mode = self.config.get(
            "mode", "predefined"
        )  # "predefined", "llm", "hybrid"
        self.predefined_features = []
        self.current_index = 0

        # Load predefined features if available
        if self.mode in ["predefined", "hybrid"]:
            self._load_predefined_features()

    def _load_predefined_features(self) -> None:
        """Load predefined features from configuration."""
        # Define some basic predefined features for the Goodreads dataset
        self.predefined_features = [
            FeatureDefinition(
                name="user_avg_rating",
                description="Average rating given by the user across all their reviews",
                code="""
def calculate_user_avg_rating(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if horizontal_user_data is not None and not horizontal_user_data.empty:
        return horizontal_user_data['rating'].mean()
    return 3.0  # Default neutral rating
""",
                required_input_columns=["user_id"],
                output_column_name="user_avg_rating",
            ),
            FeatureDefinition(
                name="user_rating_std",
                description="Standard deviation of ratings given by the user",
                code="""
def calculate_user_rating_std(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if horizontal_user_data is not None and not horizontal_user_data.empty and len(horizontal_user_data) > 1:
        return horizontal_user_data['rating'].std()
    return 0.0  # No variation if only one rating or no data
""",
                required_input_columns=["user_id"],
                output_column_name="user_rating_std",
            ),
            FeatureDefinition(
                name="book_avg_rating",
                description="Average rating for the book across all reviews",
                code="""
def calculate_book_avg_rating(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if vertical_book_data is not None and not vertical_book_data.empty:
        return vertical_book_data['rating'].mean()
    return 3.0  # Default neutral rating
""",
                required_input_columns=["book_id"],
                output_column_name="book_avg_rating",
            ),
            FeatureDefinition(
                name="user_review_count",
                description="Total number of reviews written by the user",
                code="""
def calculate_user_review_count(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if horizontal_user_data is not None and not horizontal_user_data.empty:
        return len(horizontal_user_data)
    return 1  # At least the current review
""",
                required_input_columns=["user_id"],
                output_column_name="user_review_count",
            ),
            FeatureDefinition(
                name="book_review_count",
                description="Total number of reviews for the book",
                code="""
def calculate_book_review_count(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if vertical_book_data is not None and not vertical_book_data.empty:
        return len(vertical_book_data)
    return 1  # At least the current review
""",
                required_input_columns=["book_id"],
                output_column_name="book_review_count",
            ),
            FeatureDefinition(
                name="user_rating_deviation",
                description="How much the user's rating deviates from the book's average",
                code="""
def calculate_user_rating_deviation(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    user_rating = current_review_data.get('rating', 3.0)
    if vertical_book_data is not None and not vertical_book_data.empty:
        book_avg = vertical_book_data['rating'].mean()
        return user_rating - book_avg
    return 0.0  # No deviation if no book data
""",
                required_input_columns=["rating", "book_id"],
                output_column_name="user_rating_deviation",
            ),
            FeatureDefinition(
                name="review_text_length",
                description="Length of the review text in characters",
                code="""
def calculate_review_text_length(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    review_text = current_review_data.get('review_text', '')
    if review_text and isinstance(review_text, str):
        return len(review_text)
    return 0
""",
                required_input_columns=["review_text"],
                output_column_name="review_text_length",
            ),
            FeatureDefinition(
                name="user_genre_preference",
                description="User's preference score for the book's genre based on historical ratings",
                code="""
def calculate_user_genre_preference(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    # This is a simplified version - in practice would need genre data
    if horizontal_user_data is not None and not horizontal_user_data.empty:
        # Use rating variance as a proxy for genre consistency
        rating_std = horizontal_user_data['rating'].std()
        return 5.0 - rating_std if rating_std > 0 else 5.0
    return 3.0  # Default neutral preference
""",
                required_input_columns=["user_id"],
                output_column_name="user_genre_preference",
            ),
        ]

        self.logger.info(f"Loaded {len(self.predefined_features)} predefined features")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the feature generation task.

        Args:
            context: Context containing state manager and other information

        Returns:
            Dictionary containing the generated feature or None
        """
        feature = self.generate_feature_proposal(context.get("state_manager"), context)

        return {"feature": feature}

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
        self, state_manager: StateManager, context: Optional[Dict[str, Any]] = None
    ) -> Optional[FeatureDefinition]:
        """
        Generate a new feature proposal.

        Args:
            state_manager: Current state manager
            context: Optional additional context

        Returns:
            A FeatureDefinition object or None if generation fails
        """
        if self.mode == "predefined":
            return self._generate_predefined_feature(state_manager, context)
        elif self.mode == "llm":
            return self._generate_llm_feature(state_manager, context)
        elif self.mode == "hybrid":
            # Try LLM first, fall back to predefined
            feature = self._generate_llm_feature(state_manager, context)
            if feature is None:
                feature = self._generate_predefined_feature(state_manager, context)
            return feature
        else:
            self.logger.error(f"Unknown mode: {self.mode}")
            return None

    def _generate_predefined_feature(
        self, state_manager: StateManager, context: Optional[Dict[str, Any]] = None
    ) -> Optional[FeatureDefinition]:
        """Generate a feature from the predefined list."""
        if not self.predefined_features:
            self.logger.warning("No predefined features available")
            return None

        # Get features that have already been tried
        successful_features = state_manager.get_successful_features()
        tried_feature_names = {f.name for f in successful_features}

        # Get features that have been tried from the current MCTS node
        if context and "mcts_node" in context:
            mcts_node = context["mcts_node"]
            tried_feature_names.update(mcts_node.tried_feature_names)

        # Find untried features
        untried_features = [
            f for f in self.predefined_features if f.name not in tried_feature_names
        ]

        if not untried_features:
            self.logger.info("All predefined features have been tried")
            # Optionally return a random feature or None
            return None

        # Select a random untried feature
        selected_feature = random.choice(untried_features)
        self.logger.info(f"Selected predefined feature: {selected_feature.name}")

        return selected_feature

    def _generate_llm_feature(
        self, state_manager: StateManager, context: Optional[Dict[str, Any]] = None
    ) -> Optional[FeatureDefinition]:
        """Generate a feature using LLM (placeholder for now)."""
        # This would integrate with an LLM API to generate features
        # For now, return None to indicate LLM generation is not implemented
        self.logger.info("LLM feature generation not yet implemented")
        return None
