"""
LLM Service for VULCAN using OpenAI and LangChain.

This module provides a comprehensive LLM service for feature engineering
and reflection tasks with proper error handling, rate limiting, and logging.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMService:
    """
    Comprehensive LLM service using OpenAI and LangChain.

    Provides feature engineering and reflection capabilities with
    proper error handling, rate limiting, and logging.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM service.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # LLM configuration
        self.model_name = self.config.get("model_name", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)

        # Rate limiting
        self.rate_limit_delay = self.config.get("rate_limit_delay", 0.5)
        self.last_call_time = 0

        # Logging callback
        self.prompt_logger: Optional[Callable] = None

        # Console logging configuration
        self.console_logging = self.config.get("console_logging", True)
        self.log_prompts = self.config.get("log_prompts", True)
        self.log_responses = self.config.get("log_responses", True)
        self.max_log_length = self.config.get("max_log_length", 1000)

        # Initialize OpenAI client
        self._initialize_llm()

        logger.info(f"LLM Service initialized with model: {self.model_name}")

    def _initialize_llm(self):
        """Initialize the OpenAI LLM client."""
        try:
            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. Using mock mode.")
                self.mock_mode = True
                return

            self.mock_mode = False

            # Initialize ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
            )

            # Create output parser
            self.output_parser = StrOutputParser()

            logger.info("✅ OpenAI LLM client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            logger.warning("Falling back to mock mode")
            self.mock_mode = True

    def set_prompt_logger(self, logger_func: Callable):
        """Set the prompt logging function."""
        self.prompt_logger = logger_func

    def _log_prompt(
        self, agent_name: str, prompt_type: str, prompt: str, response: str = ""
    ):
        """Log prompt if logger is available."""
        if self.prompt_logger:
            self.prompt_logger(
                agent_name=agent_name,
                prompt_type=prompt_type,
                prompt=prompt,
                response=response,
            )

    def _log_to_console(
        self, agent_name: str, prompt_type: str, prompt: str, response: str = ""
    ):
        """Log LLM interactions to console for debugging."""
        if not self.console_logging:
            return

        print("\n" + "=" * 80)
        print(f"🤖 LLM INTERACTION - {agent_name.upper()} ({prompt_type})")
        print("=" * 80)

        if self.log_prompts and prompt:
            print("📝 PROMPT:")
            print("-" * 40)
            # Truncate if too long
            display_prompt = prompt
            if len(prompt) > self.max_log_length:
                display_prompt = prompt[: self.max_log_length] + "\n... [TRUNCATED]"
            print(display_prompt)
            print("-" * 40)

        if self.log_responses and response:
            print("🎯 RESPONSE:")
            print("-" * 40)
            # Truncate if too long
            display_response = response
            if len(response) > self.max_log_length:
                display_response = response[: self.max_log_length] + "\n... [TRUNCATED]"
            print(display_response)
            print("-" * 40)

        print("=" * 80)

    def _apply_rate_limiting(self):
        """Apply rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def _mock_llm_call(self, prompt: str, agent_name: str = "MockAgent") -> str:
        """Mock LLM call for testing and fallback."""
        # Simulate API delay
        time.sleep(1.0)

        # Generate mock responses based on prompt content
        if "feature" in prompt.lower():
            return self._generate_mock_feature_response()
        elif "reflection" in prompt.lower() or "analyze" in prompt.lower():
            return self._generate_mock_reflection_response()
        else:
            return "This is a mock response from the LLM service."

    def _generate_mock_feature_response(self) -> str:
        """Generate mock feature engineering response."""
        import random

        features = [
            {
                "name": "user_rating_momentum",
                "description": "Measures the momentum in user's rating patterns over recent reviews",
                "code": """
def calculate_user_rating_momentum(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if horizontal_user_data is not None and not horizontal_user_data.empty and len(horizontal_user_data) > 3:
        ratings = horizontal_user_data['rating'].values
        # Calculate momentum using moving average
        recent_avg = sum(ratings[-3:]) / 3
        older_avg = sum(ratings[:-3]) / len(ratings[:-3])
        return recent_avg - older_avg
    return 0.0
""",
                "columns": ["user_id", "rating"],
            },
            {
                "name": "book_review_velocity",
                "description": "Rate of new reviews for a book over time",
                "code": """
def calculate_book_review_velocity(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if vertical_book_data is not None and not vertical_book_data.empty:
        if 'date_added' in vertical_book_data.columns:
            dates = pd.to_datetime(vertical_book_data['date_added'])
            recent_reviews = (dates >= dates.max() - pd.Timedelta(days=30)).sum()
            total_reviews = len(dates)
            return recent_reviews / total_reviews if total_reviews > 0 else 0.0
    return 0.0
""",
                "columns": ["book_id", "date_added"],
            },
            {
                "name": "user_genre_consistency",
                "description": "Consistency of user preferences within genres",
                "code": """
def calculate_user_genre_consistency(current_review_data, horizontal_user_data=None, vertical_book_data=None):
    if horizontal_user_data is not None and not horizontal_user_data.empty:
        ratings = horizontal_user_data['rating'].values
        # Measure consistency as inverse of standard deviation
        if len(ratings) > 1:
            std_dev = pd.Series(ratings).std()
            return 1.0 / (1.0 + std_dev)
    return 1.0
""",
                "columns": ["user_id", "rating"],
            },
        ]

        feature = random.choice(features)

        return f"""Feature Name: {feature["name"]}
Description: {feature["description"]}
Implementation:
{feature["code"]}
Required Columns: {feature["columns"]}

This feature leverages temporal patterns and user behavior to improve recommendation accuracy."""

    def _generate_mock_reflection_response(self) -> str:
        """Generate mock reflection response."""
        insights = [
            "User behavioral features consistently outperform content-based features",
            "Temporal patterns in ratings provide strong signals for recommendation quality",
            "Feature combinations show promising results when user and item features are merged",
            "Rating velocity and momentum features capture user engagement effectively",
            "Genre diversity metrics help identify user exploration patterns",
        ]

        import random

        selected_insights = random.sample(insights, 3)

        return f"""Based on the current feature engineering progress, I observe several key patterns:

1. **Performance Trends**: The recent features show consistent improvement over baseline models.

2. **Key Insights**:
   - {selected_insights[0]}
   - {selected_insights[1]}
   - {selected_insights[2]}

3. **Strategic Recommendations**:
   - Focus on temporal user behavior patterns
   - Explore feature interactions between user and item characteristics
   - Consider ensemble approaches combining multiple feature types

4. **Next Steps**: Prioritize features that capture user engagement dynamics and book popularity trends."""

    def call_llm(
        self,
        prompt: str,
        agent_name: str = "LLMAgent",
        prompt_type: str = "general",
        system_message: Optional[str] = None,
    ) -> str:
        """
        Call the LLM with proper error handling and rate limiting.

        Args:
            prompt: The user prompt
            agent_name: Name of the calling agent
            prompt_type: Type of prompt for logging
            system_message: Optional system message

        Returns:
            LLM response string
        """
        # Combine system message and prompt for logging
        full_prompt = prompt
        if system_message:
            full_prompt = f"SYSTEM: {system_message}\n\nUSER: {prompt}"

        # Log the prompt to console
        self._log_to_console(agent_name, prompt_type, full_prompt)

        # Log the prompt via callback
        self._log_prompt(agent_name, prompt_type, prompt)

        # Use mock mode if no API key or initialization failed
        if self.mock_mode:
            logger.info(f"🤖 Using mock LLM call for {agent_name}")
            response = self._mock_llm_call(prompt, agent_name)

            # Log response to console and callback
            self._log_to_console(agent_name, prompt_type, full_prompt, response)
            self._log_prompt(agent_name, prompt_type, prompt, response)
            return response

        # Apply rate limiting
        self._apply_rate_limiting()

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"🤖 Calling OpenAI API (attempt {attempt + 1}/{self.max_retries})"
                )

                # Prepare messages
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))

                # Create prompt template and invoke
                if system_message:
                    template = ChatPromptTemplate.from_messages(
                        [("system", system_message), ("human", "{input}")]
                    )
                    chain = template | self.llm | self.output_parser
                    response = chain.invoke({"input": prompt})
                else:
                    # Direct LLM call without system message
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    # Parse the response
                    response = self.output_parser.invoke(response)

                logger.info("✅ Successfully received response from OpenAI API")

                # Log response to console and callback
                self._log_to_console(agent_name, prompt_type, full_prompt, response)
                self._log_prompt(agent_name, prompt_type, prompt, response)

                return response

            except Exception as e:
                logger.warning(f"❌ LLM call attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"⏳ Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "❌ All LLM call attempts failed. Falling back to mock response."
                    )
                    response = self._mock_llm_call(prompt, agent_name)

                    # Log response to console and callback
                    self._log_to_console(agent_name, prompt_type, full_prompt, response)
                    self._log_prompt(agent_name, prompt_type, prompt, response)
                    return response

    def generate_feature(
        self, context: Dict[str, Any], agent_name: str = "FeatureAgent"
    ) -> str:
        """
        Generate a feature using LLM.

        Args:
            context: Context containing state and iteration information
            agent_name: Name of the calling agent

        Returns:
            LLM response for feature generation
        """
        from .prompts import FeatureEngineeringPrompts

        # Generate prompt using the prompts module
        prompt = FeatureEngineeringPrompts.generate_feature_prompt(context)
        system_message = FeatureEngineeringPrompts.get_system_message()

        return self.call_llm(
            prompt=prompt,
            agent_name=agent_name,
            prompt_type="feature_generation",
            system_message=system_message,
        )

    def generate_reflection(
        self, context: Dict[str, Any], agent_name: str = "ReflectionAgent"
    ) -> str:
        """
        Generate reflection using LLM.

        Args:
            context: Context containing performance and state information
            agent_name: Name of the calling agent

        Returns:
            LLM response for reflection
        """
        from .prompts import ReflectionPrompts

        # Generate prompt using the prompts module
        prompt = ReflectionPrompts.generate_reflection_prompt(context)
        system_message = ReflectionPrompts.get_system_message()

        return self.call_llm(
            prompt=prompt,
            agent_name=agent_name,
            prompt_type="reflection",
            system_message=system_message,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the LLM service."""
        return {
            "mock_mode": self.mock_mode,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "rate_limit_delay": self.rate_limit_delay,
            "api_key_available": bool(os.getenv("OPENAI_API_KEY")),
        }
