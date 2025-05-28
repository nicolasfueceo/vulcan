"""Reflection engine for reasoning about feature engineering decisions using LLMs."""

import os
import time
from typing import Any, Dict, List, Optional

from ..feature import FeatureDefinition
from ..state import StateManager
from .reflection_memory import ReflectionMemory


# Import LLMService with lazy loading to avoid circular imports
def _get_llm_service():
    from ..llm import LLMService

    return LLMService


class ReflectionEngine:
    """Engine for generating reflections about feature engineering using LLMs."""

    def __init__(
        self, config: Dict[str, Any], memory: Optional[ReflectionMemory] = None
    ):
        """
        Initialize the reflection engine.

        Args:
            config: Configuration dictionary
            memory: Optional existing reflection memory to use
        """
        self.config = config
        self.reflection_config = config.get("reflection", {})
        self.memory = memory or ReflectionMemory(
            memory_dir=self.reflection_config.get("memory_dir", "memory"),
            max_entries=self.reflection_config.get("max_entries", 100),
        )

        # Initialize LLM service
        llm_config = config.get("llm", {})
        LLMService = _get_llm_service()
        self.llm_service = LLMService(llm_config)

        # Load templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for different reflection types."""
        templates = {}
        templates_dir = self.reflection_config.get(
            "templates_dir", "prompts/reflection"
        )

        # Default templates if files not found
        templates["feature_proposal"] = """
        You are an expert in recommender systems and feature engineering.
        
        # Current State:
        {state_summary}
        
        # Previous Features:
        {feature_history}
        
        # Database Schema:
        {database_schema}
        
        # Previous Reflections:
        {reflection_history}
        
        Please reflect on what kinds of features would be most valuable to try next and why.
        Consider both the performance of previous features and potential new directions.
        Your reflection should include specific ideas for new features and your reasoning.
        """

        templates["feature_evaluation"] = """
        You are an expert in recommender systems and feature engineering.
        
        # Feature Being Evaluated:
        {feature_definition}
        
        # Evaluation Results:
        {evaluation_results}
        
        # Previous Features:
        {feature_history}
        
        # Previous Reflections:
        {reflection_history}
        
        Please reflect on the performance of this feature and provide insights.
        Why did this feature perform well or poorly? What does this tell us about the problem?
        What directions should we explore or avoid based on this result?
        """

        templates["strategy"] = """
        You are an expert in recommender systems and feature engineering.
        
        # Current State:
        {state_summary}
        
        # Features Explored So Far:
        {feature_history}
        
        # Performance History:
        {performance_history}
        
        # Previous Reflections:
        {reflection_history}
        
        Please reflect on the overall feature engineering strategy.
        What patterns have emerged from our exploration? What areas are promising?
        What areas should we avoid? What high-level approach should we take next?
        """

        # Try to load templates from files
        if os.path.exists(templates_dir):
            for template_name in ["feature_proposal", "feature_evaluation", "strategy"]:
                template_path = os.path.join(templates_dir, f"{template_name}.txt")
                if os.path.exists(template_path):
                    try:
                        with open(template_path, "r") as f:
                            templates[template_name] = f.read()
                    except Exception as e:
                        print(f"Error loading template {template_path}: {e}")

        return templates

    async def generate_feature_reflection(
        self,
        state_manager: StateManager,
        feature_history: List[Dict[str, Any]],
        database_schema: Dict[str, List[str]],
    ) -> str:
        """
        Generate a reflection on what features to try next.

        Args:
            state_manager: Current state manager
            feature_history: History of previously tried features
            database_schema: Database schema for reference

        Returns:
            Reflection text
        """
        # Build the prompt from template
        prompt = self.templates["feature_proposal"].format(
            state_summary=state_manager.get_summary(),
            feature_history=self._format_feature_history(feature_history),
            database_schema=self._format_database_schema(database_schema),
            reflection_history=self.memory.get_formatted_history(
                n=5, entry_type="feature_proposal"
            ),
        )

        # Call LLM to generate reflection
        reflection = await self._call_llm(prompt)

        # Store reflection in memory
        self.memory.add_entry(
            entry_type="feature_proposal",
            content=reflection,
            metadata={
                "num_features_tried": len(feature_history),
                "current_score": state_manager.get_current_score(),
            },
        )

        return reflection

    def generate_reflection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a synchronous reflection for the dashboard.

        Args:
            context: Context containing reflection information

        Returns:
            Dictionary with reflection content and insights
        """
        try:
            # Use LLM service to generate reflection
            response = self.llm_service.generate_reflection(context, "ReflectionEngine")

            # Parse response into structured format
            content = response.strip()

            # Extract insights (simple parsing)
            insights = []
            if "insights:" in content.lower():
                insights_section = content.lower().split("insights:")[-1]
                for line in insights_section.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•"):
                        insights.append(line[1:].strip())

            # Default insights if none found
            if not insights:
                insights = [
                    "Feature performance shows consistent patterns",
                    "User behavioral features demonstrate strong predictive power",
                    "Temporal patterns in data provide valuable signals",
                ]

            return {
                "content": content,
                "insights": insights[:3],  # Limit to 3 insights
            }

        except Exception:
            # Fallback response
            return {
                "content": f"Reflection analysis for iteration {context.get('iteration', 'N/A')}",
                "insights": [
                    "Feature engineering progress is on track",
                    "Continue exploring user behavioral patterns",
                    "Monitor performance trends for optimization opportunities",
                ],
            }

    async def evaluate_feature_reflection(
        self,
        feature: FeatureDefinition,
        evaluation_results: Dict[str, Any],
        feature_history: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a reflection on a feature's performance.

        Args:
            feature: The feature being evaluated
            evaluation_results: Results of evaluating the feature
            feature_history: History of previously tried features

        Returns:
            Reflection text
        """
        # Build the prompt from template
        prompt = self.templates["feature_evaluation"].format(
            feature_definition=self._format_feature_definition(feature),
            evaluation_results=self._format_evaluation_results(evaluation_results),
            feature_history=self._format_feature_history(feature_history),
            reflection_history=self.memory.get_formatted_history(
                n=5, entry_type="feature_evaluation"
            ),
        )

        # Call LLM to generate reflection
        reflection = await self._call_llm(prompt)

        # Store reflection in memory
        self.memory.add_entry(
            entry_type="feature_evaluation",
            content=reflection,
            metadata={
                "feature_name": feature.name,
                "score": evaluation_results.get("score", 0.0),
            },
        )

        return reflection

    async def strategic_reflection(
        self,
        state_manager: StateManager,
        feature_history: List[Dict[str, Any]],
        performance_history: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a strategic reflection on overall feature engineering approach.

        Args:
            state_manager: Current state manager
            feature_history: History of previously tried features
            performance_history: History of performance metrics

        Returns:
            Reflection text
        """
        # Build the prompt from template
        prompt = self.templates["strategy"].format(
            state_summary=state_manager.get_summary(),
            feature_history=self._format_feature_history(feature_history),
            performance_history=self._format_performance_history(performance_history),
            reflection_history=self.memory.get_formatted_history(
                n=5, entry_type="strategy"
            ),
        )

        # Call LLM to generate reflection
        reflection = await self._call_llm(prompt)

        # Store reflection in memory
        self.memory.add_entry(
            entry_type="strategy",
            content=reflection,
            metadata={
                "num_features_tried": len(feature_history),
                "current_score": state_manager.get_current_score(),
                "best_score": max([p.get("score", 0.0) for p in performance_history])
                if performance_history
                else 0.0,
            },
        )

        return reflection

    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM to generate a reflection.

        This is a placeholder that should be replaced with actual LLM API calls.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Generated reflection text
        """
        # This should be replaced with actual LLM API calls
        # For now, just return a placeholder
        print(
            "LLM API call placeholder - this should be replaced with actual implementation"
        )
        print(f"Prompt length: {len(prompt)} characters")
        time.sleep(1)  # Simulate API call delay

        # Placeholder response
        return "This is a placeholder reflection that would normally be generated by an LLM. In a real implementation, this method would call an LLM API with the provided prompt and return the generated text."

    def _format_feature_history(self, feature_history: List[Dict[str, Any]]) -> str:
        """Format feature history for inclusion in prompts."""
        if not feature_history:
            return "No features have been tried yet."

        formatted = []
        for i, feature in enumerate(feature_history):
            name = feature.get("name", f"Feature {i + 1}")
            score = feature.get("score", "N/A")
            description = feature.get("description", "No description")

            formatted.append(f"- {name} (Score: {score}): {description}")

        return "\n".join(formatted)

    def _format_feature_definition(self, feature: FeatureDefinition) -> str:
        """Format a feature definition for inclusion in prompts."""
        return f"""
        Name: {feature.name}
        Description: {feature.description}
        Required Input Columns: {", ".join(feature.required_input_columns)}
        Output Column: {feature.output_column_name}
        
        Code:
        ```python
        {feature.code}
        ```
        """

    def _format_evaluation_results(self, results: Dict[str, Any]) -> str:
        """Format evaluation results for inclusion in prompts."""
        formatted = ["Evaluation Results:"]

        for key, value in results.items():
            formatted.append(f"- {key}: {value}")

        return "\n".join(formatted)

    def _format_database_schema(self, schema: Dict[str, List[str]]) -> str:
        """Format database schema for inclusion in prompts."""
        if not schema:
            return "No database schema available."

        formatted = ["Database Schema:"]

        for table, columns in schema.items():
            formatted.append(f"\n## {table} Table")
            formatted.append("Columns:")
            for col in columns:
                formatted.append(f"- {col}")

        return "\n".join(formatted)

    def _format_performance_history(
        self, performance_history: List[Dict[str, Any]]
    ) -> str:
        """Format performance history for inclusion in prompts."""
        if not performance_history:
            return "No performance history available."

        formatted = ["Performance History:"]

        for i, perf in enumerate(performance_history):
            iteration = perf.get("iteration", i + 1)
            score = perf.get("score", "N/A")
            features = perf.get("features", [])
            feature_names = [
                f.get("name", f"Feature {j + 1}") for j, f in enumerate(features)
            ]

            formatted.append(
                f"- Iteration {iteration}: Score {score} with features [{', '.join(feature_names)}]"
            )

        return "\n".join(formatted)
