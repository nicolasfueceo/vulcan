"""Intelligent feature generation agent for VULCAN system."""

import os
import time
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vulcan.types import (
    ActionContext,
    DataContext,
    FeatureDefinition,
    FeatureGenerationResponse,
    FeatureType,
    FeatureTypeEnum,
    MCTSAction,
    ReflectionResponse,
    VulcanConfig,
)

from .base_agent import BaseAgent

logger = structlog.get_logger(__name__)

# Enhanced prompts with JSON output instructions
FEATURE_GENERATION_SYSTEM_PROMPT = """You are an expert feature engineer working on a recommender system. Your task is to generate new features that could improve user clustering and recommendation quality.

You must respond with a valid JSON object that matches the exact schema provided. Do not include any additional text, explanations, or markdown formatting outside the JSON response.

CRITICAL CODING REQUIREMENTS:
1. For code_based features: Your code MUST set a variable called 'result'
2. Use 'df' to access the DataFrame (never use 'data')
3. Available libraries: pandas (as pd), numpy (as np)
4. The result should be a pandas Series indexed by user_id or a single aggregated value
5. Handle missing values with .fillna() or .dropna()

DATABASE SCHEMA:
- user_id: Integer, unique identifier for users
- book_id: Integer, unique identifier for books  
- rating: Float, user rating for book (1.0-5.0)
- title: String, book title
- authors: String, comma-separated author names
- average_rating: Float, book's average rating across all users
- ratings_count: Integer, total number of ratings for the book

VALID CODE PATTERNS:
```python
# Pattern 1: User-level aggregation
result = df.groupby('user_id')['rating'].mean()

# Pattern 2: User-book interaction features  
result = df.groupby('user_id')['rating'].count()

# Pattern 3: Complex feature with multiple operations
user_stats = df.groupby('user_id').agg({{
    'rating': ['mean', 'std', 'count'],
    'average_rating': 'mean'
}}).round(4)
result = user_stats.fillna(0)

# Pattern 4: Text-based features (simple)
result = df.groupby('user_id')['authors'].apply(lambda x: len(set(x)))
```

For llm_based features: Provide a clear prompt that can extract numerical values from text.
For hybrid features: This will be handled by the executor with proper LLM integration."""

FEATURE_GENERATION_USER_PROMPT = """Current Context:
- Action: {action}
- Current features: {current_features}
- Performance history: {performance_summary}
- Data schema: {data_schema}
- Available text columns: {text_columns}

Data Statistics:
- Number of users: {n_users}
- Number of items: {n_items}
- Data sparsity: {sparsity:.3f}

{action_specific_prompt}

Requirements:
1. Generate a feature that addresses gaps in current feature set
2. Consider computational cost vs. potential benefit
3. Provide clear implementation following the required patterns
4. Explain reasoning behind the feature choice
5. ENSURE your code sets a 'result' variable and uses 'df' not 'data'

{format_instructions}"""

ACTION_SPECIFIC_PROMPTS = {
    MCTSAction.ADD: """
You are ADDING a new feature to the existing set. Focus on:
- Complementing existing features
- Filling gaps in user representation  
- Adding new dimensions of user behavior
- Using the provided database schema effectively

MANDATORY: Your implementation must follow this pattern:
```python
# Your feature logic here using df
result = df.groupby('user_id')[column].operation()
# or
result = computed_feature_series
```
""",
    MCTSAction.MUTATE: """
You are MUTATING an existing feature: {{target_feature}}
Current implementation: {{target_implementation}}
Focus on:
- Improving the existing feature logic
- Adjusting parameters or thresholds
- Enhancing the feature extraction method
- Fixing any issues in the current implementation

MANDATORY: Your implementation must set 'result' variable.
""",
    MCTSAction.REPLACE: """
You are REPLACING the worst performing feature: {{target_feature}}
Current implementation: {{target_implementation}}
Focus on:
- Creating a better alternative
- Addressing the limitations of the replaced feature
- Maintaining or improving performance
- Using different columns or operations

MANDATORY: Your implementation must set 'result' variable.
""",
    MCTSAction.COMBINE: """
You are COMBINING two features: {{feature_1}} and {{feature_2}}
Current implementations:
Feature 1: {{implementation_1}}
Feature 2: {{implementation_2}}
Focus on:
- Creating a composite feature that captures both aspects
- Reducing feature redundancy
- Improving overall representation

MANDATORY: Your implementation must set 'result' variable.
""",
}

REFLECTION_SYSTEM_PROMPT = """You are an expert data scientist analyzing feature performance in a recommender system. 

You must respond with a valid JSON object that matches the exact schema provided. Do not include any additional text, explanations, or markdown formatting outside the JSON response.

Provide detailed analysis and actionable recommendations based on the performance data."""

REFLECTION_USER_PROMPT = """Current Feature Set:
{feature_descriptions}

Performance Metrics:
{performance_metrics}

Performance History:
{performance_history}

Data Context:
- Users: {n_users}, Items: {n_items}, Sparsity: {sparsity:.3f}

Analyze the results and provide insights:
1. Which features are performing well and why?
2. What patterns do you see in the performance trends?
3. Are there any feature interactions or redundancies?
4. What should be the next action to improve performance?
5. What types of features might be missing?

{format_instructions}"""


class FeatureAgent(BaseAgent):
    """Intelligent agent for generating and reasoning about features using structured JSON output."""

    def __init__(self, config: VulcanConfig) -> None:
        """Initialize feature agent.

        Args:
            config: VULCAN configuration.
        """
        super().__init__(config, "FeatureAgent")
        self.llm_client = None
        self.feature_chain = None
        self.reflection_chain = None
        self.use_llm = config.llm.provider != "local"

    async def initialize(self) -> bool:
        """Initialize the feature agent with LangChain structured output."""
        try:
            if self.use_llm:
                await self._initialize_langchain()

            self.logger.info(
                "Feature agent initialized",
                use_llm=self.use_llm,
                provider=self.config.llm.provider,
            )
            self._set_initialized(True)
            return True

        except Exception as e:
            self.logger.error("Failed to initialize feature agent", error=str(e))
            return False

    async def _initialize_langchain(self) -> None:
        """Initialize LangChain with structured output."""
        if self.config.llm.provider == "openai":
            try:
                api_key = os.getenv(self.config.llm.api_key_env)
                if not api_key:
                    self.logger.warning("OpenAI API key not found")
                    self.use_llm = False
                    return

                # Initialize LangChain OpenAI client
                self.llm_client = ChatOpenAI(
                    model=self.config.llm.model_name,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    api_key=api_key,
                )

                # Create structured output chains
                self._setup_feature_generation_chain()
                self._setup_reflection_chain()

                self.logger.info("LangChain initialized with structured output")

            except ImportError:
                self.logger.warning("LangChain OpenAI package not installed")
                self.use_llm = False

    def _setup_feature_generation_chain(self) -> None:
        """Setup the feature generation chain with structured output."""
        # Create parser for FeatureGenerationResponse
        parser = JsonOutputParser(pydantic_object=FeatureGenerationResponse)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FEATURE_GENERATION_SYSTEM_PROMPT),
                ("human", FEATURE_GENERATION_USER_PROMPT),
            ]
        )

        # Create a simple callable chain
        async def feature_chain_func(inputs):
            formatted_prompt = await prompt.ainvoke(inputs)
            llm_response = await self.llm_client.ainvoke(formatted_prompt)
            return await parser.ainvoke(llm_response)

        self.feature_chain = feature_chain_func

    def _setup_reflection_chain(self) -> None:
        """Setup the reflection chain with structured output."""
        # Create parser for ReflectionResponse
        parser = JsonOutputParser(pydantic_object=ReflectionResponse)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REFLECTION_SYSTEM_PROMPT),
                ("human", REFLECTION_USER_PROMPT),
            ]
        )

        # Create a simple callable chain
        async def reflection_chain_func(inputs):
            formatted_prompt = await prompt.ainvoke(inputs)
            llm_response = await self.llm_client.ainvoke(formatted_prompt)
            return await parser.ainvoke(llm_response)

        self.reflection_chain = reflection_chain_func

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new feature based on the current context.

        Args:
            context: Execution context containing data and action information.

        Returns:
            Dictionary containing the generated feature and reasoning.
        """
        if not self.validate_context(context):
            return {"error": "Invalid context provided"}

        try:
            data_context = context["data_context"]
            action_context = context["action_context"]

            # Decide on the action to take
            action = await self._decide_action(action_context, data_context)

            # Generate feature based on action
            feature = await self._generate_feature(action, action_context, data_context)

            if feature:
                self.logger.info(
                    "Generated new feature",
                    feature_name=feature.name,
                    action=action,
                    feature_type=feature.feature_type,
                )

                return {
                    "success": True,
                    "feature": feature,
                    "action": action,
                    "reasoning": f"Generated {feature.feature_type} feature via {action} action",
                }
            else:
                return {"error": "Failed to generate feature"}

        except Exception as e:
            self.logger.error("Feature generation failed", error=str(e))
            return {"error": f"Feature generation failed: {str(e)}"}

    async def _decide_action(
        self,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> MCTSAction:
        """Decide which action to take based on current context.

        Args:
            action_context: Action context.
            data_context: Data context.

        Returns:
            MCTS action to take.
        """
        # Simple heuristic for action selection
        feature_count = len(action_context.current_features.features)

        if feature_count == 0:
            return MCTSAction.ADD
        elif feature_count < 3:
            return MCTSAction.ADD
        else:
            # For more features, use performance history to decide
            if action_context.performance_history:
                recent_performance = action_context.performance_history[-3:]
                avg_performance = sum(
                    p.overall_score for p in recent_performance
                ) / len(recent_performance)

                if avg_performance < 0.3:
                    return MCTSAction.REPLACE
                elif feature_count >= 5:
                    return MCTSAction.COMBINE
                else:
                    return MCTSAction.MUTATE
            else:
                return MCTSAction.ADD

    async def _generate_feature(
        self,
        action: MCTSAction,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Optional[FeatureDefinition]:
        """Generate a feature based on the specified action.

        Args:
            action: MCTS action to take.
            action_context: Action context.
            data_context: Data context.

        Returns:
            Generated feature definition.
        """
        if self.use_llm and self.feature_chain:
            return await self._generate_llm_feature(
                action, action_context, data_context
            )
        else:
            return self._generate_heuristic_feature(
                action, action_context, data_context
            )

    async def _generate_llm_feature(
        self,
        action: MCTSAction,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Optional[FeatureDefinition]:
        """Generate feature using LangChain structured output.

        Args:
            action: MCTS action to take.
            action_context: Action context.
            data_context: Data context.

        Returns:
            Generated feature definition.
        """
        try:
            # Prepare context for LLM
            current_features = [
                f.name for f in action_context.current_features.features
            ]
            performance_summary = self._summarize_performance(
                action_context.performance_history
            )

            # Get action-specific prompt
            action_prompt = ACTION_SPECIFIC_PROMPTS[action]

            # Add action-specific context
            if action == MCTSAction.MUTATE:
                target_feature = action_context.get_worst_performing_feature()
                if target_feature:
                    target_impl = self._get_feature_implementation(
                        target_feature, action_context
                    )
                    action_prompt = action_prompt.replace(
                        "{{target_feature}}", target_feature
                    ).replace("{{target_implementation}}", target_impl)
            elif action == MCTSAction.REPLACE:
                target_feature = action_context.get_worst_performing_feature()
                if target_feature:
                    target_impl = self._get_feature_implementation(
                        target_feature, action_context
                    )
                    action_prompt = action_prompt.replace(
                        "{{target_feature}}", target_feature
                    ).replace("{{target_implementation}}", target_impl)
            elif action == MCTSAction.COMBINE:
                features = action_context.current_features.features
                if len(features) >= 2:
                    action_prompt = (
                        action_prompt.replace("{{feature_1}}", features[0].name)
                        .replace("{{feature_2}}", features[1].name)
                        .replace(
                            "{{implementation_1}}",
                            features[0].code
                            or features[0].llm_prompt
                            or "No implementation",
                        )
                        .replace(
                            "{{implementation_2}}",
                            features[1].code
                            or features[1].llm_prompt
                            or "No implementation",
                        )
                    )

            # Get format instructions from parser
            parser = JsonOutputParser(pydantic_object=FeatureGenerationResponse)
            format_instructions = parser.get_format_instructions()

            # Log the prompt for auditing
            self.logger.info(
                "Sending prompt to LLM",
                action=action.value,
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

            # Invoke the structured chain
            response = await self.feature_chain(
                {
                    "action": action.value,
                    "current_features": current_features,
                    "performance_summary": performance_summary,
                    "data_schema": list(data_context.data_schema.keys()),
                    "text_columns": data_context.text_columns,
                    "n_users": data_context.n_users,
                    "n_items": data_context.n_items,
                    "sparsity": data_context.sparsity,
                    "action_specific_prompt": action_prompt,
                    "format_instructions": format_instructions,
                }
            )

            # Log the structured response
            self.logger.info(
                "Received structured LLM response",
                action=action.value,
                feature_name=response.get("feature_name"),
                feature_type=response.get("feature_type"),
            )
            self.logger.debug("Full structured response", response=response)

            # Convert to FeatureDefinition
            feature = self._convert_response_to_feature(response)
            return feature

        except Exception as e:
            self.logger.error("Structured LLM feature generation failed", error=str(e))
            return None

    def _convert_response_to_feature(
        self, response: Dict[str, Any]
    ) -> Optional[FeatureDefinition]:
        """Convert LangChain structured response to FeatureDefinition with validation.

        Args:
            response: Structured LLM response.

        Returns:
            FeatureDefinition object.
        """
        try:
            # Map FeatureTypeEnum to FeatureType
            feature_type_mapping = {
                FeatureTypeEnum.CODE_BASED: FeatureType.CODE_BASED,
                FeatureTypeEnum.LLM_BASED: FeatureType.LLM_BASED,
                FeatureTypeEnum.HYBRID: FeatureType.HYBRID,
            }

            feature_type = feature_type_mapping.get(
                response["feature_type"], FeatureType.CODE_BASED
            )

            # Validate and fix code-based implementations
            if feature_type == FeatureType.CODE_BASED:
                implementation = response["implementation"]

                # Validate the code follows required patterns
                validated_code = self._validate_and_fix_code(
                    implementation, response["feature_name"]
                )

                return FeatureDefinition(
                    name=response["feature_name"],
                    feature_type=feature_type,
                    description=response["description"],
                    code=validated_code,
                    dependencies=response.get("dependencies", []),
                )
            elif feature_type == FeatureType.LLM_BASED:
                return FeatureDefinition(
                    name=response["feature_name"],
                    feature_type=feature_type,
                    description=response["description"],
                    llm_prompt=response["implementation"],
                    text_columns=response.get("text_columns", ["title", "authors"]),
                    dependencies=response.get("dependencies", []),
                )
            else:  # HYBRID
                return FeatureDefinition(
                    name=response["feature_name"],
                    feature_type=feature_type,
                    description=response["description"],
                    llm_prompt=response["implementation"],
                    text_columns=response.get("text_columns", ["title"]),
                    postprocessing_code="result = llm_df['llm_value'].astype(float)",
                    dependencies=response.get("dependencies", []),
                )

        except Exception as e:
            self.logger.error("Failed to convert response to feature", error=str(e))
            return None

    def _validate_and_fix_code(self, code: str, feature_name: str) -> str:
        """Validate and fix generated code to ensure it follows required patterns.

        Args:
            code: Generated code string
            feature_name: Name of the feature for error reporting

        Returns:
            Validated and potentially fixed code
        """
        # Remove any markdown formatting
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # Check for common issues and fix them
        issues_found = []

        # 1. Check if code sets 'result' variable
        if "result =" not in code and "result=" not in code:
            issues_found.append("No 'result' variable assignment found")

            # Try to fix by looking for common patterns
            if code.strip().endswith(")") and ("df.groupby" in code or "df." in code):
                # Likely missing result assignment
                code = f"result = {code}"
                issues_found.append("Fixed: Added 'result =' assignment")

        # 2. Check for 'data' instead of 'df'
        if "data." in code or "data[" in code or "data.groupby" in code:
            issues_found.append("Found 'data' reference, replacing with 'df'")
            code = (
                code.replace("data.", "df.")
                .replace("data[", "df[")
                .replace("data.groupby", "df.groupby")
            )

        # 3. Fix scalar fillna issues - common LLM error
        if ".fillna(" in code and (
            "mean()" in code or "sum()" in code or "std()" in code
        ):
            # Check if fillna is being called on potential scalar results
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if "result =" in line and ".fillna(" in line:
                    # If the line has aggregation functions that might return scalars
                    if any(
                        agg in line
                        for agg in [
                            ".mean()",
                            ".sum()",
                            ".std()",
                            ".var()",
                            ".median()",
                        ]
                    ):
                        # Check if it's a groupby operation that should return a Series
                        if ".groupby(" not in line:
                            # This is likely a scalar operation, remove fillna
                            new_line = line.replace(".fillna(0)", "").replace(
                                ".fillna()", ""
                            )
                            lines[i] = new_line
                            issues_found.append(
                                "Fixed: Removed fillna from scalar operation"
                            )
                        else:
                            # This is a groupby, ensure it's properly structured
                            if "pd.Series(" not in line and "pd.DataFrame(" not in line:
                                # Wrap the result to ensure it's a Series
                                assignment_part = line.split("=")[0].strip()
                                expression_part = line.split("=", 1)[1].strip()
                                lines[i] = (
                                    f"{assignment_part} = pd.Series({expression_part})"
                                )
                                issues_found.append(
                                    "Fixed: Wrapped groupby result in pd.Series"
                                )
            code = "\n".join(lines)

        # 4. Fix user_id KeyError issues - another common LLM error
        if "'user_id'" in code and ".reset_index()" in code and ".explode(" in code:
            # This pattern often loses the user_id index
            issues_found.append("Found complex operation that might lose user_id index")
            # Try to fix by ensuring user_id is preserved
            if ".reset_index()" in code and ".groupby('user_id')" in code:
                # Replace reset_index() with reset_index().set_index('user_id') if appropriate
                code = code.replace(".reset_index()", ".reset_index(drop=False)")
                issues_found.append("Fixed: Preserved user_id index in reset_index")

        # 5. Ensure proper pandas operations
        required_patterns = ["df.", "result ="]
        missing_patterns = [p for p in required_patterns if p not in code]
        if missing_patterns:
            issues_found.append(f"Missing required patterns: {missing_patterns}")

        # 6. Add fillna() for robustness if missing (but only for Series operations)
        if "groupby" in code and "fillna" not in code and "result =" in code:
            # Add fillna to the result assignment, but only if it's a Series operation
            if code.count("result =") == 1:
                result_line = [line for line in code.split("\n") if "result =" in line][
                    0
                ]
                # Only add fillna if it's a groupby operation (which returns Series)
                if (
                    ".groupby(" in result_line
                    and not any(
                        scalar_op in result_line
                        for scalar_op in [".mean()", ".sum()", ".count()"]
                    )
                    or ".groupby(" in result_line
                ):
                    if not result_line.strip().endswith(".fillna(0)"):
                        new_result_line = result_line.rstrip() + ".fillna(0)"
                        code = code.replace(result_line, new_result_line)
                        issues_found.append("Added .fillna(0) for robustness")

        # 7. Fix common string operations that cause errors
        if ".split(',')" in code and ".apply(" in code:
            # Common pattern that can cause issues
            if "lambda x:" in code and "len(set(" in code:
                # This often creates complex nested operations that fail
                issues_found.append("Simplified complex string operation")
                # Replace with simpler pattern
                code = code.replace(
                    "lambda x: len(set(x))",
                    "lambda x: len(set(str(x).split(',')) if pd.notna(x) else [])",
                )

        # 8. Ensure imports for complex operations
        if "combinations(" in code and "from itertools import combinations" not in code:
            code = "from itertools import combinations\n" + code
            issues_found.append("Added missing itertools import")

        # 9. Fix DataFrame column access patterns
        if "df['authors'].str.split(',').apply(" in code:
            # This pattern often causes issues with complex operations
            code = code.replace(
                "df['authors'].str.split(',').apply(",
                "df['authors'].fillna('').str.split(',').apply(",
            )
            issues_found.append("Added fillna for string operations")

        # Log validation results
        if issues_found:
            self.logger.info(
                "Code validation and fixes applied",
                feature_name=feature_name,
                issues=issues_found,
                original_code_length=len(code),
            )

        # Final validation - try to parse the code
        try:
            compile(code, "<feature_code>", "exec")
        except SyntaxError as e:
            self.logger.error(
                "Generated code has syntax errors after validation",
                feature_name=feature_name,
                error=str(e),
                code=code,
            )
            # Return a safe fallback
            return "# Fallback due to syntax error in generated code\nresult = df.groupby('user_id')['rating'].mean().fillna(0)"

        return code

    def _generate_heuristic_feature(
        self,
        action: MCTSAction,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Optional[FeatureDefinition]:
        """Generate feature using heuristic rules when LLM is not available.

        Args:
            action: MCTS action to take.
            action_context: Action context.
            data_context: Data context.

        Returns:
            Generated feature definition.
        """
        # Simple heuristic feature generation
        feature_count = len(action_context.current_features.features)

        if action == MCTSAction.ADD:
            # Generate basic statistical features
            if feature_count == 0:
                return FeatureDefinition(
                    name="user_avg_rating",
                    feature_type=FeatureType.CODE_BASED,
                    description="Average rating given by user",
                    code="result = df.groupby('user_id')['rating'].mean()",
                    dependencies=["user_id", "rating"],
                )
            elif feature_count == 1:
                return FeatureDefinition(
                    name="user_rating_count",
                    feature_type=FeatureType.CODE_BASED,
                    description="Number of ratings given by user",
                    code="result = df.groupby('user_id').size()",
                    dependencies=["user_id"],
                )
            else:
                return FeatureDefinition(
                    name="user_rating_variance",
                    feature_type=FeatureType.CODE_BASED,
                    description="Variance in user ratings",
                    code="result = df.groupby('user_id')['rating'].var().fillna(0)",
                    dependencies=["user_id", "rating"],
                )

        # For other actions, return a simple variation
        return FeatureDefinition(
            name=f"heuristic_feature_{feature_count + 1}",
            feature_type=FeatureType.CODE_BASED,
            description="Heuristically generated feature",
            code="result = df.groupby('user_id')['rating'].std().fillna(0)",
            dependencies=["user_id", "rating"],
        )

    def _summarize_performance(self, performance_history: List) -> str:
        """Summarize performance history for LLM context.

        Args:
            performance_history: List of performance evaluations.

        Returns:
            Performance summary string.
        """
        if not performance_history:
            return "No previous performance data available."

        recent_scores = [eval.overall_score for eval in performance_history[-5:]]

        summary = f"Recent performance scores: {recent_scores}\n"
        summary += f"Best score: {max(recent_scores):.3f}\n"
        summary += f"Latest score: {recent_scores[-1]:.3f}\n"

        if len(recent_scores) > 1:
            trend = (
                "improving" if recent_scores[-1] > recent_scores[-2] else "declining"
            )
            summary += f"Trend: {trend}"

        return summary

    def _get_feature_implementation(
        self, feature_name: str, action_context: ActionContext
    ) -> str:
        """Get implementation of a specific feature.

        Args:
            feature_name: Name of the feature.
            action_context: Action context.

        Returns:
            Feature implementation string.
        """
        feature = action_context.current_features.get_feature_by_name(feature_name)
        if feature:
            return feature.code or feature.llm_prompt or "No implementation found"
        return "Feature not found"

    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate execution context.

        Args:
            context: Context to validate.

        Returns:
            True if context is valid.
        """
        required_keys = ["data_context", "action_context"]
        return all(key in context for key in required_keys)

    async def reflect_on_performance(
        self,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Dict[str, Any]:
        """Reflect on current performance using structured output.

        Args:
            action_context: Action context with performance history.
            data_context: Data context.

        Returns:
            Structured reflection results with insights and recommendations.
        """
        if not self.use_llm or not self.reflection_chain:
            return {"reflection": "LLM not available for reflection"}

        try:
            # Prepare context for reflection
            feature_descriptions = [
                f"{f.name}: {f.description}"
                for f in action_context.current_features.features
            ]

            performance_metrics = "No metrics available"
            if action_context.performance_history:
                latest = action_context.performance_history[-1]
                performance_metrics = f"Overall score: {latest.overall_score:.3f}"

            performance_history = self._summarize_performance(
                action_context.performance_history
            )

            # Get format instructions
            parser = JsonOutputParser(pydantic_object=ReflectionResponse)
            format_instructions = parser.get_format_instructions()

            # Log the reflection prompt for auditing
            self.logger.info(
                "Sending structured reflection prompt to LLM",
                model=self.config.llm.model_name,
            )

            # Invoke structured reflection chain
            response = await self.reflection_chain(
                {
                    "feature_descriptions": "\n".join(feature_descriptions),
                    "performance_metrics": performance_metrics,
                    "performance_history": performance_history,
                    "n_users": data_context.n_users,
                    "n_items": data_context.n_items,
                    "sparsity": data_context.sparsity,
                    "format_instructions": format_instructions,
                }
            )

            # Log the structured reflection response
            self.logger.info(
                "Received structured reflection response",
                recommended_action=response.get("recommended_action"),
            )
            self.logger.debug("Full reflection response", response=response)

            return {
                "reflection": response,
                "timestamp": time.time(),
                "context": "performance_analysis",
            }

        except Exception as e:
            self.logger.error("Structured reflection failed", error=str(e))
            return {"error": f"Reflection failed: {str(e)}"}

    async def cleanup(self) -> None:
        """Cleanup feature agent resources."""
        if self.llm_client:
            # Close LLM client if needed
            pass
        await super().cleanup()
