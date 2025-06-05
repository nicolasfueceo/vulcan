#!/usr/bin/env python
"""Feature generation agent for VULCAN system."""

import os
import re
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vulcan.schemas import (
    ActionContext,
    DataContext,
    EvolutionAction,
    FeatureDefinition,
    FeatureGenerationResponse,
    FeatureType,
    FeatureTypeEnum,
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
For hybrid features: This will be handled by the executor with proper LLM integration.

IMPORTANT: Generate features that are DIFFERENT from existing features. Avoid simple variations of the same concept."""

FEATURE_GENERATION_USER_PROMPT = """Before generating the feature, list 2-3 different strategies to derive a new user feature. Then select the most promising one and explain why. Finally, implement it.

Current Context:
- Action: {action}
- Current features: {current_features}
- Performance history: {performance_summary}
- Data schema: {data_schema}
- Available text columns: {text_columns}

EXISTING FEATURES (DO NOT DUPLICATE):
{existing_features_detail}

Data Statistics:
- Number of users: {n_users}
- Number of items: {n_items}
- Data sparsity: {sparsity:.3f}

{action_specific_prompt}

Requirements:
1. Generate a feature that is SUBSTANTIALLY DIFFERENT from existing features
2. Explore new aspects of user behavior not covered by current features
3. Consider computational cost vs. potential benefit
4. Provide clear implementation following the required patterns
5. ENSURE your code sets a 'result' variable and uses 'df' not 'data'

DIVERSITY GUIDELINES:
- If existing features focus on ratings, try author/genre preferences
- If existing features use means, try variance/distribution measures
- If existing features are simple counts, try temporal patterns
- If existing features are user-centric, try item-centric perspectives
- Consider cross-feature interactions or ratios

{format_instructions}"""

ACTION_SPECIFIC_PROMPTS = {
    EvolutionAction.GENERATE_NEW: """
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
    EvolutionAction.MUTATE_EXISTING: """
You are MUTATING an existing feature: {{target_feature}}
Current implementation: {{target_implementation}}
Focus on:
- Improving the existing feature logic
- Adjusting parameters or thresholds
- Enhancing the feature extraction method
- Fixing any issues in the current implementation

MANDATORY: Your implementation must set 'result' variable.
""",
}


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
            action = context.get("action_to_perform")
            if not isinstance(action, EvolutionAction):
                self.logger.error(
                    "Action not provided or invalid in context for FeatureAgent.execute",
                    received_action=action,
                )
                pass

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

    async def _generate_feature(
        self,
        action: EvolutionAction,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Optional[FeatureDefinition]:
        """Generate a feature based on the specified action.

        Args:
            action: EvolutionAction to take.
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
        action: EvolutionAction,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Optional[FeatureDefinition]:
        """Generate a feature using LLM with structured output."""
        if not self.use_llm or not self.feature_chain:
            self.logger.warning(
                "LLM not configured or chain not initialized, skipping LLM generation."
            )
            return None

        parser = JsonOutputParser(pydantic_object=FeatureGenerationResponse)
        format_instructions = parser.get_format_instructions()

        # Prepare context for the prompt
        current_features_summary = (
            ", ".join([f.name for f in action_context.current_features.features])
            if action_context.current_features.features
            else "No features yet"
        )
        performance_summary = self._summarize_performance(
            action_context.performance_history
        )
        data_schema_summary = ", ".join(data_context.data_schema.keys())
        text_columns_summary = ", ".join(data_context.text_columns)

        # Apply prompt memory configuration
        features_for_prompt = action_context.current_features.features
        num_original_features = len(features_for_prompt)
        prompt_memory_type = self.config.llm.prompt_memory_type
        prompt_memory_max = self.config.llm.prompt_memory_max_features

        if (
            prompt_memory_type == "limited"
            and prompt_memory_max is not None
            and prompt_memory_max > 0
        ):
            if num_original_features > prompt_memory_max:
                features_for_prompt = features_for_prompt[-prompt_memory_max:]
                self.logger.info(
                    "Applying 'limited' prompt memory.",
                    action=action.value,
                    max_features=prompt_memory_max,
                    features_included=len(features_for_prompt),
                    features_omitted=num_original_features - len(features_for_prompt),
                )
        elif prompt_memory_type == "none":
            features_for_prompt = []
            self.logger.info(
                "Applying 'none' prompt memory. No existing features included in prompt.",
                action=action.value,
                features_omitted=num_original_features,
            )
        else:  # full or limited with no max_features set (effectively full)
            self.logger.info(
                "Applying 'full' prompt memory. All existing features included in prompt.",
                action=action.value,
                features_included=num_original_features,
            )

        existing_features_detail_str = "\n".join(
            [
                "- Name: {f.name}, Type: {f.feature_type}, Desc: {f.description}, Code: {f.code_snippet_and_dependencies}"
                for f in features_for_prompt  # Use the (potentially) filtered list
            ]
        )
        if not existing_features_detail_str:
            existing_features_detail_str = (
                "No existing features included in prompt based on memory settings."
            )

        action_specific_prompt_str = self._prepare_action_specific_llm_prompt(
            action, action_context, ACTION_SPECIFIC_PROMPTS[action]
        )

        # Log the prompt for auditing
        self.logger.info(
            "Sending prompt to LLM for feature generation",
            action=action.value,
            model=self.config.llm.model_name,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            # Prompt content itself is too large for a single log line;
            # refer to saved artifacts if FeatureAgent has callbacks for that.
            prompt_artifact_info="Full prompt details saved via callback if configured.",
        )

        # Invoke the structured chain
        response = await self.feature_chain(
            {
                "action": action.value,
                "current_features": current_features_summary,
                "performance_summary": performance_summary,
                "data_schema": data_schema_summary,
                "text_columns": text_columns_summary,
                "n_users": data_context.n_users,
                "n_items": data_context.n_items,
                "sparsity": data_context.sparsity,
                "action_specific_prompt": action_specific_prompt_str,
                "format_instructions": format_instructions,
                "existing_features_detail": existing_features_detail_str,
            }
        )

        # Log the structured response
        self.logger.info(
            "Received structured LLM response",
            action=action.value,
            feature_name=response.get("feature_name"),
            feature_type=response.get("feature_type"),
            # Full response content is too large; refer to saved artifacts.
            response_artifact_info="Full response details saved via callback if configured.",
        )
        self.logger.debug(
            "Full structured LLM response details", response_details=response
        )  # Keep debug for full details

        # Convert to FeatureDefinition
        feature = self._convert_response_to_feature(response)
        return feature

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
                    llm_chain_of_thought_reasoning=response.get(
                        "chain_of_thought_reasoning"
                    ),
                )
            elif feature_type == FeatureType.LLM_BASED:
                return FeatureDefinition(
                    name=response["feature_name"],
                    feature_type=feature_type,
                    description=response["description"],
                    llm_prompt=response["implementation"],
                    text_columns=response.get("text_columns", ["title", "authors"]),
                    dependencies=response.get("dependencies", []),
                    llm_chain_of_thought_reasoning=response.get(
                        "chain_of_thought_reasoning"
                    ),
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
                    llm_chain_of_thought_reasoning=response.get(
                        "chain_of_thought_reasoning"
                    ),
                )

        except Exception as e:
            self.logger.error(
                "Failed to convert response to feature", error=str(e), exc_info=True
            )
            return None

    def _vfc_strip_markdown(self, code: str, feature_name: str) -> str:
        """Strips markdown formatting (e.g., ```python) from the code string."""
        self.logger.debug(
            "Attempting to strip markdown",
            feature_name=feature_name,
            code_prefix=code[:20],
        )
        stripped_code = code.strip()
        if stripped_code.startswith("```python"):
            stripped_code = stripped_code[9:].strip()
            self.logger.info("Stripped ```python prefix", feature_name=feature_name)
        elif stripped_code.startswith("```"):
            stripped_code = stripped_code[3:].strip()
            self.logger.info("Stripped ``` prefix", feature_name=feature_name)
        if stripped_code.endswith("```"):
            stripped_code = stripped_code[:-3].strip()
            self.logger.info("Stripped ``` suffix", feature_name=feature_name)
        return stripped_code

    def _vfc_ensure_result_assignment(self, code: str, feature_name: str) -> str:
        """Ensures the code assigns to a 'result' variable, attempting a fix if common patterns are missed."""
        if "result =" not in code and "result=" not in code:
            self.logger.warning(
                f"No 'result' variable assignment found in code for feature '{feature_name}'",
                issue_type="missing_result_assignment",
            )
            # Attempt fix for common LLM pattern: forgetting 'result =' before a pandas expression
            if code.strip().endswith(")") and (
                "df.groupby" in code or "df." in code or "pd." in code
            ):
                code = f"result = {code}"
                self.logger.info(
                    f"Fixed: Added 'result =' assignment to feature '{feature_name}' based on common pattern",
                    fix_type="added_result_assignment",
                )
        return code

    def _vfc_replace_data_with_df(self, code: str, feature_name: str) -> str:
        """Replaces common incorrect 'data' references with 'df' for DataFrame access."""
        if "data." in code or "data[" in code or "data.groupby" in code:
            self.logger.warning(
                f"Found 'data' reference instead of 'df' in code for feature '{feature_name}'",
                issue_type="data_reference_found",
            )
            original_code = code
            code = (
                code.replace("data.", "df.")
                .replace("data[", "df[")
                .replace("data.groupby", "df.groupby")
            )
            if code != original_code:
                self.logger.info(
                    f"Fixed: Replaced 'data' with 'df' for feature '{feature_name}'",
                    fix_type="replaced_data_with_df",
                )
        return code

    def _vfc_fix_scalar_fillna(self, code: str, feature_name: str) -> str:
        """Attempts to fix common LLM errors of using .fillna() on likely scalar results from aggregations."""
        if ".fillna(" not in code:  # Quick check
            return code

        lines = code.split("\n")
        modified_lines = []
        changed = False

        # Regex to find a pandas scalar aggregation method call
        # (e.g., .mean(), .sum()) possibly followed by .fillna(...)
        # This tries to avoid matching .fillna on a Series resulting from a groupby chain.
        scalar_agg_fillna_pattern = re.compile(
            r"(\.mean\(\)|\.sum\(\)|\.std\(\)|\.var\(\)|\.median\(\)|\.min\(\)|\.max\(\)|\.count\(\)|\.size\(\))"  # Scalar aggregation
            r"\s*(\.fillna\s*\([^)]*\))"  # Followed by .fillna(...)
        )

        for line_idx, line in enumerate(lines):
            new_line = line
            # Only apply if 'result =' is in the line to target primary result assignments
            if "result =" in line:
                # Check if the line contains a groupby operation. If so, fillna might be appropriate for the resulting Series.
                is_groupby_chain = ".groupby(" in line

                match = scalar_agg_fillna_pattern.search(line)
                if match and not is_groupby_chain:
                    # If it is NOT a groupby chain and a scalar_agg_fillna pattern is found, remove fillna.
                    fillna_part = match.group(2)
                    new_line = line.replace(fillna_part, "")
                    self.logger.info(
                        f"Fixed: Removed '{fillna_part}' from likely scalar operation on feature '{feature_name}' (line {line_idx + 1})",
                        original_line_snippet=line.strip(),
                        new_line_snippet=new_line.strip(),
                        fix_type="removed_scalar_fillna",
                    )
                    changed = True
            modified_lines.append(new_line)

        return "\n".join(modified_lines) if changed else code

    def _vfc_preserve_user_id_index(self, code: str, feature_name: str) -> str:
        """Placeholder for logic to fix operations (e.g., explode, pivot_table with reset_index)
        that might unintentionally drop 'user_id' index. Currently logs a warning."""
        # Robustly fixing this requires deep semantic understanding of the pandas operations.
        # Current implementation primarily logs potential issues for manual review.
        if (
            "'user_id'" in code
            and ".reset_index()" in code
            and (".explode(" in code or ".pivot_table(" in code or ".unstack(" in code)
        ):
            self.logger.warning(
                f"Found complex operation pattern (e.g., explode/pivot/unstack with reset_index) in feature '{feature_name}'. "
                "This might lose 'user_id' index. Manual review of the generated code is recommended.",
                issue_type="potential_user_id_loss_risk",
            )
        return code

    def _vfc_check_required_patterns(self, code: str, feature_name: str) -> str:
        """Checks for presence of required patterns like 'df.' (implying pandas usage) and 'result ='. Primarily logs warnings."""
        required_patterns = ["df.", "result ="]
        missing_patterns = [p for p in required_patterns if p not in code]
        if missing_patterns:
            self.logger.warning(
                f"Missing required patterns {missing_patterns} for feature '{feature_name}'. 'df.' implies pandas usage, 'result =' for final assignment.",
                issue_type="missing_required_patterns",
            )
        return code

    def _vfc_add_robust_fillna(self, code: str, feature_name: str) -> str:
        """Adds .fillna(0) for robustness to GroupBy operations if not already present and seems appropriate for numerical results."""
        if "groupby" not in code or "result =" not in code:  # Quick check
            return code

        lines = code.split("\n")
        modified_lines = []
        changed = False
        for line_idx, line in enumerate(lines):
            new_line = line
            # Target lines assigning a result from a groupby operation, not already having fillna.
            if (
                "result =" in line
                and ".groupby(" in line
                and not re.search(r"\.fillna\s*\([^)]*\)", line)
            ):
                # Heuristic: Add .fillna(0) if it looks like a groupby creating a Series expected to be numeric.
                # Avoid adding .fillna(0) if the result is clearly non-numeric (e.g., .agg(list) or string ops).
                # This is a common pattern for numerical aggregations by group.
                # Check if common scalar-producing aggregations are NOT the very last step (if so, _vfc_fix_scalar_fillna handles it)
                if not line.strip().endswith(".fillna(0)") and not any(
                    line.strip().endswith(s_agg) for s_agg in [".first()", ".last()"]
                ):  # Add more non-numeric terminal ops if needed
                    new_line = line.rstrip() + ".fillna(0)"
                    self.logger.info(
                        f"Fixed: Added .fillna(0) for robustness to groupby operation in feature '{feature_name}' (line {line_idx + 1})",
                        fix_type="added_groupby_fillna",
                    )
                    changed = True
            modified_lines.append(new_line)
        return "\n".join(modified_lines) if changed else code

    def _vfc_simplify_string_operations(self, code: str, feature_name: str) -> str:
        """Simplifies common complex or error-prone string operations generated by LLMs, ensuring NaN and type safety."""
        # Example: .str.split(',').apply(lambda x: len(set(x))) needs to handle NaNs and non-strings.
        if ".split(" in code and ".apply(" in code and "lambda x: len(set(" in code:
            self.logger.info(
                f"Attempting to simplify complex string operation pattern in feature '{feature_name}'",
                issue_type="complex_string_op_simplify",
            )
            original_code = code
            # Robust replacement for patterns like: lambda x: len(set(x)) or lambda x: len(set(x.split(',')))
            # This handles potential NaNs and ensures string conversion before split, returning 0 for problematic cases.
            code = re.sub(
                # Matches: lambda x: len(set( <optional str(x)> <optional .split(...) > ))
                r"lambda\s+x\s*:\s*len\s*\(\s*set\s*\(\s*(?:str\s*\(\s*x\s*\))?s*x?\s*(?:\.split\s*\(\s*['\"]?,['\"]?\s*\))?\s*\)\s*\)",
                "lambda x: len(set(str(x).split(',') if pd.notna(x) and x is not None else [])) if pd.notna(x) and x is not None else 0",
                code,
            )
            if code != original_code:
                self.logger.info(
                    f"Fixed: Simplified string operation with NaN/type handling for feature '{feature_name}'",
                    fix_type="simplified_string_op",
                )
                # Ensure pd is imported due to pd.notna() - handled by _vfc_add_missing_imports
        return code

    def _vfc_add_missing_imports(self, code: str, feature_name: str) -> str:
        """Adds common missing imports (e.g., itertools, pandas) based on code content. Ensures import is at the top."""
        required_imports = []
        if "combinations(" in code and "from itertools import combinations" not in code:
            required_imports.append("from itertools import combinations")

        # Check for pandas usage (pd.function, pd.Series, etc.)
        if re.search(r"pd\.(notna|Series|DataFrame|to_datetime|Timestamp)", code) or (
            "pd " in code and "import pandas as pd" not in code.split("\n")[0]
        ):
            if not any(
                imp_line.startswith("import pandas as pd")
                for imp_line in code.split("\n")
            ):
                required_imports.append("import pandas as pd")

        if not required_imports:
            return code

        self.logger.info(
            f"Adding missing imports for feature '{feature_name}': {required_imports}",
            fix_type="added_missing_imports",
        )
        current_lines = code.split("\n")
        existing_imports = [
            line
            for line in current_lines
            if line.startswith("import") or line.startswith("from")
        ]
        other_code_lines = [
            line
            for line in current_lines
            if not (line.startswith("import") or line.startswith("from"))
        ]

        # Add new imports only if not already present (case-sensitive check for simplicity)
        final_imports = list(
            dict.fromkeys(
                existing_imports
                + [imp for imp in required_imports if imp not in existing_imports]
            )
        )

        return "\n".join(final_imports + other_code_lines)

    def _vfc_fix_df_column_access_fillna(self, code: str, feature_name: str) -> str:
        """Adds .fillna('') before .str access on DataFrame columns (e.g., 'authors', 'title') to prevent errors on NaN values."""
        # Heuristic based on common text column names that might contain NaNs.
        text_columns_heuristic = getattr(
            self.config.data, "text_columns", ["authors", "title"]
        )  # Use config if available
        if not text_columns_heuristic:
            return code  # No columns to check

        lines = code.split("\n")
        modified_lines = []
        changed = False

        for line_idx, line in enumerate(lines):
            new_line = line
            for col_name in text_columns_heuristic:
                # Regex to find df['col_name'].str (but not df['col_name'].fillna(...).str)
                # Handles single or double quotes around column name.
                pattern = re.compile(
                    rf"(df\[['\"]{re.escape(col_name)}['\"]\])(?!\.fillna\s*\([^)]*\))(\s*\.str\.)"
                )

                def replacer_for_str_access(match):
                    nonlocal changed  # To modify 'changed' in outer scope
                    changed = True
                    self.logger.info(
                        f"Fixed: Added .fillna('') for robust string access on df['{col_name}'] in feature '{feature_name}' (line {line_idx + 1})",
                        fix_type="added_str_fillna_for_column_access",
                    )
                    return f"{match.group(1)}.fillna(''){match.group(2)}"  # construct: df['col'].fillna('').str.

                new_line = pattern.sub(replacer_for_str_access, new_line)
            modified_lines.append(new_line)

        return "\n".join(modified_lines) if changed else code

    def _vfc_compile_code(self, code: str, feature_name: str) -> str:
        """Final compilation check for the generated code. Returns a safe fallback if SyntaxError occurs."""
        try:
            compile(code, f"<feature_code_for_{feature_name}>", "exec")
            self.logger.debug(
                f"Code for feature '{feature_name}' compiled successfully after validation."
            )
        except SyntaxError as e:
            self.logger.error(
                f"Generated code for feature '{feature_name}' has syntax errors after all validation attempts.",
                error_message=str(e),
                line_number=e.lineno,
                offset=e.offset,
                code_submitted_to_compile=code,
                exc_info=True,
            )
            # Return a safe, simple fallback feature code
            fallback_code = "import pandas as pd\n# Fallback due to syntax error in generated code\nresult = df.groupby('user_id')['rating'].mean().fillna(0)"
            self.logger.warning(
                f"Returning fallback code for feature '{feature_name}' due to compilation failure."
            )
            return fallback_code
        return code

    def _validate_and_fix_code(self, code: str, feature_name: str) -> str:
        """Validate and fix generated Python code for features to ensure it follows required patterns and is executable.

        This method applies a series of specific validation and correction rules, executed in a defined order.
        Each rule is encapsulated in a helper method (_vfc_*).

        Args:
            code: The generated code string from the LLM.
            feature_name: Name of the feature, used for logging and error reporting.

        Returns:
            Validated and potentially fixed code string. If unfixable syntax errors persist,
            a safe fallback code is returned.
        """
        self.logger.info(
            f"Starting code validation/fixing process for feature: '{feature_name}'",
            original_code_prefix=code[:100] + "...",
        )
        original_code_for_comparison = str(
            code
        )  # Keep an exact copy for final comparison

        # 0. Initial: Add any obviously missing imports based on common patterns first.
        code = self._vfc_add_missing_imports(code, feature_name)
        # 1. Strip markdown like ```python ``` that LLMs sometimes add.
        code = self._vfc_strip_markdown(code, feature_name)
        # 2. Ensure 'result = ' for final assignment.
        code = self._vfc_ensure_result_assignment(code, feature_name)
        # 3. Replace common 'data. ...' with 'df. ...'.
        code = self._vfc_replace_data_with_df(code, feature_name)
        # 4. Fix .fillna() on likely scalar pandas results (common LLM error).
        code = self._vfc_fix_scalar_fillna(code, feature_name)
        # 5. Log if operations might drop user_id index (complex to auto-fix reliably).
        code = self._vfc_preserve_user_id_index(code, feature_name)
        # 6. Check for presence of essential patterns ('df.', 'result ='). Logs only.
        code = self._vfc_check_required_patterns(code, feature_name)
        # 7. Add .fillna(0) to groupby results for robustness if numeric and not handled.
        code = self._vfc_add_robust_fillna(code, feature_name)
        # 8. Simplify common complex/error-prone string operations.
        code = self._vfc_simplify_string_operations(code, feature_name)
        # 9. Add .fillna('') before .str access on known text columns.
        code = self._vfc_fix_df_column_access_fillna(code, feature_name)

        # 10. Re-run import adder in case some fixes introduced new needs (e.g., pd.notna from string simplification).
        code = self._vfc_add_missing_imports(code, feature_name)

        # 11. Final compilation check. If this fails, a fallback code is returned.
        final_code = self._vfc_compile_code(code, feature_name)

        if final_code != original_code_for_comparison:
            self.logger.info(
                f"Code for feature '{feature_name}' was modified by validation/fixing process.",
                status="CodeModified",
                # For brevity in standard logs, detailed diffs are not logged here.
                # Original snippet: original_code_for_comparison[:150] + "...",
                # Final snippet: final_code[:150] + "..."
            )
            self.logger.debug(
                "Code change details",
                feature_name=feature_name,
                original_code=original_code_for_comparison,
                final_code=final_code,
            )
        else:
            self.logger.info(
                f"Code for feature '{feature_name}' passed validation without substantive changes or used fallback.",
                status="CodeUnchangedOrFallback",
            )
        return final_code

    def _prepare_action_specific_llm_prompt(
        self,
        action: EvolutionAction,
        action_context: ActionContext,
        base_action_prompt_template: str,
    ) -> str:
        """Prepares the action-specific part of the LLM prompt by filling placeholders (e.g., {{target_feature}}).

        Args:
            action: The EvolutionAction (GENERATE_NEW, MUTATE_EXISTING) being performed.
            action_context: The context for the action, including current features and potential targets.
            base_action_prompt_template: The basic string template for the action prompt (from ACTION_SPECIFIC_PROMPTS).

        Returns:
            A string with placeholders filled, tailored for the specific action and context.
        """
        self.logger.debug(
            f"Preparing action-specific LLM prompt for action: {action.value}"
        )
        action_prompt = base_action_prompt_template  # Start with the generic template for the action type

        target_feature_name: Optional[str] = None
        target_implementation: str = "N/A (Not Applicable or Not Found)"  # Default placeholder if no specific impl is found

        if action == EvolutionAction.MUTATE_EXISTING:
            # Determine the target feature for mutation.
            current_features_list = action_context.current_features.features
            explicit_target_name = getattr(
                action_context, "target_feature_name", None
            ) or getattr(action_context, "mutation_target", None)

            if explicit_target_name and any(
                f.name == explicit_target_name for f in current_features_list
            ):
                target_feature_name = explicit_target_name
                self.logger.info(
                    f"Using explicit target for {action.value}: {target_feature_name}"
                )
            elif current_features_list:
                # Fallback: use the worst-performing feature if no explicit target is set.
                target_feature_name = action_context.get_worst_performing_feature()
                if target_feature_name:
                    self.logger.info(
                        f"No explicit target for {action.value}, selected worst feature: '{target_feature_name}' from context."
                    )
                else:
                    target_feature_name = current_features_list[-1].name
                    self.logger.warning(
                        f"Could not determine worst feature for {action.value}; picking last feature '{target_feature_name}' as target."
                    )
            else:
                self.logger.warning(
                    f"No features in context to target for {action.value}. Prompt will be generic."
                )
                target_feature_name = (
                    "a_selected_feature"  # Generic placeholder for LLM
                )

            if target_feature_name:
                target_implementation = self._get_feature_implementation(
                    target_feature_name, action_context
                )

            action_prompt = action_prompt.replace(
                "{{target_feature}}", target_feature_name or "selected_feature"
            ).replace("{{target_implementation}}", target_implementation)
            self.logger.debug(
                f"Prepared {action.value} prompt targeting: '{target_feature_name}'"
            )

        action_prompt = action_prompt
        return action_prompt

    def _generate_heuristic_feature(
        self,
        action: EvolutionAction,
        action_context: ActionContext,
        data_context: DataContext,
    ) -> Optional[FeatureDefinition]:
        """Generate feature using heuristic rules when LLM is not available.

        Args:
            action: EvolutionAction to take.
            action_context: Action context.
            data_context: Data context.

        Returns:
            Generated feature definition.
        """
        # Simple heuristic feature generation
        feature_count = len(action_context.current_features.features)

        if action == EvolutionAction.GENERATE_NEW:
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

    async def cleanup(self) -> None:
        """Cleanup feature agent resources."""
        if self.llm_client:
            # Close LLM client if needed
            pass
        await super().cleanup()
