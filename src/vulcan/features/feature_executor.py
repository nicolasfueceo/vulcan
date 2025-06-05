"""Feature execution engine for VULCAN system."""

import asyncio
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vulcan.schemas import (
    DataContext,
    FeatureDefinition,
    FeatureType,
    FeatureValue,
    LLMFeatureExtractionResponse,
    VulcanConfig,
)
from vulcan.utils import get_vulcan_logger

logger = structlog.get_logger(__name__)

# LLM feature extraction prompt with JSON output instructions
LLM_FEATURE_EXTRACTION_SYSTEM_PROMPT = """You are a feature extraction expert. Your task is to analyze text content and extract a numerical feature value based on the given prompt.

You must respond with a valid JSON object that matches the exact schema provided. Do not include any additional text, explanations, or markdown formatting outside the JSON response.

The feature_value should be a numerical value (float) that represents the characteristic described in the prompt.
The confidence should be between 0 and 1, indicating how confident you are in the extracted value.
"""

LLM_FEATURE_EXTRACTION_USER_PROMPT = """{feature_prompt}

Text to analyze:
{text_content}

Extract a numerical feature value from this text and provide your confidence in the extraction.

{format_instructions}"""


class FeatureExecutor:
    """Executes feature definitions on real data with structured LLM output."""

    def __init__(self, config: VulcanConfig) -> None:
        """Initialize feature executor.

        Args:
            config: VULCAN configuration.
        """
        self.config = config
        self.logger = get_vulcan_logger(__name__)
        self.llm_client = None
        self.llm_chain = None
        self._execution_cache: Dict[str, Any] = {}
        self._full_data_cache: Dict[str, pd.DataFrame] = {}  # Cache for full data

    async def initialize(self) -> bool:
        """Initialize the feature executor with LangChain structured output."""
        try:
            if self.config.llm.provider == "openai":
                await self._initialize_langchain()

            self.logger.info("Feature executor initialized")
            return True

        except Exception as e:
            self.logger.error("Failed to initialize feature executor", error=str(e))
            return False

    async def _initialize_langchain(self) -> None:
        """Initialize LangChain client for structured LLM-based features."""
        try:
            import os

            api_key = os.getenv(self.config.llm.api_key_env)
            if api_key:
                # Initialize LangChain OpenAI client
                self.llm_client = ChatOpenAI(
                    model=self.config.llm.model_name,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    api_key=api_key,
                )

                # Setup structured LLM extraction chain
                self._setup_llm_extraction_chain()

                self.logger.info("LangChain client initialized for feature execution")
            else:
                self.logger.warning("OpenAI API key not found")

        except ImportError:
            self.logger.warning("LangChain OpenAI package not installed")

    def _setup_llm_extraction_chain(self) -> None:
        """Setup the LLM feature extraction chain with structured output."""
        # Create parser for LLMFeatureExtractionResponse
        parser = JsonOutputParser(pydantic_object=LLMFeatureExtractionResponse)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", LLM_FEATURE_EXTRACTION_SYSTEM_PROMPT),
                ("human", LLM_FEATURE_EXTRACTION_USER_PROMPT),
            ]
        )

        # Create a simple callable chain
        async def llm_chain_func(inputs):
            formatted_prompt = await prompt.ainvoke(inputs)
            llm_response = await self.llm_client.ainvoke(formatted_prompt)
            return await parser.ainvoke(llm_response)

        self.llm_chain = llm_chain_func

    async def execute_feature(
        self,
        feature: FeatureDefinition,
        data_context: DataContext,
        target_split: str = "train",
        max_records: Optional[int] = None,
    ) -> List[FeatureValue]:
        """Execute a single feature definition and extract values.

        Args:
            feature: Feature definition to execute.
            data_context: Data context containing all splits.
            target_split: Which data split to use ('train', 'validation', 'test').
            max_records: Maximum number of records to process (None for all).

        Returns:
            List of feature values for all users.
        """
        self.logger.info(
            "Executing feature",
            feature_name=feature.name,
            feature_type=feature.feature_type,
            target_split=target_split,
            max_records=max_records,
        )

        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"{feature.name}_{target_split}_full"
            if cache_key in self._execution_cache:
                self.logger.debug(
                    "Using cached feature values", feature_name=feature.name
                )
                return self._execution_cache[cache_key]

            # Load target split data
            data = await self._load_split_data(data_context, target_split, max_records)

            # Execute based on feature type with self-correction
            if feature.feature_type == FeatureType.CODE_BASED:
                feature_values = await self._execute_code_feature_with_correction(
                    feature, data, data_context
                )
            elif feature.feature_type == FeatureType.LLM_BASED:
                feature_values = await self._execute_llm_feature(
                    feature, data, data_context
                )
            elif feature.feature_type == FeatureType.HYBRID:
                feature_values = await self._execute_hybrid_feature(
                    feature, data, data_context
                )
            else:
                raise ValueError(f"Unknown feature type: {feature.feature_type}")

            # Cache results
            self._execution_cache[cache_key] = feature_values

            execution_time = time.time() - start_time
            self.logger.info(
                "Feature executed successfully",
                feature_name=feature.name,
                feature_type=feature.feature_type.value,
                target_split=target_split,
                execution_time=f"{execution_time:.3f}s",
                result_count=len(feature_values),
            )

            return feature_values

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                "Feature execution failed",
                feature_name=feature.name,
                error=str(e),
                execution_time=f"{execution_time:.3f}s",
            )
            raise

    async def _load_split_data(
        self,
        data_context: DataContext,
        target_split: str,
        max_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load actual data for specified split.

        Args:
            data_context: Data context.
            target_split: Target split name.
            max_records: Maximum number of records to process (None for all).

        Returns:
            Data dictionary with real data.
        """
        # For feature execution, we need the FULL data, not just samples
        if hasattr(data_context, "get_full_split_data"):
            # Use the full data loading method for feature execution
            self.logger.info(f"Loading full {target_split} data for feature execution")
            full_df = data_context.get_full_split_data(target_split)
            # Return the DataFrame directly - our execution methods can handle DataFrames
            return full_df

        # Fallback to sample batch for other data contexts
        if hasattr(data_context, "get_sample_batch"):
            self.logger.warning(
                f"Using sample batch for {target_split} - full data method not available"
            )
            return data_context.get_sample_batch(target_split, max_records=max_records)

        # Final fallback to data_context properties
        if target_split == "train":
            return data_context.train_data
        elif target_split in ["validation", "val"]:
            return data_context.validation_data
        elif target_split == "test":
            return data_context.test_data

        raise ValueError(f"Unknown target split: {target_split}")

    async def _execute_code_feature_with_correction(
        self,
        feature: FeatureDefinition,
        data: Dict[str, Any],
        data_context: DataContext,
        max_attempts: int = 3,
    ) -> List[FeatureValue]:
        """Execute code-based feature with self-correction capabilities.

        Args:
            feature: Feature definition.
            data: Target data split.
            data_context: Full data context.
            max_attempts: Maximum correction attempts.

        Returns:
            List of feature values.
        """
        if not feature.code:
            raise ValueError("Code-based feature missing implementation code")

        # Convert data to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        # Log data info for debugging
        self.logger.debug(
            "Executing feature with data info",
            feature_name=feature.name,
            df_shape=df.shape,
            df_columns=list(df.columns),
            data_schema=list(data_context.data_schema.keys()),
        )

        current_code = feature.code
        attempt = 1

        while attempt <= max_attempts:
            try:
                self.logger.debug(
                    "Attempting feature execution",
                    feature_name=feature.name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    code=current_code[:100] + "..."
                    if len(current_code) > 100
                    else current_code,
                )

                # Try to execute the current code
                feature_values = await self._execute_code_feature_single_attempt(
                    feature.name, current_code, df
                )

                if attempt > 1:
                    self.logger.info(
                        "Feature execution succeeded after correction",
                        feature_name=feature.name,
                        attempt=attempt,
                        corrected_code=current_code,
                    )

                return feature_values

            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__

                self.logger.warning(
                    "Feature execution attempt failed",
                    feature_name=feature.name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=error_msg,
                    error_type=error_type,
                )

                if attempt >= max_attempts:
                    # Final attempt failed, raise the error
                    self.logger.error(
                        "Feature execution failed after all correction attempts",
                        feature_name=feature.name,
                        final_error=error_msg,
                        attempts=max_attempts,
                    )
                    raise ValueError(
                        f"Feature execution failed for '{feature.name}' after {max_attempts} attempts. "
                        f"Final error: {error_msg}. "
                        f"Make sure your code sets a 'result' variable and uses 'df' to access the DataFrame."
                    )

                # Try to correct the code using LLM reflection
                try:
                    corrected_code = await self._correct_feature_code(
                        feature.name,
                        current_code,
                        error_msg,
                        error_type,
                        df,
                        data_context,
                        attempt,
                    )

                    if corrected_code and corrected_code != current_code:
                        current_code = corrected_code
                        self.logger.info(
                            "Generated corrected code",
                            feature_name=feature.name,
                            attempt=attempt,
                            corrected_code=corrected_code,
                        )
                    else:
                        self.logger.warning(
                            "Could not generate corrected code, retrying with original",
                            feature_name=feature.name,
                            attempt=attempt,
                        )

                except Exception as correction_error:
                    self.logger.warning(
                        "Code correction failed",
                        feature_name=feature.name,
                        correction_error=str(correction_error),
                    )

                attempt += 1

    async def _execute_code_feature_single_attempt(
        self, feature_name: str, code: str, df: pd.DataFrame
    ) -> List[FeatureValue]:
        """Execute feature code in a single attempt.

        Args:
            feature_name: Name of the feature.
            code: Code to execute.
            df: DataFrame to operate on.

        Returns:
            List of feature values.
        """
        feature_values = []

        # Import numpy for the execution context
        import numpy as np

        # Create comprehensive execution context
        exec_context = {
            "df": df,
            "pd": pd,
            "np": np,
            "result": None,
            # Add common helper functions
            "len": len,
            "set": set,
            "list": list,
            "dict": dict,
            "str": str,
            "float": float,
            "int": int,
        }

        # Execute the feature code
        exec(code, exec_context)

        result = exec_context.get("result")
        if result is None:
            raise ValueError(
                f"Feature code did not set 'result' variable. "
                f"Available variables after execution: {list(exec_context.keys())}"
            )

        self.logger.debug(
            "Feature code executed successfully",
            feature_name=feature_name,
            result_type=type(result).__name__,
            result_shape=getattr(result, "shape", None),
        )

        # Convert result to feature values with robust handling
        if isinstance(result, pd.Series):
            # Handle Series result (most common case)
            for user_id, value in result.items():
                feature_values.append(
                    FeatureValue(
                        user_id=str(user_id),
                        feature_name=feature_name,
                        value=float(value) if pd.notna(value) else None,
                        confidence=1.0,
                    )
                )
        elif isinstance(result, pd.DataFrame):
            # Handle DataFrame result - flatten to Series
            if "user_id" in result.columns:
                # Use the first non-user_id column as the value
                value_col = [col for col in result.columns if col != "user_id"][0]
                for _, row in result.iterrows():
                    feature_values.append(
                        FeatureValue(
                            user_id=str(row["user_id"]),
                            feature_name=feature_name,
                            value=float(row[value_col])
                            if pd.notna(row[value_col])
                            else None,
                            confidence=1.0,
                        )
                    )
            else:
                # Use the first column as values, index as user_id
                for idx, value in result.iloc[:, 0].items():
                    feature_values.append(
                        FeatureValue(
                            user_id=str(idx),
                            feature_name=feature_name,
                            value=float(value) if pd.notna(value) else None,
                            confidence=1.0,
                        )
                    )
        elif isinstance(result, (dict)):
            # Handle dictionary result
            for user_id, value in result.items():
                feature_values.append(
                    FeatureValue(
                        user_id=str(user_id),
                        feature_name=feature_name,
                        value=float(value) if pd.notna(value) else None,
                        confidence=1.0,
                    )
                )
        else:
            # Handle single value result - broadcast to all users
            unique_users = (
                df["user_id"].unique() if "user_id" in df.columns else ["default"]
            )
            for user_id in unique_users:
                feature_values.append(
                    FeatureValue(
                        user_id=str(user_id),
                        feature_name=feature_name,
                        value=float(result) if pd.notna(result) else None,
                        confidence=1.0,
                    )
                )

        self.logger.info(
            "Feature values generated successfully",
            feature_name=feature_name,
            value_count=len(feature_values),
        )

        return feature_values

    async def _correct_feature_code(
        self,
        feature_name: str,
        original_code: str,
        error_msg: str,
        error_type: str,
        df: pd.DataFrame,
        data_context: DataContext,
        attempt: int,
    ) -> Optional[str]:
        """Use LLM to correct failing feature code.

        Args:
            feature_name: Name of the feature.
            original_code: Original failing code.
            error_msg: Error message from execution.
            error_type: Type of error (e.g., KeyError).
            df: DataFrame the code was run on.
            data_context: Data context.
            attempt: Current attempt number.

        Returns:
            Corrected code or None if correction failed.
        """
        if not self.llm_chain:
            # No LLM available, try basic rule-based correction
            return self._apply_basic_corrections(
                original_code, error_msg, error_type, df
            )

        try:
            # Create detailed error context for LLM
            error_context = {
                "feature_name": feature_name,
                "original_code": original_code,
                "error_message": error_msg,
                "error_type": error_type,
                "attempt": attempt,
                "available_columns": list(df.columns),
                "data_shape": df.shape,
                "sample_data": df.head(3).to_dict(),
                "data_schema": list(data_context.data_schema.keys()),
            }

            # Create correction prompt
            correction_prompt = f"""
You are an expert Python data scientist. The following feature code failed to execute and needs correction.

FEATURE: {feature_name}
ATTEMPT: {attempt}

FAILING CODE:
```python
{original_code}
```

ERROR:
Type: {error_type}
Message: {error_msg}

DATA CONTEXT:
- Available columns: {list(df.columns)}
- Data shape: {df.shape}
- Expected schema: {list(data_context.data_schema.keys())}

SAMPLE DATA:
{df.head(3).to_string()}

REQUIREMENTS:
1. Fix the error while maintaining the original intent
2. Code must set a 'result' variable
3. Use 'df' to access the DataFrame
4. Handle missing values appropriately
5. Return corrected code only, no explanations

CORRECTED CODE:
```python
"""

            # Use LLM to generate correction
            if hasattr(self.llm_client, "ainvoke"):
                from langchain_core.prompts import ChatPromptTemplate

                prompt = ChatPromptTemplate.from_messages(
                    [("human", correction_prompt)]
                )

                response = await self.llm_client.ainvoke(await prompt.ainvoke({}))

                corrected_code = response.content
            else:
                # Fallback for older LangChain versions
                corrected_code = await self.llm_client.agenerate([correction_prompt])
                corrected_code = corrected_code.generations[0][0].text

            # Extract code from response
            corrected_code = self._extract_code_from_response(corrected_code)

            self.logger.debug(
                "LLM generated corrected code",
                feature_name=feature_name,
                attempt=attempt,
                original_length=len(original_code),
                corrected_length=len(corrected_code),
            )

            return corrected_code

        except Exception as e:
            self.logger.warning(
                "LLM code correction failed",
                feature_name=feature_name,
                error=str(e),
            )
            # Fallback to basic corrections
            return self._apply_basic_corrections(
                original_code, error_msg, error_type, df
            )

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response.

        Args:
            response: LLM response text.

        Returns:
            Extracted Python code.
        """
        # Remove markdown formatting
        code = response.strip()

        # Extract code from markdown blocks
        if "```python" in code:
            start = code.find("```python") + 9
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()
        elif "```" in code:
            start = code.find("```") + 3
            end = code.find("```", start)
            if end != -1:
                code = code[start:end].strip()

        return code

    def _apply_basic_corrections(
        self, original_code: str, error_msg: str, error_type: str, df: pd.DataFrame
    ) -> str:
        """Apply basic rule-based corrections to failing code.

        Args:
            original_code: Original failing code.
            error_msg: Error message.
            error_type: Error type.
            df: DataFrame.

        Returns:
            Corrected code.
        """
        code = original_code

        # Handle KeyError for missing columns
        if error_type == "KeyError":
            missing_col = error_msg.strip("'\"")
            available_cols = list(df.columns)

            # Try to find similar column names
            similar_cols = [
                col for col in available_cols if missing_col.lower() in col.lower()
            ]

            if similar_cols:
                replacement_col = similar_cols[0]
                code = code.replace(f"'{missing_col}'", f"'{replacement_col}'")
                code = code.replace(f'"{missing_col}"', f'"{replacement_col}"')
                code = code.replace(f"['{missing_col}']", f"['{replacement_col}']")
                code = code.replace(f'["{missing_col}"]', f'["{replacement_col}"]')
            else:
                # Replace with a safe alternative
                if "rating" in available_cols:
                    code = code.replace(f"'{missing_col}'", "'rating'")
                    code = code.replace(f'"{missing_col}"', '"rating"')

        # Handle division by zero
        if "division by zero" in error_msg.lower():
            code = code.replace("/ x", "/ x.clip(lower=1)")
            code = code.replace("/x", "/x.clip(lower=1)")

        # Handle empty aggregations
        if "empty" in error_msg.lower():
            if ".fillna(" not in code:
                code = code.rstrip() + ".fillna(0)"

        # Ensure result assignment
        if "result =" not in code and "result=" not in code:
            code = f"result = {code}"

        return code

    async def _execute_llm_feature(
        self,
        feature: FeatureDefinition,
        data: Dict[str, Any],
        data_context: DataContext,
    ) -> List[FeatureValue]:
        """Execute LLM-based feature with structured output.

        Args:
            feature: Feature definition.
            data: Target data split.
            data_context: Full data context.

        Returns:
            List of feature values.
        """
        if not feature.llm_prompt or not feature.text_columns:
            raise ValueError("LLM-based feature missing prompt or text columns")

        if not self.llm_chain:
            raise ValueError("LLM client not initialized")

        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        feature_values = []

        # Process each user's text data
        for user_id in df["user_id"].unique():
            user_data = df[df["user_id"] == user_id]

            # Combine text from specified columns
            text_content = self._extract_text_content(user_data, feature.text_columns)

            if not text_content.strip():
                # No text content for this user
                feature_values.append(
                    FeatureValue(
                        user_id=str(user_id),
                        feature_name=feature.name,
                        value=None,
                        confidence=0.0,
                    )
                )
                continue

            # Make structured LLM call
            try:
                # Get format instructions
                parser = JsonOutputParser(pydantic_object=LLMFeatureExtractionResponse)
                format_instructions = parser.get_format_instructions()

                # Invoke structured LLM chain
                response = await self.llm_chain(
                    {
                        "feature_prompt": feature.llm_prompt,
                        "text_content": text_content,
                        "format_instructions": format_instructions,
                    }
                )

                self.logger.debug(
                    "Received structured LLM feature response",
                    user_id=user_id,
                    feature_name=feature.name,
                    feature_value=response.get("feature_value"),
                    confidence=response.get("confidence"),
                )

                feature_values.append(
                    FeatureValue(
                        user_id=str(user_id),
                        feature_name=feature.name,
                        value=response["feature_value"],
                        confidence=response["confidence"],
                    )
                )

            except Exception as e:
                self.logger.warning(
                    "Structured LLM call failed for user",
                    user_id=user_id,
                    feature_name=feature.name,
                    error=str(e),
                )
                feature_values.append(
                    FeatureValue(
                        user_id=str(user_id),
                        feature_name=feature.name,
                        value=None,
                        confidence=0.0,
                    )
                )

        return feature_values

    async def _execute_hybrid_feature(
        self,
        feature: FeatureDefinition,
        data: Dict[str, Any],
        data_context: DataContext,
    ) -> List[FeatureValue]:
        """Execute hybrid feature (code + LLM) with structured output.

        Args:
            feature: Feature definition.
            data: Target data split.
            data_context: Full data context.

        Returns:
            List of feature values.
        """
        if not feature.llm_prompt or not feature.postprocessing_code:
            raise ValueError("Hybrid feature missing LLM prompt or postprocessing code")

        # First, execute LLM component
        llm_values = []
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        # Process each user with LLM
        for user_id in df["user_id"].unique():
            user_data = df[df["user_id"] == user_id]
            text_content = self._extract_text_content(
                user_data, feature.text_columns or ["title", "authors"]
            )

            if text_content.strip() and self.llm_chain:
                try:
                    # Get format instructions
                    parser = JsonOutputParser(
                        pydantic_object=LLMFeatureExtractionResponse
                    )
                    format_instructions = parser.get_format_instructions()

                    # Invoke structured LLM chain
                    response = await self.llm_chain(
                        {
                            "feature_prompt": feature.llm_prompt,
                            "text_content": text_content,
                            "format_instructions": format_instructions,
                        }
                    )

                    llm_values.append(
                        {
                            "user_id": user_id,
                            "llm_value": response["feature_value"],
                            "confidence": response["confidence"],
                        }
                    )

                except Exception as e:
                    self.logger.warning(
                        "LLM processing failed in hybrid feature",
                        user_id=user_id,
                        error=str(e),
                    )
                    llm_values.append(
                        {"user_id": user_id, "llm_value": 0.0, "confidence": 0.0}
                    )
            else:
                llm_values.append(
                    {"user_id": user_id, "llm_value": 0.0, "confidence": 0.0}
                )

        # Create DataFrame with LLM results
        llm_df = pd.DataFrame(llm_values)

        # Execute postprocessing code
        feature_values = []
        try:
            exec_context = {
                "df": df,
                "llm_df": llm_df,
                "pd": pd,
                "result": None,
            }

            exec(feature.postprocessing_code, exec_context)

            result = exec_context.get("result")
            if result is None:
                raise ValueError(
                    "Hybrid feature postprocessing code did not set 'result' variable"
                )

            # Convert result to feature values
            if isinstance(result, pd.Series):
                for user_id, value in result.items():
                    # Find corresponding confidence from LLM results
                    user_llm = llm_df[llm_df["user_id"] == user_id]
                    confidence = (
                        user_llm["confidence"].iloc[0] if not user_llm.empty else 1.0
                    )

                    feature_values.append(
                        FeatureValue(
                            user_id=str(user_id),
                            feature_name=feature.name,
                            value=float(value) if pd.notna(value) else None,
                            confidence=confidence,
                        )
                    )

        except Exception as e:
            self.logger.error(
                "Hybrid feature postprocessing failed",
                feature_name=feature.name,
                error=str(e),
            )
            raise

        return feature_values

    def _extract_text_content(
        self, user_data: pd.DataFrame, text_columns: List[str]
    ) -> str:
        """Extract and combine text content from specified columns.

        Args:
            user_data: User's data rows.
            text_columns: Columns containing text.

        Returns:
            Combined text content.
        """
        text_parts = []

        for column in text_columns:
            if column in user_data.columns:
                # Get all non-null text values for this user
                text_values = user_data[column].dropna().astype(str).tolist()
                if text_values:
                    text_parts.extend(text_values)

        return " ".join(text_parts)

    async def cleanup(self) -> None:
        """Cleanup feature executor resources."""
        if self.llm_client:
            # Close LLM client if needed
            pass
        self._execution_cache.clear()
        self._full_data_cache.clear()

    async def execute_feature_set(
        self,
        features: List[FeatureDefinition],
        data_context: DataContext,
        target_split: str = "train",
        max_records: Optional[int] = None,
    ) -> Dict[str, List[FeatureValue]]:
        """Execute multiple features in parallel.

        Args:
            features: List of feature definitions.
            data_context: Data context.
            target_split: Target data split.
            max_records: Maximum number of records to process (None for all).

        Returns:
            Dictionary mapping feature names to their values.
        """
        tasks = []
        for feature in features:
            task = self.execute_feature(
                feature, data_context, target_split, max_records
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        feature_results = {}
        for i, (feature, result) in enumerate(zip(features, results)):
            if isinstance(result, Exception):
                self.logger.error(
                    "Feature execution failed",
                    feature_name=feature.name,
                    error=str(result),
                )
                feature_results[feature.name] = []
            else:
                feature_results[feature.name] = result

        return feature_results

    async def _execute_code_feature(
        self,
        feature: FeatureDefinition,
        data: Dict[str, Any],
        data_context: DataContext,
    ) -> List[FeatureValue]:
        """Backward-compatible wrapper for code feature execution.

        Args:
            feature: Feature definition.
            data: Target data split.
            data_context: Full data context.

        Returns:
            List of feature values.
        """
        # Use the new self-correcting method with full 3 attempts
        return await self._execute_code_feature_with_correction(
            feature, data, data_context, max_attempts=3
        )

    def clear_cache(
        self, feature_name: Optional[str] = None, split: Optional[str] = None
    ):
        """Clear execution cache.

        Args:
            feature_name: If provided, only clear cache for this feature
            split: If provided, only clear cache for this split
        """
        if feature_name and split:
            cache_key = f"{feature_name}_{split}_full"
            if cache_key in self._execution_cache:
                del self._execution_cache[cache_key]
                self.logger.info(f"Cleared cache for {feature_name} on {split}")
        elif feature_name:
            # Clear all splits for this feature
            keys_to_remove = [
                k
                for k in self._execution_cache.keys()
                if k.startswith(f"{feature_name}_")
            ]
            for key in keys_to_remove:
                del self._execution_cache[key]
            self.logger.info(f"Cleared all cache entries for {feature_name}")
        elif split:
            # Clear all features for this split
            keys_to_remove = [
                k for k in self._execution_cache.keys() if k.endswith(f"_{split}_full")
            ]
            for key in keys_to_remove:
                del self._execution_cache[key]
            self.logger.info(f"Cleared all cache entries for {split} split")
        else:
            # Clear everything
            self._execution_cache.clear()
            self._full_data_cache.clear()
            self.logger.info("Cleared all execution caches")
