# src/utils/schemas.py
import ast
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


class Insight(BaseModel):
    title: str = Field(description="A concise, descriptive title for the insight.")
    finding: str = Field(
        description="The detailed finding or observation, explaining what was discovered."
    )
    supporting_code: Optional[str] = Field(
        None, description="The exact SQL or Python code used to generate the finding."
    )
    source_representation: str = Field(
        description="The name of the SQL View or Graph used for analysis (e.g., 'vw_user_review_summary' or 'g_user_book_bipartite')."
    )
    plot_path: Optional[str] = Field(
        None, description="The absolute path to the plot that visualizes the finding."
    )
    plot_interpretation: Optional[str] = Field(
        None,
        description="A detailed, LLM-generated analysis of what the plot shows and its implications.",
    )
    quality_score: Optional[float] = Field(
        None, description="A score from 1-10 indicating the quality of the insight."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the insight, like the round it was added.",
    )
    tables_used: List[str] = Field(
        default_factory=list,
        description="List of table names used to generate the insight.",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning chain or trace of how this insight was derived. Each entry should represent a reasoning step, tool call, or reflection.",
    )


class Hypothesis(BaseModel):
    """
    Represents a hypothesis for feature engineering, including explicit data dependencies.
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="A unique identifier for the hypothesis, e.g., a UUID."
    )
    summary: str = Field(
        ..., description="A concise, one-sentence statement of the hypothesis."
    )
    rationale: str = Field(
        ..., description="A clear explanation of why this hypothesis is useful and worth testing."
    )
    depends_on: List[str] = Field(
        ..., description="A list of fully qualified column names (e.g., 'reviews.user_id', 'books.genre') required to test this hypothesis."
    )

    @validator("rationale")
    def rationale_must_be_non_empty(cls, v):
        if not v:
            raise ValueError("Rationale cannot be empty.")
        return v


class PrioritizedHypothesis(BaseModel):
    id: str = Field(..., description="The unique identifier for the hypothesis.")
    priority: int = Field(
        ..., ge=1, le=5, description="The priority score from 1 to 5."
    )
    feasibility: int = Field(
        ..., ge=1, le=5, description="The feasibility score from 1 to 5."
    )
    notes: str = Field(..., description="A brief justification for the scores.")


class ParameterSpec(BaseModel):
    type: Literal["int", "float", "categorical"] = Field(..., description="Parameter type: int, float, or categorical.")
    low: Optional[Union[int, float]] = Field(None, description="Lower bound (for int/float)")
    high: Optional[Union[int, float]] = Field(None, description="Upper bound (for int/float)")
    step: Optional[Union[int, float]] = Field(None, description="Step size (for int)")
    log: Optional[bool] = Field(False, description="Log scale (for float)")
    choices: Optional[List[Any]] = Field(None, description="Allowed choices (for categorical)")
    default: Optional[Any] = Field(None, description="Default value")

class CandidateFeature(BaseModel):
    name: str = Field(..., description="A unique, descriptive name for the feature.")
    type: Literal["code"] = Field(..., description="The type of feature to be realized. Only 'code' is supported.")
    spec: str = Field(..., description="The core logic of the feature: a Python expression or formula.")
    depends_on: List[str] = Field(
        default_factory=list,
        description="A list of other feature names this feature depends on (for compositions).",
    )
    parameters: Dict[str, ParameterSpec] = Field(
        default_factory=dict,
        description="A dictionary specifying each tunable parameter and its constraints.",
    )
    rationale: str = Field(..., description="A detailed explanation of why this feature is useful.")

    def validate_spec(self):
        """
        Validates the 'spec' field based on the feature type.
        Raises ValueError for invalid specs.
        """
        if self.type == "code":
            try:
                ast.parse(self.spec)
            except SyntaxError as e:
                raise ValueError(
                    f"Invalid Python syntax in 'spec' for feature '{self.name}': {e}"
                ) from e
        return True


class VettedFeature(CandidateFeature):
    pass


class RealizedFeature(BaseModel):
    """
    Represents a feature that has been converted into executable code.
    """
    name: str
    code_str: str
    parameters: Dict[str, ParameterSpec]
    passed_test: bool
    type: Literal["code"]
    source_candidate: CandidateFeature
    depends_on: List[str] = []  # Data dependencies, copied from CandidateFeature
    source_hypothesis_summary: Optional[str] = None  # For traceability

    def validate_code(self) -> None:
        from loguru import logger
        logger.debug(f"Validating code for realized feature '{self.name}'...")
        try:
            tree = ast.parse(self.code_str)
        except SyntaxError as e:
            logger.error(f"Invalid Python syntax in generated code for '{self.name}': {e}")
            raise ValueError(
                f"Invalid Python syntax in generated code for '{self.name}': {e}"
            ) from e

        # Find the function definition in the AST
        func_defs = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        if not func_defs or len(func_defs) > 1:
            logger.error(f"Generated code for '{self.name}' must contain exactly one function definition.")
            raise ValueError(
                f"Generated code for '{self.name}' must contain exactly one function definition."
            )

        func_def = func_defs[0]

        # Check function name
        if func_def.name != self.name:
            logger.error(f"Function name '{func_def.name}' does not match feature name '{self.name}'.")
            raise ValueError(
                f"Function name '{func_def.name}' does not match feature name '{self.name}'."
            )

        # Check for expected parameters in the function signature
        arg_names = {arg.arg for arg in func_def.args.args}
        expected_params = set(self.parameters.keys())

        # The function should accept 'df' plus all tunable params
        if "df" not in arg_names:
            logger.error(f"Generated function for '{self.name}' must accept a 'df' argument.")
            raise ValueError(
                f"Generated function for '{self.name}' must accept a 'df' argument."
            )

        missing_params = expected_params - (arg_names - {"df"})
        if missing_params:
            logger.error(f"Missing parameters in function signature for '{self.name}': {missing_params}")
            raise ValueError(
                f"Missing parameters in function signature for '{self.name}': {missing_params}"
            )
