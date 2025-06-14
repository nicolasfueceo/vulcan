# src/utils/schemas.py
import ast
from typing import Any, Dict, List, Literal, Optional

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


class Hypothesis(BaseModel):
    id: str = Field(
        ...,
        description="A unique, sequential identifier for the hypothesis, e.g., 'H-01'.",
    )
    summary: str = Field(
        ..., description="A concise, one-sentence statement of the hypothesis."
    )
    rationale: str = Field(
        ...,
        description="A clear explanation of why this hypothesis is useful and worth testing.",
    )
    source_insight: Optional[str] = Field(
        None,
        description="The title of the insight from the report that inspired this hypothesis.",
    )
    estimated_gain: float = Field(
        ...,
        ge=0,
        le=10,
        description="A subjective estimate of the potential value (0-10) if the hypothesis is true.",
    )
    difficulty: Literal["low", "medium", "high"] = Field(
        ...,
        description="The estimated difficulty to implement and test this hypothesis.",
    )
    tags: List[str] = Field(
        ...,
        description="A list of relevant tags, e.g., ['per-user', 'temporal', 'text-based'].",
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


class CandidateFeature(BaseModel):
    name: str = Field(..., description="A unique, descriptive name for the feature.")
    type: Literal["code", "llm", "composition"] = Field(
        ..., description="The type of feature to be realized."
    )
    spec: str = Field(
        ...,
        description="The core logic of the feature: a Python expression, an LLM prompt, or a composition formula.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="A list of other feature names this feature depends on (for compositions).",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of tunable parameters for the feature.",
    )
    rationale: str = Field(
        ..., description="A detailed explanation of why this feature is useful."
    )

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
        # Add more validation for 'llm' or 'composition' types if needed
        return True


class VettedFeature(CandidateFeature):
    pass


class RealizedFeature(BaseModel):
    """
    Represents a feature that has been converted into executable code.
    """

    name: str
    code_str: str
    params: Dict[str, Any]
    passed_test: bool
    type: Literal["code", "llm", "composition"]
    source_candidate: CandidateFeature

    def validate_code(self) -> None:
        """
        Validates the generated code string for correctness.
        - Parses the code to ensure it's valid Python.
        - Checks that the function name matches the feature name.
        - Verifies that all specified params are in the function signature.
        """
        try:
            tree = ast.parse(self.code_str)
        except SyntaxError as e:
            raise ValueError(
                f"Invalid Python syntax in generated code for '{self.name}': {e}"
            ) from e

        # Find the function definition in the AST
        func_defs = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        if not func_defs or len(func_defs) > 1:
            raise ValueError(
                f"Generated code for '{self.name}' must contain exactly one function definition."
            )

        func_def = func_defs[0]

        # Check function name
        if func_def.name != self.name:
            raise ValueError(
                f"Function name '{func_def.name}' does not match feature name '{self.name}'."
            )

        # Check for expected parameters in the function signature
        arg_names = {arg.arg for arg in func_def.args.args}
        expected_params = set(self.params.keys())

        # The function should accept 'df' plus all tunable params
        if "df" not in arg_names:
            raise ValueError(
                f"Generated function for '{self.name}' must accept a 'df' argument."
            )

        missing_params = expected_params - (arg_names - {"df"})
        if missing_params:
            raise ValueError(
                f"Missing parameters in function signature for '{self.name}': {missing_params}"
            )
