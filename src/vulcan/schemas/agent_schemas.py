from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LLMFeatureOutput(BaseModel):
    """
    Defines the structured output expected from the LLM for generating or modifying a feature.
    This schema is used by a PydanticOutputParser to format instructions for the LLM
    and parse the LLM's response.
    """

    name: str = Field(
        ...,
        description="A short, descriptive, and unique name for the new feature, using snake_case.",
    )
    description: str = Field(
        ..., description="A one-sentence description of what the feature represents."
    )
    feature_type: str = Field(
        ...,
        description="The type of the feature, must be one of: 'code_based', 'llm_based', 'hybrid'.",
    )
    code: Optional[str] = Field(
        None,
        description="The complete, syntactically correct Python code to generate the feature for 'code_based' or 'hybrid' types. It must define a variable named 'result'.",
    )
    llm_prompt: Optional[str] = Field(
        None,
        description="The detailed prompt for an LLM to generate the feature for 'llm_based' or 'hybrid' types.",
    )
    chain_of_thought_reasoning: str = Field(
        ...,
        description="Step-by-step reasoning for why this feature is useful and how the implementation was derived.",
    )


class LLMInteractionLog(BaseModel):
    """Log of a single interaction between an agent and the LLM."""

    agent_name: str = Field(..., description="Name of the agent invoking the LLM.")
    timestamp: float = Field(..., description="Timestamp of the interaction.")
    prompt_input: Dict[str, Any] = Field(
        ..., description="The full dictionary of inputs provided to the LLM prompt."
    )
    raw_response: str = Field(..., description="The raw string response from the LLM.")
    parsed_response: Optional[LLMFeatureOutput] = Field(
        None, description="The Pydantic model parsed from the LLM response."
    )
    error_message: Optional[str] = Field(
        None, description="Any error message if parsing or execution failed."
    )
