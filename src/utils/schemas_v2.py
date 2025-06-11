from typing import Optional

from pydantic import BaseModel, Field


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


class Hypothesis(BaseModel):
    id: str = Field(
        ..., description="A unique identifier for the hypothesis, e.g., 'H-01'."
    )
    description: str = Field(
        ..., description="The full text of the hypothesis, clearly stated."
    )
    strategic_critique: str = Field(
        ...,
        description="A detailed critique from the Strategist on how this hypothesis aligns with the Core Objective.",
    )
    feasibility_critique: str = Field(
        ...,
        description="A detailed critique from the Engineer on the technical feasibility and computational cost.",
    )
