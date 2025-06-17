"""
Strategy Team agents for feature engineering and optimization.
This team is responsible for turning hypotheses into concrete features and optimizing them.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt
from src.utils.session_state import get_run_dir


def get_strategy_team_agents(
    llm_config: Dict,
    db_schema: str = "",
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the streamlined strategy team group chat.
    Uses Jinja2 templates from src/prompts/agents/strategy_team/
    
    Args:
        llm_config: Configuration for the language model
        db_schema: Current database schema string to provide to agents
    """

    # Load agent prompts from Jinja2 templates - removed HypothesisAgent and
    # replaced FeatureIdeator & FeatureRealizer with a single FeatureEngineer
    # Pass the database schema to each agent's prompt template
    project_context = (
        "You are working on a book recommender system. "
        "The downstream task is to engineer and realize features that improve the accuracy and diversity of book recommendations. "
        "All features and code should be designed for this context."
    )
    agent_prompts = {
        "StrategistAgent": load_prompt(
            "agents/strategy_team/strategist_agent.j2",
            db_schema=db_schema,
            project_context=project_context,
        ),
    }

    agents = {}
    for name, prompt in agent_prompts.items():
        current_llm_config = llm_config.copy()
        agents[name] = autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=current_llm_config,
        )

    return agents
