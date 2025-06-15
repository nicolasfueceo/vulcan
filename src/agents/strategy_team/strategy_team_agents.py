"""
Strategy Team agents for feature engineering and optimization.
This team is responsible for turning hypotheses into concrete features and optimizing them.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


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
    agent_prompts = {
        "StrategistAgent": load_prompt("agents/strategy_team/strategist_agent.j2", db_schema=db_schema),
        "EngineerAgent": load_prompt("agents/strategy_team/engineer_agent.j2", db_schema=db_schema),
        "FeatureEngineer": load_prompt("agents/strategy_team/feature_engineer.j2", db_schema=db_schema),
    }

    # Create agents with loaded prompts
    agents = {
        name: autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=llm_config,
        )
        for name, prompt in agent_prompts.items()
    }

    # Add user proxy for code execution with faster termination condition
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Strategy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,  # Increased to allow more iterations within a single chat
        is_termination_msg=lambda x: "FINAL_FEATURES" in x.get("content", ""),  # Updated termination message
        code_execution_config={"use_docker": False},
    )

    agents["user_proxy"] = user_proxy
    return agents
