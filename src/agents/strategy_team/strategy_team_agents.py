"""
Strategy Team agents for hypothesis generation, feature ideation, and optimization.
This team is responsible for turning insights into concrete features and optimizing them.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


def get_strategy_team_agents(
    llm_config: Dict,
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the strategy team group chat.
    Uses Jinja2 templates from src/prompts/agents/strategy_team/
    """

    # Load agent prompts from Jinja2 templates
    agent_prompts = {
        "HypothesisAgent": load_prompt("agents/strategy_team/hypothesis_agent.j2"),
        "StrategistAgent": load_prompt("agents/strategy_team/strategist_agent.j2"),
        "EngineerAgent": load_prompt("agents/strategy_team/engineer_agent.j2"),
        "FeatureIdeator": load_prompt("agents/feature_ideator.j2"),
        "FeatureRealizer": load_prompt("agents/feature_realization.j2"),
        "OptimizationAgent": load_prompt("agents/optimization_agent.j2"),
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

    # Add user proxy for code execution
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Strategy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "FINAL_HYPOTHESES" in x.get("content", ""),
        code_execution_config={"use_docker": False},
    )

    agents["user_proxy"] = user_proxy
    return agents
