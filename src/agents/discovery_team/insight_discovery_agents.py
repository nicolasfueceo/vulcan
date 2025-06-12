"""
Insight Discovery Team agents for exploratory data analysis.
This team is responsible for discovering patterns and insights in the data.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


def get_insight_discovery_agents(
    llm_config: Dict,
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the insight discovery loop.
    Uses Jinja2 templates from src/prompts/agents/discovery_team/
    """

    # Load agent prompts from Jinja2 templates
    agent_prompts = {
        "DataRepresenter": load_prompt("agents/discovery_team/data_representer.j2"),
        "QuantitativeAnalyst": load_prompt(
            "agents/discovery_team/quantitative_analyst.j2"
        ),
        "PatternSeeker": load_prompt("agents/discovery_team/pattern_seeker.j2"),
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

    return agents
