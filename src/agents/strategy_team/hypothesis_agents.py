"""
Hypothesis & Strategy Team agents for refining insights into concrete hypotheses.
This team is responsible for strategic analysis and hypothesis generation.
"""

from typing import Dict

import autogen

from src.utils.prompt_utils import load_prompt


def get_hypothesis_agents(
    llm_config: Dict, insights_report: str
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the hypothesis and strategy loop.
    Uses Jinja2 templates from src/prompts/agents/strategy_team/
    """

    # Load agent prompts from Jinja2 templates
    hypothesis_prompt = load_prompt(
        "agents/strategy_team/hypothesis_agent.j2", insights_report=insights_report
    )
    strategist_prompt = load_prompt("agents/strategy_team/strategist_agent.j2")
    engineer_prompt = load_prompt("agents/strategy_team/engineer_agent.j2")

    agent_defs = [
        ("HypothesisAgent", hypothesis_prompt),
        ("StrategistAgent", strategist_prompt),
        ("EngineerAgent", engineer_prompt),
    ]

    # Create agents with loaded prompts
    agents = {
        name: autogen.AssistantAgent(
            name=name, system_message=prompt, llm_config=llm_config
        )
        for name, prompt in agent_defs
    }

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Hypothesis",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "SUCCESS" in x.get("content", ""),
        code_execution_config={"use_docker": False},
    )

    agents["user_proxy"] = user_proxy
    return agents
