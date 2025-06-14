"""
Hypothesis & Strategy Team agents for refining insights into concrete hypotheses.
This team is responsible for strategic analysis and hypothesis generation.
"""

from typing import Dict

import autogen


def get_hypothesis_agents(
    llm_config: Dict, system_prompt: str
) -> Dict[str, autogen.ConversableAgent]:
    """
    Initializes and returns the agents for the hypothesis and strategy loop.
    All agents will share the same system prompt to ensure a shared context.
    Individual roles are defined in separate, more concise prompts.
    """
    # Define agent roles
    agent_roles = {
        "HypothesisAgent": "Your role is to propose initial hypotheses based on the insight report.",
        "StrategistAgent": "Your role is to critique hypotheses for business and scientific value.",
        "EngineerAgent": "Your role is to critique hypotheses for technical feasibility and finalize the list.",
    }

    # Create agents with a shared system prompt and specific roles
    agents = {
        name: autogen.AssistantAgent(
            name=name,
            system_message=system_prompt + "\n\n**YOUR SPECIFIC ROLE:**\n" + role,
            llm_config=llm_config,
        )
        for name, role in agent_roles.items()
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
