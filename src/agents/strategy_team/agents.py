from typing import List

from autogen import Agent, AssistantAgent, UserProxyAgent

from src.config.settings import LLM_CONFIG
from src.core.prompt_utils import load_prompt
from src.utils.session_state import SessionState


def get_strategy_agents(session_state: SessionState) -> List[Agent]:
    """
    Initializes and returns the agents for the strategy and feasibility team.
    """
    # Base configuration for all strategy agents
    base_strategy_config = {
        "is_termination_msg": lambda x: "TERMINATE" in x.get("content", ""),
        "code_execution_config": False,  # No direct code execution
    }

    # User Proxy Agent for code execution and tool interaction
    code_execution_agent = UserProxyAgent(
        name="CodeExecutionAgent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config={
            "work_dir": session_state.get_run_output_dir(),
            "use_docker": False,
            "timeout": 120,
        },
        system_message="You are a code execution agent.",
    )

    # Register the finalize_hypotheses function
    available_functions = {
        "finalize_hypotheses": session_state.finalize_hypotheses,
    }
    code_execution_agent.register_function(function_map=available_functions)

    # Strategist Agent
    strategist_agent = AssistantAgent(
        name="StrategistAgent",
        llm_config=LLM_CONFIG,
        system_message=load_prompt("agents/strategy_team/strategist"),
        **base_strategy_config,
    )

    # Engineer Agent
    engineer_agent = AssistantAgent(
        name="EngineerAgent",
        llm_config=LLM_CONFIG,
        system_message=load_prompt("agents/strategy_team/engineer"),
        **base_strategy_config,
    )

    return [code_execution_agent, strategist_agent, engineer_agent]
