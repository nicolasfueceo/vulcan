from typing import List

from autogen import Agent, AssistantAgent, UserProxyAgent

from ...config.settings import LLM_CONFIG
from ...core.prompt_utils import load_prompt
from ...core.tools import get_table_sample
from ...utils.session_state import SessionState


def get_insight_discovery_agents(session_state: SessionState) -> List[Agent]:
    """
    Initializes and returns the agents for the Insight Discovery team.
    """
    # Define a user proxy agent for code execution
    code_execution_agent = UserProxyAgent(
        name="CodeExecutionAgent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "content" in x
        and x["content"] is not None
        and x["content"].rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": session_state.get_run_output_dir(),
            "use_docker": False,
            "timeout": 120,
        },
        system_message="You are a code execution agent.",
    )

    # Make helper functions available to the agents in the code execution environment
    available_functions = {
        "get_table_sample": get_table_sample,
        "add_insight_to_report": session_state.add_insight_to_report,
        "vision_tool": session_state.vision_tool,
    }
    code_execution_agent.register_function(function_map=available_functions)

    # Base configuration for all analyst agents
    base_analyst_config = {
        "is_termination_msg": lambda x: "TERMINATE" in x.get("content", ""),
        "code_execution_config": False,  # No direct code execution for these agents
    }

    # Data Representer Agent
    data_representer = AssistantAgent(
        name="DataRepresenterAgent",
        llm_config=LLM_CONFIG,
        system_message=load_prompt("agents/discovery_team/data_representer"),
        **base_analyst_config,
    )

    # Pattern Seeker Agent
    pattern_seeker = AssistantAgent(
        name="PatternSeekerAgent",
        llm_config=LLM_CONFIG,
        system_message=load_prompt("agents/discovery_team/pattern_seeker"),
        **base_analyst_config,
    )

    # Quantitative Analyst Agent
    quantitative_analyst = AssistantAgent(
        name="QuantitativeAnalystAgent",
        llm_config=LLM_CONFIG,
        system_message=load_prompt("agents/discovery_team/quantitative_analyst"),
        **base_analyst_config,
    )

    return [
        code_execution_agent,
        data_representer,
        pattern_seeker,
        quantitative_analyst,
    ]
