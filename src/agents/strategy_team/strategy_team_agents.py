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
    agent_prompts = {
        "StrategistAgent": load_prompt("agents/strategy_team/strategist_agent.j2", db_schema=db_schema),
        "EngineerAgent": load_prompt("agents/strategy_team/engineer_agent.j2", db_schema=db_schema),
        "FeatureEngineer": load_prompt("agents/feature_realization.j2"),
    }

    # Define the schema for the save_candidate_features tool
    save_candidate_features_tool_schema = {
        "type": "function",
        "function": {
            "name": "save_candidate_features",
            "description": "Saves a list of candidate feature specifications. Each feature spec should be a dictionary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_features_data": {
                        "type": "array",
                        "description": "A list of candidate features, where each feature is a dictionary defining its 'name', 'description', 'dependencies', and 'parameters'.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Unique, snake_case name."},
                                "description": {"type": "string", "description": "Explanation of the feature."},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of source column names."
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Dictionary of tunable parameters (name: {type, description}). Empty if no params."
                                }
                            },
                            "required": ["name", "description", "dependencies", "parameters"]
                        }
                    }
                },
                "required": ["candidate_features_data"]
            }
        }
    }

    # Create agents with loaded prompts
    agents = {}
    for name, prompt in agent_prompts.items():
        current_llm_config = llm_config.copy()
        if name == "FeatureEngineer":
            current_llm_config["tools"] = [save_candidate_features_tool_schema]
        
        agents[name] = autogen.AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=current_llm_config,
        )


    # Add user proxy for code execution with faster termination condition
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_Strategy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,  # Increased to allow more iterations within a single chat
        is_termination_msg=lambda x: "FINAL_FEATURES" in x.get("content", ""),  # Updated termination message
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    # Register save_candidate_features tool for user_proxy
    # Assume session_state will be passed in or made available at runtime
    # This is a placeholder; actual registration should occur where session_state is available
    # user_proxy.register_tool(get_save_candidate_features_tool(session_state))

    agents["user_proxy"] = user_proxy
    return agents
