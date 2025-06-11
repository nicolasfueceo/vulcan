import os
from typing import Dict, List

import autogen

from src.agents.insight_discovery_agents import (
    DATA_REPRESENTER_PROMPT,
    PATTERN_SEEKER_PROMPT,
    QUANTITATIVE_ANALYST_PROMPT,
)
from src.agents.strategy_agents import (
    ENGINEER_AGENT_PROMPT,
    HYPOTHESIS_AGENT_PROMPT,
    STRATEGIST_AGENT_PROMPT,
)
from src.utils.run_utils import init_run
from src.utils.session_state import SessionState
from src.utils.tools_v2 import (
    create_graph_from_query,
    create_sql_view,
    execute_python_with_db_connection,
    get_add_insight_tool,
    get_db_schema,
    get_finalize_hypotheses_tool,
    get_graph_metrics,
    vision_tool,
)


def get_discovery_tool_definitions() -> List[Dict]:
    """Returns the JSON schemas for the tools in the discovery loop."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_db_schema",
                "description": "Connects to the DuckDB database and returns the schema of all tables.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_sql_view",
                "description": "Creates or replaces a temporary SQL view.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "view_name": {"type": "string"},
                        "sql_query": {"type": "string"},
                    },
                    "required": ["view_name", "sql_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_graph_from_query",
                "description": "Creates a NetworkX graph from node and edge queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "graph_name": {"type": "string"},
                        "node_query": {"type": "string"},
                        "edge_query": {"type": "string"},
                    },
                    "required": ["graph_name", "node_query", "edge_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_graph_metrics",
                "description": "Loads a graph and returns key metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {"graph_name": {"type": "string"}},
                    "required": ["graph_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vision_tool",
                "description": "Analyzes an image file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["image_path", "prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_insight_to_report",
                "description": "Logs a structured insight to the session report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "finding": {"type": "string"},
                        "source_representation": {"type": "string"},
                        "supporting_code": {"type": "string"},
                        "plot_path": {"type": "string"},
                        "plot_interpretation": {"type": "string"},
                    },
                    "required": ["title", "finding", "source_representation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_with_db_connection",
                "description": "Executes a string of Python code that has access to a DuckDB connection object `conn` for custom queries and data manipulation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute. The code should use the `conn` object to interact with the database.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
    ]


def run_discovery_loop(
    session_state: SessionState, llm_config: Dict, max_turns: int = 25
):
    """Orchestrates the Insight Discovery Team."""
    llm_config_with_tools = llm_config.copy()
    llm_config_with_tools["tools"] = get_discovery_tool_definitions()

    agents = [
        autogen.AssistantAgent(
            name=name, system_message=prompt, llm_config=llm_config_with_tools
        )
        for name, prompt in [
            ("DataRepresenter", DATA_REPRESENTER_PROMPT),
            ("QuantitativeAnalyst", QUANTITATIVE_ANALYST_PROMPT),
            ("PatternSeeker", PATTERN_SEEKER_PROMPT),
        ]
    ]

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
        function_map={
            "get_db_schema": get_db_schema,
            "create_sql_view": create_sql_view,
            "create_graph_from_query": create_graph_from_query,
            "get_graph_metrics": get_graph_metrics,
            "vision_tool": vision_tool,
            "add_insight_to_report": get_add_insight_tool(session_state),
            "execute_python_with_db_connection": execute_python_with_db_connection,
        },
    )
    agents.append(user_proxy)

    group_chat = autogen.GroupChat(agents=agents, messages=[], max_round=max_turns)
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    initial_message = "Begin. Explore the database (especially the `curated_books` and `curated_reviews` tables) and generate 2 insights. Use python for analysis and plotting. Call tools for other tasks. End with TERMINATE."
    user_proxy.initiate_chat(manager, message=initial_message)


def get_strategy_tool_definition() -> List[Dict]:
    """Returns the JSON schema for the tools in the strategy loop."""
    return [
        {
            "type": "function",
            "function": {
                "name": "finalize_hypotheses",
                "description": "Finalizes and saves the vetted list of hypotheses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hypotheses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "description": {"type": "string"},
                                    "strategic_critique": {"type": "string"},
                                    "feasibility_critique": {"type": "string"},
                                },
                                "required": [
                                    "id",
                                    "description",
                                    "strategic_critique",
                                    "feasibility_critique",
                                ],
                            },
                        }
                    },
                    "required": ["hypotheses"],
                },
            },
        }
    ]


def run_strategy_loop(
    session_state: SessionState, llm_config: Dict, max_turns: int = 15
):
    """Orchestrates the Hypothesis & Strategy Team."""
    insight_report = session_state.get_final_insight_report()
    if "No insights" in insight_report:
        print("Skipping strategy loop: no insights were generated.")
        return

    llm_config_with_tool = llm_config.copy()
    llm_config_with_tool["tools"] = get_strategy_tool_definition()

    hypothesis_agent = autogen.AssistantAgent(
        name="HypothesisAgent",
        system_message=HYPOTHESIS_AGENT_PROMPT,
        llm_config=llm_config_with_tool,
    )
    agents = [
        hypothesis_agent,
        autogen.AssistantAgent(
            name="StrategistAgent",
            system_message=STRATEGIST_AGENT_PROMPT,
            llm_config=llm_config,
        ),
        autogen.AssistantAgent(
            name="EngineerAgent",
            system_message=ENGINEER_AGENT_PROMPT,
            llm_config=llm_config,
        ),
    ]

    tool_executor_proxy = autogen.UserProxyAgent(
        name="ToolExecutor",
        human_input_mode="NEVER",
        code_execution_config=False,
        function_map={
            "finalize_hypotheses": get_finalize_hypotheses_tool(session_state)
        },
    )
    agents.append(tool_executor_proxy)

    group_chat = autogen.GroupChat(agents=agents, messages=[], max_round=max_turns)
    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
        is_termination_msg=lambda x: "SUCCESS" in x.get("content", ""),
    )

    initial_message = (
        f"Welcome Team. Refine these insights into hypotheses:\n\n{insight_report}"
    )
    hypothesis_agent.initiate_chat(manager, message=initial_message)


def main():
    """Main entry point for the V2 orchestrator."""
    run_id, run_dir = init_run()
    print(f"Starting V2 orchestration for run: {run_id}\nRun directory: {run_dir}")

    session_state = SessionState()

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY must be set.")

    config_list = [
        {
            "model": "gemini-1.5-pro-latest",
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "api_type": "google",
        }
    ]
    llm_config = {"config_list": config_list, "temperature": 0.1}

    print("\n--- Running Insight Discovery Loop ---")
    run_discovery_loop(session_state, llm_config)
    print("\n--- Insight Discovery Loop Complete ---")

    print("\n--- Running Strategy Loop ---")
    run_strategy_loop(session_state, llm_config)
    print("\n--- Strategy Loop Complete ---")

    print("\n\n--- FINAL RESULTS ---")
    print("\n--- Insights ---")
    print(session_state.get_final_insight_report())
    print("\n--- Hypotheses ---")
    hypotheses = session_state.get_final_hypotheses()
    if hypotheses:
        for i, h in enumerate(hypotheses, 1):
            print(f"H{i}: {h.id} - {h.description}")
            print(f"  - Strategic Critique: {h.strategic_critique}")
            print(f"  - Feasibility Critique: {h.feasibility_critique}\n")
    else:
        print("No hypotheses were generated.")


if __name__ == "__main__":
    main()
