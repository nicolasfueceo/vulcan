"""
Orchestration module for the insight discovery team.
"""

from typing import Dict
import json 
import autogen
from loguru import logger

from src.agents.discovery_team.insight_discovery_agents import (
    get_insight_discovery_agents,
)
from src.utils.prompt_utils import load_prompt
from src.utils.run_utils import get_run_dir


def run_insight_discovery_chat(llm_config: Dict) -> Dict:
    """
    Runs the insight discovery team group chat to find patterns in the data.

    Args:
        llm_config: LLM configuration for the agents

    Returns:
        Dictionary containing the insights and view descriptions
    """
    logger.info("Starting insight discovery team chat...")

    # Initialize agents
    agents = get_insight_discovery_agents(llm_config)
    user_proxy = agents.pop("user_proxy")

    # Load chat initiator prompt
    initiator_prompt = load_prompt(
        "globals/discovery_team_chat_initiator.j2",
        view_descriptions="No views created yet.",
    )

    # Create group chat
    group_chat = autogen.GroupChat(
        agents=[user_proxy] + list(agents.values()),
        messages=[],
        max_round=50,
        allow_repeat_speaker=True,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Start the chat
    user_proxy.initiate_chat(
        manager,
        message=initiator_prompt,
    )

    # Extract results from the chat
    results = {
        "insights": _extract_insights(group_chat.messages),
        "view_descriptions": _extract_view_descriptions(group_chat.messages),
    }

    # Save results
    run_dir = get_run_dir()
    results_path = run_dir / "insight_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Insight discovery chat completed. Results saved to {results_path}")
    return results


def _extract_insights(messages: list) -> list:
    """Extract insights from the chat messages."""
    insights = []
    for msg in messages:
        if "add_insight_to_report" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    insight = json.loads(content[start_idx:end_idx])
                    insights.append(insight)
            except Exception as e:
                logger.error(f"Error parsing insight: {e}")
    return insights


def _extract_view_descriptions(messages: list) -> dict:
    """Extract SQL view descriptions from the chat messages."""
    views = {}
    for msg in messages:
        if "create_analysis_view" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    view = json.loads(content[start_idx:end_idx])
                    views[view["name"]] = view["description"]
            except Exception as e:
                logger.error(f"Error parsing view description: {e}")
    return views
