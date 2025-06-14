"""
Orchestration module for the strategy team group chat.
"""

import json
from typing import Dict, List

import autogen
from loguru import logger

from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.utils.prompt_utils import load_prompt
from src.utils.run_utils import get_run_dir


def run_strategy_team_chat(
    llm_config: Dict,
    insight_report: Dict,
    view_descriptions: Dict[str, str],
) -> Dict:
    """
    Runs the strategy team group chat to generate and optimize features.

    Args:
        llm_config: LLM configuration for the agents
        insight_report: Dictionary containing insights from the discovery team
        view_descriptions: Dictionary mapping view names to their descriptions

    Returns:
        Dictionary containing the final hypotheses, features, and optimization results
    """
    logger.info("Starting strategy team group chat...")

    # Initialize agents
    agents = get_strategy_team_agents(llm_config)
    user_proxy = agents.pop("user_proxy")

    # Load chat initiator prompt
    initiator_prompt = load_prompt(
        "globals/strategy_team_chat_initiator.j2",
        insight_report=json.dumps(insight_report, indent=2),
        view_descriptions=json.dumps(view_descriptions, indent=2),
    )

    # Create group chat
    groupchat = autogen.GroupChat(
        agents=list(agents.values()),
        messages=[],
        max_round=50,
        speaker_selection_method="round_robin",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat)

    # Start the chat
    user_proxy.initiate_chat(
        manager,
        message=initiator_prompt,
    )

    # Extract results from the chat
    results = {
        "hypotheses": _extract_hypotheses(groupchat.messages),
        "features": _extract_features(groupchat.messages),
        "optimization_results": _extract_optimization_results(groupchat.messages),
    }

    # Save results
    run_dir = get_run_dir()
    results_path = run_dir / "strategy_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Strategy team chat completed. Results saved to {results_path}")
    return results


def _extract_hypotheses(messages: List[Dict]) -> List[Dict]:
    """Extract hypotheses from the chat messages."""
    hypotheses = []
    for msg in messages:
        if "FINAL_HYPOTHESES" in msg.get("content", ""):
            # Parse the hypotheses from the message
            try:
                content = msg["content"]
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                if start_idx != -1 and end_idx != -1:
                    hypotheses = json.loads(content[start_idx:end_idx])
            except Exception as e:
                logger.error(f"Error parsing hypotheses: {e}")
    return hypotheses


def _extract_features(messages: List[Dict]) -> List[Dict]:
    """Extract feature specifications from the chat messages."""
    features = []
    for msg in messages:
        if "save_candidate_features" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                if start_idx != -1 and end_idx != -1:
                    features = json.loads(content[start_idx:end_idx])
            except Exception as e:
                logger.error(f"Error parsing features: {e}")
    return features


def _extract_optimization_results(messages: List[Dict]) -> Dict:
    """Extract optimization results from the chat messages."""
    results = {}
    for msg in messages:
        if "save_optimization_results" in msg.get("content", ""):
            try:
                content = msg["content"]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    results = json.loads(content[start_idx:end_idx])
            except Exception as e:
                logger.error(f"Error parsing optimization results: {e}")
    return results
