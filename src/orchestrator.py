import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import autogen
from loguru import logger

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from src.agents.discovery_team.insight_discovery_agents import (
    get_insight_discovery_agents,
)
from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.agents.strategy_team.reflection_agent import ReflectionAgent
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.config.log_config import setup_logging
from src.utils.run_utils import config_list_from_json, get_run_dir, init_run
from src.utils.session_state import SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    create_analysis_view,
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    get_table_sample,
    run_sql_query,
    vision_tool,
)


# --- Helper Functions for SmartGroupChatManager ---

def should_continue_exploration(
    session_state: SessionState, round_count: int
) -> bool:
    """Determines if exploration should continue based on insights and coverage."""
    insights = session_state.insights
    if not insights:
        return True  # Must find at least one insight

    high_quality_insights = [i for i in insights if i.quality_score >= 8]
    if len(high_quality_insights) >= 5:
        logger.info("Termination condition: Found 5+ high-quality insights.")
        return False

    if len(insights) >= 15:
        logger.info("Termination condition: Found 15+ total insights.")
        return False

    if round_count > 50:
        last_insight_round = max((i.metadata.get("round_added", 0) for i in insights), default=0)
        if round_count - last_insight_round > 20:
            logger.info("Termination condition: No new insights in the last 20 rounds.")
            return False

    if len(insights) > 5:
        tables_in_insights = {t for i in insights for t in i.tables_used}
        all_tables = set(session_state.get_all_table_names())
        coverage = len(tables_in_insights) / len(all_tables) if all_tables else 0
        if coverage < 0.3:
            logger.info(f"Continuation condition: Low table coverage ({coverage:.1%}). Encouraging more exploration.")
            return True
        if coverage > 0.7 and len(insights) >= 8:
            logger.info(f"Termination condition: High table coverage ({coverage:.1%}) with sufficient insights.")
            return False

    return True

def get_progress_prompt(session_state: SessionState, round_count: int) -> Optional[str]:
    """Generate a progress prompt to guide agents when they seem stuck."""
    insights = session_state.insights
    if not insights:
        return "It's been a while and no insights have been reported. As a reminder, your goal is to find interesting patterns. Please review the schema and propose a query."

    tables_in_insights = {t for i in insights for t in i.tables_used}
    all_tables = set(session_state.get_all_table_names())
    unexplored_tables = all_tables - tables_in_insights

    if round_count > 20 and unexplored_tables:
        return f"Great work so far. We've analyzed {len(tables_in_insights)} tables, but these remain unexplored: {', '.join(list(unexplored_tables)[:3])}. Consider formulating a hypothesis involving one of these."

    low_detail_insights = [i for i in insights if len(i.description) < 100]
    if low_detail_insights:
        return f"The insight '{low_detail_insights[0].title}' is a bit brief. Can the DataScientist elaborate on its significance or provide more supporting evidence?"

    return None

def _fallback_compression(messages: List[Dict], keep_recent: int = 20) -> List[Dict]:
    """Fallback keyword-based compression if LLM compression fails."""
    logger.warning("Executing fallback context compression.")
    if len(messages) <= keep_recent:
        return messages

    compressed_messages = []
    keywords = ["insight", "hypothesis", "important", "significant", "surprising"]
    for msg in messages[:-keep_recent]:
        if any(keyword in msg.get("content", "").lower() for keyword in keywords):
            new_content = f"(Summarized): {msg['content'][:200]}..."
            compressed_messages.append({**msg, "content": new_content})

    return compressed_messages + messages[-keep_recent:]

def compress_conversation_context(messages: List[Dict], keep_recent: int = 20) -> List[Dict]:
    """Intelligently compress conversation context using LLM summarization."""
    if len(messages) <= keep_recent:
        return messages

    logger.info(f"Compressing conversation context, keeping last {keep_recent} messages.")
    try:
        config_list = config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": {"gpt-4-turbo-preview"}})
        if not config_list:
            raise ValueError("No config found for summarization model.")

        summarizer_llm_config = {"config_list": config_list, "cache_seed": None, "temperature": 0.2}
        summarizer_client = autogen.AssistantAgent("summarizer", llm_config=summarizer_llm_config)

        conversation_to_summarize = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages[:-keep_recent]])
        prompt = f"Please summarize the key findings, decisions, and unresolved questions from the following conversation history. Be concise, but do not lose critical information. The summary will be used as context for an ongoing AI agent discussion.\n\n---\n{conversation_to_summarize}\n---"

        response = summarizer_client.generate_reply(messages=[{"role": "user", "content": prompt}])
        summary_message = {"role": "system", "content": f"## Conversation Summary ##\n{response}"}
        return [summary_message] + messages[-keep_recent:]
    except Exception as e:
        logger.error(f"LLM-based context compression failed: {e}")
        return _fallback_compression(messages, keep_recent)


# --- Enhanced Conversation Manager ---

class SmartGroupChatManager(autogen.GroupChatManager):
    def run_chat(self, messages, sender, config):
        """Override the main chat runner to add smart features."""
        session_state = config.get("session_state")
        if not session_state:
            logger.error("SessionState not found in config for SmartGroupChatManager.")
            return super().run_chat(messages, sender, config)

        self.round_count += 1

        if self.round_count % 10 == 0:
            insights_count = len(session_state.insights)
            logger.info("Exploration progress: Round {}, {} insights captured", self.round_count, insights_count)

        if self.round_count % 25 == 0:
            try:
                self.groupchat.messages = compress_conversation_context(self.groupchat.messages)
                logger.info("Applied LLM context compression at round {}", self.round_count)
            except Exception as e:
                logger.warning("Context compression failed: {}", e)

        if self.round_count > 15 and not should_continue_exploration(session_state, self.round_count):
            logger.info("Exploration criteria met, terminating conversation")
            self.groupchat.messages.append({"role": "assistant", "content": "TERMINATE", "name": "SystemCoordinator"})
            return True, None

        if self.round_count > 5 and self.round_count % 15 == 0:
            if progress_prompt := get_progress_prompt(session_state, self.round_count):
                logger.info("Adding progress guidance at round {}", self.round_count)
                self.groupchat.messages.append({"role": "user", "content": progress_prompt, "name": "SystemCoordinator"})

        return super().run_chat(messages, sender, config)


# --- Orchestration Loops ---

def run_discovery_loop(session_state: SessionState) -> str:
    """Orchestrates the Insight Discovery Team to find patterns in the data."""
    logger.info("--- Running Insight Discovery Loop ---")
    config_list = config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": {"gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-32k"}})
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key: [c.update({"api_key": api_key}) for c in config_list]

    if not config_list or not all(c.get("api_key") or c.get("base_url") for c in config_list):
        raise ValueError("No valid LLM configurations found. Check OAI_CONFIG_LIST or OPENAI_API_KEY.")

    llm_config = {"config_list": config_list, "cache_seed": None}
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").strip(),
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    assistant_agents = get_insight_discovery_agents(llm_config)
    for agent in assistant_agents.values():
        autogen.register_function(run_sql_query, caller=agent, executor=user_proxy, name="run_sql_query", description="Run a SQL query.")
        autogen.register_function(get_table_sample, caller=agent, executor=user_proxy, name="get_table_sample", description="Get a sample of rows from a table.")
        autogen.register_function(create_analysis_view, caller=agent, executor=user_proxy, name="create_analysis_view", description="Create a temporary SQL view.")
        autogen.register_function(vision_tool, caller=agent, executor=user_proxy, name="vision_tool", description="Analyze an image.")
        autogen.register_function(get_add_insight_tool(session_state), caller=agent, executor=user_proxy, name="add_insight_to_report", description="Saves insights to the report.")

    group_chat = autogen.GroupChat(agents=[user_proxy] + list(assistant_agents.values()), messages=[], max_round=500, allow_repeat_speaker=True)
    manager = SmartGroupChatManager(groupchat=group_chat, llm_config=llm_config)

    logger.info("Closing database connection for agent execution...")
    session_state.close_connection()
    try:
        initial_message = "Team, let's begin our analysis. The database schema and our mission are in your system prompts. Please start by planning your first exploration step."
        user_proxy.initiate_chat(manager, message=initial_message, session_state=session_state)

        logger.info("Exploration completed after {} rounds with {} insights", manager.round_count, len(session_state.insights))
        logger.info("--- FINAL INSIGHTS SUMMARY ---")
        logger.info(session_state.get_final_insight_report())

        run_dir = get_run_dir()
        views_file = run_dir / "generated_views.json"
        if views_file.exists():
            with open(views_file, "r", encoding="utf-8") as f:
                views_data = json.load(f)
            logger.info("Total views created: {}", len(views_data.get("views", [])))
        else:
            logger.info("Total views created: 0")
    finally:
        logger.info("Reopening database connection...")
        session_state.reconnect()

    logger.info("--- Insight Discovery Loop Complete ---")
    return session_state.get_final_insight_report()

def run_strategy_loop(session_state: SessionState) -> Optional[Dict[str, Any]]:
    """Orchestrates the Strategy Team to refine insights and generate features."""
    logger.info("--- Running Strategy Loop ---")
    insights_report = session_state.get_final_insight_report()
    if not session_state.insights:
        logger.warning("No insights found, skipping strategy loop.")
        return None

    run_dir = get_run_dir()
    views_file = run_dir / "generated_views.json"
    view_descriptions = "No views created yet."
    if views_file.exists():
        with open(views_file, "r", encoding="utf-8") as f:
            views_data = json.load(f)
        if views_data.get("views"):
            view_descriptions = json.dumps(views_data, indent=2)

    config_list = config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": {"gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-32k"}})
    if api_key := os.getenv("OPENAI_API_KEY"): [c.update({"api_key": api_key}) for c in config_list]
    llm_config = {"config_list": config_list, "cache_seed": None}

    agents = get_strategy_team_agents(llm_config)
    user_proxy = agents.pop("user_proxy")
    user_proxy.register_function(function_map={"run_sql_query": run_sql_query, "finalize_hypotheses": get_finalize_hypotheses_tool(session_state)})

    group_chat = autogen.GroupChat(agents=[user_proxy] + list(agents.values()), messages=[], max_round=50, allow_repeat_speaker=True)
    manager = SmartGroupChatManager(groupchat=group_chat, llm_config=llm_config)

    logger.info("Closing database connection for strategy agent execution...")
    session_state.close_connection()
    try:
        user_proxy.initiate_chat(
            manager,
            message=f'Welcome strategists. Your task is to:\n1. Refine insights into hypotheses\n2. Convert hypotheses to features\n3. Implement and optimize features\n4. Reflect on results.\nUse `run_sql_query` to verify findings. Call `finalize_hypotheses` when done.\n\n--- INSIGHTS REPORT ---\n{insights_report}\n\n--- AVAILABLE VIEWS ---\n{view_descriptions}',
            session_state=session_state,
        )

        logger.info("--- Running Feature Realization ---")
        FeatureRealizationAgent(llm_config=llm_config, session_state=session_state).run()
        logger.info("--- Feature Realization Complete ---")

        reflection_results = ReflectionAgent(llm_config).run(session_state)
    finally:
        session_state.reconnect()

    logger.info("--- Strategy Loop Complete ---")
    return reflection_results


def main() -> str:
    """Main function to run the VULCAN agent orchestration."""
    run_id, run_dir = init_run()
    logger.info(f"Starting VULCAN Run ID: {run_id}")
    setup_logging()
    session_state = SessionState(run_dir)

    discovery_report = "Discovery loop did not generate a report."
    strategy_report = "Strategy loop was not run or did not generate a report."

    try:
        should_continue = True
        while should_continue:
            discovery_report = run_discovery_loop(session_state)
            reflection_results = run_strategy_loop(session_state)

            if reflection_results:
                strategy_report = json.dumps(reflection_results, indent=2)

            if not session_state.get_final_hypotheses():
                logger.info("No hypotheses generated. Ending pipeline.")
                break

            should_continue = reflection_results.get("should_continue", False) if reflection_results else False
            if not should_continue:
                logger.info("Pipeline completion criteria met. Ending pipeline.")

    except Exception as e:
        logger.error(f"An error occurred during orchestration: {e}", exc_info=True)
        strategy_report += f"\n\n--- ORCHESTRATION FAILED ---\n{e}"
    finally:
        session_state.close_connection()
        cleanup_analysis_views(Path(session_state.run_dir))
        logger.info("View cleanup process initiated.")
        logger.info("Run finished. Session state saved.")

    final_report = (
        f"# VULCAN Run Complete: {run_id}\n\n"
        f"## Discovery Loop Report\n{discovery_report}\n\n"
        f"## Strategy Refinement Report\n{strategy_report}\n"
    )
    logger.info("VULCAN has completed its run.")
    return final_report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"VULCAN run failed: {e}", exc_info=True)
        sys.exit(1)
