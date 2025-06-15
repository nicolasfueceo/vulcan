import json
import os
import sys

# Ensure DB views are set up for pipeline compatibility
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import autogen
from autogen import Agent
from dotenv import load_dotenv
from loguru import logger

import scripts.setup_views
from src.agents.discovery_team.insight_discovery_agents import get_insight_discovery_agents
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.config.log_config import setup_logging
from src.core.database import get_db_schema_string
from src.utils.run_utils import config_list_from_json, get_run_dir, init_run
from src.utils.session_state import CoverageTracker, SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    create_analysis_view,
    execute_python,  # FIX: Import execute_python for agent registration
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    get_table_sample,
    run_sql_query,
    vision_tool,
    get_save_features_tool,  # Register save_features tool for agents
)

# Ensure DB views are set up for pipeline compatibility
scripts.setup_views.setup_views()

# Load environment variables from .env file at the beginning.
load_dotenv()


# --- Helper Functions for SmartGroupChatManager ---

def get_insight_context(session_state: SessionState) -> str:
    """Generate a context message based on available insights."""
    if not session_state.insights:
        return ""
        
    # Format the top insights for context
    insights = session_state.insights[:5]  # Limit to top 5 insights
    insights_text = "\n\n".join([f"**{i.title}**: {i.finding[:150]}..." for i in insights])
    
    return f"""
## Context from Discovery Team

These insights were discovered by the previous team:

{insights_text}

Please reference these insights when building your features.
"""


def should_continue_exploration(session_state: SessionState, round_count: int) -> bool:
    """Determines if exploration should continue based on insights and coverage."""
    insights = session_state.insights
    if not insights:
        logger.info("Cannot terminate: No insights found yet. Forcing continuation.")
        return True  # Never terminate with zero insights

    high_quality_insights = [
        i for i in insights if i.quality_score is not None and i.quality_score >= 8
    ]
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
            logger.info(
                f"Continuation condition: Low table coverage ({coverage:.1%}). Encouraging more exploration."
            )
            return True
        if coverage > 0.7 and len(insights) >= 8:
            logger.info(
                f"Termination condition: High table coverage ({coverage:.1%}) with sufficient insights."
            )
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

    low_detail_insights = [i for i in insights if len(i.finding) < 100]
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
        config_file_path = os.getenv("OAI_CONFIG_LIST")
        if not config_file_path:
            raise ValueError("OAI_CONFIG_LIST environment variable not set.")
        config_list_all = config_list_from_json(config_file_path)
        config_list = [config for config in config_list_all if config.get("model") == "gpt-4o"]
        if not config_list:
            raise ValueError("No config found for summarization model.")

        summarizer_llm_config = {
            "config_list": config_list,
            "cache_seed": None,
            "temperature": 0.2,
        }
        summarizer_client = autogen.AssistantAgent("summarizer", llm_config=summarizer_llm_config)

        conversation_to_summarize = "\n".join(
            [f"{m.get('role')}: {m.get('content')}" for m in messages[:-keep_recent]]
        )
        prompt = f"Please summarize the key findings, decisions, and unresolved questions from the following conversation history. Be concise, but do not lose critical information. The summary will be used as context for an ongoing AI agent discussion.\n\n---\n{conversation_to_summarize}\n---"

        response = summarizer_client.generate_reply(messages=[{"role": "user", "content": prompt}])
        summary_message = {
            "role": "system",
            "content": f"## Conversation Summary ##\n{response}",
        }
        return [summary_message] + messages[-keep_recent:]
    except ValueError as e:
        logger.error(
            f"Could not initialize LLM config. Please check your configuration. Error: {e}"
        )
        # Re-raise to be caught by main and terminate the run.
        raise
    except Exception as e:
        logger.error(f"LLM-based context compression failed: {e}")
        return _fallback_compression(messages, keep_recent)


def get_llm_config_list() -> Optional[Dict[str, Any]]:
    """
    Loads LLM configuration from the path specified in OAI_CONFIG_LIST,
    injects the API key, and returns a dictionary for autogen.

    Returns:
        A dictionary containing the 'config_list' and 'cache_seed', or None if config fails.
    """
    try:
        config_file_path = os.getenv("OAI_CONFIG_LIST")
        if not config_file_path:
            logger.error("OAI_CONFIG_LIST environment variable not set.")
            raise ValueError("OAI_CONFIG_LIST environment variable not set.")

        logger.debug(f"Loading LLM configuration from: {config_file_path}")
        config_list = config_list_from_json(file_path=config_file_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Relying on config file.")
        else:
            logger.debug("Injecting OPENAI_API_KEY into LLM config.")
            for c in config_list:
                c.update({"api_key": api_key})

        if not config_list:
            logger.error(
                "No valid LLM configurations found after loading. Check file content and path."
            )
            raise ValueError("No valid LLM configurations found.")

        logger.info(f"Successfully loaded {len(config_list)} LLM configurations.")
        return {"config_list": config_list, "cache_seed": None}

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse LLM config: {e}", exc_info=True)
        return None


# --- Enhanced Conversation Manager ---


class SmartGroupChatManager(autogen.GroupChatManager):
    """A customized GroupChatManager with context compression and progress monitoring."""

    round_count: int = 0

    def __init__(self, groupchat: autogen.GroupChat, llm_config: Dict[str, Any]):
        super().__init__(groupchat=groupchat, llm_config=llm_config)
        self.round_count = 0  # Reset round count for each new chat

    def run_chat(
        self, messages: List[Dict[str, Any]], sender: autogen.Agent, config: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any], None]:
        """Run the chat with additional tracking and feedback mechanisms."""
        self.round_count += 1
        session_state = globals().get("session_state")

        # If we're at round 1, attach insights/discovery context
        if self.round_count == 1:
            if session_state and hasattr(session_state, "insights"):
                context_message = get_insight_context(session_state)
                if context_message:
                    self.groupchat.messages.append(
                        {
                            "role": "user",
                            "content": context_message,
                            "name": "SystemCoordinator",
                        }
                    )

        # Try to compress context if it's getting too long
        if self.round_count > 10 and self.round_count % 10 == 0:
            try:
                self.groupchat.messages = compress_conversation_context(self.groupchat.messages)
                logger.info("Applied LLM context compression at round {}", self.round_count)
            except Exception as e:
                logger.warning("Context compression failed: {}", e)

        # Check if we should terminate based on discovery criteria
        if session_state and self.round_count > 15 and not should_continue_exploration(session_state, self.round_count):
            if len(session_state.insights) > 0:
                logger.info(
                    "Exploration criteria met and insights found, terminating conversation"
                )
                self.groupchat.messages.append(
                    {
                        "role": "assistant",
                        "content": "TERMINATE",
                        "name": "SystemCoordinator",
                    }
                )
            else:
                logger.info(
                    "Termination criteria met, but no insights found. Forcing continuation."
                )

        # Reset agents if potential loop detected
        if self.round_count > 0 and self.round_count % 20 == 0:
            logger.warning("Potential loop detected. Resetting agents.")
            # Reset all agents to clear their memory
            for agent in self.groupchat.agents:
                # Use getattr for safer access to reset method
                reset_method = getattr(agent, "reset", None)
                if reset_method and callable(reset_method):
                    reset_method()

        # Add progress prompts to guide agents periodically
        if session_state and self.round_count > 5 and self.round_count % 15 == 0:
            progress_guidance = get_progress_prompt(session_state, self.round_count)
            if progress_guidance:
                logger.info("Adding progress guidance at round {}", self.round_count)
                self.groupchat.messages.append(
                    {
                        "role": "user",
                        "content": progress_guidance,
                        "name": "SystemCoordinator",
                    }
                )

        # Let the parent class handle the actual chat execution
        # Pass the GroupChat object as config for correct typing
        result = super().run_chat(messages, sender, self.groupchat)  # type: ignore
        # Handle possible tuple return value from parent class
        if isinstance(result, tuple) and len(result) == 2:
            success, response = result
            if success and response:
                return response
            return None
        return result


# --- Orchestration Loops ---


def run_discovery_loop(session_state: SessionState) -> str:
    """Orchestrates the Insight Discovery Team to find patterns in the data."""
    logger.info("--- Running Insight Discovery Loop ---")
    llm_config = get_llm_config_list()
    if not llm_config:
        raise RuntimeError("Failed to get LLM configuration, cannot proceed with discovery.")

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").strip(),
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    assistant_agents = get_insight_discovery_agents(llm_config)
    analyst = assistant_agents["QuantitativeAnalyst"]
    researcher = assistant_agents["DataRepresenter"]
    critic = assistant_agents["PatternSeeker"]

    for agent in [analyst, researcher, critic]:
        autogen.register_function(
            run_sql_query,
            caller=agent,
            executor=user_proxy,
            name="run_sql_query",
            description="Run a SQL query.",
        )
        autogen.register_function(
            get_table_sample,
            caller=agent,
            executor=user_proxy,
            name="get_table_sample",
            description="Get a sample of rows from a table.",
        )
        autogen.register_function(
            create_analysis_view,
            caller=agent,
            executor=user_proxy,
            name="create_analysis_view",
            description="Create a temporary SQL view.",
        )
        autogen.register_function(
            vision_tool,
            caller=agent,
            executor=user_proxy,
            name="vision_tool",
            description="Analyze an image.",
        )
        autogen.register_function(
            get_add_insight_tool(session_state),
            caller=agent,
            executor=user_proxy,
            name="add_insight_to_report",
            description="Saves insights to the report.",
        )
        # Register execute_python for all discovery agents
        autogen.register_function(
            execute_python,
            caller=agent,
            executor=user_proxy,
            name="execute_python",
            description="Execute arbitrary Python code for analysis, stats, or plotting.",
        )

    agents: Sequence[Agent] = [user_proxy, analyst, researcher, critic]
    group_chat = autogen.GroupChat(
        agents=agents, messages=[], max_round=100, allow_repeat_speaker=False
    )
    manager = SmartGroupChatManager(groupchat=group_chat, llm_config=llm_config)

    logger.info("Closing database connection for agent execution...")
    session_state.close_connection()
    try:
        initial_message = "Team, let's begin our analysis. The database schema and our mission are in your system prompts. Please start by planning your first exploration step."
        user_proxy.initiate_chat(manager, message=initial_message, session_state=session_state)

        logger.info(
            "Exploration completed after {} rounds with {} insights",
            manager.round_count,
            len(session_state.insights),
        )
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


def run_strategy_loop(
    session_state: SessionState,
    strategy_agents_with_proxy: Dict[str, autogen.ConversableAgent],
    llm_config: Dict,
) -> Optional[Dict[str, Any]]:
    """
    Runs the streamlined strategy team loop with the following agents:
    - StrategistAgent: Validates features from a business/strategy perspective
    - EngineerAgent: Validates features from a technical perspective
    - FeatureEngineer: Designs and implements features based on pre-generated hypotheses
    - UserProxy_Strategy: Handles tool execution and stores features

    The session_state should already contain hypotheses generated by the discovery team.
    """
    logger.info("--- Running Strategy Loop ---")
    if not session_state.insights:
        logger.warning("No insights found, skipping strategy loop.")
        return None

    run_dir = get_run_dir()
    views_file = run_dir / "generated_views.json"
    if views_file.exists():
        with open(views_file, "r", encoding="utf-8") as f:
            json.load(f)

    # Extract agents from pre-initialized dictionary
    agents = {k: v for k, v in strategy_agents_with_proxy.items() if k != "user_proxy"}
    user_proxy = strategy_agents_with_proxy["user_proxy"]

    # Register strategy team tools with the user proxy
    user_proxy.register_function(
        function_map={
            "run_sql_query": run_sql_query,
            "finalize_hypotheses": get_finalize_hypotheses_tool(session_state),
        }
    )

    # Initialize group chat with all discovery agents
    discovery_agents = list(agents.values())
    # Using Sequence instead of List[Agent] for better type compatibility
    agent_sequence: Sequence[autogen.ConversableAgent] = discovery_agents + [user_proxy]
    group_chat = autogen.GroupChat(
        agents=agent_sequence,  # type: ignore # We're handling type compatibility with Sequence
        messages=[],
        max_round=25,
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config=llm_config
    )  # Using standard GroupChatManager

    logger.info("Closing database connection for strategy agent execution...")
    session_state.close_connection()

    # Format the hypotheses data
    hypotheses = session_state.get_final_hypotheses()
    hypothesis_str = (
        "\n".join([f"- {getattr(h, 'summary', str(h))}" for h in hypotheses])
        if hypotheses
        else "No hypotheses available."
    )

    # DB schema is now in agent system messages, no need to repeat it here
    initial_message = f"""
# Strategy Team Task: Turn Hypotheses into Features

## Available Hypotheses
{hypothesis_str}

## Task
Analyze these hypotheses and create features that can be used by the optimization team.
Start by having the FeatureEngineer create specifications and implementations for each feature.
The team will validate the features before finalizing them.

Ready to begin?

IMPORTANT INSTRUCTIONS:
1. There is NO HypothesisAgent in this conversation - the hypotheses are already provided above.
2. FeatureEngineer should take the lead with concrete implementation.
3. To run tools, you MUST prefix your request with '@UserProxy_Strategy please run'
4. To finalize hypotheses, use: '@UserProxy_Strategy please run finalize_hypotheses([{{"summary": "...", "rationale": "..."}}])'
5. For SQL queries, use: '@UserProxy_Strategy please run run_sql_query("SELECT * FROM table")'
6. Focus on producing production-ready code with detailed explanations.
7. Feature implementations will be automatically stored in session_state.features
8. End with FINAL_FEATURES when complete.

YOUR GOAL: Efficiently translate the pre-generated hypotheses into implemented features, with the FeatureEngineer driving the creation process while StrategistAgent and EngineerAgent provide critical feedback."""

    try:
        user_proxy.initiate_chat(manager, message=initial_message, session_state=session_state)

        # Check for realized features in session_state
        features = getattr(session_state, "features", None)

        # Attempt to get a strategy report from session_state or fallback to features/hypotheses
        if hasattr(session_state, "get_final_strategy_report"):
            report = session_state.get_final_strategy_report()
        elif features:
            report = {
                "features_count": len(features),
                "hypotheses_count": len(session_state.hypotheses),
            }
        elif hasattr(session_state, "get_final_hypotheses"):
            hypotheses = session_state.get_final_hypotheses()
            # Convert List[Hypothesis] to Dict[str, Any]
            report = {"hypotheses": [h.__dict__ for h in hypotheses] if hypotheses else []}
        else:
            report = {"message": "No strategy report, features, or hypotheses available."}
        return report
    finally:
        logger.info("Reopening database connection after strategy loop...")
        session_state.reconnect()


def main(epochs: int = 1, fast_mode_frac: float = 0.15) -> str:
    """
    Main function to run the VULCAN agent orchestration.
    Now supports epoch-based execution. Each epoch runs the full pipeline in fast_mode (subsampled data).
    After all epochs, a final full-data optimization and evaluation is performed.
    """

    run_id, run_dir = init_run()
    logger.info(f"Starting VULCAN Run ID: {run_id}")
    setup_logging()
    session_state = SessionState(run_dir)
    session_state.set_state("fast_mode_sample_frac", fast_mode_frac)

    # Get the database schema once to be reused by agents
    try:
        db_schema = get_db_schema_string()
        logger.info("Successfully retrieved database schema for agents")
    except Exception as e:
        logger.warning(f"Could not get database schema: {e}")
        db_schema = "[Error retrieving schema]"

    # Initialize LLM configuration once to reuse
    llm_config = get_llm_config_list()
    if not llm_config:
        raise RuntimeError("Failed to get LLM configuration, cannot proceed with orchestration.")

    # Initialize strategy agents once with the schema
    strategy_agents = get_strategy_team_agents(llm_config=llm_config, db_schema=db_schema)

    all_epoch_reports = []
    coverage_tracker = CoverageTracker()

    try:
        for epoch in range(epochs):
            logger.info(f"=== Starting Epoch {epoch + 1} / {epochs} (fast_mode) ===")
            session_state.set_state("fast_mode_sample_frac", fast_mode_frac)
            discovery_report = run_discovery_loop(session_state)
            logger.info(session_state.get_final_insight_report())

            # --- MANDATORY HYPOTHESIS GENERATION ---
            # --- MANDATORY HYPOTHESIS GENERATION ---
            # Note: Discovery team should handle hypothesis generation now
            # If no hypotheses are found, log the issue but don't attempt to generate them ourselves
            if session_state.insights and not session_state.get_final_hypotheses():
                logger.warning(
                    "No hypotheses found after discovery. Continuing with strategy without hypotheses."
                )

            if not session_state.get_final_hypotheses():
                logger.info("No hypotheses found, skipping strategy loop.")
                strategy_report = "Strategy loop skipped: No hypotheses were generated."
            else:
                # Pass the pre-initialized strategy agents to the strategy loop
                reflection_results = run_strategy_loop(session_state, strategy_agents, llm_config)
                if reflection_results:
                    strategy_report = json.dumps(reflection_results, indent=2)
                else:
                    strategy_report = "Strategy loop did not return results."

            summary = session_state.summarise_central_memory()
            session_state.epoch_summary = summary
            session_state.save_to_disk()
            session_state.clear_central_memory()

            coverage_tracker.update_coverage(session_state)
            all_epoch_reports.append(
                {
                    "epoch": epoch + 1,
                    "discovery_report": discovery_report,
                    "strategy_report": strategy_report,
                    "epoch_summary": summary,
                    "coverage": coverage_tracker.get_coverage(),
                }
            )
        # from src.agents.strategy_team.evaluation_agent import EvaluationAgent
        # EvaluationAgent().run(session_state)

    except Exception as e:
        logger.error(
            f"An uncaught exception occurred during orchestration: {type(e).__name__}: {e}"
        )
        logger.error(traceback.format_exc())
        strategy_report = f"Run failed with error: {e}"
    finally:
        session_state.close_connection()
        cleanup_analysis_views(Path(session_state.run_dir))
        logger.info("View cleanup process initiated.")
        logger.info("Run finished. Session state saved.")

    final_report = (
        f"# VULCAN Run Complete: {run_id}\n\n"
        f"## Epoch Reports\n{json.dumps(all_epoch_reports, indent=2)}\n\n"
        f"## Final Strategy Refinement Report\n{strategy_report}\n"
    )
    logger.info("VULCAN has completed its run.")
    print(final_report)
    return final_report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"VULCAN run failed: {e}", exc_info=True)
        sys.exit(1)
