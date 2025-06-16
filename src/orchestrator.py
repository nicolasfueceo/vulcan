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
from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.config.log_config import setup_logging
from src.core.database import get_db_schema_string, refresh_global_db_schema
from src.utils.run_utils import config_list_from_json, get_run_dir, init_run
from src.utils.prompt_utils import load_prompt
from src.utils.session_state import CoverageTracker, SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    create_analysis_view,
    execute_python,
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    get_table_sample,
    run_sql_query,
    vision_tool,
    get_save_features_tool,
    get_save_candidate_features_tool,
    get_add_to_central_memory_tool,
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

        # --- ENFORCE: Do not allow termination until finalize_hypotheses has been called ---
        # If a termination signal is detected, but session_state.hypotheses is empty, inject a reminder and prevent termination
        if session_state:
            # Detect attempted termination in the last message
            last_msg = self.groupchat.messages[-1]["content"].strip() if self.groupchat.messages else ""
            attempted_termination = any(term in last_msg for term in ["FINAL_INSIGHTS", "TERMINATE"])
            hypotheses_finalized = hasattr(session_state, "hypotheses") and len(session_state.hypotheses) > 0
            if attempted_termination and not hypotheses_finalized:
                logger.warning("Termination signal received, but no hypotheses have been finalized. Blocking termination.")
                # Inject a message that forces the Hypothesizer to act.
                self.groupchat.messages.append(
                    {
                        "role": "user",
                        "name": "SystemCoordinator",
                        "content": (
                            "A termination request was detected, but no hypotheses have been finalized. **Hypothesizer, it is now your turn to act.** "
                            "Please synthesize the team's insights and call the `finalize_hypotheses` tool."
                        ),
                    }
                )
                # Prevent actual termination this round
                return super().run_chat(self.groupchat.messages, sender, self.groupchat)
        # --- END ENFORCE ---
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

        # --- VERBOSE LOGGING: Trace agent selection ---
        try:
            next_agent = self.groupchat.select_speaker(last_speaker=self.last_speaker, selector=self.selector)
            logger.info(f"[DEBUG] Next agent selected by groupchat.select_speaker(): {getattr(next_agent, 'name', next_agent)}")
        except Exception as e:
            logger.error(f"[DEBUG] Exception during agent selection: {e}")
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
        max_consecutive_auto_reply=100,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").strip(),
        code_execution_config={"work_dir": str(get_run_dir()), "use_docker": False},
    )

    assistant_agents = get_insight_discovery_agents(llm_config)
    analyst = assistant_agents["QuantitativeAnalyst"]
    researcher = assistant_agents["DataRepresenter"]
    critic = assistant_agents["PatternSeeker"]
    hypothesizer = assistant_agents["Hypothesizer"]

    # Register tools for analysis agents only
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
        autogen.register_function(
            execute_python,
            caller=agent,
            executor=user_proxy,
            name="execute_python",
            description="Execute arbitrary Python code for analysis, stats, or plotting.",
        )

    # Register finalize_hypotheses only for the Hypothesizer
    autogen.register_function(
        get_finalize_hypotheses_tool(session_state),
        caller=hypothesizer,
        executor=user_proxy,
        name="finalize_hypotheses",
        description="Finalize and submit a list of all validated hypotheses. This is the mandatory final step before the discovery loop can end.",
    )

    agents: Sequence[Agent] = [user_proxy, analyst, researcher, critic, hypothesizer]
    group_chat = autogen.GroupChat(
        agents=agents, messages=[], max_round=100, allow_repeat_speaker=True
    )
    manager = SmartGroupChatManager(groupchat=group_chat, llm_config=llm_config)

    logger.info("Closing database connection for agent execution...")
    session_state.close_connection()
    try:
        initial_message = (
            "Team, let's begin our analysis.\n"
            "- **Analysts (QuantitativeAnalyst, PatternSeeker, DataRepresenter):** Your goal is to explore the data and use the `add_insight_to_report` tool to log your findings.\n"
            "- **Hypothesizer:** Your role is to monitor the conversation. Once enough insights have been gathered, your job is to synthesize them and call the `finalize_hypotheses` tool.\n\n"
            "Let the analysis begin."
        )
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
        # --- NEW: Always reconnect after discovery loop to refresh DB schema/views ---
        logger.info("Refreshing DB connection after discovery loop to ensure new views are visible...")
        session_state.reconnect()
    finally:
        logger.info("Reopening database connection (final cleanup in discovery loop)...")
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
    - StrategistAgent: Validates features from a business/strategy perspective.
    - EngineerAgent: Validates features from a technical perspective.
    - FeatureEngineer: Designs feature contracts based on pre-generated hypotheses.
    - UserProxy_Strategy: Handles tool execution and stores features.

    The session_state should already contain hypotheses generated by the discovery team.
    """
    logger.info("--- Running Strategy Loop ---")
    if not session_state.get_final_hypotheses():
        logger.warning("No hypotheses found, skipping strategy loop.")
        return {"message": "Strategy loop skipped: No hypotheses were generated."}

    # Extract agents from the pre-initialized dictionary
    strategist = strategy_agents_with_proxy["StrategistAgent"]
    engineer = strategy_agents_with_proxy["EngineerAgent"]
    feature_engineer = strategy_agents_with_proxy["FeatureEngineer"]
    user_proxy = strategy_agents_with_proxy["user_proxy"]

    # --- Tool Registration for this specific loop ---
    # The user proxy needs access to the session_state to save features.
    # CRITICAL: Only register tools relevant to this team. `finalize_hypotheses`
    # belongs to the discovery team and was causing confusion.
    # Register the tools the agents in this group chat can use.
    # The user proxy needs access to the session_state to save features.
    user_proxy.register_function(
        function_map={
            "save_candidate_features": get_save_candidate_features_tool(session_state),
            "execute_python": execute_python,
        }
    )

    # Create the group chat with the necessary agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, strategist, engineer, feature_engineer],
        messages=[],
        max_round=1000,
        speaker_selection_method="auto",
    )

    manager = SmartGroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Format hypotheses for the initial message
    hypotheses_json = json.dumps(
        [h.model_dump() for h in session_state.get_final_hypotheses()], indent=2
    )

    # Construct the initial message to kick off the conversation.
    # This message is a direct command to the FeatureEngineer to ensure it acts first.
    initial_message = f"""You are the FeatureEngineer. Your task is to design a set of `CandidateFeature` contracts based on the following hypotheses.

**Hypotheses:**
```json
{hypotheses_json}
```

**Your Instructions:**
1.  Analyze the hypotheses.
2.  Design a list of `CandidateFeature` contracts. Each contract must be a dictionary with `name`, `description`, `dependencies`, and `parameters`.
3.  Use the `save_candidate_features` tool to submit your designs. Your response MUST be a call to this tool.

The StrategistAgent and EngineerAgent will then review your work. Begin now.
"""

    logger.info("Reopening database connection before strategy loop...")
    session_state.reconnect()

    # --- NEW: Ensure DB connection is refreshed before strategy loop ---
    logger.info("Refreshing DB connection before strategy loop to ensure all views are visible...")
    session_state.reconnect()

    report: Dict[str, Any] = {}
    try:
        # The user_proxy initiates the chat. The `message` is the first thing said.
        user_proxy.initiate_chat(manager, message=initial_message)

        # After the chat, we check the session_state for the results.
        features = getattr(session_state, "candidate_features", [])
        hypotheses = session_state.get_final_hypotheses()

        # --- Feature Realization Step ---
        if features:
            feature_realization_agent = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)
            feature_realization_agent.run()
            realized_features = getattr(session_state, "features", {})
            realized_features_list = list(realized_features.values()) if isinstance(realized_features, dict) else realized_features
        else:
            realized_features_list = []

        report = {
            "features_generated": len(features),
            "hypotheses_processed": len(hypotheses),
            "features": features,  # candidate_features are dicts, not Pydantic models
            "realized_features": [f["name"] if isinstance(f, dict) and "name" in f else getattr(f, "name", None) for f in realized_features_list],
            "hypotheses": [h.model_dump() for h in hypotheses],
        }

    except Exception as e:
        logger.error("Strategy loop failed", exc_info=True)
        report = {"error": str(e)}
    finally:
        logger.info("Reopening database connection after strategy loop...")
        session_state.reconnect()  # Reconnect again to be safe.

    return report


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

            # Refresh DB schema for prompt context ONCE per epoch
            refresh_global_db_schema()

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

        # === Optimization Step ===
        logger.info("Starting optimization step with realized features...")
        realized_features = list(session_state.features.values()) if hasattr(session_state, 'features') and session_state.features else []
        if not realized_features:
            logger.warning("No realized features found for optimization. Skipping optimization step.")
            optimization_report = "No realized features found. Optimization skipped."
        else:
            optimizer = VULCANOptimizer(session=session_state)
            try:
                optimization_result = optimizer.optimize(features=realized_features, n_trials=10, use_fast_mode=True)
                optimization_report = optimization_result.json(indent=2)
                logger.info(f"Optimization completed. Best score: {optimization_result.best_score}")
            except Exception as opt_e:
                logger.error(f"Optimization failed: {opt_e}")
                optimization_report = f"Optimization failed: {opt_e}"

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
