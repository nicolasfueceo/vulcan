import json
import os
import sys

# Ensure DB views are set up for pipeline compatibility
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import autogen
from autogen import Agent
from dotenv import load_dotenv
from loguru import logger

import scripts.setup_views
from src.agents.discovery_team.insight_discovery_agents import get_insight_discovery_agents
from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents
from src.config.log_config import setup_logging
from src.core.database import get_db_schema_string
from src.utils.prompt_utils import refresh_global_db_schema
from src.utils.run_utils import config_list_from_json, get_run_dir, init_run
from src.utils.session_state import CoverageTracker, SessionState
from src.utils.tools import (
    cleanup_analysis_views,
    create_analysis_view,
    execute_python,
    get_add_insight_tool,
    get_finalize_hypotheses_tool,
    get_save_candidate_features_tool,
    get_table_sample,
    run_sql_query,
    vision_tool,
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
    """Determines if exploration should continue. Main termination: hypotheses finalized. Fallback: max rounds/no new insights."""
    # If hypotheses have been finalized, allow termination
    if session_state.get_final_hypotheses():
        logger.info("Hypotheses have been finalized. Discovery loop can terminate.")
        return False

    # Always continue if no insights yet (prevents empty runs)
    if not session_state.insights:
        logger.info("Cannot terminate: No insights found yet. Forcing continuation.")
        return True

    # Fallback: prevent infinite loops if agents are stuck
    if round_count > 50:
        last_insight_round = max((i.metadata.get("round_added", 0) for i in session_state.insights), default=0)
        if round_count - last_insight_round > 20:
            logger.info("Termination condition: No new insights in the last 20 rounds (fallback). Hypotheses not finalized.")
            return False

    # Default: continue until hypotheses are finalized
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
        return {"config_list": config_list, "cache_seed": None, "max_tokens": 16384}

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
        self,
        messages: List[Dict[str, Any]],
        sender: autogen.Agent,
        config: Optional[autogen.GroupChat] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Run the chat with additional tracking and feedback mechanisms."""
        self.round_count += 1
        session_state = globals().get("session_state")

        # The config is the groupchat.
        groupchat = config or self.groupchat

        # --- EARLY TERMINATION: If hypotheses are finalized, end the chat ---
        if session_state and session_state.get_final_hypotheses():
            logger.info("Hypotheses have been finalized. Terminating discovery loop.")
            return True, "TERMINATE"  # Do NOT call super().run_chat if terminating

        # --- CONTEXT INJECTION: Add context on first round ---
        if self.round_count == 1 and session_state:
            context_message = get_insight_context(session_state)
            if context_message:
                messages.append(
                    {"role": "user", "content": context_message, "name": "SystemCoordinator"}
                )

        # --- CONTEXT COMPRESSION ---
        if self.round_count > 10 and self.round_count % 10 == 0:
            try:
                groupchat.messages = compress_conversation_context(messages)
                logger.info("Applied LLM context compression at round {}", self.round_count)
            except Exception as e:
                logger.warning("Context compression failed: {}", e)

        # --- TERMINATION BLOCKER: Enforce hypothesis finalization ---
        last_msg_content = messages[-1]["content"].strip().upper() if messages else ""
        if "TERMINATE" in last_msg_content and session_state and not session_state.get_final_hypotheses():
            logger.warning("Termination signal received, but no hypotheses finalized. Blocking termination.")
            messages.append(
                {
                    "role": "user",
                    "name": "SystemCoordinator",
                    "content": (
                        "A termination request was detected, but no hypotheses have been finalized. **Hypothesizer, it is now your turn to act.** "
                        "Please synthesize the team's insights and call the `finalize_hypotheses` tool."
                    ),
                }
            )

        # --- FALLBACK TERMINATION: Prevent infinite loops ---
        if session_state and not should_continue_exploration(session_state, self.round_count):
            logger.info("Exploration criteria met (fallback), terminating conversation.")
            return True, "TERMINATE"

        # --- LOOP PREVENTION: Reset agents periodically ---
        if self.round_count > 0 and self.round_count % 20 == 0:
            logger.warning("Potential loop detected at round {}. Resetting agents.", self.round_count)
            for agent in groupchat.agents:
                if hasattr(agent, "reset"):
                    agent.reset()

        # --- GUIDANCE: Add progress prompts periodically ---
        if session_state and self.round_count > 5 and self.round_count % 15 == 0:
            progress_guidance = get_progress_prompt(session_state, self.round_count)
            if progress_guidance:
                logger.info("Adding progress guidance at round {}", self.round_count)
                messages.append(
                    {"role": "user", "content": progress_guidance, "name": "SystemCoordinator"}
                )
        prev_message_count = len(self.groupchat.messages)
        result = super().run_chat(messages, sender, self.groupchat)  # type: ignore
        # --- LOGGING: Log every message in the groupchat ---
        if session_state and hasattr(session_state, 'run_logger'):
            # Only log new messages since the last call
            new_messages = self.groupchat.messages[prev_message_count:]
            for msg in new_messages:
                session_state.run_logger.log_message(
                    sender=msg.get('name', msg.get('role', 'unknown')),
                    recipient=None,  # Not tracked at message level
                    content=msg.get('content', ''),
                    role=msg.get('role', None),
                    extra={k: v for k, v in msg.items() if k not in ['content', 'role', 'name']}
                )
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

    # --- Tool Registration with Logging ---
    from src.utils.tools_logging import log_tool_call

    # A dictionary of tool functions to be wrapped and registered.
    # The key is the name the agent will use to call the tool.
    tool_functions = {
        "run_sql_query": run_sql_query,
        "get_table_sample": get_table_sample,
        "create_analysis_view": create_analysis_view,
        "vision_tool": vision_tool,
        "add_insight_to_report": get_add_insight_tool(session_state),
        "execute_python": execute_python,
        "finalize_hypotheses": get_finalize_hypotheses_tool(session_state),
    }

    # Wrap all tool functions with the logger
    logged_tools = {
        name: log_tool_call(func, session_state, tool_name=name)
        for name, func in tool_functions.items()
    }

    # Register tools for the appropriate agents
    for agent in [analyst, researcher, critic]:
        for name in ["run_sql_query", "get_table_sample", "create_analysis_view", "vision_tool", "add_insight_to_report", "execute_python"]:
            autogen.register_function(
                logged_tools[name],
                caller=agent,
                executor=user_proxy,
                name=name,
                description=tool_functions[name].__doc__.strip().split('\n')[0] # Use first line of docstring
            )

    # Register finalize_hypotheses only for the Hypothesizer
    autogen.register_function(
        logged_tools["finalize_hypotheses"],
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
        # --- REFLECTION HANDOVER: Prepend latest reflection's next_steps (if any) ---
        reflection_intro = ""
        if getattr(session_state, 'reflections', None):
            latest_reflection = session_state.reflections[-1]
            if isinstance(latest_reflection, dict):
                # Try to extract next_steps, novel_ideas, expansion_ideas
                next_steps = latest_reflection.get("next_steps")
                novel_ideas = latest_reflection.get("novel_ideas")
                expansion_ideas = latest_reflection.get("expansion_ideas")
                if next_steps:
                    reflection_intro += "\n---\n**Reflection Agent's Next Steps:**\n"
                    if isinstance(next_steps, list):
                        for i, step in enumerate(next_steps, 1):
                            reflection_intro += f"{i}. {step}\n"
                    else:
                        reflection_intro += str(next_steps) + "\n"
                if novel_ideas:
                    reflection_intro += "\n**Novel Unexplored Ideas:**\n"
                    if isinstance(novel_ideas, list):
                        for idea in novel_ideas:
                            reflection_intro += f"- {idea}\n"
                    else:
                        reflection_intro += str(novel_ideas) + "\n"
                if expansion_ideas:
                    reflection_intro += "\n**Promising Expansions:**\n"
                    if isinstance(expansion_ideas, list):
                        for idea in expansion_ideas:
                            reflection_intro += f"- {idea}\n"
                    else:
                        reflection_intro += str(expansion_ideas) + "\n"
        initial_message = (
            reflection_intro +
            "Team, let's begin our analysis.\n"
            "- **Analysts (QuantitativeAnalyst, PatternSeeker, DataRepresenter):** Explore the data and use the `add_insight_to_report` tool to log findings. When you believe enough insights have been gathered, prompt the Hypothesizer to finalize hypotheses. Do NOT call `TERMINATE` yourself for this reason.\n"
            "- **Hypothesizer:** Only you can end the discovery phase by calling the `finalize_hypotheses` tool. Listen for cues from the team, and when prompted (or when you believe enough insights are present), synthesize the insights and call `finalize_hypotheses`.\n\n"
            "**IMPORTANT:** The discovery phase ends ONLY when the Hypothesizer calls `finalize_hypotheses`. All other agents should prompt the Hypothesizer when ready, but only the Hypothesizer can end the phase.\n\n"
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

    # --- Tool Registration ---
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


    # --- NEW: Ensure DB connection is refreshed before strategy loop ---
    logger.info("Refreshing DB connection before strategy loop to ensure all views are visible...")
    session_state.reconnect()

    report: Dict[str, Any] = {}
    try:
        # The user_proxy initiates the chat. The `message` is the first thing said.
        user_proxy.initiate_chat(manager, message=initial_message, session_state=session_state)
    
        # After the chat, we check the session_state for the results.
        # Ensure features and hypotheses are always defined for downstream use
        features = getattr(session_state, "candidate_features", [])
        hypotheses = session_state.get_final_hypotheses()
        insights = getattr(session_state, "insights", [])
        logger.info(f"Exploration completed after {manager.round_count} rounds with {len(insights)} insights")
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
    optimization_report = "Optimization step did not run."
    """
    Main orchestration function for the VULCAN pipeline.
    """
    # --- Set up logging and run context ---
    
    try:
        run_id, run_dir = init_run()
    except Exception as e:
        logger.error(f"Failed to initialize run context: {e}")
        sys.exit(1)
    logger.info(f"Starting VULCAN run: {run_id}")

    setup_logging()
    
    # --- Start TensorBoard for experiment tracking (after run context is initialized) ---
    from src.config.tensorboard import start_tensorboard
    logger.info("Launching TensorBoard server on port 6006 with global logdir: runtime/tensorboard_global")
    start_tensorboard()

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
        realized_features = list(session_state.features.values()) if hasattr(session_state, 'features') and session_state.features else []  # pylint: disable=no-member
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
        f"## Final Strategy Refinement Report\n{strategy_report}\n\n"
        f"## Final Optimization Report\n{optimization_report}\n"
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
