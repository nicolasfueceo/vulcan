import json
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# This is no longer needed with a proper project structure and installation.
# # Add project root to Python path for proper imports
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))
import autogen
from loguru import logger

from src.agents.discovery_team.insight_discovery_agents import (
    get_insight_discovery_agents,
)
from src.agents.strategy_team.hypothesis_agents import get_hypothesis_agents
from src.config.logging import setup_logging
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

# Load environment variables


def run_discovery_loop(session_state: SessionState):
    """Orchestrates the Insight Discovery Team to find patterns in the data."""
    logging.info("--- Running Insight Discovery Loop ---")

    config_list = config_list_from_json(
        os.getenv("OAI_CONFIG_LIST", "config/OAI_CONFIG_LIST.json")
    )

    # Validate and substitute the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("OPENAI_API_KEY environment variable is empty or not set.")

    logging.info(f"API key loaded with length: {len(api_key)}")

    # Manually substitute the API key to ensure it's loaded
    for config in config_list:
        if config.get("api_key") == "${OPENAI_API_KEY}":
            config["api_key"] = api_key
            logging.info("Substituted API key in config")
        elif not config.get("api_key"):
            raise ValueError(f"No API key found in config: {config}")

    # Validate final config
    for i, config in enumerate(config_list):
        if not config.get("api_key") or config.get("api_key").strip() == "":
            raise ValueError(
                f"Config {i} has empty API key after substitution: {config}"
            )

    logging.info(f"Final config list validated with {len(config_list)} configurations")

    llm_config = {"config_list": config_list, "cache_seed": None}

    # This agent will execute code blocks and call functions
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy_ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config={"work_dir": str(get_run_dir())},  # Enable code execution
    )

    assistant_agents = get_insight_discovery_agents(llm_config)

    # Register the granular tools for the agents
    for agent in assistant_agents.values():
        autogen.register_function(
            run_sql_query,
            caller=agent,
            executor=user_proxy,
            name="run_sql_query",
            description="Executes a read-only SQL query and returns the result as markdown.",
        )
        autogen.register_function(
            get_table_sample,
            caller=agent,
            executor=user_proxy,
            name="get_table_sample",
            description="Retrieves a random sample of rows from a specified table.",
        )
        autogen.register_function(
            create_analysis_view,
            caller=agent,
            executor=user_proxy,
            name="create_analysis_view",
            description="Creates a SQL view with documentation for complex analysis.",
        )
        autogen.register_function(
            vision_tool,
            caller=agent,
            executor=user_proxy,
            name="vision_tool",
            description="Analyzes image files using vision AI.",
        )
        autogen.register_function(
            get_add_insight_tool(session_state),
            caller=agent,
            executor=user_proxy,
            name="add_insight_to_report",
            description="Saves structured insights to the session report.",
        )

    # Enhanced GroupChat with intelligent context management
    group_chat = autogen.GroupChat(
        agents=[user_proxy] + list(assistant_agents.values()),
        messages=[],
        max_round=500,  # High limit instead of -1 (uncapped), our SmartGroupChatManager will handle termination
        allow_repeat_speaker=True,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Close the session database connection to avoid lock conflicts during agent execution
    logging.info("Closing database connection for agent execution...")
    session_state.close_connection()

    def should_continue_exploration(
        session_state: SessionState, round_count: int
    ) -> bool:
        """Determines if exploration should continue based on insights and coverage."""
        insights = len(session_state.insights)
        min_insights = 8  # Minimum insights needed for comprehensive analysis

        # Always continue if we don't have enough insights
        if insights < min_insights:
            logging.info(
                f"Continuing exploration: {insights}/{min_insights} insights captured"
            )
            return True

        # After minimum insights, check for completeness every 10 rounds
        if round_count % 10 == 0 and insights >= min_insights:
            logging.info(
                f"Evaluating exploration completeness: {insights} insights, round {round_count}"
            )

            # Check if we have coverage across major areas
            insight_titles = [
                insight.title.lower() for insight in session_state.insights
            ]
            coverage_areas = {
                "rating": any("rating" in title for title in insight_titles),
                "genre": any(
                    any(term in title for term in ["genre", "shelf", "category"])
                    for title in insight_titles
                ),
                "author": any("author" in title for title in insight_titles),
                "temporal": any(
                    any(term in title for term in ["time", "year", "date", "temporal"])
                    for title in insight_titles
                ),
                "user": any("user" in title for title in insight_titles),
            }

            covered_areas = sum(coverage_areas.values())
            if covered_areas >= 4:  # Need coverage of at least 4/5 major areas
                logging.info(
                    f"Sufficient coverage achieved: {covered_areas}/5 areas covered"
                )
                return False

        # Safety limit - don't run indefinitely
        if round_count > 200:
            logging.warning(
                f"Reached maximum round limit ({round_count}), stopping exploration"
            )
            return False

        # Continue if we haven't reached the insight threshold or coverage
        return True

    def get_progress_prompt(session_state: SessionState, round_count: int) -> str:
        """Generate a progress prompt to guide agents when they seem stuck."""
        insights = len(session_state.insights)

        if insights == 0 and round_count > 5:
            return "\n\nIMPORTANT: No insights have been captured yet. Please ensure you call `add_insight_to_report()` after each analysis to record your findings. Focus on generating actual insights, not just data exploration."

        if insights < 4 and round_count > 15:
            return f"\n\nPROGRESS CHECK: Only {insights} insights captured after {round_count} rounds. Please focus on generating concrete insights using `add_insight_to_report()` and ensure comprehensive coverage of rating patterns, genres, authors, and user behavior."

        # Check coverage gaps
        insight_titles = [insight.title.lower() for insight in session_state.insights]
        missing_areas = []
        if not any("rating" in title for title in insight_titles):
            missing_areas.append("rating analysis")
        if not any(
            any(term in title for term in ["genre", "shelf", "category"])
            for title in insight_titles
        ):
            missing_areas.append("genre/category analysis")
        if not any("author" in title for title in insight_titles):
            missing_areas.append("author analysis")
        if not any(
            any(term in title for term in ["time", "year", "date", "temporal"])
            for title in insight_titles
        ):
            missing_areas.append("temporal analysis")
        if not any("user" in title for title in insight_titles):
            missing_areas.append("user behavior analysis")

        if missing_areas and round_count % 20 == 0:
            return f"\n\nCOVERAGE GAP: Missing analysis in: {', '.join(missing_areas)}. Please prioritize these areas in your next analysis."

        return ""

    def compress_conversation_context(
        messages: list, keep_recent: int = 20, llm_config: dict = None
    ) -> list:
        """Intelligently compress conversation context using LLM summarization."""
        if len(messages) <= keep_recent * 2:
            return messages  # No compression needed if conversation is still short

        logging.info(
            f"Compressing context with LLM: {len(messages)} -> target ~{keep_recent * 2} messages"
        )

        # Always preserve system messages and recent messages
        system_messages = [
            msg for msg in messages[:3] if msg.get("role") in ["user", "system"]
        ]
        recent_messages = messages[-keep_recent:]
        middle_messages = messages[len(system_messages) : -keep_recent]

        if not middle_messages:
            return messages

        try:
            # Create an LLM client for compression
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Group middle messages into chunks for summarization
            chunk_size = 15  # Process in chunks to avoid token limits
            compressed_summaries = []

            for i in range(0, len(middle_messages), chunk_size):
                chunk = middle_messages[i : i + chunk_size]

                # Create conversation text for summarization
                conversation_text = ""
                for msg in chunk:
                    role = msg.get("name", msg.get("role", "unknown"))
                    content = msg.get("content", "")
                    conversation_text += f"{role}: {content}\n\n"

                # Ask LLM to compress this chunk
                compression_prompt = f"""You are compressing a conversation between data analysis agents exploring a book recommendation database. 

Please create a concise summary that preserves:
1. All specific insights and findings discovered
2. Key analysis results, correlations, and statistics  
3. Important database views created
4. Significant plots generated and their interpretations
5. Critical decision points and conclusions

Focus on preserving factual discoveries and actionable insights while removing redundant discussion.

CONVERSATION CHUNK TO COMPRESS:
{conversation_text}

COMPRESSED SUMMARY (preserve all insights and key findings):"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use mini for cost efficiency in compression
                    messages=[{"role": "user", "content": compression_prompt}],
                    max_tokens=64000,
                    temperature=0.1,  # Low temperature for consistent compression
                )

                summary = response.choices[0].message.content

                # Create a compressed message
                compressed_msg = {
                    "role": "assistant",
                    "name": "ContextCompressor",
                    "content": f"[COMPRESSED SUMMARY of rounds {i + len(system_messages) + 1}-{i + len(system_messages) + len(chunk)}]\n\n{summary}",
                }
                compressed_summaries.append(compressed_msg)

            logging.info(
                f"LLM compression: {len(middle_messages)} messages -> {len(compressed_summaries)} summaries"
            )

            # Combine preserved and compressed content
            compressed_context = (
                system_messages + compressed_summaries + recent_messages
            )
            logging.info(
                f"Context compression completed: {len(messages)} -> {len(compressed_context)} messages"
            )
            return compressed_context

        except Exception as e:
            logging.warning(
                f"LLM compression failed: {e}. Falling back to keyword-based compression."
            )
            return _fallback_compression(messages, keep_recent)

    def _fallback_compression(messages: list, keep_recent: int = 20) -> list:
        """Fallback keyword-based compression if LLM compression fails."""
        if len(messages) <= keep_recent * 2:
            return messages

        # Always keep the initial system message and recent messages
        system_messages = [
            msg for msg in messages[:3] if msg.get("role") in ["user", "system"]
        ]
        recent_messages = messages[-keep_recent:]

        # Extract key insights and tool execution results from middle messages
        key_messages = []
        for msg in messages[len(system_messages) : -keep_recent]:
            content = msg.get("content", "")
            # Keep messages with insights, analysis results, or tool outputs
            if any(
                keyword in content.lower()
                for keyword in [
                    "insight:",
                    "analysis:",
                    "correlation:",
                    "finding:",
                    "plot_saved:",
                    "view_created:",
                    "stdout:",
                    "pattern:",
                    "hypothesis:",
                    "recommendation:",
                    "conclusion:",
                ]
            ):
                key_messages.append(msg)

        # Combine and return compressed context
        compressed = (
            system_messages + key_messages[-8:] + recent_messages
        )  # Keep max 8 key messages
        logging.info(
            f"Fallback compression completed: retained {len(compressed)} messages"
        )
        return compressed

    try:
        round_count = 0
        initial_message = "Team, let's begin our analysis. The database schema and our mission are in your system prompts. Please start by planning your first exploration step."

        # Enhanced conversation with custom termination logic
        class SmartGroupChatManager(autogen.GroupChatManager):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.round_count = 0

            def run_chat(self, messages, sender, config=None):
                """Override the main chat runner to add our smart features."""
                self.round_count += 1

                # Log progress
                if self.round_count % 10 == 0:
                    insights_count = len(session_state.insights)
                    logging.info(
                        f"Exploration progress: Round {self.round_count}, {insights_count} insights captured"
                    )
                    if session_state.insights:
                        recent_insights = session_state.insights[-3:]
                        logging.info(
                            f"Recent insights: {[insight.title for insight in recent_insights]}"
                        )

                # Apply context compression periodically
                if self.round_count % 25 == 0:
                    try:
                        self.groupchat.messages = compress_conversation_context(
                            self.groupchat.messages, llm_config=llm_config
                        )
                        logging.info(
                            f"Applied LLM context compression at round {self.round_count}"
                        )
                    except Exception as e:
                        logging.warning(f"Context compression failed: {e}")

                # Check termination only after sufficient rounds
                if self.round_count > 15:
                    if not should_continue_exploration(session_state, self.round_count):
                        logging.info(
                            "Exploration criteria met, terminating conversation"
                        )
                        # Add termination message to conversation
                        termination_msg = {
                            "role": "assistant",
                            "content": "TERMINATE - Exploration criteria met. Sufficient insights captured with good coverage.",
                            "name": "SystemCoordinator",
                        }
                        self.groupchat.messages.append(termination_msg)
                        return True  # Signal termination

                # Add progress prompts when needed
                if self.round_count > 5 and self.round_count % 15 == 0:
                    progress_prompt = get_progress_prompt(
                        session_state, self.round_count
                    )
                    if progress_prompt:
                        logging.info(
                            f"Adding progress guidance at round {self.round_count}"
                        )
                        guidance_msg = {
                            "role": "user",
                            "content": progress_prompt,
                            "name": "SystemCoordinator",
                        }
                        self.groupchat.messages.append(guidance_msg)

                # Call the parent implementation
                res = super().run_chat(messages, sender, config)
                # super() returns a tuple (final, reply). If reply is None, substitute a fallback.
                if isinstance(res, tuple) and len(res) == 2:
                    final, reply = res
                    if reply is None:
                        logging.warning(
                            "Parent run_chat returned None reply; substituting fallback message to avoid crash."
                        )
                        reply = {
                            "role": "assistant",
                            "content": "ERROR: Reply generation failed (empty). Please review the previous tool call outputs and ensure a proper assistant response.",
                        }
                    return final, reply
                # In unexpected cases, just return res as-is
                return res

        # Create the enhanced manager
        smart_manager = SmartGroupChatManager(
            groupchat=group_chat, llm_config=llm_config
        )

        # Start the conversation using AutoGen's standard pattern
        user_proxy.initiate_chat(smart_manager, message=initial_message)

        insights_count = len(session_state.insights)
        logging.info(
            f"Exploration completed after {smart_manager.round_count} rounds with {insights_count} insights"
        )

        logging.info("--- FINAL INSIGHTS SUMMARY ---")
        logging.info(session_state.get_final_insight_report())

        run_dir = get_run_dir()
        views_file = run_dir / "generated_views.json"
        if views_file.exists():
            with open(views_file, "r") as f:
                views_data = json.load(f)
            logging.info(f"Total views created: {len(views_data.get('views', []))}")
        else:
            logging.info("Total views created: 0")

    finally:
        # Reopen the connection after agent execution
        logging.info("Reopening database connection...")
        session_state.reconnect()

    logging.info("--- Insight Discovery Loop Complete ---")


def run_strategy_loop(session_state: SessionState):
    """Orchestrates the Hypothesis & Strategy Team to refine insights."""
    logging.info("--- Running Strategy Loop ---")
    insights_report = session_state.get_final_insight_report()
    if "No insights" in insights_report:
        logging.warning("Skipping strategy loop as no insights were generated.")
        return None

    config_list = config_list_from_json(
        os.getenv("OAI_CONFIG_LIST", "config/OAI_CONFIG_LIST.json")
    )

    # Validate and substitute the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("OPENAI_API_KEY environment variable is empty or not set.")

    logging.info(f"API key loaded with length: {len(api_key)}")

    # Manually substitute the API key to ensure it's loaded
    for config in config_list:
        if config.get("api_key") == "${OPENAI_API_KEY}":
            config["api_key"] = api_key
            logging.info("Substituted API key in config")
        elif not config.get("api_key"):
            raise ValueError(f"No API key found in config: {config}")

    # Validate final config
    for i, config in enumerate(config_list):
        if not config.get("api_key") or config.get("api_key").strip() == "":
            raise ValueError(
                f"Config {i} has empty API key after substitution: {config}"
            )

    logging.info(f"Final config list validated with {len(config_list)} configurations")

    llm_config = {"config_list": config_list, "cache_seed": None}

    agents_and_proxy = get_hypothesis_agents(llm_config, insights_report)
    user_proxy = agents_and_proxy.pop("user_proxy")
    agents = list(agents_and_proxy.values())

    user_proxy.register_function(
        function_map={
            "run_sql_query": run_sql_query,
            "finalize_hypotheses": get_finalize_hypotheses_tool(session_state),
        }
    )

    group_chat = autogen.GroupChat(
        agents=[user_proxy] + agents,
        messages=[],
        max_round=20,
        allow_repeat_speaker=True,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Close the session database connection to avoid lock conflicts during agent execution
    logging.info("Closing database connection for strategy agent execution...")
    session_state.close_connection()

    try:
        user_proxy.initiate_chat(
            manager,
            message=f"""Welcome strategists. Your task is to refine the following insights into a set of concrete, testable hypotheses. You can use `run_sql_query` to re-verify findings. When the final list is agreed upon, the EngineerAgent must call `finalize_hypotheses`.

--- INSIGHTS REPORT ---
{insights_report}
""",
        )
    finally:
        # Reopen the connection after agent execution
        logging.info("Reopening database connection...")
        session_state.reconnect()

    logging.info("--- Strategy Loop Complete ---")
    return session_state.get_final_hypotheses()


def main():
    """
    Main function to run the VULCAN agent orchestration.
    """
    # Initialize the run
    run_id, run_dir = init_run()
    logger.info(f"Starting VULCAN Run ID: {run_id}")

    # Setup logging
    setup_logging()

    # Initialize session state
    session_state = SessionState(run_dir)

    try:
        run_discovery_loop(session_state)
        # final_hypotheses = run_strategy_loop(session_state)

        logging.info("Orchestration complete.")
        # if final_hypotheses is not None:
        #     logging.info("--- FINAL VETTED HYPOTHESES ---")
        #     for h in final_hypotheses:
        #         logging.info(f"- {h.model_dump_json(indent=2)}")
        # else:
        #     logging.info("--- No hypotheses were finalized. ---")

    except Exception as e:
        logging.error(f"An error occurred during orchestration: {e}", exc_info=True)
    finally:
        session_state.close_connection()

        # Clean up any generated views
        cleanup_analysis_views(Path(session_state.run_dir))
        logging.info("View cleanup process initiated.")

        logging.info("Run finished. Session state saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"VULCAN run failed: {e}")
        logger.exception(e)
        # Optional: Add any cleanup logic here
        sys.exit(1)
