# src/agents/feature_ideation_agent.py
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.schemas.models import CandidateFeature
from src.utils.decorators import agent_run_decorator
from src.utils.prompt_utils import load_prompt
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.session_state import SessionState


class FeatureIdeationAgent:
    """
    An agent responsible for generating candidate features from hypotheses.
    """

    def __init__(self, llm_config: dict, session_state: SessionState):
        logger.info("FeatureIdeationAgent initialized.")
        self.session_state = session_state
        self.assistant = autogen.AssistantAgent(
            name="FeatureIdeator",
            system_message="",  # Will be set dynamically before each run.
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_FeatureIdeationAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/FeatureIdeationAgent")

    @agent_run_decorator("FeatureIdeationAgent")
    def run(self, message: dict = {}):
        """
        Runs the feature ideation pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:FeatureIdeationAgent"
        if not acquire_lock(lock_name):
            logger.info("FeatureIdeationAgent is already running. Skipping.")
            return

        try:
            prioritized_hypotheses = self.session_state.get_prioritized_hypotheses()
            if not prioritized_hypotheses:
                logger.warning(
                    "No prioritized hypotheses found in session state. Skipping feature ideation."
                )
                return

            def save_candidate_features(features: List[CandidateFeature]):
                """Saves the generated candidate features to session state and publishes an event."""
                self.session_state.set_candidate_features(
                    [f.model_dump() for f in features]
                )
                logger.info(
                    f"Saved {len(features)} candidate features to session state."
                )
                publish(
                    "features_ready",
                    {"candidate_features": [f.model_dump() for f in features]},
                )
                run_count = self.session_state.increment_ideation_run_count()
                self.writer.add_scalar("features_generated", len(features), run_count)
                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={"save_candidate_features": save_candidate_features}
            )

            # Load the full prompt for this specific run
            prompt = load_prompt(
                "agents/feature_ideator.j2",
                prioritized_hypotheses=prioritized_hypotheses,
            )

            # Update the system message and initiate the chat
            self.assistant.update_system_message(prompt)
            self.user_proxy.initiate_chat(
                self.assistant,
                message="Generate candidate features based on the hypotheses provided in your system message. You must call the `save_candidate_features` function with your results.",
            )
            self.writer.close()
        finally:
            release_lock(lock_name)
