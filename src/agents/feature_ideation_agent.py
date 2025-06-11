# src/agents/feature_ideation_agent.py
import json
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.memory import get_mem, set_mem
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.schemas import CandidateFeature


class FeatureIdeationAgent:
    """
    An agent responsible for generating candidate features from hypotheses.
    """

    def __init__(self, llm_config: dict):
        logger.info("FeatureIdeationAgent initialized.")
        self.assistant = autogen.AssistantAgent(
            name="FeatureIdeator",
            system_message="You are a creative data scientist. Your goal is to generate candidate features based on the provided hypotheses. You must call the `save_candidate_features` function with your results.",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_FeatureIdeationAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/FeatureIdeationAgent")
        self.run_count = get_mem("ideation_run_count") or 0

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
            prioritized_hypotheses = get_mem("prioritized_hypotheses")
            if not prioritized_hypotheses:
                logger.warning(
                    "No prioritized hypotheses found in memory. Skipping feature ideation."
                )
                return

            def save_candidate_features(features: List[CandidateFeature]):
                """Saves the generated candidate features to memory and publishes an event."""
                set_mem("candidate_features", [f.model_dump() for f in features])
                logger.info(f"Saved {len(features)} candidate features to memory.")

                publish(
                    "features_ready",
                    {
                        "status": "success",
                        "candidate_features": [f.model_dump() for f in features],
                    },
                )
                self.writer.add_scalar(
                    "features_generated", len(features), self.run_count
                )
                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={"save_candidate_features": save_candidate_features}
            )

            prompt = f"""
            Given the following prioritized hypotheses, brainstorm and propose
            candidate features that could be used to test them. For each feature,
            provide a name, type (code, llm, or composition), a detailed spec,
            a rationale, and an estimated effort and impact score (1-5).
            
            Call the `save_candidate_features` function with your results.

            Prioritized Hypotheses:
            {json.dumps(prioritized_hypotheses, indent=2)}
            """

            self.user_proxy.initiate_chat(self.assistant, message=prompt)
            self.run_count += 1
            set_mem("ideation_run_count", self.run_count)
            self.writer.close()
        finally:
            release_lock(lock_name)
