# src/agents/reasoning_agent.py
import json
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.memory import get_mem, set_mem
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.schemas import PrioritizedHypothesis


class ReasoningAgent:
    """
    An agent responsible for prioritizing hypotheses based on feasibility and impact.
    """

    def __init__(self, llm_config: dict):
        logger.info("ReasoningAgent initialized.")
        self.assistant = autogen.AssistantAgent(
            name="ReasoningAgent",
            system_message="You are an expert data strategist. Your goal is to prioritize hypotheses based on their feasibility and potential impact. You must call the `save_prioritized_hypotheses` function with your results.",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_ReasoningAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReasoningAgent")
        self.run_count = get_mem("reasoning_run_count") or 0

    @agent_run_decorator("ReasoningAgent")
    def run(self, message: dict = {}):
        """
        Runs the hypothesis prioritization pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:ReasoningAgent"
        if not acquire_lock(lock_name):
            logger.info("ReasoningAgent is already running. Skipping.")
            return

        try:
            hypotheses = get_mem("hypotheses")
            if not hypotheses:
                logger.warning(
                    "No hypotheses found in memory. Skipping prioritization."
                )
                return

            def save_prioritized_hypotheses(
                prioritized_hypotheses: List[PrioritizedHypothesis],
            ):
                """Saves the prioritized hypotheses to memory and publishes an event."""
                set_mem(
                    "prioritized_hypotheses",
                    [h.model_dump() for h in prioritized_hypotheses],
                )
                logger.info(
                    f"Saved {len(prioritized_hypotheses)} prioritized hypotheses to memory."
                )

                # Publish an event to trigger the next agent
                publish(
                    "priorities_ready",
                    {
                        "status": "success",
                        "prioritized_hypotheses": [
                            h.model_dump() for h in prioritized_hypotheses
                        ],
                    },
                )
                self.writer.add_scalar(
                    "hypotheses_prioritized",
                    len(prioritized_hypotheses),
                    self.run_count,
                )
                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={
                    "save_prioritized_hypotheses": save_prioritized_hypotheses
                }
            )

            prompt = f"""
            Given the following hypotheses, please assess each one for its
            feasibility (1-5) and potential impact (1-5). A higher score is better.
            Call the `save_prioritized_hypotheses` function with your results.

            Hypotheses:
            {json.dumps(hypotheses, indent=2)}
            """

            self.user_proxy.initiate_chat(self.assistant, message=prompt)
            self.run_count += 1
            set_mem("reasoning_run_count", self.run_count)
            self.writer.close()
        finally:
            release_lock(lock_name)
