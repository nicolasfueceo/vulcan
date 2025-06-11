# src/agents/hypothesis_agent.py
import json
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.logging_utils import (
    log_agent_context,
    log_agent_error,
    log_agent_response,
    log_llm_prompt,
    log_llm_response,
    setup_agent_logger,
)
from src.utils.memory import get_mem, set_mem
from src.utils.pubsub import acquire_lock, publish, release_lock


class HypothesisAgent:
    """
    An agent responsible for generating hypotheses based on EDA results.
    """

    def __init__(self, llm_config: dict):
        setup_agent_logger(self.__class__.__name__)
        logger.info("HypothesisAgent initialized.")
        self.assistant = autogen.AssistantAgent(
            name="HypothesisGenerator",
            system_message="You are an expert data analyst. Your goal is to generate insightful hypotheses based on the provided Exploratory Data Analysis (EDA) results. You must call the `save_hypotheses` function with your results.",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_HypothesisAgent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,  # Limit retries to prevent infinite loops
            code_execution_config={
                "work_dir": "runtime/code_execution",
                "use_docker": False,
            },
        )
        self.writer = SummaryWriter("runtime/tensorboard/HypothesisAgent")
        self.run_count = get_mem("hypothesis_run_count") or 0

    @agent_run_decorator("HypothesisAgent")
    def run(self, message: dict = {}):
        """
        Runs the hypothesis generation pipeline. Triggered by a pub/sub event.
        """
        log_agent_context(self.__class__.__name__, message)

        lock_name = "lock:HypothesisAgent"
        if not acquire_lock(lock_name):
            logger.info("HypothesisAgent is already running. Skipping.")
            return

        try:
            eda_results = get_mem("eda")
            logger.info(f"Retrieved EDA results from memory: {bool(eda_results)}")

            if not eda_results:
                logger.warning(
                    "No EDA results found in memory. Skipping hypothesis generation."
                )
                return

            logger.info(
                f"EDA results summary: {list(eda_results.keys()) if eda_results else 'None'}"
            )

            hypotheses_generated = []

            def save_hypotheses(
                hypotheses: List[str],
            ):  # Changed to List[str] for simpler LLM output
                """Saves the generated hypotheses to memory and publishes an event."""
                nonlocal hypotheses_generated
                hypotheses_generated = hypotheses

                # Convert to Hypothesis objects
                hypothesis_objects = [
                    {"hypothesis": h, "confidence": 0.8} for h in hypotheses
                ]

                set_mem("hypotheses", hypothesis_objects)
                logger.info(f"Saved {len(hypotheses)} hypotheses to memory.")
                self.writer.add_scalar(
                    "hypotheses_generated", len(hypotheses), self.run_count
                )

                # Publish an event to trigger the next agent
                publish(
                    "hypotheses_ready",
                    {
                        "status": "success",
                        "hypotheses": hypothesis_objects,
                    },
                )
                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={"save_hypotheses": save_hypotheses}
            )

            prompt = f"""
            Given the following EDA results, generate 5-7 interesting hypotheses
            about user behavior or item properties. Call the `save_hypotheses`
            function with a list of hypothesis strings.

            EDA Results:
            {json.dumps(eda_results, indent=2)}
            """

            log_llm_prompt(self.__class__.__name__, prompt)

            # Initiate chat with the assistant
            result = self.user_proxy.initiate_chat(self.assistant, message=prompt)

            log_llm_response(self.__class__.__name__, str(result))

            if hypotheses_generated:
                log_agent_response(
                    self.__class__.__name__,
                    {"hypotheses_count": len(hypotheses_generated)},
                )
            else:
                logger.warning("No hypotheses were generated successfully")

            self.run_count += 1
            set_mem("hypothesis_run_count", self.run_count)

        except Exception as e:
            log_agent_error(self.__class__.__name__, e)
            raise
        finally:
            release_lock(lock_name)
            self.writer.close()
