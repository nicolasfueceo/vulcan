# src/agents/reflection_agent.py
import json
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.memory import get_mem, set_mem
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.schemas import Hypothesis


class ReflectionAgent:
    """
    An agent responsible for reflecting on the optimization results and
    suggesting next steps.
    """

    def __init__(self, llm_config: dict):
        logger.info("ReflectionAgent initialized.")
        self.assistant = autogen.AssistantAgent(
            name="ReflectionAgent",
            system_message="You are an expert data scientist and strategist. Your goal is to analyze the results of a feature engineering and optimization pipeline and suggest insightful next steps. You must call the `save_reflections` function with your results.",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_ReflectionAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReflectionAgent")
        self.run_count = get_mem("reflection_run_count") or 0

    @agent_run_decorator("ReflectionAgent")
    def run(self, message: dict = {}):
        """
        Runs the reflection pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:ReflectionAgent"
        if not acquire_lock(lock_name):
            logger.info("ReflectionAgent is already running. Skipping.")
            return

        try:
            bo_history = get_mem("bo_history")
            best_params = get_mem("best_params")

            if not bo_history or not best_params:
                logger.warning(
                    "No optimization history or best parameters found. Skipping reflection."
                )
                publish(
                    "pipeline_done",
                    {
                        "status": "success",
                        "reason": "No optimization results to reflect on.",
                    },
                )
                return

            def save_reflections(reflections: List[str], next_steps: List[Hypothesis]):
                """Saves the generated reflections and next steps to memory and publishes an event."""
                current_reflections = get_mem("reflections") or []
                current_reflections.append(
                    {
                        "reflections": reflections,
                        "next_steps": [h.model_dump() for h in next_steps],
                    }
                )
                set_mem("reflections", current_reflections)

                # If there are next steps, add them to the hypotheses list
                if next_steps:
                    current_hypotheses = get_mem("hypotheses") or []
                    current_hypotheses.extend([h.model_dump() for h in next_steps])
                    set_mem("hypotheses", current_hypotheses)

                logger.info("Saved reflections and next steps to memory.")

                # Publish an event to trigger the next loop or end the pipeline
                if next_steps:
                    publish(
                        "start_eda",
                        {
                            "status": "success",
                            "reason": "Reflection suggested new hypotheses.",
                        },
                    )
                else:
                    publish(
                        "pipeline_done",
                        {"status": "success", "reason": "Pipeline converged."},
                    )

                self.writer.add_scalar(
                    "new_hypotheses_from_reflection", len(next_steps), self.run_count
                )

                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={"save_reflections": save_reflections}
            )

            prompt = f"""
            Given the following Bayesian optimization history and best parameters,
            please provide some reflections on the feature engineering process and
            suggest next steps. The next steps should be new hypotheses to explore.
            If you don't have any new hypotheses, provide an empty list for next_steps.

            BO History:
            {json.dumps(bo_history, indent=2)}

            Best Parameters:
            {json.dumps(best_params, indent=2)}

            Call the `save_reflections` function with your results.
            """

            self.user_proxy.initiate_chat(self.assistant, message=prompt)

            self.run_count += 1
            set_mem("reflection_run_count", self.run_count)
            self.writer.close()
        finally:
            release_lock(lock_name)
