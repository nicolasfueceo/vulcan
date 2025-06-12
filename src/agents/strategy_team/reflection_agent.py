# src/agents/reflection_agent.py
from typing import List

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.schemas.models import Hypothesis
from src.utils.decorators import agent_run_decorator
from src.utils.prompt_utils import load_prompt
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.session_state import SessionState


class ReflectionAgent:
    """
    An agent responsible for reflecting on the optimization results and
    suggesting next steps.
    """

    def __init__(self, llm_config: dict, session_state: SessionState):
        logger.info("ReflectionAgent initialized.")
        self.session_state = session_state
        self.assistant = autogen.AssistantAgent(
            name="ReflectionAgent",
            system_message="",  # Will be set dynamically before each run.
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_ReflectionAgent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReflectionAgent")

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
            bo_history = self.session_state.get_bo_history()
            best_params = self.session_state.get_best_params()

            if not bo_history and not best_params:
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
                """Saves the generated reflections and next steps to session state and publishes an event."""
                reflection_data = {
                    "reflections": reflections,
                    "next_steps": [h.model_dump() for h in next_steps],
                }
                self.session_state.add_reflection(reflection_data)

                if next_steps:
                    self.session_state.finalize_hypotheses(next_steps)

                logger.info("Saved reflections and next steps to session state.")

                if next_steps:
                    publish(
                        "start_eda",
                        {"reason": "Reflection suggested new hypotheses."},
                    )
                else:
                    publish(
                        "pipeline_done",
                        {"reason": "Pipeline converged."},
                    )

                run_count = self.session_state.increment_reflection_run_count()
                self.writer.add_scalar(
                    "new_hypotheses_from_reflection", len(next_steps), run_count
                )

                return "TERMINATE"

            self.user_proxy.register_function(
                function_map={"save_reflections": save_reflections}
            )

            prompt = load_prompt(
                "agents/reflection_agent.j2",
                bo_history=bo_history,
                best_params=best_params,
            )

            self.assistant.update_system_message(prompt)
            self.user_proxy.initiate_chat(
                self.assistant,
                message="Reflect on the provided results and suggest next steps. You must call the `save_reflections` function with your results.",
            )
            self.writer.close()
        finally:
            release_lock(lock_name)
