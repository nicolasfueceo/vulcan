 src/agents/reflection_agent.py
import json
from typing import Dict

import autogen
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.prompt_utils import load_prompt


class ReflectionAgent:
    """
    An agent responsible for reflecting on the optimization results and
    suggesting next steps.
    """

    def __init__(self, llm_config: Dict):
        """Initialize the reflection agent."""
        self.llm_config = llm_config
        self.assistant = autogen.AssistantAgent(
            name="ReflectionAgent",
            system_message="""You are an expert data scientist and strategist. Your role is to:
1. Analyze the results of the current pipeline iteration
2. Evaluate the quality and completeness of insights and features
3. Identify gaps or areas that need more exploration
4. Decide if another iteration of the pipeline would be valuable
5. Provide clear reasoning for your decision""",
            llm_config=llm_config,
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy_Reflection",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
        )
        self.writer = SummaryWriter("runtime/tensorboard/ReflectionAgent")

    @agent_run_decorator("ReflectionAgent")
    def run(self, session_state) -> Dict:
        """
        Run the reflection process and decide if more exploration is needed.

        Args:
            session_state: The current session state containing insights and hypotheses

        Returns:
            Dict containing:
            - should_continue: bool indicating if more exploration is needed
            - reasoning: str explaining the decision
            - next_steps: list of suggested areas to explore
        """
        logger.info("Starting reflection process...")

         Gather current state
        insights = session_state.get_final_insight_report()
        hypotheses = session_state.get_final_hypotheses()
        views = session_state.get_available_views()

         Load reflection prompt
        reflection_prompt = load_prompt(
            "agents/reflection_agent.j2",
            insights=insights,
            hypotheses=json.dumps(hypotheses, indent=2),
            views=json.dumps(views, indent=2),
        )

         Run reflection chat
        self.user_proxy.initiate_chat(
            self.assistant,
            message=reflection_prompt,
        )

         Get the last message from the reflection agent
        last_message_obj = self.user_proxy.last_message()
        last_message_content = last_message_obj.get("content") if last_message_obj else None

        if not last_message_content:
            logger.error("Could not retrieve a response from the reflection agent.")
            return {
                "should_continue": False,
                "reasoning": "Failed to get a response from the reflection agent.",
                "next_steps": "Investigate the reflection agent's chat history for errors.",
            }

        try:
             Parse the response from the reflection agent
            response = json.loads(last_message_content)
            should_continue = response.get("should_continue", False)
            reasoning = response.get("reasoning", "")
            next_steps = response.get("next_steps", "")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing reflection agent response: {e}")
            logger.error(f"Raw response: {last_message_content}")
             Provide a default response
            should_continue = False
            reasoning = "Error parsing response from reflection agent."
            next_steps = "Investigate the error in the reflection agent."

        logger.info(f"Reflection decision: {should_continue}")
        logger.info(f"Reasoning: {reasoning}")
        logger.info(f"Next steps: {next_steps}")
        return {
            "should_continue": should_continue,
            "reasoning": reasoning,
            "next_steps": next_steps,
        }
