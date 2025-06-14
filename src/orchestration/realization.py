import logging
from typing import Dict

from src.agents.strategy_team.feature_realization_agent import FeatureRealizationAgent
from src.utils.session_state import SessionState

logger = logging.getLogger(__name__)


def run_feature_realization(session_state: SessionState, llm_config: Dict):
    """
    Orchestrates the Feature Realization phase by invoking the FeatureRealizationAgent.

    This function instantiates the agent and calls its run() method. The agent is
    responsible for the entire feature realization lifecycle, including:
    - Reading candidate features from the session state.
    - Interacting with the LLM to generate code.
    - Validating the generated code in a sandbox.
    - Retrying with a self-correction loop if validation fails.
    - Writing the final realized features back to the session state.
    """
    logger.info("--- Running Feature Realization Step ---")

    # Instantiate the agent. It will use the session_state to get the candidates
    # and other necessary info like db_path.
    agent = FeatureRealizationAgent(llm_config=llm_config, session_state=session_state)

    # The agent's run method encapsulates all the logic for generation and validation.
    agent.run()

    logger.info("--- Feature Realization Step Complete ---")
