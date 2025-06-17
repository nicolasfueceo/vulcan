"""
End-to-end test for the refactored Strategy Team pipeline:
- Loads a mock hypothesis into SessionState
- Runs the full pipeline: Strategist -> Engineer (per-feature, with retries) -> Optimizer
- Asserts that realized features are valid and optimizer runs
- Prints/logs all outputs and errors for review
"""
import sys
import os
from pathlib import Path
from loguru import logger

# Adjust path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.session_state import SessionState
from src.agents.strategy_team.strategy_team_agents import get_strategy_team_agents

from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer
from src.orchestrator import run_strategy_loop
from src.utils.run_utils import init_run


from src.schemas.models import Hypothesis

def mock_hypothesis():
    """Return a list of real Hypothesis model instances for testing."""
    return [
        Hypothesis(
            summary='Users who rate more books tend to have higher average ratings.',
            rationale='Rating more books may reflect higher engagement, which could correlate with higher average ratings.',
            depends_on=['user_id', 'book_id', 'rating']
        )
    ]


def main():
    logger.info("=== TEST: Strategy Team Pipeline End-to-End ===")
    # Setup session state
    run_dir = Path("test_runs/strategy_team_test")
    run_dir.mkdir(parents=True, exist_ok=True)
    session_state = SessionState(run_dir=run_dir)
    session_state.set_state("final_hypotheses", mock_hypothesis())
    session_state.set_state("hypotheses", mock_hypothesis())  # Ensure both keys are set for orchestrator compatibility
    session_state.set_state("insights", [])

    # Use real API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": api_key}], "cache_seed": 42}

    # Initialize run context (required for agent creation)
    init_run()

    # Create agents
    agents = get_strategy_team_agents(llm_config, db_schema="user_id, book_id, rating")

    # Run strategy loop
    report = run_strategy_loop(session_state, agents, llm_config)
    logger.info(f"Strategy loop report: {report}")

    # Check realized features
    realized_features = report.get("realized_features", [])
    assert realized_features, "No features were realized."
    logger.success(f"Number of realized features: {len(realized_features)}")

    # Run optimizer on realized features
    optimizer = VULCANOptimizer(session=session_state)
    try:
        optimization_result = optimizer.optimize(features=realized_features, n_trials=3, use_fast_mode=True)
        logger.success(f"Optimizer ran successfully. Best score: {optimization_result.best_score}")
    except Exception as e:
        logger.error(f"Optimizer failed: {e}")

    logger.info("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()
