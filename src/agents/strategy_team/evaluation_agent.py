# src/agents/evaluation_agent.py
from typing import Optional
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.session_state import SessionState


class EvaluationAgent:
    def __init__(self, llm_config: Optional[dict] = None):
        self.writer = SummaryWriter("runtime/tensorboard/EvaluationAgent")
        self.run_count = 0

    @agent_run_decorator("EvaluationAgent")
    def run(self, session_state: SessionState):
        """
        Runs a final evaluation on the best model and logs metrics.
        This is a procedural step called by the orchestrator.
        """
        logger.info("Starting final evaluation...")

        opt_results = session_state.get_state("optimization_results", {})
        best_trial = opt_results.get("best_trial")

        if not best_trial:
            logger.warning("No optimization results found. Skipping evaluation.")
            return

        best_params = best_trial.params
        best_value = best_trial.value

        # In a real implementation, this would involve retraining and testing on a holdout set.
        # For now, we'll consider the best validation score as the final metric.
        logger.info(f"Final evaluation with best params: {best_params}")
        logger.info(f"Best validation score (e.g., RMSE): {best_value}")

        # Log metrics to TensorBoard
        self.writer.add_scalar("final_test_metric", best_value, self.run_count)
        self.writer.add_hparams(best_params, {"hparam/final_test_metric": best_value})

        # Store final results in the session state
        evaluation_results = {
            "final_metric": best_value,
            "best_params": best_params,
            "status": "success",
        }
        session_state.set_state("evaluation_results", evaluation_results)

        logger.info("Final evaluation complete. Results saved to session state.")
        self.run_count += 1
        self.writer.close()
