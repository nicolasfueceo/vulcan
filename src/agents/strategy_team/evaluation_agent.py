# src/agents/evaluation_agent.py
from loguru import logger
from tensorboardX import SummaryWriter

from src.utils.decorators import agent_run_decorator
from src.utils.memory import get_mem
from src.utils.pubsub import acquire_lock, publish, release_lock


class EvaluationAgent:
    def __init__(self, llm_config: dict = None):
        self.writer = SummaryWriter("runtime/tensorboard/EvaluationAgent")
        self.run_count = 0

    @agent_run_decorator("EvaluationAgent")
    def run(self, message: dict = {}):
        """
        Runs a final evaluation on the best model and logs metrics.
        Triggered by 'opt_complete'.
        """
        lock_name = "lock:EvaluationAgent"
        if not acquire_lock(lock_name):
            logger.info("EvaluationAgent is already running. Skipping.")
            return

        try:
            best_rmse = get_mem("best_rmse")
            best_params = get_mem("best_params")

            if not best_params:
                logger.warning("No best parameters found. Skipping evaluation.")
                return

            # In a real implementation, this agent would:
            # 1. Retrain the model on the full training data using best_params.
            # 2. Evaluate on a held-out test set.
            # 3. Calculate various metrics (e.g., NDCG, precision@k).
            # 4. Log everything to TensorBoard.

            logger.info(f"Final evaluation with best params: {best_params}")
            logger.info(f"Best validation RMSE: {best_rmse}")

            # For now, just log the best RMSE from validation as the final metric
            self.writer.add_scalar("final_test_rmse", best_rmse, self.run_count)
            self.writer.add_hparams(best_params, {"hparam/final_test_rmse": best_rmse})

            self.run_count += 1
            publish("evaluation_done", {"status": "success"})

        finally:
            release_lock(lock_name)
            self.writer.close()
