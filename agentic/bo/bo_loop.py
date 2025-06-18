from agentic.bo.reward_function_manager import RewardFunctionManager
from agentic.core.agent_memory import AgentMemory
from typing import Any, Dict

class BayesianOptimizationLoop:
    """
    Bayesian Optimization loop with pluggable reward functions and memory integration.
    Now supports passing data_manager and arbitrary kwargs to reward functions.

    Example reward function signature:
        def reward_fn(data_manager, features, model_params, subsample_frac=1.0, **kwargs):
            ...
    """
    def __init__(self, reward_manager: 'RewardFunctionManager', memory: 'AgentMemory'):
        self.reward_manager = reward_manager
        self.memory = memory

    def run(self, config: Dict[str, Any], reward_fn_name: str, data_manager=None, **kwargs) -> Dict[str, Any]:
        """
        Run a Bayesian Optimization step.
        Args:
            config: Dict of hyperparameters/model/feature settings.
            reward_fn_name: Name of the reward function to use.
            data_manager: Data manager object to provide data access (optional).
            **kwargs: Additional arguments for reward function (e.g., subsample_frac, fold_idx, etc.)
        Returns:
            Dict with result and any additional outputs.
        """
        reward_fn = self.reward_manager.get(reward_fn_name)
        # Pass data_manager, config, and any extra kwargs to the reward function
        result = reward_fn(data_manager=data_manager, **config, **kwargs)
        self.memory.set("last_bo_result", result)
        return {"result": result}
