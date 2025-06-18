from agentic.bo.reward_function_manager import RewardFunctionManager
from agentic.bo.bo_loop import BayesianOptimizationLoop
from agentic.core.agent_memory import AgentMemory

def test_bo_loop_runs():
    mgr = RewardFunctionManager()
    # Example reward function that expects data_manager and arbitrary kwargs
    def reward_fn(data_manager, x, y, extra=None):
        # Use data_manager if present, otherwise just sum
        base = x + y
        if data_manager is not None and hasattr(data_manager, "tag"):  # Dummy attribute
            base += 10
        if extra:
            base += extra
        return base
    mgr.register("test_fn", reward_fn)
    mem = AgentMemory()
    class DummyDataManager:
        tag = True
    bo = BayesianOptimizationLoop(mgr, mem)
    # Should add 10 because data_manager has 'tag', plus extra=5
    result = bo.run({"x": 3, "y": 4}, reward_fn_name="test_fn", data_manager=DummyDataManager(), extra=5)
    assert result["result"] == 22
    assert mem.get("last_bo_result") == 22

# Test stratified subsampling
import pandas as pd
from agentic.langgraph.data.cv_fold_manager import CVFoldManager

def test_stratified_subsample():
    df = pd.DataFrame({"label": [0]*50 + [1]*50, "value": range(100)})
    manager = CVFoldManager()
    sample = manager.get_stratified_subsample(df, stratify_col="label", frac=0.2, random_state=42)
    # Should have 10 of each label
    counts = sample["label"].value_counts().to_dict()
    assert counts[0] == 10 and counts[1] == 10
    assert len(sample) == 20
