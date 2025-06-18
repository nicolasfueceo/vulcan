import pytest
import pandas as pd
from agentic.langgraph.nodes.evaluation_node import EvaluationNode
from agentic.core.metric_logger import MetricLogger
from agentic.core.agent_memory import AgentMemory
import numpy as np

def make_fake_interactions(n_users=3, n_inter=10):
    users = [f"u{i+1}" for i in range(n_users)]
    data = []
    for u in users:
        for i in range(n_inter):
            data.append({"user_id": u, "interaction_id": f"{u}_int{i+1}", "timestamp": i})
    return pd.DataFrame(data)

def test_evaluation_node_runs_and_logs(tmp_path):
    # Setup
    log_dir = tmp_path / "metrics"
    metric_logger = MetricLogger(log_dir=log_dir)
    memory = AgentMemory()
    node = EvaluationNode(memory, metric_logger, db_path=":memory:", splits_dir=tmp_path / "splits")
    interactions_df = make_fake_interactions(n_users=2, n_inter=5)
    realized_feature = {"name": "feature_X", "feature_fn": lambda x: x}
    bo_result = {"best_value": 0.77, "stddev": 0.02, "params": {"alpha": 1.0}}
    state = {
        "realized_feature": realized_feature,
        "interactions_df": interactions_df,
        "bo_result": bo_result,
        "round": 1
    }
    # Run node
    out_state = node.run(state)
    # Check metrics in state
    assert "metrics" in out_state
    assert "metrics_stddev" in out_state
    # Check log file
    import json
    log_path = log_dir / "metrics.json"
    with open(log_path, "r") as f:
        logs = json.load(f)
    assert logs[-1]["feature_name"] == "feature_X"
    assert "rmse" in logs[-1]["metrics"]
    assert "ndcg@5" in logs[-1]["metrics"]
    assert logs[-1]["bo_best_value"] == 0.77
    assert logs[-1]["params"] == {"alpha": 1.0}
