import os
import shutil
import json
import pytest
from agentic.core.metric_logger import MetricLogger

def test_metric_logger_json_and_csv(tmp_path):
    log_dir = tmp_path / "metrics"
    logger = MetricLogger(log_dir=log_dir, log_name="metrics.json")
    # Log a first round
    logger.log_round(
        feature_name="feature_A",
        metrics={"rmse": 1.0, "ndcg": 0.8},
        stddevs={"rmse": 0.1, "ndcg": 0.05},
        bo_best_value=0.8,
        bo_stddev=0.02,
        params={"param1": 1, "param2": 2},
        extra={"round": 1}
    )
    # Log a second round
    logger.log_round(
        feature_name="feature_B",
        metrics={"rmse": 0.9, "ndcg": 0.85},
        stddevs={"rmse": 0.08, "ndcg": 0.04},
        bo_best_value=0.85,
        bo_stddev=0.01,
        params={"param1": 3},
        extra={"round": 2}
    )
    # Check JSON file
    json_path = log_dir / "metrics.json"
    with open(json_path, "r") as f:
        records = json.load(f)
    assert len(records) == 2
    assert records[0]["feature_name"] == "feature_A"
    assert records[1]["feature_name"] == "feature_B"
    # Save and check CSV
    logger.save_csv("metrics.csv")
    csv_path = log_dir / "metrics.csv"
    with open(csv_path, "r") as f:
        lines = f.readlines()
    assert "feature_name" in lines[0]
    assert "feature_A" in lines[1]
    assert "feature_B" in lines[2]
