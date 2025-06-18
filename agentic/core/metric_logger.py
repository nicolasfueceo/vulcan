import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class MetricLogger:
    """
    Logs evaluation metrics, std devs, and feature additions for each round of the pipeline.
    Supports logging to JSON and CSV for easy parsing and plotting.
    """
    def __init__(self, log_dir: str = "logs/metrics", log_name: str = "metrics.json"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_name
        self.records: List[Dict[str, Any]] = []
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []

    def log_round(self, feature_name: str, metrics: Dict[str, float], stddevs: Dict[str, float],
                  bo_best_value: Optional[float] = None, bo_stddev: Optional[float] = None,
                  params: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None):
        """
        Log the results of a pipeline round.
        Args:
            feature_name: The feature added this round.
            metrics: Dict of evaluation metrics (mean across folds).
            stddevs: Dict of std devs for each metric (across folds).
            bo_best_value: Best value from BO (if applicable).
            bo_stddev: Std dev from BO (if applicable).
            params: Hyperparameters used (if desired).
            timestamp: Optional timestamp (auto-filled if None).
            extra: Any extra metadata.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        record = {
            "feature_name": feature_name,
            "metrics": metrics,
            "stddevs": stddevs,
            "bo_best_value": bo_best_value,
            "bo_stddev": bo_stddev,
            "params": params,
            "timestamp": timestamp,
        }
        if extra:
            record.update(extra)
        self.records.append(record)
        self._save_json()

    def _save_json(self):
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def save_csv(self, csv_name: str = "metrics.csv"):
        # Flatten records for CSV output
        if not self.records:
            return
        keys = set()
        for rec in self.records:
            keys.update(rec.keys())
            if isinstance(rec.get("metrics"), dict):
                keys.update(f"metrics_{k}" for k in rec["metrics"].keys())
            if isinstance(rec.get("stddevs"), dict):
                keys.update(f"stddevs_{k}" for k in rec["stddevs"].keys())
        keys.discard("metrics")
        keys.discard("stddevs")
        keys = sorted(keys)
        csv_path = self.log_dir / csv_name
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in self.records:
                flat = {k: v for k, v in rec.items() if k not in ("metrics", "stddevs")}
                if "metrics" in rec and isinstance(rec["metrics"], dict):
                    for k, v in rec["metrics"].items():
                        flat[f"metrics_{k}"] = v
                if "stddevs" in rec and isinstance(rec["stddevs"], dict):
                    for k, v in rec["stddevs"].items():
                        flat[f"stddevs_{k}"] = v
                writer.writerow(flat)

# TODO: Define standard recommender evaluation metrics rigorously (e.g., RMSE, NDCG@k, Precision@k, Recall@k, MAP, coverage, etc.)
# Specify computation details, cutoffs, and logging format in pipeline documentation.
