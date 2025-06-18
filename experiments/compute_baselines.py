#!/usr/bin/env python3
"""Compute baseline RMSE scores for specified recommender models.

Usage:
    python compute_baselines.py --run_dir /path/to/run --output_file baseline_scores.json \
        --models lightfm deepfm popularity --n_folds 3

The script loads the cross-validation folds via SessionState / CVDataManager,
trains each baseline on every fold, records RMSE, and writes the mean / std
per model to a JSON file that can later be consumed by `run_manual_bo.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from loguru import logger
import numpy as np

from src.data.cv_data_manager import CVDataManager

# Baseline runners are imported lazily to avoid heavy deps when not needed
BASELINE_IMPORT_PATHS = {
    "lightfm": "src.baselines.recommender.lightfm_baseline.run_lightfm_baseline",
    "svd": "src.baselines.recommender.svd_baseline.run_svd_baseline",
    "random_forest": "src.baselines.recommender.random_forest_baseline.run_random_forest_baseline",
    "popularity": "src.baselines.recommender.popularity_baseline.run_popularity_baseline",
}

def _import_callable(dotted_path: str):
    """Import a callable from a dotted string path."""
    module_path, func_name = dotted_path.rsplit(".", 1)
    mod = __import__(module_path, fromlist=[func_name])
    return getattr(mod, func_name)

def compute_baseline_for_model(cv_manager: CVDataManager, model: str, n_folds: int) -> Dict[str, float]:
    """Train the baseline model over the first `n_folds` and aggregate RMSE."""
    logger.info(f"Computing baseline for {model} over {n_folds} fold(s)...")
    run_baseline = _import_callable(BASELINE_IMPORT_PATHS[model])

    rmse_scores: List[float] = []
    total_folds = cv_manager.get_fold_summary().get("n_folds", 0)
    if total_folds == 0:
        logger.warning("No CV folds found – aborting.")
        return {"rmse": float('nan'), "std": 0.0, "n_folds": 0}

    folds_to_iterate = min(n_folds, total_folds)
    for fold_idx in range(folds_to_iterate):
        train_df, test_df = cv_manager.get_fold_data(fold_idx=fold_idx, split_type="full_train")

        metrics = run_baseline(train_df, test_df)
        rmse_scores.append(float(metrics.get("rmse", np.nan)))
        logger.info(f"Fold {fold_idx}: RMSE={rmse_scores[-1]:.4f}")

    return {
        "rmse": mean(rmse_scores),
        "std": pstdev(rmse_scores) if len(rmse_scores) > 1 else 0.0,
        "n_folds": len(rmse_scores),
    }

def main():
    parser = argparse.ArgumentParser(description="Compute baseline RMSE scores")
    parser.add_argument("--db_path", required=True, type=str, help="Path to DuckDB database file")
    parser.add_argument("--splits_dir", required=True, type=str, help="Path to CV splits directory")
    parser.add_argument("--output_file", type=str, default="baseline_scores.json", help="Where to write the JSON results")
    parser.add_argument("--models", nargs="*", default=["lightfm", "svd", "random_forest", "popularity"], help="Which baseline models to evaluate")
    parser.add_argument("--n_folds", type=int, default=3, help="How many CV folds to evaluate (<= available)")
    parser.add_argument("--undersample_frac", type=float, default=1.0, help="Fraction of users to stratified-sample for each CV fold (e.g. 0.05 for 5%)")
    args = parser.parse_args()

    cv_manager = CVDataManager(db_path=args.db_path, splits_dir=args.splits_dir, undersample_frac=args.undersample_frac)

    baseline_scores: Dict[str, Dict[str, float]] = {}
    for model in args.models:
        if model not in BASELINE_IMPORT_PATHS:
            logger.warning(f"Unknown model '{model}', skipping.")
            continue
        baseline_scores[model] = compute_baseline_for_model(cv_manager, model, args.n_folds)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline_scores, f, indent=4)
    logger.info(f"Baseline scores written to {output_path}")

if __name__ == "__main__":
    main()
