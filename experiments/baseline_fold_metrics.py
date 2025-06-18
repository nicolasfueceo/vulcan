"""
Compute and save per-fold baseline metrics (precision@k, recall@k, f1@k, rmse, etc.) for all baseline recommenders.
Evaluates for k=5 and k=10, and reports mean and stddev across all folds.
Output: experiments/baseline_fold_metrics.json
"""
import json
from pathlib import Path
import numpy as np
from agentic.recommenders.popularity import PopularityRecommender
from agentic.recommenders.knn import KNNRecommender
from agentic.recommenders.rf import RandomForestRecommender
from agentic.recommenders.lightfm import LightFMRecommender
from agentic.langgraph.data.cv_fold_manager import CVFoldManager

import pandas as pd

def compute_metrics_across_folds(recommender_cls, cv_manager, n_folds=5, k_list=[5, 10]):
    fold_metrics = []
    folds = cv_manager.load_folds_json()  # Should return a list of dicts, each dict has 'train', 'val', 'test'
    if folds is None:
        raise RuntimeError("No CV folds found in splits_dir")
    for fold_idx, fold in enumerate(folds[:n_folds]):
        # Each fold is a dict with keys 'train', 'val', 'test', each a list of dicts with user_id, item_id, rating
        train_df = pd.DataFrame(fold.get('train', []))
        test_df = pd.DataFrame(fold.get('test', []))
        rec = recommender_cls() if recommender_cls is not LightFMRecommender else recommender_cls(no_components=10, loss='warp', epochs=5)
        rec.fit(train_df, user_col="user_id", item_col="item_id", rating_col="rating")
        fold_result = {}
        for k in k_list:
            metrics = rec.score(test_df, user_col="user_id", item_col="item_id", k=k)
            for metric, value in metrics.items():
                fold_result[f"{metric}"] = value
        fold_metrics.append(fold_result)
    # Aggregate mean and stddev for each metric
    metrics_keys = fold_metrics[0].keys()
    agg = {}
    for key in metrics_keys:
        values = [fm[key] for fm in fold_metrics]
        agg[key + "_mean"] = float(np.mean(values))
        agg[key + "_std"] = float(np.std(values))
    return {"per_fold": fold_metrics, "aggregate": agg}

def main():
    db_path = "data/goodreads_curated.duckdb"
    splits_dir = "data/cv_splits"
    output_file = "experiments/baseline_fold_metrics.json"
    n_folds = 5
    k_list = [5, 10]
    cv_manager = CVFoldManager(splits_dir=splits_dir, db_path=db_path)
    baselines = {
        "popularity": PopularityRecommender,
        "knn": KNNRecommender,
        "random_forest": RandomForestRecommender,
        "lightfm": LightFMRecommender,
    }
    results = {}
    for name, rec_cls in baselines.items():
        print(f"Evaluating {name} recommender...")
        results[name] = compute_metrics_across_folds(rec_cls, cv_manager, n_folds=n_folds, k_list=k_list)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_file}")

if __name__ == "__main__":
    main()
