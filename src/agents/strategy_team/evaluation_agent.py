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
        Runs a final, paper-ready evaluation on the best model and logs metrics and artifacts.
        """
        import json
        from pathlib import Path
        import pandas as pd
        import numpy as np
        from src.data.cv_data_manager import CVDataManager
        from src.evaluation.scoring import _train_and_evaluate_lightfm
        from src.evaluation.clustering import cluster_users_kmeans
        from src.evaluation.beyond_accuracy import (
            compute_novelty, compute_diversity, compute_catalog_coverage
        )
        from src.agents.strategy_team.optimization_agent_v2 import VULCANOptimizer

        logger.info("Starting final evaluation...")
        opt_results = session_state.get_state("optimization_results", {})
        best_trial = opt_results.get("best_trial")
        realized_features = session_state.get_state("realized_features", [])
        run_dir = getattr(session_state, 'run_dir', Path("runtime/runs/unknown"))
        artifacts_dir = Path(run_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if not best_trial:
            logger.warning("No optimization results found. Skipping evaluation.")
            return
        best_params = best_trial.params
        # --- 1. Load hold-out data ---
        data_manager = CVDataManager(
            db_path=session_state.db_path,
            splits_dir="data/processed/cv_splits",
        )
        n_folds = data_manager.get_fold_summary().get("n_folds", 1)
        full_train_df, test_df = data_manager.get_fold_data(fold_idx=n_folds-1, split_type="full_train")
        # --- 2. Generate feature matrices ---
        X_train = VULCANOptimizer._generate_feature_matrix(full_train_df, realized_features, best_params)
        X_test = VULCANOptimizer._generate_feature_matrix(test_df, realized_features, best_params)
        # --- 3. Global LightFM model ---
        from lightfm.data import Dataset
        dataset = Dataset()
        all_users = pd.concat([full_train_df["user_id"], test_df["user_id"]]).unique()
        all_items = pd.concat([full_train_df["book_id"], test_df["book_id"]]).unique()
        dataset.fit(users=all_users, items=all_items)
        (test_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in test_df.iterrows()]
        )
        user_features_train = dataset.build_user_features(
            (user_id, {col: X_train.loc[user_id, col] for col in X_train.columns})
            for user_id in X_train.index
        )
        global_metrics = {}
        for k in [5, 10, 20]:
            scores = _train_and_evaluate_lightfm(
                dataset, full_train_df, test_interactions, user_features=user_features_train, k=k
            )
            global_metrics[f"precision_at_{k}"] = scores.get(f"precision_at_{k}", 0)
            global_metrics[f"recall_at_{k}"] = scores.get(f"recall_at_{k}", 0)
            global_metrics[f"hit_rate_at_{k}"] = scores.get(f"hit_rate_at_{k}", 0)
        # --- 4. Clustering and Intra-Cluster Models ---
        cluster_labels = cluster_users_kmeans(X_train, n_clusters=5, random_state=42)
        clusters = {}
        cluster_metrics = {}
        for label in set(cluster_labels.values()):
            user_ids = [user_id for user_id, cluster_label in cluster_labels.items() if cluster_label == label]
            train_sub = full_train_df[full_train_df["user_id"].isin(user_ids)]
            test_sub = test_df[test_df["user_id"].isin(user_ids)]
            X_train_sub = X_train.loc[user_ids]
            user_features_sub = dataset.build_user_features(
                (user_id, {col: X_train_sub.loc[user_id, col] for col in X_train_sub.columns})
                for user_id in X_train_sub.index
            )
            (test_interactions_sub, _) = dataset.build_interactions(
                [(row["user_id"], row["book_id"]) for _, row in test_sub.iterrows()]
            )
            metrics = {}
            for k in [5, 10, 20]:
                scores = _train_and_evaluate_lightfm(
                    dataset, train_sub, test_interactions_sub, user_features=user_features_sub, k=k
                )
                metrics[f"precision_at_{k}"] = scores.get(f"precision_at_{k}", 0)
                metrics[f"recall_at_{k}"] = scores.get(f"recall_at_{k}", 0)
                metrics[f"hit_rate_at_{k}"] = scores.get(f"hit_rate_at_{k}", 0)
            cluster_metrics[label] = metrics
            clusters[label] = user_ids
        # --- 5. Beyond-Accuracy Metrics ---
        def get_recommendations(model, dataset, user_ids, k):
            # Recommend top-k for each user (returns a sparse matrix)
            recs = {}
            for i, user_id in enumerate(user_ids):
                scores = model.predict(i, np.arange(len(all_items)), user_features=None)
                top_items = np.argsort(-scores)[:k]
                rec_items = [all_items[j] for j in top_items]
                recs[user_id] = rec_items
            return recs
        # Global recommendations for beyond-accuracy
        # (Assume last trained model is global)
        from lightfm import LightFM
        model = LightFM(loss="warp", random_state=42)
        (train_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in full_train_df.iterrows()]
        )
        model.fit(train_interactions, user_features=user_features_train, epochs=5, num_threads=4)
        global_recs = get_recommendations(model, dataset, list(X_test.index), k=10)
        novelty = compute_novelty(global_recs, full_train_df)
        diversity = compute_diversity(global_recs)
        catalog = set(all_items)
        coverage = compute_catalog_coverage(global_recs, catalog)
        global_metrics.update({"novelty": novelty, "diversity": diversity, "catalog_coverage": coverage})
        # Cluster beyond-accuracy
        for label, user_ids in clusters.items():
            recs = get_recommendations(model, dataset, user_ids, k=10)
            cluster_metrics[label]["novelty"] = compute_novelty(recs, full_train_df)
            cluster_metrics[label]["diversity"] = compute_diversity(recs)
            cluster_metrics[label]["catalog_coverage"] = compute_catalog_coverage(recs, catalog)
        # --- 6. Logging and Artifact Saving ---
        self.writer.add_hparams(best_params, global_metrics)
        session_state.set_state("final_evaluation_metrics", {
            "global": global_metrics,
            "clusters": cluster_metrics
        })
        # Save final report
        report = {
            "best_params": best_params,
            "global_metrics": global_metrics,
            "cluster_metrics": cluster_metrics,
        }
        with open(artifacts_dir / "final_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Final evaluation complete. Results and artifacts saved.")
        self.run_count += 1
        self.writer.close()
