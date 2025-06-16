import pandas as pd
from loguru import logger
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse

from src.evaluation.ranking_metrics import evaluate_ranking_metrics
from .ranking_utils import get_top_n_recommendations


def run_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Runs the SVD baseline, evaluating with RMSE, MAE, and NDCG@10.
    """
    logger.info("Starting SVD baseline...")

    # 1. Load Data
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    testset = list(test_df[['user_id', 'book_id', 'rating']].itertuples(index=False, name=None))

    # NOTE: Do not build a full anti-test set (can be huge). Instead, compute top-N recommendations per user in batches.
    # anti_testset = trainset.build_anti_testset()

    # 2. Train Model
    logger.info("Training SVD model...")
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)
    model.fit(trainset)

    # 3. Evaluate for Accuracy (RMSE, MAE)
    logger.info("Evaluating model for accuracy (RMSE, MAE)...")
    accuracy_predictions = model.test(testset)
    rmse_score = rmse(accuracy_predictions, verbose=False)
    mae_score = mae(accuracy_predictions, verbose=False)
    logger.info(f"SVD baseline RMSE: {rmse_score:.4f}, MAE: {mae_score:.4f}")

    # 4. Evaluate for Ranking (NDCG, Precision@K, Recall@K) using RankerEval
    logger.info("Evaluating model for ranking metrics (RankerEval)...")
    import numpy as np
    user_ids = test_df['user_id'].unique()
    item_ids = train_df['book_id'].unique()
    top_n = {}
    for user_id in user_ids:
        seen_items = set(train_df[train_df['user_id'] == user_id]['book_id'])
        candidate_items = [iid for iid in item_ids if iid not in seen_items]
        preds = []
        for book_id in candidate_items:
            try:
                pred = model.predict(user_id, book_id)
                preds.append((book_id, pred.est))
            except Exception:
                continue
        preds.sort(key=lambda x: -x[1])
        top_n[user_id] = [bid for bid, _ in preds[:20]]  # up to 20 for all K
    ground_truth = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
    ranking_metrics = evaluate_ranking_metrics(top_n, ground_truth, k_list=[5, 10, 20])
    logger.info(f"SVD baseline ranking metrics: {ranking_metrics}")

    # 5. Return Metrics
    metrics = {"rmse": rmse_score, "mae": mae_score}
    metrics.update(ranking_metrics)
    logger.info(f"SVD metrics: {metrics}")
    logger.success("SVD baseline finished successfully.")
    return metrics
