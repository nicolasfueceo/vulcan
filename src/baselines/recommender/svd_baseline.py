import pandas as pd
from loguru import logger
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse

from src.evaluation.ranking_metrics import evaluate_ranking_metrics
from .ranking_utils import get_top_n_recommendations


def run_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k_list=[5, 10, 20]) -> dict:
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
    top_n = {}
    # Build mapping from inner item id to raw item id
    item_inner_id_to_raw = {iid: model.trainset.to_raw_iid(iid) for iid in range(len(model.qi))}
    for user_id in user_ids:
        try:
            inner_uid = model.trainset.to_inner_uid(user_id)
        except ValueError:
            continue  # user not in training set
        # Get seen items in internal ids
        seen_inner_iids = set()
        for iid in train_df[train_df['user_id'] == user_id]['book_id']:
            if model.trainset.knows_item(iid):
                try:
                    seen_inner_iids.add(model.trainset.to_inner_iid(iid))
                except ValueError:
                    continue
        user_vec = model.pu[inner_uid]
        scores = model.qi @ user_vec  # shape (n_items,)
        # Mask seen items
        if seen_inner_iids:
            scores[list(seen_inner_iids)] = -np.inf
        # Get top-20 indices efficiently
        top_indices = np.argpartition(scores, -20)[-20:]
        top_indices_sorted = top_indices[np.argsort(scores[top_indices])[::-1]]
        # Map back to raw ids
        top_n[user_id] = [item_inner_id_to_raw[iid] for iid in top_indices_sorted]

    ground_truth = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
    ranking_metrics = evaluate_ranking_metrics(top_n, ground_truth, k_list=k_list)
    logger.info(f"SVD baseline ranking metrics: {ranking_metrics}")

    # 5. Return Metrics
    metrics = {"rmse": rmse_score, "mae": mae_score}
    metrics.update(ranking_metrics)
    logger.info(f"SVD metrics: {metrics}")
    logger.success("SVD baseline finished successfully.")
    return metrics
