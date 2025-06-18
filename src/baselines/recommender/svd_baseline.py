import pandas as pd
from loguru import logger
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse
from sklearn.metrics import ndcg_score
import numpy as np



def run_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k_list=[5, 10, 20]) -> dict:
    """
    Runs the SVD baseline, evaluating with RMSE, MAE.
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

    # 4. Compute NDCG@10
    metrics = {"rmse": rmse_score, "mae": mae_score}
    try:
        # Build user->items mapping from test set
        user_items = test_df.groupby('user_id')['book_id'].apply(list)
        all_items = np.array(train_df['book_id'].unique())
        batch_size = 1000
        ndcg_at_10 = []
        precision_at_5_list = []
        for i in range(0, len(user_items), batch_size):
            batch_user_ids = list(user_items.index[i:i+batch_size])
            batch_user_items = user_items.iloc[i:i+batch_size]
            # Predicted scores for all items
            scores = np.array([model.predict(user_id, item_id).est for user_id in batch_user_ids for item_id in all_items]).reshape(-1, len(all_items))
            # Relevance: 1 if in test set, 0 otherwise
            true_relevance = np.array([np.isin(all_items, np.array(true_items)).astype(int) for true_items in batch_user_items])
            # Compute NDCG for all users in the batch
            batch_ndcg = ndcg_score(true_relevance, scores, k=5)
            ndcg_at_5.extend(batch_ndcg)
            # Compute precision@5 for all users in the batch
            top5_indices = np.argpartition(-scores, 5, axis=1)[:, :5]
            for idx, user_true_items in enumerate(batch_user_items):
                top5_items = all_items[top5_indices[idx]]
                hits = np.isin(top5_items, np.array(user_true_items))
                precision = np.sum(hits) / 5
                precision_at_5_list.append(precision)
        ndcg_at_5 = float(np.mean(ndcg_at_5)) if ndcg_at_5 else float('nan')
        precision_at_5 = float(np.mean(precision_at_5_list)) if precision_at_5_list else float('nan')
        metrics['ndcg_at_5'] = ndcg_at_5
        metrics['precision_at_5'] = precision_at_5
        logger.info(f"SVD NDCG@5: {ndcg_at_5:.4f}, Precision@5: {precision_at_5:.4f}")
    except Exception as ndcg_e:
        logger.warning(f"Could not compute NDCG@10: {ndcg_e}")
        metrics['ndcg_at_10'] = float('nan')

    logger.info(f"SVD metrics: {metrics}")
    logger.success("SVD baseline finished successfully.")
    return metrics
