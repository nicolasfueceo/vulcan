import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, ndcg_score, mean_absolute_error
from loguru import logger

def run_random_forest_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Train a Random Forest regressor to predict ratings and return RMSE.
    Uses all numeric columns except IDs and 'rating'.
    """
    logger.info("Running Random Forest baseline...")

    # Identify feature columns (exclude IDs and target)
    ignore_cols = {'user_id', 'book_id', 'rating'}
    feature_cols = [col for col in train_df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[col])]

    if not feature_cols:
        logger.error("No numeric feature columns found for Random Forest baseline.")
        return {"rmse": np.nan}

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['rating']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['rating']

    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    metrics = {"rmse": rmse, "mae": mae}
    logger.info(f"Random Forest RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # NDCG@10 calculation
    try:
        # Fast NDCG@10: batch predictions for all users at once
        all_items = train_df['book_id'].unique()
        user_ids = test_df['user_id'].unique()
        n_users = len(user_ids)
        n_items = len(all_items)
        # Build user-item true relevance matrix (binary)
        user_item_matrix = np.zeros((n_users, n_items), dtype=int)
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_id_to_idx = {iid: i for i, iid in enumerate(all_items)}
        for row in test_df.itertuples():
            uidx = user_id_to_idx[row.user_id]
            iidx = item_id_to_idx.get(row.book_id, None)
            if iidx is not None:
                user_item_matrix[uidx, iidx] = 1
        # Build feature matrix for each user (mean feature vector)
        user_feat_mat = np.zeros((n_users, len(feature_cols)))
        for i, user_id in enumerate(user_ids):
            feats = test_df[test_df['user_id'] == user_id][feature_cols]
            if not feats.empty:
                user_feat_mat[i] = feats.mean().values
        # Predict scores for all users and items in batch
        all_feats = np.repeat(user_feat_mat, n_items, axis=0)
        all_feats_df = pd.DataFrame(all_feats, columns=feature_cols)
        scores_flat = model.predict(all_feats_df)
        scores_mat = scores_flat.reshape(n_users, n_items)
        # Compute NDCG@10 for all users with at least one relevant item
        ndcg_scores = []
        for u in range(n_users):
            true_rel = user_item_matrix[u]
            if np.sum(true_rel) == 0:
                continue
            ndcg = ndcg_score([true_rel], [scores_mat[u]], k=5)
            ndcg_scores.append(ndcg)
        ndcg_at_5 = float(np.mean(ndcg_scores)) if ndcg_scores else float('nan')
        metrics['ndcg_at_5'] = ndcg_at_5
        logger.info(f"Random Forest NDCG@5: {ndcg_at_5:.4f} (fast batch)")
    except Exception as ndcg_e:
        logger.warning(f"Could not compute NDCG@10: {ndcg_e}")
        metrics['ndcg_at_10'] = float('nan')

    return metrics
