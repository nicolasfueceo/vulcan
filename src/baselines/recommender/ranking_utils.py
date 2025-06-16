import numpy as np
import pandas as pd


def get_top_n_recommendations(
    predictions_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "book_id",
    rating_col: str = "rating",
    n: int = 10,
) -> dict:
    """
    Get the top-N recommendations for each user from a predictions dataframe.

    Args:
        predictions_df (pd.DataFrame): DataFrame with user, item, and rating columns.
        user_col (str): Name of the user ID column.
        item_col (str): Name of the item ID column.
        rating_col (str): Name of the rating/prediction column.
        n (int): The number of recommendations to output for each user.

    Returns:
        A dict where keys are user IDs and values are lists of tuples:
        [(item ID, estimated rating), ...]
    """
    top_n = {}
    for user_id, group in predictions_df.groupby(user_col):
        top_n[user_id] = list(
            group.nlargest(n, rating_col)[[item_col, rating_col]].itertuples(
                index=False, name=None
            )
        )
    return top_n


def calculate_ndcg(
    recommendations: dict,
    ground_truth: dict,
    k: int = 10,
    batch_size: int = 1000,
) -> float:
    """
    Efficiently calculate mean NDCG@k for a set of recommendations and ground truth using numpy, preserving all samples.
    recommendations: {user_id: [rec1, rec2, ...]}
    ground_truth: {user_id: [item1, item2, ...]}
    batch_size: Number of users to process at once (to avoid OOM)
    """
    import numpy as np
    user_ids = list(recommendations.keys())
    ndcgs = []
    for i in range(0, len(user_ids), batch_size):
        batch_users = user_ids[i:i+batch_size]
        # Precompute log2 denominators
        log2s = np.log2(np.arange(2, k + 2))
        for user_id in batch_users:
            recs = recommendations[user_id][:k]
            gt = set(ground_truth.get(user_id, []))
            if not gt:
                continue
            # DCG: 1/log2(rank+1) for each hit
            hits = np.array([item in gt for item in recs], dtype=np.float32)
            dcg = np.sum(hits / log2s[:len(recs)])
            # Ideal DCG is sum for min(len(gt), k)
            ideal_len = min(len(gt), k)
            ideal_dcg = np.sum(1.0 / log2s[:ideal_len])
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
