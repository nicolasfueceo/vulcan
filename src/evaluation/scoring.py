import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


def _train_and_evaluate_lightfm(
    dataset: Dataset,
    train_df: pd.DataFrame,
    test_interactions,
    user_features=None,
    k=10,
    batch_size=100000,
) -> Dict[str, float]:
    """
    Helper to train a LightFM model in batches and score it.
    """
    model = LightFM(loss="warp", random_state=42)

    # Train in batches using fit_partial
    for i in range(0, train_df.shape[0], batch_size):
        chunk = train_df.iloc[i : i + batch_size]
        # Build interactions for the current chunk only
        (chunk_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in chunk.iterrows()]
        )
        model.fit_partial(
            chunk_interactions,
            user_features=user_features,
            epochs=1,  # One pass over each chunk
            num_threads=4,
        )

    # Evaluation logic remains the same
    auc = auc_score(
        model,
        test_interactions,
        user_features=user_features,
        num_threads=4,
    ).mean()

    prec_at_k = precision_at_k(
        model,
        test_interactions,
        k=k,
        user_features=user_features,
        num_threads=4,
    ).mean()

    recall_at_k_scores = recall_at_k(
        model,
        test_interactions,
        k=k,
        user_features=user_features,
        num_threads=4,
    )
    recall_at_k_mean = recall_at_k_scores.mean()
    hit_rate_at_k = np.mean(recall_at_k_scores > 0)

    return {
        "auc": auc,
        f"precision_at_{k}": prec_at_k,
        f"recall_at_{k}": recall_at_k_mean,
        f"hit_rate_at_{k}": hit_rate_at_k,
    }


def score_trial(
    X_val: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    weights: Dict[str, float] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Evaluates a feature matrix (X) by training a LightFM model in batches.
    """
    if weights is None:
        weights = {"auc": 0.6, "precision": 0.2, "recall": 0.2}

    # 1. Build dataset mapping and test interactions
    dataset = Dataset()
    all_users = pd.concat([train_df["user_id"], val_df["user_id"]]).unique()
    all_items = pd.concat([train_df["book_id"], val_df["book_id"]]).unique()
    dataset.fit(users=all_users, items=all_items)

    (test_interactions, _) = dataset.build_interactions(
        [(row["user_id"], row["book_id"]) for _, row in val_df.iterrows()]
    )

    # 2. Build user features sparse matrix
    user_features = dataset.build_user_features(
        (user_id, {col: X_val.loc[user_id, col] for col in X_val.columns})
        for user_id in X_val.index
    )

    # 3. Train (in batches) and evaluate the model
    scores = _train_and_evaluate_lightfm(
        dataset, train_df, test_interactions, user_features=user_features
    )

    # 4. Calculate final objective
    final_objective = -(
        weights["auc"] * scores.get("auc", 0)
        + weights["precision"] * scores.get("precision_at_10", 0)
        + weights["recall"] * scores.get("recall_at_10", 0)
    )

    logger.info(f"Trial scores: {scores} -> Final objective: {final_objective:.4f}")
    return scores, final_objective
