# ranking_metrics.py: Unified ranking metric evaluation using RankerEval

import numpy as np
from rankereval import Rankings, BinaryLabels, NumericLabels
from rankereval.metrics import NDCG, Precision, Recall, F1, HitRate, FirstRelevantRank
from loguru import logger

def evaluate_ranking_metrics(recommendations, ground_truth, k_list=[5, 10, 20]):
    """
    Compute mean NDCG@k, Precision@k, Recall@k, F1@k, HitRate@k, and FirstRelevantRank using rankerEval.
    Args:
        recommendations: dict {user_id: [item_id1, item_id2, ...]} (ranked list)
        ground_truth: dict {user_id: [item_id1, item_id2, ...]} (relevant items)
        k_list: list of cutoff values for metrics
    Returns:
        metrics: dict with keys like 'ndcg@10', 'precision@5', etc.
    """
    # Only evaluate users present in both dicts
    user_ids = sorted(set(recommendations) & set(ground_truth))
    if not user_ids:
        logger.warning("No overlapping users between recommendations and ground_truth. Returning empty metrics.")
        return {}
    logger.info(f"Starting RankerEval metrics for {len(user_ids)} users, k_list={k_list}...")
    y_pred = [recommendations[u] for u in user_ids]
    y_true = [ground_truth[u] for u in user_ids]

    # Rankings: lists of indices (ranked)
    # BinaryLabels: lists of positive indices
    # For each user, map items in y_pred to indices in y_true, or use global item index mapping
    # We'll use positive indices for ground truth, and ranked indices for predictions
    # Build a global item index mapping
    all_items = set()
    for items in y_pred:
        all_items.update(items)
    for items in y_true:
        all_items.update(items)
    item2idx = {item: idx for idx, item in enumerate(sorted(all_items))}
    # Convert to index-based format
    y_pred_idx = [[item2idx[i] for i in recs] for recs in y_pred]
    y_true_idx = [[item2idx[i] for i in gts] for gts in y_true]

    # Rankings and BinaryLabels objects
    rankings = Rankings.from_ranked_indices(y_pred_idx)
    binary_labels = BinaryLabels.from_positive_indices(y_true_idx)
    # For NDCG, NumericLabels (all 1s for binary relevance)
    numeric_labels = NumericLabels.from_matrix([
        [1 if idx in label else 0 for idx in range(len(item2idx))] for label in y_true_idx
    ])

    metrics = {}
    for k in k_list:
        logger.info(f"Computing metrics for k={k}...")
        # NDCG
        ndcg_scores = NDCG(k=k).score(numeric_labels, rankings)
        metrics[f'ndcg@{k}'] = float(np.nanmean(ndcg_scores))
        logger.debug(f"NDCG@{k} mean: {metrics[f'ndcg@{k}']}")
        # Precision
        precision_scores = Precision(k=k).score(binary_labels, rankings)
        metrics[f'precision@{k}'] = float(np.nanmean(precision_scores))
        logger.debug(f"Precision@{k} mean: {metrics[f'precision@{k}']}")
        # Recall
        recall_scores = Recall(k=k).score(binary_labels, rankings)
        metrics[f'recall@{k}'] = float(np.nanmean(recall_scores))
        logger.debug(f"Recall@{k} mean: {metrics[f'recall@{k}']}")
        # F1
        f1_scores = F1(k=k).score(binary_labels, rankings)
        metrics[f'f1@{k}'] = float(np.nanmean(f1_scores))
        logger.debug(f"F1@{k} mean: {metrics[f'f1@{k}']}")
        # HitRate (only valid if exactly one relevant per user)
        try:
            hitrate_scores = HitRate(k=k).score(binary_labels, rankings)
            metrics[f'hitrate@{k}'] = float(np.nanmean(hitrate_scores))
            logger.debug(f"HitRate@{k} mean: {metrics[f'hitrate@{k}']}")
        except Exception as e:
            logger.warning(f"HitRate@{k} failed: {e}")
    # FirstRelevantRank
    try:
        logger.info("Computing FirstRelevantRank...")
        frr_scores = FirstRelevantRank().score(binary_labels, rankings)
        metrics['first_relevant_rank'] = float(np.nanmean(frr_scores))
        logger.debug(f"FirstRelevantRank mean: {metrics['first_relevant_rank']}")
    except Exception as e:
        logger.warning(f"FirstRelevantRank failed: {e}")
    logger.info("RankerEval metrics computation complete.")
    return metrics
