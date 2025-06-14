# src/evaluation/beyond_accuracy.py
"""
Metrics for beyond-accuracy evaluation of recommender systems.
Implements novelty, diversity, and catalog coverage.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Any

def compute_novelty(recommendations: Dict[Any, List[Any]], train_df: pd.DataFrame) -> float:
    """
    Novelty: Inverse log-popularity of recommended items (higher is more novel).
    Args:
        recommendations: {user_id: [item_id, ...]}
        train_df: DataFrame with columns ['user_id', 'item_id'] (training interactions)
    Returns:
        Mean novelty across all recommendations.
    """
    item_counts = train_df['item_id'].value_counts().to_dict()
    total_users = train_df['user_id'].nunique()
    novelty_scores = []
    for user, recs in recommendations.items():
        for item in recs:
            pop = item_counts.get(item, 1)
            novelty = -np.log2(pop / total_users)
            novelty_scores.append(novelty)
    return float(np.mean(novelty_scores)) if novelty_scores else 0.0

def compute_diversity(recommendations: Dict[Any, List[Any]], item_features: pd.DataFrame = None) -> float:
    """
    Diversity: Mean pairwise dissimilarity between recommended items (per user, then averaged).
    If item_features is None, uses unique item count as a proxy.
    Args:
        recommendations: {user_id: [item_id, ...]}
        item_features: DataFrame indexed by item_id (optional)
    Returns:
        Mean diversity across users.
    """
    from itertools import combinations
    diversities = []
    for user, recs in recommendations.items():
        if not recs or len(recs) == 1:
            diversities.append(1.0)
            continue
        if item_features is not None:
            feats = item_features.loc[recs].values
            sims = [np.dot(feats[i], feats[j]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[j]) + 1e-8)
                    for i, j in combinations(range(len(recs)), 2)]
            mean_sim = np.mean(sims)
            diversities.append(1 - mean_sim)
        else:
            # Proxy: fraction of unique items
            diversities.append(len(set(recs)) / len(recs))
    return float(np.mean(diversities)) if diversities else 0.0

def compute_catalog_coverage(recommendations: Dict[Any, List[Any]], catalog: Set[Any]) -> float:
    """
    Catalog coverage: Fraction of catalog items recommended to any user.
    Args:
        recommendations: {user_id: [item_id, ...]}
        catalog: Set of all item_ids
    Returns:
        Fraction of unique recommended items over catalog size.
    """
    recommended = set()
    for recs in recommendations.values():
        recommended.update(recs)
    return len(recommended) / len(catalog) if catalog else 0.0
