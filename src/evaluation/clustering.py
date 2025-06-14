# src/evaluation/clustering.py
"""
User clustering utility for evaluation (e.g., KMeans).
"""
from typing import Dict, Any
import pandas as pd
from sklearn.cluster import KMeans

def cluster_users_kmeans(X: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> Dict[Any, int]:
    """
    Clusters users via KMeans on their feature vectors.
    Args:
        X: pd.DataFrame, indexed by user_id, user feature matrix
        n_clusters: number of clusters
        random_state: for reproducibility
    Returns:
        Dict mapping user_id to cluster label
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X.values)
    return dict(zip(X.index, labels))
