# tests/evaluation/test_clustering.py
import pandas as pd
from src.evaluation.clustering import cluster_users_kmeans

def test_cluster_users_kmeans_basic():
    # 6 users, 2D features, obvious clusters
    X = pd.DataFrame({
        'f1': [0, 0, 0, 10, 10, 10],
        'f2': [0, 1, 2, 10, 11, 12],
    }, index=[101, 102, 103, 201, 202, 203])
    clusters = cluster_users_kmeans(X, n_clusters=2, random_state=42)
    assert set(clusters.keys()) == set(X.index)
    # Should be 2 clusters
    assert len(set(clusters.values())) == 2
    # Reproducibility
    clusters2 = cluster_users_kmeans(X, n_clusters=2, random_state=42)
    assert clusters == clusters2
