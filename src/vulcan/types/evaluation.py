# Additional evaluation-specific metrics
precision_at_10: Optional[float] = 0.0
recall_at_10: Optional[float] = 0.0
ndcg_at_10: Optional[float] = 0.0
num_clusters: Optional[int] = None
cluster_coverage: Optional[float] = 1.0
improvement_over_baseline: Optional[float] = 0.0

# Global model metrics
global_with_features_p10: Optional[float] = 0.0
improvement_global: Optional[float] = 0.0
