from agentic.core.metric_logger import MetricLogger
from agentic.langgraph.data.cv_fold_manager import CVFoldManager
from agentic.langgraph.data.data_loader import DataLoader
import numpy as np
import pandas as pd

class EvaluationNode:
    """
    Node for running cross-validation evaluation of a realized feature/model, logging metrics for each round.
    Integrates with CVFoldManager for interaction-level 5CV and MetricLogger for result logging.
    """
    def __init__(self, memory, metric_logger: MetricLogger, db_path: str = "data/goodreads_curated.duckdb", splits_dir: str = "data/splits"):
        self.memory = memory
        self.metric_logger = metric_logger
        self.db_path = db_path
        self.splits_dir = splits_dir
        self.cv_manager = CVFoldManager(splits_dir=splits_dir, db_path=db_path)
        self.data_loader = DataLoader(db_path=db_path)

    def run(self, state: dict):
        """
        Run evaluation for the current realized feature using 5-fold CV.
        Expects state to contain:
            - 'realized_feature': dict or object with 'name' and callable 'feature_fn'
            - 'interactions_df': DataFrame of all interactions (must have user_id, interaction_id, etc.)
            - 'bo_result': dict with best_value, stddev, params, etc. (optional)
        """
        realized_feature = state.get('realized_feature')
        interactions_df = state.get('interactions_df')
        bo_result = state.get('bo_result', {})
        if realized_feature is None or interactions_df is None:
            raise ValueError("State must contain 'realized_feature' and 'interactions_df'")
        # Generate or load 5CV splits
        folds = self.cv_manager.generate_interaction_folds(interactions_df, n_folds=5, save=False)
        metrics_per_fold = []
        for fold_idx, fold_df in enumerate(folds):
            # Placeholder: apply feature_fn and evaluate model here
            # For demo, fake RMSE and NDCG@5
            # Replace this with actual model training and prediction logic
            rmse = np.random.normal(1.0, 0.1)
            ndcg5 = np.random.normal(0.8, 0.05)
            metrics_per_fold.append({'rmse': rmse, 'ndcg@5': ndcg5})
        # Aggregate metrics
        metrics = {}
        stddevs = {}
        for key in metrics_per_fold[0].keys():
            vals = [m[key] for m in metrics_per_fold]
            metrics[key] = float(np.mean(vals))
            stddevs[key] = float(np.std(vals))
        # Log results
        self.metric_logger.log_round(
            feature_name=realized_feature['name'] if isinstance(realized_feature, dict) else str(realized_feature),
            metrics=metrics,
            stddevs=stddevs,
            bo_best_value=bo_result.get('best_value'),
            bo_stddev=bo_result.get('stddev'),
            params=bo_result.get('params'),
            extra={'bo_result': bo_result, 'round': state.get('round')}
        )
        # Add metrics to state for downstream nodes
        state['metrics'] = metrics
        state['metrics_stddev'] = stddevs
        return state

# TODO: Replace placeholder metric computation with actual model training and evaluation logic.
# TODO: Accept and use evaluation metric configuration (e.g., which metrics, k values, etc.) from pipeline config or state.
