from typing import Any, Dict, Optional

class BaseRecommender:
    """
    Abstract base class for all recommender models.
    Defines the standard interface for fit, predict, score, and (optionally) suggest_params for BO.
    """
    def fit(self, train_data: Any, **kwargs):
        raise NotImplementedError

    def predict(self, user_ids, item_ids=None) -> Any:
        raise NotImplementedError

    def score(self, test_data: Any, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Compute evaluation metrics on test_data. Metrics dict can specify which metrics to compute."""
        raise NotImplementedError

    def suggest_params(self, trial):
        """Suggest hyperparameters for Bayesian Optimization. Not required for all recommenders."""
        raise NotImplementedError
