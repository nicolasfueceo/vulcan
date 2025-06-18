from .base import BaseRecommender
import pandas as pd
import numpy as np

class PopularityRecommender(BaseRecommender):
    """
    Simple popularity-based recommender: ranks items by overall popularity (e.g., count of interactions).
    No hyperparameters or BO integration.
    """
    def __init__(self):
        self.item_scores = pd.Series(dtype=float)
        self.fitted = False

    def fit(self, train_data: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating"):
        # Compute item popularity
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.item_scores = train_data.groupby(item_col)[rating_col].sum().sort_values(ascending=False)
        self.item_scores.name = "popularity"
        self.fitted = True

    def predict(self, user_ids, item_ids=None, top_k=10):
        if not self.fitted:
            raise ValueError("Model not fitted")
        if item_ids is not None:
            scores = self.item_scores[self.item_scores.index.isin(item_ids)]
        else:
            scores = self.item_scores
        return [list(scores.index[:top_k])] * len(user_ids)

    def score(self, test_data: pd.DataFrame, metrics=None, item_col: str = "item_id", user_col: str = "user_id", k: int = 2):
        # Compute precision@k, recall@k, f1@k, and coverage
        if self.item_scores is None:
            raise ValueError("Model not fitted")
        recommended = list(self.item_scores.index[:k])
        users = test_data[user_col].unique()
        precisions, recalls, f1s = [], [], []
        for user in users:
            user_items = set(test_data[test_data[user_col] == user][item_col])
            hits = len(set(recommended) & user_items)
            prec = hits / k if k > 0 else 0.0
            rec = hits / len(user_items) if len(user_items) > 0 else 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        coverage = len(set(recommended) & set(test_data[item_col])) / len(set(test_data[item_col]))
        return {
            f"precision@{k}": float(np.mean(precisions)),
            f"recall@{k}": float(np.mean(recalls)),
            f"f1@{k}": float(np.mean(f1s)),
            "coverage": coverage,
        }

    def suggest_params(self, trial):
        raise NotImplementedError("Popularity recommender does not support BO")
