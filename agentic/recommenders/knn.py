from .base import BaseRecommender
import pandas as pd
import numpy as np

class KNNRecommender(BaseRecommender):
    """
    k-Nearest Neighbors recommender for collaborative filtering.
    Can be used for user-user or item-item similarity.
    Supports BO via suggest_params.
    """
    def __init__(self, k=20, similarity='cosine', user_based=True):
        self.k = k
        self.similarity = similarity
        self.user_based = user_based
        self.model = None  # Placeholder for fitted matrix

    def fit(self, train_data: pd.DataFrame, user_col: str = 'user_id', item_col: str = 'item_id', rating_col: str = 'rating'):
        # For demo: create user-item matrix
        self.user_item_matrix = train_data.pivot_table(index=user_col, columns=item_col, values=rating_col, fill_value=0)
        # TODO: Compute similarity matrix here
        self.model = True  # Mark as fitted

    def predict(self, user_ids, item_ids=None, top_k=10):
        # Dummy: recommend random items for now
        if not self.model:
            raise ValueError("Model not fitted")
        all_items = self.user_item_matrix.columns
        return [list(np.random.choice(all_items, size=top_k, replace=False)) for _ in user_ids]

    def score(self, test_data: pd.DataFrame, metrics=None, user_col: str = "user_id", item_col: str = "item_id", k: int = 2):
        # Compute precision@k, recall@k, f1@k using dummy predict()
        users = test_data[user_col].unique()
        precisions, recalls, f1s = [], [], []
        for user in users:
            user_items = set(test_data[test_data[user_col] == user][item_col])
            preds = set(self.predict([user], top_k=k)[0])
            hits = len(preds & user_items)
            prec = hits / k if k > 0 else 0.0
            rec = hits / len(user_items) if len(user_items) > 0 else 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        return {
            f"precision@{k}": float(np.mean(precisions)),
            f"recall@{k}": float(np.mean(recalls)),
            f"f1@{k}": float(np.mean(f1s)),
        }

    def suggest_params(self, trial):
        k = trial.suggest_int('k', 5, 100)
        similarity = trial.suggest_categorical('similarity', ['cosine', 'pearson'])
        user_based = trial.suggest_categorical('user_based', [True, False])
        return {'k': k, 'similarity': similarity, 'user_based': user_based}
