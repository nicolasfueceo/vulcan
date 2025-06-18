from .base import BaseRecommender
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

class LightFMRecommender(BaseRecommender):
    """
    LightFM hybrid recommender supporting user/item features and end-to-end metric evaluation.
    - fit: accepts train_interactions, user_features, item_features (all scipy sparse matrices)
    - predict: recommends top-k items for given users
    - score: computes precision@k and recall@k using LightFM's utilities
    """
    def __init__(self, no_components=30, loss='warp', epochs=10, num_threads=1):
        self.no_components = no_components
        self.loss = loss
        self.epochs = epochs
        self.num_threads = num_threads
        self.model = None
        self.fitted = False
        self.user_features = None
        self.item_features = None

    def fit(self, train_interactions, user_features=None, item_features=None):
        """
        Fit LightFM model. All arguments should be scipy sparse matrices.
        """
        if LightFM is None:
            raise ImportError("LightFM package not installed")
        self.model = LightFM(no_components=self.no_components, loss=self.loss)
        self.model.fit(train_interactions,
                       user_features=user_features,
                       item_features=item_features,
                       epochs=self.epochs,
                       num_threads=self.num_threads)
        self.fitted = True
        self.user_features = user_features
        self.item_features = item_features

    def predict(self, user_ids, item_ids=None, user_features=None, item_features=None, top_k=10):
        """
        Recommend top-k items for each user in user_ids. Uses features if provided, else falls back to those from fit().
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        user_features = user_features if user_features is not None else self.user_features
        item_features = item_features if item_features is not None else self.item_features
        n_items = self.model.item_embeddings.shape[0]
        all_items = np.arange(n_items) if item_ids is None else np.array(item_ids)
        results = []
        for user_id in user_ids:
            user_id_array = np.repeat(user_id, len(all_items))
            scores = self.model.predict(user_id_array, all_items, user_features=user_features, item_features=item_features, num_threads=self.num_threads)
            top_items = all_items[np.argsort(scores)[::-1][:top_k]]
            results.append(list(top_items))
        return results

    def score(self, test_interactions, train_interactions=None, user_features=None, item_features=None, k=None):
        """
        Compute precision@k, recall@k, and F1@k for k=5 and k=10. All interactions/features should be scipy sparse matrices.
        Returns a dict with keys: precision@5, recall@5, f1@5, precision@10, recall@10, f1@10
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        user_features = user_features if user_features is not None else self.user_features
        item_features = item_features if item_features is not None else self.item_features
        metrics = {}
        for cutoff in [5, 10]:
            prec = precision_at_k(self.model, test_interactions,
                                  train_interactions=train_interactions,
                                  user_features=user_features,
                                  item_features=item_features,
                                  k=cutoff, num_threads=self.num_threads)
            rec = recall_at_k(self.model, test_interactions,
                              train_interactions=train_interactions,
                              user_features=user_features,
                              item_features=item_features,
                              k=cutoff, num_threads=self.num_threads)
            p = float(np.mean(prec))
            r = float(np.mean(rec))
            if p + r == 0:
                f1 = 0.0
            else:
                f1 = 2 * p * r / (p + r)
            metrics[f"precision@{cutoff}"] = p
            metrics[f"recall@{cutoff}"] = r
            metrics[f"f1@{cutoff}"] = f1
        return metrics

    def suggest_params(self, trial):
        no_components = trial.suggest_int('no_components', 10, 100)
        loss = trial.suggest_categorical('loss', ['warp', 'bpr', 'warp-kos'])
        epochs = trial.suggest_int('epochs', 5, 50)
        return {'no_components': no_components, 'loss': loss, 'epochs': epochs}
