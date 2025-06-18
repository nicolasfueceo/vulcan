import pandas as pd
import numpy as np
from agentic.recommenders.knn import KNNRecommender

def test_knn_recommender():
    data = {
        "user_id": ["u1", "u2", "u3", "u1", "u2"],
        "item_id": ["i1", "i2", "i3", "i2", "i1"],
        "rating": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)
    rec = KNNRecommender(k=2)
    rec.fit(df, user_col="user_id", item_col="item_id", rating_col="rating")
    preds = rec.predict(["u1", "u2"], item_ids=["i1", "i2", "i3"], top_k=2)
    assert len(preds) == 2
    assert all(len(p) == 2 for p in preds)
    metrics = rec.score(df)
    assert "precision@2" in metrics
    assert 0.0 <= metrics["precision@2"] <= 1.0
    print("KNNRecommender test passed.")
