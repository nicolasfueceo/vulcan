import numpy as np
import scipy.sparse as sp
from agentic.recommenders.lightfm import LightFMRecommender

def make_dummy_data(n_users=5, n_items=8, n_user_features=3, n_item_features=4):
    # Interactions: random binary matrix
    rng = np.random.default_rng(42)
    interactions = sp.csr_matrix(rng.integers(0, 2, size=(n_users, n_items), dtype=np.int32))
    # User features: random binary
    user_features = sp.csr_matrix(rng.integers(0, 2, size=(n_users, n_user_features), dtype=np.int32))
    # Item features: random binary
    item_features = sp.csr_matrix(rng.integers(0, 2, size=(n_items, n_item_features), dtype=np.int32))
    return interactions, user_features, item_features

def test_lightfm_recommender_end_to_end():
    interactions, user_features, item_features = make_dummy_data()
    # Create train/test split with no overlap
    train = interactions.copy().tolil()
    test = interactions.copy().tolil()
    # Hold out 3 entries for user 0 in test only
    heldout = [(0, 0), (0, 1), (0, 2)]
    for u, i in heldout:
        train[u, i] = 0
    test[:, :] = 0  # Zero out all
    for u, i in heldout:
        test[u, i] = interactions[u, i]
    train = train.tocsr()
    test = test.tocsr()
    # Instantiate and fit
    rec = LightFMRecommender(no_components=5, loss='warp', epochs=3, num_threads=1)
    rec.fit(train, user_features=user_features, item_features=item_features)
    # Predict for all users
    user_ids = np.arange(train.shape[0])
    preds = rec.predict(user_ids, user_features=user_features, item_features=item_features, top_k=3)
    assert len(preds) == len(user_ids)
    assert all(len(p) == 3 for p in preds)
    # Score
    metrics = rec.score(test, train_interactions=train, user_features=user_features, item_features=item_features)
    for cutoff in [5, 10]:
        assert f"precision@{cutoff}" in metrics
        assert f"recall@{cutoff}" in metrics
        assert f"f1@{cutoff}" in metrics
        assert 0.0 <= metrics[f"precision@{cutoff}"] <= 1.0
        assert 0.0 <= metrics[f"recall@{cutoff}"] <= 1.0
        assert 0.0 <= metrics[f"f1@{cutoff}"] <= 1.0
