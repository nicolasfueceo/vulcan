import pandas as pd
import numpy as np
from agentic.utils.data_preparer import RecommenderDataPreparer

def test_data_preparer_basic():
    users = ["u1", "u2", "u3"]
    items = ["i1", "i2", "i3"]
    data = {
        "user_id": ["u1", "u2", "u3", "u1"],
        "item_id": ["i1", "i2", "i3", "i2"],
        "rating": [1, 2, 3, 4],
    }
    interactions = pd.DataFrame(data)
    # User/item features
    user_feat = pd.Series([0.1, 0.2, 0.3], index=users, name="uf1")
    item_feat = pd.Series([1.1, 1.2, 1.3], index=items, name="if1")

    prep = RecommenderDataPreparer()
    prep.fit_id_maps(interactions, "user_id", "item_id")
    prep.add_feature(user_feat, scope="user")
    prep.add_feature(item_feat, scope="item")
    train_df, test_df = prep.split_train_test(interactions, "user_id", "item_id", "rating", test_frac=0.5, random_state=0)
    mats = prep.prepare_for_lightfm(train_df, test_df, "user_id", "item_id", "rating")
    assert mats["train"].shape == (3, 3)
    assert mats["test"].shape == (3, 3)
    assert mats["user_features"].shape[0] == 3
    assert mats["item_features"].shape[0] == 3
    print("DataPreparer basic test passed.")
