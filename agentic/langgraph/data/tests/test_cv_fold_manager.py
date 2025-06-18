import os
import shutil
import pytest
from agentic.langgraph.data.cv_fold_manager import CVFoldManager
import pandas as pd
import numpy as np

def test_cv_fold_manager_save_load():
    test_dir = "data/test_splits"
    os.makedirs(test_dir, exist_ok=True)
    mgr = CVFoldManager(splits_dir=test_dir)
    folds = [
        {"train": ["u1", "u2"], "val": ["u3"], "test": ["u4"]},
        {"train": ["u5"], "val": ["u6"], "test": ["u7"]}
    ]
    mgr.save_folds_json(folds, name="test_splits.json")
    loaded = mgr.load_folds_json(name="test_splits.json")
    assert loaded == folds
    shutil.rmtree(test_dir)


def test_cv_fold_manager_save_load_db():
    test_db = "data/test_cv_folds.duckdb"
    mgr = CVFoldManager(db_path=test_db)
    folds = [
        {"train": ["u1", "u2"], "val": ["u3"], "test": ["u4"]},
        {"train": ["u5"], "val": ["u6"], "test": ["u7"]}
    ]
    mgr.save_folds_db(folds, table_name="test_cv_folds")
    loaded = mgr.load_folds_db(table_name="test_cv_folds", n_folds=2)
    assert loaded == folds
    os.remove(test_db)
def test_generate_interaction_folds():
    # Synthetic data: 3 users, 10 interactions each
    n_users = 3
    n_inter = 10
    user_ids = [f"u{i+1}" for i in range(n_users)]
    data = []
    for u in user_ids:
        for i in range(n_inter):
            data.append({"user_id": u, "interaction_id": f"{u}_int{i+1}", "timestamp": i})
    df = pd.DataFrame(data)
    mgr = CVFoldManager(splits_dir="data/test_interaction_splits")
    folds = mgr.generate_interaction_folds(df, n_folds=5, random_state=42, save=False)
    # Check: All interactions are assigned to exactly one fold
    all_ids = set(df["interaction_id"])
    fold_ids = set()
    for fold in folds:
        fold_ids.update(fold["interaction_id"])
    assert all_ids == fold_ids
    # Check: Each user's interactions are split across folds, each fold has at most ceil(n_inter/5) for that user
    for u in user_ids:
        user_counts = [fold[fold["user_id"] == u].shape[0] for fold in folds]
        assert sum(user_counts) == n_inter
        assert max(user_counts) - min(user_counts) <= 1
    import shutil
    shutil.rmtree("data/test_interaction_splits", ignore_errors=True)

def test_cv_fold_manager_assign_and_lock():

    # Test JSON-based assignment and lock
    test_dir = "data/test_splits2"
    os.makedirs(test_dir, exist_ok=True)
    mgr = CVFoldManager(splits_dir=test_dir)
    folds = [
        {"train": ["u1"], "val": [], "test": []},
        {"train": [], "val": ["u2"], "test": []}
    ]
    mgr.save_folds_json(folds, name="test2.json")
    # Use load_folds_json for JSON-based assign_fold
    loaded = mgr.load_folds_json(name="test2.json")
    assert loaded == folds
    shutil.rmtree(test_dir)

    # Test DB-based assignment and lock
    test_db = "data/test_cv_folds2.duckdb"
    mgr = CVFoldManager(db_path=test_db)
    folds = [
        {"train": ["u1"], "val": [], "test": []},
        {"train": [], "val": ["u2"], "test": []}
    ]
    mgr.save_folds_db(folds, table_name="test_cv_folds2")
    assert mgr.assign_fold("u1", n_folds=2, table_name="test_cv_folds2") == 0
    assert mgr.assign_fold("u2", n_folds=2, table_name="test_cv_folds2") == 1
    assert mgr.assign_fold("u3", n_folds=2, table_name="test_cv_folds2") is None
    assert mgr.lock_fold(0) is True
    assert mgr.lock_fold(0) is False
    mgr.unlock_fold(0)
    os.remove(test_db)
