import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from threading import Lock
import duckdb

class CVFoldManager:
    """
    Manages CV fold assignments, retrieval, and locking for concurrency safety.
    Supports both JSON and DuckDB storage for folds.
    """
    def __init__(self, splits_dir: str = "data/splits", db_path: str = "data/goodreads_curated.duckdb"):
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.db_path = db_path

    def get_stratified_subsample(self, df, stratify_col, frac, random_state=None):
        """
        Return a stratified subsample of the input DataFrame.
        Args:
            df: Input pandas DataFrame.
            stratify_col: Column name to stratify on.
            frac: Fraction of data to sample (e.g., 0.1 for 10%).
            random_state: Optional random seed for reproducibility.
        Returns:
            Stratified subsample DataFrame.
        """
        import pandas as pd
        if frac >= 1.0:
            return df.copy()
        # Compute sample size per stratum
        stratified = (
            df.groupby(stratify_col, group_keys=False)
            .apply(lambda x: x.sample(frac=frac, random_state=random_state))
        )
        return stratified.reset_index(drop=True)

    def generate_interaction_folds(self, interactions_df, n_folds=5, random_state=None, user_col='user_id', interaction_col='interaction_id', stratify_col=None, save=True, save_name='interaction_folds.json'):
        """
        Generate n-fold cross-validation splits at the interaction level, stratified by user activity.
        Each user's interactions are split across folds so that each fold contains a subset of every user's interactions.
        Args:
            interactions_df: DataFrame with at least [user_id, interaction_id] columns.
            n_folds: Number of folds (default 5).
            random_state: Optional random seed.
            user_col: Name of the user ID column.
            interaction_col: Name of the interaction ID column.
            stratify_col: Optional column to stratify by (e.g., activity bin).
            save: If True, save splits to disk as JSON.
            save_name: Filename for saving splits.
        Returns:
            List of DataFrames, one per fold, each with an added 'fold' column.
        """
        import pandas as pd
        import numpy as np
        rng = np.random.default_rng(random_state)
        interactions_df = interactions_df.copy()
        interactions_df['fold'] = -1
        grouped = interactions_df.groupby(user_col)
        for user, group in grouped:
            idxs = group.index.tolist()
            rng.shuffle(idxs)
            fold_sizes = [len(idxs) // n_folds + (1 if x < len(idxs) % n_folds else 0) for x in range(n_folds)]
            start = 0
            for fold_idx, size in enumerate(fold_sizes):
                for i in idxs[start:start+size]:
                    interactions_df.at[i, 'fold'] = fold_idx
                start += size
        folds = [interactions_df[interactions_df['fold'] == i].reset_index(drop=True) for i in range(n_folds)]
        if save:
            out_path = self.splits_dir / save_name
            folds_json = [df[[user_col, interaction_col, 'fold']].to_dict(orient='records') for df in folds]
            with open(out_path, 'w') as f:
                json.dump(folds_json, f, indent=2)
        return folds

    def save_folds_json(self, folds: List[Dict], name: str = "splits.json"):
        with self.lock:
            out_path = self.splits_dir / name
            with open(out_path, "w") as f:
                json.dump(folds, f, indent=2)

    def load_folds_json(self, name: str = "splits.json") -> Optional[List[Dict]]:
        path = self.splits_dir / name
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def save_folds_db(self, folds: List[Dict], table_name: str = "cv_folds"):
        with self.lock, duckdb.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} (fold INTEGER, split_type VARCHAR, user_id VARCHAR)")
            for fold_idx, fold in enumerate(folds):
                for split_type in ["train", "val", "test"]:
                    for user_id in fold.get(split_type, []):
                        conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?)", [fold_idx, split_type, user_id])

    def load_folds_db(self, table_name: str = "cv_folds", n_folds: int = 5) -> List[Dict]:
        with duckdb.connect(self.db_path) as conn:
            folds = [{"train": [], "val": [], "test": []} for _ in range(n_folds)]
            res = conn.execute(f"SELECT fold, split_type, user_id FROM {table_name}").fetchall()
            for fold_idx, split_type, user_id in res:
                folds[fold_idx][split_type].append(user_id)
            return folds

    def assign_fold(self, user_id: str, n_folds: int = 5, table_name: str = "cv_folds") -> Optional[int]:
        folds = self.load_folds_db(table_name=table_name, n_folds=n_folds)
        for i, fold in enumerate(folds):
            if user_id in fold.get("train", []) or user_id in fold.get("val", []) or user_id in fold.get("test", []):
                return i
        return None

    def lock_fold(self, fold_idx: int) -> bool:
        lockfile = self.splits_dir / f"fold_{fold_idx}.lock"
        if lockfile.exists():
            return False  # Already locked
        lockfile.touch()
        return True

    def unlock_fold(self, fold_idx: int):
        lockfile = self.splits_dir / f"fold_{fold_idx}.lock"
        if lockfile.exists():
            lockfile.unlink()
