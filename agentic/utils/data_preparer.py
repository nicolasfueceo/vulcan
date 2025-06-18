import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Optional, Tuple, Union, List

class RecommenderDataPreparer:
    """
    Utility class for preparing data for recommender systems.
    Handles ID mapping, matrix construction, feature integration, and LightFM formatting.
    """
    def __init__(self):
        self.user_id_map: Dict = {}
        self.item_id_map: Dict = {}
        self.user_features: pd.DataFrame = pd.DataFrame()
        self.item_features: pd.DataFrame = pd.DataFrame()
        self.fitted = False

    def fit_id_maps(self, interactions: pd.DataFrame, user_col: str, item_col: str):
        self.user_id_map = {uid: idx for idx, uid in enumerate(sorted(interactions[user_col].unique()))}
        self.item_id_map = {iid: idx for idx, iid in enumerate(sorted(interactions[item_col].unique()))}

    def map_ids(self, df: pd.DataFrame, user_col: str, item_col: str) -> pd.DataFrame:
        df = df.copy()
        df[user_col] = df[user_col].map(self.user_id_map)
        df[item_col] = df[item_col].map(self.item_id_map)
        return df

    def build_interaction_matrix(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str) -> sp.csr_matrix:
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        rows = df[user_col].values
        cols = df[item_col].values
        data = df[rating_col].values
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_items))
        return mat.tocsr()

    def add_feature(self, feature: pd.Series, scope: str):
        """
        Add a feature (Series) to user or item features. Scope must be 'user' or 'item'.
        """
        if scope == "user":
            if not set(feature.index).issubset(set(self.user_id_map.keys())):
                raise ValueError("User feature index does not match user ids")
            self.user_features[feature.name] = feature.reindex(self.user_id_map.keys()).values
        elif scope == "item":
            if not set(feature.index).issubset(set(self.item_id_map.keys())):
                raise ValueError("Item feature index does not match item ids")
            self.item_features[feature.name] = feature.reindex(self.item_id_map.keys()).values
        else:
            raise ValueError("Scope must be 'user' or 'item'")

    def build_feature_matrix(self, scope: str) -> sp.csr_matrix:
        if scope == "user":
            return sp.csr_matrix(self.user_features.values)
        elif scope == "item":
            return sp.csr_matrix(self.item_features.values)
        else:
            raise ValueError("Scope must be 'user' or 'item'")

    def split_train_test(self, interactions: pd.DataFrame, user_col: str, item_col: str, rating_col: str, test_frac: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple random split of interaction rows into train/test.
        """
        test_idx = interactions.sample(frac=test_frac, random_state=random_state).index
        test_df = interactions.loc[test_idx]
        train_df = interactions.drop(test_idx)
        return train_df, test_df

    def prepare_for_lightfm(self, train_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str, item_col: str, rating_col: str) -> Dict[str, sp.csr_matrix]:
        """
        Returns dict with train, test, user_features, item_features matrices for LightFM.
        """
        train_df = self.map_ids(train_df, user_col, item_col)
        test_df = self.map_ids(test_df, user_col, item_col)
        train_mat = self.build_interaction_matrix(train_df, user_col, item_col, rating_col)
        test_mat = self.build_interaction_matrix(test_df, user_col, item_col, rating_col)
        user_feat_mat = self.build_feature_matrix("user") if not self.user_features.empty else None
        item_feat_mat = self.build_feature_matrix("item") if not self.item_features.empty else None
        return {
            "train": train_mat,
            "test": test_mat,
            "user_features": user_feat_mat,
            "item_features": item_feat_mat,
        }

    def get_user_id_map(self):
        return self.user_id_map

    def get_item_id_map(self):
        return self.item_id_map
