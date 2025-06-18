import pandas as pd
import numpy as np
from typing import Dict, Any

# --- RELIABLE, PASSING FEATURE FUNCTIONS ---

def ratings_count_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Raw ratings count (book popularity proxy)
    Required columns:
        - ratings_count (int)
    """
    scale = params.get("scale", 1.0)
    if "ratings_count" not in df.columns:
        raise ValueError("ratings_count_feature requires ratings_count column")
    return pd.Series(df["ratings_count"].fillna(0) * scale, index=df.index, name="ratings_count_feature")

def average_rating_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Raw average rating
    Required columns:
        - avg_rating (float)  # curated_books: avg_rating
    """
    offset = params.get("offset", 0.0)
    if "avg_rating" not in df.columns:
        raise ValueError("average_rating_feature requires avg_rating column")
    return pd.Series(df["avg_rating"].fillna(df["avg_rating"].median()) + offset, index=df.index, name="average_rating_feature")

def num_pages_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Book length (number of pages)
    Required columns:
        - num_pages (int)
    """
    scale = params.get("scale", 1.0)
    if "num_pages" not in df.columns:
        raise ValueError("num_pages_feature requires num_pages column")
    return pd.Series(df["num_pages"].fillna(df["num_pages"].median()) * scale, index=df.index, name="num_pages_feature")

def user_books_read_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Number of books read by user
    Required columns:
        - books_read (int)  # user_reading_trends: books_read
    """
    scale = params.get("scale", 1.0)
    if "books_read" not in df.columns:
        raise ValueError("user_books_read_feature requires books_read column")
    return pd.Series(df["books_read"].fillna(0) * scale, index=df.index, name="user_books_read_feature")

def interaction_rating_feature(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature: Explicit rating from user-item interaction
    Required columns:
        - rating (float)  # interactions: rating
    """
    bias = params.get("bias", 0.0)
    if "rating" not in df.columns:
        raise ValueError("interaction_rating_feature requires rating column")
    return pd.Series(df["rating"].fillna(df["rating"].median()) + bias, index=df.index, name="interaction_rating_feature")
