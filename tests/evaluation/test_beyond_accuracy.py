# tests/evaluation/test_beyond_accuracy.py
import pandas as pd
import numpy as np
from src.evaluation.beyond_accuracy import compute_novelty, compute_diversity, compute_catalog_coverage

def test_compute_novelty():
    train_df = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2, 3],
        'item_id': ['a', 'a', 'b', 'b', 'c', 'c']
    })
    recommendations = {1: ['a', 'b'], 2: ['c'], 3: ['a', 'c']}
    novelty = compute_novelty(recommendations, train_df)
    assert novelty > 0

def test_compute_diversity():
    recommendations = {1: ['a', 'b', 'c'], 2: ['c', 'd']}
    # Proxy test (no item_features)
    diversity = compute_diversity(recommendations)
    assert 0 <= diversity <= 1

def test_compute_catalog_coverage():
    recommendations = {1: ['a', 'b'], 2: ['c'], 3: ['a', 'c']}
    catalog = {'a', 'b', 'c', 'd'}
    coverage = compute_catalog_coverage(recommendations, catalog)
    assert 0 <= coverage <= 1
    assert np.isclose(coverage, 0.75)
