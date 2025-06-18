import numpy as np
from typing import Tuple, Any, List, Dict
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from loguru import logger
import warnings
warnings.filterwarnings('ignore')


def calculate_precision_gain_reward(p5_feature: float, p5_baseline: float) -> float:
    """
    Reward for LightFM: relative gain in precision@5.
    Returns (p5_feature - p5_baseline) / p5_baseline, or a large value if baseline is 0 and feature > 0.
    """
    if p5_baseline == 0:
        if p5_feature > 0:
            return 100.0
        return 0.0
    return (p5_feature - p5_baseline) / p5_baseline

def calculate_rmse_gain_reward(rmse_feature: float, rmse_baseline: float) -> float:
    """
    Reward for SVD: relative gain in RMSE (lower is better).
    Returns (rmse_baseline - rmse_feature) / rmse_baseline, or 0 if baseline is 0.
    """
    if rmse_baseline == 0:
        return 0.0
    return (rmse_baseline - rmse_feature) / rmse_baseline

def evaluate_feature_with_model(feature_values: pd.Series, train_df: pd.DataFrame, test_df: pd.DataFrame, model_type: str = 'lightfm') -> float:
    """
    Evaluate a feature by training a model and returning the relevant metric.
    For LightFM: returns precision@5
    For SVD: returns RMSE
    For Random Forest: returns RMSE
    """
    try:
        if model_type == 'lightfm':
            from src.baselines.recommender.lightfm_baseline import run_lightfm_baseline
            metrics = run_lightfm_baseline(train_df, test_df)
            return metrics.get('precision_at_5', np.nan)
        elif model_type == 'svd':
            from src.baselines.recommender.svd_baseline import run_svd_baseline
            metrics = run_svd_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        elif model_type == 'random_forest':
            from src.baselines.recommender.random_forest_baseline import run_random_forest_baseline
            metrics = run_random_forest_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        import logging
        logging.error(f"Error evaluating feature with {model_type}: {e}")
        return 0.0


def calculate_scaled_rmse_improvement(rmse_feature: float, rmse_baseline: float, delta_max: float = 0.05) -> float:
    """
    Calculate scaled RMSE improvement normalized to [0, 1].
    
    RmseImprovement = (RMSE_baseline - RMSE_feature) / delta_max
    
    Args:
        ndcg_feature: NDCG score with the feature
        ndcg_baseline: NDCG score of the baseline model
        delta_max: Maximum expected improvement (hyperparameter)
        
    Returns:
        Scaled RecLift score in [0, 1]
    """
    improvement_raw = rmse_baseline - rmse_feature
    improvement_scaled = improvement_raw / delta_max
    return np.clip(improvement_scaled, 0, 1)


def find_optimal_clusters(features: np.ndarray, k_range: range = range(2, 11), 
                         random_state: int = 42) -> Tuple[int, float]:
    """
    Find optimal number of clusters that maximizes silhouette score.
    
    Args:
        features: Feature matrix for clustering
        k_range: Range of k values to try
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal_k, best_silhouette_score)
    """
    if len(features) < 2:
        logger.warning("Not enough samples for clustering")
        return 2, 0.0
        
    best_k = 2
    best_score = -1
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    for k in k_range:
        if k >= len(features):
            continue
            
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k
                
        except Exception as e:
            logger.warning(f"Error with k={k}: {e}")
            continue
    
    return best_k, max(best_score, 0.0)


def calculate_scaled_silhouette(features: np.ndarray, s_min: float = 0.1, s_max: float = 0.6,
                               k_range: range = range(2, 11), random_state: int = 42) -> Tuple[float, int]:
    """
    Calculate scaled Silhouette score normalized to [0, 1] with optimal cluster tuning.
    
    Args:
        features: Feature matrix for clustering
        s_min: Minimum expected silhouette score
        s_max: Maximum expected silhouette score
        k_range: Range of k values to try for optimal clustering
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (scaled_silhouette_score, optimal_k)
    """
    optimal_k, silhouette_raw = find_optimal_clusters(features, k_range, random_state)
    
    # Scale to [0, 1]
    silhouette_scaled = (silhouette_raw - s_min) / (s_max - s_min)
    silhouette_scaled = np.clip(silhouette_scaled, 0, 1)
    
    return silhouette_scaled, optimal_k


def calculate_composite_reward(rmse_feature: float, rmse_baseline: float, features: np.ndarray,
                              w1: float = 0.7, w2: float = 0.3, delta_max: float = 0.05,
                              s_min: float = 0.1, s_max: float = 0.6, 
                              k_range: range = range(2, 11), random_state: int = 42) -> Dict[str, Any]:
    """
    Calculate the composite reward function J = w1 * RecLift_scaled + w2 * Silhouette_scaled.
    
    Args:
        ndcg_feature: NDCG score with the feature
        ndcg_baseline: NDCG score of the baseline model
        features: Feature matrix for clustering
        w1: Weight for RecLift component (default: 0.7)
        w2: Weight for Silhouette component (default: 0.3)
        delta_max: Maximum expected NDCG improvement
        s_min: Minimum expected silhouette score
        s_max: Maximum expected silhouette score
        k_range: Range of k values for clustering
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing all reward components and final score
    """
    # Calculate RMSE improvement component
    rmse_improvement_scaled = calculate_scaled_rmse_improvement(rmse_feature, rmse_baseline, delta_max)
    
    # Calculate Silhouette component
    silhouette_scaled, optimal_k = calculate_scaled_silhouette(
        features, s_min, s_max, k_range, random_state
    )
    
    # Calculate composite reward
    composite_reward = w1 * rmse_improvement_scaled + w2 * silhouette_scaled
    
    return {
        'rmse_improvement_scaled': rmse_improvement_scaled,
        'silhouette_scaled': silhouette_scaled,
        'optimal_k': optimal_k,
        'composite_reward': composite_reward,
        'rmse_feature': rmse_feature,
        'rmse_baseline': rmse_baseline,
        'weights': {'w1': w1, 'w2': w2}
    }


def evaluate_feature_with_model(feature_values: pd.Series, train_df: pd.DataFrame, 
                               test_df: pd.DataFrame, model_type: str = 'lightfm') -> float:
    """
    Evaluate a feature by training a model and calculating RMSE.
    
    Args:
        feature_values: The engineered feature values
        train_df: Training data
        test_df: Test data
        model_type: Type of model to use ('lightfm', 'deepfm', 'popularity')
        
    Returns:
        RMSE score
    """
    try:
        # This is a placeholder - in practice, you'd integrate the feature
        # into your model training pipeline and evaluate
        
        if model_type == 'lightfm':
            from src.baselines.recommender.lightfm_baseline import run_lightfm_baseline
            metrics = run_lightfm_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
            
        elif model_type == 'svd':
            from src.baselines.recommender.svd_baseline import run_svd_baseline
            metrics = run_svd_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        elif model_type == 'deepfm':
            from src.baselines.recommender.deepfm_baseline import run_deepfm_baseline
            metrics = run_deepfm_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
            
        elif model_type == 'random_forest':
            from src.baselines.recommender.random_forest_baseline import run_random_forest_baseline
            metrics = run_random_forest_baseline(train_df, test_df)
            return metrics.get('rmse', np.nan)
        elif model_type == 'popularity':
            from src.baselines.recommender.popularity_baseline import run_popularity_baseline
            result = run_popularity_baseline(train_df, test_df)
            return result.get('rmse', np.nan)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error evaluating feature with {model_type}: {e}")
        return 0.0
