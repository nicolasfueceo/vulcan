#!/usr/bin/env python3
"""
Test script for manual feature functions with Bayesian Optimization

This script tests our manually implemented feature functions using BO
to optimize their hyperparameters on real data.
"""

import argparse
from pathlib import Path
import pandas as pd
import optuna
import numpy as np
from typing import Dict, Any
from src.utils.session_state import SessionState
from src.contingency.functions import rating_popularity_momentum


def load_books_data(session_state: SessionState) -> pd.DataFrame:
    """Load books data with ratings information."""
    conn = session_state.db_connection
    
    # Get books with their ratings data
    sql = """
    SELECT 
        b.book_id,
        b.title,
        b.average_rating,
        b.ratings_count,
        b.text_reviews_count,
        b.publication_year,
        b.num_pages
    FROM books b
    WHERE b.average_rating IS NOT NULL 
      AND b.ratings_count IS NOT NULL
      AND b.ratings_count > 0
    LIMIT 10000
    """
    
    try:
        df = conn.execute(sql).fetchdf()
        print(f"Loaded {len(df)} books with rating data")
        print(f"Average rating range: {df['average_rating'].min():.2f} - {df['average_rating'].max():.2f}")
        print(f"Ratings count range: {df['ratings_count'].min()} - {df['ratings_count'].max()}")
        return df
    except Exception as e:
        print(f"Error loading books data: {e}")
        # Fallback: create synthetic data for testing
        print("Creating synthetic data for testing...")
        np.random.seed(42)
        n_books = 1000
        synthetic_df = pd.DataFrame({
            'book_id': range(n_books),
            'title': [f'Book_{i}' for i in range(n_books)],
            'average_rating': np.random.uniform(1.0, 5.0, n_books),
            'ratings_count': np.random.exponential(50, n_books).astype(int),
            'text_reviews_count': np.random.exponential(20, n_books).astype(int),
            'publication_year': np.random.randint(1950, 2024, n_books),
            'num_pages': np.random.randint(100, 800, n_books)
        })
        return synthetic_df


def test_rating_popularity_momentum(df: pd.DataFrame, n_trials: int = 20):
    """Test the rating_popularity_momentum feature with Bayesian Optimization."""
    print(f"\n=== Testing rating_popularity_momentum feature ===")
    print(f"Data shape: {df.shape}")
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'rating_weight': trial.suggest_float('rating_weight', 0.1, 2.0),
            'count_weight': trial.suggest_float('count_weight', 0.1, 1.5),
            'momentum_power': trial.suggest_float('momentum_power', 0.3, 1.2),
            'min_ratings_threshold': trial.suggest_int('min_ratings_threshold', 5, 50),
            'rating_scale': trial.suggest_float('rating_scale', 4.0, 6.0)
        }
        
        try:
            # Compute feature
            feature_values = rating_popularity_momentum(df, params)
            
            # Evaluation metric: we want features that correlate well with actual popularity
            # Use ratings_count as proxy for true popularity
            correlation = np.corrcoef(feature_values, df['ratings_count'])[0, 1]
            
            # Handle NaN correlation (can happen if feature is constant)
            if np.isnan(correlation):
                return -1.0
                
            # We want high positive correlation
            return correlation
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return -1.0
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"\nBest correlation: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test best feature
    best_feature = rating_popularity_momentum(df, study.best_params)
    print(f"\nFeature statistics:")
    print(f"  Mean: {best_feature.mean():.4f}")
    print(f"  Std: {best_feature.std():.4f}")
    print(f"  Min: {best_feature.min():.4f}")
    print(f"  Max: {best_feature.max():.4f}")
    print(f"  Non-zero values: {(best_feature > 0).sum()}/{len(best_feature)}")
    
    # Correlation analysis
    correlations = {
        'ratings_count': np.corrcoef(best_feature, df['ratings_count'])[0, 1],
        'average_rating': np.corrcoef(best_feature, df['average_rating'])[0, 1],
        'text_reviews_count': np.corrcoef(best_feature, df['text_reviews_count'])[0, 1] if 'text_reviews_count' in df.columns else None
    }
    
    print(f"\nCorrelations with other variables:")
    for var, corr in correlations.items():
        if corr is not None:
            print(f"  {var}: {corr:.4f}")
    
    return study.best_params, study.best_value, best_feature


def main():
    parser = argparse.ArgumentParser(description="Test manual feature functions with BO")
    parser.add_argument("--run_dir", type=str, required=True, 
                       help="Path to run directory with session_state.json")
    parser.add_argument("--n_trials", type=int, default=20, 
                       help="Number of BO trials")
    
    args = parser.parse_args()
    
    # Initialize session state
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
        
    session_state = SessionState(run_dir=run_dir)
    
    try:
        # Load data
        df = load_books_data(session_state)
        
        if df.empty:
            print("No data loaded, cannot proceed")
            return 1
            
        # Test the feature
        best_params, best_score, feature_values = test_rating_popularity_momentum(df, args.n_trials)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Best correlation score: {best_score:.4f}")
        print(f"Optimized parameters: {best_params}")
        
        # Save results
        results_file = run_dir / "manual_feature_results.json"
        import json
        results = {
            "feature_name": "rating_popularity_momentum",
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": args.n_trials,
            "data_shape": df.shape,
            "timestamp": "2025-06-17T11:33:52+02:00"
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        session_state.close_connection()
    
    return 0


if __name__ == "__main__":
    exit(main())
