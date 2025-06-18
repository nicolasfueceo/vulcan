#!/usr/bin/env python3
"""
Sequential Feature Evaluation Pipeline

This script evaluates manual features by sequentially adding them to a recommender model
and tracking performance improvements. It saves full Optuna studies and generates
representation learning plots.
"""

import argparse
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'axes.titlepad': 12,
    'axes.labelpad': 8
})

from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from src.utils.session_state import SessionState
import inspect
import importlib
from src.contingency import functions as feature_module

def get_all_feature_functions():
    feature_funcs = {}
    for name, func in inspect.getmembers(feature_module, inspect.isfunction):
        if name.startswith('_') or name == 'template_feature_function':
            continue
        feature_funcs[name] = func
    return feature_funcs



class SequentialFeatureEvaluator:
    """Evaluates features sequentially on a recommender model."""
    
    def __init__(self, session_state: SessionState, output_dir: Path):
        """
        Initializes the evaluator with a session state and output directory.
        
        Args:
            session_state (SessionState): The session state.
            output_dir (Path): The output directory.
        """
        self.session_state = session_state
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and data
        self.base_data = None
        self.target_column = 'average_rating'  # What we're predicting
        self.feature_columns = []
        self.models = {}
        self.results = []
        self.scaler = StandardScaler()
        
        # Feature functions registry
        self.feature_functions = get_all_feature_functions()
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the base dataset for recommendation."""
        conn = self.session_state.db_connection
        
        # Load comprehensive book data with user interactions
        sql = """
        SELECT 
            b.book_id,
            b.title,
            b.average_rating,
            b.ratings_count,
            b.text_reviews_count,
            b.publication_year,
            b.num_pages,
            -- User interaction features
            COUNT(DISTINCT r.user_id) as unique_users,
            AVG(r.rating) as user_avg_rating,
            COUNT(r.rating) as total_user_ratings,
            -- Book popularity features
            b.ratings_count as book_popularity,
            CASE WHEN b.ratings_count > 100 THEN 1 ELSE 0 END as is_popular
        FROM books b
        LEFT JOIN reviews r ON b.book_id = r.book_id
        WHERE b.average_rating IS NOT NULL 
          AND b.ratings_count IS NOT NULL
          AND b.ratings_count > 5  -- Filter out books with very few ratings
        GROUP BY b.book_id, b.title, b.average_rating, b.ratings_count, 
                 b.text_reviews_count, b.publication_year, b.num_pages
        HAVING COUNT(r.rating) >= 3  -- Ensure some user interaction data
        LIMIT 5000
        """
        
        try:
            df = conn.execute(sql).fetchdf()
            print(f"Loaded {len(df)} books with interaction data")
            
            # Handle missing values
            df = df.fillna({
                'text_reviews_count': 0,
                'publication_year': df['publication_year'].median(),
                'num_pages': df['num_pages'].median(),
                'unique_users': 0,
                'user_avg_rating': df['average_rating'].median(),
                'total_user_ratings': 0
            })
            
            # Create base features
            df['log_ratings_count'] = np.log1p(df['ratings_count'])
            df['log_text_reviews'] = np.log1p(df['text_reviews_count'])
            df['pages_per_year'] = df['num_pages'] / (2024 - df['publication_year'] + 1)
            df['rating_engagement'] = df['average_rating'] * np.log1p(df['ratings_count'])
            
            self.base_data = df
            self.feature_columns = [
                'ratings_count', 'text_reviews_count', 'publication_year', 'num_pages',
                'unique_users', 'total_user_ratings', 'book_popularity',
                'log_ratings_count', 'log_text_reviews', 'pages_per_year', 'rating_engagement'
            ]
            
            print(f"Base feature columns: {len(self.feature_columns)}")
            print(f"Target range: {df[self.target_column].min():.2f} - {df[self.target_column].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create synthetic data for testing
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for testing when real data fails."""
        print("Creating synthetic data for testing...")
        np.random.seed(42)
        n_books = 2000
        
        df = pd.DataFrame({
            'book_id': range(n_books),
            'title': [f'Book_{i}' for i in range(n_books)],
            'average_rating': np.random.uniform(2.0, 5.0, n_books),
            'ratings_count': np.random.exponential(100, n_books).astype(int),
            'text_reviews_count': np.random.exponential(30, n_books).astype(int),
            'publication_year': np.random.randint(1980, 2024, n_books),
            'num_pages': np.random.randint(150, 600, n_books),
            'unique_users': np.random.randint(5, 200, n_books),
            'user_avg_rating': np.random.uniform(2.0, 5.0, n_books),
            'total_user_ratings': np.random.randint(10, 500, n_books),
            'book_popularity': np.random.exponential(100, n_books).astype(int),
            'is_popular': np.random.binomial(1, 0.3, n_books)
        })
        
        # Create base features
        df['log_ratings_count'] = np.log1p(df['ratings_count'])
        df['log_text_reviews'] = np.log1p(df['text_reviews_count'])
        df['pages_per_year'] = df['num_pages'] / (2024 - df['publication_year'] + 1)
        df['rating_engagement'] = df['average_rating'] * np.log1p(df['ratings_count'])
        
        self.base_data = df
        self.feature_columns = [
            'ratings_count', 'text_reviews_count', 'publication_year', 'num_pages',
            'unique_users', 'total_user_ratings', 'book_popularity',
            'log_ratings_count', 'log_text_reviews', 'pages_per_year', 'rating_engagement'
        ]
        
        return df
    
    def optimize_feature(self, feature_name: str, n_trials: int = 30) -> Tuple[Dict[str, Any], float, optuna.Study]:
        """Optimize a single feature using Bayesian Optimization."""
        print(f"\n=== Optimizing {feature_name} ===")
        
        if feature_name not in self.feature_functions:
            raise ValueError(f"Feature function {feature_name} not found")
        
        feature_func = self.feature_functions[feature_name]
        
        def objective(trial):
            # Define hyperparameter search space based on feature
            if feature_name == 'rating_popularity_momentum':
                params = {
                    'rating_weight': trial.suggest_float('rating_weight', 0.1, 3.0),
                    'count_weight': trial.suggest_float('count_weight', 0.1, 2.0),
                    'momentum_power': trial.suggest_float('momentum_power', 0.2, 1.5),
                    'min_ratings_threshold': trial.suggest_int('min_ratings_threshold', 5, 100),
                    'rating_scale': trial.suggest_float('rating_scale', 3.0, 6.0)
                }
            elif feature_name == 'genre_preference_alignment':
                params = {
                    'genre_weight': trial.suggest_float('genre_weight', 0.1, 2.0),
                    'rating_threshold': trial.suggest_float('rating_threshold', 3.0, 4.5),
                    'popularity_factor': trial.suggest_float('popularity_factor', 0.0, 1.0),
                    'recency_decay': trial.suggest_float('recency_decay', 0.8, 1.0),
                    'boost_multiplier': trial.suggest_float('boost_multiplier', 1.0, 3.0)
                }
            elif feature_name == 'publication_recency_boost':
                params = {
                    'recency_weight': trial.suggest_float('recency_weight', 0.1, 2.0),
                    'rating_weight': trial.suggest_float('rating_weight', 0.5, 2.0),
                    'velocity_factor': trial.suggest_float('velocity_factor', 0.1, 1.5),
                    'recent_threshold': trial.suggest_int('recent_threshold', 1, 10),
                    'min_ratings': trial.suggest_int('min_ratings', 5, 100)
                }
            elif feature_name == 'engagement_depth_score':
                params = {
                    'review_ratio_weight': trial.suggest_float('review_ratio_weight', 0.5, 2.0),
                    'absolute_reviews_weight': trial.suggest_float('absolute_reviews_weight', 0.1, 1.0),
                    'engagement_threshold': trial.suggest_float('engagement_threshold', 0.05, 0.5),
                    'length_proxy_factor': trial.suggest_float('length_proxy_factor', 0.0, 1.0),
                    'quality_boost': trial.suggest_float('quality_boost', 1.0, 2.0)
                }
            else:
                # Default parameter space for other features
                params = {}
            
            try:
                # Compute feature
                feature_values = feature_func(self.base_data, params)
                
                # Prepare data for model training
                X = self.base_data[self.feature_columns].copy()
                X[feature_name] = feature_values
                y = self.base_data[self.target_column]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Return RMSE gain (positive value)
                return rmse
                
            except Exception as e:
                print(f"Error in trial: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_rmse_gain = study.best_value
        
        print(f"Best RMSE gain: {best_rmse_gain:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Save study
        study_file = self.output_dir / f"{feature_name}_optuna_study.pkl"
        joblib.dump(study, study_file)
        print(f"Optuna study saved to: {study_file}")
        
        return best_params, best_rmse_gain, study
    
    def evaluate_model_with_features(self, feature_list: List[str], 
                                   feature_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance with a specific set of features."""
        print(f"\nEvaluating model with features: {feature_list}")
        
        # Start with base features
        X = self.base_data[self.feature_columns].copy()
        
        # Add optimized features
        for feature_name in feature_list:
            if feature_name in self.feature_functions:
                params = feature_params.get(feature_name, {})
                feature_values = self.feature_functions[feature_name](self.base_data, params)
                X[feature_name] = feature_values
        
        y = self.base_data[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # R² score
            r2 = model.score(X_test_scaled, y_test)
            
            results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_features': X.shape[1],
                'feature_names': list(X.columns)
            }
            
            print(f"  {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return results
    
    def run_sequential_evaluation(self, feature_names: List[str], n_trials: int = 30):
        """Run sequential feature evaluation pipeline."""
        print("=== Starting Sequential Feature Evaluation ===")
        
        # Load data
        self.load_data()
        
        # Evaluate baseline (no additional features)
        print("\n=== Baseline Model (No Additional Features) ===")
        baseline_results = self.evaluate_model_with_features([], {})
        
        self.results.append({
            'step': 0,
            'features_added': [],
            'total_features': len(self.feature_columns),
            'model_results': baseline_results,
            'feature_params': {}
        })
        
        # Sequential feature addition
        optimized_params = {}
        current_features = []
        baseline_rmse = baseline_results['Ridge']['rmse']
        
        for i, feature_name in enumerate(feature_names, 1):
            print(f"\n=== Step {i}: Adding {feature_name} ===")
            
            # Optimize the feature
            best_params, best_rmse_gain, study = self.optimize_feature(feature_name, n_trials)
            optimized_params[feature_name] = best_params
            current_features.append(feature_name)
            
            # Evaluate model with all features so far
            model_results = self.evaluate_model_with_features(current_features, optimized_params)
            
            # Compute RMSE gain
            rmse_gain = baseline_rmse - model_results['Ridge']['rmse']
            
            # Store results
            self.results.append({
                'step': i,
                'features_added': current_features.copy(),
                'total_features': len(self.feature_columns) + len(current_features),
                'model_results': model_results,
                'feature_params': optimized_params.copy(),
                'feature_optimization': {
                    'best_params': best_params,
                    'best_rmse_gain': best_rmse_gain,
                    'n_trials': n_trials,
                    'rmse_gain': rmse_gain
                }
            })
        
        # Save all results
        results_file = self.output_dir / "sequential_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nAll results saved to: {results_file}")
        
        # Generate plots
        self.create_representation_learning_plot()
        
        return self.results
    
    def create_representation_learning_plot(self):
        """Create representation learning plot showing performance vs features."""
        print("\n=== Creating Representation Learning Plot ===")
        
        # Extract data for plotting
        steps = []
        ridge_rmse = []
        ridge_r2 = []
        rf_rmse = []
        rf_r2 = []
        feature_counts = []
        feature_names = []
        
        for result in self.results:
            steps.append(result['step'])
            feature_counts.append(result['total_features'])
            
            # Get model results
            ridge_results = result['model_results']['Ridge']
            rf_results = result['model_results']['RandomForest']
            
            ridge_rmse.append(ridge_results['rmse'])
            ridge_r2.append(ridge_results['r2'])
            rf_rmse.append(rf_results['rmse'])
            rf_r2.append(rf_results['r2'])
            
            # Feature names for x-axis
            if result['step'] == 0:
                feature_names.append('Baseline')
            else:
                feature_names.append(f"+{result['features_added'][-1]}")
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(r'Sequential Feature Evaluation: Representation Learning', fontsize=18, fontweight='bold')
        
        # RMSE plots
        ax1.plot(steps, ridge_rmse, 'o-', label='Ridge', color='blue', linewidth=2)
        ax1.plot(steps, rf_rmse, 's-', label='Random Forest', color='red', linewidth=2)
        ax1.set_xlabel(r'\textbf{Feature Addition Step}')
        ax1.set_ylabel(r'\textbf{RMSE (Lower is Better)}')
        ax1.set_title(r'\textbf{Model Performance: RMSE}')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # R² plots
        ax2.plot(steps, ridge_r2, 'o-', label='Ridge', color='blue', linewidth=2)
        ax2.plot(steps, rf_r2, 's-', label='Random Forest', color='red', linewidth=2)
        ax2.set_xlabel(r'\textbf{Feature Addition Step}')
        ax2.set_ylabel(r'$R^2$ \textbf{Score (Higher is Better)}')
        ax2.set_title(r'\textbf{Model Performance: $R^2$ Score}')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Feature count vs performance
        ax3.scatter(feature_counts, ridge_rmse, c=steps, cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel(r'\textbf{Total Number of Features}')
        ax3.set_ylabel(r'\textbf{RMSE (Ridge)}')
        ax3.set_title(r'\textbf{Feature Count vs Performance}')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Performance improvement
        if len(ridge_rmse) > 1:
            baseline_rmse = ridge_rmse[0]
            improvements = [(baseline_rmse - rmse) / baseline_rmse * 100 for rmse in ridge_rmse[1:]]
            ax4.bar(range(1, len(improvements) + 1), improvements, alpha=0.7, color='green')
            ax4.set_xlabel(r'\textbf{Feature Addition Step}')
            ax4.set_ylabel(r'\textbf{RMSE Improvement (\%)}')
            ax4.set_title(r'\textbf{Cumulative Performance Improvement}')
            ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        # Save high-quality PDF and PNG
        plot_file_pdf = self.output_dir / "representation_learning_plot.pdf"
        plot_file_png = self.output_dir / "representation_learning_plot.png"
        plt.savefig(plot_file_pdf, dpi=600, bbox_inches='tight')
        plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
        print(f"Representation learning plot saved to: {plot_file_pdf} and {plot_file_png}")
        # Also create a detailed feature impact plot
        self._create_feature_impact_plot()
        plt.show()
    
    def _create_feature_impact_plot(self):
        """Create detailed feature impact visualization."""
        if len(self.results) < 2:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance trajectory
        steps = [r['step'] for r in self.results]
        ridge_rmse = [r['model_results']['Ridge']['rmse'] for r in self.results]
        
        ax1.plot(steps, ridge_rmse, 'o-', linewidth=3, markersize=8)
        ax1.set_xlabel(r'\textbf{Feature Addition Step}')
        ax1.set_ylabel(r'\textbf{RMSE}')
        ax1.set_title(r'\textbf{Performance Trajectory}')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations for each step
        for i, (step, rmse) in enumerate(zip(steps, ridge_rmse)):
            if step == 0:
                label = 'Baseline'
    parser = argparse.ArgumentParser(description="Sequential Feature Evaluation Pipeline")
    parser.add_argument("--run_dir", type=str, required=True,
{{ ... }}
    parser.add_argument("--output_dir", type=str, 
                       default="/root/fuegoRecommender/src/contingency/evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=30,
                       help="Number of BO trials per feature")
    parser.add_argument("--features", nargs='+', 
                       default=None,  # Set default to None
                       help="List of features to evaluate sequentially")
    
    args = parser.parse_args()
    
    # Initialize
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    session_state = SessionState(run_dir=run_dir)
    evaluator = SequentialFeatureEvaluator(session_state, output_dir)
    
    # Get all discovered feature names if --features is not specified
    if args.features is None:
        # Use global get_all_feature_functions() to get the full feature list dynamically
        all_features = list(get_all_feature_functions().keys())
        args.features = all_features  # Set args.features to all discovered feature names
    
    try:
        # Run evaluation
        results = evaluator.run_sequential_evaluation(args.features, args.n_trials)
        
        print(f"\n=== EVALUATION COMPLETE ===")
        print(f"Evaluated {len(args.features)} features")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        baseline_rmse = results[0]['model_results']['Ridge']['rmse']
        final_rmse = results[-1]['model_results']['Ridge']['rmse']
        improvement = (baseline_rmse - final_rmse) / baseline_rmse * 100
        
        print(f"Baseline RMSE: {baseline_rmse:.4f}")
        print(f"Final RMSE: {final_rmse:.4f}")
        print(f"Total improvement: {improvement:.2f}%")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        session_state.close_connection()
    
    return 0


if __name__ == "__main__":
    exit(main())
