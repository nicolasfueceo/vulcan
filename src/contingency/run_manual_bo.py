import argparse
from pathlib import Path
import pandas as pd
import json
import numpy as np
from src.contingency.reward_functions import calculate_precision_gain_reward, calculate_rmse_gain_reward
from src.contingency.reward_functions import evaluate_feature_with_model
from datetime import datetime
import optuna
from typing import Dict, Any
from src.data.cv_data_manager import CVDataManager
import inspect
import importlib
import importlib

# Dynamically collect all valid feature functions (skip template and private)
def get_all_feature_functions(feature_module):
    feature_funcs = {}
    for name, func in inspect.getmembers(feature_module, inspect.isfunction):
        if name.startswith('_') or name == 'template_feature_function':
            continue
        feature_funcs[name] = func
    return feature_funcs




# --- GLOBAL SCHEMA CACHE ---
_SCHEMA_CACHE = None

# --- COMMON JOIN KEYS ---
_COMMON_JOIN_KEYS = ["user_id", "book_id", "item_id"]

def _get_schema_map(conn):
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    schema = {}
    try:
        tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
        for t in tables:
            try:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchdf()["name"].tolist()
                for c in cols:
                    if c not in schema:
                        schema[c] = []
                    schema[c].append(t)
            except Exception:
                continue
        _SCHEMA_CACHE = schema
    except Exception as e:
        print(f"[WARN] Could not build schema map: {e}")
        schema = {}
    return schema

def prepare_dataframe(depends_on, cv_manager):
    """
    Given a list of column names (not necessarily qualified),
    dynamically map columns to tables and construct a SELECT query.
    Returns None if columns can't be found or joined.
    Always returns the DB connection to the pool.
    """
    conn = cv_manager.db_connection
    schema = _get_schema_map(conn)
    # Map: col -> table
    col_table_map = {}
    ambiguous = False
    for col in depends_on:
        tables = schema.get(col, [])
        if not tables:
            print(f"[WARN] Column '{col}' not found in any table. Skipping feature.")
            ambiguous = True
            break
        elif len(tables) == 1:
            col_table_map[col] = tables[0]
        else:
            # Heuristic: prefer curated_reviews for ratings, curated_books for book info, else first
            preferred = None
            for t in tables:
                if (col == "rating" and "review" in t) or (col in ["title", "description", "avg_rating"] and "book" in t):
                    preferred = t
                    break
            if not preferred:
                preferred = tables[0]
            col_table_map[col] = preferred
    if ambiguous:
        try:
            cv_manager._return_connection(conn)
        except Exception as e:
            print(f"[WARN] Could not return connection to pool: {e}")
        return None
    involved_tables = set(col_table_map.values())
    select_cols = [f"{col_table_map[c]}.{c}" for c in depends_on]
    # If all columns are from one table
    if len(involved_tables) == 1:
        table = list(involved_tables)[0]
        sql = f"SELECT {', '.join(select_cols)} FROM {table} LIMIT 10000"
    else:
        # Try to join on common keys
        join_keys = [k for k in _COMMON_JOIN_KEYS if all(k in schema and t in schema[k] for t in involved_tables)]
        if not join_keys:
            print(f"[WARN] Cannot join tables {involved_tables} for columns {depends_on}: no common key.")
            try:
                cv_manager._return_connection(conn)
            except Exception as e:
                print(f"[WARN] Could not return connection to pool: {e}")
            return None
        # Use the first join key
        key = join_keys[0]
        tables = list(involved_tables)
        sql = f"SELECT {', '.join(select_cols)} FROM {tables[0]}"
        for t in tables[1:]:
            sql += f" JOIN {t} USING ({key})"
        sql += " LIMIT 10000"
    try:
        df = conn.execute(sql).fetchdf()
    except Exception as e:
        print(f"[WARN] Failed to prepare dataframe for depends_on={depends_on}: {e}\nSQL: {sql}")
        df = None
    finally:
        try:
            cv_manager._return_connection(conn)
        except Exception as e:
            print(f"[WARN] Could not return connection to pool: {e}")
    return df

def load_train_test(cv_manager, fold_idx=0):
    """Utility to get train and test DataFrames from CVDataManager."""
    train_df, test_df = cv_manager.get_fold_data(fold_idx=fold_idx, split_type="train_val")
    return train_df.copy(), test_df.copy()

def run_bo_for_feature(feature_dict: Dict[str, Any], cv_manager, baseline_rmse: float, model_type: str, output_base: Path, n_trials=10, fold_idx=0):
    """
    Run Bayesian Optimization for a single feature.
    """
    name = feature_dict['name']
    # Prepare output directory for this feature
    feature_dir = output_base / name
    feature_dir.mkdir(parents=True, exist_ok=True)
    depends_on = feature_dict.get('depends_on', [])
    param_space = feature_dict.get('parameters', {})
    df = prepare_dataframe(depends_on, cv_manager)
    if df is None:
        print(f"[WARN] Skipping feature '{name}' because required columns could not be loaded from DB.")
        return None
    train_df, test_df = load_train_test(cv_manager, fold_idx=fold_idx)

    def objective(trial):
        params = {}
        for param, spec in param_space.items():
            if spec['type'] == 'float':
                params[param] = trial.suggest_float(param, spec['min'], spec['max'])
            elif spec['type'] == 'int':
                params[param] = trial.suggest_int(param, spec['min'], spec['max'])
            elif spec['type'] == 'categorical':
                params[param] = trial.suggest_categorical(param, spec['choices'])
            else:
                raise ValueError(f'Unknown param type: {spec}')
        # Compute feature using the actual function
        feature_func = feature_dict['function']
        feature_col = feature_func(df, params)
        # Append feature to train/test
        train_df_aug = train_df.copy()
        test_df_aug = test_df.copy()
        train_df_aug[name] = feature_col.reindex(train_df_aug.index).fillna(0)
        test_df_aug[name] = feature_col.reindex(test_df_aug.index).fillna(0)

        # Evaluate feature with model and get relevant metric
        eval_metric = evaluate_feature_with_model(feature_col, train_df_aug, test_df_aug, model_type=model_type)
        # Compute reward based on model type
        if model_type == 'lightfm':
            # eval_metric should be precision@5
            reward = calculate_precision_gain_reward(eval_metric, baseline_rmse)  # baseline_rmse is actually baseline_p5 for lightfm
        elif model_type == 'svd':
            # eval_metric should be RMSE
            reward = calculate_rmse_gain_reward(eval_metric, baseline_rmse)
        else:
            raise ValueError(f"Unknown model_type {model_type}")
        return reward

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    # Persist results
    result = {
        "feature": name,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(feature_dir / "bo_result.json", "w") as f:
        json.dump(result, f, indent=4)
    # Optional: save Optuna study for further analysis
    study.trials_dataframe().to_csv(feature_dir / "trials.csv", index=False)
    print(f'Feature: {name} | Best params: {study.best_params} | Best value: {study.best_value}')
    return study.best_params, study.best_value

def main(db_path: str, splits_dir: str, undersample_frac: float, baseline_model: str, baseline_scores_path: str, n_trials: int = 30, use_reliable_features: bool = False):
    # Load baseline RMSE
    with open(baseline_scores_path, 'r') as f:
        baseline_scores = json.load(f)
    if baseline_model not in baseline_scores:
        raise ValueError(f"Baseline model {baseline_model} not found in {baseline_scores_path}")
    if baseline_model == 'popularity':
        raise ValueError("Bayesian Optimization is not run for the 'popularity' baseline as it has no trainable parameters. Choose 'lightfm' or 'deepfm'.")
    baseline_rmse = baseline_scores[baseline_model]['rmse']

    # Initialize CVDataManager
    cv_manager = CVDataManager(
        db_path=db_path,
        splits_dir=splits_dir,
        undersample_frac=undersample_frac,
        read_only=True
    )
    # Base directory for this baseline model
    output_base = Path("experiments") / baseline_model
    output_base.mkdir(parents=True, exist_ok=True)

    # Dynamically discover all feature functions
    if use_reliable_features:
        print("[INFO] Using reliable feature functions from reliable_functions.py")
        feature_module = importlib.import_module("src.contingency.reliable_functions")
    else:
        print("[INFO] Using standard feature functions from functions.py")
        feature_module = importlib.import_module("src.contingency.functions")
    feature_functions = get_all_feature_functions(feature_module)

    # Run BO for each feature function
    for name, func in feature_functions.items():
        # Try to infer depends_on from docstring
        depends_on = []
        docstring = inspect.getdoc(func)
        if docstring and "Required columns:" in docstring:
            lines = docstring.splitlines()
            req_idx = None
            for i, line in enumerate(lines):
                if "Required columns:" in line:
                    req_idx = i
                    break
            if req_idx is not None:
                for l in lines[req_idx+1:]:
                    l = l.strip()
                    # Only accept lines like '- column_name (type)'
                    if l.startswith('- '):
                        try:
                            col_part = l[2:].split(' (')[0]
                            if col_part:
                                depends_on.append(col_part)
                        except Exception:
                            continue
                    elif not l:
                        break
        # Construct feature_dict
        feature_dict = {
            'name': name,
            'function': func,
            'parameters': {},  # default param space
            'depends_on': depends_on
        }
        # Defensive: skip if no depends_on
        if not depends_on:
            print(f"[WARN] Skipping feature '{name}' due to missing depends_on.")
            continue
        try:
            run_bo_for_feature(feature_dict, cv_manager, baseline_rmse, baseline_model, output_base, n_trials=n_trials)
        except Exception as e:
            print(f"[WARN] Skipping feature '{name}' due to error: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Bayesian Optimization for Features")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the DuckDB database file")
    parser.add_argument("--splits_dir", type=str, required=True, help="Path to the CV splits directory")
    parser.add_argument("--undersample_frac", type=float, default=1.0, help="Fraction of data to use for subsampling (e.g., 0.05 for 5%)")
    parser.add_argument("--baseline_model", type=str, required=True, choices=["lightfm", "svd", "random_forest"], help="Which baseline model to optimize for")
    parser.add_argument("--baseline_scores", type=str, default="/root/fuegoRecommender/experiments/baseline_scores.json", help="Path to baseline scores JSON")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of BO trials per feature")
    parser.add_argument("--use-reliable-features", action="store_true", help="Use only reliable, passing features from reliable_functions.py")
    args = parser.parse_args()
    main(args.db_path, args.splits_dir, args.undersample_frac, args.baseline_model, args.baseline_scores, args.n_trials, use_reliable_features=args.use_reliable_features)
