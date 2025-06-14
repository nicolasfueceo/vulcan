import hashlib
import json
import logging
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from src.schemas.models import RealizedFeature
from src.utils.run_utils import get_run_dir

logger = logging.getLogger(__name__)


def _get_feature_cache_key(feature: RealizedFeature, params: Dict[str, Any]) -> str:
    """Creates a unique cache key for a feature and its parameter values."""
    param_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{feature.name}:{param_string}".encode()).hexdigest()


def _execute_feature_code(
    feature: RealizedFeature, df: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """Executes the code for a single feature and returns the resulting Series."""
    exec_globals = {"pd": pd, "np": np}
    exec(feature.code_str, exec_globals)
    feature_func: Callable = exec_globals[feature.name]

    # Filter for only the params this function expects
    func_params = {k: v for k, v in params.items() if k in feature.params}

    return feature_func(df, **func_params)


def generate_feature_matrix(
    realized_features: List[RealizedFeature],
    df: pd.DataFrame,
    trial_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generates the full feature matrix X by executing or loading from cache.

    Args:
        realized_features: List of realized feature objects.
        df: The input DataFrame (e.g., train or validation split).
        trial_params: The parameter values for the current optimization trial.

    Returns:
        A pandas DataFrame representing the user-feature matrix X.
    """
    cache_dir = get_run_dir() / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    all_feature_series = []

    for feature in realized_features:
        if not feature.passed_test:
            logger.warning(f"Skipping feature '{feature.name}' as it failed tests.")
            continue

        feature_trial_params = {
            p_name: trial_params.get(f"{feature.name}__{p_name}")
            for p_name in feature.params
        }

        cache_key = _get_feature_cache_key(feature, feature_trial_params)
        cache_file = cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            logger.debug(f"Loading feature '{feature.name}' from cache.")
            feature_series = pd.read_parquet(cache_file).squeeze("columns")
        else:
            logger.debug(f"Computing feature '{feature.name}'.")
            try:
                feature_series = _execute_feature_code(
                    feature, df.copy(), feature_trial_params
                )
                feature_series.to_parquet(cache_file)
            except Exception as e:
                logger.error(f"Failed to execute feature '{feature.name}': {e}")
                continue  # Skip this feature if it fails

        all_feature_series.append(feature_series)

    if not all_feature_series:
        logger.warning("No features were successfully generated.")
        return pd.DataFrame(index=df.index)

    # Combine all feature series into a single DataFrame
    X = pd.concat(all_feature_series, axis=1).fillna(0)

    logger.info(f"Generated feature matrix X with shape: {X.shape}")
    return X
