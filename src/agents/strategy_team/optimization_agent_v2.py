"""
Optimization Agent for VULCAN.

This module provides an implementation of the optimization agent that:
1. Uses k-fold cross-validation for robust evaluation
2. Leverages Optuna for efficient Bayesian optimization
3. Implements early stopping and pruning
4. Integrates with VULCAN's feature registry and session state
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np
import optuna
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
from pydantic import BaseModel, Field
from scipy.sparse import coo_matrix, csr_matrix

from src.data.cv_data_manager import CVDataManager
from src.utils.run_utils import get_run_dir, get_run_tensorboard_dir
from src.utils.session_state import SessionState

# Type aliases for better readability
FeatureParams = Dict[str, Any]
TrialResults = List[Dict[str, Any]]
PathLike = Union[str, Path]
T = TypeVar("T")  # For generic type hints


class OptimizationResult(BaseModel):
    """Container for optimization results."""

    best_params: Dict[str, Any] = Field(
        ..., description="Best parameters found during optimization"
    )
    best_score: float = Field(
        ..., description="Best score achieved during optimization", ge=0.0, le=1.0
    )
    trial_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed results from all trials"
    )
    feature_importances: Dict[str, float] = Field(
        default_factory=dict, description="Importance scores for each feature parameter"
    )

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            np.float32: float,
            np.float64: float,
        }


class VULCANOptimizer:
    """Optimization agent for VULCAN feature engineering."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        n_jobs: Optional[int] = None,
        random_state: int = 42,
        session: Optional[SessionState] = None,
        db_path: Union[str, Path] = "data/goodreads_curated.duckdb",
    ) -> None:
        """Initialize the optimizer.

        Args:
            data_dir: Directory containing the data files
            n_jobs: Number of parallel jobs to run (-1 for all CPUs, None for 1)
            random_state: Random seed for reproducibility
            session: Optional session state for tracking experiments
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.session = session or SessionState()
        self.current_trial: Optional[optuna.Trial] = None  # Track the current trial

        # Set up data manager
        self.data_manager = CVDataManager(
            db_path=db_path,
            splits_dir="data/processed/cv_splits",
        )

        # Set up logging
        self.run_dir = get_run_dir()
        self.log_dir = self.run_dir / "optimization_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Set up TensorBoard writer if available
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.writer = SummaryWriter(log_dir=str(get_run_tensorboard_dir() / "optimization"))
        except ImportError as e:
            logger.warning("TensorBoard not available, logging will be limited: %s", str(e))

    def _objective(
        self,
        trial: optuna.Trial,
        features: List[Dict[str, Any]],
        use_fast_mode: bool,
    ) -> float:
        """Objective function for optimization."""
        logger.info(f"--- Starting Trial {trial.number} ---")
        self.current_trial = trial
        trial_number = trial.number
        logger.info(f"Starting trial {trial_number}...")

        try:
            # Ensure CV folds are loaded and get summary
            self.data_manager.load_cv_folds()
            summary = self.data_manager.get_fold_summary()
            n_folds = summary.get("n_folds", 0)
            if n_folds == 0:
                raise ValueError("No CV folds found. Please generate them first.")

            # Sample parameters for this trial
            params = self._sample_parameters(trial, features)

            # Determine sampling for fast mode
            sample_frac = 0.1 if use_fast_mode else None
            logger.info(
                f"Running trial with {n_folds} folds. Fast mode: {use_fast_mode} (sample_frac={sample_frac})"
            )

            fold_scores = []
            for fold_idx in range(n_folds):
                # Get data for the current fold
                fold_data = self.data_manager.get_fold_data(
                    fold_idx=fold_idx,
                    split_type="train_val",
                    sample_frac=sample_frac,
                )
                # Since split_type is 'train_val', we expect a tuple of two dataframes
                if not (isinstance(fold_data, tuple) and len(fold_data) == 2):
                    raise TypeError(f"Expected (train_df, val_df), but got {type(fold_data)}")
                train_df, val_df = fold_data

                # Evaluate on the current fold
                fold_metrics = self._evaluate_fold(
                    fold_idx=fold_idx,
                    train_df=train_df,
                    val_df=val_df,
                    features=features,
                    params=params,
                )
                score = float(fold_metrics["val_score"])
                fold_scores.append(score)

                # Report intermediate score after each fold for pruning
                trial.report(float(np.mean(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(fold_scores) if fold_scores else 0.0
            logger.info(f"Trial {trial.number} -> Average Score: {mean_score:.4f}")
            return float(mean_score)

        except optuna.TrialPruned:
            logger.debug(f"Trial {trial_number} was pruned.")
            raise
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}", exc_info=True)
            # Prune trial if it fails
            raise optuna.exceptions.TrialPruned()
        finally:
            logger.info(f"--- Finished Trial {trial.number} ---")

    def _sample_parameters(self, trial: Trial, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sample parameters for a trial.

        Args:
            trial: Optuna trial object
            features: List of feature configurations

        Returns:
            Dictionary of sampled parameters

        Raises:
            ValueError: If a feature configuration is invalid
            KeyError: If required configuration keys are missing
        """
        params: Dict[str, Any] = {}

        for feature in features:
            try:
                # Store feature name in a variable that will be used
                feature_name = feature["name"]

                # Process each parameter in the feature configuration
                for param_name, param_config in feature.get("parameters", {}).items():
                    full_param_name = f"{feature_name}__{param_name}"
                    param_type = param_config.get("type", "float")

                    if param_type == "int":
                        params[full_param_name] = trial.suggest_int(
                            full_param_name,
                            low=param_config["low"],
                            high=param_config["high"],
                            step=param_config.get("step", 1),
                        )
                    elif param_type == "float":
                        params[full_param_name] = trial.suggest_float(
                            full_param_name,
                            low=param_config.get("low", 0.0),
                            high=param_config.get("high", 1.0),
                            log=param_config.get("log", False),
                        )
                    elif param_type == "categorical":
                        params[full_param_name] = trial.suggest_categorical(
                            full_param_name, choices=param_config["choices"]
                        )
                    else:
                        logger.warning(
                            "Unknown parameter type '%s' for %s", param_type, full_param_name
                        )

            except KeyError as e:
                logger.error(
                    "Missing required configuration for feature %s: %s",
                    feature.get("name", "unknown"),
                    str(e),
                )
                raise
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error sampling parameters for feature %s: %s",
                    feature.get("name", "unknown"),
                    str(e),
                )
                raise ValueError(f"Invalid parameter configuration: {str(e)}") from e

        return params

    def _generate_user_features(
        self,
        df: pd.DataFrame,
        features: List[Dict[str, Any]],
        params: Dict[str, Any],
        user_map: Dict[Any, int],
    ) -> Optional[csr_matrix]:
        """Generate user features matrix for LightFM.

        Args:
            df: DataFrame containing user data
            features: List of feature configurations
            params: Dictionary of parameters for feature generation
            user_map: Dictionary mapping user IDs to indices

        Returns:
            Sparse matrix of user features (n_users x n_features) or None if no features
        """
        if not features:
            return None

        # Generate features using the existing method
        feature_df = self._generate_feature_matrix(df, features, params)

        # Convert to sparse matrix format expected by LightFM
        from scipy.sparse import csr_matrix

        # Create mapping from user_id to feature vector
        user_features = {}
        for user_id, group in df.groupby("user_id"):
            user_idx = user_map[user_id]
            user_features[user_idx] = feature_df.loc[
                group.index[0]
            ].values  # Take first row per user

        # Convert to sparse matrix
        n_users = len(user_map)
        n_features = len(features)

        if not user_features:
            return None

        # Create COO matrix and convert to CSR for LightFM
        rows, cols, data = [], [], []
        for user_idx, feat_vec in user_features.items():
            for feat_idx, val in enumerate(feat_vec):
                rows.append(user_idx)
                cols.append(feat_idx)
                data.append(float(val))

        return csr_matrix((data, (rows, cols)), shape=(n_users, n_features))

    def _evaluate_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train and evaluate a model on a single fold.

        Args:
            fold_idx: Index of the current fold
            train_df: Training data
            val_df: Validation data
            features: List of feature configurations
            params: Dictionary of parameters for the model and features

        Returns:
            Dictionary containing evaluation metrics and parameters
        """
        # Create user and item mappings
        user_ids = {user_id: i for i, user_id in enumerate(train_df["user_id"].unique())}
        item_ids = {item_id: i for i, item_id in enumerate(train_df["item_id"].unique())}

        # Create interaction matrices in COO format
        from scipy.sparse import coo_matrix

        def create_interaction_matrix(df, user_map, item_map):
            # Map user and item IDs to indices
            user_indices = df["user_id"].map(user_map).values
            item_indices = df["item_id"].map(item_map).values
            # Create COO matrix (users x items)
            return coo_matrix(
                (np.ones(len(df)), (user_indices, item_indices)),
                shape=(len(user_map), len(item_map)),
            )

        # Create interaction matrices
        X_train = create_interaction_matrix(train_df, user_ids, item_ids)
        X_val = create_interaction_matrix(
            val_df[val_df["item_id"].isin(item_ids)],  # Only include items seen in training
            user_ids,
            item_ids,
        )

        # Train model with parameters from the trial
        model_params = {
            "loss": "warp",
            "random_state": self.random_state,
            **{k: v for k, v in params.items() if k.startswith("model__")},
        }
        model = LightFM(**model_params)

        # Fit the model
        fit_params = {
            "epochs": params.get("fit__epochs", 30),
            "num_threads": self.n_jobs,
            "verbose": params.get("fit__verbose", False),
        }

        # Generate user features if available
        user_features = None
        if features:
            user_features = self._generate_user_features(train_df, features, params, user_ids)

        try:
            model.fit(interactions=X_train, user_features=user_features, **fit_params)
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            logger.error(f"X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
            logger.error(f"X_train type: {type(X_train)}")
            if user_features is not None:
                logger.error(f"user_features shape: {user_features.shape}")
            raise

        # Evaluate
        val_score = self._evaluate_model(
            model,
            X_val,
            user_features=user_features,  # Pass user features for evaluation
        )

        # Log metrics if writer is available and we have a valid trial number
        trial_number = (
            getattr(self.current_trial, "number", None) if hasattr(self, "current_trial") else None
        )
        if self.writer is not None and trial_number is not None:
            self.writer.add_scalar(f"val/auc_fold_{fold_idx}", val_score, trial_number)

        return {
            "val_score": val_score,
            "params": params,
            "model": model,
            "features": [f["name"] for f in features],
        }

    @staticmethod
    def _generate_feature_matrix(
        df: pd.DataFrame, features: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate feature matrix from input data and parameters.

        Args:
            df: Input DataFrame containing the data
            features: List of feature configurations
            params: Dictionary of parameters for feature generation

        Returns:
            DataFrame with generated features

        Raises:
            RuntimeError: If feature generation fails
        """
        # Initialize empty feature matrix
        feature_matrix = pd.DataFrame(index=df.index)

        # Generate each feature
        for feature in features:
            feature_name = feature.get("name", "unnamed_feature")
            try:
                # Extract feature parameters from the params dict
                feature_params = {
                    k.split("__", 1)[1]: v
                    for k, v in params.items()
                    if k.startswith(f"{feature_name}__")
                }

                # Generate feature using the feature registry
                from src.utils.feature_registry import feature_registry

                feature_data = feature_registry.get(feature_name)
                if feature_data and "func" in feature_data:
                    feature_func = feature_data["func"]
                    if not callable(feature_func):
                        raise TypeError(
                            f"Feature '{feature_name}' in registry is not a callable function."
                        )

                    feature_values = feature_func(df, **feature_params)
                    feature_matrix[feature_name] = feature_values
                else:
                    logger.warning(f"Feature '{feature_name}' not found or invalid in registry.")

            except (ValueError, KeyError) as e:
                logger.warning("Failed to generate feature %s: %s", feature_name, str(e))
            except RuntimeError as e:
                logger.error("Runtime error generating feature %s: %s", feature_name, str(e))

        # If no features were generated, add a dummy feature
        if feature_matrix.empty:
            feature_matrix["dummy_feature"] = 1.0

        return feature_matrix

    @staticmethod
    def _evaluate_model(
        model: LightFM,
        X_val: Union[np.ndarray, coo_matrix],
        user_features: Optional[csr_matrix] = None,
    ) -> float:
        """Evaluate model and return validation score.

        Args:
            model: Trained LightFM model
            X_val: Validation data as sparse COO matrix or numpy array
            user_features: Optional user features as CSR matrix

        Returns:
            AUC score (higher is better)

        Raises:
            ValueError: If model evaluation fails
        """
        try:
            # Calculate AUC score (higher is better)
            auc = auc_score(
                model,
                X_val,
                user_features=user_features,
                num_threads=1,  # Avoid OpenMP issues
            ).mean()
            return float(auc)
        except (ValueError, RuntimeError) as e:
            logger.error("Error in model evaluation: %s", str(e))
            return 0.0

    def optimize(
        self,
        features: List[Dict[str, Any]],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        use_fast_mode: bool = False,
    ) -> OptimizationResult:
        """Run the optimization process.

        Args:
            features: List of feature configurations to optimize
            n_trials: Maximum number of trials to run
            timeout: Maximum time in seconds to run optimization
            use_fast_mode: Whether to use fast mode (subsample data)

        Returns:
            OptimizationResult containing the best parameters and results
        """
        # Set up study
        logger.info(f"🚀 Starting optimization with {n_trials} trials...")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, features, use_fast_mode=use_fast_mode),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Extract results
        best_params = study.best_params
        best_score = study.best_value

        # Get all trial results
        trial_results = [
            {
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state),
            }
            for trial in study.trials
        ]

        # Calculate feature importances (simplified)
        feature_importances = self._calculate_feature_importances(study, features)

        logger.info(f"✅ Optimization finished. Best score: {best_score:.4f}")
        logger.info(f"🏆 Best params: {best_params}")

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trial_results=trial_results,
            feature_importances=feature_importances,
        )
        logger.debug(f"Full optimization result: {result}")
        return result

    @staticmethod
    def _calculate_feature_importances(
        study: optuna.Study,
        features: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate feature importances from optimization results.

        Args:
            study: Optuna study containing trial results
            features: List of feature configurations (unused, kept for future use)

        Returns:
            Dictionary mapping feature names to their importance scores

        Note:
            This is a simplified implementation. In production, consider using
            more sophisticated methods like SHAP values or permutation importance.
        """
        # Calculate importance based on parameter sensitivity across trials
        importances: Dict[str, float] = {}

        # Group parameters by feature
        feature_params: Dict[str, List[str]] = {}
        for param_name in study.best_params:
            feature_name = param_name.split("__")[0]
            if feature_name not in feature_params:
                feature_params[feature_name] = []
            feature_params[feature_name].append(param_name)

        # Calculate importance as the average absolute value of the best parameters
        for feature_name, param_names in feature_params.items():
            param_importance = 0.0
            for param_name in param_names:
                param_value = study.best_params[param_name]
                if isinstance(param_value, (int, float)):
                    param_importance += abs(param_value)
                else:
                    # For non-numeric parameters, use a default importance
                    param_importance += 1.0

            # Average importance across parameters for this feature
            importances[feature_name] = param_importance / max(1, len(param_names))

        return importances


def run_optimization(
    features: List[Dict[str, Any]],
    data_dir: Union[str, Path] = "data",
    n_trials: int = 100,
    timeout: Optional[int] = None,
    use_fast_mode: bool = False,
    n_jobs: Optional[int] = None,
    random_state: int = 42,
) -> OptimizationResult:
    """Run the optimization pipeline.

    Args:
        features: List of feature configurations to optimize
        data_dir: Directory containing the data files
        n_trials: Maximum number of trials to run
        timeout: Maximum time in seconds to run optimization
        use_fast_mode: Whether to use fast mode (subsample data)
        n_jobs: Number of parallel jobs to run (-1 for all CPUs, None for 1)
        random_state: Random seed for reproducibility

    Returns:
        OptimizationResult containing the best parameters and results
    """
    optimizer = VULCANOptimizer(
        data_dir=data_dir,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return optimizer.optimize(
        features=features,
        n_trials=n_trials,
        timeout=timeout,
        use_fast_mode=use_fast_mode,
    )
