# src/agents/optimization_agent.py
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from loguru import logger
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from tensorboardX import SummaryWriter

from src.utils.data_utils import time_based_split
from src.utils.decorators import agent_run_decorator
from src.utils.feature_registry import feature_registry
from src.utils.memory import set_mem
from src.utils.pubsub import acquire_lock, publish, release_lock
from src.utils.tools import sql_tool


class OptimizationAgent:
    """
    An agent responsible for optimizing feature parameters and model hyperparameters.
    """

    def __init__(self):
        logger.info("OptimizationAgent initialized.")
        self.writer = SummaryWriter("runtime/tensorboard")
        self.trial_count = 0
        self.run_count = 0

    def _define_search_space(self) -> list:
        """Defines the search space for the optimization."""
        logger.info("Defining search space...")
        search_space = []

        # Add realized feature parameters to the search space
        all_realized_fns = feature_registry.get_all()
        for feature_name, data in all_realized_fns.items():
            for param_name in data.get("params", []):
                # For simplicity, we assume all code params are real-valued
                # and LLM params have a 'scale' name
                if "scale" in param_name:
                    search_space.append(
                        Real(0.1, 2.0, name=f"{feature_name}__{param_name}")
                    )
                else:
                    search_space.append(
                        Real(0.1, 10.0, name=f"{feature_name}__{param_name}")
                    )

        # Add FM hyperparameters to the search space
        search_space.extend(
            [
                Integer(8, 128, name="fm_n_factors"),
                Real(1e-5, 1.0, "log-uniform", name="fm_reg"),
                Categorical([True, False], name="fm_use_bias"),
            ]
        )

        logger.info(f"...search space defined with {len(search_space)} dimensions.")
        return search_space

    def _objective_function(self, params: list) -> float:
        """The objective function to minimize (e.g., validation RMSE)."""
        logger.info(f"Running trial {self.trial_count} with parameters: {params}")

        # 1. Parse parameters
        param_dict = {dim.name: value for dim, value in zip(self.search_space, params)}

        # 2. Generate features
        all_reviews = sql_tool(
            "SELECT user_id, book_id, rating, timestamp FROM reviews"
        )
        all_books = sql_tool("SELECT book_id, title, description FROM books")

        all_realized_fns = feature_registry.get_all()
        user_features_dfs = []
        item_features_dfs = []

        for name, data in all_realized_fns.items():
            func = data["func"]
            func_params = {
                p: param_dict.get(f"{name}__{p}", 1.0) for p in data.get("params", [])
            }
            # This is a simplification: we assume the function returns a Series
            # and we can determine if it's a user or item feature by its index name
            feature_output = func(all_reviews, all_books, **func_params)
            if feature_output.index.name == "user_id":
                user_features_dfs.append(feature_output.rename(name))
            else:  # assuming item_id
                item_features_dfs.append(feature_output.rename(name))

        user_features = (
            pd.concat(user_features_dfs, axis=1).fillna(0)
            if user_features_dfs
            else None
        )
        item_features = (
            pd.concat(item_features_dfs, axis=1).fillna(0)
            if item_features_dfs
            else None
        )

        # 3. Split data
        train_df, val_df = time_based_split(all_reviews)

        # 4. Train LightFM model
        dataset = Dataset()
        dataset.fit(
            users=all_reviews["user_id"].unique(),
            items=all_reviews["book_id"].unique(),
            user_features=user_features.columns if user_features is not None else None,
            item_features=item_features.columns if item_features is not None else None,
        )

        (train_interactions, _) = dataset.build_interactions(
            (row["user_id"], row["book_id"], row["rating"])
            for _, row in train_df.iterrows()
        )

        user_features_matrix = (
            dataset.build_user_features(
                (
                    user_id,
                    {
                        col: user_features.loc[user_id, col]
                        for col in user_features.columns
                    },
                )
                for user_id in user_features.index
            )
            if user_features is not None
            else None
        )

        item_features_matrix = (
            dataset.build_item_features(
                (
                    item_id,
                    {
                        col: item_features.loc[item_id, col]
                        for col in item_features.columns
                    },
                )
                for item_id in item_features.index
            )
            if item_features is not None
            else None
        )

        model = LightFM(
            no_components=param_dict["fm_n_factors"],
            loss="warp",
            user_alpha=param_dict["fm_reg"],
            item_alpha=param_dict["fm_reg"],
        )
        model.fit(
            train_interactions,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            epochs=10,
            num_threads=4,
        )

        # 5. Evaluate model
        user_id_map, _, item_id_map, _ = dataset.mapping()

        val_user_ids = [user_id_map[u] for u in val_df["user_id"]]
        val_item_ids = [item_id_map[i] for i in val_df["book_id"]]

        predictions = model.predict(
            np.array(val_user_ids),
            np.array(val_item_ids),
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=4,
        )

        val_rmse = np.sqrt(mean_squared_error(val_df["rating"], predictions))

        # 6. Log results
        self.writer.add_scalar("Validation_RMSE", val_rmse, self.trial_count)
        for name, value in param_dict.items():
            if isinstance(value, str) or isinstance(value, bool):
                self.writer.add_text(f"param/{name}", str(value), self.trial_count)
            else:
                self.writer.add_scalar(f"param/{name}", value, self.trial_count)

        self.trial_count += 1
        return val_rmse

    @agent_run_decorator("OptimizationAgent")
    def run(self, message: dict = {}):
        """
        Runs the Bayesian optimization pipeline. Triggered by a pub/sub event.
        """
        lock_name = "lock:OptimizationAgent"
        if not acquire_lock(lock_name):
            logger.info("OptimizationAgent is already running. Skipping.")
            return

        try:
            self.search_space = self._define_search_space()
            if not self.search_space:
                logger.warning("Search space is empty. Skipping optimization.")
                return

            result = gp_minimize(
                self._objective_function,
                self.search_space,
                n_calls=20,
                random_state=42,
            )

            best_params_list = result.x
            best_params = {
                dim.name: value
                for dim, value in zip(self.search_space, best_params_list)
            }
            best_rmse = result.fun

            set_mem("best_params", best_params)
            set_mem("best_rmse", best_rmse)

            self.writer.close()
            logger.info(
                f"OptimizationAgent: Best RMSE: {best_rmse}, Best Params: {best_params}"
            )

            # Publish an event to signal completion
            publish(
                "optimization_done",
                {
                    "status": "success",
                    "best_rmse": best_rmse,
                    "best_params": best_params,
                },
            )
        finally:
            self.run_count += 1
            release_lock(lock_name)
