import featuretools as ft
import pandas as pd


def run_featuretools_baseline(
    train_df: pd.DataFrame, books_df: pd.DataFrame, users_df: pd.DataFrame, test_df: pd.DataFrame = None
) -> dict:
    # Featuretools requires nanosecond precision for datetime columns.
    # Convert all relevant columns to ensure compatibility.
    import logging
    logger = logging.getLogger("featuretools_baseline")
    def clean_datetime_columns(df):
        for col in ["date_added", "date_updated", "read_at", "started_at"]:
            if col in df.columns and str(df[col].dtype).startswith("datetime64"):
                if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
                df[col] = df[col].astype("datetime64[ns]")
        return df

    train_df = clean_datetime_columns(train_df)
    books_df = clean_datetime_columns(books_df)
    users_df = clean_datetime_columns(users_df)
    """
    Runs the Featuretools baseline to generate features for the recommender system.

    This function takes the raw training dataframes, creates a Featuretools EntitySet,
    defines the relationships between them, and then runs Deep Feature Synthesis (DFS)
    to automatically generate a feature matrix.

    Args:
        train_df: DataFrame containing the training interactions (e.g., ratings).
                  Expected columns: ['user_id', 'book_id', 'rating', 'rating_id'].
        books_df: DataFrame containing book metadata.
                  Expected columns: ['book_id', ...].
        users_df: DataFrame containing user metadata.
                  Expected columns: ['user_id', ...].

    Returns:
        A pandas DataFrame containing the generated feature matrix. The matrix will
        have the same index as the input `train_df`.
    """
    logger.info("Starting Featuretools baseline...")

    # 1. Create an EntitySet
    logger.info("Creating EntitySet and adding dataframes...")
    es = ft.EntitySet(id="goodreads_recsys")

    es = es.add_dataframe(
        dataframe_name="ratings",
        dataframe=train_df,
        index="rating_id",
        make_index=True,
        time_index="date_added",
    )

    es = es.add_dataframe(
        dataframe_name="users", dataframe=users_df, index="user_id"
    )

    es = es.add_dataframe(
        dataframe_name="books", dataframe=books_df, index="book_id"
    )

    # 2. Define Relationships
    logger.info("Defining relationships between entities...")
    es = es.add_relationship("users", "user_id", "ratings", "user_id")
    es = es.add_relationship("books", "book_id", "ratings", "book_id")

    # 3. Run Deep Feature Synthesis (DFS)
    logger.info("Running Deep Feature Synthesis (DFS)...")
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="ratings",
        agg_primitives=["mean", "sum", "count", "std", "max", "min", "mode"],
        trans_primitives=["month", "weekday", "time_since_previous"],
        max_depth=2,
        verbose=True,
        n_jobs=-1,  # Use all available cores
    )

    logger.info(f"Featuretools generated {feature_matrix.shape[1]} features.")
    logger.info(f"Shape of the resulting feature matrix: {feature_matrix.shape}")

    # 4. Evaluate with LightFM (if test_df is provided)
    if test_df is not None:
        from lightfm.data import Dataset
        import numpy as np
        from src.evaluation.scoring import _train_and_evaluate_lightfm
        from src.evaluation.beyond_accuracy import compute_novelty, compute_diversity, compute_catalog_coverage
        # Build LightFM dataset
        dataset = Dataset()
        all_users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()
        all_items = pd.concat([train_df["book_id"], test_df["book_id"]]).unique()
        dataset.fit(users=all_users, items=all_items)
        (test_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in test_df.iterrows()]
        )
        user_features_train = dataset.build_user_features(
            (user_id, {col: feature_matrix.loc[user_id, col] for col in feature_matrix.columns})
            for user_id in feature_matrix.index
        )
        metrics = {}
        for k in [5, 10, 20]:
            scores = _train_and_evaluate_lightfm(
                dataset, train_df, test_interactions, user_features=user_features_train, k=k
            )
            metrics[f"precision_at_{k}"] = scores.get(f"precision_at_{k}", 0)
            metrics[f"recall_at_{k}"] = scores.get(f"recall_at_{k}", 0)
            metrics[f"hit_rate_at_{k}"] = scores.get(f"hit_rate_at_{k}", 0)
        # Beyond-accuracy metrics
        from lightfm import LightFM
        model = LightFM(loss="warp", random_state=42)
        (train_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in train_df.iterrows()]
        )
        model.fit(train_interactions, user_features=user_features_train, epochs=5, num_threads=4)
        def get_recommendations(model, dataset, user_ids, k):
            recs = {}
            for i, user_id in enumerate(user_ids):
                scores = model.predict(i, np.arange(len(all_items)), user_features=None)
                top_items = np.argsort(-scores)[:k]
                rec_items = [all_items[j] for j in top_items]
                recs[user_id] = rec_items
            return recs
        global_recs = get_recommendations(model, dataset, list(feature_matrix.index), k=10)
        novelty = compute_novelty(global_recs, train_df)
        diversity = compute_diversity(global_recs)
        catalog = set(all_items)
        coverage = compute_catalog_coverage(global_recs, catalog)
        metrics.update({"novelty": novelty, "diversity": diversity, "catalog_coverage": coverage})
        logger.success(f"Featuretools+LightFM metrics: {metrics}")
        return metrics
    logger.success("Featuretools baseline finished successfully.")
    return feature_matrix
