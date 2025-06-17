import featuretools as ft
import pandas as pd
from loguru import logger 

def run_featuretools_baseline(
    train_df: pd.DataFrame, books_df: pd.DataFrame, users_df: pd.DataFrame, test_df: pd.DataFrame = None, k_list=[5, 10, 20]
) -> dict:
    # Featuretools requires nanosecond precision for datetime columns.
    # Convert all relevant columns to ensure compatibility.
    import logging
    logger = logging.getLogger("featuretools_baseline")
    import pandas as pd
    import numpy as np

    # Robust timestamp filtering utility
    def filter_out_of_bounds_timestamps(df, name, extra_cols=None):
        # Default columns and any extra columns
        timestamp_cols = ["date_added", "date_updated", "read_at", "started_at"]
        if extra_cols:
            timestamp_cols += extra_cols
        timestamp_cols = [col for col in timestamp_cols if col in df.columns]
        if not timestamp_cols:
            return df
        before = len(df)
        # Convert and coerce errors
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Remove timezone if present
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
            # Explicitly cast to datetime64[ns]
            if not pd.api.types.is_datetime64_ns_dtype(df[col]):
                df[col] = df[col].astype('datetime64[ns]')
        # Remove rows with NaT or out-of-bounds
        mask = pd.Series(True, index=df.index)
        for col in timestamp_cols:
            vals = pd.to_datetime(df[col], errors='coerce')
            mask &= (vals >= pd.Timestamp.min) & (vals <= pd.Timestamp.max)
            mask &= vals.notna()
        cleaned = df[mask].copy()
        dropped = before - len(cleaned)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows from {name} due to out-of-bounds or invalid timestamps in columns {timestamp_cols}.")
        return cleaned

    train_df = filter_out_of_bounds_timestamps(train_df, "train_df")
    books_df = filter_out_of_bounds_timestamps(books_df, "books_df")
    users_df = filter_out_of_bounds_timestamps(users_df, "users_df")
    if test_df is not None:
        test_df = filter_out_of_bounds_timestamps(test_df, "test_df")

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

    # Optionally: Save feature matrix for visualization
    try:
        feature_matrix.head(100).to_html("reports/featuretools_feature_matrix_head.html")
        feature_matrix.describe().to_csv("reports/featuretools_feature_matrix_stats.csv")
        logger.info("Featuretools feature matrix head (100 rows) saved to reports/featuretools_feature_matrix_head.html")
        logger.info("Featuretools feature matrix stats saved to reports/featuretools_feature_matrix_stats.csv")
    except Exception as e:
        logger.warning(f"Could not save featuretools feature matrix visualizations: {e}")

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
        # Train LightFM model
        from lightfm import LightFM
        model = LightFM(loss="warp", random_state=42)
        (train_interactions, _) = dataset.build_interactions(
            [(row["user_id"], row["book_id"]) for _, row in train_df.iterrows()]
        )
        model.fit(train_interactions, user_features=user_features_train, epochs=5, num_threads=4)
        # Generate recommendations for all test users (top 20 for all K)
        test_user_ids = test_df["user_id"].unique()
        all_items = pd.concat([train_df["book_id"], test_df["book_id"]]).unique()
        recommendations = {}
        import numpy as np
        for i, user_id in enumerate(test_user_ids):
            scores = model.predict(i, np.arange(len(all_items)), user_features=None)
            top_items = np.argsort(-scores)[:20]
            rec_items = [all_items[j] for j in top_items]
            recommendations[user_id] = rec_items
        ground_truth = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
        from src.evaluation.ranking_metrics import evaluate_ranking_metrics
        ranking_metrics = evaluate_ranking_metrics(recommendations, ground_truth, k_list=k_list)
        metrics = dict(ranking_metrics)
        # Beyond-accuracy metrics
        global_recs = {user_id: recommendations.get(user_id, [])[:10] for user_id in test_user_ids}
        novelty = compute_novelty(global_recs, train_df)
        diversity = compute_diversity(global_recs)
        catalog = set(all_items)
        coverage = compute_catalog_coverage(global_recs, catalog)
        metrics["novelty"] = novelty
        metrics["diversity"] = diversity
        metrics["catalog_coverage"] = coverage
        logger.info(f"Featuretools+LightFM metrics: {metrics}")
        return metrics
    logger.success("Featuretools baseline finished successfully.")
    return feature_matrix
