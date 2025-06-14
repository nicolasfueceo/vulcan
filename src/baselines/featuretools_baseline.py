import featuretools as ft
import pandas as pd


def run_featuretools_baseline(
    train_df: pd.DataFrame, books_df: pd.DataFrame, users_df: pd.DataFrame
) -> pd.DataFrame:
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

    # 4. Return & Save
    logger.success("Featuretools baseline finished successfully.")
    return feature_matrix
