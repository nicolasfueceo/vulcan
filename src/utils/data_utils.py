import pandas as pd


def time_based_split(
    df: pd.DataFrame, train_size: float = 0.8
) -> (pd.DataFrame, pd.DataFrame):
    """Splits a DataFrame into training and validation sets based on a timestamp."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * train_size)
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
    return train_df, val_df
