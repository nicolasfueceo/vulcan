import json

import numpy as np
import pandas as pd
from jsonschema import validate


def assert_json_schema(instance: dict, schema_path: str) -> None:
    """Raises AssertionError if instance doesn't match schema."""
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        validate(instance=instance, schema=schema)
    except Exception as e:
        raise AssertionError(f"JSON schema validation failed: {e}")


def load_test_data(
    n_reviews: int, n_items: int, n_users: int
) -> (pd.DataFrame, pd.DataFrame):
    """Creates a synthetic toy dataset with random ratings, random words."""
    # Create reviews data
    review_data = {
        "user_id": np.random.randint(0, n_users, n_reviews),
        "book_id": np.random.randint(0, n_items, n_reviews),
        "rating": np.random.randint(1, 6, n_reviews),
        "review_text": [
            " ".join(
                np.random.choice(
                    ["good", "bad", "fantasy", "sci-fi", "grimdark"], size=10
                )
            )
            for _ in range(n_reviews)
        ],
        "timestamp": pd.to_datetime(
            np.random.randint(1577836800, 1609459200, n_reviews), unit="s"
        ),
    }
    df_reviews = pd.DataFrame(review_data)

    # Create items data
    item_data = {
        "book_id": np.arange(n_items),
        "author": [f"Author_{i}" for i in range(n_items)],
        "genre": np.random.choice(["Fantasy", "Sci-Fi"], size=n_items),
    }
    df_items = pd.DataFrame(item_data)

    return df_reviews, df_items
