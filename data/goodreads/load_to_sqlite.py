import json
import logging
import os
import sqlite3

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="sqlite_loading.log",
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(console)

# Define the columns we want to keep for each table
BOOKS_COLUMNS = {
    "book_id",
    "title",
    "authors",
    "average_rating",
    "ratings_count",
    "num_pages",
    "publication_year",
    "publisher",
    "language_code",
    "description",
}

REVIEWS_COLUMNS = {
    "review_id",
    "user_id",
    "book_id",
    "rating",
    "review_text",
    "date_added",
}


def peek_json_structure(file_path, num_lines=5):
    """Preview the structure of the JSON file."""
    logger.info(f"Peeking at JSON structure in {file_path}")
    structures = []

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            if line.strip():
                try:
                    item = json.loads(line)
                    structures.append(set(item.keys()))
                    logger.info(f"Sample record {i + 1} columns: {list(item.keys())}")
                except json.JSONDecodeError:
                    logger.warning(f"Couldn't decode line {i + 1}")

    # Find common columns across all samples
    if structures:
        common_columns = set.intersection(*structures)
        logger.info(f"Common columns across samples: {list(common_columns)}")
    else:
        logger.warning("No valid JSON structures found in preview")
        common_columns = set()

    return common_columns


def load_json_to_sqlite(
    file_path,
    table_name,
    db_path="goodreads.db",
    batch_size=10000,
    columns_to_keep=None,
):
    """Load a JSON file directly to SQLite database with specified columns.

    Args:
        file_path: Path to the JSON file
        table_name: Name of the table to create
        db_path: Path to the SQLite database
        batch_size: Number of records to process in each batch
        columns_to_keep: Set of column names to keep (all others will be excluded)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist")

    logger.info(f"Will keep only these columns in {table_name}: {columns_to_keep}")

    conn = sqlite3.connect(db_path)
    logger.info(f"Connected to database: {db_path}")

    # Setup progress tracking
    total_lines = sum(1 for _ in open(file_path, "r"))
    logger.info(f"Processing {total_lines} lines from {file_path}")

    with open(file_path, "r") as f:
        batch = []
        for i, line in enumerate(
            tqdm(f, total=total_lines, desc=f"Loading {table_name}")
        ):
            if line.strip():
                try:
                    item = json.loads(line)

                    # Keep only the specified columns
                    filtered_item = {
                        k: v for k, v in item.items() if k in columns_to_keep
                    }

                    # Convert lists and dicts to JSON strings
                    for key, value in filtered_item.items():
                        if isinstance(value, (list, dict)):
                            filtered_item[key] = json.dumps(value)

                    # Type conversion for specific columns
                    if table_name == "books":
                        # Convert numeric fields
                        for field in [
                            "average_rating",
                            "ratings_count",
                            "num_pages",
                            "publication_year",
                        ]:
                            if field in filtered_item:
                                try:
                                    if field == "average_rating":
                                        filtered_item[field] = float(
                                            filtered_item[field]
                                        )
                                    else:
                                        filtered_item[field] = int(filtered_item[field])
                                except (ValueError, TypeError):
                                    filtered_item[field] = None

                    elif table_name == "reviews":
                        # Convert rating to integer
                        if "rating" in filtered_item:
                            try:
                                filtered_item["rating"] = int(filtered_item["rating"])
                            except (ValueError, TypeError):
                                filtered_item["rating"] = None

                    batch.append(filtered_item)

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {i + 1}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {i + 1}: {e}")
                    continue

            if len(batch) >= batch_size or (i == total_lines - 1 and batch):
                try:
                    # Convert batch to DataFrame
                    df = pd.DataFrame(batch)

                    # Write to SQLite - either create new table or append
                    if i < batch_size:
                        logger.info(f"Creating table {table_name}")
                        df.to_sql(table_name, conn, if_exists="replace", index=False)
                    else:
                        df.to_sql(table_name, conn, if_exists="append", index=False)

                    logger.info(
                        f"Processed batch ending at line {i + 1} - {len(batch)} records"
                    )
                    batch = []  # Clear memory
                except Exception as e:
                    logger.error(f"Error processing batch at line {i + 1}: {e}")
                    raise

    # Create indexes
    logger.info(f"Creating indexes for table {table_name}")

    if table_name == "books":
        conn.execute("CREATE INDEX IF NOT EXISTS idx_books_book_id ON books(book_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_books_title ON books(title)")
    elif table_name == "reviews":
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_book_id ON reviews(book_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_review_id ON reviews(review_id)"
        )

    conn.commit()
    logger.info(f"Successfully loaded table {table_name}")
    return conn


if __name__ == "__main__":
    try:
        # First, remove the existing database if it exists
        if os.path.exists("data/goodreads/goodreads.db"):
            logger.info("Removing existing database...")
            os.remove("data/goodreads/goodreads.db")

        logger.info("Starting data loading process")

        logger.info("Loading books...")
        load_json_to_sqlite(
            "data/goodreads/fantasy_books.json",
            "books",
            db_path="data/goodreads/goodreads.db",
            columns_to_keep=BOOKS_COLUMNS,
        )

        logger.info("Loading reviews...")
        load_json_to_sqlite(
            "data/goodreads/fantasy_reviews.json",
            "reviews",
            db_path="data/goodreads/goodreads.db",
            columns_to_keep=REVIEWS_COLUMNS,
        )

        logger.info("All data successfully loaded to SQLite")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
