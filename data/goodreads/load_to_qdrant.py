import logging
import sqlite3
from typing import Dict

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPLITS = ["train", "test", "validation"]


class GoodreadsDataLoader:
    def __init__(self):
        """Initialize the data loader with split databases."""
        # Initialize SQL connections for each split
        self.connections: Dict[str, sqlite3.Connection] = {}
        for split in SPLITS:
            db_path = f"data/{split}.db"
            self.connections[split] = sqlite3.connect(db_path)
            logger.info(f"Connected to {db_path}")

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.vector_size = 128  # Reduced vector size

        # Create Qdrant collections
        self.setup_qdrant_collections()

    def drop_all_collections(self):
        """Drop all collections in Qdrant."""
        existing_collections = self.qdrant_client.get_collections().collections
        for collection in existing_collections:
            self.qdrant_client.delete_collection(collection.name)
            logger.info(f"Dropped collection: {collection.name}")
        logger.info("All collections have been dropped")

    def setup_qdrant_collections(self):
        """Create Qdrant collections for books and reviews for each split."""
        distance = Distance.COSINE

        # Delete existing collections if they exist
        existing_collections = self.qdrant_client.get_collections().collections
        for collection in existing_collections:
            self.qdrant_client.delete_collection(collection.name)
            logger.info(f"Deleted existing collection: {collection.name}")

        # Create new collections for each split
        for split in SPLITS:
            for data_type in ["books", "reviews"]:
                collection_name = f"{split}_{data_type}"
                self.qdrant_client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=distance
                    ),
                )
                logger.info(f"Created collection: {collection_name}")

    def load_books(self, split: str) -> pd.DataFrame:
        """Load books data from a specific split, excluding description."""
        query = """
        SELECT 
            book_id,
            title,
            authors,
            average_rating,
            ratings_count,
            num_pages,
            publication_year,
            publisher,
            language_code
        FROM books
        """
        return pd.read_sql_query(query, self.connections[split])

    def load_reviews(self, split: str) -> pd.DataFrame:
        """Load reviews data from a specific split, excluding review_text."""
        query = """
        SELECT 
            review_id,
            user_id,
            book_id,
            rating,
            date_added
        FROM reviews
        """
        return pd.read_sql_query(query, self.connections[split])

    def upload_to_qdrant(self, data: pd.DataFrame, collection_name: str):
        """Upload data to Qdrant."""
        batch_size = 1000  # Reasonable default batch size
        points = []

        logger.info(f"Uploading {len(data)} records to {collection_name}")

        for idx, row in data.iterrows():
            point = models.PointStruct(
                id=idx,
                vector=[0.0] * self.vector_size,  # Placeholder vector
                payload=row.to_dict(),
            )
            points.append(point)

            if len(points) >= batch_size:
                self.qdrant_client.upsert(
                    collection_name=collection_name, points=points
                )
                logger.info(
                    f"Uploaded batch of {batch_size} records to {collection_name}"
                )
                points = []

        if points:
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
            logger.info(
                f"Uploaded final batch of {len(points)} records to {collection_name}"
            )

    def close(self):
        """Close all database connections."""
        for conn in self.connections.values():
            conn.close()
        self.qdrant_client.close()


if __name__ == "__main__":
    loader = GoodreadsDataLoader()

    # Drop all collections first
    #loader.drop_all_collections()

    try:
        for split in SPLITS:
            # Load and upload books
            books_df = loader.load_books(split)
            loader.upload_to_qdrant(books_df, f"{split}_books")
            logger.info(f"Completed uploading books for {split}")

            # Load and upload reviews
            reviews_df = loader.load_reviews(split)
            loader.upload_to_qdrant(reviews_df, f"{split}_reviews")
            logger.info(f"Completed uploading reviews for {split}")

    except Exception as e:
        logger.error(f"Error during data upload: {e}")
    finally:
        loader.close()
        logger.info("Closed all connections")
