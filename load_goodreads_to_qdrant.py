#!/usr/bin/env python3
"""
Streamlined script for loading Goodreads data, processing it, and loading it into Qdrant.

This script handles the complete workflow:
1. Loading the Goodreads Fantasy & Paranormal dataset
2. Processing and cleaning the data
3. Creating embeddings
4. Loading the data into Qdrant for efficient similarity search

Usage:
    python load_goodreads_to_qdrant.py --interactions_path <path> --books_path <path> --reviews_path <path>
"""

import argparse
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GoodreadsProcessor:
    """
    Processor for Goodreads Fantasy & Paranormal dataset.

    This class handles loading, cleaning, and processing the Goodreads dataset
    to prepare it for recommendation tasks.
    """

    def __init__(self, output_dir: str = "./processed_data"):
        """
        Initialize the Goodreads processor.

        Args:
            output_dir (str, optional): Directory to save processed data. Defaults to "./processed_data".
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"Goodreads processor initialized with output_dir: {self.output_dir}"
        )

    def load_interactions(self, file_path: str) -> pd.DataFrame:
        """
        Load Goodreads interactions data from a gzipped JSON file.

        Args:
            file_path (str): Path to the interactions file.

        Returns:
            pd.DataFrame: DataFrame containing interactions data.
        """
        logger.info(f"Loading interactions from {file_path}")

        # Read gzipped JSON file line by line
        interactions = []
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading interactions"):
                try:
                    interaction = json.loads(line)
                    interactions.append(interaction)
                except json.JSONDecodeError:
                    logger.warning(f"Error decoding JSON line in {file_path}")

        # Convert to DataFrame
        df = pd.DataFrame(interactions)

        logger.info(f"Loaded {len(df)} interactions")
        return df

    def load_books(self, file_path: str) -> pd.DataFrame:
        """
        Load Goodreads books data from a gzipped JSON file.

        Args:
            file_path (str): Path to the books file.

        Returns:
            pd.DataFrame: DataFrame containing books data.
        """
        logger.info(f"Loading books from {file_path}")

        # Read gzipped JSON file line by line
        books = []
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading books"):
                try:
                    book = json.loads(line)
                    books.append(book)
                except json.JSONDecodeError:
                    logger.warning(f"Error decoding JSON line in {file_path}")

        # Convert to DataFrame
        df = pd.DataFrame(books)

        logger.info(f"Loaded {len(df)} books")
        return df

    def load_reviews(self, file_path: str) -> pd.DataFrame:
        """
        Load Goodreads reviews data from a gzipped JSON file.

        Args:
            file_path (str): Path to the reviews file.

        Returns:
            pd.DataFrame: DataFrame containing reviews data.
        """
        logger.info(f"Loading reviews from {file_path}")

        # Read gzipped JSON file line by line
        reviews = []
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading reviews"):
                try:
                    review = json.loads(line)
                    reviews.append(review)
                except json.JSONDecodeError:
                    logger.warning(f"Error decoding JSON line in {file_path}")

        # Convert to DataFrame
        df = pd.DataFrame(reviews)

        logger.info(f"Loaded {len(df)} reviews")
        return df

    def clean_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess interactions data.

        Args:
            interactions_df (pd.DataFrame): Raw interactions DataFrame.

        Returns:
            pd.DataFrame: Cleaned interactions DataFrame.
        """
        logger.info("Cleaning interactions data")

        # Make a copy to avoid modifying the original
        df = interactions_df.copy()

        # Extract relevant columns
        if "user_id" in df.columns and "book_id" in df.columns:
            # Keep only necessary columns
            keep_cols = ["user_id", "book_id", "rating", "date_added", "date_updated"]
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]

            # Convert ratings to float
            if "rating" in df.columns:
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

                # Fill missing ratings with the median
                median_rating = df["rating"].median()
                df["rating"].fillna(median_rating, inplace=True)

            # Convert dates to datetime
            date_cols = ["date_added", "date_updated"]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # Add timestamp column (seconds since epoch)
            if "date_updated" in df.columns:
                df["timestamp"] = df["date_updated"].astype(int) // 10**9
            elif "date_added" in df.columns:
                df["timestamp"] = df["date_added"].astype(int) // 10**9

            logger.info(f"Cleaned interactions data: {len(df)} rows")
        else:
            logger.warning(
                "Required columns 'user_id' and 'book_id' not found in interactions data"
            )

        return df

    def clean_books(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess books data.

        Args:
            books_df (pd.DataFrame): Raw books DataFrame.

        Returns:
            pd.DataFrame: Cleaned books DataFrame.
        """
        logger.info("Cleaning books data")

        # Make a copy to avoid modifying the original
        df = books_df.copy()

        # Extract relevant columns
        if "book_id" in df.columns:
            # Keep only necessary columns
            keep_cols = [
                "book_id",
                "title",
                "description",
                "authors",
                "average_rating",
                "num_pages",
                "publication_year",
                "language_code",
                "genres",
            ]
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]

            # Convert numeric columns
            numeric_cols = ["average_rating", "num_pages", "publication_year"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Process authors and genres (if they are lists or strings)
            if "authors" in df.columns:
                if isinstance(df["authors"].iloc[0], list):
                    # Join author names into a single string
                    df["authors"] = df["authors"].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else x
                    )

            if "genres" in df.columns:
                if isinstance(df["genres"].iloc[0], list):
                    # Join genres into a single string
                    df["genres"] = df["genres"].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else x
                    )

            logger.info(f"Cleaned books data: {len(df)} rows")
        else:
            logger.warning("Required column 'book_id' not found in books data")

        return df

    def filter_data(
        self,
        interactions_df: pd.DataFrame,
        books_df: pd.DataFrame,
        min_interactions: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter data to include only users and books with sufficient interactions.

        Args:
            interactions_df (pd.DataFrame): Interactions DataFrame.
            books_df (pd.DataFrame): Books DataFrame.
            min_interactions (int, optional): Minimum number of interactions per user/book. Defaults to 5.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered interactions and books DataFrames.
        """
        logger.info(f"Filtering data with min_interactions={min_interactions}")

        # Make copies to avoid modifying the originals
        interactions = interactions_df.copy()
        books = books_df.copy()

        # Count interactions per user and book
        user_counts = interactions["user_id"].value_counts()
        book_counts = interactions["book_id"].value_counts()

        # Filter users and books with sufficient interactions
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_books = book_counts[book_counts >= min_interactions].index

        # Filter interactions
        filtered_interactions = interactions[
            interactions["user_id"].isin(valid_users)
            & interactions["book_id"].isin(valid_books)
        ]

        # Filter books
        filtered_books = books[books["book_id"].isin(valid_books)]

        logger.info(
            f"Filtered data: {len(filtered_interactions)} interactions, "
            f"{len(valid_users)} users, {len(filtered_books)} books"
        )

        return filtered_interactions, filtered_books

    def create_id_mappings(
        self, interactions_df: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Create mappings from original IDs to integer IDs.

        Args:
            interactions_df (pd.DataFrame): Interactions DataFrame.

        Returns:
            Tuple[Dict[str, int], Dict[str, int]]: User and book ID mappings.
        """
        logger.info("Creating ID mappings")

        # Get unique user and book IDs
        unique_user_ids = interactions_df["user_id"].unique()
        unique_book_ids = interactions_df["book_id"].unique()

        # Create mappings
        user_id_map = {user_id: i for i, user_id in enumerate(unique_user_ids)}
        book_id_map = {book_id: i for i, book_id in enumerate(unique_book_ids)}

        logger.info(
            f"Created mappings for {len(user_id_map)} users and {len(book_id_map)} books"
        )

        return user_id_map, book_id_map

    def apply_id_mappings(
        self,
        interactions_df: pd.DataFrame,
        books_df: pd.DataFrame,
        user_id_map: Dict[str, int],
        book_id_map: Dict[str, int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply ID mappings to interactions and books data.

        Args:
            interactions_df (pd.DataFrame): Interactions DataFrame.
            books_df (pd.DataFrame): Books DataFrame.
            user_id_map (Dict[str, int]): User ID mapping.
            book_id_map (Dict[str, int]): Book ID mapping.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with mapped IDs.
        """
        logger.info("Applying ID mappings")

        # Make copies to avoid modifying the originals
        interactions = interactions_df.copy()
        books = books_df.copy()

        # Apply mappings
        interactions["user_id_mapped"] = interactions["user_id"].map(user_id_map)
        interactions["book_id_mapped"] = interactions["book_id"].map(book_id_map)

        books["book_id_mapped"] = books["book_id"].map(book_id_map)

        # Drop rows with missing mapped IDs
        interactions = interactions.dropna(subset=["user_id_mapped", "book_id_mapped"])
        books = books.dropna(subset=["book_id_mapped"])

        # Convert mapped IDs to integers
        interactions["user_id_mapped"] = interactions["user_id_mapped"].astype(int)
        interactions["book_id_mapped"] = interactions["book_id_mapped"].astype(int)
        books["book_id_mapped"] = books["book_id_mapped"].astype(int)

        logger.info(
            f"Applied mappings: {len(interactions)} interactions, {len(books)} books"
        )

        return interactions, books

    def split_train_test(
        self,
        interactions_df: pd.DataFrame,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into training and test sets.

        Args:
            interactions_df (pd.DataFrame): Interactions DataFrame.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
        logger.info(
            f"Splitting data with test_size={test_size}, random_seed={random_seed}"
        )

        # Make a copy to avoid modifying the original
        df = interactions_df.copy()

        # Set random seed
        np.random.seed(random_seed)

        # Shuffle the data
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Calculate split point
        split_idx = int(len(df) * (1 - test_size))

        # Split the data
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        logger.info(f"Split data: {len(train_df)} training, {len(test_df)} test")

        return train_df, test_df

    def process_data(
        self,
        interactions_path: str,
        books_path: str,
        reviews_path: Optional[str] = None,
        min_interactions: int = 5,
        test_size: float = 0.2,
        random_seed: int = 42,
        save_output: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process Goodreads data from raw files to cleaned, filtered, and split data.

        Args:
            interactions_path (str): Path to interactions file.
            books_path (str): Path to books file.
            reviews_path (Optional[str], optional): Path to reviews file. Defaults to None.
            min_interactions (int, optional): Minimum interactions per user/book. Defaults to 5.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            save_output (bool, optional): Whether to save processed data. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, test, and books DataFrames.
        """
        logger.info("Processing Goodreads data")

        # Load data
        interactions_df = self.load_interactions(interactions_path)
        books_df = self.load_books(books_path)

        reviews_df = None
        if reviews_path:
            reviews_df = self.load_reviews(reviews_path)

        # Clean data
        interactions_df = self.clean_interactions(interactions_df)
        books_df = self.clean_books(books_df)

        # Filter data
        interactions_df, books_df = self.filter_data(
            interactions_df, books_df, min_interactions=min_interactions
        )

        # Create ID mappings
        user_id_map, book_id_map = self.create_id_mappings(interactions_df)

        # Apply ID mappings
        interactions_df, books_df = self.apply_id_mappings(
            interactions_df, books_df, user_id_map, book_id_map
        )

        # Split data
        train_df, test_df = self.split_train_test(
            interactions_df, test_size=test_size, random_seed=random_seed
        )

        # Save processed data
        if save_output:
            train_path = self.output_dir / "train.csv"
            test_path = self.output_dir / "test.csv"
            books_path = self.output_dir / "books.csv"
            mappings_path = self.output_dir / "id_mappings.json"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            books_df.to_csv(books_path, index=False)

            # Save ID mappings
            with open(mappings_path, "w") as f:
                json.dump({"user_id_map": user_id_map, "book_id_map": book_id_map}, f)

            logger.info(f"Saved processed data to {self.output_dir}")

        return train_df, test_df, books_df


class QdrantLoader:
    """
    Loader for Goodreads data into Qdrant vector database.

    This class handles creating embeddings and loading data into Qdrant
    for efficient similarity search.
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        use_local: bool = True,
        collection_prefix: str = "goodreads",
    ):
        """
        Initialize the Qdrant loader.

        Args:
            qdrant_host (str, optional): Qdrant server host. Defaults to "localhost".
            qdrant_port (int, optional): Qdrant server port. Defaults to 6333.
            use_local (bool, optional): Whether to use local Qdrant instance. Defaults to True.
            collection_prefix (str, optional): Prefix for collection names. Defaults to "goodreads".
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.use_local = use_local
        self.collection_prefix = collection_prefix

        # Import Qdrant client
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.models = models
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except ImportError:
            logger.error(
                "Qdrant client not installed. Please install with: pip install qdrant-client"
            )
            raise

        # Import sentence-transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            logger.error(
                "Sentence-transformers not installed. Please install with: pip install sentence-transformers"
            )
            raise

    def create_book_embeddings(self, books_df: pd.DataFrame) -> Dict[int, List[float]]:
        """
        Create embeddings for books based on title, description, and other metadata.

        Args:
            books_df (pd.DataFrame): Books DataFrame.

        Returns:
            Dict[int, List[float]]: Dictionary mapping book IDs to embeddings.
        """
        logger.info("Creating book embeddings")

        book_embeddings = {}

        # Prepare text for embedding
        texts = []
        book_ids = []

        for _, row in tqdm(
            books_df.iterrows(), total=len(books_df), desc="Preparing book texts"
        ):
            book_id = row["book_id_mapped"]

            # Combine title, authors, and description
            text_parts = []

            if "title" in row and not pd.isna(row["title"]):
                text_parts.append(f"Title: {row['title']}")

            if "authors" in row and not pd.isna(row["authors"]):
                text_parts.append(f"Authors: {row['authors']}")

            if "description" in row and not pd.isna(row["description"]):
                # Truncate description to avoid very long texts
                description = row["description"]
                if isinstance(description, str) and len(description) > 1000:
                    description = description[:1000] + "..."
                text_parts.append(f"Description: {description}")

            if "genres" in row and not pd.isna(row["genres"]):
                text_parts.append(f"Genres: {row['genres']}")

            # Combine all parts
            text = " ".join(text_parts)

            texts.append(text)
            book_ids.append(book_id)

        # Generate embeddings in batches
        batch_size = 32
        for i in tqdm(
            range(0, len(texts), batch_size), desc="Generating book embeddings"
        ):
            batch_texts = texts[i : i + batch_size]
            batch_ids = book_ids[i : i + batch_size]

            # Generate embeddings
            batch_embeddings = self.embedding_model.encode(batch_texts)

            # Store embeddings
            for j, book_id in enumerate(batch_ids):
                book_embeddings[book_id] = batch_embeddings[j].tolist()

        logger.info(f"Created embeddings for {len(book_embeddings)} books")

        return book_embeddings

    def create_user_embeddings(
        self, interactions_df: pd.DataFrame, book_embeddings: Dict[int, List[float]]
    ) -> Dict[int, List[float]]:
        """
        Create user embeddings by aggregating book embeddings weighted by ratings.

        Args:
            interactions_df (pd.DataFrame): Interactions DataFrame.
            book_embeddings (Dict[int, List[float]]): Book embeddings.

        Returns:
            Dict[int, List[float]]: Dictionary mapping user IDs to embeddings.
        """
        logger.info("Creating user embeddings")

        user_embeddings = {}

        # Group interactions by user
        user_groups = interactions_df.groupby("user_id_mapped")

        # Process each user
        for user_id, group in tqdm(user_groups, desc="Generating user embeddings"):
            # Get books and ratings
            user_books = group["book_id_mapped"].values
            user_ratings = (
                group["rating"].values
                if "rating" in group.columns
                else np.ones_like(user_books)
            )

            # Normalize ratings to sum to 1
            if len(user_ratings) > 0:
                user_ratings = user_ratings / user_ratings.sum()

            # Aggregate book embeddings weighted by ratings
            user_embedding = np.zeros(
                self.embedding_model.get_sentence_embedding_dimension()
            )

            for i, book_id in enumerate(user_books):
                if book_id in book_embeddings:
                    book_embedding = np.array(book_embeddings[book_id])
                    user_embedding += book_embedding * user_ratings[i]

            # Normalize user embedding
            norm = np.linalg.norm(user_embedding)
            if norm > 0:
                user_embedding = user_embedding / norm

            user_embeddings[user_id] = user_embedding.tolist()

        logger.info(f"Created embeddings for {len(user_embeddings)} users")

        return user_embeddings

    def create_collections(self):
        """Create Qdrant collections for books and users."""
        logger.info("Creating Qdrant collections")

        # Get embedding dimension
        vector_size = self.embedding_model.get_sentence_embedding_dimension()

        # Create books collection
        books_collection = f"{self.collection_prefix}_books"
        try:
            self.client.get_collection(books_collection)
            logger.info(f"Collection {books_collection} already exists")
        except Exception:
            self.client.create_collection(
                collection_name=books_collection,
                vectors_config=self.models.VectorParams(
                    size=vector_size, distance=self.models.Distance.COSINE
                ),
            )
            logger.info(f"Created collection {books_collection}")

        # Create users collection
        users_collection = f"{self.collection_prefix}_users"
        try:
            self.client.get_collection(users_collection)
            logger.info(f"Collection {users_collection} already exists")
        except Exception:
            self.client.create_collection(
                collection_name=users_collection,
                vectors_config=self.models.VectorParams(
                    size=vector_size, distance=self.models.Distance.COSINE
                ),
            )
            logger.info(f"Created collection {users_collection}")

    def upload_books(
        self, books_df: pd.DataFrame, book_embeddings: Dict[int, List[float]]
    ):
        """
        Upload books to Qdrant.

        Args:
            books_df (pd.DataFrame): Books DataFrame.
            book_embeddings (Dict[int, List[float]]): Book embeddings.
        """
        logger.info("Uploading books to Qdrant")

        books_collection = f"{self.collection_prefix}_books"

        # Prepare points
        points = []

        for _, row in tqdm(
            books_df.iterrows(), total=len(books_df), desc="Preparing book points"
        ):
            book_id = row["book_id_mapped"]

            # Skip if no embedding
            if book_id not in book_embeddings:
                continue

            # Create payload
            payload = {
                "book_id": int(book_id),
                "original_id": row["book_id"],
                "metadata": {},
            }

            # Add metadata
            metadata_cols = [
                "title",
                "authors",
                "average_rating",
                "num_pages",
                "publication_year",
                "language_code",
                "genres",
            ]
            for col in metadata_cols:
                if col in row and not pd.isna(row[col]):
                    payload["metadata"][col] = row[col]

            # Create point
            points.append(
                self.models.PointStruct(
                    id=int(book_id), vector=book_embeddings[book_id], payload=payload
                )
            )

        # Upload points in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading books"):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=books_collection, points=batch)

        logger.info(f"Uploaded {len(points)} books to Qdrant")

    def upload_users(
        self, interactions_df: pd.DataFrame, user_embeddings: Dict[int, List[float]]
    ):
        """
        Upload users to Qdrant.

        Args:
            interactions_df (pd.DataFrame): Interactions DataFrame.
            user_embeddings (Dict[int, List[float]]): User embeddings.
        """
        logger.info("Uploading users to Qdrant")

        users_collection = f"{self.collection_prefix}_users"

        # Get unique users
        unique_users = interactions_df["user_id_mapped"].unique()

        # Prepare points
        points = []

        for user_id in tqdm(unique_users, desc="Preparing user points"):
            # Skip if no embedding
            if user_id not in user_embeddings:
                continue

            # Get user interactions
            user_interactions = interactions_df[
                interactions_df["user_id_mapped"] == user_id
            ]

            # Create payload
            payload = {
                "user_id": int(user_id),
                "original_id": user_interactions["user_id"].iloc[0],
                "interactions": [],
            }

            # Add interactions
            for _, row in user_interactions.iterrows():
                interaction = {
                    "book_id": int(row["book_id_mapped"]),
                    "original_book_id": row["book_id"],
                }

                if "rating" in row:
                    interaction["rating"] = float(row["rating"])

                if "timestamp" in row:
                    interaction["timestamp"] = int(row["timestamp"])

                payload["interactions"].append(interaction)

            # Create point
            points.append(
                self.models.PointStruct(
                    id=int(user_id), vector=user_embeddings[user_id], payload=payload
                )
            )

        # Upload points in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading users"):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=users_collection, points=batch)

        logger.info(f"Uploaded {len(points)} users to Qdrant")

    def load_data(self, train_df: pd.DataFrame, books_df: pd.DataFrame):
        """
        Load Goodreads data into Qdrant.

        Args:
            train_df (pd.DataFrame): Training interactions DataFrame.
            books_df (pd.DataFrame): Books DataFrame.
        """
        logger.info("Loading Goodreads data into Qdrant")

        # Create collections
        self.create_collections()

        # Create book embeddings
        book_embeddings = self.create_book_embeddings(books_df)

        # Create user embeddings
        user_embeddings = self.create_user_embeddings(train_df, book_embeddings)

        # Upload books
        self.upload_books(books_df, book_embeddings)

        # Upload users
        self.upload_users(train_df, user_embeddings)

        logger.info("Finished loading Goodreads data into Qdrant")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load Goodreads data into Qdrant")

    # Data paths
    parser.add_argument(
        "--interactions_path",
        type=str,
        required=True,
        help="Path to Goodreads interactions file (gzipped JSON)",
    )
    parser.add_argument(
        "--books_path",
        type=str,
        required=True,
        help="Path to Goodreads books file (gzipped JSON)",
    )
    parser.add_argument(
        "--reviews_path",
        type=str,
        default=None,
        help="Path to Goodreads reviews file (gzipped JSON)",
    )

    # Processing parameters
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=5,
        help="Minimum number of interactions per user/book",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directory to save processed data",
    )

    # Qdrant parameters
    parser.add_argument(
        "--qdrant_host", type=str, default="localhost", help="Qdrant server host"
    )
    parser.add_argument(
        "--qdrant_port", type=int, default=6333, help="Qdrant server port"
    )
    parser.add_argument(
        "--use_local", action="store_true", help="Use local Qdrant instance"
    )
    parser.add_argument(
        "--collection_prefix",
        type=str,
        default="goodreads",
        help="Prefix for Qdrant collection names",
    )

    return parser.parse_args()


def main():
    """Run the data processing and loading pipeline."""
    # Parse arguments
    args = parse_args()

    # Initialize processor
    processor = GoodreadsProcessor(output_dir=args.output_dir)

    # Process data
    train_df, test_df, books_df = processor.process_data(
        interactions_path=args.interactions_path,
        books_path=args.books_path,
        reviews_path=args.reviews_path,
        min_interactions=args.min_interactions,
        test_size=args.test_size,
        random_seed=args.random_seed,
        save_output=True,
    )

    # Initialize Qdrant loader
    loader = QdrantLoader(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        use_local=args.use_local,
        collection_prefix=args.collection_prefix,
    )

    # Load data into Qdrant
    loader.load_data(train_df, books_df)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
