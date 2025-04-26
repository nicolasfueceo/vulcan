"""
This module provides integration with Qdrant vector database for storing and retrieving
embeddings for the benchmark recommender systems.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QdrantManager:
    """Class to manage Qdrant vector database for FUEGO benchmark recommenders."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        https: bool = False,
        api_key: Optional[str] = None,
        project_root: Optional[str] = None,
    ):
        """
        Initialize the QdrantManager.

        Args:
            host (str, optional): Qdrant server host. Defaults to "localhost".
            port (int, optional): Qdrant server HTTP port. Defaults to 6333.
            grpc_port (int, optional): Qdrant server gRPC port. Defaults to 6334.
            prefer_grpc (bool, optional): Whether to prefer gRPC over HTTP. Defaults to True.
            https (bool, optional): Whether to use HTTPS. Defaults to False.
            api_key (str, optional): API key for Qdrant Cloud. Defaults to None.
            project_root (str, optional): Root directory of the project. Defaults to current working directory.
        """
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
        )

        # Set up project directories
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"

        logger.info(f"Qdrant manager initialized with host: {host}, port: {port}")

        # Collection names for benchmark recommenders
        self.collections = {
            "user_embeddings": "user_embeddings_collection",
            "item_embeddings": "item_embeddings_collection",
            "item_similarities": "item_similarities_collection",
        }

    def create_collection(self, collection_name: str, vector_size: int = 100):
        """
        Create a collection in Qdrant for storing embeddings.

        Args:
            collection_name (str): Name of the collection to create
            vector_size (int, optional): Size of the embedding vectors. Defaults to 100.

        Returns:
            str: Status of collection creation
        """
        logger.info(
            f"Creating collection {collection_name} with vector size: {vector_size}"
        )

        # Check if collection already exists
        collections_list = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections_list]

        if collection_name in collection_names:
            logger.info(f"Collection {collection_name} already exists")
            return "already_exists"

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        logger.info(f"Created collection: {collection_name}")
        return "created"

    def create_collections(self, vector_size: int = 100):
        """
        Create all collections needed for benchmark recommenders.

        Args:
            vector_size (int, optional): Size of the embedding vectors. Defaults to 100.

        Returns:
            dict: Status of collection creation
        """
        logger.info(f"Creating collections with vector size: {vector_size}")

        results = {}

        # Create collections for each data type
        for name, collection_name in self.collections.items():
            results[collection_name] = self.create_collection(
                collection_name, vector_size
            )

        return results

    def upload_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[Union[int, str]],
        payloads: Optional[List[Dict]] = None,
    ):
        """
        Upload vectors to a Qdrant collection.

        Args:
            collection_name (str): Name of the collection to upload to
            vectors (List[List[float]]): List of vectors to upload
            ids (List[Union[int, str]]): List of IDs for the vectors
            payloads (Optional[List[Dict]], optional): List of payloads for the vectors. Defaults to None.

        Returns:
            int: Number of vectors uploaded
        """
        logger.info(f"Uploading {len(vectors)} vectors to collection {collection_name}")

        # Prepare points for upload
        points = []

        for i, (vector_id, vector) in enumerate(zip(ids, vectors)):
            # Create payload
            payload = payloads[i] if payloads is not None else {}

            # Add point
            points.append(PointStruct(id=vector_id, vector=vector, payload=payload))

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)
            logger.info(
                f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}"
            )

        logger.info(f"Uploaded {len(points)} vectors to Qdrant")

        return len(points)

    def search_similar_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_params: Optional[Dict] = None,
    ):
        """
        Search for similar vectors in a collection.

        Args:
            collection_name (str): Name of the collection to search in.
            query_vector (List[float]): Query vector to search for.
            limit (int, optional): Maximum number of results to return. Defaults to 10.
            filter_params (Dict, optional): Filter parameters for the search. Defaults to None.

        Returns:
            List: List of search results
        """
        logger.info(f"Searching for similar vectors in collection: {collection_name}")

        # Perform search
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_params,
        )

        logger.info(f"Found {len(search_results)} similar vectors")

        return search_results

    def get_vector_by_id(self, collection_name: str, vector_id: Union[int, str]):
        """
        Get a vector by its ID.

        Args:
            collection_name (str): Name of the collection to search in.
            vector_id (Union[int, str]): ID of the vector to retrieve.

        Returns:
            Dict: Vector data
        """
        logger.info(
            f"Getting vector with ID {vector_id} from collection: {collection_name}"
        )

        # Get vector
        vector = self.client.retrieve(collection_name=collection_name, ids=[vector_id])

        if not vector:
            logger.warning(
                f"Vector with ID {vector_id} not found in collection {collection_name}"
            )
            return None

        return vector[0]

    def delete_collection(self, collection_name: str):
        """
        Delete a collection.

        Args:
            collection_name (str): Name of the collection to delete.

        Returns:
            bool: Whether the deletion was successful
        """
        logger.info(f"Deleting collection: {collection_name}")

        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    def delete_all_collections(self):
        """
        Delete all collections used by the benchmark system.

        Returns:
            dict: Status of collection deletion
        """
        logger.info("Deleting all collections")

        results = {}

        for name, collection_name in self.collections.items():
            results[collection_name] = self.delete_collection(collection_name)

        return results

    def get_collection_info(self, collection_name: str):
        """
        Get information about a collection.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            dict: Collection information
        """
        logger.info(f"Getting information for collection: {collection_name}")

        try:
            collection_info = self.client.get_collection(
                collection_name=collection_name
            )
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None

    def list_collections(self):
        """
        List all collections in the Qdrant instance.

        Returns:
            list: List of collection names
        """
        logger.info("Listing all collections")

        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            return collection_names
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
