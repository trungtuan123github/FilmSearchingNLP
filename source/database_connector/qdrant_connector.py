from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Optional, Union, Dict, Any

import time

def connect_to_qdrant(qdrant_uri, api_key):
    """Connect to Qdrant client and verify the connection."""
    try:


        client = QdrantClient(url=qdrant_uri, api_key=api_key, timeout=60)

        # Try listing collections to verify connection
        _ = client.get_collections()
        print("Successfully connected to Qdrant!")
        return client

    except UnexpectedResponse as e:
        raise Exception(f"Connection failed due to unexpected response: {e}")
    except Exception as e:
        raise Exception(f"Connection failed: {e}")

def get_collection(qdrant_client: QdrantClient, collection_name: str):
    """Get collection info or create it if it doesn't exist."""
    try:
        # Try to get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' found with {collection_info.points_count} points")
        return collection_info
    
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return None

def create_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int = 1536, distance: Distance = Distance.COSINE):
    """Create a new collection with specified vector configuration."""
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"Collection '{collection_name}' created successfully with vector size {vector_size}!")
        return True
    except Exception as e:
        print(f"Failed to create collection '{collection_name}': {e}")
        return False

def get_vector_size(vector):
    """Get the size of a vector, handling different vector formats."""
    if hasattr(vector, 'shape') and len(vector.shape) > 0:
        # Handle numpy arrays or similar
        return vector.shape[-1] if len(vector.shape) > 1 else len(vector)
    elif isinstance(vector, (list, tuple)):
        # Handle regular Python lists/tuples
        return len(vector)
    else:
        # Try to convert to list and get length
        try:
            return len(list(vector))
        except:
            raise ValueError(f"Cannot determine vector size for type: {type(vector)}")

def ensure_collection_exists(qdrant_client: QdrantClient, collection_name: str, vector_size: int):
    """Ensure collection exists, create if it doesn't."""
    collection_info = get_collection(qdrant_client, collection_name)
    
    if collection_info is None:
        print(f"Creating new collection '{collection_name}' with vector size {vector_size}")
        return create_collection(qdrant_client, collection_name, vector_size)
    else:
        # Verify vector size matches
        expected_size = collection_info.config.params.vectors.size
        if expected_size != vector_size:
            print(f"Warning: Collection '{collection_name}' exists with vector size {expected_size}, but trying to insert vector of size {vector_size}")
            return False
        print(f"Collection '{collection_name}' already exists with matching vector size {vector_size}")
        return True

def get_all_points(qdrant_client: QdrantClient, collection_name: str, limit: int = 100, offset: Optional[str] = None):
    """Retrieve all points from a collection with pagination support."""
    try:
        points = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        
        print(f"Retrieved {len(points[0])} points from collection '{collection_name}'")
        return points[0], points[1]  # points, next_page_offset
    
    except Exception as e:
        print(f"Failed to retrieve points from collection '{collection_name}': {e}")
        return [], None


def insert_points_batch_to_qdrant(
    qdrant_client,
    collection_name: str,
    qdrant_points: List,
    batch_size: int = 100,
    max_retries: int = 3
):
    """Insert multiple QdrantPoints to the collection in batches with retry support."""

    try:
        if not qdrant_points:
            print("No points to insert")
            return True


        vector_size = get_vector_size(qdrant_points[0].vector)

        if not ensure_collection_exists(qdrant_client, collection_name, vector_size):
            raise Exception(f"Failed to ensure collection '{collection_name}' exists with vector size {vector_size}")

        total_inserted = 0
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i:i + batch_size]
            point_structs = []

            for qdrant_point in batch:
                current_vector_size = get_vector_size(qdrant_point.vector)
                if current_vector_size != vector_size:
                    raise ValueError(f"Inconsistent vector sizes: expected {vector_size}, got {current_vector_size}")

                point_struct = PointStruct(
                    id=qdrant_point.id,
                    vector=qdrant_point.vector.tolist() if hasattr(qdrant_point.vector, 'tolist') else list(qdrant_point.vector),
                    payload={
                        "text": qdrant_point.text,
                        "metadata": qdrant_point.metadata.dict() if hasattr(qdrant_point.metadata, 'dict') else qdrant_point.metadata
                    }
                )
                point_structs.append(point_struct)

            # Retry logic
            for attempt in range(max_retries):
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=point_structs
                    )
                    print(f"Inserted batch {i // batch_size + 1}: {len(point_structs)} points")
                    total_inserted += len(point_structs)
                    break  # Success
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for batch {i // batch_size + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise  # Re-raise after last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff

        print(f"Successfully inserted {total_inserted} total points in batches")
        return True


    except Exception as e:
        print(f"Failed to insert points in batch: {e}")
        return False


def search_points(qdrant_client: QdrantClient, collection_name: str, query_vector: List[float], 
                 limit: int = 10, score_threshold: Optional[float] = None, 
                 filter_conditions: Optional[Dict] = None):
    """Search for similar points using vector similarity."""
    try:
        # Prepare filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ) for key, value in filter_conditions.items()
                ]
            )
        
        # Perform search
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )

        # print(f"Found {len(search_result)} similar points")
        return search_result
        
    except Exception as e:
        # print(f"Failed to search points: {e}")

        return []

def delete_point(qdrant_client: QdrantClient, collection_name: str, point_id: Union[int, str]):
    """Delete a point by ID."""
    try:
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=[point_id]
        )
        print(f"Successfully deleted point with ID: {point_id}")
        return True
        
    except Exception as e:
        print(f"Failed to delete point with ID {point_id}: {e}")
        return False

def get_point_by_id(qdrant_client: QdrantClient, collection_name: str, point_id: Union[int, str]):
    """Retrieve a specific point by ID."""
    try:
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True
        )
        
        if points:
            print(f"Found point with ID: {point_id}")
            return points[0]
        else:
            print(f"No point found with ID: {point_id}")
            return None
            
    except Exception as e:
        print(f"Failed to retrieve point with ID {point_id}: {e}")
        return None