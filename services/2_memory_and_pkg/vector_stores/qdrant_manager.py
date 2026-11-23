import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages Qdrant client connection, collections, and hybrid/vector search.
    """
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if self.url and self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        elif self.url:
            self.client = QdrantClient(url=self.url)
        else:
            self.client = QdrantClient(location=":memory:")
            logger.info("Initialized in-memory Qdrant instance.")

    def ensure_collection(self, collection_name: str, vector_size: int = 1024):
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}.")

    def upsert_points(self, collection_name: str, points: List[models.PointStruct]):
        return self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def search(self, collection_name: str, query_vector: List[float], limit: int = 3, score_threshold: Optional[float] = None):
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        ).points

    def scroll_all(self, collection_name: str, limit: int = 100):
        chunks = []
        offset = None
        while True:
            pts, offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(pts)
            if offset is None:
                break
        return sorted(chunks, key=lambda p: p.payload.get("chunk_id", 0))
