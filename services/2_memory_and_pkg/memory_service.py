import re
import uuid
import logging
from typing import List, Tuple, Optional
from qdrant_client.http import models

from common.contracts import UnifiedDocumentAST, EvidenceSpan, PKGConcept, PKGEdge
from .embeddings.bge_m3_embedder import BGEM3Embedder
from .vector_stores.qdrant_manager import QdrantManager
from .knowledge_graph.concept_extractor import ConceptExtractor
from .knowledge_graph.prerequisite_miner import PrerequisiteMiner

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Service 2 Coordinator:
    - Text Cleaning and Chunking (with overlap)
    - Vector Embedding with BGE-M3
    - Vector Indexing in Qdrant
    - Concept Extraction and PKG Construction
    """
    def __init__(
        self,
        collection_name: str = "academic_lectures",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedder = BGEM3Embedder()
        self.qdrant = QdrantManager(url=qdrant_url, api_key=qdrant_api_key)
        self.concept_extractor = ConceptExtractor()
        self.prereq_miner = PrerequisiteMiner()
        
        # Ensure collection exists (BGE-M3 vector size 1024)
        self.qdrant.ensure_collection(self.collection_name, vector_size=1024)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    @staticmethod
    def create_chunks(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start += chunk_size - overlap
        return chunks

    def process_and_index_text(self, text: str, lecture_id: str = "lec_01") -> Tuple[int, List[PKGConcept], List[PKGEdge]]:
        cleaned = self.clean_text(text)
        chunks = self.create_chunks(cleaned)
        
        if not chunks:
            return 0, [], []
            
        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": i + 1,
                        "text": chunk,
                        "lecture_id": lecture_id,
                        "hierarchy_level": 0
                    }
                )
            )
            
        self.qdrant.upsert_points(self.collection_name, points)
        
        # PKG
        concepts = self.concept_extractor.extract_concepts(cleaned)
        edges = self.prereq_miner.mine_prerequisites(concepts)
        
        return len(points), concepts, edges
