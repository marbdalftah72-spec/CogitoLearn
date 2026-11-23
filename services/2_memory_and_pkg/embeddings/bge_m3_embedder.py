import os
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

class BGEM3Embedder:
    """
    Embedding service using BAAI/bge-m3 model.
    Supports dense vectors and batch encoding.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, batch_size: int = 32) -> np.ndarray:
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return embeddings

    def encode_query(self, query: str) -> List[float]:
        return self.encode([query])[0].tolist()
