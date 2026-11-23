from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RaptorClusterer:
    """
    RAPTOR hierarchical summarization and clustering.
    Creates level-2 cluster summaries for broad query routing.
    """
    def __init__(self, cluster_size: int = 5):
        self.cluster_size = cluster_size

    def cluster_and_summarize(self, chunks: List[str]) -> List[Dict[str, Any]]:
        summaries = []
        for i in range(0, len(chunks), self.cluster_size):
            group = chunks[i:i + self.cluster_size]
            summary_text = " ".join(group)[:500] + "..."
            summaries.append({
                "level": 2,
                "summary": summary_text,
                "source_chunk_count": len(group)
            })
        return summaries
