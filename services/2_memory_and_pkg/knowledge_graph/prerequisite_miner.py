from typing import List
from common.contracts import PKGConcept, PKGEdge, RelationType

class PrerequisiteMiner:
    """
    Identifies prerequisite relations among concepts to form the Knowledge Graph.
    """
    def mine_prerequisites(self, concepts: List[PKGConcept]) -> List[PKGEdge]:
        edges = []
        concept_ids = {c.concept_id for c in concepts}
        
        rules = [
            ("tokenization", "stemming"),
            ("tokenization", "lemmatization"),
            ("tokenization", "bag_of_words"),
            ("bag_of_words", "tf_idf"),
            ("tokenization", "n_gram"),
            ("natural_language_processing", "nlu"),
            ("natural_language_processing", "nlg"),
        ]
        
        idx = 0
        for src, dst in rules:
            if src in concept_ids and dst in concept_ids:
                idx += 1
                edges.append(
                    PKGEdge(
                        edge_id=f"edge_{idx}_{src}_{dst}",
                        src_concept_id=src,
                        dst_concept_id=dst,
                        relation_type=RelationType.PREREQUISITE_OF,
                        provenance_span_ids=["default_span"],
                        entailment_conf=0.95
                    )
                )
        return edges
