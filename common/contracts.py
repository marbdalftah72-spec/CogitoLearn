"""
CogitoLearn - Central Domain Contracts & Unified Schemas
=========================================================
This module defines the immutable Pydantic v2 data contracts exchanged
between the 5 decoupled domain services, ensuring zero circular dependencies
and contract-first isolation.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ==========================================
# 1. Modality & Parsing Contracts
# ==========================================

class MediaModality(str, Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    SLIDES = "SLIDES"


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class EvidenceSpan(BaseModel):
    span_id: str = Field(..., description="Unique UUID for this text/formula span")
    lecture_id: str
    hierarchy_level: int = Field(0, description="0: Micro-Chunk, 1: Parent Section, 2: RAPTOR Summary")
    modality_kind: str = Field("PROSE", description="PROSE, LATEX_EQ, TABLE_MD, DIAGRAM_AST, ASR_SPAN")
    content_text: str
    page_no: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    qdrant_point_id: Optional[str] = None


class UnifiedDocumentAST(BaseModel):
    lecture_id: str
    course_id: str
    title: str
    modality: MediaModality
    total_spans: int
    spans: List[EvidenceSpan] = Field(default_factory=list)
    raw_markdown: Optional[str] = None


# ==========================================
# 2. Pedagogical Knowledge Graph (PKG) Contracts
# ==========================================

class ConceptType(str, Enum):
    DEFINITION = "DEFINITION"
    THEOREM = "THEOREM"
    ALGORITHM = "ALGORITHM"
    FORMULA = "FORMULA"


class RelationType(str, Enum):
    PREREQUISITE_OF = "PREREQUISITE_OF"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRASTS_WITH = "CONTRASTS_WITH"


class PKGConcept(BaseModel):
    concept_id: str
    course_id: str
    name_ar: str
    name_en: str
    concept_type: ConceptType
    base_difficulty: float = 0.50


class PKGEdge(BaseModel):
    edge_id: str
    src_concept_id: str
    dst_concept_id: str
    relation_type: RelationType
    provenance_span_ids: List[str] = Field(..., min_length=1, description="Direct provenance spans grounding this relation")
    entailment_conf: float = Field(..., ge=0.0, le=1.0)


# ==========================================
# 3. Retrieval & Routing Contracts
# ==========================================

class RetrievalStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    ABSTAIN = "ABSTAIN"


class QueryPlan(BaseModel):
    query: str
    route: str
    sub_queries: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)


class RetrievalBundle(BaseModel):
    status: RetrievalStatus
    spans: List[EvidenceSpan] = Field(default_factory=list)
    plan: Optional[QueryPlan] = None
    abstention_reason: Optional[str] = None


# ==========================================
# 4. Verified Response & Verification Gate
# ==========================================

class AtomicClaim(BaseModel):
    claim_id: str
    statement: str
    cited_span_ids: List[str]
    is_entailed: bool = False
    is_algebraically_verified: bool = False


class VerifiedResponse(BaseModel):
    query: str
    draft_answer: str
    verified_answer: str
    status: str  # "VERIFIED", "PARTIALLY_GROUNDED", "ABSTAINED"
    faithfulness_score: float
    claims: List[AtomicClaim] = Field(default_factory=list)
    contradictions_detected: List[str] = Field(default_factory=list)


# ==========================================
# 5. Pedagogy & Assessment Contracts
# ==========================================

class BloomLevel(str, Enum):
    REMEMBER = "REMEMBER"
    UNDERSTAND = "UNDERSTAND"
    APPLY = "APPLY"
    ANALYZE = "ANALYZE"
    EVALUATE = "EVALUATE"


class MCQOption(BaseModel):
    option_id: str
    text: str
    is_correct: bool
    misconception_id: Optional[str] = None
    misconception_explanation: Optional[str] = None


class MCQQuestion(BaseModel):
    item_id: str
    concept_id: str
    stem: str
    bloom_level: BloomLevel
    options: List[MCQOption]
    explanation: str
    grounding_span_ids: List[str] = Field(default_factory=list)


class RubricCriterion(BaseModel):
    name: str
    description: str
    max_score: float


class EssayQuestion(BaseModel):
    item_id: str
    concept_id: str
    stem: str
    bloom_level: BloomLevel
    model_answer: str
    rubric: List[RubricCriterion]
    grounding_span_ids: List[str] = Field(default_factory=list)


class StudentConceptMastery(BaseModel):
    student_id: str
    concept_id: str
    bkt_p_mastery: float = 0.15
    bkt_p_slip: float = 0.10
    bkt_p_guess: float = 0.20
    bkt_p_transit: float = 0.15
    fsrs_stability: float = 1.0
    fsrs_difficulty: float = 5.0
    next_review_due: Optional[str] = None
    diagnosed_misconceptions: List[str] = Field(default_factory=list)
