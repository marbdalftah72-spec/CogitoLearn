"""
Misconception Analyzer and Adaptive Remediation Diagnoser.
==========================================================
Identifies cumulative conceptual misconceptions, traverses the Pedagogical
Knowledge Graph (PKG) to isolate prerequisite knowledge gaps, and constructs
scaffolded remediation pathways within the student's Zone of Proximal Development (ZPD).
"""

import logging
from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field

from common.contracts import (
    PKGConcept,
    PKGEdge,
    RelationType,
    EvidenceSpan,
    StudentConceptMastery,
)

logger = logging.getLogger(__name__)


class RemediationStep(BaseModel):
    concept_id: str
    concept_name_ar: str
    concept_name_en: str
    is_prerequisite: bool
    current_mastery: float
    remediation_explanation: str
    grounding_span_ids: List[str] = Field(default_factory=list)


class RemediationPlan(BaseModel):
    student_id: str
    target_concept_id: str
    target_concept_name_ar: str
    target_concept_name_en: str
    prerequisite_gaps: List[str]
    steps: List[RemediationStep]
    diagnosed_misconceptions: List[str]
    estimated_duration_minutes: int
    guidance_summary: str


class ErrorDiagnoser:
    """
    Analyzes student error patterns and synthesizes adaptive remediation plans.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.api_key = gemini_api_key
        # Track errors per student: student_id -> {concept_id -> [misconception_ids]}
        self._error_history: Dict[str, Dict[str, List[str]]] = {}

    def record_error(
        self, student_id: str, concept_id: str, misconception_id: str
    ) -> None:
        """
        Logs a diagnosed misconception occurrence for a student.
        """
        if student_id not in self._error_history:
            self._error_history[student_id] = {}
        if concept_id not in self._error_history[student_id]:
            self._error_history[student_id][concept_id] = []

        self._error_history[student_id][concept_id].append(misconception_id)

    def get_error_patterns(self, student_id: str) -> Dict[str, List[str]]:
        """
        Returns all logged misconceptions grouped by concept_id.
        """
        return self._error_history.get(student_id, {})

    def find_prerequisite_gaps(
        self,
        target_concept_id: str,
        pkg_concepts: List[PKGConcept],
        pkg_edges: List[PKGEdge],
        mastery_states: List[StudentConceptMastery],
        mastery_threshold: float = 0.65,
    ) -> List[PKGConcept]:
        """
        Traverses PKG prerequisite edges backwards from target_concept_id to locate
        all unmastered foundational concepts in topological order.
        """
        concepts_by_id = {c.concept_id: c for c in pkg_concepts}
        mastery_by_id = {m.concept_id: m.bkt_p_mastery for m in mastery_states}

        # Build prerequisite adjacency graph:
        # If Edge: src_id -> dst_id with PREREQUISITE_OF, then src is prerequisite of dst.
        prereqs_of: Dict[str, List[str]] = {}
        for edge in pkg_edges:
            if edge.relation_type == RelationType.PREREQUISITE_OF:
                prereqs_of.setdefault(edge.dst_concept_id, []).append(edge.src_concept_id)

        # Topological traversal (BFS / DFS) collecting unmastered prerequisites
        visited: Set[str] = set()
        unmastered_prereqs: List[PKGConcept] = []

        def traverse(curr_id: str):
            for prereq_id in prereqs_of.get(curr_id, []):
                if prereq_id not in visited:
                    visited.add(prereq_id)
                    p_mastery = mastery_by_id.get(prereq_id, 0.15)
                    if p_mastery < mastery_threshold:
                        if prereq_id in concepts_by_id:
                            unmastered_prereqs.append(concepts_by_id[prereq_id])
                    # Continue inspecting deeper prerequisites
                    traverse(prereq_id)

        traverse(target_concept_id)
        # Reverse to get foundational (earliest) prerequisites first
        unmastered_prereqs.reverse()
        return unmastered_prereqs

    def generate_remediation_plan(
        self,
        student_id: str,
        target_concept: PKGConcept,
        prerequisite_gaps: List[PKGConcept],
        evidence_spans: List[EvidenceSpan],
        mastery_states: List[StudentConceptMastery],
        active_misconceptions: Optional[List[str]] = None,
    ) -> RemediationPlan:
        """
        Constructs a structured remediation sequence targeting prerequisite gaps first,
        followed by resolving specific misconceptions for the target concept.
        """
        mastery_by_id = {m.concept_id: m.bkt_p_mastery for m in mastery_states}
        span_ids = [s.span_id for s in evidence_spans]

        misconceptions = active_misconceptions or self.get_error_patterns(student_id).get(
            target_concept.concept_id, []
        )

        steps: List[RemediationStep] = []

        # 1. Prerequisite gap remediation steps
        for prereq in prerequisite_gaps:
            m_val = mastery_by_id.get(prereq.concept_id, 0.15)
            steps.append(
                RemediationStep(
                    concept_id=prereq.concept_id,
                    concept_name_ar=prereq.name_ar,
                    concept_name_en=prereq.name_en,
                    is_prerequisite=True,
                    current_mastery=round(m_val, 4),
                    remediation_explanation=(
                        f"يجب مراجعة المفهوم التأسيسي '{prereq.name_ar}' أولاً "
                        f"لأنه متطلب أساسي لفهم '{target_concept.name_ar}'."
                    ),
                    grounding_span_ids=[],
                )
            )

        # 2. Target concept remediation step
        target_m = mastery_by_id.get(target_concept.concept_id, 0.15)
        target_explanation = (
            f"مراجعة المفهوم المستهدف '{target_concept.name_ar}' "
            f"مع التركيز على إزالة المفاهيم المغلوطة: {', '.join(misconceptions) if misconceptions else 'المراجعة الشاملة'}."
        )

        steps.append(
            RemediationStep(
                concept_id=target_concept.concept_id,
                concept_name_ar=target_concept.name_ar,
                concept_name_en=target_concept.name_en,
                is_prerequisite=False,
                current_mastery=round(target_m, 4),
                remediation_explanation=target_explanation,
                grounding_span_ids=span_ids[:3],
            )
        )

        duration = max(10, len(steps) * 8)
        summary = (
            f"خطة علاجية مخصصة للطالب لمعالجة الفجوات في مفهوم '{target_concept.name_ar}'. "
            f"تتضمن الخطة {len(prerequisite_gaps)} متطلبات سابقة و{len(misconceptions)} مفاهيم خاطئة مشخصة."
        )

        return RemediationPlan(
            student_id=student_id,
            target_concept_id=target_concept.concept_id,
            target_concept_name_ar=target_concept.name_ar,
            target_concept_name_en=target_concept.name_en,
            prerequisite_gaps=[p.concept_id for p in prerequisite_gaps],
            steps=steps,
            diagnosed_misconceptions=misconceptions,
            estimated_duration_minutes=duration,
            guidance_summary=summary,
        )
