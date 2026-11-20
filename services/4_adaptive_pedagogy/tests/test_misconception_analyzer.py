"""
Tests for Misconception Analyzer and PKG Prerequisite Gap Detection.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

from common.contracts import (
    PKGConcept,
    PKGEdge,
    ConceptType,
    RelationType,
    StudentConceptMastery,
    EvidenceSpan,
)
from misconception_analyzer.error_diagnoser import ErrorDiagnoser


def test_find_prerequisite_gaps_topological():
    diagnoser = ErrorDiagnoser()

    # Concepts: Linear Algebra -> Vectors -> Attention Mechanism
    c_la = PKGConcept(
        concept_id="c_la",
        course_id="nlp_101",
        name_ar="الجبر الخطي",
        name_en="Linear Algebra",
        concept_type=ConceptType.DEFINITION,
    )
    c_vec = PKGConcept(
        concept_id="c_vec",
        course_id="nlp_101",
        name_ar="المتجهات",
        name_en="Vectors",
        concept_type=ConceptType.DEFINITION,
    )
    c_attn = PKGConcept(
        concept_id="c_attn",
        course_id="nlp_101",
        name_ar="آلية الانتباه",
        name_en="Attention Mechanism",
        concept_type=ConceptType.ALGORITHM,
    )

    # Edges: c_la is prerequisite of c_vec, c_vec is prerequisite of c_attn
    edges = [
        PKGEdge(
            edge_id="e1",
            src_concept_id="c_la",
            dst_concept_id="c_vec",
            relation_type=RelationType.PREREQUISITE_OF,
            provenance_span_ids=["s1"],
            entailment_conf=0.95,
        ),
        PKGEdge(
            edge_id="e2",
            src_concept_id="c_vec",
            dst_concept_id="c_attn",
            relation_type=RelationType.PREREQUISITE_OF,
            provenance_span_ids=["s2"],
            entailment_conf=0.92,
        ),
    ]

    mastery = [
        StudentConceptMastery(student_id="std_1", concept_id="c_la", bkt_p_mastery=0.20),
        StudentConceptMastery(student_id="std_1", concept_id="c_vec", bkt_p_mastery=0.30),
        StudentConceptMastery(student_id="std_1", concept_id="c_attn", bkt_p_mastery=0.15),
    ]

    gaps = diagnoser.find_prerequisite_gaps(
        target_concept_id="c_attn",
        pkg_concepts=[c_la, c_vec, c_attn],
        pkg_edges=edges,
        mastery_states=mastery,
        mastery_threshold=0.65,
    )

    gap_ids = [g.concept_id for g in gaps]
    assert "c_la" in gap_ids
    assert "c_vec" in gap_ids
    assert gap_ids.index("c_la") < gap_ids.index("c_vec")


def test_remediation_plan_generation():
    diagnoser = ErrorDiagnoser()
    c_target = PKGConcept(
        concept_id="c_target",
        course_id="nlp_101",
        name_ar="التحليل الصرفي",
        name_en="Morphological Analysis",
        concept_type=ConceptType.ALGORITHM,
    )
    c_prereq = PKGConcept(
        concept_id="c_prereq",
        course_id="nlp_101",
        name_ar="الجذور والأوزان",
        name_en="Roots and Patterns",
        concept_type=ConceptType.DEFINITION,
    )

    span = EvidenceSpan(
        span_id="span_1",
        lecture_id="lec_1",
        content_text="التحليل الصرفي يعتمد على اشتقاق الجذور العربية.",
    )

    plan = diagnoser.generate_remediation_plan(
        student_id="std_rem",
        target_concept=c_target,
        prerequisite_gaps=[c_prereq],
        evidence_spans=[span],
        mastery_states=[],
        active_misconceptions=["MISC_ROOT_CONFUSION"],
    )

    assert plan.target_concept_id == "c_target"
    assert len(plan.steps) == 2
    assert plan.steps[0].is_prerequisite is True
    assert plan.steps[1].is_prerequisite is False
    assert "MISC_ROOT_CONFUSION" in plan.diagnosed_misconceptions
