"""
Tests for PedagogyCoordinator v2: signal fusion, fitting, unified report.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import random

import pytest
from common.contracts import (
    ConceptType,
    PKGConcept,
    PKGEdge,
    RelationType,
)

from pedagogy_coordinator import PedagogyCoordinator
from misconception_analyzer.advanced_diagnoser import AdvancedErrorDiagnoser
from irt_engine.irt_model import IRTItem


@pytest.fixture
def coordinator():
    return PedagogyCoordinator()


def _concept(cid):
    return PKGConcept(
        concept_id=cid, course_id="course_1", name_ar=f"مفهوم {cid}",
        name_en=f"Concept {cid}", concept_type=ConceptType.DEFINITION,
        base_difficulty=0.5,
    )


def _edge(src, dst):
    return PKGEdge(
        edge_id=f"e_{src}_{dst}", src_concept_id=src, dst_concept_id=dst,
        relation_type=RelationType.PREREQUISITE_OF,
        provenance_span_ids=["span_1"], entailment_conf=0.9,
    )


def test_coordinator_uses_advanced_diagnoser(coordinator):
    assert isinstance(coordinator.diagnoser, AdvancedErrorDiagnoser)


def test_fit_bkt_parameters_returns_params_and_metrics(coordinator):
    rng = random.Random(5)
    sequences = []
    for _ in range(20):
        learned, seq = False, []
        for _ in range(10):
            seq.append(rng.random() < (0.9 if learned else 0.2))
            if not learned and rng.random() < 0.3:
                learned = True
        sequences.append(seq)
    params, metrics = coordinator.fit_bkt_parameters("c1", sequences)
    assert 0.0 < params.p_guess < 1.0
    assert metrics.n_predictions == 200
    assert 0.0 <= metrics.auc <= 1.0


def test_estimate_student_ability_sign(coordinator):
    items = [IRTItem(f"i{i}", "c1", 1.2, -1.0 + 0.3 * i) for i in range(8)]
    strong = coordinator.estimate_student_ability(items, [True] * 8)
    weak = coordinator.estimate_student_ability(items, [False] * 8)
    assert strong.theta > 0 > weak.theta


def test_unified_report_without_optional_data(coordinator):
    coordinator.state_store.get_or_create_state("s1", "c1")
    report = coordinator.get_unified_student_report("s1")
    assert report.student_id == "s1"
    assert report.irt_theta is None
    assert report.total_concepts_tracked == 1
    assert report.recommendations_ar  # never empty


def test_unified_report_fuses_all_signals(coordinator):
    # Seed mastery + a persistent misconception.
    state = coordinator.state_store.get_or_create_state("s1", "A")
    coordinator.state_store.update_state(state.model_copy(update={"bkt_p_mastery": 0.9}))
    for _ in range(3):
        coordinator.diagnoser.record_error("s1", "A", "MISC_INVERSE_LOGIC")

    items = [IRTItem(f"i{i}", "A", 1.2, 0.0) for i in range(6)]
    concepts = [_concept("A"), _concept("B")]
    edges = [_edge("A", "B")]
    report = coordinator.get_unified_student_report(
        "s1",
        irt_items=items,
        irt_responses=[True, True, False, True, True, False],
        pkg_concepts=concepts,
        pkg_edges=edges,
    )
    assert report.irt_theta is not None
    assert report.persistent_misconception_count == 1
    assert report.top_misconceptions[0].misconception_id == "MISC_INVERSE_LOGIC"
    assert report.next_session_plan is not None
    assert any("MISC" not in r and "عكس" in r or "عالج" in r for r in report.recommendations_ar)


def test_plan_next_session_includes_due_review(coordinator):
    concepts = [_concept("A"), _concept("B")]
    edges = [_edge("A", "B")]
    state = coordinator.state_store.get_or_create_state("s1", "A")
    # next_review_due = None means "never reviewed -> due immediately".
    coordinator.state_store.update_state(state.model_copy(update={"bkt_p_mastery": 0.9}))
    plan = coordinator.plan_next_session("s1", concepts, edges)
    review_steps = [s for s in plan.steps if s.step_type == "REVIEW"]
    assert review_steps and review_steps[0].concept_id == "A"


def test_evaluate_essay_v2_through_coordinator(coordinator):
    from common.contracts import BloomLevel, EssayQuestion, RubricCriterion

    question = EssayQuestion(
        item_id="e1", concept_id="A", stem="حلّل المفهوم",
        bloom_level=BloomLevel.ANALYZE,
        model_answer="المفهوم يقوم على ربط المعطيات بالاستنتاجات عبر أوزان تركيز دقيقة.",
        rubric=[RubricCriterion(name="الدقة", description="ربط المعطيات بالاستنتاجات", max_score=10.0)],
        grounding_span_ids=["span_1"],
    )
    result = coordinator.evaluate_essay_v2(
        question,
        "حلّل المفهوم: يقوم على ربط المعطيات بالاستنتاجات عبر أوزان تركيز، وقارن بينه وبين البدائل، "
        "لأن ذلك يوضح الفروق الجوهرية في المعالجة وبالتالي يقيّم جودة النتائج بدقة عالية.",
    )
    assert 0.0 <= result.score_percentage <= 100.0
    assert result.max_score == 10.0
