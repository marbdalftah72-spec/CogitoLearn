"""
Tests for the prerequisite-graph Learning Path planner (v2).
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import pytest
from common.contracts import ConceptType, PKGConcept, PKGEdge, RelationType

from learning_path.path_planner import LearningPathPlanner


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


@pytest.fixture
def planner():
    # Chain: A -> B -> C, plus independent D.
    concepts = [_concept(c) for c in ("A", "B", "C", "D")]
    edges = [_edge("A", "B"), _edge("B", "C")]
    return LearningPathPlanner(concepts, edges)


def test_topological_order_respects_prerequisites(planner):
    order = planner.topological_order()
    assert order.index("A") < order.index("B") < order.index("C")
    assert set(order) == {"A", "B", "C", "D"}


def test_topological_order_cycle_raises():
    concepts = [_concept("X"), _concept("Y")]
    edges = [_edge("X", "Y"), _edge("Y", "X")]
    cyclic = LearningPathPlanner(concepts, edges)
    with pytest.raises(ValueError):
        cyclic.topological_order()


def test_unlocked_requires_mastered_prerequisites(planner):
    assert "B" not in planner.unlocked_concepts({"A": 0.40})
    assert "B" in planner.unlocked_concepts({"A": 0.90})
    assert "C" not in planner.unlocked_concepts({"A": 0.90})  # B not mastered


def test_blocked_by_lists_unmastered_prerequisites(planner):
    assert planner.blocked_by("C", {"B": 0.30}) == ["B"]
    assert planner.blocked_by("C", {"B": 0.95}) == []


def test_effective_gate_adjusts_with_theta(planner):
    base = planner.effective_gate(None)
    assert planner.effective_gate(2.0) < base       # strong student: lower gate
    assert planner.effective_gate(-2.0) > base      # struggling: higher gate


def test_plan_session_reviews_come_first(planner):
    plan = planner.plan_session(
        student_id="s1",
        mastery_map={"A": 0.90, "D": 0.80},
        due_concept_ids=["D"],
    )
    assert plan.steps[0].step_type == "REVIEW"
    assert plan.steps[0].concept_id == "D"
    types = [s.step_type for s in plan.steps]
    assert "NEW" in types  # B is unlocked (A mastered) and unstudied


def test_plan_session_blocked_concepts_listed(planner):
    plan = planner.plan_session(student_id="s1", mastery_map={"A": 0.10})
    assert "C" in plan.blocked_concepts
    step_ids = [s.concept_id for s in plan.steps]
    assert "C" not in step_ids and "B" not in step_ids


def test_plan_session_reinforce_for_weak_unlocked(planner):
    plan = planner.plan_session(
        student_id="s1", mastery_map={"A": 0.95, "B": 0.40}
    )
    reinforce = [s for s in plan.steps if s.step_type == "REINFORCE"]
    assert reinforce and reinforce[0].concept_id == "B"


def test_plan_session_respects_max_new(planner):
    concepts = [_concept(f"K{i}") for i in range(8)]
    wide = LearningPathPlanner(concepts, [])
    plan = wide.plan_session(student_id="s1", mastery_map={}, max_new=2)
    new_steps = [s for s in plan.steps if s.step_type == "NEW"]
    assert len(new_steps) == 2
    assert plan.total_estimated_minutes == sum(s.estimated_minutes for s in plan.steps)
