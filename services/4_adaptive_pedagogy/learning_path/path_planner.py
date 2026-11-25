"""
Learning-path planner over the Pedagogical Knowledge Graph (PKG).
==================================================================
Builds ordered study plans from three fused signals:

* **Prerequisite structure** — PKG ``PREREQUISITE_OF`` edges define a DAG;
  a concept is *unlocked* only when every prerequisite is mastered
  (BKT probability >= mastery gate).
* **Mastery map** — per-concept BKT mastery probabilities (optionally
  moderated by the student's global IRT ability ``theta``: high-ability
  students get a slightly lower gate, struggling students a higher one).
* **FSRS due reviews** — concepts whose spaced-repetition review is due
  are scheduled first, before any new material.

The output is a :class:`SessionPlan`: reviews first (retention), then new
concepts in topological order (progression), each step annotated with a
reason and an estimated duration.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set

from pydantic import BaseModel, Field

from common.contracts import PKGConcept, PKGEdge, RelationType

from config import PATH_MASTERY_GATE, PATH_SESSION_MAX_NEW, PATH_SESSION_MAX_REVIEW

logger = logging.getLogger(__name__)


class SessionStep(BaseModel):
    """One ordered step inside a study session plan."""

    order: int
    concept_id: str
    concept_name_ar: str
    concept_name_en: str
    step_type: str  # REVIEW | NEW | REINFORCE
    current_mastery: float
    reason_ar: str
    estimated_minutes: int


class SessionPlan(BaseModel):
    """A complete next-session plan for one student."""

    student_id: str
    steps: List[SessionStep] = Field(default_factory=list)
    total_estimated_minutes: int = 0
    blocked_concepts: List[str] = Field(default_factory=list)
    summary_ar: str = ""


class LearningPathPlanner:
    """Plans study sessions over a prerequisite DAG."""

    def __init__(
        self,
        concepts: Sequence[PKGConcept],
        edges: Sequence[PKGEdge],
        mastery_gate: float = PATH_MASTERY_GATE,
    ) -> None:
        self.concepts: Dict[str, PKGConcept] = {c.concept_id: c for c in concepts}
        self.mastery_gate = mastery_gate
        # prereqs_of[dst] = [src, ...] where src is a prerequisite of dst.
        self.prereqs_of: Dict[str, List[str]] = {}
        self.dependents_of: Dict[str, List[str]] = {}
        for edge in edges:
            if edge.relation_type == RelationType.PREREQUISITE_OF:
                self.prereqs_of.setdefault(edge.dst_concept_id, []).append(edge.src_concept_id)
                self.dependents_of.setdefault(edge.src_concept_id, []).append(edge.dst_concept_id)

    # -- graph utilities -------------------------------------------------------
    def topological_order(self) -> List[str]:
        """Kahn topological sort of all concept ids (prerequisites first).

        Raises ``ValueError`` if the prerequisite graph contains a cycle.
        """
        indegree: Dict[str, int] = {cid: 0 for cid in self.concepts}
        for dst, srcs in self.prereqs_of.items():
            if dst in indegree:
                indegree[dst] = len([s for s in srcs if s in self.concepts])
        queue = sorted([cid for cid, d in indegree.items() if d == 0])
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dep in sorted(self.dependents_of.get(node, [])):
                if dep not in indegree:
                    continue
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)
        if len(order) != len(self.concepts):
            raise ValueError("Prerequisite graph contains a cycle")
        return order

    def effective_gate(self, theta: Optional[float] = None) -> float:
        """Adjust the mastery gate by global IRT ability (±0.05 band)."""
        if theta is None:
            return self.mastery_gate
        adjustment = max(-0.05, min(0.05, -0.02 * theta))
        return float(max(0.4, min(0.9, self.mastery_gate + adjustment)))

    def unlocked_concepts(
        self, mastery_map: Dict[str, float], theta: Optional[float] = None
    ) -> List[str]:
        """Concepts whose prerequisites are all mastered, in topo order."""
        gate = self.effective_gate(theta)
        unlocked: List[str] = []
        for cid in self.topological_order():
            prereqs = self.prereqs_of.get(cid, [])
            if all(mastery_map.get(p, 0.0) >= gate for p in prereqs):
                unlocked.append(cid)
        return unlocked

    def blocked_by(self, concept_id: str, mastery_map: Dict[str, float],
                   theta: Optional[float] = None) -> List[str]:
        """Unmastered prerequisites currently blocking ``concept_id``."""
        gate = self.effective_gate(theta)
        return [
            p
            for p in self.prereqs_of.get(concept_id, [])
            if mastery_map.get(p, 0.0) < gate
        ]

    # -- session planning --------------------------------------------------------
    def plan_session(
        self,
        student_id: str,
        mastery_map: Dict[str, float],
        due_concept_ids: Optional[Sequence[str]] = None,
        theta: Optional[float] = None,
        max_new: int = PATH_SESSION_MAX_NEW,
        max_review: int = PATH_SESSION_MAX_REVIEW,
    ) -> SessionPlan:
        """Build the next study session: due reviews first, then new concepts.

        * ``REVIEW`` — FSRS-due concepts (already studied, retention at risk).
        * ``REINFORCE`` — unlocked concepts below the gate that are not due.
        * ``NEW`` — unlocked concepts never studied (mastery at prior level).
        """
        due = list(due_concept_ids or [])
        unlocked = set(self.unlocked_concepts(mastery_map, theta))
        steps: List[SessionStep] = []
        order = 0

        def add_step(cid: str, step_type: str, reason: str, minutes: int) -> None:
            nonlocal order
            order += 1
            concept = self.concepts.get(cid)
            steps.append(
                SessionStep(
                    order=order,
                    concept_id=cid,
                    concept_name_ar=concept.name_ar if concept else cid,
                    concept_name_en=concept.name_en if concept else cid,
                    step_type=step_type,
                    current_mastery=round(mastery_map.get(cid, 0.0), 4),
                    reason_ar=reason,
                    estimated_minutes=minutes,
                )
            )

        # 1) Due reviews first — retention before progression.
        for cid in due[:max_review]:
            if cid in self.concepts:
                add_step(cid, "REVIEW", "مراجعة مستحقة وفق جدولة FSRS لتثبيت الذاكرة قبل النسيان.", 6)

        # 2) New / reinforce unlocked concepts in topological order.
        topo = self.topological_order()
        new_count = 0
        for cid in topo:
            if new_count >= max_new:
                break
            if cid in due or cid not in unlocked:
                continue
            mastery = mastery_map.get(cid, 0.0)
            if mastery >= self.effective_gate(theta):
                continue  # already mastered — nothing to schedule
            if cid in mastery_map:
                add_step(cid, "REINFORCE", "مفهوم مفتوح لكن إتقانه دون العتبة؛ يحتاج تعزيزًا موجّهًا.", 12)
            else:
                add_step(cid, "NEW", "مفهوم جديد متطلباته السابقة مكتملة الإتقان.", 15)
            new_count += 1

        blocked = [
            cid
            for cid in topo
            if cid not in unlocked and mastery_map.get(cid, 0.0) < self.effective_gate(theta)
        ]
        total_minutes = sum(s.estimated_minutes for s in steps)
        summary = (
            f"خطة جلسة للطالب: {sum(1 for s in steps if s.step_type == 'REVIEW')} مراجعات مستحقة، "
            f"{sum(1 for s in steps if s.step_type != 'REVIEW')} مفاهيم للتعلم/التعزيز، "
            f"بإجمالي {total_minutes} دقيقة تقديرية."
        )
        return SessionPlan(
            student_id=student_id,
            steps=steps,
            total_estimated_minutes=total_minutes,
            blocked_concepts=blocked,
            summary_ar=summary,
        )
