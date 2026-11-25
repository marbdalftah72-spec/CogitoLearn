#!/usr/bin/env python3
"""
End-to-end demo of CogitoLearn Service 4 v2 with a simulated student.
=====================================================================
Runs a full adaptive-pedagogy loop without any external services:

1. Builds a small prerequisite graph (Tokenization -> Embeddings -> Attention).
2. Simulates a student answering MCQs (with a recurring misconception).
3. Fits BKT parameters from cohort data and reports prediction quality.
4. Estimates the student's IRT ability (theta) and runs an adaptive session.
5. Scores an essay with the multi-criteria v2 evaluator.
6. Prints the fused UnifiedStudentReport incl. the next-session plan.

Usage:
    python -m examples.demo          (from the service root)
    python examples/demo.py          (equivalent)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))
_repo_root = _service_root.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from common.contracts import (  # noqa: E402
    BloomLevel,
    ConceptType,
    EvidenceSpan,
    PKGConcept,
    PKGEdge,
    RelationType,
)

from pedagogy_coordinator import PedagogyCoordinator  # noqa: E402
from irt_engine.irt_model import IRTItem, prob_correct  # noqa: E402


def build_pkg():
    """Three-concept prerequisite chain used by the demo."""
    concepts = [
        PKGConcept(concept_id="tok", course_id="nlp101", name_ar="التقطيع المعجمي",
                   name_en="Tokenization", concept_type=ConceptType.DEFINITION, base_difficulty=0.3),
        PKGConcept(concept_id="emb", course_id="nlp101", name_ar="التضمينات",
                   name_en="Embeddings", concept_type=ConceptType.DEFINITION, base_difficulty=0.5),
        PKGConcept(concept_id="attn", course_id="nlp101", name_ar="آلية الانتباه",
                   name_en="Attention", concept_type=ConceptType.ALGORITHM, base_difficulty=0.7),
    ]
    edges = [
        PKGEdge(edge_id="e1", src_concept_id="tok", dst_concept_id="emb",
                relation_type=RelationType.PREREQUISITE_OF,
                provenance_span_ids=["span_1"], entailment_conf=0.95),
        PKGEdge(edge_id="e2", src_concept_id="emb", dst_concept_id="attn",
                relation_type=RelationType.PREREQUISITE_OF,
                provenance_span_ids=["span_2"], entailment_conf=0.93),
    ]
    spans = [
        EvidenceSpan(span_id="span_1", lecture_id="lec1",
                     content_text="التقطيع أساس التضمينات."),
        EvidenceSpan(span_id="span_2", lecture_id="lec2",
                     content_text="الانتباه يعتمد على التضمينات."),
    ]
    return concepts, edges, spans


def main() -> int:
    rng = random.Random(42)
    concepts, edges, spans = build_pkg()
    by_id = {c.concept_id: c for c in concepts}
    coordinator = PedagogyCoordinator()
    student = "demo_student"

    print("=" * 68)
    print("CogitoLearn Service 4 v2 — End-to-End Demo")
    print("=" * 68)

    # 1) Simulated MCQ practice on Tokenization + Embeddings ---------------
    print("\n[1] Simulated MCQ practice (with a recurring misconception)...")
    for concept_id in ("tok", "emb"):
        package = coordinator.generate_assessment(
            by_id[concept_id], spans, include_essay=False, questions_per_bloom_level=1
        )
        for level_questions in package.mcqs_by_bloom.values():
            for question in level_questions[:2]:
                correct = next(o for o in question.options if o.is_correct)
                wrong = next(o for o in question.options if not o.is_correct)
                # Student answers correctly 70% of the time; when wrong, the
                # distractor's misconception gets recorded by the coordinator.
                chosen = correct if rng.random() < 0.7 else wrong
                feedback = coordinator.process_mcq_response(student, question, chosen.option_id)
        state = coordinator.get_student_mastery(student, concept_id)
        print(f"    {concept_id}: mastery -> {state.bkt_p_mastery:.2f} "
              f"({coordinator.bkt.classify_mastery(state.bkt_p_mastery)})")

    # Force one persistent misconception for the report demo.
    for _ in range(3):
        coordinator.diagnoser.record_error(student, "attn", "MISC_INVERSE_LOGIC")

    # 2) Fit BKT parameters from cohort data --------------------------------
    print("\n[2] Fitting BKT parameters from simulated cohort data...")
    cohort = []
    for _ in range(30):
        learned, seq = False, []
        for _ in range(12):
            seq.append(rng.random() < (0.92 if learned else 0.18))
            if not learned and rng.random() < 0.28:
                learned = True
        cohort.append(seq)
    params, metrics = coordinator.fit_bkt_parameters("tok", cohort)
    print(f"    fitted: P(L0)={params.p_l0:.2f} P(T)={params.p_transit:.2f} "
          f"P(G)={params.p_guess:.2f} P(S)={params.p_slip:.2f}")
    print(f"    prediction quality: AUC={metrics.auc:.3f} RMSE={metrics.rmse:.3f} "
          f"ACC={metrics.accuracy:.3f} (n={metrics.n_predictions})")

    # 3) IRT ability + adaptive session --------------------------------------
    print("\n[3] IRT ability estimation + adaptive item selection...")
    bank = [IRTItem(f"item_{i}", "attn", 1.3, -2.0 + 0.4 * i) for i in range(11)]
    true_theta = 0.8
    estimate, administered = coordinator.irt_selector.run_adaptive_session(
        bank,
        respond=lambda item: rng.random() < prob_correct(
            true_theta, item.discrimination, item.difficulty
        ),
        max_items=8,
    )
    print(f"    true theta={true_theta}  estimated theta={estimate.theta} "
          f"(SE={estimate.standard_error}) after {len(administered)} items")

    # 4) Essay scoring with evaluator v2 --------------------------------------
    print("\n[4] Essay evaluation (multi-criteria v2)...")
    essay_q = coordinator.essay_eval.generate_essay_question(by_id["attn"], spans)
    answer = (
        "تقوم آلية الانتباه على حساب أوزان تركيز لكل عنصر في التسلسل، ولذلك تربط المعطيات بالاستنتاجات.\n\n"
        "أولاً، حلّل آلية العمل يوضح أنها تقلل التشوهات في المعالجة لأن النموذج يركز على العناصر ذات الصلة. "
        "ثانياً، قارن بين الانتباه والأساليب التقليدية يُظهر تفوقه في ربط العناصر البعيدة، بينما تفقد الأساليب القديمة السياق الطويل.\n\n"
        "أخيراً، قيّم النتائج يثبت أن الانتباه حسّن الدقة، وبالتالي أصبح أساس النماذج الحديثة في معالجة اللغات الطبيعية."
    )
    essay_result = coordinator.evaluate_essay_v2(essay_q, answer)
    print(f"    score: {essay_result.total_score}/{essay_result.max_score} "
          f"({essay_result.score_percentage}%)")
    print(f"    feedback: {essay_result.feedback_text}")

    # 5) Unified fused report ---------------------------------------------------
    print("\n[5] Unified student report (BKT + IRT + FSRS-5 + diagnoses)...")
    report = coordinator.get_unified_student_report(
        student,
        irt_items=bank[:6],
        irt_responses=[True, True, False, True, True, False],
        pkg_concepts=concepts,
        pkg_edges=edges,
    )
    print(f"    concepts tracked : {report.total_concepts_tracked} "
          f"(mastered {report.mastered_count}, learning {report.learning_count}, "
          f"unlearned {report.unlearned_count})")
    print(f"    average mastery  : {report.average_mastery:.2f}")
    print(f"    IRT theta        : {report.irt_theta} (SE {report.irt_theta_standard_error})")
    print(f"    due reviews      : {report.due_review_count} now, "
          f"{report.forecast_reviews_next_7_days} forecast over 7 days")
    print(f"    persistent misc. : {report.persistent_misconception_count}")
    for diag in report.top_misconceptions[:2]:
        print(f"      - {diag.misconception_id} on {diag.concept_id}: "
              f"confidence {diag.confidence:.0%}, risk {diag.risk_score:.2f}")
    if report.next_session_plan:
        print("    next session plan:")
        for step in report.next_session_plan.steps:
            print(f"      {step.order}. [{step.step_type}] {step.concept_id} "
                  f"({step.estimated_minutes} min) — {step.reason_ar}")
    print("    recommendations:")
    for rec in report.recommendations_ar:
        print(f"      * {rec}")

    print("\nDemo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
