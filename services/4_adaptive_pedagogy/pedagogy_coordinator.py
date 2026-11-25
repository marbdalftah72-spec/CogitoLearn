"""
Pedagogy Coordinator for CogitoLearn Service 4.
===============================================
Central orchestrator linking Bloom Question Generation, Bayesian Knowledge Tracing (BKT),
FSRS Spaced Repetition Scheduling, and PKG-Driven Misconception Remediation.
Provides high-level APIs consumed by Service 5 (Interactive UI) and Service 3 (Router/RAG).
"""

import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

from datetime import datetime, timezone
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from common.contracts import (
    PKGConcept,
    PKGEdge,
    EvidenceSpan,
    BloomLevel,
    MCQQuestion,
    EssayQuestion,
    StudentConceptMastery,
)

from bkt_engine.bkt_updater import BKTUpdater
from bkt_engine.student_state_store import StudentStateStore
from bkt_engine.bkt_fitter import (
    BKTParameterFitter,
    BKTParameters,
    BKTPredictionEvaluator,
    PredictionMetrics,
)
from fsrs_scheduler.memory_optimizer import FSRSScheduler
from fsrs_scheduler.fsrs5_optimizer import FSRS5Scheduler
from misconception_analyzer.error_diagnoser import (
    ErrorDiagnoser,
    RemediationPlan,
)
from misconception_analyzer.advanced_diagnoser import (
    AdvancedErrorDiagnoser,
    DiagnosedMisconception,
)
from bloom_engine.mcq_generator import MCQGenerator
from bloom_engine.essay_evaluator import (
    EssayEvaluator,
    EssayEvaluationResult,
)
from bloom_engine.essay_evaluator_v2 import EssayEvaluatorV2
from irt_engine.irt_model import (
    AbilityEstimate,
    AbilityEstimator,
    AdaptiveItemSelector,
    IRTItem,
)
from learning_path.path_planner import LearningPathPlanner, SessionPlan

logger = logging.getLogger(__name__)


class AssessmentPackage(BaseModel):
    concept_id: str
    concept_name_ar: str
    concept_name_en: str
    mcqs_by_bloom: Dict[str, List[MCQQuestion]]
    essay_questions: List[EssayQuestion] = Field(default_factory=list)
    generated_at: str


class ResponseFeedback(BaseModel):
    student_id: str
    concept_id: str
    item_id: str
    is_correct: bool
    selected_option_id: str
    correct_option_id: str
    misconception_id: Optional[str] = None
    misconception_explanation: Optional[str] = None
    previous_mastery: float
    updated_mastery: float
    mastery_level: str
    next_review_due: Optional[str] = None
    feedback_message: str


class EssayFeedback(BaseModel):
    student_id: str
    concept_id: str
    item_id: str
    evaluation: EssayEvaluationResult
    previous_mastery: float
    updated_mastery: float
    mastery_level: str
    next_review_due: Optional[str] = None


class UnifiedStudentReport(BaseModel):
    """v2 fused report: BKT mastery + IRT ability + FSRS load + diagnoses."""

    student_id: str
    generated_at: str
    total_concepts_tracked: int
    mastered_count: int
    learning_count: int
    unlearned_count: int
    average_mastery: float
    irt_theta: Optional[float] = None
    irt_theta_standard_error: Optional[float] = None
    due_review_count: int = 0
    forecast_reviews_next_7_days: int = 0
    top_misconceptions: List[DiagnosedMisconception] = Field(default_factory=list)
    persistent_misconception_count: int = 0
    next_session_plan: Optional[SessionPlan] = None
    recommendations_ar: List[str] = Field(default_factory=list)


class StudentDashboard(BaseModel):
    student_id: str
    total_concepts_tracked: int
    mastered_count: int
    learning_count: int
    unlearned_count: int
    average_mastery: float
    weak_concepts: List[StudentConceptMastery]
    due_reviews: List[StudentConceptMastery]
    diagnosed_misconceptions_summary: Dict[str, List[str]]
    recommended_action: str


class PedagogyCoordinator:
    """
    Main Service 4 coordinator implementing the end-to-end adaptive pedagogy lifecycle.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        state_persistence_path: Optional[str] = None,
        desired_retention: float = 0.90,
    ):
        self.api_key = gemini_api_key
        self.bkt = BKTUpdater()
        self.state_store = StudentStateStore(persistence_filepath=state_persistence_path)
        self.fsrs = FSRSScheduler(desired_retention=desired_retention)
        # v2: AdvancedErrorDiagnoser subclasses ErrorDiagnoser, so every v1
        # call site keeps working while gaining confidence-scored diagnoses.
        self.diagnoser = AdvancedErrorDiagnoser(gemini_api_key=gemini_api_key)
        self.mcq_gen = MCQGenerator(gemini_api_key=gemini_api_key)
        self.essay_eval = EssayEvaluator(gemini_api_key=gemini_api_key)
        # --- v2 engines -------------------------------------------------
        self.bkt_fitter = BKTParameterFitter()
        self.fsrs5 = FSRS5Scheduler(desired_retention=desired_retention)
        self.essay_eval_v2 = EssayEvaluatorV2()
        self.irt_estimator = AbilityEstimator()
        self.irt_selector = AdaptiveItemSelector(estimator=self.irt_estimator)

    def generate_assessment(
        self,
        concept: PKGConcept,
        evidence_spans: List[EvidenceSpan],
        include_essay: bool = True,
        questions_per_bloom_level: int = 1,
    ) -> AssessmentPackage:
        """
        Synthesizes an assessment package encompassing Bloom MCQ levels and essay questions.
        """
        bloom_bank = self.mcq_gen.generate_bloom_bank(
            concept=concept,
            evidence_spans=evidence_spans,
            questions_per_level=questions_per_bloom_level,
        )

        essay_questions: List[EssayQuestion] = []
        if include_essay:
            essay = self.essay_eval.generate_essay_question(
                concept=concept,
                evidence_spans=evidence_spans,
                bloom_level=BloomLevel.ANALYZE,
            )
            essay_questions.append(essay)

        return AssessmentPackage(
            concept_id=concept.concept_id,
            concept_name_ar=concept.name_ar,
            concept_name_en=concept.name_en,
            mcqs_by_bloom=bloom_bank,
            essay_questions=essay_questions,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def process_mcq_response(
        self,
        student_id: str,
        question: MCQQuestion,
        selected_option_id: str,
    ) -> ResponseFeedback:
        """
        Evaluates student choice, diagnoses misconceptions, updates BKT mastery,
        and reschedules spaced review via FSRS.
        """
        # 1. Identify selected option and correct option
        selected_option = next(
            (opt for opt in question.options if opt.option_id == selected_option_id), None
        )
        correct_option = next((opt for opt in question.options if opt.is_correct), None)

        if not selected_option or not correct_option:
            raise ValueError(f"Option ID {selected_option_id} not found in question {question.item_id}")

        is_correct = selected_option.is_correct
        misc_id = selected_option.misconception_id
        misc_exp = selected_option.misconception_explanation

        # 2. Record misconception if incorrect
        if not is_correct and misc_id:
            self.diagnoser.record_error(student_id, question.concept_id, misc_id)

        # 3. Retrieve and update BKT mastery
        state = self.state_store.get_or_create_state(student_id, question.concept_id)
        prev_mastery = state.bkt_p_mastery

        updated_state = self.bkt.update_mastery(state, is_correct)

        # Update diagnosed misconceptions list in state
        if misc_id and misc_id not in updated_state.diagnosed_misconceptions:
            updated_state = updated_state.model_copy(
                update={
                    "diagnosed_misconceptions": updated_state.diagnosed_misconceptions + [misc_id]
                }
            )

        # 4. Update FSRS memory schedule
        # Rating: 3 (Good) on correct, 1 (Again) on error
        fsrs_rating = 3 if is_correct else 1
        scheduled_state = self.fsrs.schedule_review(updated_state, rating=fsrs_rating)

        # Save to store
        self.state_store.update_state(scheduled_state)

        # 5. Formulate feedback
        classification = self.bkt.classify_mastery(scheduled_state.bkt_p_mastery)
        if is_correct:
            msg = f"إجابة صحيحة وممتازة! ارتفع مستوى إتقانك للمفهوم إلى {round(scheduled_state.bkt_p_mastery * 100, 1)}%."
        else:
            msg = f"إجابة غير دقيقة. {misc_exp or question.explanation}"

        return ResponseFeedback(
            student_id=student_id,
            concept_id=question.concept_id,
            item_id=question.item_id,
            is_correct=is_correct,
            selected_option_id=selected_option_id,
            correct_option_id=correct_option.option_id,
            misconception_id=misc_id,
            misconception_explanation=misc_exp,
            previous_mastery=prev_mastery,
            updated_mastery=scheduled_state.bkt_p_mastery,
            mastery_level=classification,
            next_review_due=scheduled_state.next_review_due,
            feedback_message=msg,
        )

    def process_essay_response(
        self,
        student_id: str,
        question: EssayQuestion,
        student_answer: str,
    ) -> EssayFeedback:
        """
        Evaluates essay answer with rubrics, updates BKT based on percentage score,
        and reschedules review.
        """
        eval_result = self.essay_eval.evaluate_answer(question, student_answer)

        # Binary equivalent for BKT update: score >= 70% is considered successful acquisition
        is_success = eval_result.score_percentage >= 70.0

        state = self.state_store.get_or_create_state(student_id, question.concept_id)
        prev_mastery = state.bkt_p_mastery

        updated_state = self.bkt.update_mastery(state, is_success)

        # Map percentage to FSRS rating
        if eval_result.score_percentage >= 85.0:
            rating = 4  # Easy
        elif eval_result.score_percentage >= 70.0:
            rating = 3  # Good
        elif eval_result.score_percentage >= 50.0:
            rating = 2  # Hard
        else:
            rating = 1  # Again

        scheduled_state = self.fsrs.schedule_review(updated_state, rating=rating)
        self.state_store.update_state(scheduled_state)

        classification = self.bkt.classify_mastery(scheduled_state.bkt_p_mastery)

        return EssayFeedback(
            student_id=student_id,
            concept_id=question.concept_id,
            item_id=question.item_id,
            evaluation=eval_result,
            previous_mastery=prev_mastery,
            updated_mastery=scheduled_state.bkt_p_mastery,
            mastery_level=classification,
            next_review_due=scheduled_state.next_review_due,
        )

    def get_student_mastery(self, student_id: str, concept_id: str) -> StudentConceptMastery:
        """
        Retrieves live mastery state for Service 3 (Router/RAG) difficulty adaptation.
        """
        return self.state_store.get_or_create_state(student_id, concept_id)

    def get_student_dashboard(self, student_id: str) -> StudentDashboard:
        """
        Synthesizes aggregate telemetry for Service 5 (Student UI Dashboard).
        """
        summary = self.state_store.get_mastery_summary(student_id)
        profile = self.state_store.get_student_profile(student_id)
        weak = self.state_store.get_weak_concepts(student_id, threshold=0.50)
        due = self.fsrs.get_due_concepts(profile)
        errors = self.diagnoser.get_error_patterns(student_id)

        if len(due) > 0:
            rec_action = f"لديك {len(due)} مفاهيم مستحقة للمراجعة اليوم لتثبيت الذاكرة وفق خوارزمية FSRS."
        elif len(weak) > 0:
            rec_action = f"ينصح بمراجعة {len(weak)} مفاهيم تواجه فيها صعوبة وبدء خطة علاجية."
        else:
            rec_action = "أداؤك ممتاز ومستمر في التقدم المعرفي! استمر في الاختبارات الدورية."

        return StudentDashboard(
            student_id=student_id,
            total_concepts_tracked=summary["total_concepts"],
            mastered_count=summary["mastered_count"],
            learning_count=summary["learning_count"],
            unlearned_count=summary["unlearned_count"],
            average_mastery=summary["average_mastery"],
            weak_concepts=weak,
            due_reviews=due,
            diagnosed_misconceptions_summary=errors,
            recommended_action=rec_action,
        )

    def get_remediation_plan(
        self,
        student_id: str,
        target_concept: PKGConcept,
        pkg_concepts: List[PKGConcept],
        pkg_edges: List[PKGEdge],
        evidence_spans: List[EvidenceSpan],
    ) -> RemediationPlan:
        """
        Traverses PKG prerequisite structure and synthesizes targeted remediation plan.
        """
        profile = self.state_store.get_student_profile(student_id)
        gaps = self.diagnoser.find_prerequisite_gaps(
            target_concept_id=target_concept.concept_id,
            pkg_concepts=pkg_concepts,
            pkg_edges=pkg_edges,
            mastery_states=profile,
            mastery_threshold=0.65,
        )
        return self.diagnoser.generate_remediation_plan(
            student_id=student_id,
            target_concept=target_concept,
            prerequisite_gaps=gaps,
            evidence_spans=evidence_spans,
            mastery_states=profile,
        )

    def get_due_reviews(
        self, student_id: str, current_date: Optional[datetime] = None
    ) -> List[StudentConceptMastery]:
        """
        Fetches all concepts that are currently due for spaced repetition review.
        """
        profile = self.state_store.get_student_profile(student_id)
        return self.fsrs.get_due_concepts(profile, current_date=current_date)

    # =====================================================================
    # v2 — advanced professional capabilities
    # =====================================================================

    def fit_bkt_parameters(
        self,
        concept_id: str,
        response_sequences: List[List[bool]],
        method: str = "em",
    ) -> tuple[BKTParameters, PredictionMetrics]:
        """
        Learn BKT parameters for ``concept_id`` from cohort response data.

        The fitted slip/guess/transit values are written back into every
        stored state for the concept, and one-step-ahead prediction metrics
        (AUC / RMSE / accuracy / log-loss) are returned alongside.
        """
        params = self.bkt_fitter.fit(response_sequences, method=method)
        metrics = BKTPredictionEvaluator(params).evaluate(response_sequences)
        # Persist fitted parameters into stored states for this concept.
        for state in list(self.state_store._store.values()):
            if state.concept_id == concept_id:
                self.state_store.update_state(
                    state.model_copy(update=params.as_state_update())
                )
        logger.info(
            "Fitted BKT for %s: %s (AUC=%.3f)", concept_id, params, metrics.auc
        )
        return params, metrics

    def estimate_student_ability(
        self, items: List[IRTItem], responses: List[bool]
    ) -> AbilityEstimate:
        """Estimate a student's latent ability (theta) with the 2PL model."""
        return self.irt_estimator.estimate(items, responses)

    def evaluate_essay_v2(
        self, question: EssayQuestion, student_answer: str
    ) -> EssayEvaluationResult:
        """Score an essay with the deterministic multi-criteria v2 evaluator."""
        return self.essay_eval_v2.evaluate(question, student_answer)

    def plan_next_session(
        self,
        student_id: str,
        pkg_concepts: List[PKGConcept],
        pkg_edges: List[PKGEdge],
        theta: Optional[float] = None,
    ) -> SessionPlan:
        """
        Build the next study session from the prerequisite graph, the
        student's BKT mastery map and the FSRS due list.
        """
        profile = self.state_store.get_student_profile(student_id)
        mastery_map = {s.concept_id: s.bkt_p_mastery for s in profile}
        due_ids = [s.concept_id for s in self.fsrs.get_due_concepts(profile)]
        planner = LearningPathPlanner(pkg_concepts, pkg_edges)
        return planner.plan_session(
            student_id=student_id,
            mastery_map=mastery_map,
            due_concept_ids=due_ids,
            theta=theta,
        )

    def get_unified_student_report(
        self,
        student_id: str,
        irt_items: Optional[List[IRTItem]] = None,
        irt_responses: Optional[List[bool]] = None,
        pkg_concepts: Optional[List[PKGConcept]] = None,
        pkg_edges: Optional[List[PKGEdge]] = None,
    ) -> UnifiedStudentReport:
        """
        Fuse all four v2 signal families into one student report:

        1. **BKT** — mastery distribution across tracked concepts.
        2. **IRT** — latent ability theta (when item responses are given).
        3. **FSRS** — due reviews now + FSRS-5 workload forecast (7 days).
        4. **Misconceptions** — confidence-scored diagnoses, riskiest first.

        When PKG structure is supplied, a concrete next-session plan is
        attached as well.
        """
        summary = self.state_store.get_mastery_summary(student_id)
        profile = self.state_store.get_student_profile(student_id)
        due = self.fsrs.get_due_concepts(profile)

        theta: Optional[float] = None
        theta_se: Optional[float] = None
        if irt_items and irt_responses:
            estimate = self.irt_estimator.estimate(irt_items, irt_responses)
            theta, theta_se = estimate.theta, estimate.standard_error

        forecast = self.fsrs5.forecast_workload(profile, horizon_days=7)
        forecast_total = sum(int(day["due_count"]) for day in forecast)

        diagnoses = self.diagnoser.diagnose_student(student_id)
        persistent = [d for d in diagnoses if d.is_persistent]

        session_plan: Optional[SessionPlan] = None
        if pkg_concepts is not None and pkg_edges is not None:
            session_plan = self.plan_next_session(
                student_id, pkg_concepts, pkg_edges, theta=theta
            )

        recommendations: List[str] = []
        if due:
            recommendations.append(
                f"ابدأ بـ {len(due)} مراجعات مستحقة اليوم لتثبيت الذاكرة (FSRS)."
            )
        for diag in diagnoses[:3]:
            if diag.remediation_action_ar:
                recommendations.append(
                    f"عالج «{diag.pattern_name_ar or diag.misconception_id}» "
                    f"(ثقة التشخيص {diag.confidence:.0%}): {diag.remediation_action_ar}"
                )
        if theta is not None and theta < -0.5:
            recommendations.append(
                "قدرة الطالب التقديرية (IRT) دون المتوسط؛ خفّف صعوبة المفردات القادمة واعتمد التدرّج."
            )
        if not recommendations:
            recommendations.append("لا إجراءات عاجلة؛ واصل خطة الجلسات المنتظمة.")

        return UnifiedStudentReport(
            student_id=student_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_concepts_tracked=summary["total_concepts"],
            mastered_count=summary["mastered_count"],
            learning_count=summary["learning_count"],
            unlearned_count=summary["unlearned_count"],
            average_mastery=summary["average_mastery"],
            irt_theta=theta,
            irt_theta_standard_error=theta_se,
            due_review_count=len(due),
            forecast_reviews_next_7_days=forecast_total,
            top_misconceptions=diagnoses[:5],
            persistent_misconception_count=len(persistent),
            next_session_plan=session_plan,
            recommendations_ar=recommendations,
        )
