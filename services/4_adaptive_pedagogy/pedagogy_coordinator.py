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
from fsrs_scheduler.memory_optimizer import FSRSScheduler
from misconception_analyzer.error_diagnoser import (
    ErrorDiagnoser,
    RemediationPlan,
)
from bloom_engine.mcq_generator import MCQGenerator
from bloom_engine.essay_evaluator import (
    EssayEvaluator,
    EssayEvaluationResult,
)

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
        self.diagnoser = ErrorDiagnoser(gemini_api_key=gemini_api_key)
        self.mcq_gen = MCQGenerator(gemini_api_key=gemini_api_key)
        self.essay_eval = EssayEvaluator(gemini_api_key=gemini_api_key)

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
