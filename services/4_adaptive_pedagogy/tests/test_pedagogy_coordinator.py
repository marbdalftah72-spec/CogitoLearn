"""
End-to-end Integration Tests for Pedagogy Coordinator.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

from common.contracts import (
    PKGConcept,
    ConceptType,
    EvidenceSpan,
)
from pedagogy_coordinator import PedagogyCoordinator


def test_coordinator_full_cycle():
    coord = PedagogyCoordinator()
    concept = PKGConcept(
        concept_id="c_pos",
        course_id="nlp_101",
        name_ar="تمييز أجزاء الكلام",
        name_en="Part-of-Speech Tagging",
        concept_type=ConceptType.ALGORITHM,
    )
    span = EvidenceSpan(
        span_id="sp_pos_1",
        lecture_id="lec_pos",
        content_text="تحديد الصنف النحوي لكل كلمة في الجملة.",
    )

    # 1. Generate assessment
    package = coord.generate_assessment(concept, [span], include_essay=True)
    assert len(package.mcqs_by_bloom) == 5
    assert len(package.essay_questions) == 1

    # 2. Student takes an MCQ
    remember_mcq = package.mcqs_by_bloom["REMEMBER"][0]
    correct_opt = next(o for o in remember_mcq.options if o.is_correct)
    wrong_opt = next(o for o in remember_mcq.options if not o.is_correct)

    # First attempt: wrong answer
    fb1 = coord.process_mcq_response("std_test", remember_mcq, wrong_opt.option_id)
    assert fb1.is_correct is False
    assert fb1.misconception_id is not None
    assert fb1.next_review_due is not None

    # Second attempt: correct answer
    fb2 = coord.process_mcq_response("std_test", remember_mcq, correct_opt.option_id)
    assert fb2.is_correct is True
    assert fb2.updated_mastery > fb1.updated_mastery

    # 3. Student submits essay response
    essay_q = package.essay_questions[0]
    essay_fb = coord.process_essay_response(
        "std_test",
        essay_q,
        "يعمل تمييز أجزاء الكلام على تصنيف كل كلمة إلى اسم أو فعل أو حرف بالاعتماد على سياق الجملة.",
    )
    assert essay_fb.evaluation.total_score > 0

    # 4. Query student dashboard
    dashboard = coord.get_student_dashboard("std_test")
    assert dashboard.total_concepts_tracked == 1
    assert dashboard.student_id == "std_test"
