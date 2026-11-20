"""
Tests for Bloom Engine (MCQ Generator and Essay Evaluator).
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

from common.contracts import (
    PKGConcept,
    ConceptType,
    BloomLevel,
    EvidenceSpan,
)
from bloom_engine.mcq_generator import MCQGenerator
from bloom_engine.essay_evaluator import EssayEvaluator


def test_mcq_generation_distractor_mapping():
    generator = MCQGenerator()
    concept = PKGConcept(
        concept_id="c_embed",
        course_id="nlp_101",
        name_ar="التضمين الدلالي",
        name_en="Word Embeddings",
        concept_type=ConceptType.DEFINITION,
    )
    span = EvidenceSpan(
        span_id="sp_1",
        lecture_id="lec_1",
        content_text="التضمين الدلالي يحول الكلمات إلى متجهات كثيفة.",
    )

    questions = generator.generate_mcq(
        concept=concept,
        evidence_spans=[span],
        bloom_level=BloomLevel.UNDERSTAND,
        num_questions=1,
    )

    assert len(questions) == 1
    q = questions[0]
    assert q.concept_id == "c_embed"
    assert q.bloom_level == BloomLevel.UNDERSTAND
    assert len(q.options) == 4

    correct_opts = [o for o in q.options if o.is_correct]
    wrong_opts = [o for o in q.options if not o.is_correct]

    assert len(correct_opts) == 1
    assert len(wrong_opts) == 3

    for opt in wrong_opts:
        assert opt.misconception_id is not None
        assert opt.misconception_explanation is not None


def test_bloom_question_bank():
    generator = MCQGenerator()
    concept = PKGConcept(
        concept_id="c_rnn",
        course_id="nlp_101",
        name_ar="الشبكات العصبية المتكررة",
        name_en="RNNs",
        concept_type=ConceptType.ALGORITHM,
    )

    bank = generator.generate_bloom_bank(concept=concept, evidence_spans=[])
    for level in BloomLevel:
        assert level.value in bank
        assert len(bank[level.value]) == 1


def test_essay_generation_and_evaluation():
    evaluator = EssayEvaluator()
    concept = PKGConcept(
        concept_id="c_transformer",
        course_id="nlp_101",
        name_ar="المحولات",
        name_en="Transformers",
        concept_type=ConceptType.ALGORITHM,
    )

    question = evaluator.generate_essay_question(
        concept=concept,
        evidence_spans=[],
        bloom_level=BloomLevel.EVALUATE,
    )

    assert question.concept_id == "c_transformer"
    assert len(question.rubric) > 0

    good_answer = (
        "تعد المحولات نقلة نوعية في معالجة اللغات الطبيعية لأنها تعتمد على آلية الانتباه الذاتي "
        "بدلاً من المعالجة التسلسلية، مما يتيح التوازي الحسابي والتقاط العلاقات بعيدة المدى."
    )
    result = evaluator.evaluate_answer(question, good_answer)
    assert result.total_score > 0.0
    assert result.max_score > 0.0
    assert result.score_percentage > 50.0
    assert len(result.criterion_scores) == len(question.rubric)

    bad_result = evaluator.evaluate_answer(question, "مش عارف")
    assert bad_result.total_score == 0.0
    assert bad_result.score_percentage == 0.0
