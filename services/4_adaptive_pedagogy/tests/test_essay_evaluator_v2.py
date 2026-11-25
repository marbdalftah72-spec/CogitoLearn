"""
Tests for the multi-criteria Essay Evaluator v2 and TF-IDF helpers.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import pytest
from common.contracts import BloomLevel, EssayQuestion, RubricCriterion

from bloom_engine.essay_evaluator_v2 import (
    EssayEvaluatorV2,
    cosine_similarity,
    tfidf_vectors,
)


@pytest.fixture
def question():
    return EssayQuestion(
        item_id="essay_1",
        concept_id="concept_attention",
        stem="حلّل آلية الانتباه في معالجة اللغات الطبيعية وقارنها بالأساليب التقليدية.",
        bloom_level=BloomLevel.ANALYZE,
        model_answer=(
            "تقوم آلية الانتباه على حساب أوزان تركيز لكل عنصر في التسلسل، "
            "مما يسمح للنموذج بربط المعطيات بالاستنتاجات وتقليل التشوهات في المعالجة."
        ),
        rubric=[
            RubricCriterion(name="الدقة المفاهيمية", description="تعريف آلية الانتباه وأوزان التركيز", max_score=4.0),
            RubricCriterion(name="العمق التحليلي", description="تحليل آلية العمل والمقارنة بالبدائل", max_score=3.0),
            RubricCriterion(name="الاستدلال", description="حجج وأمثلة مدعومة", max_score=3.0),
        ],
        grounding_span_ids=["span_1"],
    )


STRONG_ANSWER = (
    "تقوم آلية الانتباه على حساب أوزان تركيز لكل عنصر في التسلسل، ولذلك فهي تربط المعطيات بالاستنتاجات مباشرة.\n\n"
    "أولاً، حلّل الباحثون آلية العمل فوجدوا أنها تقلل التشوهات في المعالجة مقارنة بالأساليب التقليدية، "
    "لأن النموذج يركز على العناصر الأكثر صلة بالسياق. ثانياً، قارن بين الانتباه والأساليب التسلسلية القديمة "
    "يُظهر تفوقه في ربط العناصر البعيدة، بينما كانت الأساليب التقليدية تفقد المعلومات الطويلة.\n\n"
    "أخيراً، قيّم النتائج يوضح أن الانتباه حسّن دقة المعالجة، وبالتالي أصبح أساس النماذج الحديثة، "
    "إذن فهو يمثل نقلة نوعية في معالجة اللغات الطبيعية مع الحفاظ على قابلية التحقق من النتائج."
)

WEAK_ANSWER = "الانتباه شيء جيد ومفيد في النماذج الحديثة ويستخدم كثيراً."


def test_strong_answer_scores_higher_than_weak(question):
    ev = EssayEvaluatorV2()
    strong = ev.evaluate(question, STRONG_ANSWER)
    weak = ev.evaluate(question, WEAK_ANSWER)
    assert strong.score_percentage > weak.score_percentage
    assert strong.score_percentage > 40.0


def test_short_answer_scores_zero(question):
    ev = EssayEvaluatorV2()
    result = ev.evaluate(question, "قصير")
    assert result.total_score == 0.0
    assert result.score_percentage == 0.0


def test_result_contract_matches_rubric(question):
    ev = EssayEvaluatorV2()
    result = ev.evaluate(question, STRONG_ANSWER)
    assert set(result.criterion_scores.keys()) == {r.name for r in question.rubric}
    assert result.total_score <= result.max_score
    assert 0.0 <= result.score_percentage <= 100.0
    assert result.feedback_text


def test_content_coverage_identical_text_is_high(question):
    ev = EssayEvaluatorV2()
    coverage = ev.content_coverage(question, question.model_answer)
    assert coverage > 0.5


def test_content_coverage_unrelated_text_is_low(question):
    ev = EssayEvaluatorV2()
    coverage = ev.content_coverage(question, "كرة القدم رياضة شعبية تُلعب في ملاعب كبيرة ويتابعها الملايين حول العالم بشغف")
    assert coverage < 0.35


def test_bloom_verb_score_rewards_target_level_verbs(question):
    ev = EssayEvaluatorV2()
    with_verbs = "حلّل النتائج وقارن بين الأساليب وميّز الفروق وصنّف العناصر وحلّل الآلية بعمق."
    without_verbs = "هذا موضوع مهم وجميل ويستحق الدراسة والاهتمام من الجميع دائماً."
    assert ev.bloom_verb_score(question, with_verbs) > ev.bloom_verb_score(question, without_verbs)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        EssayEvaluatorV2(weights={"content_coverage": 0.5, "structure": 0.1,
                                  "bloom_verbs": 0.1, "coherence": 0.1})


def test_tfidf_cosine_identical_documents():
    vecs = tfidf_vectors(["attention mechanism weights", "attention mechanism weights"])
    assert cosine_similarity(vecs[0], vecs[1]) == pytest.approx(1.0)


def test_tfidf_cosine_disjoint_documents():
    vecs = tfidf_vectors(["alpha beta gamma", "delta epsilon zeta"])
    assert cosine_similarity(vecs[0], vecs[1]) == 0.0
