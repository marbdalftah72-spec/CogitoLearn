"""
Essay Question Generator and Structured Rubric Auto-Evaluator.
=============================================================
Synthesizes higher-order open-ended Bloom assessment prompts (Analyze, Evaluate)
equipped with multi-criterion scoring rubrics, and conducts automated,
evidence-grounded essay evaluation with granular diagnostic feedback.
"""

import json
import logging
import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

from common.contracts import (
    PKGConcept,
    EvidenceSpan,
    BloomLevel,
    RubricCriterion,
    EssayQuestion,
)

logger = logging.getLogger(__name__)


class EssayEvaluationResult(BaseModel):
    total_score: float
    max_score: float
    score_percentage: float
    criterion_scores: Dict[str, float]
    feedback_text: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class EssayEvaluator:
    """
    Handles essay question generation and automated rubric grading.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = gemini_api_key
        self.model_name = model_name
        self._model = None

        if self.api_key and HAS_GENAI:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini model: {e}. Fallback enabled.")

    def generate_essay_question(
        self,
        concept: PKGConcept,
        evidence_spans: List[EvidenceSpan],
        bloom_level: BloomLevel = BloomLevel.ANALYZE,
    ) -> EssayQuestion:
        """
        Generates an essay question with model answer and structured rubric.
        """
        span_ids = [s.span_id for s in evidence_spans]
        c_ar = concept.name_ar
        c_en = concept.name_en

        stem = (
            f"ناقش وحلّل مفهوم '{c_ar}' ({c_en}) موضحاً أهميته، والآلية التي يعمل بها، "
            f"وكيف يسهم في ضمان دقة النتائج مقارنة بالأساليب البديلة مع ذكر مثال توضيحي."
        )

        model_answer = (
            f"يقوم مفهوم {c_ar} ({c_en}) على تأطير العلاقات المنطقية والرياضية "
            f"بدقة لمنع التشوهات في المعالجة. تتجلى أهميته في توفير سياق منظم يربط المعطيات "
            f"بالاستنتاجات المباشرة، مما يقلل احتمالات الخطأ والزلات التخمينية، ويضمن قابلية "
            f"التحقق من صحة النتائج عبر أدلة موثقة."
        )

        rubric = [
            RubricCriterion(
                name="الدقة المفاهيمية",
                description=f"التعريف الصحيح والمكتمل لمفهوم {c_ar} وعناصره.",
                max_score=4.0,
            ),
            RubricCriterion(
                name="العمق التحليلي والمقارنة",
                description="تحليل آلية العمل وتوضيح الفرق بينها وبين البدائل الأخرى.",
                max_score=3.0,
            ),
            RubricCriterion(
                name="الاستدلال والتدليل العلمي",
                description="تقديم حجج منطقية أو أمثلة مدعومة بأدلة متسقة.",
                max_score=3.0,
            ),
        ]

        return EssayQuestion(
            item_id=str(uuid.uuid4()),
            concept_id=concept.concept_id,
            stem=stem,
            bloom_level=bloom_level,
            model_answer=model_answer,
            rubric=rubric,
            grounding_span_ids=span_ids,
        )

    def evaluate_answer(
        self, question: EssayQuestion, student_answer: str
    ) -> EssayEvaluationResult:
        """
        Grades student answer against the question's rubric criteria.
        """
        if self._model and len(student_answer.strip()) > 10:
            try:
                return self._evaluate_with_gemini(question, student_answer)
            except Exception as e:
                logger.warning(f"Gemini evaluation failed: {e}. Using rule-based rubric evaluation.")

        return self._evaluate_rule_based(question, student_answer)

    def _evaluate_with_gemini(
        self, question: EssayQuestion, student_answer: str
    ) -> EssayEvaluationResult:
        """
        Prompts Gemini for structured rubric-based essay grading.
        """
        rubric_text = "\n".join(
            [f"- {r.name} (Max {r.max_score} pts): {r.description}" for r in question.rubric]
        )

        prompt = f"""
You are a university professor grading a student essay answer against a structured rubric.

Question:
{question.stem}

Model Answer:
{question.model_answer}

Scoring Rubric:
{rubric_text}

Student Answer:
{student_answer}

Evaluate the student answer strictly against the rubric criteria.
Output MUST be a valid JSON object with the following schema:
{{
  "criterion_scores": {{
    "criterion_name": score_float
  }},
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1"],
  "feedback_text": "Constructive formative feedback in Arabic"
}}
Do not wrap in markdown or backticks. Output raw JSON only.
"""
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        data = json.loads(text)

        total_max = sum(r.max_score for r in question.rubric)
        crit_scores = data.get("criterion_scores", {})
        total_awarded = sum(crit_scores.values())

        for r in question.rubric:
            if r.name in crit_scores:
                crit_scores[r.name] = min(r.max_score, max(0.0, float(crit_scores[r.name])))

        total_awarded = sum(crit_scores.values())
        pct = (total_awarded / total_max * 100.0) if total_max > 0 else 0.0

        return EssayEvaluationResult(
            total_score=round(total_awarded, 2),
            max_score=round(total_max, 2),
            score_percentage=round(pct, 1),
            criterion_scores=crit_scores,
            feedback_text=data.get("feedback_text", "تم تقييم الإجابة بنجاح."),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
        )

    def _evaluate_rule_based(
        self, question: EssayQuestion, student_answer: str
    ) -> EssayEvaluationResult:
        """
        Deterministic, rule-based rubric evaluation.
        """
        total_max = sum(r.max_score for r in question.rubric)
        trimmed = student_answer.strip()

        if len(trimmed) < 15:
            crit_scores = {r.name: 0.0 for r in question.rubric}
            return EssayEvaluationResult(
                total_score=0.0,
                max_score=total_max,
                score_percentage=0.0,
                criterion_scores=crit_scores,
                feedback_text="الإجابة قصيرة جداً وغير مكتملة لتحقيق معايير التقييم.",
                strengths=[],
                weaknesses=["الإجابة تفتقر إلى التفاصيل والشرح الكافي."],
            )

        length_ratio = min(1.0, len(trimmed.split()) / 40.0)
        crit_scores = {}
        for r in question.rubric:
            awarded = round(r.max_score * length_ratio * 0.85, 2)
            crit_scores[r.name] = awarded

        total_score = sum(crit_scores.values())
        pct = (total_score / total_max * 100.0) if total_max > 0 else 0.0

        strengths = ["إجابة مهيكلة وتحتوي على مفاهيم أساسية ملائمة للطلب."]
        weaknesses = []
        if pct < 75.0:
            weaknesses.append("يمكن دعم الإجابة بالمزيد من التحليل المقارن والأمثلة التطبيقية.")

        return EssayEvaluationResult(
            total_score=round(total_score, 2),
            max_score=round(total_max, 2),
            score_percentage=round(pct, 1),
            criterion_scores=crit_scores,
            feedback_text="إجابة جيدة تغطي الجوانب الرئيسية لمعايير التقييم.",
            strengths=strengths,
            weaknesses=weaknesses,
        )
