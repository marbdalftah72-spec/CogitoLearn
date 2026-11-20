"""
Bloom-Taxonomy Multiple Choice Question (MCQ) Generator.
========================================================
Generates curriculum-grounded assessment items mapped across Bloom's Taxonomy
levels (Remember, Understand, Apply, Analyze, Evaluate) with Diagnostic Distractor
Mapping linking incorrect options to specific cognitive misconceptions.
"""

import json
import logging
import uuid
from typing import List, Dict, Optional

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
    MCQOption,
    MCQQuestion,
)

logger = logging.getLogger(__name__)


class MCQGenerator:
    """
    Generates diagnostically-mapped MCQ questions using Gemini models
    with deterministic fallback generation.
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
                logger.warning(f"Could not initialize Gemini model: {e}. Will use algorithmic generator.")

    def generate_mcq(
        self,
        concept: PKGConcept,
        evidence_spans: List[EvidenceSpan],
        bloom_level: BloomLevel,
        num_questions: int = 1,
    ) -> List[MCQQuestion]:
        """
        Generates MCQ questions for a specified Bloom level and concept.
        """
        if self._model and evidence_spans:
            try:
                return self._generate_with_gemini(concept, evidence_spans, bloom_level, num_questions)
            except Exception as e:
                logger.warning(f"Gemini MCQ generation failed: {e}. Falling back to rule-based generation.")

        return self._generate_rule_based(concept, evidence_spans, bloom_level, num_questions)

    def generate_bloom_bank(
        self,
        concept: PKGConcept,
        evidence_spans: List[EvidenceSpan],
        questions_per_level: int = 1,
    ) -> Dict[str, List[MCQQuestion]]:
        """
        Generates a comprehensive question bank spanning all 5 Bloom levels.
        """
        bank: Dict[str, List[MCQQuestion]] = {}
        for level in BloomLevel:
            bank[level.value] = self.generate_mcq(
                concept=concept,
                evidence_spans=evidence_spans,
                bloom_level=level,
                num_questions=questions_per_level,
            )
        return bank

    def _generate_with_gemini(
        self,
        concept: PKGConcept,
        evidence_spans: List[EvidenceSpan],
        bloom_level: BloomLevel,
        num_questions: int,
    ) -> List[MCQQuestion]:
        """
        Uses Gemini to generate strictly structured MCQs.
        """
        evidence_text = "\n".join([f"[{s.span_id}]: {s.content_text}" for s in evidence_spans[:5]])
        prompt = f"""
You are an expert university professor creating diagnostic multiple-choice questions for an NLP and computer science course.
Generate exactly {num_questions} MCQ question(s) at Bloom's Taxonomy level '{bloom_level.value}'.

Target Concept:
- Name (Arabic): {concept.name_ar}
- Name (English): {concept.name_en}
- Type: {concept.concept_type.value}

Grounding Evidence from Lecture:
{evidence_text}

Requirements:
1. The question must strictly test the concept at the '{bloom_level.value}' level.
2. Provide exactly 4 options. Exactly ONE option must be correct (is_correct: true).
3. Every INCORRECT option (distractor) MUST have a specific diagnostic misconception_id (e.g., 'MISC_CONFLATION', 'MISC_OVERSIMPLIFICATION', 'MISC_INVERSE_LOGIC') and an explanation of the underlying misunderstanding.
4. Output MUST be a valid JSON array of objects with the following schema:
[
  {{
    "stem": "Question text in Arabic or English matching lecture language",
    "explanation": "Detailed explanation of correct answer",
    "options": [
      {{"option_id": "opt_a", "text": "...", "is_correct": true, "misconception_id": null, "misconception_explanation": null}},
      {{"option_id": "opt_b", "text": "...", "is_correct": false, "misconception_id": "MISC_1", "misconception_explanation": "..."}},
      {{"option_id": "opt_c", "text": "...", "is_correct": false, "misconception_id": "MISC_2", "misconception_explanation": "..."}},
      {{"option_id": "opt_d", "text": "...", "is_correct": false, "misconception_id": "MISC_3", "misconception_explanation": "..."}}
    ]
  }}
]
Do not include any markdown fences or preamble, output ONLY the raw JSON.
"""
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        data = json.loads(text)
        span_ids = [s.span_id for s in evidence_spans]

        questions = []
        for item in data:
            options = [
                MCQOption(
                    option_id=opt.get("option_id", str(uuid.uuid4())[:8]),
                    text=opt["text"],
                    is_correct=bool(opt.get("is_correct", False)),
                    misconception_id=opt.get("misconception_id"),
                    misconception_explanation=opt.get("misconception_explanation"),
                )
                for opt in item["options"]
            ]
            questions.append(
                MCQQuestion(
                    item_id=str(uuid.uuid4()),
                    concept_id=concept.concept_id,
                    stem=item["stem"],
                    bloom_level=bloom_level,
                    options=options,
                    explanation=item.get("explanation", ""),
                    grounding_span_ids=span_ids,
                )
            )
        return questions

    def _generate_rule_based(
        self,
        concept: PKGConcept,
        evidence_spans: List[EvidenceSpan],
        bloom_level: BloomLevel,
        num_questions: int,
    ) -> List[MCQQuestion]:
        """
        Deterministic, rule-based MCQ generator with diagnostic distractor mapping.
        Guarantees robust testability without external network access.
        """
        span_ids = [s.span_id for s in evidence_spans]
        c_ar = concept.name_ar
        c_en = concept.name_en

        bloom_stems = {
            BloomLevel.REMEMBER: (
                f"ما هو التعريف العلمي الدقيق لمفهوم '{c_ar}' ({c_en}) استناداً للمحاضرة؟",
                f"هو آلية أساسية تمثل {c_en} وتعمل على ضبط ومعالجة البيانات بدقة.",
            ),
            BloomLevel.UNDERSTAND: (
                f"كيف يفسر دور '{c_ar}' ({c_en}) في تحسين استقرار النموذج التعليمي؟",
                f"يعمل على ربط السياق التفاعلي وتخفيض التشتت الحسابي للبيانات.",
            ),
            BloomLevel.APPLY: (
                f"عند تطبيق '{c_ar}' ({c_en}) على مسألة معالجة لغات، ما الناتج المتوقع؟",
                f"تحسين التقارب والحد من الأخطاء التراكمية في الاستدلال.",
            ),
            BloomLevel.ANALYZE: (
                f"ما وجه الاختلاف الجوهري بين '{c_ar}' والبدائل التقليدية غير المهيكلة؟",
                f"القدرة على عزل المتغيرات التراكمية وتقديم تحقق استدلالي مدعوم بالأدلة.",
            ),
            BloomLevel.EVALUATE: (
                f"ما المعيار الأنسب لتقييم كفاءة تطبيق خوارزمية '{c_ar}' في البيئات الواقعية؟",
                f"قياس الدقة التنبؤية مع التحقق من الاتساق المنطقي ومعدل انخفاض الزلات.",
            ),
        }

        stem_text, correct_ans = bloom_stems.get(
            bloom_level,
            (f"اختر العبارة الصحيحة بخصوص {c_ar}:", f"العبارة التي تمثل تطبيق {c_ar} بدقة."),
        )

        questions: List[MCQQuestion] = []
        for i in range(num_questions):
            options = [
                MCQOption(
                    option_id=f"opt_correct_{i}",
                    text=correct_ans,
                    is_correct=True,
                    misconception_id=None,
                    misconception_explanation=None,
                ),
                MCQOption(
                    option_id=f"opt_distractor_1_{i}",
                    text=f"إنه مجرد إجراء شكلي لا يؤثر على الأداء الاستدلالي في {c_en}.",
                    is_correct=False,
                    misconception_id="MISC_SUPERFICIAL_REDUCTION",
                    misconception_explanation=f"الخلط بين الجدوى العملية لـ {c_ar} والعمليات الهامشية.",
                ),
                MCQOption(
                    option_id=f"opt_distractor_2_{i}",
                    text=f"يطبق حصراً كنموذج إحصائي عشوائي دون الحاجة لأدلة موثقة.",
                    is_correct=False,
                    misconception_id="MISC_STOCHASTIC_CONFUSION",
                    misconception_explanation=f"افتراض العشوائية الكاملة وإغفال القيود المنطقية لـ {c_ar}.",
                ),
                MCQOption(
                    option_id=f"opt_distractor_3_{i}",
                    text=f"يعطي نتائج متطابقة تماماً مهما تغيرت المدخلات والسياق المعرفي.",
                    is_correct=False,
                    misconception_id="MISC_INVARIANCE_FALLACY",
                    misconception_explanation=f"افتراض ثبات غير مبرر في سلوك خوارزمية {c_en}.",
                ),
            ]

            questions.append(
                MCQQuestion(
                    item_id=str(uuid.uuid4()),
                    concept_id=concept.concept_id,
                    stem=stem_text,
                    bloom_level=bloom_level,
                    options=options,
                    explanation=f"الإجابة الصحيحة تمثل الاستيعاب المعرفي السليم لـ {c_ar} وفق مستوى {bloom_level.value}.",
                    grounding_span_ids=span_ids,
                )
            )

        return questions
