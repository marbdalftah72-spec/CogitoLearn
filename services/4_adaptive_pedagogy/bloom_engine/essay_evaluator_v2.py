"""
Essay Evaluator v2 — deterministic multi-criteria scoring.
==========================================================
Upgrades the v1 rule-based fallback (which scored mostly by length) to a
transparent, reproducible four-criteria model:

1. **Content coverage (40%)** — TF-IDF cosine similarity between the
   student answer and the reference concept set (model answer + rubric
   descriptions), computed over a bilingual (Arabic/English) tokeniser.
2. **Structure (20%)** — paragraphing, length adequacy and discourse
   markers (introduction / body / conclusion signals).
3. **Bloom verb usage (25%)** — density of cognitive verbs mapped to the
   target Bloom level (e.g. حلّل، قارن، قيّم for ANALYZE/EVALUATE).
4. **Coherence (15%)** — connective density and sentence-length variance
   (penalising both telegram-style fragments and run-on monotony).

The weighted sum yields a 0..100 percentage plus per-criterion sub-scores
mapped back onto the question's rubric criteria proportionally, so the
v1 :class:`EssayEvaluationResult` contract is preserved.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from common.contracts import EssayQuestion

from bloom_engine.essay_evaluator import EssayEvaluationResult
from config import ESSAY_MIN_WORDS_FOR_FULL_STRUCTURE, ESSAY_V2_WEIGHTS

logger = logging.getLogger(__name__)

# Cognitive verbs per Bloom level (Arabic + English stems).
BLOOM_VERBS: Dict[str, List[str]] = {
    "REMEMBER": ["عرّف", "اذكر", "عدّد", "define", "list", "recall", "name"],
    "UNDERSTAND": ["اشرح", "فسّر", "لخّص", "وضّح", "explain", "describe", "summarize"],
    "APPLY": ["طبّق", "استخدم", "احسب", "apply", "use", "solve", "demonstrate"],
    "ANALYZE": ["حلّل", "قارن", "ميّز", "صنّف", "analyze", "compare", "distinguish", "examine"],
    "EVALUATE": ["قيّم", "حاكم", "برّر", "انتقد", "evaluate", "justify", "critique", "assess"],
}

CONNECTIVES: List[str] = [
    "لأن", "لذلك", "إذن", "حيث", "بينما", "علاوة", "بالإضافة", "نتيجة",
    "أولاً", "ثانياً", "أخيراً", "من ناحية", "في المقابل", "بالتالي",
    "because", "therefore", "however", "moreover", "finally", "in contrast",
]

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens (Arabic + Latin scripts)."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def tfidf_vectors(documents: Sequence[str]) -> List[Dict[str, float]]:
    """Compute TF-IDF vectors for a small in-memory corpus."""
    tokenised = [tokenize(d) for d in documents]
    n_docs = len(documents)
    df: Counter = Counter()
    for toks in tokenised:
        for term in set(toks):
            df[term] += 1
    vectors: List[Dict[str, float]] = []
    for toks in tokenised:
        tf = Counter(toks)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            vec[term] = (count / max(1, len(toks))) * idf
        vectors.append(vec)
    return vectors


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (na * nb))))


class EssayEvaluatorV2:
    """Multi-criteria deterministic essay evaluator."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = dict(weights or ESSAY_V2_WEIGHTS)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")

    # -- criteria ------------------------------------------------------------
    def content_coverage(self, question: EssayQuestion, answer: str) -> float:
        """TF-IDF cosine of the answer against model answer + rubric text."""
        reference = question.model_answer + " " + " ".join(
            f"{r.name} {r.description}" for r in question.rubric
        )
        vecs = tfidf_vectors([reference, answer])
        return cosine_similarity(vecs[0], vecs[1])

    def structure_score(self, answer: str) -> float:
        """Paragraphing + length adequacy + discourse-marker presence."""
        words = tokenize(answer)
        paragraphs = [p for p in re.split(r"\n\s*\n", answer) if p.strip()]
        if not words:
            return 0.0
        length_part = min(1.0, len(words) / ESSAY_MIN_WORDS_FOR_FULL_STRUCTURE)
        para_part = min(1.0, len(paragraphs) / 3.0)
        lowered = answer.lower()
        markers = sum(1 for c in CONNECTIVES if c in lowered)
        marker_part = min(1.0, markers / 4.0)
        return float(0.5 * length_part + 0.25 * para_part + 0.25 * marker_part)

    def bloom_verb_score(self, question: EssayQuestion, answer: str) -> float:
        """Density of target-level cognitive verbs, saturating at 5 hits."""
        level = question.bloom_level.value
        verbs = BLOOM_VERBS.get(level, [])
        lowered = answer.lower()
        hits = sum(lowered.count(v) for v in verbs)
        # Credit also half for verbs of adjacent (lower) levels.
        lower_levels = ["REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE", "EVALUATE"]
        idx = lower_levels.index(level) if level in lower_levels else 0
        partial = 0
        for lv in lower_levels[:idx]:
            partial += sum(lowered.count(v) for v in BLOOM_VERBS[lv])
        score = (hits + 0.3 * partial) / 5.0
        return float(min(1.0, score))

    def coherence_score(self, answer: str) -> float:
        """Connective density + healthy sentence-length variance."""
        sentences = [s for s in re.split(r"[.!?؟\n]+", answer) if s.strip()]
        words = tokenize(answer)
        if len(sentences) < 2 or not words:
            return 0.2 if words else 0.0
        lowered = answer.lower()
        connective_density = sum(lowered.count(c) for c in CONNECTIVES) / len(sentences)
        density_part = min(1.0, connective_density / 1.5)
        lengths = [len(tokenize(s)) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
        # Moderate variance (std 3..10 words) is healthiest.
        std = math.sqrt(variance)
        variance_part = 1.0 if 3.0 <= std <= 10.0 else max(0.2, 1.0 - abs(std - 6.5) / 12.0)
        return float(0.6 * density_part + 0.4 * variance_part)

    # -- main API --------------------------------------------------------------
    def evaluate(self, question: EssayQuestion, student_answer: str) -> EssayEvaluationResult:
        """Score ``student_answer`` and map results onto the v1 contract."""
        total_max = sum(r.max_score for r in question.rubric)
        trimmed = student_answer.strip()
        if len(trimmed) < 15:
            return EssayEvaluationResult(
                total_score=0.0,
                max_score=total_max,
                score_percentage=0.0,
                criterion_scores={r.name: 0.0 for r in question.rubric},
                feedback_text="الإجابة قصيرة جداً وغير مكتملة لتحقيق معايير التقييم.",
                strengths=[],
                weaknesses=["الإجابة تفتقر إلى التفاصيل والشرح الكافي."],
            )

        sub_scores = {
            "content_coverage": self.content_coverage(question, trimmed),
            "structure": self.structure_score(trimmed),
            "bloom_verbs": self.bloom_verb_score(question, trimmed),
            "coherence": self.coherence_score(trimmed),
        }
        weighted = sum(self.weights[k] * sub_scores[k] for k in self.weights)
        percentage = round(weighted * 100.0, 1)

        # Map the global percentage onto rubric criteria proportionally,
        # nudged by content coverage for the first (conceptual) criterion.
        crit_scores: Dict[str, float] = {}
        for i, r in enumerate(question.rubric):
            factor = weighted
            if i == 0:
                factor = 0.5 * weighted + 0.5 * sub_scores["content_coverage"]
            crit_scores[r.name] = round(r.max_score * min(1.0, factor), 2)
        total_score = round(sum(crit_scores.values()), 2)
        percentage = round((total_score / total_max * 100.0) if total_max else 0.0, 1)

        strengths: List[str] = []
        weaknesses: List[str] = []
        labels = {
            "content_coverage": ("تغطية مفاهيمية قوية للمحتوى المرجعي.", "التغطية المفاهيمية ضعيفة؛ راجع المفاهيم الأساسية للسؤال."),
            "structure": ("بنية إجابة منظمة وواضحة.", "بنية الإجابة تحتاج فقرات وروابط أوضح."),
            "bloom_verbs": ("استخدام جيد لأفعال المستوى المعرفي المستهدف.", "ادعم الإجابة بأفعال تحليل/تقييم أقوى (قارن، حلّل، برّر)."),
            "coherence": ("ترابط وتماسك جيد بين الجمل.", "اربط الجمل بأدوات ربط منطقية لتحسين التماسك."),
        }
        for key, value in sub_scores.items():
            if value >= 0.6:
                strengths.append(labels[key][0])
            elif value < 0.4:
                weaknesses.append(labels[key][1])

        feedback = (
            f"تقييم آلي متعدد المعايير: التغطية {sub_scores['content_coverage']:.0%}، "
            f"البنية {sub_scores['structure']:.0%}، أفعال بلوم {sub_scores['bloom_verbs']:.0%}، "
            f"التماسك {sub_scores['coherence']:.0%}."
        )
        logger.debug("EssayV2 sub-scores: %s -> %.1f%%", sub_scores, percentage)
        return EssayEvaluationResult(
            total_score=total_score,
            max_score=round(total_max, 2),
            score_percentage=percentage,
            criterion_scores=crit_scores,
            feedback_text=feedback,
            strengths=strengths,
            weaknesses=weaknesses,
        )
