"""
MCQ v2 analytics: distractor plausibility ranking + classical item analysis.
=============================================================================
Two deterministic tools that upgrade the v1 MCQ generator's output quality:

1. **Distractor plausibility ranking** — scores every distractor of an
   :class:`MCQQuestion` by lexical/conceptual proximity to the correct
   answer (TF-IDF cosine against the key + shared-token overlap) so weak,
   obviously-wrong options can be flagged and replaced.  A good distractor
   is *plausible but wrong*; a distractor nobody would pick adds no
   measurement value.

2. **Classical item analysis** — from a response matrix
   (students x options chosen) computes per item:
   * difficulty index ``p`` = proportion answering correctly,
   * discrimination index (point-biserial correlation between item score
     and total test score),
   * distractor functioning (selection rate of each wrong option; a
     distractor chosen by <5% of students is non-functioning),
   * a quality verdict (RETAIN / REVISE / DISCARD) with reasons.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from common.contracts import MCQQuestion

from bloom_engine.essay_evaluator_v2 import cosine_similarity, tfidf_vectors, tokenize
from config import (
    ITEM_DIFFICULTY_IDEAL_MAX,
    ITEM_DIFFICULTY_IDEAL_MIN,
    ITEM_DISCRIMINATION_GOOD,
    ITEM_DISCRIMINATION_POOR,
)

logger = logging.getLogger(__name__)


@dataclass
class DistractorScore:
    """Plausibility assessment for one distractor option."""

    option_id: str
    plausibility: float          # 0..1, higher = more plausible
    shared_terms: List[str] = field(default_factory=list)
    verdict: str = "PLAUSIBLE"   # PLAUSIBLE | WEAK | IMPLAUSIBLE


@dataclass
class ItemStatistics:
    """Classical test-theory statistics for one administered item."""

    item_id: str
    n_students: int
    difficulty_index: float                  # proportion correct (p-value)
    discrimination_index: float              # point-biserial r_pb
    distractor_rates: Dict[str, float]       # option_id -> selection rate
    non_functioning_distractors: List[str]   # option_ids chosen by < 5%
    verdict: str                             # RETAIN | REVISE | DISCARD
    reasons: List[str] = field(default_factory=list)


class DistractorAnalyzer:
    """Rank distractors by plausibility relative to the correct answer."""

    def rank_distractors(self, question: MCQQuestion) -> List[DistractorScore]:
        """Score and sort the distractors of ``question`` (most plausible first)."""
        correct = next((o for o in question.options if o.is_correct), None)
        if correct is None:
            raise ValueError(f"Question {question.item_id} has no correct option")
        distractors = [o for o in question.options if not o.is_correct]
        if not distractors:
            return []

        docs = [correct.text] + [d.text for d in distractors]
        vecs = tfidf_vectors(docs)
        key_tokens = set(tokenize(correct.text))
        scores: List[DistractorScore] = []
        for d, vec in zip(distractors, vecs[1:]):
            sim = cosine_similarity(vecs[0], vec)
            shared = sorted(key_tokens & set(tokenize(d.text)))
            overlap_bonus = min(0.25, 0.05 * len(shared))
            plausibility = float(min(1.0, sim + overlap_bonus))
            if plausibility >= 0.35:
                verdict = "PLAUSIBLE"
            elif plausibility >= 0.15:
                verdict = "WEAK"
            else:
                verdict = "IMPLAUSIBLE"
            scores.append(
                DistractorScore(
                    option_id=d.option_id,
                    plausibility=round(plausibility, 4),
                    shared_terms=shared,
                    verdict=verdict,
                )
            )
        scores.sort(key=lambda s: s.plausibility, reverse=True)
        return scores


class ItemAnalyzer:
    """Classical item analysis over administered response data."""

    def analyze_item(
        self,
        question: MCQQuestion,
        chosen_option_ids: Sequence[str],
        total_scores: Sequence[float],
    ) -> ItemStatistics:
        """Analyse one item.

        ``chosen_option_ids[i]`` is the option student *i* picked and
        ``total_scores[i]`` that student's total test score (used for the
        point-biserial discrimination index).
        """
        if len(chosen_option_ids) != len(total_scores):
            raise ValueError("chosen_option_ids and total_scores must align")
        n = len(chosen_option_ids)
        correct_id = next(o.option_id for o in question.options if o.is_correct)
        item_scores = [1.0 if c == correct_id else 0.0 for c in chosen_option_ids]

        difficulty = sum(item_scores) / n if n else 0.0
        discrimination = _point_biserial(item_scores, list(total_scores))

        counts: Dict[str, int] = {o.option_id: 0 for o in question.options}
        for c in chosen_option_ids:
            if c in counts:
                counts[c] += 1
        rates = {oid: (cnt / n if n else 0.0) for oid, cnt in counts.items()}
        non_functioning = [
            oid
            for oid, rate in rates.items()
            if oid != correct_id and rate < 0.05
        ]

        reasons: List[str] = []
        if difficulty > 0.95:
            reasons.append("Item is too easy (p > 0.95); it does not differentiate students.")
        elif difficulty > ITEM_DIFFICULTY_IDEAL_MAX:
            reasons.append("Item is easier than the ideal band (p > 0.80).")
        if difficulty < 0.10:
            reasons.append("Item is too hard (p < 0.10); check for errors or untaught content.")
        elif difficulty < ITEM_DIFFICULTY_IDEAL_MIN:
            reasons.append("Item is harder than the ideal band (p < 0.30).")
        if discrimination < 0:
            reasons.append("Negative discrimination: strong students fail this item — likely a keying error.")
        elif discrimination < ITEM_DISCRIMINATION_POOR:
            reasons.append("Poor discrimination (r_pb < 0.10).")
        if non_functioning:
            reasons.append(
                f"{len(non_functioning)} non-functioning distractor(s) chosen by <5% of students."
            )

        if discrimination < 0 or difficulty < 0.05 or difficulty > 0.98:
            verdict = "DISCARD"
        elif reasons:
            verdict = "REVISE"
        else:
            verdict = "RETAIN"
            reasons.append(
                f"Difficulty and discrimination (r_pb >= {ITEM_DISCRIMINATION_GOOD}) are in the healthy band."
            )

        return ItemStatistics(
            item_id=question.item_id,
            n_students=n,
            difficulty_index=round(difficulty, 4),
            discrimination_index=round(discrimination, 4),
            distractor_rates={k: round(v, 4) for k, v in rates.items()},
            non_functioning_distractors=non_functioning,
            verdict=verdict,
            reasons=reasons,
        )

    def analyze_bank(
        self,
        questions: Sequence[MCQQuestion],
        responses_by_item: Dict[str, Sequence[str]],
        total_scores: Sequence[float],
    ) -> List[ItemStatistics]:
        """Analyse every question that has response data."""
        return [
            self.analyze_item(q, responses_by_item[q.item_id], total_scores)
            for q in questions
            if q.item_id in responses_by_item
        ]


def _point_biserial(item_scores: List[float], total_scores: List[float]) -> float:
    """Point-biserial correlation between a dichotomous item and totals."""
    n = len(item_scores)
    if n < 3:
        return 0.0
    mean_total = sum(total_scores) / n
    sd_total = math.sqrt(sum((t - mean_total) ** 2 for t in total_scores) / n)
    if sd_total == 0:
        return 0.0
    correct_totals = [t for s, t in zip(item_scores, total_scores) if s == 1.0]
    wrong_totals = [t for s, t in zip(item_scores, total_scores) if s == 0.0]
    if not correct_totals or not wrong_totals:
        return 0.0
    p = len(correct_totals) / n
    q = 1.0 - p
    m1 = sum(correct_totals) / len(correct_totals)
    m0 = sum(wrong_totals) / len(wrong_totals)
    return float(((m1 - m0) / sd_total) * math.sqrt(p * q))
