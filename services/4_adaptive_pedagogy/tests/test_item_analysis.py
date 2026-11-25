"""
Tests for MCQ v2 analytics: distractor plausibility + classical item analysis.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import pytest
from common.contracts import BloomLevel, MCQOption, MCQQuestion

from bloom_engine.item_analysis import DistractorAnalyzer, ItemAnalyzer


@pytest.fixture
def question():
    return MCQQuestion(
        item_id="mcq_1",
        concept_id="concept_attention",
        stem="ما وظيفة آلية الانتباه؟",
        bloom_level=BloomLevel.UNDERSTAND,
        options=[
            MCQOption(option_id="key", text="حساب أوزان تركيز لعناصر التسلسل وربط المعطيات بالاستنتاجات", is_correct=True),
            MCQOption(option_id="near", text="حساب أوزان تركيز لعناصر التسلسل دون ربطها بالسياق", is_correct=False,
                      misconception_id="MISC_PARTIAL_APPLICATION"),
            MCQOption(option_id="far", text="طهي الطعام في المطبخ باستخدام الفرن", is_correct=False,
                      misconception_id="MISC_UNKNOWN"),
        ],
        explanation="الانتباه يحسب الأوزان ويربط السياق.",
        grounding_span_ids=["span_1"],
    )


def test_distractor_ranking_prefers_plausible(question):
    scores = DistractorAnalyzer().rank_distractors(question)
    assert scores[0].option_id == "near"
    assert scores[0].plausibility > scores[-1].plausibility


def test_distractor_verdicts(question):
    scores = {s.option_id: s for s in DistractorAnalyzer().rank_distractors(question)}
    assert scores["far"].verdict == "IMPLAUSIBLE"
    assert scores["near"].verdict in ("PLAUSIBLE", "WEAK")


def test_distractor_ranking_requires_correct_option():
    bad = MCQQuestion(
        item_id="bad", concept_id="c", stem="?", bloom_level=BloomLevel.REMEMBER,
        options=[MCQOption(option_id="x", text="نص", is_correct=False)],
        explanation="", grounding_span_ids=[],
    )
    with pytest.raises(ValueError):
        DistractorAnalyzer().rank_distractors(bad)


def test_difficulty_index_matches_proportion(question):
    # 6 of 10 correct
    chosen = ["key"] * 6 + ["near"] * 3 + ["far"] * 1
    totals = [10, 9, 9, 8, 8, 7, 5, 4, 3, 2]
    stats = ItemAnalyzer().analyze_item(question, chosen, totals)
    assert stats.difficulty_index == pytest.approx(0.6)
    assert stats.n_students == 10


def test_discrimination_positive_for_good_item(question):
    # Strong students (high totals) pick the key; weak students miss.
    chosen = ["key"] * 5 + ["near"] * 3 + ["far"] * 2
    totals = [20, 19, 18, 17, 16, 8, 7, 6, 5, 4]
    stats = ItemAnalyzer().analyze_item(question, chosen, totals)
    assert stats.discrimination_index > 0.3
    assert stats.verdict in ("RETAIN", "REVISE")


def test_negative_discrimination_flags_discard(question):
    # Weak students pick the key, strong students miss -> keying problem.
    chosen = ["near"] * 5 + ["key"] * 5
    totals = [20, 19, 18, 17, 16, 8, 7, 6, 5, 4]
    stats = ItemAnalyzer().analyze_item(question, chosen, totals)
    assert stats.discrimination_index < 0
    assert stats.verdict == "DISCARD"


def test_non_functioning_distractor_detected(question):
    # Nobody picks "far" (0% < 5%) in a class of 20.
    chosen = ["key"] * 12 + ["near"] * 8
    totals = list(range(20, 0, -1))
    stats = ItemAnalyzer().analyze_item(question, chosen, totals)
    assert "far" in stats.non_functioning_distractors
    assert stats.distractor_rates["far"] == 0.0


def test_analyze_bank_skips_items_without_data(question):
    other = MCQQuestion(
        item_id="mcq_2", concept_id="c", stem="؟", bloom_level=BloomLevel.REMEMBER,
        options=[MCQOption(option_id="k", text="صواب", is_correct=True)],
        explanation="", grounding_span_ids=[],
    )
    results = ItemAnalyzer().analyze_bank(
        [question, other], {question.item_id: ["key", "near", "key"]}, [5, 3, 4]
    )
    assert len(results) == 1 and results[0].item_id == "mcq_1"


def test_analyze_item_length_mismatch_raises(question):
    with pytest.raises(ValueError):
        ItemAnalyzer().analyze_item(question, ["key"], [1, 2])
