"""
Tests for the pattern library and confidence-scored AdvancedErrorDiagnoser (v2).
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import pytest

from misconception_analyzer.advanced_diagnoser import AdvancedErrorDiagnoser
from misconception_analyzer.error_diagnoser import ErrorDiagnoser
from misconception_analyzer.pattern_library import (
    PATTERN_LIBRARY,
    get_pattern,
    patterns_by_category,
)


def test_pattern_library_is_expanded():
    assert len(PATTERN_LIBRARY) >= 12
    assert get_pattern("MISC_INVERSE_LOGIC") is not None
    assert get_pattern("MISC_DOES_NOT_EXIST") is None


def test_patterns_by_category_sorted_by_severity():
    logic = patterns_by_category("LOGIC")
    assert logic and all(p.category == "LOGIC" for p in logic)
    severities = [p.severity for p in logic]
    assert severities == sorted(severities, reverse=True)


def test_confidence_grows_with_occurrences():
    d = AdvancedErrorDiagnoser()
    assert d.confidence_for(1) < d.confidence_for(2) < d.confidence_for(5)
    assert d.confidence_for(50) <= 0.98


def test_confidence_spread_bonus():
    d = AdvancedErrorDiagnoser()
    assert d.confidence_for(2, concept_spread=3) > d.confidence_for(2, concept_spread=1)


def test_diagnose_student_scores_and_sorts():
    d = AdvancedErrorDiagnoser()
    for _ in range(4):
        d.record_error("s1", "c1", "MISC_INVERSE_LOGIC")
    d.record_error("s1", "c1", "MISC_TERMINOLOGY_CONFUSION")
    diagnoses = d.diagnose_student("s1")
    assert diagnoses[0].misconception_id == "MISC_INVERSE_LOGIC"
    assert diagnoses[0].occurrences == 4
    assert diagnoses[0].is_persistent is True
    assert diagnoses[0].pattern_name_ar is not None
    assert diagnoses[0].risk_score > diagnoses[-1].risk_score


def test_diagnose_unknown_pattern_still_reported():
    d = AdvancedErrorDiagnoser()
    d.record_error("s1", "c1", "MISC_BRAND_NEW")
    diagnoses = d.diagnose_student("s1")
    assert len(diagnoses) == 1
    assert diagnoses[0].pattern_name_ar is None
    assert diagnoses[0].severity == 0.5  # neutral fallback


def test_persistent_filter():
    d = AdvancedErrorDiagnoser()
    d.record_error("s1", "c1", "MISC_CONFLATION")
    for _ in range(3):
        d.record_error("s1", "c2", "MISC_NEGATION_ERROR")
    persistent = d.persistent_misconceptions("s1")
    assert [p.misconception_id for p in persistent] == ["MISC_NEGATION_ERROR"]


def test_remediation_recommendations_shape_and_limit():
    d = AdvancedErrorDiagnoser()
    for mid in ("MISC_INVERSE_LOGIC", "MISC_CONFLATION", "MISC_NEGATION_ERROR"):
        d.record_error("s1", "c1", mid)
    recs = d.remediation_recommendations("s1", top_n=2)
    assert len(recs) == 2
    assert recs[0]["target_bloom_level"] == "ANALYZE"  # inverse logic remediation level
    assert recs[0]["action_ar"]


def test_v1_behaviour_preserved_in_subclass():
    d = AdvancedErrorDiagnoser()
    assert isinstance(d, ErrorDiagnoser)
    d.record_error("s9", "cX", "MISC_CONFLATION")
    assert d.get_error_patterns("s9") == {"cX": ["MISC_CONFLATION"]}


def test_known_pattern_ids_sorted():
    ids = AdvancedErrorDiagnoser.known_pattern_ids()
    assert ids == sorted(ids)
    assert "MISC_INVARIANCE_FALLACY" in ids
